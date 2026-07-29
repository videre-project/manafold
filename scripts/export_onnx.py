#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from manafold.models.card_ranking import build_family_card_ranking
from manafold.models.data import load_training_dataset
from manafold.models.family_backoff import (
  build_family_vocab,
  extract_proposed_edges,
)
from manafold.models.model_artifacts import load_model_artifact

INPUT_NAMES = [
  "cards",
  "zones",
  "quantities",
  "quantity_weights",
  "deck_idx",
  "deck_count",
  "package_features",
]
OUTPUT_NAMES = ["logits"]
DEFAULT_OPSET_VERSION = 17
DEFAULT_TOKEN_COUNT = 16
DEFAULT_DECK_COUNT = 1


class ONNXWrapper(torch.nn.Module):
  def __init__(self, net: torch.nn.Module) -> None:
    super().__init__()
    self.net = net

  def forward(
    self,
    card_idx: torch.Tensor,
    zone_idx: torch.Tensor,
    quantity_idx: torch.Tensor,
    quantity_weight: torch.Tensor,
    deck_idx: torch.Tensor,
    deck_count: torch.Tensor,
    package_features: torch.Tensor,
  ) -> torch.Tensor:
    return self.net.forward_onnx(
      card_idx,
      zone_idx,
      quantity_idx,
      quantity_weight,
      deck_idx,
      deck_count,
      package_features,
    )


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  if importlib.util.find_spec("onnx") is None:
    raise SystemExit(
      "ONNX export requires the optional 'onnx' package. Install it in the "
      "Manafold environment before running this script."
    )

  workspace = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", Path.cwd()))
  artifact_dir = _resolve_path(args.model_artifact, workspace)
  output_dir = _resolve_path(args.output_dir, workspace)
  if args.family_relations is not None:
    args.family_relations = _resolve_path(args.family_relations, workspace)
  if args.ranking_dataset is not None:
    args.ranking_dataset = _resolve_path(args.ranking_dataset, workspace)
  output_dir.mkdir(parents=True, exist_ok=True)
  onnx_path = output_dir / args.onnx_name

  artifact = load_model_artifact(artifact_dir, device="cpu")
  model = artifact["model"]
  net = getattr(model, "_network", None)
  if net is None or not hasattr(net, "forward_onnx"):
    raise SystemExit(
      f"Artifact {artifact_dir} does not expose an ONNX forward pass."
    )

  model_config = dict(artifact["model_config"])
  package_count = int(model_config.get("package_count") or 0)
  if package_count and not args.allow_package_features:
    raise SystemExit(
      "Package-conditioned artifacts are not enabled for Worker export. Pass "
      "--allow-package-features only if the Worker runtime will provide the "
      "package_features tensor."
    )
  if args.deck_count != 1:
    print(
      "warning: forward_onnx currently traces deck_count as a fixed shape; "
      "Cloudflare Worker exports should normally keep --deck-count 1.",
      file=sys.stderr,
    )

  net = net.cpu().eval()
  wrapper = ONNXWrapper(net).eval()
  torch.backends.mha.set_fastpath_enabled(False)
  sample_inputs = _sample_inputs(
    card_count=int(model_config["card_count"]),
    zone_count=int(model_config["zone_count"]),
    quantity_count=int(model_config["quantity_count"]),
    package_count=package_count,
    token_count=args.sample_token_count,
    deck_count=args.deck_count,
  )
  pytorch_logits = _run_pytorch(wrapper, sample_inputs)

  export_kwargs: dict[str, Any] = {
    "input_names": INPUT_NAMES,
    "output_names": OUTPUT_NAMES,
    "opset_version": args.opset_version,
    "do_constant_folding": True,
    "external_data": False,
  }
  if args.dynamic_tokens:
    export_kwargs["dynamic_axes"] = {
      "cards": {0: "token_count"},
      "zones": {0: "token_count"},
      "quantities": {0: "token_count"},
      "quantity_weights": {0: "token_count"},
      "deck_idx": {0: "token_count"},
    }
  if args.dynamo_export:
    export_kwargs["dynamo"] = True
  else:
    export_kwargs["dynamo"] = False

  with torch.no_grad():
    torch.onnx.export(wrapper, sample_inputs, onnx_path, **export_kwargs)

  onnx_info = _inspect_onnx(onnx_path)
  verification = _verify_onnx(
    onnx_path,
    sample_inputs=sample_inputs,
    pytorch_logits=pytorch_logits,
    input_names=onnx_info["input_names"],
  )
  files = _write_worker_metadata(
    artifact=artifact,
    artifact_dir=artifact_dir,
    output_dir=output_dir,
    onnx_path=onnx_path,
    onnx_info=onnx_info,
    verification=verification,
    args=args,
  )

  if args.copy_artifact_metadata:
    _copy_artifact_metadata(artifact_dir, output_dir)
  if args.worker_assets_dir is not None:
    _copy_worker_assets(output_dir, _resolve_path(args.worker_assets_dir, workspace))

  print(json.dumps({
    "status": "completed",
    "onnx_path": str(onnx_path),
    "onnx_bytes": onnx_path.stat().st_size,
    "worker_manifest": str(files["worker_manifest"]),
    "worker_assets_dir": (
      str(_resolve_path(args.worker_assets_dir, workspace))
      if args.worker_assets_dir else None
    ),
    "verification": verification,
  }, indent=2, sort_keys=True))
  return 0


def _resolve_path(path: Path, workspace: Path) -> Path:
  path = path.expanduser()
  if not path.is_absolute():
    path = workspace / path
  return path.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Export a Manafold set-model artifact to a single-file ONNX bundle for "
      "Cloudflare Worker inference."
    )
  )
  parser.add_argument(
    "--model-artifact",
    required=True,
    type=Path,
    help="Path to a Manafold model artifact directory.",
  )
  parser.add_argument(
    "--output-dir",
    required=True,
    type=Path,
    help="Directory for the ONNX model and Worker metadata bundle.",
  )
  parser.add_argument(
    "--format",
    required=True,
    help="Deck format served by the exported model bundle.",
  )
  parser.add_argument(
    "--family-relations",
    type=Path,
    help=(
      "Optional seed-free auto-ontology or compact relation artifact. Semantic "
      "edges are folded into the serving family map; the source artifact is "
      "not copied into the Worker bundle."
    ),
  )
  parser.add_argument(
    "--ranking-dataset",
    type=Path,
    help=(
      "Dataset whose train split supplies mainboard family-card "
      "distinctiveness rankings. Release builds should pass the dataset used "
      "to train the artifact."
    ),
  )
  parser.add_argument(
    "--onnx-name",
    default="model.onnx",
    help="Filename for the exported ONNX graph inside --output-dir.",
  )
  parser.add_argument(
    "--opset-version",
    default=DEFAULT_OPSET_VERSION,
    type=int,
    help="ONNX opset version to request from torch.onnx.export.",
  )
  parser.add_argument(
    "--sample-token-count",
    default=DEFAULT_TOKEN_COUNT,
    type=int,
    help="Token count used for tracing and verification.",
  )
  parser.add_argument(
    "--deck-count",
    default=DEFAULT_DECK_COUNT,
    type=int,
    help="Deck count used for tracing. Worker exports should normally use 1.",
  )
  parser.add_argument(
    "--fixed-token-count",
    dest="dynamic_tokens",
    action="store_false",
    default=True,
    help="Export a fixed token axis instead of a dynamic token axis.",
  )
  parser.add_argument(
    "--dynamo-export",
    action="store_true",
    help=(
      "Use PyTorch's torch.export-based ONNX exporter. The default uses the "
      "legacy exporter because it keeps the single-deck Worker graph compact "
      "and prunes unused package/deck-count inputs."
    ),
  )
  parser.add_argument(
    "--allow-package-features",
    action="store_true",
    help="Permit export of package-conditioned artifacts.",
  )
  parser.add_argument(
    "--copy-artifact-metadata",
    action="store_true",
    help="Copy model_config.json, temperature.json, and training_manifest.json.",
  )
  parser.add_argument(
    "--include-card-vocab-maps",
    action="store_true",
    help=(
      "Include precomputed oracle/name lookup maps in card_vocab.json. By "
      "default the Worker bundle writes compact card entries and lets the "
      "runtime build maps at startup."
    ),
  )
  parser.add_argument(
    "--worker-assets-dir",
    type=Path,
    help=(
      "Optional Cloudflare Worker asset directory to receive the generated "
      "bundle after export. This copies files only; it does not run wrangler."
    ),
  )
  args = parser.parse_args(argv)
  if args.sample_token_count <= 0:
    parser.error("--sample-token-count must be positive")
  if args.deck_count <= 0:
    parser.error("--deck-count must be positive")
  if args.opset_version <= 0:
    parser.error("--opset-version must be positive")
  return args


def _sample_inputs(
  *,
  card_count: int,
  zone_count: int,
  quantity_count: int,
  package_count: int,
  token_count: int,
  deck_count: int,
) -> tuple[torch.Tensor, ...]:
  card_idx = torch.arange(token_count, dtype=torch.long) % card_count
  zone_idx = torch.zeros(token_count, dtype=torch.long)
  if zone_count > 1 and token_count > 1:
    zone_idx[1::2] = 1
  quantity_weight = torch.arange(
    1,
    token_count + 1,
    dtype=torch.float32,
  )
  quantity_idx = quantity_weight.to(dtype=torch.long)
  quantity_idx.clamp_(max=max(quantity_count - 1, 0))
  deck_idx = torch.zeros(token_count, dtype=torch.long)
  if deck_count > 1:
    deck_idx = torch.arange(token_count, dtype=torch.long) % deck_count
  deck_count_tensor = torch.tensor([deck_count], dtype=torch.long)
  package_features = torch.zeros((deck_count, package_count), dtype=torch.float32)
  return (
    card_idx,
    zone_idx,
    quantity_idx,
    quantity_weight,
    deck_idx,
    deck_count_tensor,
    package_features,
  )


def _run_pytorch(
  wrapper: ONNXWrapper,
  sample_inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
  with torch.no_grad():
    return wrapper(*sample_inputs).detach().cpu()


def _inspect_onnx(onnx_path: Path) -> dict[str, Any]:
  import onnx

  model = onnx.load(onnx_path)
  onnx.checker.check_model(model)
  return {
    "input_names": [value.name for value in model.graph.input],
    "output_names": [value.name for value in model.graph.output],
    "opsets": [
      {"domain": opset.domain, "version": int(opset.version)}
      for opset in model.opset_import
    ],
    "initializer_count": len(model.graph.initializer),
  }


def _verify_onnx(
  onnx_path: Path,
  *,
  sample_inputs: tuple[torch.Tensor, ...],
  pytorch_logits: torch.Tensor,
  input_names: list[str],
) -> dict[str, Any]:
  if importlib.util.find_spec("onnxruntime") is None:
    return {
      "enabled": False,
      "reason": "onnxruntime is not installed.",
    }

  import numpy as np
  import onnxruntime as ort

  by_name = dict(zip(INPUT_NAMES, sample_inputs, strict=True))
  feed = {
    name: by_name[name].detach().cpu().numpy()
    for name in input_names
  }
  session = ort.InferenceSession(
    str(onnx_path),
    providers=["CPUExecutionProvider"],
  )
  onnx_logits = session.run(OUTPUT_NAMES, feed)[0]
  diff = np.max(np.abs(pytorch_logits.numpy() - onnx_logits))
  return {
    "enabled": True,
    "runtime": "onnxruntime CPUExecutionProvider",
    "max_abs_logit_diff": float(diff),
    "pytorch_shape": list(pytorch_logits.shape),
    "onnx_shape": list(onnx_logits.shape),
  }


def _write_worker_metadata(
  *,
  artifact: dict[str, Any],
  artifact_dir: Path,
  output_dir: Path,
  onnx_path: Path,
  onnx_info: dict[str, Any],
  verification: dict[str, Any],
  args: argparse.Namespace,
) -> dict[str, Path]:
  card_vocab_path = output_dir / "card_vocab.json"
  label_vocab_path = output_dir / "label_vocab.json"
  family_vocab_path = output_dir / "family_vocab.json"
  card_ranking_path = output_dir / "family_card_ranking.json"
  zone_vocab_path = output_dir / "zone_vocab.json"
  worker_manifest_path = output_dir / "worker_manifest.json"

  card_rows = sorted(
    (
      {
        "card_idx": int(row["card_idx"]),
        "oracle_id": str(row["oracle_id"]),
        "primary_name": str(row["primary_name"]),
      }
      for row in artifact["card_vocab"]
    ),
    key=lambda row: int(row["card_idx"]),
  )
  label_rows = sorted(
    artifact["label_metadata"].values(),
    key=lambda row: row["label_id"],
  )
  zone_vocab = {
    zone: int(index)
    for zone, index in sorted(
      artifact["zone_vocab"].items(),
      key=lambda item: item[1],
    )
  }
  card_vocab_payload: dict[str, Any] = {"entries": card_rows}
  if args.include_card_vocab_maps:
    card_vocab_payload["oracle_id_to_card_idx"] = {
      str(row["oracle_id"]): int(row["card_idx"])
      for row in card_rows
      if row.get("oracle_id") is not None
    }
    card_vocab_payload["primary_name_to_card_idx"] = {
      str(row["primary_name"]): int(row["card_idx"])
      for row in card_rows
      if row.get("primary_name") is not None
    }
  _write_json(card_vocab_path, card_vocab_payload)
  _write_json(label_vocab_path, {"entries": [_json_safe(row) for row in label_rows]})
  relation_source = None
  proposed_edges: tuple[dict[str, Any], ...] = ()
  if args.family_relations is not None:
    relation_path = args.family_relations
    relation_payload = json.loads(relation_path.read_text(encoding="utf-8"))
    proposed_edges = extract_proposed_edges(relation_payload)
    relation_source = {
      "filename": relation_path.name,
      "sha256": _sha256(relation_path),
      "proposed_edge_count": len(proposed_edges),
    }
  family_vocab = build_family_vocab(
    artifact["label_metadata"],
    proposed_edges=proposed_edges,
  )
  if relation_source is not None:
    family_vocab["relation_source"] = relation_source
  _write_json(family_vocab_path, family_vocab)
  ranking_source = None
  card_ranking = {
    "version": "disabled",
    "method": "none",
    "scope": "none",
    "training_deck_count": 0,
    "parameters": {},
    "families": {},
  }
  if args.ranking_dataset is not None:
    ranking_dataset = load_training_dataset(
      args.ranking_dataset,
      target_source=str(artifact["model_config"].get("target_source") or ""),
    )
    if ranking_dataset.dataset_version != artifact["model_config"].get(
      "dataset_version"
    ):
      raise ValueError(
        "Ranking dataset version does not match the model artifact: "
        f"{ranking_dataset.dataset_version!r} != "
        f"{artifact['model_config'].get('dataset_version')!r}."
      )
    card_ranking = build_family_card_ranking(
      ranking_dataset,
      artifact_card_vocab=tuple(artifact["card_vocab"]),
      family_vocab=family_vocab,
    )
    ranking_source = {
      "dataset": str(args.ranking_dataset),
      "dataset_version": ranking_dataset.dataset_version,
      "training_deck_count": card_ranking["training_deck_count"],
    }
  _write_json(card_ranking_path, card_ranking)
  _write_json(zone_vocab_path, zone_vocab)

  model_config = dict(artifact["model_config"])
  onnx_sha256 = _sha256(onnx_path)
  family_vocab_sha256 = _sha256(family_vocab_path)
  card_ranking_sha256 = _sha256(card_ranking_path)
  serving_fingerprint = hashlib.sha256(
    (
      f"{onnx_sha256}:{family_vocab_sha256}:"
      f"{card_ranking_sha256}"
    ).encode("utf-8")
  ).hexdigest()[:12]
  created_at = datetime.now(timezone.utc)
  serving_model = "manafold-a3"
  serving_version = (
    f"{str(args.format).strip().lower()}-"
    f"{created_at.strftime('%Y%m%d')}-{serving_fingerprint}"
  )
  manifest = {
    "bundle_schema": {
      "name": "manafold_worker_onnx_bundle",
      "version": 1,
    },
    "created_at": created_at.isoformat(),
    "source_artifact": str(artifact_dir),
    "model_version": artifact["model_version"],
    "model_family": artifact["model_family"],
    "serving": {
      "model": serving_model,
      "version": serving_version,
    },
    "format": str(args.format).strip().lower(),
    "dataset_version": model_config.get("dataset_version"),
    "target_source": model_config.get("target_source"),
    "temperature": float(artifact["temperature"]),
    "fixed_deck_count": int(args.deck_count),
    "dynamic_token_count": bool(args.dynamic_tokens),
    "onnx": {
      "file": onnx_path.name,
      "sha256": onnx_sha256,
      "bytes": onnx_path.stat().st_size,
      "requested_opset_version": int(args.opset_version),
      **onnx_info,
    },
    "runtime_inputs": [
      {
        "name": name,
        "dtype": _input_dtype(name),
        "shape": _input_shape(
          name,
          dynamic_tokens=bool(args.dynamic_tokens),
          fixed_deck_count=int(args.deck_count),
          package_count=int(model_config.get("package_count") or 0),
          sample_token_count=int(args.sample_token_count),
        ),
      }
      for name in onnx_info["input_names"]
    ],
    "runtime_outputs": [
      {
        "name": "logits",
        "dtype": "float32",
        "shape": [int(args.deck_count), int(model_config["label_count"])],
      }
    ],
    "vocab_files": {
      "cards": card_vocab_path.name,
      "families": family_vocab_path.name,
      "family_card_ranking": card_ranking_path.name,
      "labels": label_vocab_path.name,
      "zones": zone_vocab_path.name,
    },
    "family_backoff": {
      "version": family_vocab["version"],
      "sha256": family_vocab_sha256,
      "family_count": len(family_vocab["families"]),
      "raw_label_count": len(family_vocab["entries"]),
      "policy": family_vocab["policy"],
      "relation_source": relation_source,
    },
    "card_ranking": (
      {
        "enabled": True,
        "version": card_ranking["version"],
        "method": card_ranking["method"],
        "scope": card_ranking["scope"],
        "sha256": card_ranking_sha256,
        "source": ranking_source,
        "response_min_score": 0.15,
        "response_max_cards": 8,
      }
      if ranking_source is not None
      else {
        "enabled": False,
        "sha256": card_ranking_sha256,
      }
    ),
    "model_config": {
      key: _json_safe(value)
      for key, value in model_config.items()
      if key in {
        "architecture",
        "attention_heads",
        "attention_layers",
        "card_count",
        "dropout",
        "embedding_dim",
        "hidden_dim",
        "hypergeometric_draw_count",
        "label_count",
        "label_smoothing",
        "package_count",
        "pooling",
        "quantity_count",
        "rho_hidden_dim",
        "seed",
        "observation_conditioning",
        "token_scope",
        "zone_count",
      }
    },
    "verification": verification,
  }
  _write_json(worker_manifest_path, manifest)
  return {
    "card_vocab": card_vocab_path,
    "family_vocab": family_vocab_path,
    "family_card_ranking": card_ranking_path,
    "label_vocab": label_vocab_path,
    "zone_vocab": zone_vocab_path,
    "worker_manifest": worker_manifest_path,
  }


def _copy_artifact_metadata(artifact_dir: Path, output_dir: Path) -> None:
  for name in (
    "model_config.json",
    "temperature.json",
    "training_manifest.json",
  ):
    source = artifact_dir / name
    if source.exists():
      shutil.copy2(source, output_dir / name)


def _copy_worker_assets(output_dir: Path, worker_assets_dir: Path) -> None:
  worker_assets_dir.mkdir(parents=True, exist_ok=True)
  for source in sorted(output_dir.iterdir()):
    if source.is_file():
      shutil.copy2(source, worker_assets_dir / source.name)


def _input_dtype(name: str) -> str:
  if name == "quantity_weights" or name == "package_features":
    return "float32"
  return "int64"


def _input_shape(
  name: str,
  *,
  dynamic_tokens: bool,
  fixed_deck_count: int,
  package_count: int,
  sample_token_count: int,
) -> list[str | int]:
  token_dim: str | int = "token_count" if dynamic_tokens else sample_token_count
  if name in {"cards", "zones", "quantities", "quantity_weights", "deck_idx"}:
    return [token_dim]
  if name == "deck_count":
    return [1]
  if name == "package_features":
    return [fixed_deck_count, package_count]
  return []


def _write_json(path: Path, payload: Any) -> None:
  path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def _json_safe(value: Any) -> Any:
  if hasattr(value, "isoformat"):
    return value.isoformat()
  if isinstance(value, dict):
    return {str(key): _json_safe(inner) for key, inner in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(inner) for inner in value]
  return value


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


if __name__ == "__main__":
  raise SystemExit(main())

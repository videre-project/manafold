from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from manafold.datasets.model_inputs import TrainingDataset
from manafold.models.classifiers import (
  POOLING_QUANTITY_WEIGHTED,
  DeepSetsClassifier,
  PooledLinearClassifier,
  SetTransformerClassifier,
  max_quantity,
)
from manafold.serialization import sha256_file, write_json

MODEL_POOLED_LINEAR = "pooled-linear"
MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED = (
  "deepsets-quantity-weighted-regularized"
)
MODEL_DEEPSETS_PLUSPLUS_REGULARIZED = "deepsets-plusplus-regularized"
MODEL_DEEPSETS_PLUSPLUS_REGULARIZED_LABEL_SMOOTHING_005 = (
  "deepsets-plusplus-regularized-label-smoothing-005"
)
MODEL_DEEPSETS_PLUSPLUS_REGULARIZED_LABEL_SMOOTHING_010 = (
  "deepsets-plusplus-regularized-label-smoothing-010"
)
SAVED_MODEL_FORMAT_VERSION = "manafold_saved_model_v0"


def save_trained_model(
  saved_model_dir: Path,
  *,
  model_name: str,
  model: Any,
  dataset: TrainingDataset,
  model_result: dict[str, Any],
  training_result: dict[str, Any],
  seed: int,
  family_relations: Path | None = None,
) -> dict[str, Any]:
  if not hasattr(model, "state_dict_for_saving"):
    raise ValueError(f"Model {model_name!r} cannot be saved.")
  saved_model_dir.mkdir(parents=True, exist_ok=True)

  state_path = saved_model_dir / "model.pt"
  torch.save(
    {
      "format_version": SAVED_MODEL_FORMAT_VERSION,
      "model_family": model_name,
      "state_dict": model.state_dict_for_saving(),
    },
    state_path,
  )
  model_config = _saved_model_configuration(
    model_name=model_name,
    model=model,
    dataset=dataset,
    model_result=model_result,
    seed=seed,
  )
  write_json(saved_model_dir / "model_config.json", model_config)
  write_json(
    saved_model_dir / "label_vocab.json",
    {
      "labels": list(model.labels),
      "entries": _label_vocab_entries(dataset, model.labels),
    },
  )
  write_json(
    saved_model_dir / "zone_vocab.json",
    dataset.zone_vocab,
  )
  calibration_override = model_result.get("saved_model_calibration") or {}
  calibration = calibration_override.get("calibration") or (
    model_result.get("metrics", {}).get("calibration")
  )
  write_json(
    saved_model_dir / "temperature.json",
    {
      "temperature": model_config.get("temperature", 1.0),
      "selection_split": calibration_override.get("selection_split")
      or ((calibration or {}).get("selection_split")),
      "calibration": calibration,
      "source_model_version": calibration_override.get("source_model_version"),
      "source_temperature_sha256": calibration_override.get(
        "source_temperature_sha256"
      ),
    },
  )
  _copy_card_vocab(dataset, saved_model_dir / "card_vocab.parquet")
  family_relations_output = None
  if family_relations is not None:
    family_relations_output = saved_model_dir / "family_relations.json"
    if family_relations.resolve() != family_relations_output.resolve():
      shutil.copy2(family_relations, family_relations_output)
  training_manifest = _saved_model_training_manifest(
    training_result=training_result,
    model_name=model_name,
    model_result=model_result,
    saved_model_dir=saved_model_dir,
    family_relations=family_relations_output,
  )
  write_json(saved_model_dir / "training_manifest.json", training_manifest)
  files = {
    "model": str(state_path),
    "model_config": str(saved_model_dir / "model_config.json"),
    "card_vocab": str(saved_model_dir / "card_vocab.parquet"),
    "label_vocab": str(saved_model_dir / "label_vocab.json"),
    "zone_vocab": str(saved_model_dir / "zone_vocab.json"),
    "temperature": str(saved_model_dir / "temperature.json"),
    "training_manifest": str(saved_model_dir / "training_manifest.json"),
  }
  if family_relations_output is not None:
    files["family_relations"] = str(family_relations_output)
  return {
    "path": str(saved_model_dir),
    "model_family": model_name,
    "format_version": SAVED_MODEL_FORMAT_VERSION,
    "files": files,
  }


def load_saved_model(
  saved_model_dir: Path,
  *,
  device: str = "auto",
) -> dict[str, Any]:
  saved_model_dir = saved_model_dir.resolve()
  model_config = json.loads(
    (saved_model_dir / "model_config.json").read_text(encoding="utf-8")
  )
  card_vocab = tuple(
    pq.read_table(saved_model_dir / "card_vocab.parquet").to_pylist()
  )
  zone_vocab = json.loads(
    (saved_model_dir / "zone_vocab.json").read_text(encoding="utf-8")
  )
  label_vocab = json.loads(
    (saved_model_dir / "label_vocab.json").read_text(encoding="utf-8")
  )
  temperature_config = json.loads(
    (saved_model_dir / "temperature.json").read_text(encoding="utf-8")
  )
  label_entries = _label_entries(label_vocab)
  labels = tuple(entry["label_id"] for entry in label_entries)
  model_family = str(model_config["model_family"])
  model = _restore_classifier(
    model_family,
    labels=labels,
    model_config=model_config,
    device=device,
  )
  payload = torch.load(
    saved_model_dir / "model.pt",
    map_location="cpu",
    weights_only=False,
  )
  model.load_saved_state_dict(payload["state_dict"])
  return {
    "saved_model_dir": saved_model_dir,
    "model": model,
    "model_family": model_family,
    "model_config": model_config,
    "card_vocab": card_vocab,
    "zone_vocab": {
      str(zone): int(index) for zone, index in zone_vocab.items()
    },
    "label_metadata": {
      entry["label_id"]: entry for entry in label_entries
    },
    "temperature": float(temperature_config.get("temperature") or 1.0),
    "model_version": str(model_config["model_version"]),
  }


def _saved_model_configuration(
  *,
  model_name: str,
  model: Any,
  dataset: TrainingDataset,
  model_result: dict[str, Any],
  seed: int,
) -> dict[str, Any]:
  base_config = dict(model_result.get("model_config", {}))
  return {
    "format_version": SAVED_MODEL_FORMAT_VERSION,
    "model_family": model_name,
    "model_version": _model_version(model_name, dataset.dataset_version, seed),
    "dataset_version": dataset.dataset_version,
    "target_source": dataset.target_source,
    "seed": seed,
    "card_count": dataset.card_count,
    "zone_count": dataset.zone_count,
    "quantity_count": int(
      getattr(
        model, "quantity_count", max(1, max_quantity(dataset.examples) + 1)
      )
    ),
    "label_count": len(model.labels),
    "pooling": base_config.get("pooling", getattr(model, "pooling", None)),
    "token_scope": base_config.get("token_scope", "all"),
    "architecture": base_config.get(
      "architecture",
      getattr(model, "architecture", "base"),
    ),
    "dropout": base_config.get("dropout", getattr(model, "dropout", 0.0)),
    "label_smoothing": base_config.get(
      "label_smoothing",
      getattr(model, "label_smoothing", 0.0),
    ),
    "embedding_dim": base_config.get("embedding_dim"),
    "hidden_dim": base_config.get("hidden_dim"),
    "attention_heads": base_config.get("attention_heads"),
    "attention_layers": base_config.get("attention_layers"),
    "hypergeometric_draw_count": base_config.get(
      "hypergeometric_draw_count",
      getattr(model, "hypergeometric_draw_count", None),
    ),
    "a8_objectives": base_config.get(
      "a8_objectives",
      getattr(model, "a8_objective_config", lambda: {})(),
    ),
    "observation_conditioning": base_config.get(
      "observation_conditioning",
      base_config.get("partial_observation", {}).get(
        "observation_conditioning",
        getattr(model, "observation_conditioning", False),
      ),
    ),
    "rho_hidden_dim": base_config.get(
      "rho_hidden_dim",
      getattr(model, "rho_hidden_dim", None),
    ),
    "weight_decay": base_config.get(
      "weight_decay",
      model_result.get("weight_decay"),
    ),
    "learning_rate": base_config.get(
      "learning_rate",
      model_result.get("learning_rate"),
    ),
    "temperature": base_config.get("temperature", 1.0),
    "package_mining": base_config.get("package_mining", {"enabled": False}),
    "package_count": int(getattr(model, "package_count", 0)),
    "package_projection_dim": int(getattr(model, "package_projection_dim", 0)),
    "package_projection_bias": bool(
      getattr(model, "package_projection_bias", False)
    ),
    "package_scale": float(getattr(model, "package_scale", 1.0)),
    "extra_feature_dim": int(getattr(model, "extra_feature_dim", 0)),
    "extra_feature_value": float(getattr(model, "extra_feature_value", 0.0)),
    "preserve_base_rho_init": bool(
      getattr(model, "preserve_base_rho_init", False)
    ),
  }


def _saved_model_training_manifest(
  *,
  training_result: dict[str, Any],
  model_name: str,
  model_result: dict[str, Any],
  saved_model_dir: Path,
  family_relations: Path | None,
) -> dict[str, Any]:
  return {
    "run_id": "saved_model_export",
    "saved_model_dir": str(saved_model_dir),
    "source_training_run_id": training_result.get("run_id"),
    "dataset_path": training_result.get("dataset_path"),
    "dataset_version": training_result.get("dataset_version"),
    "formats": training_result.get("formats"),
    "target_source": training_result.get("target_source"),
    "training_target": training_result.get("training_target"),
    "family_relations": (
      {
        "path": family_relations.name,
        "sha256": sha256_file(family_relations),
      }
      if family_relations is not None
      else None
    ),
    "saved_model_export": training_result.get("saved_model_export"),
    "fit_scope": training_result.get("fit_scope"),
    "saved_model_calibration": (
      {
        key: value
        for key, value in model_result.get("saved_model_calibration", {}).items()
        if key != "calibration"
      }
      or None
    ),
    "model_family": model_name,
    "seed": model_result.get("seed"),
    "epochs": model_result.get("epochs"),
    "batch_size": model_result.get("batch_size"),
    "max_steps": model_result.get("max_steps"),
    "best_validation_epoch": model_result.get("best_validation_epoch"),
    "best_validation_metric": model_result.get("best_validation_metric"),
    "primary_evaluation_split": model_result.get("primary_evaluation_split"),
    "primary_metrics_at_best_validation_epoch": model_result.get(
      "primary_metrics_at_best_validation_epoch"
    ),
  }


def _copy_card_vocab(dataset: TrainingDataset, output: Path) -> None:
  manifest = json.loads(
    (dataset.dataset_path / "dataset_manifest.json").read_text(
      encoding="utf-8"
    )
  )
  source = dataset.dataset_path / manifest["artifacts"]["card_vocab"]
  shutil.copy2(source, output)


def _label_vocab_entries(
  dataset: TrainingDataset,
  labels: tuple[str, ...],
) -> list[dict[str, Any]]:
  manifest = json.loads(
    (dataset.dataset_path / "dataset_manifest.json").read_text(
      encoding="utf-8"
    )
  )
  dataset_files = manifest["artifacts"]
  proxy_targets = _read_proxy_targets_if_present(
    dataset.dataset_path,
    dataset_files,
  )
  display_by_label_id: dict[str, str] = {}
  target_source_by_label_id: dict[str, str | None] = {}
  for target in proxy_targets:
    label_id = str(target["proxy_label_id"])
    if target.get("display_label") is not None:
      display_by_label_id.setdefault(label_id, str(target["display_label"]))
    if target.get("target_source") is not None:
      target_source_by_label_id.setdefault(
        label_id, str(target["target_source"])
      )
  return [
    {
      "label_id": label_id,
      "display_label": display_by_label_id.get(
        label_id,
        _display_label_from_id(label_id),
      ),
      "target_source": target_source_by_label_id.get(
        label_id,
        dataset.target_source,
      ),
    }
    for label_id in labels
  ]


def _label_entries(label_vocab: dict[str, Any]) -> list[dict[str, Any]]:
  entries = label_vocab.get("entries")
  if isinstance(entries, list) and entries:
    return [
      {
        "label_id": str(entry["label_id"]),
        "display_label": str(
          entry.get("display_label")
          or _display_label_from_id(str(entry["label_id"]))
        ),
        "target_source": (
          str(entry["target_source"])
          if entry.get("target_source") is not None
          else None
        ),
      }
      for entry in entries
    ]
  return [
    {
      "label_id": str(label_id),
      "display_label": _display_label_from_id(str(label_id)),
      "target_source": None,
    }
    for label_id in label_vocab["labels"]
  ]


def _read_proxy_targets_if_present(
  dataset_path: Path,
  dataset_files: dict[str, Any],
) -> list[dict[str, Any]]:
  relative_path = dataset_files.get("proxy_targets")
  if relative_path is None:
    return []
  path = dataset_path / relative_path
  if not path.exists():
    return []
  return pq.read_table(path).to_pylist()


def _restore_classifier(
  model_family: str,
  *,
  labels: tuple[str, ...],
  model_config: dict[str, Any],
  device: str,
) -> Any:
  if model_family in (
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
    MODEL_DEEPSETS_PLUSPLUS_REGULARIZED,
    MODEL_DEEPSETS_PLUSPLUS_REGULARIZED_LABEL_SMOOTHING_005,
    MODEL_DEEPSETS_PLUSPLUS_REGULARIZED_LABEL_SMOOTHING_010,
  ):
    if int(model_config.get("package_count") or 0):
      raise ValueError(
        "Package-conditioned saved models are not supported by model-score"
        " v0."
      )
    return DeepSetsClassifier(
      labels=labels,
      card_count=int(model_config["card_count"]),
      zone_count=int(model_config["zone_count"]),
      quantity_count=int(model_config["quantity_count"]),
      embedding_dim=int(model_config["embedding_dim"]),
      hidden_dim=int(model_config["hidden_dim"]),
      pooling=str(model_config.get("pooling") or POOLING_QUANTITY_WEIGHTED),
      learning_rate=float(model_config.get("learning_rate") or 0.0),
      weight_decay=float(model_config.get("weight_decay") or 0.0),
      seed=int(model_config.get("seed") or 13),
      device=device,
      rho_hidden_dim=(
        int(model_config["rho_hidden_dim"])
        if model_config.get("rho_hidden_dim") is not None
        else None
      ),
      extra_feature_dim=int(model_config.get("extra_feature_dim") or 0),
      extra_feature_value=float(
        model_config.get("extra_feature_value") or 0.0
      ),
      preserve_base_rho_init=bool(model_config.get("preserve_base_rho_init")),
      architecture=str(model_config.get("architecture") or "base"),
      dropout=float(model_config.get("dropout") or 0.0),
      label_smoothing=float(model_config.get("label_smoothing") or 0.0),
    )
  if model_family == MODEL_POOLED_LINEAR:
    return PooledLinearClassifier(
      labels=labels,
      card_count=int(model_config["card_count"]),
      zone_count=int(model_config["zone_count"]),
      learning_rate=float(model_config.get("learning_rate") or 0.0),
      weight_decay=float(model_config.get("weight_decay") or 0.0),
      seed=int(model_config.get("seed") or 13),
      device=device,
    )
  if model_family.startswith("set-transformer-"):
    a8_objectives = dict(model_config.get("a8_objectives") or {})
    return SetTransformerClassifier(
      labels=labels,
      card_count=int(model_config["card_count"]),
      zone_count=int(model_config["zone_count"]),
      quantity_count=int(model_config["quantity_count"]),
      embedding_dim=int(model_config["embedding_dim"]),
      hidden_dim=int(model_config["hidden_dim"]),
      attention_heads=int(model_config["attention_heads"]),
      attention_layers=int(model_config["attention_layers"]),
      pooling=str(model_config["pooling"]),
      learning_rate=float(model_config.get("learning_rate") or 0.0),
      weight_decay=float(model_config.get("weight_decay") or 0.0),
      label_smoothing=float(model_config.get("label_smoothing") or 0.0),
      hypergeometric_draw_count=int(
        model_config.get("hypergeometric_draw_count") or 7
      ),
      observation_conditioning=bool(
        model_config.get("observation_conditioning")
      ),
      supcon_loss_weight=float(a8_objectives.get("supcon_weight") or 0.0),
      anchor_gating=bool(a8_objectives.get("anchor_gating")),
      use_product_manifold=bool(a8_objectives.get("product_manifold")),
      product_manifold_tree_weight=float(
        a8_objectives.get("hyperbolic_family_weight") or 0.0
      ),
      causal_target_alpha=float(
        a8_objectives.get("causal_target_alpha") or 0.0
      ),
      adaptive_manifold_scaling=bool(
        a8_objectives.get("adaptive_manifold_scaling")
      ),
      adaptive_manifold_ramp_cards=float(
        a8_objectives.get("adaptive_manifold_ramp_cards") or 20.0
      ),
      residual_anchor_logits=bool(a8_objectives.get("residual_anchor_logits")),
      residual_anchor_gamma=float(
        a8_objectives.get("residual_anchor_gamma") or 0.25
      ),
      seed=int(model_config.get("seed") or 13),
      device=device,
    )
  raise ValueError(f"Unsupported saved model family: {model_family}")


def _display_label_from_id(label_id: str) -> str:
  slug = label_id.rsplit(".", 1)[-1]
  return " ".join(word.capitalize() for word in slug.split("_") if word)


def _model_version(model_name: str, dataset_version: str, seed: int) -> str:
  safe_dataset = dataset_version.replace("/", "_")
  return f"{model_name}:{safe_dataset}:seed-{seed}"

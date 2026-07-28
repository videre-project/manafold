from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from manafold.data.export import SOURCE_ARCHETYPE_PROXY
from manafold.data.validate import write_json
from manafold.models.data import (
  TrainingDataset,
  UNKNOWN_LABEL_ID,
  build_scoring_examples,
)
from manafold.models.deepsets import (
  POOLING_QUANTITY_WEIGHTED,
  DeepSetsClassifier,
  PooledLinearClassifier,
  PrototypeClassifier,
  SetTransformerClassifier,
  max_quantity,
)

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
MODEL_ARTIFACT_VERSION = "manafold_model_artifact_v0"


def save_model_artifact(
  artifact_dir: Path,
  *,
  model_name: str,
  model: Any,
  dataset: TrainingDataset,
  model_result: dict[str, Any],
  training_result: dict[str, Any],
  seed: int,
) -> dict[str, Any]:
  if not hasattr(model, "state_dict_for_artifact"):
    raise ValueError(f"Model {model_name!r} does not support artifact export.")
  artifact_dir.mkdir(parents=True, exist_ok=True)

  state_path = artifact_dir / "model.pt"
  torch.save(
    {
      "artifact_version": MODEL_ARTIFACT_VERSION,
      "model_family": model_name,
      "state_dict": model.state_dict_for_artifact(),
    },
    state_path,
  )
  model_config = _model_config(
    model_name=model_name,
    model=model,
    dataset=dataset,
    model_result=model_result,
    seed=seed,
  )
  write_json(artifact_dir / "model_config.json", model_config)
  write_json(
    artifact_dir / "label_vocab.json",
    {
      "labels": list(model.labels),
      "entries": _label_vocab_entries(dataset, model.labels),
    },
  )
  write_json(
    artifact_dir / "zone_vocab.json",
    dataset.zone_vocab,
  )
  write_json(
    artifact_dir / "temperature.json",
    {
      "temperature": model_config.get("temperature", 1.0),
      "selection_split": (
        model_result.get("metrics", {})
        .get("calibration", {})
        .get("selection_split")
      ),
      "calibration": model_result.get("metrics", {}).get("calibration"),
    },
  )
  _copy_card_vocab(dataset, artifact_dir / "card_vocab.parquet")
  training_manifest = _training_manifest(
    training_result=training_result,
    model_name=model_name,
    model_result=model_result,
    artifact_dir=artifact_dir,
  )
  write_json(artifact_dir / "training_manifest.json", training_manifest)
  return {
    "path": str(artifact_dir),
    "model_family": model_name,
    "artifact_version": MODEL_ARTIFACT_VERSION,
    "files": {
      "model": str(state_path),
      "model_config": str(artifact_dir / "model_config.json"),
      "card_vocab": str(artifact_dir / "card_vocab.parquet"),
      "label_vocab": str(artifact_dir / "label_vocab.json"),
      "zone_vocab": str(artifact_dir / "zone_vocab.json"),
      "temperature": str(artifact_dir / "temperature.json"),
      "training_manifest": str(artifact_dir / "training_manifest.json"),
    },
  }


def run_model_scoring(
  *,
  model_artifact: Path,
  dataset_path: Path,
  output: Path,
  target_source: str = SOURCE_ARCHETYPE_PROXY,
  batch_size: int = 1024,
  top_k: int = 3,
  low_confidence_threshold: float = 0.5,
  device: str = "auto",
  deck_embedding_output: Path | None = None,
  taxonomy_eval_version: str | None = None,
) -> dict[str, Any]:
  if top_k <= 0:
    raise ValueError("top_k must be positive.")
  if batch_size <= 0:
    raise ValueError("batch_size must be positive.")
  if not 0.0 <= low_confidence_threshold <= 1.0:
    raise ValueError("low_confidence_threshold must be between 0 and 1.")

  artifact = load_model_artifact(model_artifact, device=device)
  dataset = _load_scoring_dataset(
    dataset_path,
    target_source=target_source,
    artifact=artifact,
  )
  _validate_dataset_compatibility(artifact, dataset)
  examples = list(dataset.examples)
  model = artifact["model"]
  temperature = float(artifact["temperature"])
  prototype_stats, prototype_scoring = _prototype_novelty_scores(
    model,
    examples,
    batch_size=batch_size,
    top_k=top_k,
  )
  predictions = _score_examples(
    model,
    examples,
    label_metadata=artifact["label_metadata"],
    temperature=temperature,
    batch_size=batch_size,
    top_k=top_k,
    low_confidence_threshold=low_confidence_threshold,
    model_version=artifact["model_version"],
    taxonomy_eval_version=taxonomy_eval_version,
    prototype_stats=prototype_stats,
  )

  output.parent.mkdir(parents=True, exist_ok=True)
  pq.write_table(pa.Table.from_pylist(predictions), output)

  embedding_path: Path | None = deck_embedding_output
  embedding_count = 0
  if hasattr(model, "deck_embedding_rows"):
    embedding_path = (
      embedding_path
      or output.with_name(f"{output.stem}_deck_embeddings.parquet")
    )
    embedding_rows = model.deck_embedding_rows(
      examples,
      batch_size=batch_size,
    )
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(embedding_rows), embedding_path)
    embedding_count = len(embedding_rows)

  manifest = {
    "run_id": "model_scoring",
    "model_artifact": str(model_artifact),
    "dataset_path": str(dataset_path),
    "output": str(output),
    "deck_embedding_output": str(embedding_path) if embedding_path else None,
    "model_family": artifact["model_family"],
    "model_version": artifact["model_version"],
    "target_source": target_source,
    "prediction_count": len(predictions),
    "deck_embedding_count": embedding_count,
    "prototype_scoring": prototype_scoring,
    "top_k": top_k,
    "low_confidence_threshold": low_confidence_threshold,
    "taxonomy_eval_version": taxonomy_eval_version,
  }
  write_json(output.with_suffix(".manifest.json"), manifest)
  return manifest


def load_model_artifact(
  artifact_dir: Path,
  *,
  device: str = "auto",
) -> dict[str, Any]:
  artifact_dir = artifact_dir.resolve()
  model_config = json.loads(
    (artifact_dir / "model_config.json").read_text(encoding="utf-8")
  )
  card_vocab = tuple(
    pq.read_table(artifact_dir / "card_vocab.parquet").to_pylist()
  )
  zone_vocab = json.loads(
    (artifact_dir / "zone_vocab.json").read_text(encoding="utf-8")
  )
  label_vocab = json.loads(
    (artifact_dir / "label_vocab.json").read_text(encoding="utf-8")
  )
  temperature_config = json.loads(
    (artifact_dir / "temperature.json").read_text(encoding="utf-8")
  )
  label_entries = _label_entries_from_vocab(label_vocab)
  labels = tuple(entry["label_id"] for entry in label_entries)
  model_family = str(model_config["model_family"])
  model = _build_artifact_model(
    model_family,
    labels=labels,
    model_config=model_config,
    device=device,
  )
  payload = torch.load(
    artifact_dir / "model.pt",
    map_location="cpu",
    weights_only=False,
  )
  model.load_state_dict_from_artifact(payload["state_dict"])
  return {
    "artifact_dir": artifact_dir,
    "model": model,
    "model_family": model_family,
    "model_config": model_config,
    "card_vocab": card_vocab,
    "zone_vocab": {
      str(zone): int(index)
      for zone, index in zone_vocab.items()
    },
    "label_metadata": {
      entry["label_id"]: entry
      for entry in label_entries
    },
    "temperature": float(temperature_config.get("temperature") or 1.0),
    "model_version": str(model_config["model_version"]),
  }


def _model_config(
  *,
  model_name: str,
  model: Any,
  dataset: TrainingDataset,
  model_result: dict[str, Any],
  seed: int,
) -> dict[str, Any]:
  base_config = dict(model_result.get("model_config", {}))
  return {
    "artifact_version": MODEL_ARTIFACT_VERSION,
    "model_family": model_name,
    "model_version": _model_version(model_name, dataset.dataset_version, seed),
    "dataset_version": dataset.dataset_version,
    "target_source": dataset.target_source,
    "seed": seed,
    "card_count": dataset.card_count,
    "zone_count": dataset.zone_count,
    "quantity_count": int(
      getattr(model, "quantity_count", max(1, max_quantity(dataset.examples) + 1))
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


def _training_manifest(
  *,
  training_result: dict[str, Any],
  model_name: str,
  model_result: dict[str, Any],
  artifact_dir: Path,
) -> dict[str, Any]:
  return {
    "run_id": "model_artifact_export",
    "artifact_dir": str(artifact_dir),
    "source_training_run_id": training_result.get("run_id"),
    "dataset_path": training_result.get("dataset_path"),
    "dataset_version": training_result.get("dataset_version"),
    "formats": training_result.get("formats"),
    "target_source": training_result.get("target_source"),
    "training_target": training_result.get("training_target"),
    "model_artifact_export": training_result.get("model_artifact_export"),
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
    (dataset.dataset_path / "dataset_manifest.json").read_text(encoding="utf-8")
  )
  source = dataset.dataset_path / manifest["artifacts"]["card_vocab"]
  shutil.copy2(source, output)


def _label_vocab_entries(
  dataset: TrainingDataset,
  labels: tuple[str, ...],
) -> list[dict[str, Any]]:
  manifest = json.loads(
    (dataset.dataset_path / "dataset_manifest.json").read_text(encoding="utf-8")
  )
  proxy_targets = _read_proxy_targets_if_present(dataset.dataset_path, manifest["artifacts"])
  display_by_label_id: dict[str, str] = {}
  target_source_by_label_id: dict[str, str | None] = {}
  for target in proxy_targets:
    label_id = str(target["proxy_label_id"])
    if target.get("display_label") is not None:
      display_by_label_id.setdefault(label_id, str(target["display_label"]))
    if target.get("target_source") is not None:
      target_source_by_label_id.setdefault(label_id, str(target["target_source"]))
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


def _label_entries_from_vocab(label_vocab: dict[str, Any]) -> list[dict[str, Any]]:
  entries = label_vocab.get("entries")
  if isinstance(entries, list) and entries:
    return [
      {
        "label_id": str(entry["label_id"]),
        "display_label": str(
          entry.get("display_label") or _display_label_from_id(str(entry["label_id"]))
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


def _load_scoring_dataset(
  dataset_path: Path,
  *,
  target_source: str,
  artifact: dict[str, Any],
) -> TrainingDataset:
  manifest = json.loads(
    (dataset_path / "dataset_manifest.json").read_text(encoding="utf-8")
  )
  artifacts = manifest["artifacts"]
  artifact_card_vocab = tuple(artifact["card_vocab"])
  artifact_zone_vocab = dict(artifact["zone_vocab"])
  card_idx_by_oracle_id = {
    str(card["oracle_id"]): int(card["card_idx"])
    for card in artifact_card_vocab
  }
  remapped_tokens = []
  missing_oracle_ids: set[str] = set()
  for token in pq.read_table(
    dataset_path / artifacts["deck_tokens"],
  ).to_pylist():
    oracle_id = str(token["oracle_id"])
    card_idx = card_idx_by_oracle_id.get(oracle_id)
    if card_idx is None:
      missing_oracle_ids.add(oracle_id)
      continue
    zone_name = str(token.get("zone") or "")
    if zone_name not in artifact_zone_vocab:
      raise ValueError(
        "Scoring dataset references a zone that is absent from the model "
        f"artifact: {zone_name!r}."
      )
    remapped = dict(token)
    remapped["card_idx"] = card_idx
    remapped["zone_idx"] = artifact_zone_vocab[zone_name]
    remapped_tokens.append(remapped)
  if missing_oracle_ids:
    sample = ", ".join(sorted(missing_oracle_ids)[:10])
    raise ValueError(
      "Scoring dataset references cards that are absent from the model artifact "
      f"vocabulary: {sample}."
    )

  proxy_targets = _read_proxy_targets_if_present(dataset_path, artifacts)
  examples = build_scoring_examples(
    deck_tokens=remapped_tokens,
    split_manifest=pq.read_table(
      dataset_path / artifacts["split_manifest"],
    ).to_pylist(),
    proxy_targets=proxy_targets,
    target_source=target_source,
  )
  if str(artifact["model_config"].get("token_scope") or "all") == "mainboard":
    main_zone_idx = artifact_zone_vocab.get("main")
    if main_zone_idx is None:
      raise ValueError(
        "Mainboard-only artifact requires a 'main' zone in the artifact vocabulary."
      )
    examples = [
      replace(
        example,
        tokens=tuple(
          token
          for token in example.tokens
          if token.zone_idx == main_zone_idx
        ),
        expected_mainboard_size=sum(
          max(token.quantity, 0)
          for token in example.tokens
          if token.zone_idx == main_zone_idx
        ),
      )
      for example in examples
    ]
  labels = tuple(
    sorted(
      {
        example.target_label_id
        for example in examples
        if example.split_name == "train"
      }
    )
  )
  return TrainingDataset(
    dataset_path=dataset_path,
    dataset_version=str(manifest["dataset_version"]),
    target_source=target_source,
    card_vocab=artifact_card_vocab,
    zone_vocab=artifact_zone_vocab,
    examples=tuple(examples),
    labels=labels,
  )


def _read_proxy_targets_if_present(
  dataset_path: Path,
  artifacts: dict[str, Any],
) -> list[dict[str, Any]]:
  relative_path = artifacts.get("proxy_targets")
  if relative_path is None:
    return []
  path = dataset_path / relative_path
  if not path.exists():
    return []
  return pq.read_table(path).to_pylist()


def _build_artifact_model(
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
      raise ValueError("Package-conditioned artifacts are not supported by model-score v0.")
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
      extra_feature_value=float(model_config.get("extra_feature_value") or 0.0),
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
      seed=int(model_config.get("seed") or 13),
      device=device,
    )
  raise ValueError(f"Unsupported model artifact family: {model_family}")


def _validate_dataset_compatibility(
  artifact: dict[str, Any],
  dataset: TrainingDataset,
) -> None:
  config = artifact["model_config"]
  if dataset.card_count > int(config["card_count"]):
    raise ValueError(
      "Dataset card vocabulary is larger than the model artifact vocabulary "
      f"({dataset.card_count} > {config['card_count']})."
    )
  if dataset.zone_count > int(config["zone_count"]):
    raise ValueError(
      "Dataset zone vocabulary is larger than the model artifact vocabulary "
      f"({dataset.zone_count} > {config['zone_count']})."
    )
  # Inference clips copy counts to the final learned quantity embedding. This
  # matches training batches and the ONNX request adapter, and allows unusual
  # basic-land counts that were not present in the training split.


def _score_examples(
  model: Any,
  examples: list[Any],
  *,
  label_metadata: dict[str, dict[str, Any]],
  temperature: float,
  batch_size: int,
  top_k: int,
  low_confidence_threshold: float,
  model_version: str,
  taxonomy_eval_version: str | None,
  prototype_stats: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
  stats = model.predict_top_k_with_stats_many(
    examples,
    k=top_k,
    batch_size=batch_size,
  )
  logits = model.logits_many(examples, batch_size=batch_size)
  scaled_probabilities = torch.softmax(
    logits.float() / max(float(temperature), 1e-12),
    dim=1,
  )
  scaled_max = scaled_probabilities.max(dim=1).values.tolist()
  rows: list[dict[str, Any]] = []
  prototype_rows = prototype_stats or [None] * len(examples)
  for example, prediction_stats, temperature_probability, prototype_row in zip(
    examples,
    stats,
    scaled_max,
    prototype_rows,
    strict=True,
  ):
    top_predictions = prediction_stats["top_predictions"]
    top1 = top_predictions[0] if top_predictions else (None, 0.0)
    top1_label_id = top1[0]
    top_prediction_ids = [label for label, _ in top_predictions]
    rows.append({
      "deck_id": example.deck_id,
      "event_id": example.event_id,
      "event_date": (
        example.event_date.isoformat()
        if example.event_date is not None
        else None
      ),
      "format": example.format_code,
      "split_name": example.split_name,
      "target_source": example.target_source,
      "source_label": _display_label_or_none(
        example.source_label_id,
        label_metadata,
      ),
      "source_label_id": (
        None
        if example.source_label_id == UNKNOWN_LABEL_ID
        else example.source_label_id
      ),
      "top1_label": _display_label_or_none(top1_label_id, label_metadata),
      "top1_label_id": top1_label_id,
      "top1_probability": float(top1[1]),
      "top3_labels": [
        _display_label_or_none(label_id, label_metadata)
        for label_id in top_prediction_ids
      ],
      "top3_label_ids": top_prediction_ids,
      "top3_probabilities": [float(score) for _, score in top_predictions],
      "temperature_scaled_probability": float(temperature_probability),
      "energy_score": prediction_stats.get("energy"),
      "msp_score": prediction_stats.get("max_probability", top1[1]),
      "nearest_prototype_distance": (
        prototype_row.get("nearest_prototype_distance")
        if prototype_row is not None
        else None
      ),
      "prototype_margin": (
        prototype_row.get("prototype_margin")
        if prototype_row is not None
        else None
      ),
      "entropy": prediction_stats.get("entropy"),
      "normalized_entropy": prediction_stats.get("normalized_entropy"),
      "is_low_confidence": (
        float(temperature_probability) < low_confidence_threshold
      ),
      "embedding_id": example.deck_id,
      "embedding_path": None,
      "model_version": model_version,
      "taxonomy_eval_version": taxonomy_eval_version,
    })
  return rows


def _prototype_novelty_scores(
  model: Any,
  examples: list[Any],
  *,
  batch_size: int,
  top_k: int,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
  if not isinstance(model, DeepSetsClassifier):
    return None, {
      "enabled": False,
      "reason": "Model does not expose a Deep Sets deck encoder.",
    }
  train_examples = [
    example
    for example in examples
    if example.split_name == "train"
  ]
  if not train_examples:
    return None, {
      "enabled": False,
      "reason": "No train split examples are available for prototype scoring.",
    }
  try:
    prototype_model = PrototypeClassifier.from_deepsets(
      model,
      train_examples,
      batch_size=batch_size,
    )
  except ValueError as exc:
    return None, {
      "enabled": False,
      "reason": str(exc),
    }
  rows = prototype_model.predict_top_k_with_stats_many(
    examples,
    k=top_k,
    batch_size=batch_size,
  )
  populated_prototype_count = sum(
    1
    for count in prototype_model.prototype_counts.values()
    if count > 0
  )
  return rows, {
    "enabled": True,
    "prototype_count": len(prototype_model.prototype_counts),
    "populated_prototype_count": populated_prototype_count,
    "distance": prototype_model.distance,
    "train_example_count": sum(prototype_model.prototype_counts.values()),
  }


def _display_label_or_none(
  label_id: str | None,
  label_metadata: dict[str, dict[str, Any]],
) -> str | None:
  if label_id is None or label_id == UNKNOWN_LABEL_ID:
    return None
  return str(
    label_metadata.get(label_id, {}).get("display_label")
    or _display_label_from_id(label_id)
  )


def _display_label_from_id(label_id: str) -> str:
  slug = label_id.rsplit(".", 1)[-1]
  return " ".join(
    word.capitalize()
    for word in slug.split("_")
    if word
  )


def _model_version(model_name: str, dataset_version: str, seed: int) -> str:
  safe_dataset = dataset_version.replace("/", "_")
  return f"{model_name}:{safe_dataset}:seed-{seed}"

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from manafold.datasets.model_inputs import (
  TrainingDataset,
  UNKNOWN_LABEL_ID,
  build_scoring_inputs,
)
from manafold.datasets.mtgo.build import SOURCE_ARCHETYPE_PROXY
from manafold.models.classifiers import DeepSetsClassifier, PrototypeClassifier
from manafold.models.saved_model import load_saved_model
from manafold.serialization import write_json


def score_dataset(
  *,
  saved_model_dir: Path,
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

  saved_model = load_saved_model(saved_model_dir, device=device)
  dataset = load_dataset_for_scoring(
    dataset_path,
    target_source=target_source,
    saved_model=saved_model,
  )
  validate_dataset_for_saved_model(saved_model, dataset)
  examples = list(dataset.examples)
  model = saved_model["model"]
  temperature = float(saved_model["temperature"])
  prototype_stats, prototype_scoring = _prototype_novelty_scores(
    model,
    examples,
    batch_size=batch_size,
    top_k=top_k,
  )
  predictions = _score_examples(
    model,
    examples,
    label_metadata=saved_model["label_metadata"],
    temperature=temperature,
    batch_size=batch_size,
    top_k=top_k,
    low_confidence_threshold=low_confidence_threshold,
    model_version=saved_model["model_version"],
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
    "saved_model_dir": str(saved_model_dir),
    "dataset_path": str(dataset_path),
    "output": str(output),
    "deck_embedding_output": str(embedding_path) if embedding_path else None,
    "model_family": saved_model["model_family"],
    "model_version": saved_model["model_version"],
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


def load_dataset_for_scoring(
  dataset_path: Path,
  *,
  target_source: str,
  saved_model: dict[str, Any],
) -> TrainingDataset:
  manifest = json.loads(
    (dataset_path / "dataset_manifest.json").read_text(encoding="utf-8")
  )
  dataset_files = manifest["artifacts"]
  saved_card_vocab = tuple(saved_model["card_vocab"])
  saved_zone_vocab = dict(saved_model["zone_vocab"])
  card_idx_by_oracle_id = {
    str(card["oracle_id"]): int(card["card_idx"]) for card in saved_card_vocab
  }
  remapped_tokens = []
  for token in pq.read_table(
    dataset_path / dataset_files["deck_tokens"],
  ).to_pylist():
    oracle_id = str(token["oracle_id"])
    card_idx = card_idx_by_oracle_id.get(oracle_id)
    if card_idx is None:
      # Drop cards absent from the saved vocabulary. Later windows (forward
      # transfer / continuous learning) routinely introduce printings that
      # were not present at training time; failing hard would block Track F
      # and production-like scoring on future exports. Decks that lose every
      # token fall out of scoring inputs.
      continue
    zone_name = str(token.get("zone") or "")
    if zone_name not in saved_zone_vocab:
      raise ValueError(
        "Scoring dataset references a zone that is absent from the saved "
        f"model: {zone_name!r}."
      )
    remapped = dict(token)
    remapped["card_idx"] = card_idx
    remapped["zone_idx"] = saved_zone_vocab[zone_name]
    remapped_tokens.append(remapped)

  proxy_targets = _read_proxy_targets_if_present(dataset_path, dataset_files)
  examples = build_scoring_inputs(
    deck_tokens=remapped_tokens,
    split_manifest=pq.read_table(
      dataset_path / dataset_files["split_manifest"],
    ).to_pylist(),
    proxy_targets=proxy_targets,
    target_source=target_source,
  )
  if str(saved_model["model_config"].get("token_scope") or "all") == "mainboard":
    main_zone_idx = saved_zone_vocab.get("main")
    if main_zone_idx is None:
      raise ValueError(
        "A mainboard-only saved model requires a 'main' zone in its vocabulary."
      )
    examples = [
      replace(
        example,
        tokens=tuple(
          token for token in example.tokens if token.zone_idx == main_zone_idx
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
    card_vocab=saved_card_vocab,
    zone_vocab=saved_zone_vocab,
    examples=tuple(examples),
    labels=labels,
  )


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


def validate_dataset_for_saved_model(
  saved_model: dict[str, Any],
  dataset: TrainingDataset,
) -> None:
  config = saved_model["model_config"]
  if dataset.card_count > int(config["card_count"]):
    raise ValueError(
      "Dataset card vocabulary is larger than the saved model vocabulary "
      f"({dataset.card_count} > {config['card_count']})."
    )
  if dataset.zone_count > int(config["zone_count"]):
    raise ValueError(
      "Dataset zone vocabulary is larger than the saved model vocabulary "
      f"({dataset.zone_count} > {config['zone_count']})."
    )


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
    example for example in examples if example.split_name == "train"
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
    1 for count in prototype_model.prototype_counts.values() if count > 0
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
  return " ".join(word.capitalize() for word in slug.split("_") if word)

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from manafold.data.validate import write_json
from manafold.models.family_backoff import (
  SEMANTIC_AUTO_RELATIONS,
  aggregate_family_probabilities,
  build_family_mapping,
  extract_proposed_edges,
)
from manafold.models.data import UNKNOWN_LABEL_ID
from manafold.models.model_artifacts import (
  _load_scoring_dataset,
  _validate_dataset_compatibility,
  load_model_artifact,
)

SUPPORT_BUCKETS = (
  ("absent_from_train", 0, 0),
  ("train_support_1_9", 1, 9),
  ("train_support_10_49", 10, 49),
  ("train_support_50_99", 50, 99),
  ("train_support_ge_100", 100, None),
)


def evaluate_family_backoff(
  *,
  model_artifact: Path,
  dataset_path: Path,
  output: Path,
  family_relations: Path | None = None,
  split_name: str = "test",
  batch_size: int = 1024,
  device: str = "auto",
) -> dict[str, Any]:
  """Evaluate an artifact through its serving-time family policy."""
  artifact = load_model_artifact(model_artifact, device=device)
  target_source = str(artifact["model_config"].get("target_source") or "")
  dataset = _load_scoring_dataset(
    dataset_path,
    target_source=target_source,
    artifact=artifact,
  )
  _validate_dataset_compatibility(artifact, dataset)

  proposed_edges: tuple[dict[str, Any], ...] = ()
  if family_relations is not None:
    proposed_edges = extract_proposed_edges(
      json.loads(family_relations.read_text(encoding="utf-8"))
    )

  split_examples = [
    example
    for example in dataset.examples
    if example.split_name == split_name
    and example.target_label_id != UNKNOWN_LABEL_ID
  ]
  if not split_examples:
    raise ValueError(f"Dataset has no examples in split {split_name!r}.")

  all_label_ids = {
    example.target_label_id
    for example in dataset.examples
  }
  label_metadata = {
    label_id: artifact["label_metadata"].get(
      label_id,
      {
        "label_id": label_id,
        "display_label": _display_label_from_id(label_id),
        "target_source": target_source,
      },
    )
    for label_id in all_label_ids
  }
  family_by_label = build_family_mapping(
    label_metadata,
    proposed_edges=proposed_edges,
  )

  logits = artifact["model"].logits_many(
    split_examples,
    batch_size=batch_size,
  )
  probabilities = torch.softmax(
    logits.float() / max(float(artifact["temperature"]), 1e-12),
    dim=1,
  )
  model_labels = tuple(artifact["model"].labels)
  model_label_index = {
    label_id: index
    for index, label_id in enumerate(model_labels)
  }
  raw_predictions = [
    model_labels[index]
    for index in probabilities.argmax(dim=1).tolist()
  ]
  raw_correct = [
    actual.target_label_id == predicted
    for actual, predicted in zip(
      split_examples,
      raw_predictions,
      strict=True,
    )
  ]
  predicted_family_by_label = {
    label_id: family_by_label[label_id]
    for label_id in model_labels
  }
  families, family_probabilities = aggregate_family_probabilities(
    probabilities,
    labels=model_labels,
    family_by_label=predicted_family_by_label,
  )
  family_index = {family: index for index, family in enumerate(families)}
  actual_families = [
    family_by_label[example.target_label_id]
    for example in split_examples
  ]
  predicted_families = [
    families[index]
    for index in family_probabilities.argmax(dim=1).tolist()
  ]
  ranked_indexes = family_probabilities.topk(
    min(3, len(families)),
    dim=1,
  ).indices.tolist()
  top_families = [
    {families[index] for index in row}
    for row in ranked_indexes
  ]

  train_family_support = Counter(
    family_by_label[example.target_label_id]
    for example in dataset.examples
    if example.split_name == "train"
  )
  metrics = _classification_metrics(
    actual_families,
    predicted_families,
    top_families=top_families,
    represented_families=set(families),
  )
  raw_accuracy = sum(raw_correct) / len(raw_correct)
  if metrics["accuracy"] + 1e-12 < raw_accuracy:
    raise AssertionError(
      "Family aggregation reduced exact accuracy; evaluation and serving "
      "policies are inconsistent."
    )
  support_buckets = {}
  for name, minimum, maximum in SUPPORT_BUCKETS:
    indexes = [
      index
      for index, family in enumerate(actual_families)
      if _in_support_bucket(
        train_family_support.get(family, 0),
        minimum=minimum,
        maximum=maximum,
      )
    ]
    support_buckets[name] = _classification_metrics(
      [actual_families[index] for index in indexes],
      [predicted_families[index] for index in indexes],
      top_families=[top_families[index] for index in indexes],
      represented_families=set(families),
    )

  report = {
    "run_id": "family_backoff_release_evaluation_v1",
    "dataset_path": str(dataset_path),
    "model_artifact": str(model_artifact),
    "model_version": artifact["model_version"],
    "dataset_version": dataset.dataset_version,
    "format": split_examples[0].format_code,
    "target_source": target_source,
    "split_name": split_name,
    "policy": {
      "color_relaxation": True,
      "generic_marker_normalization": True,
      "accepted_relations": sorted(SEMANTIC_AUTO_RELATIONS),
      "family_relations": str(family_relations) if family_relations else None,
      "proposed_edge_count": len(proposed_edges),
    },
    "raw_label_count": len(model_labels),
    "family_count": len(families),
    "raw_exact": {
      "count": len(split_examples),
      "accuracy": raw_accuracy,
      "train_unseen_count": sum(
        example.target_label_id not in model_label_index
        for example in split_examples
      ),
    },
    "metrics": metrics,
    "support_buckets": support_buckets,
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  write_json(output, report)
  return report


def _classification_metrics(
  actual: list[str],
  predicted: list[str],
  *,
  top_families: list[set[str]],
  represented_families: set[str],
) -> dict[str, Any]:
  count = len(actual)
  if count == 0:
    return {
      "count": 0,
      "actual_family_count": 0,
      "accuracy": None,
      "closed_set_count": 0,
      "closed_set_accuracy": None,
      "top3_accuracy": None,
      "macro_f1": None,
      "unrepresented_family_count": 0,
    }

  correct = [left == right for left, right in zip(actual, predicted, strict=True)]
  represented = [family in represented_families for family in actual]
  closed_indexes = [
    index
    for index, is_represented in enumerate(represented)
    if is_represented
  ]
  labels = sorted(set(actual))
  f1_values = []
  for label in labels:
    true_positive = sum(
      left == label and right == label
      for left, right in zip(actual, predicted, strict=True)
    )
    false_positive = sum(
      left != label and right == label
      for left, right in zip(actual, predicted, strict=True)
    )
    false_negative = sum(
      left == label and right != label
      for left, right in zip(actual, predicted, strict=True)
    )
    denominator = (2 * true_positive) + false_positive + false_negative
    f1_values.append(
      (2 * true_positive / denominator)
      if denominator
      else 0.0
    )

  return {
    "count": count,
    "actual_family_count": len(set(actual)),
    "accuracy": sum(correct) / count,
    "closed_set_count": len(closed_indexes),
    "closed_set_accuracy": (
      sum(correct[index] for index in closed_indexes) / len(closed_indexes)
      if closed_indexes
      else None
    ),
    "top3_accuracy": sum(
      family in top
      for family, top in zip(actual, top_families, strict=True)
    ) / count,
    "macro_f1": sum(f1_values) / len(f1_values),
    "unrepresented_family_count": len({
      family
      for family in actual
      if family not in represented_families
    }),
  }


def _in_support_bucket(
  support: int,
  *,
  minimum: int,
  maximum: int | None,
) -> bool:
  return support >= minimum and (maximum is None or support <= maximum)


def _display_label_from_id(label_id: str) -> str:
  slug = label_id.rsplit(".", 1)[-1]
  return " ".join(word.capitalize() for word in slug.split("_") if word)

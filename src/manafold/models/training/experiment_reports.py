from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from manafold.datasets.model_inputs import DeckModelInput
from manafold.models.training.model_names import (
  MODEL_DEEPSETS_QUANTITY_WEIGHTED,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES,
  MODEL_POOLED_LINEAR,
  SPLIT_ORDER_INDEX,
)
from manafold.models.features.card_packages import PackageFeatureSet
from manafold.models.evaluation.summary_statistics import summary_statistics


def _paired_correctness_report(
  reference_pairing: list[dict[str, Any]],
  comparison_pairing: list[dict[str, Any]],
  *,
  seed: int,
  bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
  reference_by_deck = {
    str(row["deck_id"]): bool(row["is_correct"]) for row in reference_pairing
  }
  comparison_by_deck = {
    str(row["deck_id"]): bool(row["is_correct"]) for row in comparison_pairing
  }
  deck_ids = sorted(set(reference_by_deck) & set(comparison_by_deck))
  if not deck_ids:
    return {
      "count": 0,
      "reference_accuracy": None,
      "comparison_accuracy": None,
      "accuracy_delta": None,
      "bootstrap_accuracy_delta_ci_95": None,
      "mcnemar_chi_square": None,
      "mcnemar_p_value_approx": None,
      "reference_only_correct": 0,
      "comparison_only_correct": 0,
      "both_correct": 0,
      "both_wrong": 0,
    }

  reference_correct = [reference_by_deck[deck_id] for deck_id in deck_ids]
  comparison_correct = [comparison_by_deck[deck_id] for deck_id in deck_ids]
  deltas = [
    (1 if reference else 0) - (1 if comparison else 0)
    for reference, comparison in zip(
      reference_correct,
      comparison_correct,
      strict=True,
    )
  ]
  reference_only = sum(
    1
    for reference, comparison in zip(
      reference_correct,
      comparison_correct,
      strict=True,
    )
    if reference and not comparison
  )
  comparison_only = sum(
    1
    for reference, comparison in zip(
      reference_correct,
      comparison_correct,
      strict=True,
    )
    if comparison and not reference
  )
  both_correct = sum(
    1
    for reference, comparison in zip(
      reference_correct,
      comparison_correct,
      strict=True,
    )
    if reference and comparison
  )
  both_wrong = sum(
    1
    for reference, comparison in zip(
      reference_correct,
      comparison_correct,
      strict=True,
    )
    if not reference and not comparison
  )
  delta = sum(deltas) / len(deltas)
  bootstrap_ci = _bootstrap_mean_ci(
    deltas,
    seed=seed,
    iterations=bootstrap_iterations,
  )
  discordant = reference_only + comparison_only
  if discordant:
    chi_square = ((abs(reference_only - comparison_only) - 1) ** 2) / discordant
    p_value = math.erfc(math.sqrt(chi_square / 2))
  else:
    chi_square = None
    p_value = None

  return {
    "count": len(deck_ids),
    "reference_accuracy": sum(reference_correct) / len(deck_ids),
    "comparison_accuracy": sum(comparison_correct) / len(deck_ids),
    "accuracy_delta": delta,
    "bootstrap_accuracy_delta_ci_95": bootstrap_ci,
    "mcnemar_chi_square": chi_square,
    "mcnemar_p_value_approx": p_value,
    "reference_only_correct": reference_only,
    "comparison_only_correct": comparison_only,
    "both_correct": both_correct,
    "both_wrong": both_wrong,
  }


def _bootstrap_mean_ci(
  values: list[int],
  *,
  seed: int,
  iterations: int,
) -> list[float] | None:
  if not values or iterations <= 0:
    return None

  rng = random.Random(seed)
  bootstrapped: list[float] = []
  for _ in range(iterations):
    total = 0
    for _ in values:
      total += values[rng.randrange(len(values))]
    bootstrapped.append(total / len(values))
  bootstrapped.sort()
  lower_idx = int(0.025 * (len(bootstrapped) - 1))
  upper_idx = int(0.975 * (len(bootstrapped) - 1))
  return [bootstrapped[lower_idx], bootstrapped[upper_idx]]


def _multi_seed_summary(
  seed_runs: list[dict[str, Any]],
  *,
  primary_evaluation_split: str,
) -> dict[str, Any]:
  metric_paths = {
    "primary_accuracy": (
      "metrics",
      "splits",
      primary_evaluation_split,
      "accuracy",
    ),
    "primary_closed_set_accuracy": (
      "metrics",
      "evaluation_views",
      primary_evaluation_split,
      "closed_set_seen_labels",
      "accuracy",
    ),
    "primary_top_3_accuracy": (
      "metrics",
      "splits",
      primary_evaluation_split,
      "top_3_accuracy",
    ),
    "primary_macro_f1": (
      "metrics",
      "splits",
      primary_evaluation_split,
      "macro_f1",
    ),
    "primary_source_label_accuracy": (
      "source_label_evaluation",
      "metrics",
      "splits",
      primary_evaluation_split,
      "accuracy",
    ),
    "primary_source_label_closed_set_accuracy": (
      "source_label_evaluation",
      "metrics",
      "evaluation_views",
      primary_evaluation_split,
      "closed_set_seen_labels",
      "accuracy",
    ),
    "primary_source_label_top_3_accuracy": (
      "source_label_evaluation",
      "metrics",
      "splits",
      primary_evaluation_split,
      "top_3_accuracy",
    ),
    "primary_source_label_macro_f1": (
      "source_label_evaluation",
      "metrics",
      "splits",
      primary_evaluation_split,
      "macro_f1",
    ),
    "primary_canonical_accuracy": (
      "taxonomy_evaluation",
      "metrics",
      "splits",
      primary_evaluation_split,
      "accuracy",
    ),
    "primary_canonical_closed_set_accuracy": (
      "taxonomy_evaluation",
      "metrics",
      "evaluation_views",
      primary_evaluation_split,
      "closed_set_seen_labels",
      "accuracy",
    ),
    "primary_canonical_top_3_accuracy": (
      "taxonomy_evaluation",
      "metrics",
      "splits",
      primary_evaluation_split,
      "top_3_accuracy",
    ),
    "primary_canonical_macro_f1": (
      "taxonomy_evaluation",
      "metrics",
      "splits",
      primary_evaluation_split,
      "macro_f1",
    ),
    "primary_temperature_scaled_nll": (
      "metrics",
      "calibration",
      "temperature_scaled",
      "splits",
      primary_evaluation_split,
      "nll",
    ),
    "primary_temperature_scaled_brier": (
      "metrics",
      "calibration",
      "temperature_scaled",
      "splits",
      primary_evaluation_split,
      "brier",
    ),
    "primary_temperature_scaled_ece": (
      "metrics",
      "calibration",
      "temperature_scaled",
      "splits",
      primary_evaluation_split,
      "ece",
    ),
    "primary_msp_auroc": (
      "metrics",
      "abstention",
      "max_softmax_probability",
      "splits",
      primary_evaluation_split,
      "auroc",
    ),
    "primary_temperature_scaled_msp_auroc": (
      "metrics",
      "abstention",
      "temperature_scaled_max_softmax_probability",
      "splits",
      primary_evaluation_split,
      "auroc",
    ),
    "primary_energy_auroc": (
      "metrics",
      "abstention",
      "energy",
      "splits",
      primary_evaluation_split,
      "auroc",
    ),
    "primary_known_false_abstention_rate": (
      "metrics",
      "abstention",
      "max_softmax_probability",
      "splits",
      primary_evaluation_split,
      "known_false_abstention_rate",
    ),
    "primary_unknown_recall": (
      "metrics",
      "abstention",
      "max_softmax_probability",
      "splits",
      primary_evaluation_split,
      "unknown_recall",
    ),
    "primary_energy_known_false_abstention_rate": (
      "metrics",
      "abstention",
      "energy",
      "splits",
      primary_evaluation_split,
      "known_false_abstention_rate",
    ),
    "primary_energy_unknown_recall": (
      "metrics",
      "abstention",
      "energy",
      "splits",
      primary_evaluation_split,
      "unknown_recall",
    ),
    "primary_nearest_prototype_distance_auroc": (
      "metrics",
      "abstention",
      "nearest_prototype_distance",
      "splits",
      primary_evaluation_split,
      "auroc",
    ),
    "primary_nearest_prototype_distance_known_false_abstention_rate": (
      "metrics",
      "abstention",
      "nearest_prototype_distance",
      "splits",
      primary_evaluation_split,
      "known_false_abstention_rate",
    ),
    "primary_prototype_margin_auroc": (
      "metrics",
      "abstention",
      "prototype_margin",
      "splits",
      primary_evaluation_split,
      "auroc",
    ),
    "primary_prototype_margin_known_false_abstention_rate": (
      "metrics",
      "abstention",
      "prototype_margin",
      "splits",
      primary_evaluation_split,
      "known_false_abstention_rate",
    ),
    "temperature": ("metrics", "calibration", "temperature"),
  }
  summary: dict[str, Any] = {}
  for metric_name, path in metric_paths.items():
    summary[metric_name] = summary_statistics(
      [_nested_value(seed_run, path) for seed_run in seed_runs]
    )
  return summary


def _nested_value(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
  current: Any = row
  for key in path:
    if not isinstance(current, dict) or key not in current:
      return None
    current = current[key]
  if current is None:
    return None
  return float(current)


def _accuracy_comparison(
  pooled_linear_metrics: dict[str, Any],
  model_metrics: dict[str, Any],
) -> dict[str, Any]:
  rows: dict[str, Any] = {}
  split_names = sorted(
    set(pooled_linear_metrics.get("splits", {}))
    | set(model_metrics.get("splits", {})),
    key=lambda split_name: (
      SPLIT_ORDER_INDEX.get(split_name, 10_000),
      split_name,
    ),
  )
  for split_name in split_names:
    pooled_linear_accuracy = _split_accuracy(
      pooled_linear_metrics, split_name
    )
    model_accuracy = _split_accuracy(model_metrics, split_name)
    rows[split_name] = {
      "pooled_linear_accuracy": pooled_linear_accuracy,
      "model_accuracy": model_accuracy,
      "accuracy_delta": (
        model_accuracy - pooled_linear_accuracy
        if model_accuracy is not None and pooled_linear_accuracy is not None
        else None
      ),
      "closed_set_seen_labels": _view_accuracy_comparison(
        pooled_linear_metrics,
        model_metrics,
        split_name,
        "closed_set_seen_labels",
      ),
      "open_set_unseen_labels": _view_accuracy_comparison(
        pooled_linear_metrics,
        model_metrics,
        split_name,
        "open_set_unseen_labels",
      ),
    }

  return rows


def _paired_model_tests(result: dict[str, Any]) -> dict[str, Any]:
  reference_name = MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES
  if reference_name not in result["models"]:
    return {}

  comparisons = (
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES,
  )
  reference = result["models"][reference_name]
  reference_pairing = reference.get("prediction_pairing", [])
  split_name = str(result["primary_evaluation_split"])
  rows: dict[str, Any] = {}
  for comparison_name in comparisons:
    comparison = result["models"].get(comparison_name)
    if comparison is None:
      continue
    rows[f"{reference_name}__vs__{comparison_name}"] = {
      "split_name": split_name,
      "seed": result.get("primary_seed", result.get("seeds", [None])[0]),
      "reference_model": reference_name,
      "comparison_model": comparison_name,
      "paired_correctness": _paired_correctness_report(
        reference_pairing,
        comparison.get("prediction_pairing", []),
        seed=int(reference.get("primary_seed", reference.get("seed", 13))),
      ),
    }
  return rows


def _package_signal_by_label(
  result: dict[str, Any],
  *,
  train_label_support: dict[str, int],
  evaluation_examples: list[DeckModelInput],
  evaluation_split: str,
  package_features: PackageFeatureSet,
  card_vocab: tuple[dict[str, Any], ...],
  zone_vocab: dict[str, int],
) -> list[dict[str, Any]]:
  required_models = (
    MODEL_POOLED_LINEAR,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES,
  )
  if any(model_name not in result["models"] for model_name in required_models):
    return []

  per_label = {
    model_name: _per_label_accuracy_by_id(
      result["models"][model_name]["metrics"],
      split_name=evaluation_split,
    )
    for model_name in required_models
  }
  top_packages = _top_packages_by_label(
    evaluation_examples,
    package_features=package_features,
    card_vocab=card_vocab,
    zone_vocab=zone_vocab,
  )
  labels = sorted(
    set(per_label[MODEL_POOLED_LINEAR])
    | set(per_label[MODEL_DEEPSETS_QUANTITY_WEIGHTED])
    | set(per_label[MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES])
  )
  rows: list[dict[str, Any]] = []
  for label in labels:
    pooled = per_label[MODEL_POOLED_LINEAR].get(label, {})
    deepsets = per_label[MODEL_DEEPSETS_QUANTITY_WEIGHTED].get(label, {})
    packages = per_label[MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES].get(
      label, {}
    )
    deepsets_accuracy = deepsets.get("accuracy")
    package_accuracy = packages.get("accuracy")
    pooled_accuracy = pooled.get("accuracy")
    rows.append({
      "label_id": label,
      "evaluation_split": evaluation_split,
      "train_support": train_label_support.get(label, 0),
      "evaluation_support": packages.get(
        "count",
        deepsets.get("count", pooled.get("count", 0)),
      ),
      "pooled_linear_accuracy": pooled_accuracy,
      "deepsets_quantity_weighted_accuracy": deepsets_accuracy,
      "deepsets_quantity_weighted_packages_accuracy": package_accuracy,
      "package_delta_vs_deepsets": (
        package_accuracy - deepsets_accuracy
        if package_accuracy is not None and deepsets_accuracy is not None
        else None
      ),
      "package_delta_vs_pooled_linear": (
        package_accuracy - pooled_accuracy
        if package_accuracy is not None and pooled_accuracy is not None
        else None
      ),
      "top_active_packages": top_packages.get(label, []),
    })

  rows.sort(
    key=lambda row: (
      -_none_to_negative_infinity(row["package_delta_vs_deepsets"]),
      -int(row["evaluation_support"] or 0),
      row["label_id"],
    )
  )
  return rows


def _per_label_accuracy_by_id(
  metrics: dict[str, Any],
  *,
  split_name: str,
) -> dict[str, dict[str, Any]]:
  rows = (
    metrics.get("splits", {}).get(split_name, {}).get("per_label_accuracy", [])
  )
  return {str(row["label_id"]): row for row in rows}


def _top_packages_by_label(
  test_examples: list[DeckModelInput],
  *,
  package_features: PackageFeatureSet,
  card_vocab: tuple[dict[str, Any], ...],
  zone_vocab: dict[str, int],
  limit: int = 5,
) -> dict[str, list[dict[str, Any]]]:
  if not package_features:
    return {}

  counts_by_label: dict[str, dict[int, int]] = {}
  support_by_label: dict[str, int] = {}
  for example in test_examples:
    label = example.target_label_id
    support_by_label[label] = support_by_label.get(label, 0) + 1
    label_counts = counts_by_label.setdefault(label, {})
    for package_idx in package_features.activation_indexes(example):
      label_counts[package_idx] = label_counts.get(package_idx, 0) + 1

  card_name_by_idx = {
    int(card["card_idx"]): str(card.get("primary_name") or card["card_idx"])
    for card in card_vocab
  }
  zone_by_idx = {
    int(zone_idx): str(zone_name) for zone_name, zone_idx in zone_vocab.items()
  }
  features_by_idx = {
    feature.package_idx: feature for feature in package_features.features
  }
  rows: dict[str, list[dict[str, Any]]] = {}
  for label, package_counts in counts_by_label.items():
    label_support = support_by_label[label]
    rows[label] = [
      _package_label_row(
        features_by_idx[package_idx],
        activation_count=count,
        label_support=label_support,
        card_name_by_idx=card_name_by_idx,
        zone_by_idx=zone_by_idx,
      )
      for package_idx, count in sorted(
        package_counts.items(),
        key=lambda item: (-item[1], item[0]),
      )[:limit]
      if package_idx in features_by_idx
    ]
  return rows


def _package_label_row(
  feature: Any,
  *,
  activation_count: int,
  label_support: int,
  card_name_by_idx: dict[int, str],
  zone_by_idx: dict[int, str],
) -> dict[str, Any]:
  row = feature.to_dict()
  row["zone"] = zone_by_idx.get(feature.zone_idx, str(feature.zone_idx))
  row["card_names"] = [
    card_name_by_idx.get(card_idx, str(card_idx))
    for card_idx in feature.card_idxs
  ]
  row["label_activation_count"] = activation_count
  row["label_activation_rate"] = (
    activation_count / label_support if label_support else None
  )
  return row


def _none_to_negative_infinity(value: float | None) -> float:
  return float("-inf") if value is None else float(value)


def _split_accuracy(
  metrics: dict[str, Any],
  split_name: str,
) -> float | None:
  split_metrics = metrics.get("splits", {}).get(split_name)
  if split_metrics is None:
    return None
  return split_metrics.get("accuracy")


def _view_accuracy_comparison(
  pooled_linear_metrics: dict[str, Any],
  model_metrics: dict[str, Any],
  split_name: str,
  view_name: str,
) -> dict[str, float | None]:
  pooled_linear_accuracy = _view_accuracy(
    pooled_linear_metrics,
    split_name,
    view_name,
  )
  model_accuracy = _view_accuracy(
    model_metrics,
    split_name,
    view_name,
  )
  return {
    "pooled_linear_accuracy": pooled_linear_accuracy,
    "model_accuracy": model_accuracy,
    "accuracy_delta": (
      model_accuracy - pooled_linear_accuracy
      if model_accuracy is not None and pooled_linear_accuracy is not None
      else None
    ),
  }


def _view_accuracy(
  metrics: dict[str, Any],
  split_name: str,
  view_name: str,
) -> float | None:
  split_views = metrics.get("evaluation_views", {}).get(split_name, {})
  view_metrics = split_views.get(view_name)
  if view_metrics is None:
    return None
  return view_metrics.get("accuracy")


def _write_result(
  result: dict[str, Any],
  output: Path | None,
) -> dict[str, Any]:
  if output is None:
    return result

  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return result

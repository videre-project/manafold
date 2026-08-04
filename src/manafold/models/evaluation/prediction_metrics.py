from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import torch
from torch.nn import functional as F

SUPPORT_BUCKETS = (
  ("absent_from_train",     0,     0),
  ("train_support_1_9",     1,     9),
  ("train_support_10_49",  10,    49),
  ("train_support_50_99",  50,    99),
  ("train_support_ge_100", 100, None),
)


def classification_metrics(
  predictions: list[dict[str, Any]],
  *,
  train_label_support: dict[str, int] | None = None,
) -> dict[str, Any]:
  label_support = (
    dict(train_label_support)
    if train_label_support is not None
    else _label_support_from_training_predictions(predictions)
  )
  split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for prediction in predictions:
    split_rows[prediction["split_name"]].append(prediction)

  return {
    "prediction_count": len(predictions),
    "overall": _metric_block(predictions),
    "splits": {
      split_name: _metric_block(rows)
      for split_name, rows in sorted(split_rows.items())
    },
    "evaluation_views": {
      split_name: _evaluation_views(rows, label_support)
      for split_name, rows in sorted(split_rows.items())
    },
    "support_buckets": {
      split_name: _support_bucket_metrics(rows, label_support)
      for split_name, rows in sorted(split_rows.items())
    },
    "abstention": _abstention_report(split_rows, label_support),
  }


def calibration_metrics(
  *,
  logits: torch.Tensor | None,
  labels: tuple[str, ...],
  examples: list[Any],
  bin_count: int = 15,
) -> dict[str, Any]:
  if logits is None or not labels:
    return {
      "status": "skipped",
      "reason": "Model does not expose logits for calibration.",
    }
  if logits.shape != (len(examples), len(labels)):
    return {
      "status": "skipped",
      "reason": (
        "Logit shape does not match examples and label vocabulary "
        f"({tuple(logits.shape)} != ({len(examples)}, {len(labels)}))."
      ),
    }

  label_to_idx = {label: index for index, label in enumerate(labels)}
  targets = torch.tensor(
    [label_to_idx.get(example.target_label_id, -1) for example in examples],
    dtype=torch.long,
  )
  split_indexes: dict[str, list[int]] = {}
  for index, example in enumerate(examples):
    split_indexes.setdefault(example.split_name, []).append(index)

  validation_known_indexes = [
    index
    for index in split_indexes.get("validation", [])
    if int(targets[index].item()) >= 0
  ]
  temperature = _fit_temperature(
    logits.index_select(0, _index_tensor(validation_known_indexes)),
    targets.index_select(0, _index_tensor(validation_known_indexes)),
  )

  return {
    "status": "completed",
    "target_scope": "closed_set_train_labels",
    "selection_split": "validation",
    "temperature": temperature,
    "bin_count": bin_count,
    "selection": {
      "known_count": len(validation_known_indexes),
      "excluded_unknown_count": (
        len(split_indexes.get("validation", []))
        - len(validation_known_indexes)
      ),
    },
    "uncalibrated": {
      "splits": _calibration_splits(
        logits,
        targets,
        split_indexes,
        temperature=1.0,
        bin_count=bin_count,
      ),
    },
    "temperature_scaled": {
      "splits": _calibration_splits(
        logits,
        targets,
        split_indexes,
        temperature=temperature,
        bin_count=bin_count,
      ),
    },
  }


def annotate_temperature_scaled_confidence(
  predictions: list[dict[str, Any]],
  *,
  logits: torch.Tensor | None,
  calibration: dict[str, Any],
) -> None:
  temperature = calibration.get("temperature")
  if (
    logits is None
    or not isinstance(temperature, int | float)
    or temperature <= 0
  ):
    return
  if logits.shape[0] != len(predictions):
    return

  probabilities = torch.softmax(logits.float() / float(temperature), dim=1)
  max_probabilities = probabilities.max(dim=1).values.tolist()
  for prediction, max_probability in zip(
    predictions,
    max_probabilities,
    strict=True,
  ):
    prediction["temperature_scaled_max_softmax_probability"] = float(
      max_probability
    )


def _calibration_splits(
  logits: torch.Tensor,
  targets: torch.Tensor,
  split_indexes: dict[str, list[int]],
  *,
  temperature: float,
  bin_count: int,
) -> dict[str, dict[str, Any]]:
  return {
    split_name: _calibration_block(
      logits.index_select(0, _index_tensor(indexes)),
      targets.index_select(0, _index_tensor(indexes)),
      temperature=temperature,
      bin_count=bin_count,
    )
    for split_name, indexes in sorted(split_indexes.items())
  }


def _fit_temperature(
  logits: torch.Tensor,
  targets: torch.Tensor,
) -> float:
  if logits.numel() == 0 or targets.numel() == 0 or logits.shape[1] <= 1:
    return 1.0

  logits = logits.detach().float()
  targets = targets.detach().long()
  if not torch.isfinite(logits).all():
    return 1.0

  # Cross-entropy is convex in inverse temperature. Solve its bounded scalar
  # optimum directly so confident small splits cannot drive LBFGS unbounded.
  beta_low = 0.01  # T = 100
  beta_high = 20.0  # T = 0.05
  target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)

  def derivative(beta: float) -> float:
    probabilities = torch.softmax(logits * beta, dim=1)
    expected_logits = (probabilities * logits).sum(dim=1)
    return float((expected_logits - target_logits).mean().item())

  low_derivative = derivative(beta_low)
  high_derivative = derivative(beta_high)
  if low_derivative >= 0.0:
    return 100.0
  if high_derivative <= 0.0:
    return 0.05
  for _ in range(32):
    beta_mid = (beta_low + beta_high) / 2.0
    if derivative(beta_mid) > 0.0:
      beta_high = beta_mid
    else:
      beta_low = beta_mid
  temperature = 1.0 / ((beta_low + beta_high) / 2.0)
  return temperature if math.isfinite(temperature) else 1.0


def _calibration_block(
  logits: torch.Tensor,
  targets: torch.Tensor,
  *,
  temperature: float,
  bin_count: int,
) -> dict[str, Any]:
  known_mask = targets >= 0
  known_count = int(known_mask.sum().item())
  total_count = int(targets.numel())
  if known_count == 0:
    return {
      "count": 0,
      "excluded_unknown_count": total_count,
      "accuracy": None,
      "mean_confidence": None,
      "nll": None,
      "brier": None,
      "ece": None,
    }

  known_logits = logits[known_mask].float()
  known_targets = targets[known_mask].long()
  scaled_logits = known_logits / max(float(temperature), 1e-12)
  probabilities = torch.softmax(scaled_logits, dim=1)
  confidences, predictions = probabilities.max(dim=1)
  correct = predictions == known_targets
  one_hot_targets = F.one_hot(
    known_targets,
    num_classes=known_logits.shape[1],
  ).float()

  return {
    "count": known_count,
    "excluded_unknown_count": total_count - known_count,
    "accuracy": float(correct.float().mean().item()),
    "mean_confidence": float(confidences.mean().item()),
    "nll": float(F.cross_entropy(scaled_logits, known_targets).item()),
    "brier": float(
      ((probabilities - one_hot_targets) ** 2).sum(dim=1).mean().item()
    ),
    "ece": _expected_calibration_error(
      confidences,
      correct,
      bin_count=bin_count,
    ),
  }


def _expected_calibration_error(
  confidences: torch.Tensor,
  correct: torch.Tensor,
  *,
  bin_count: int,
) -> float:
  if confidences.numel() == 0 or bin_count <= 0:
    return 0.0
  ece = 0.0
  total_count = int(confidences.numel())
  for bin_index in range(bin_count):
    lower = bin_index / bin_count
    upper = (bin_index + 1) / bin_count
    if bin_index == 0:
      in_bin = (confidences >= lower) & (confidences <= upper)
    else:
      in_bin = (confidences > lower) & (confidences <= upper)
    bin_size = int(in_bin.sum().item())
    if bin_size == 0:
      continue
    bin_confidence = float(confidences[in_bin].mean().item())
    bin_accuracy = float(correct[in_bin].float().mean().item())
    ece += (bin_size / total_count) * abs(bin_confidence - bin_accuracy)
  return float(ece)


def _index_tensor(indexes: list[int]) -> torch.Tensor:
  return torch.tensor(indexes, dtype=torch.long)


def _metric_block(predictions: list[dict[str, Any]]) -> dict[str, Any]:
  if not predictions:
    return {
      "count": 0,
      "accuracy": None,
      "top_3_accuracy": None,
      "macro_f1": None,
      "label_coverage": {
        "actual_label_count": 0,
        "predicted_label_count": 0,
        "actual_labels": [],
        "predicted_labels": [],
        "actual_never_predicted": [],
        "predicted_never_actual": [],
      },
      "largest_confusion_pairs": [],
      "per_label_accuracy": [],
      "per_label_accuracy_top_20": [],
      "confidence": _confidence_summary(predictions),
    }

  correct = sum(1 for prediction in predictions if prediction["is_correct"])
  top_3_correct = sum(
    1 for prediction in predictions if prediction.get("is_top_3_correct")
  )
  return {
    "count": len(predictions),
    "accuracy": correct / len(predictions),
    "top_3_accuracy": top_3_correct / len(predictions),
    "macro_f1": _macro_f1(predictions),
    "label_coverage": _label_coverage(predictions),
    "largest_confusion_pairs": _largest_confusion_pairs(predictions),
    "per_label_accuracy": _per_label_accuracy(predictions, limit=None),
    "per_label_accuracy_top_20": _per_label_accuracy(predictions),
    "confidence": _confidence_summary(predictions),
  }


def _evaluation_views(
  predictions: list[dict[str, Any]],
  train_label_support: dict[str, int],
) -> dict[str, Any]:
  return {
    "overall_proxy_score": _metric_block(predictions),
    "closed_set_seen_labels": _metric_block(
      [
        prediction
        for prediction in predictions
        if _actual_label_support(prediction, train_label_support) > 0
      ]
    ),
    "open_set_unseen_labels": _metric_block(
      [
        prediction
        for prediction in predictions
        if _actual_label_support(prediction, train_label_support) == 0
      ]
    ),
  }


def _support_bucket_metrics(
  predictions: list[dict[str, Any]],
  train_label_support: dict[str, int],
) -> dict[str, Any]:
  rows: dict[str, Any] = {}
  for bucket_name, minimum, maximum in SUPPORT_BUCKETS:
    bucket_rows = [
      prediction
      for prediction in predictions
      if _support_in_range(
        _actual_label_support(prediction, train_label_support),
        minimum,
        maximum,
      )
    ]
    metrics = _metric_block(bucket_rows)
    metrics["train_support_range"] = {
      "minimum": minimum,
      "maximum": maximum,
    }
    rows[bucket_name] = metrics

  return rows


def _support_in_range(
  support: int,
  minimum: int,
  maximum: int | None,
) -> bool:
  if support < minimum:
    return False
  return maximum is None or support <= maximum


def _actual_label_support(
  prediction: dict[str, Any],
  train_label_support: dict[str, int],
) -> int:
  if "actual_label_train_support" in prediction:
    return int(prediction["actual_label_train_support"])
  actual_label = prediction.get("actual_label_id")
  if actual_label is None:
    return 0
  return int(train_label_support.get(str(actual_label), 0))


def _label_support_from_training_predictions(
  predictions: list[dict[str, Any]],
) -> dict[str, int]:
  counts: Counter[str] = Counter()
  for prediction in predictions:
    if prediction.get("split_name") != "train":
      continue
    actual_label = prediction.get("actual_label_id")
    if actual_label is not None:
      counts[str(actual_label)] += 1
  return dict(counts)


def _macro_f1(predictions: list[dict[str, Any]]) -> float | None:
  labels = sorted(
    {
      prediction["actual_label_id"]
      for prediction in predictions
      if prediction.get("actual_label_id") is not None
    }
  )
  if not labels:
    return None

  true_positive: Counter[str] = Counter()
  false_positive: Counter[str] = Counter()
  false_negative: Counter[str] = Counter()

  for prediction in predictions:
    actual = prediction.get("actual_label_id")
    predicted = prediction.get("predicted_label_id")
    if actual == predicted and actual is not None:
      true_positive[str(actual)] += 1
      continue
    if predicted is not None:
      false_positive[str(predicted)] += 1
    if actual is not None:
      false_negative[str(actual)] += 1

  scores: list[float] = []
  for label in labels:
    precision_denominator = true_positive[label] + false_positive[label]
    recall_denominator = true_positive[label] + false_negative[label]
    precision = (
      true_positive[label] / precision_denominator
      if precision_denominator
      else 0.0
    )
    recall = (
      true_positive[label] / recall_denominator
      if recall_denominator
      else 0.0
    )
    if precision + recall == 0:
      scores.append(0.0)
    else:
      scores.append(2 * precision * recall / (precision + recall))

  return sum(scores) / len(scores)


def _label_coverage(predictions: list[dict[str, Any]]) -> dict[str, Any]:
  actual_labels = sorted(
    {
      prediction["actual_label_id"]
      for prediction in predictions
      if prediction.get("actual_label_id") is not None
    }
  )
  predicted_labels = sorted(
    {
      prediction["predicted_label_id"]
      for prediction in predictions
      if prediction.get("predicted_label_id") is not None
    }
  )
  return {
    "actual_label_count": len(actual_labels),
    "predicted_label_count": len(predicted_labels),
    "actual_labels": actual_labels,
    "predicted_labels": predicted_labels,
    "actual_never_predicted": sorted(
      set(actual_labels) - set(predicted_labels)
    ),
    "predicted_never_actual": sorted(
      set(predicted_labels) - set(actual_labels)
    ),
  }


def _largest_confusion_pairs(
  predictions: list[dict[str, Any]],
  *,
  limit: int = 10,
) -> list[dict[str, Any]]:
  counts: Counter[tuple[str, str]] = Counter()
  for prediction in predictions:
    actual = prediction.get("actual_label_id")
    predicted = prediction.get("predicted_label_id")
    if actual is None or predicted is None or actual == predicted:
      continue
    counts[(str(actual), str(predicted))] += 1

  return [
    {
      "actual_label_id": actual,
      "predicted_label_id": predicted,
      "count": count,
    }
    for (actual, predicted), count in counts.most_common(limit)
  ]


def _per_label_accuracy(
  predictions: list[dict[str, Any]],
  *,
  limit: int | None = 20,
) -> list[dict[str, Any]]:
  counts: Counter[str] = Counter()
  correct: Counter[str] = Counter()
  for prediction in predictions:
    actual = prediction.get("actual_label_id")
    if actual is None:
      continue
    label = str(actual)
    counts[label] += 1
    if prediction.get("is_correct"):
      correct[label] += 1

  rows = [
    {
      "label_id": label,
      "count": count,
      "accuracy": correct[label] / count if count else None,
    }
    for label, count in counts.most_common(limit)
  ]
  if limit is None:
    rows = [
      {
        "label_id": label,
        "count": counts[label],
        "accuracy": correct[label] / counts[label] if counts[label] else None,
      }
      for label in sorted(counts)
    ]
  return rows


def _confidence_summary(
  predictions: list[dict[str, Any]],
) -> dict[str, float | None]:
  scores = sorted(
    float(prediction["score"])
    for prediction in predictions
    if prediction.get("score") is not None
  )
  if not scores:
    return {
      "mean_max_score": None,
      "median_max_score": None,
      "mean_entropy": None,
      "median_entropy": None,
      "mean_normalized_entropy": None,
      "median_normalized_entropy": None,
      "mean_energy": None,
      "median_energy": None,
    }

  entropies = sorted(
    float(prediction["entropy"])
    for prediction in predictions
    if prediction.get("entropy") is not None
  )
  normalized_entropies = sorted(
    float(prediction["normalized_entropy"])
    for prediction in predictions
    if prediction.get("normalized_entropy") is not None
  )
  energies = sorted(
    float(prediction["energy"])
    for prediction in predictions
    if prediction.get("energy") is not None
  )
  return {
    "mean_max_score": sum(scores) / len(scores),
    "median_max_score": _median(scores),
    "mean_entropy": (
      sum(entropies) / len(entropies) if entropies else None
    ),
    "median_entropy": _median(entropies) if entropies else None,
    "mean_normalized_entropy": (
      sum(normalized_entropies) / len(normalized_entropies)
      if normalized_entropies
      else None
    ),
    "median_normalized_entropy": (
      _median(normalized_entropies) if normalized_entropies else None
    ),
    "mean_energy": (sum(energies) / len(energies) if energies else None),
    "median_energy": _median(energies) if energies else None,
  }


def _median(values: list[float]) -> float:
  midpoint = len(values) // 2
  if len(values) % 2:
    return values[midpoint]
  return (values[midpoint - 1] + values[midpoint]) / 2


def _abstention_report(
  split_rows: dict[str, list[dict[str, Any]]],
  train_label_support: dict[str, int],
) -> dict[str, Any]:
  report = {
    "max_softmax_probability": _score_abstention_report(
      split_rows,
      train_label_support,
      score_key="max_softmax_probability",
      higher_means_unknown=False,
    ),
    "entropy": _score_abstention_report(
      split_rows,
      train_label_support,
      score_key="entropy",
      higher_means_unknown=True,
    ),
    "energy": _score_abstention_report(
      split_rows,
      train_label_support,
      score_key="energy",
      higher_means_unknown=True,
    ),
  }
  if _has_score(split_rows, "temperature_scaled_max_softmax_probability"):
    report["temperature_scaled_max_softmax_probability"] = (
      _score_abstention_report(
        split_rows,
        train_label_support,
        score_key="temperature_scaled_max_softmax_probability",
        higher_means_unknown=False,
      )
    )
  if _has_score(split_rows, "nearest_prototype_distance"):
    report["nearest_prototype_distance"] = _score_abstention_report(
      split_rows,
      train_label_support,
      score_key="nearest_prototype_distance",
      higher_means_unknown=True,
    )
  if _has_score(split_rows, "prototype_margin"):
    report["prototype_margin"] = _score_abstention_report(
      split_rows,
      train_label_support,
      score_key="prototype_margin",
      higher_means_unknown=False,
    )
  return report


def _has_score(
  split_rows: dict[str, list[dict[str, Any]]],
  score_key: str,
) -> bool:
  return any(
    prediction.get(score_key) is not None
    for rows in split_rows.values()
    for prediction in rows
  )


def _score_abstention_report(
  split_rows: dict[str, list[dict[str, Any]]],
  train_label_support: dict[str, int],
  *,
  score_key: str,
  higher_means_unknown: bool,
) -> dict[str, Any]:
  validation_rows = split_rows.get("validation", [])
  threshold = _select_abstention_threshold(
    validation_rows,
    train_label_support,
    score_key=score_key,
    higher_means_unknown=higher_means_unknown,
  )
  return {
    "score_key": score_key,
    "unknown_direction": "higher" if higher_means_unknown else "lower",
    "selection_split": "validation",
    "selection_metric": "balanced_unknown_recall_known_acceptance",
    "validation_selected_threshold": threshold,
    "splits": {
      split_name: _thresholded_abstention_metrics(
        rows,
        train_label_support,
        score_key=score_key,
        threshold=threshold,
        higher_means_unknown=higher_means_unknown,
      )
      for split_name, rows in sorted(split_rows.items())
    },
  }


def _select_abstention_threshold(
  predictions: list[dict[str, Any]],
  train_label_support: dict[str, int],
  *,
  score_key: str,
  higher_means_unknown: bool,
) -> float | None:
  rows = _unknown_scored_rows(
    predictions,
    train_label_support,
    score_key=score_key,
    higher_means_unknown=higher_means_unknown,
  )
  known_count = sum(1 for row in rows if not row["is_unknown"])
  unknown_count = len(rows) - known_count
  if not rows or known_count == 0 or unknown_count == 0:
    return None

  scores = torch.tensor(
    [row["unknown_score"] for row in rows],
    dtype=torch.float64,
  )
  unknown = torch.tensor(
    [row["is_unknown"] for row in rows],
    dtype=torch.int64,
  )
  correct_known = torch.tensor(
    [not row["is_unknown"] and row["is_correct"] for row in rows],
    dtype=torch.int64,
  )
  order = torch.argsort(scores, descending=True, stable=True)
  ordered_scores = scores.index_select(0, order)
  ordered_unknown = unknown.index_select(0, order)
  ordered_correct_known = correct_known.index_select(0, order)
  group_ends = torch.cat(
    (
      torch.nonzero(
        ordered_scores[1:] != ordered_scores[:-1],
        as_tuple=False,
      ).flatten()
      + 1,
      torch.tensor([len(rows)], dtype=torch.long),
    )
  )
  prefix_unknown = torch.cumsum(ordered_unknown, dim=0).index_select(
    0,
    group_ends - 1,
  )
  prefix_known_correct = torch.cumsum(
    ordered_correct_known,
    dim=0,
  ).index_select(0, group_ends - 1)
  group_unknown = torch.diff(
    torch.cat((torch.tensor([0]), prefix_unknown))
  )
  group_known_correct = torch.diff(
    torch.cat((torch.tensor([0]), prefix_known_correct))
  )
  group_sizes = torch.diff(torch.cat((torch.tensor([0]), group_ends)))
  group_known = group_sizes - group_unknown
  unknown_abstained_values = torch.cumsum(group_unknown, dim=0).tolist()
  known_abstained_values = torch.cumsum(group_known, dim=0).tolist()
  known_correct_abstained_values = torch.cumsum(
    group_known_correct,
    dim=0,
  ).tolist()
  group_scores = ordered_scores.index_select(0, group_ends - 1).tolist()
  max_score = float(ordered_scores[0].item())
  min_score = float(ordered_scores[-1].item())

  best_threshold: float | None = None
  best_key: tuple[float, float, float] | None = None
  total_known_correct = int(correct_known.sum().item())
  epsilon = 1e-12

  candidates = [
    (max_score + epsilon, 0, 0, 0),
  ]
  for index, score in enumerate(group_scores):
    next_threshold = (
      (score + group_scores[index + 1]) / 2
      if index + 1 < len(group_scores)
      else min_score - epsilon
    )
    candidates.append(
      (
        next_threshold,
        int(unknown_abstained_values[index]),
        int(known_abstained_values[index]),
        int(known_correct_abstained_values[index]),
      )
    )
  for (
    unknown_threshold,
    unknown_abstained,
    known_abstained,
    known_correct_abstained,
  ) in candidates:
    known_accepted = known_count - known_abstained
    known_correct_accepted = total_known_correct - known_correct_abstained
    unknown_recall = unknown_abstained / unknown_count
    known_acceptance = 1.0 - (known_abstained / known_count)
    known_accuracy = (
      known_correct_accepted / known_accepted if known_accepted else 0.0
    )
    coverage = (
      known_accepted + (unknown_count - unknown_abstained)
    ) / len(rows)
    key = (
      (unknown_recall + known_acceptance) / 2,
      known_accuracy,
      coverage,
    )
    if best_key is None or key > best_key:
      best_key = key
      best_threshold = _original_threshold(
        unknown_threshold,
        higher_means_unknown=higher_means_unknown,
      )

  return best_threshold


def _original_threshold(
  unknown_threshold: float,
  *,
  higher_means_unknown: bool,
) -> float:
  return unknown_threshold if higher_means_unknown else -unknown_threshold


def _thresholded_abstention_metrics(
  predictions: list[dict[str, Any]],
  train_label_support: dict[str, int],
  *,
  score_key: str,
  threshold: float | None,
  higher_means_unknown: bool,
) -> dict[str, Any]:
  partition = _threshold_partition_metrics(
    predictions,
    train_label_support,
    score_key=score_key,
    threshold=threshold,
    higher_means_unknown=higher_means_unknown,
  )

  return {
    **partition,
    "threshold": threshold,
    "auroc": _unknown_detection_auroc(
      predictions,
      train_label_support,
      score_key=score_key,
      higher_means_unknown=higher_means_unknown,
    ),
    "aupr": _unknown_detection_aupr(
      predictions,
      train_label_support,
      score_key=score_key,
      higher_means_unknown=higher_means_unknown,
    ),
    "fpr_at_95_unknown_recall": (
      _known_false_abstention_rate_at_unknown_recall(
        predictions,
        train_label_support,
        score_key=score_key,
        higher_means_unknown=higher_means_unknown,
        target_unknown_recall=0.95,
      )
    ),
  }


def _threshold_partition_metrics(
  predictions: list[dict[str, Any]],
  train_label_support: dict[str, int],
  *,
  score_key: str,
  threshold: float | None,
  higher_means_unknown: bool,
) -> dict[str, Any]:
  known_count = 0
  unknown_count = 0
  accepted_count = 0
  abstained_count = 0
  known_accepted_count = 0
  known_abstained_count = 0
  unknown_abstained_count = 0
  known_correct_accepted = 0
  unknown_accepted_count = 0

  for prediction in predictions:
    is_known = _actual_label_support(prediction, train_label_support) > 0
    should_abstain = _should_abstain(
      prediction,
      score_key=score_key,
      threshold=threshold,
      higher_means_unknown=higher_means_unknown,
    )
    if is_known:
      known_count += 1
    else:
      unknown_count += 1

    if should_abstain:
      abstained_count += 1
      if is_known:
        known_abstained_count += 1
      else:
        unknown_abstained_count += 1
      continue

    accepted_count += 1
    if is_known:
      known_accepted_count += 1
      if prediction.get("is_correct"):
        known_correct_accepted += 1
    else:
      unknown_accepted_count += 1

  return {
    "count": len(predictions),
    "known_count": known_count,
    "unknown_count": unknown_count,
    "accepted_count": accepted_count,
    "abstained_count": abstained_count,
    "known_accepted_count": known_accepted_count,
    "known_abstained_count": known_abstained_count,
    "unknown_accepted_count": unknown_accepted_count,
    "unknown_abstained_count": unknown_abstained_count,
    "coverage": _ratio(accepted_count, len(predictions)),
    "known_coverage": _ratio(known_accepted_count, known_count),
    "unknown_acceptance_rate": _ratio(unknown_accepted_count, unknown_count),
    "known_accuracy": _ratio(known_correct_accepted, known_accepted_count),
    "accepted_known_accuracy": _ratio(
      known_correct_accepted,
      known_accepted_count,
    ),
    "closed_set_accuracy_at_accepted_coverage": _ratio(
      known_correct_accepted,
      known_accepted_count,
    ),
    "known_accuracy_with_abstentions": _ratio(
      known_correct_accepted,
      known_count,
    ),
    "unknown_recall": _ratio(unknown_abstained_count, unknown_count),
    "known_false_abstention_rate": _ratio(
      known_abstained_count,
      known_count,
    ),
    "unknown_precision": _ratio(unknown_abstained_count, abstained_count),
  }


def _should_abstain(
  prediction: dict[str, Any],
  *,
  score_key: str,
  threshold: float | None,
  higher_means_unknown: bool,
) -> bool:
  if threshold is None or prediction.get(score_key) is None:
    return False
  value = float(prediction[score_key])
  if higher_means_unknown:
    return value >= threshold
  return value < threshold


def _unknown_detection_score(
  prediction: dict[str, Any],
  *,
  score_key: str,
  higher_means_unknown: bool,
) -> float | None:
  if prediction.get(score_key) is None:
    return None
  value = float(prediction[score_key])
  return value if higher_means_unknown else -value


def _unknown_scored_rows(
  predictions: list[dict[str, Any]],
  train_label_support: dict[str, int],
  *,
  score_key: str,
  higher_means_unknown: bool,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for prediction in predictions:
    unknown_score = _unknown_detection_score(
      prediction,
      score_key=score_key,
      higher_means_unknown=higher_means_unknown,
    )
    if unknown_score is None:
      continue
    rows.append({
      "unknown_score": unknown_score,
      "is_unknown": (
        _actual_label_support(
          prediction,
          train_label_support,
        )
        == 0
      ),
      "is_correct": bool(prediction.get("is_correct")),
    })
  return rows


def _unknown_detection_auroc(
  predictions: list[dict[str, Any]],
  train_label_support: dict[str, int],
  *,
  score_key: str,
  higher_means_unknown: bool,
) -> float | None:
  rows = _unknown_scored_rows(
    predictions,
    train_label_support,
    score_key=score_key,
    higher_means_unknown=higher_means_unknown,
  )
  positive_count = sum(1 for row in rows if row["is_unknown"])
  negative_count = len(rows) - positive_count

  if positive_count == 0 or negative_count == 0:
    return None

  rows.sort(key=lambda row: row["unknown_score"])
  rank_sum = 0.0
  index = 0
  while index < len(rows):
    end = index + 1
    while (
      end < len(rows)
      and rows[end]["unknown_score"] == rows[index]["unknown_score"]
    ):
      end += 1
    average_rank = (index + 1 + end) / 2
    positive_in_group = sum(1 for row in rows[index:end] if row["is_unknown"])
    rank_sum += positive_in_group * average_rank
    index = end

  return (rank_sum - (positive_count * (positive_count + 1) / 2)) / (
    positive_count * negative_count
  )


def _known_false_abstention_rate_at_unknown_recall(
  predictions: list[dict[str, Any]],
  train_label_support: dict[str, int],
  *,
  score_key: str,
  higher_means_unknown: bool,
  target_unknown_recall: float,
) -> dict[str, Any]:
  rows = _unknown_scored_rows(
    predictions,
    train_label_support,
    score_key=score_key,
    higher_means_unknown=higher_means_unknown,
  )
  known_count = sum(1 for row in rows if not row["is_unknown"])
  unknown_count = len(rows) - known_count
  if known_count == 0 or unknown_count == 0:
    return {
      "target_unknown_recall": target_unknown_recall,
      "known_false_abstention_rate": None,
      "threshold": None,
      "unknown_recall": None,
    }

  rows.sort(key=lambda row: row["unknown_score"], reverse=True)
  unknown_abstained = 0
  known_abstained = 0
  index = 0
  epsilon = 1e-12
  while index < len(rows):
    score = rows[index]["unknown_score"]
    while index < len(rows) and rows[index]["unknown_score"] == score:
      if rows[index]["is_unknown"]:
        unknown_abstained += 1
      else:
        known_abstained += 1
      index += 1

    unknown_recall = unknown_abstained / unknown_count
    if unknown_recall >= target_unknown_recall:
      unknown_threshold = (
        (score + rows[index]["unknown_score"]) / 2
        if index < len(rows)
        else score - epsilon
      )
      return {
        "target_unknown_recall": target_unknown_recall,
        "known_false_abstention_rate": known_abstained / known_count,
        "threshold": _original_threshold(
          unknown_threshold,
          higher_means_unknown=higher_means_unknown,
        ),
        "unknown_recall": unknown_recall,
      }

  return {
    "target_unknown_recall": target_unknown_recall,
    "known_false_abstention_rate": 1.0,
    "threshold": _original_threshold(
      rows[-1]["unknown_score"],
      higher_means_unknown=higher_means_unknown,
    ),
    "unknown_recall": unknown_abstained / unknown_count,
  }


def _unknown_detection_aupr(
  predictions: list[dict[str, Any]],
  train_label_support: dict[str, int],
  *,
  score_key: str,
  higher_means_unknown: bool,
) -> float | None:
  rows: list[tuple[float, bool]] = []
  for prediction in predictions:
    score = _unknown_detection_score(
      prediction,
      score_key=score_key,
      higher_means_unknown=higher_means_unknown,
    )
    if score is None:
      continue
    rows.append((
      score,
      _actual_label_support(prediction, train_label_support) == 0,
    ))
  positive_count = sum(1 for _, is_positive in rows if is_positive)
  if positive_count == 0:
    return None

  rows.sort(key=lambda row: row[0], reverse=True)
  true_positive = 0
  precision_sum = 0.0
  for index, (_, is_positive) in enumerate(rows, start=1):
    if not is_positive:
      continue
    true_positive += 1
    precision_sum += true_positive / index

  return precision_sum / positive_count


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
  if denominator == 0:
    return None
  return numerator / denominator

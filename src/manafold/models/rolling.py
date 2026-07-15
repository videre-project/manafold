from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from manafold.data.export import DatasetExportOptions, export_dataset
from manafold.data.validate import verify_dataset_export, write_json
from manafold.models.deepsets import POOLING_SUM
from manafold.models.train import (
  DEFAULT_LEARNING_RATE,
  DEFAULT_MODEL_SET,
  DEFAULT_REGULARIZED_WEIGHT_DECAY,
  MODEL_ALL,
  PREDICTION_OUTPUT_ERRORS,
  run_model_training,
)

ROLLING_PRIMARY_METRICS = (
  "primary_accuracy",
  "primary_closed_set_accuracy",
  "primary_top_3_accuracy",
  "primary_macro_f1",
  "primary_source_label_accuracy",
  "primary_source_label_closed_set_accuracy",
  "primary_source_label_top_3_accuracy",
  "primary_source_label_macro_f1",
  "primary_canonical_accuracy",
  "primary_canonical_closed_set_accuracy",
  "primary_canonical_top_3_accuracy",
  "primary_canonical_macro_f1",
  "primary_temperature_scaled_nll",
  "primary_temperature_scaled_brier",
  "primary_temperature_scaled_ece",
  "primary_msp_auroc",
  "primary_temperature_scaled_msp_auroc",
  "primary_energy_auroc",
  "primary_known_false_abstention_rate",
  "primary_unknown_recall",
  "primary_energy_known_false_abstention_rate",
  "primary_energy_unknown_recall",
)
ROLLING_RANKING_METRICS = (
  "primary_accuracy",
  "primary_closed_set_accuracy",
  "primary_macro_f1",
  "primary_source_label_accuracy",
  "primary_canonical_accuracy",
  "primary_canonical_macro_f1",
  "primary_temperature_scaled_nll",
  "primary_energy_auroc",
)
LOWER_IS_BETTER_METRICS = {
  "primary_temperature_scaled_nll",
  "primary_temperature_scaled_brier",
  "primary_temperature_scaled_ece",
  "primary_known_false_abstention_rate",
  "primary_energy_known_false_abstention_rate",
}
POOLED_LINEAR_MODEL_NAME = "pooled-linear"

_WINDOW_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class RollingWindow:
  name: str
  start: date
  train_end: date
  validation_end: date
  dev_test_end: date
  end: date

  def to_dict(self) -> dict[str, str]:
    return {
      "name": self.name,
      "start": self.start.isoformat(),
      "train_end": self.train_end.isoformat(),
      "validation_end": self.validation_end.isoformat(),
      "dev_test_end": self.dev_test_end.isoformat(),
      "end": self.end.isoformat(),
    }


def parse_rolling_window(value: str) -> RollingWindow:
  separator = "," if "," in value else ":"
  parts = [
    part.strip()
    for part in value.split(separator)
  ]
  if len(parts) != 6 or any(not part for part in parts):
    raise ValueError(
      "Rolling windows must be "
      "name,start,train_end,validation_end,dev_test_end,end."
    )

  name = parts[0]
  if not _WINDOW_NAME_PATTERN.fullmatch(name):
    raise ValueError(
      "Rolling window names may only contain letters, numbers, '.', '_', and '-'."
    )

  try:
    return RollingWindow(
      name=name,
      start=date.fromisoformat(parts[1]),
      train_end=date.fromisoformat(parts[2]),
      validation_end=date.fromisoformat(parts[3]),
      dev_test_end=date.fromisoformat(parts[4]),
      end=date.fromisoformat(parts[5]),
    )
  except ValueError as error:
    raise ValueError(
      "Rolling window dates must be ISO dates: YYYY-MM-DD."
    ) from error


def run_rolling_evaluation(
  *,
  windows: tuple[RollingWindow, ...],
  output: Path,
  format_code: str = "modern",
  target_source: str = "source_archetype_name_proxy",
  target_label_level: str = "source",
  canonical_targets: Path | None = None,
  env_file: Path | None = None,
  limit_events: int | None = None,
  epochs: int = 40,
  learning_rate: float = DEFAULT_LEARNING_RATE,
  pooled_learning_rate: float | None = None,
  neural_learning_rate: float | None = None,
  weight_decay: float = 0.0,
  deepsets_regularized_weight_decay: float = DEFAULT_REGULARIZED_WEIGHT_DECAY,
  seed: int = 13,
  seeds: tuple[int, ...] | None = None,
  embedding_dim: int = 32,
  hidden_dim: int = 64,
  batch_size: int = 1024,
  shuffle: bool = True,
  max_steps: int | None = None,
  device: str = "auto",
  prediction_output: str = PREDICTION_OUTPUT_ERRORS,
  taxonomy_eval: Path | None = None,
) -> dict[str, Any]:
  if not windows:
    raise ValueError("At least one rolling window is required.")

  output.mkdir(parents=True, exist_ok=True)
  window_results: list[dict[str, Any]] = []
  for window in windows:
    window_output = output / "windows" / window.name
    export_summary = export_dataset(
      DatasetExportOptions(
        format_code=format_code,
        start=window.start,
        end=window.end,
        output=window_output,
        dataset_version=None,
        train_end=window.train_end,
        validation_end=window.validation_end,
        dev_test_end=window.dev_test_end,
        env_file=env_file,
        limit_events=limit_events,
      )
    )
    check_result = verify_dataset_export(export_summary.output)
    training_output = export_summary.output / "model_runs" / "training_results.json"
    training_result = run_model_training(
      export_summary.output,
      output=training_output,
      model_names=(MODEL_ALL,),
      pooling=POOLING_SUM,
      target_source=target_source,
      target_label_level=target_label_level,
      canonical_targets=canonical_targets,
      epochs=epochs,
      learning_rate=learning_rate,
      pooled_learning_rate=pooled_learning_rate,
      neural_learning_rate=neural_learning_rate,
      weight_decay=weight_decay,
      deepsets_regularized_weight_decay=deepsets_regularized_weight_decay,
      seed=seed,
      seeds=seeds,
      embedding_dim=embedding_dim,
      hidden_dim=hidden_dim,
      batch_size=batch_size,
      shuffle=shuffle,
      max_steps=max_steps,
      device=device,
      prediction_output=prediction_output,
      taxonomy_eval=taxonomy_eval,
    )
    window_results.append(
      _window_result(
        window=window,
        export_output=export_summary.output,
        training_output=training_output,
        dataset_version=export_summary.dataset_version,
        check_result={
          "checked_artifacts": check_result.checked_artifacts,
          "checked_parquet_artifacts": check_result.checked_parquet_artifacts,
        },
        training_result=training_result,
      )
    )

  result = {
    "run_id": "rolling_time_slice_evaluation",
    "format": format_code,
    "target_source": target_source,
    "training_target": training_result_target_config(window_results),
    "model_names": list(DEFAULT_MODEL_SET),
    "seeds": list(seeds or (seed,)),
    "max_steps": max_steps,
    "batch_size": batch_size,
    "prediction_output": prediction_output,
    "taxonomy_evaluation": training_result_taxonomy_config(window_results),
    "windows": window_results,
    "summary_by_model": summarize_rolling_window_results(window_results),
    "across_window_diagnostics": summarize_across_window_diagnostics(
      window_results
    ),
    "status": "completed",
  }
  write_json(output / "rolling_evaluation_results.json", result)
  return result


def summarize_rolling_window_results(
  window_results: list[dict[str, Any]],
) -> dict[str, Any]:
  model_names: list[str] = []
  for window in window_results:
    for model_name in window.get("models", {}):
      if model_name not in model_names:
        model_names.append(model_name)

  rows: dict[str, Any] = {}
  for model_name in model_names:
    metric_rows: dict[str, Any] = {}
    for metric_name in ROLLING_PRIMARY_METRICS:
      metric_rows[metric_name] = _stats([
        _window_metric_mean(window, model_name, metric_name)
        for window in window_results
      ])
    rows[model_name] = {
      "window_count": len(window_results),
      "metrics": metric_rows,
    }
  return rows


def summarize_across_window_diagnostics(
  window_results: list[dict[str, Any]],
) -> dict[str, Any]:
  return {
    "ranking_by_metric": {
      metric_name: _ranking_by_metric(window_results, metric_name)
      for metric_name in ROLLING_RANKING_METRICS
    },
    "vs_pooled_linear": _pooled_linear_window_comparison(window_results),
  }


def training_result_taxonomy_config(
  window_results: list[dict[str, Any]],
) -> dict[str, Any]:
  for window in window_results:
    config = window.get("taxonomy_evaluation")
    if isinstance(config, dict):
      return config
  return {"enabled": False}


def training_result_target_config(
  window_results: list[dict[str, Any]],
) -> dict[str, Any]:
  for window in window_results:
    config = window.get("training_target")
    if isinstance(config, dict):
      return config
  return {"label_level": "source"}


def _window_result(
  *,
  window: RollingWindow,
  export_output: Path,
  training_output: Path,
  dataset_version: str,
  check_result: dict[str, Any],
  training_result: dict[str, Any],
) -> dict[str, Any]:
  models: dict[str, Any] = {}
  for model_name, model_result in training_result.get("models", {}).items():
    summary = model_result.get("multi_seed_summary", {})
    models[model_name] = {
      "primary_evaluation_split": model_result.get("primary_evaluation_split"),
      "multi_seed_summary": {
        metric_name: summary.get(metric_name)
        for metric_name in ROLLING_PRIMARY_METRICS
        if metric_name in summary
      },
    }

  return {
    "window": window.to_dict(),
    "dataset_path": str(export_output),
    "training_result_path": str(training_output),
    "dataset_version": dataset_version,
    "check": check_result,
    "split_counts": training_result.get("split_counts", {}),
    "evaluation_splits": training_result.get("evaluation_splits", []),
    "primary_evaluation_split": training_result.get("primary_evaluation_split"),
    "status": training_result.get("status"),
    "training_target": training_result.get("training_target"),
    "taxonomy_evaluation": training_result.get("taxonomy_evaluation"),
    "models": models,
    "diagnostics": _window_diagnostics(training_result),
  }


def _window_diagnostics(training_result: dict[str, Any]) -> dict[str, Any]:
  primary_split = str(training_result.get("primary_evaluation_split"))
  model_results = training_result.get("models", {})
  pooled_model = model_results.get(POOLED_LINEAR_MODEL_NAME)
  return {
    "scope": {
      "summary_metrics": "multi_seed_mean",
      "label_and_confusion_details": "primary_seed",
    },
    "models": {
      model_name: _model_window_diagnostics(model_result, primary_split)
      for model_name, model_result in model_results.items()
    },
    "vs_pooled_linear": {
      model_name: _model_vs_pooled_linear_diagnostics(
        model_result,
        pooled_model,
        primary_split,
      )
      for model_name, model_result in model_results.items()
      if model_name != POOLED_LINEAR_MODEL_NAME and pooled_model is not None
    },
  }


def _model_window_diagnostics(
  model_result: dict[str, Any],
  primary_split: str,
) -> dict[str, Any]:
  metrics = model_result.get("metrics", {})
  split_metrics = metrics.get("splits", {}).get(primary_split, {})
  views = metrics.get("evaluation_views", {}).get(primary_split, {})
  support_buckets = metrics.get("support_buckets", {}).get(primary_split, {})
  label_coverage = split_metrics.get("label_coverage", {})
  total_count = _number(split_metrics.get("count"))
  open_set_count = _number(
    views.get("open_set_unseen_labels", {}).get("count")
  )
  closed_set_count = _number(
    views.get("closed_set_seen_labels", {}).get("count")
  )

  return {
    "primary_split": primary_split,
    "count": total_count,
    "closed_set_count": closed_set_count,
    "open_set_count": open_set_count,
    "train_unseen_label_share": _ratio(open_set_count, total_count),
    "label_coverage": _compact_label_coverage(label_coverage),
    "support_buckets": _compact_support_buckets(support_buckets, total_count),
    "top_labels_by_mass": split_metrics.get("per_label_accuracy_top_20", [])[:20],
    "largest_confusion_pairs": split_metrics.get("largest_confusion_pairs", [])[:10],
    "taxonomy_label_noise_report": (
      model_result.get("taxonomy_evaluation", {})
      .get("label_noise_report", {})
      .get("exact_errors_canonical_matches", [])[:10]
    ),
  }


def _model_vs_pooled_linear_diagnostics(
  model_result: dict[str, Any],
  pooled_model: dict[str, Any] | None,
  primary_split: str,
) -> dict[str, Any]:
  if pooled_model is None:
    return {}

  return {
    "primary_metric_deltas": _primary_metric_deltas(
      model_result,
      pooled_model,
    ),
    "primary_seed_metric_deltas": _primary_seed_metric_deltas(
      model_result,
      pooled_model,
      primary_split,
    ),
    "top_per_label_wins": _per_label_deltas(
      model_result,
      pooled_model,
      primary_split,
      direction="wins",
    ),
    "top_per_label_losses": _per_label_deltas(
      model_result,
      pooled_model,
      primary_split,
      direction="losses",
    ),
    "canonical_label_noise_report": (
      model_result.get("taxonomy_evaluation", {})
      .get("label_noise_report", {})
      .get("exact_errors_canonical_matches", [])[:10]
    ),
  }


def _primary_metric_deltas(
  model_result: dict[str, Any],
  pooled_model: dict[str, Any],
) -> dict[str, float | None]:
  return {
    metric_name: _subtract_optional(
      _summary_metric_mean(model_result, metric_name),
      _summary_metric_mean(pooled_model, metric_name),
    )
    for metric_name in ROLLING_PRIMARY_METRICS
  }


def _primary_seed_metric_deltas(
  model_result: dict[str, Any],
  pooled_model: dict[str, Any],
  primary_split: str,
) -> dict[str, Any]:
  model_split = (
    model_result.get("metrics", {})
    .get("splits", {})
    .get(primary_split, {})
  )
  pooled_split = (
    pooled_model.get("metrics", {})
    .get("splits", {})
    .get(primary_split, {})
  )
  model_views = (
    model_result.get("metrics", {})
    .get("evaluation_views", {})
    .get(primary_split, {})
  )
  pooled_views = (
    pooled_model.get("metrics", {})
    .get("evaluation_views", {})
    .get(primary_split, {})
  )
  return {
    "accuracy": _subtract_optional(
      _number(model_split.get("accuracy")),
      _number(pooled_split.get("accuracy")),
    ),
    "closed_set_accuracy": _subtract_optional(
      _number(model_views.get("closed_set_seen_labels", {}).get("accuracy")),
      _number(pooled_views.get("closed_set_seen_labels", {}).get("accuracy")),
    ),
    "open_set_accuracy": _subtract_optional(
      _number(model_views.get("open_set_unseen_labels", {}).get("accuracy")),
      _number(pooled_views.get("open_set_unseen_labels", {}).get("accuracy")),
    ),
    "macro_f1": _subtract_optional(
      _number(model_split.get("macro_f1")),
      _number(pooled_split.get("macro_f1")),
    ),
    "top_3_accuracy": _subtract_optional(
      _number(model_split.get("top_3_accuracy")),
      _number(pooled_split.get("top_3_accuracy")),
    ),
  }


def _per_label_deltas(
  model_result: dict[str, Any],
  pooled_model: dict[str, Any],
  primary_split: str,
  *,
  direction: str,
  limit: int = 20,
) -> list[dict[str, Any]]:
  model_rows = _per_label_accuracy_by_label(model_result, primary_split)
  pooled_rows = _per_label_accuracy_by_label(pooled_model, primary_split)
  rows: list[dict[str, Any]] = []
  for label_id in sorted(set(model_rows) | set(pooled_rows)):
    model = model_rows.get(label_id, {})
    pooled = pooled_rows.get(label_id, {})
    model_accuracy = _number(model.get("accuracy"))
    pooled_accuracy = _number(pooled.get("accuracy"))
    count = int(model.get("count") or pooled.get("count") or 0)
    delta = _subtract_optional(model_accuracy, pooled_accuracy)
    if delta is None or delta == 0:
      continue
    rows.append(
      {
        "label_id": label_id,
        "count": count,
        "model_accuracy": model_accuracy,
        "pooled_linear_accuracy": pooled_accuracy,
        "accuracy_delta": delta,
        "correct_delta": delta * count,
      }
    )

  if direction == "wins":
    rows = [row for row in rows if row["accuracy_delta"] > 0]
    return sorted(
      rows,
      key=lambda row: (row["correct_delta"], row["count"], row["label_id"]),
      reverse=True,
    )[:limit]
  if direction == "losses":
    rows = [row for row in rows if row["accuracy_delta"] < 0]
    return sorted(
      rows,
      key=lambda row: (row["correct_delta"], -row["count"], row["label_id"]),
    )[:limit]
  raise ValueError(f"Unsupported per-label delta direction: {direction}")


def _per_label_accuracy_by_label(
  model_result: dict[str, Any],
  primary_split: str,
) -> dict[str, dict[str, Any]]:
  rows = (
    model_result.get("metrics", {})
    .get("splits", {})
    .get(primary_split, {})
    .get("per_label_accuracy", [])
  )
  return {
    str(row["label_id"]): row
    for row in rows
    if "label_id" in row
  }


def _compact_label_coverage(
  label_coverage: dict[str, Any],
  *,
  sample_limit: int = 20,
) -> dict[str, Any]:
  actual_never_predicted = label_coverage.get("actual_never_predicted", [])
  predicted_never_actual = label_coverage.get("predicted_never_actual", [])
  return {
    "actual_label_count": label_coverage.get("actual_label_count"),
    "predicted_label_count": label_coverage.get("predicted_label_count"),
    "actual_never_predicted_count": len(actual_never_predicted),
    "predicted_never_actual_count": len(predicted_never_actual),
    "actual_never_predicted_sample": actual_never_predicted[:sample_limit],
    "predicted_never_actual_sample": predicted_never_actual[:sample_limit],
  }


def _compact_support_buckets(
  support_buckets: dict[str, Any],
  total_count: float | None,
) -> dict[str, Any]:
  rows: dict[str, Any] = {}
  for bucket_name, metrics in support_buckets.items():
    count = _number(metrics.get("count"))
    rows[bucket_name] = {
      "count": count,
      "share": _ratio(count, total_count),
      "accuracy": _number(metrics.get("accuracy")),
      "top_3_accuracy": _number(metrics.get("top_3_accuracy")),
      "macro_f1": _number(metrics.get("macro_f1")),
    }
  return rows


def _ranking_by_metric(
  window_results: list[dict[str, Any]],
  metric_name: str,
) -> dict[str, Any]:
  lower_is_better = metric_name in LOWER_IS_BETTER_METRICS
  window_rankings: list[dict[str, Any]] = []
  ranks_by_model: dict[str, list[float]] = {}
  for window in window_results:
    values = [
      (
        model_name,
        _window_metric_mean(window, model_name, metric_name),
      )
      for model_name in window.get("models", {})
    ]
    values = [
      (model_name, value)
      for model_name, value in values
      if value is not None
    ]
    values.sort(
      key=lambda row: row[1],
      reverse=not lower_is_better,
    )
    ranking: list[dict[str, Any]] = []
    for index, (model_name, value) in enumerate(values, start=1):
      ranks_by_model.setdefault(model_name, []).append(float(index))
      ranking.append({
        "rank": index,
        "model": model_name,
        "value": value,
      })
    window_rankings.append({
      "window": window.get("window", {}).get("name"),
      "ranking": ranking,
    })

  return {
    "lower_is_better": lower_is_better,
    "windows": window_rankings,
    "mean_rank_by_model": {
      model_name: _stats(ranks)
      for model_name, ranks in ranks_by_model.items()
    },
  }


def _pooled_linear_window_comparison(
  window_results: list[dict[str, Any]],
) -> dict[str, Any]:
  model_names: list[str] = []
  for window in window_results:
    for model_name in window.get("models", {}):
      if (
        model_name != POOLED_LINEAR_MODEL_NAME
        and model_name not in model_names
      ):
        model_names.append(model_name)

  rows: dict[str, Any] = {}
  for model_name in model_names:
    metric_rows: dict[str, Any] = {}
    for metric_name in ROLLING_RANKING_METRICS:
      lower_is_better = metric_name in LOWER_IS_BETTER_METRICS
      deltas = [
        _subtract_optional(
          _window_metric_mean(window, model_name, metric_name),
          _window_metric_mean(window, POOLED_LINEAR_MODEL_NAME, metric_name),
        )
        for window in window_results
      ]
      wins = sum(
        1
        for delta in deltas
        if delta is not None and (
          delta < 0
          if lower_is_better
          else delta > 0
        )
      )
      losses = sum(
        1
        for delta in deltas
        if delta is not None and (
          delta > 0
          if lower_is_better
          else delta < 0
        )
      )
      ties = sum(1 for delta in deltas if delta == 0)
      metric_rows[metric_name] = {
        "lower_is_better": lower_is_better,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "delta": _stats(deltas),
      }
    rows[model_name] = metric_rows
  return rows


def _window_metric_mean(
  window: dict[str, Any],
  model_name: str,
  metric_name: str,
) -> float | None:
  metric = (
    window.get("models", {})
    .get(model_name, {})
    .get("multi_seed_summary", {})
    .get(metric_name)
  )
  if not isinstance(metric, dict):
    return None
  mean = metric.get("mean")
  if mean is None:
    return None
  return float(mean)


def _summary_metric_mean(
  model_result: dict[str, Any],
  metric_name: str,
) -> float | None:
  metric = model_result.get("multi_seed_summary", {}).get(metric_name)
  if not isinstance(metric, dict):
    return None
  return _number(metric.get("mean"))


def _number(value: Any) -> float | None:
  if value is None:
    return None
  try:
    number = float(value)
  except (TypeError, ValueError):
    return None
  if not math.isfinite(number):
    return None
  return number


def _subtract_optional(
  left: float | None,
  right: float | None,
) -> float | None:
  if left is None or right is None:
    return None
  return left - right


def _ratio(
  numerator: float | None,
  denominator: float | None,
) -> float | None:
  if numerator is None or denominator is None or denominator == 0:
    return None
  return numerator / denominator


def _stats(values: list[float | None]) -> dict[str, Any]:
  numeric_values = [
    value
    for value in values
    if value is not None
  ]
  if not numeric_values:
    return {
      "count": 0,
      "mean": None,
      "std": None,
      "values": values,
    }

  mean = sum(numeric_values) / len(numeric_values)
  variance = (
    sum((value - mean) ** 2 for value in numeric_values) / len(numeric_values)
  )
  return {
    "count": len(numeric_values),
    "mean": mean,
    "std": math.sqrt(variance),
    "values": values,
  }

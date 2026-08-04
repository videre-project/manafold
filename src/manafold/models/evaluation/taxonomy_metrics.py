from __future__ import annotations

from typing import Any

from manafold.models.evaluation.prediction_metrics import classification_metrics
from manafold.taxonomy import (
  TaxonomyEvaluationConfig,
  _canonical_train_label_support,
  canonicalize_predictions,
  display_label_from_id,
)


def taxonomy_metrics(
  predictions: list[dict[str, Any]],
  config: TaxonomyEvaluationConfig,
) -> dict[str, Any]:
  if not config.enabled:
    return {"enabled": False}

  canonical_predictions = canonicalize_predictions(predictions, config)
  train_label_support = _canonical_train_label_support(canonical_predictions)
  metrics = classification_metrics(
    canonical_predictions,
    train_label_support=train_label_support,
  )
  return {
    "enabled": True,
    "label_level": "canonical_family",
    "config": config.to_dict(),
    "train_label_support": train_label_support,
    "metrics": metrics,
    "label_noise_report": _label_noise_report(
      source_predictions=predictions,
      canonical_predictions=canonical_predictions,
      config=config,
    ),
  }


def _label_noise_report(
  *,
  source_predictions: list[dict[str, Any]],
  canonical_predictions: list[dict[str, Any]],
  config: TaxonomyEvaluationConfig,
  limit: int = 20,
) -> dict[str, Any]:
  groups: dict[tuple[str, str, str], dict[str, Any]] = {}
  for source, canonical in zip(
    source_predictions,
    canonical_predictions,
    strict=True,
  ):
    if source.get("is_correct") or not canonical.get("is_correct"):
      continue
    actual = source.get("source_actual_label_id") or source.get(
      "actual_label_id"
    )
    predicted = source.get("predicted_label_id")
    if actual is None or predicted is None or actual == predicted:
      continue
    split_name = str(source.get("split_name"))
    key = (split_name, str(actual), str(predicted))
    group = groups.setdefault(
      key,
      {
        "split_name": split_name,
        "actual_source_label_id": str(actual),
        "actual_source_label": display_label_from_id(str(actual)),
        "predicted_source_label_id": str(predicted),
        "predicted_source_label": display_label_from_id(str(predicted)),
        "canonical_actual_label_id": canonical.get("actual_label_id"),
        "canonical_actual": display_label_from_id(
          str(canonical.get("actual_label_id"))
        ),
        "canonical_predicted_label_id": canonical.get("predicted_label_id"),
        "canonical_predicted": display_label_from_id(
          str(canonical.get("predicted_label_id"))
        ),
        "is_canonical_match": True,
        "count": 0,
        "example_deck_ids": [],
        "event_dates": [],
        "evidence": _matching_evidence(
          label_ids=(str(actual), str(predicted)),
          prediction=source,
          config=config,
        ),
      },
    )
    group["count"] += 1
    if len(group["example_deck_ids"]) < 10:
      group["example_deck_ids"].append(source.get("deck_id"))
    if source.get("event_date") is not None:
      group["event_dates"].append(str(source["event_date"]))

  rows: list[dict[str, Any]] = []
  for group in groups.values():
    event_dates = sorted(group.pop("event_dates"))
    group["date_window"] = {
      "min": event_dates[0] if event_dates else None,
      "max": event_dates[-1] if event_dates else None,
    }
    rows.append(group)

  rows.sort(key=lambda row: (-int(row["count"]), row["actual_source_label_id"]))
  return {
    "exact_errors_canonical_matches": rows[:limit],
    "total_exact_errors_canonical_matches": sum(
      int(row["count"]) for row in rows
    ),
  }


def _matching_evidence(
  *,
  label_ids: tuple[str, str],
  prediction: dict[str, Any],
  config: TaxonomyEvaluationConfig,
) -> dict[str, Any] | None:
  for rule in config.aliases:
    if (
      rule.source_label_id in label_ids
      and rule.applies_to_prediction(prediction)
    ):
      return rule.evidence
  return None

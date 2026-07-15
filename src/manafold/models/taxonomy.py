from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from manafold.models.metrics import classification_metrics


@dataclass(frozen=True)
class TaxonomyAliasRule:
  source_label: str
  canonical_family: str
  target_source: str
  format_code: str | None
  valid_from: date | None
  valid_to: date | None
  confidence: str | None
  evidence: dict[str, Any]

  @property
  def source_label_id(self) -> str:
    return label_id_for(self.source_label, self.target_source)

  @property
  def canonical_label_id(self) -> str:
    return label_id_for(self.canonical_family, self.target_source)

  def applies_to_prediction(self, prediction: dict[str, Any]) -> bool:
    if self.format_code is not None:
      prediction_format = str(prediction.get("format") or "").casefold()
      if prediction_format != self.format_code:
        return False

    event_date = _date_or_none(prediction.get("event_date"))
    if self.valid_from is not None and (
      event_date is None or event_date < self.valid_from
    ):
      return False
    if self.valid_to is not None and (
      event_date is None or event_date > self.valid_to
    ):
      return False
    return True

  def to_dict(self) -> dict[str, Any]:
    return {
      "source_label": self.source_label,
      "source_label_id": self.source_label_id,
      "canonical_family": self.canonical_family,
      "canonical_label_id": self.canonical_label_id,
      "target_source": self.target_source,
      "format": self.format_code,
      "valid_from": (
        self.valid_from.isoformat()
        if self.valid_from is not None
        else None
      ),
      "valid_to": (
        self.valid_to.isoformat()
        if self.valid_to is not None
        else None
      ),
      "confidence": self.confidence,
      "evidence": self.evidence,
    }


@dataclass(frozen=True)
class TaxonomyEvaluationConfig:
  path: Path | None
  aliases: tuple[TaxonomyAliasRule, ...]

  @property
  def enabled(self) -> bool:
    return bool(self.aliases)

  def to_dict(self) -> dict[str, Any]:
    return {
      "enabled": self.enabled,
      "path": str(self.path) if self.path is not None else None,
      "alias_count": len(self.aliases),
      "aliases": [
        rule.to_dict()
        for rule in self.aliases
      ],
    }


def load_taxonomy_evaluation_config(
  path: Path | None,
  *,
  target_source: str,
) -> TaxonomyEvaluationConfig:
  if path is None:
    return TaxonomyEvaluationConfig(path=None, aliases=())
  if not path.exists():
    return TaxonomyEvaluationConfig(path=path, aliases=())

  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError as error:
    raise ValueError(
      f"{path} must be JSON-compatible YAML. "
      "Use JSON object syntax or add a YAML parser dependency deliberately."
    ) from error
  if not isinstance(data, dict):
    raise ValueError(f"{path} must contain a JSON object.")

  aliases = data.get("aliases", [])
  if not isinstance(aliases, list):
    raise ValueError("taxonomy aliases must be a list.")

  return TaxonomyEvaluationConfig(
    path=path,
    aliases=tuple(
      _parse_alias_rule(row, target_source=target_source)
      for row in aliases
    ),
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


def canonicalize_predictions(
  predictions: list[dict[str, Any]],
  config: TaxonomyEvaluationConfig,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for prediction in predictions:
    row = dict(prediction)
    actual = str(row.get("actual_label_id"))
    source_actual = str(row.get("source_actual_label_id") or actual)
    predicted = row.get("predicted_label_id")
    top_label_ids = [
      str(label_id)
      for label_id in row.get("top_label_ids", [])
      if label_id is not None
    ]
    canonical_actual = canonical_label_id(actual, row, config)
    canonical_predicted = (
      canonical_label_id(str(predicted), row, config)
      if predicted is not None
      else None
    )
    canonical_top_label_ids = _unique([
      canonical_label_id(label_id, row, config)
      for label_id in top_label_ids
    ])

    row["source_actual_label_id"] = source_actual
    row["source_predicted_label_id"] = predicted
    row["source_top_label_ids"] = top_label_ids
    row["actual_label_id"] = canonical_actual
    row["predicted_label_id"] = canonical_predicted
    row["top_label_ids"] = canonical_top_label_ids
    row["is_correct"] = canonical_predicted == canonical_actual
    row["is_top_3_correct"] = canonical_actual in canonical_top_label_ids
    rows.append(row)

  train_support = _canonical_train_label_support(rows)
  for row in rows:
    support = train_support.get(str(row["actual_label_id"]), 0)
    row["actual_label_train_support"] = support
    row["is_train_unseen_label"] = support == 0
  return rows


def canonical_label_id(
  label_id: str,
  prediction: dict[str, Any],
  config: TaxonomyEvaluationConfig,
) -> str:
  for rule in config.aliases:
    if label_id == rule.source_label_id and rule.applies_to_prediction(prediction):
      return rule.canonical_label_id
  return label_id


def label_id_for(label: str, target_source: str) -> str:
  if label.startswith("proxy."):
    return label
  slug = re.sub(r"[^a-z0-9]+", "_", _normalize_label(label)).strip("_")
  if not slug:
    slug = "unknown"
  return f"proxy.{target_source}.{slug}"


def _parse_alias_rule(
  row: Any,
  *,
  target_source: str,
) -> TaxonomyAliasRule:
  if not isinstance(row, dict):
    raise ValueError("taxonomy alias rows must be objects.")
  source_label = _required_str(row, "source_label")
  canonical_family = _required_str(row, "canonical_family")
  rule_target_source = str(row.get("target_source") or target_source)
  format_code = row.get("format")
  evidence = row.get("evidence", {})
  if evidence is None:
    evidence = {}
  if not isinstance(evidence, dict):
    raise ValueError("taxonomy alias evidence must be an object.")
  return TaxonomyAliasRule(
    source_label=source_label,
    canonical_family=canonical_family,
    target_source=rule_target_source,
    format_code=(
      str(format_code).casefold()
      if format_code is not None
      else None
    ),
    valid_from=_date_or_none(row.get("valid_from")),
    valid_to=_date_or_none(row.get("valid_to")),
    confidence=(
      str(row["confidence"])
      if row.get("confidence") is not None
      else None
    ),
    evidence=evidence,
  )


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
    actual = source.get("source_actual_label_id") or source.get("actual_label_id")
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
        "canonical_actual": display_label_from_id(str(canonical.get("actual_label_id"))),
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
      int(row["count"])
      for row in rows
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


def _canonical_train_label_support(
  predictions: list[dict[str, Any]],
) -> dict[str, int]:
  counts: Counter[str] = Counter()
  for prediction in predictions:
    if prediction.get("split_name") == "train":
      counts[str(prediction["actual_label_id"])] += 1
  return dict(sorted(counts.items()))


def display_label_from_id(label_id: str) -> str:
  slug = label_id.rsplit(".", 1)[-1]
  return " ".join(
    word.capitalize()
    for word in slug.split("_")
    if word
  )


def _required_str(row: dict[str, Any], field_name: str) -> str:
  value = row.get(field_name)
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f"taxonomy alias {field_name} must be a non-empty string.")
  return value


def _normalize_label(value: str) -> str:
  return " ".join(value.casefold().split())


def _date_or_none(value: Any) -> date | None:
  if value is None:
    return None
  if isinstance(value, date):
    return value
  return date.fromisoformat(str(value))


def _unique(values: list[str]) -> list[str]:
  return list(dict.fromkeys(values))

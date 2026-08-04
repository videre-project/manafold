from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from manafold.datasets.mtgo.build import SOURCE_ARCHETYPE_PROXY
from manafold.serialization import read_jsonl, write_json

SPLIT_ORDER = ("train", "validation", "dev-test", "test", "final-test")


@dataclass(frozen=True)
class _DatasetRow:
  window_name: str
  deck_id: str
  split_name: str
  event_date: date
  format_code: str
  label_id: str


def run_weak_target_preview(
  rolling_result: Path,
  *,
  weak_relation_graph: Path,
  soft_target_suggestions: Path,
  output: Path | None = None,
  target_source: str = SOURCE_ARCHETYPE_PROXY,
) -> dict[str, Any]:
  rolling_result_path = _rolling_result_path(rolling_result)
  rolling_data = json.loads(rolling_result_path.read_text(encoding="utf-8"))
  weak_relation_graph = weak_relation_graph.resolve()
  soft_target_suggestions = soft_target_suggestions.resolve()
  graph = json.loads(weak_relation_graph.read_text(encoding="utf-8"))
  suggestions = read_jsonl(soft_target_suggestions)
  output = output or rolling_result_path.parent / "weak_target_preview.json"

  rows = _load_rolling_dataset_rows(
    rolling_data,
    rolling_result_base=rolling_result_path.parent,
    target_source=target_source,
  )
  edge_index = _edge_index(graph.get("edges", []))
  preview_rows: list[dict[str, Any]] = []
  for suggestion in suggestions:
    source_label_id = str(suggestion["source_label_id"])
    source_label = str(suggestion["source_label"])
    for target in suggestion.get("candidate_targets", []):
      relation = str(target["relation"])
      target_label_id = str(target["label_id"])
      format_code = target.get("format")
      valid_from = _date_or_none(target.get("valid_from"))
      valid_to = _date_or_none(target.get("valid_to"))
      evidence = edge_index.get(
        (
          format_code,
          target_label_id,
          source_label_id,
          relation,
        ),
        {},
      )
      counts = _affected_counts(
        rows,
        source_label_id=source_label_id,
        format_code=format_code,
        valid_from=valid_from,
        valid_to=valid_to,
      )
      preview_rows.append({
        "source_label": source_label,
        "source_label_id": source_label_id,
        "candidate_canonical_labels": [{
          "label": target["label"],
          "label_id": target_label_id,
          "relation": relation,
          "confidence": target["confidence"],
          "soft_weight": target.get("soft_weight"),
        }],
        "format": format_code,
        "valid_from": target.get("valid_from"),
        "valid_to": target.get("valid_to"),
        "evidence_splits": _evidence_splits(evidence),
        "has_heldout_evidence": bool(
          int(evidence.get("held_out_observation_count") or 0)
        ),
        "all_evidence_heldout": (
          bool(evidence)
          and int(evidence.get("usable_observation_count") or 0) == 0
        ),
        "affected_rows_by_split": counts["by_split"],
        "affected_rows_by_window": counts["by_window"],
        "total_rows_affected": counts["total"],
        "train_rows_affected": counts["by_split"].get("train", 0),
        "validation_rows_affected": counts["by_split"].get("validation", 0),
        "dev_test_rows_affected": counts["by_split"].get("dev-test", 0),
        "final_test_rows_affected": counts["by_split"].get("final-test", 0),
      })

  aggregate_counts = _aggregate_counts(preview_rows)
  result = {
    "run_id": "weak_target_preview_v0",
    "rolling_result_path": str(rolling_result_path),
    "weak_relation_graph_path": str(weak_relation_graph),
    "soft_target_suggestions_path": str(soft_target_suggestions),
    "target_source": target_source,
    "suggestion_count": len(suggestions),
    "preview_row_count": len(preview_rows),
    "rows_with_train_coverage_count": sum(
      1 for row in preview_rows if int(row["train_rows_affected"]) > 0
    ),
    "rows_with_any_coverage_count": sum(
      1 for row in preview_rows if int(row["total_rows_affected"]) > 0
    ),
    "affected_rows_by_split": aggregate_counts,
    "preview_rows": sorted(
      preview_rows,
      key=lambda row: (
        -int(row["train_rows_affected"]),
        -int(row["total_rows_affected"]),
        row["source_label"],
        row["candidate_canonical_labels"][0]["label"],
      ),
    ),
  }
  write_json(output, result)
  return result


def _rolling_result_path(path: Path) -> Path:
  if path.is_dir():
    return path / "rolling_evaluation_results.json"
  return path


def _load_rolling_dataset_rows(
  rolling_data: dict[str, Any],
  *,
  rolling_result_base: Path,
  target_source: str,
) -> list[_DatasetRow]:
  rows: list[_DatasetRow] = []
  for window in rolling_data.get("windows", []):
    window_name = str(window.get("window", {}).get("name") or "unknown")
    dataset_path = _resolve_path(window["dataset_path"], rolling_result_base)
    rows.extend(
      _load_dataset_rows(
        dataset_path,
        window_name=window_name,
        target_source=target_source,
      )
    )
  return rows


def _load_dataset_rows(
  dataset_path: Path,
  *,
  window_name: str,
  target_source: str,
) -> list[_DatasetRow]:
  manifest = json.loads((dataset_path / "dataset_manifest.json").read_text())
  artifacts = manifest["artifacts"]
  splits = {
    str(row["deck_id"]): row
    for row in pq.read_table(
      dataset_path / artifacts["split_manifest"]
    ).to_pylist()
  }
  labels = {
    str(row["deck_id"]): str(row["proxy_label_id"])
    for row in pq.read_table(
      dataset_path / artifacts["proxy_targets"]
    ).to_pylist()
    if str(row["target_source"]) == target_source
  }

  rows: list[_DatasetRow] = []
  for deck_id, label_id in labels.items():
    split = splits.get(deck_id)
    if split is None:
      continue
    event_date = split["event_date"]
    if not isinstance(event_date, date):
      event_date = date.fromisoformat(str(event_date))
    rows.append(
      _DatasetRow(
        window_name=window_name,
        deck_id=deck_id,
        split_name=str(split["split_name"]),
        event_date=event_date,
        format_code=str(split["format"]).casefold(),
        label_id=label_id,
      )
    )
  return rows


def _resolve_path(path_value: Any, base: Path) -> Path:
  path = Path(str(path_value))
  return path if path.is_absolute() else base / path


def _edge_index(
  edges: list[dict[str, Any]],
) -> dict[tuple[Any, str, str, str], dict[str, Any]]:
  return {
    (
      edge.get("format"),
      str(edge["label_a_id"]),
      str(edge["label_b_id"]),
      str(edge["relation"]),
    ): edge
    for edge in edges
  }


def _affected_counts(
  rows: list[_DatasetRow],
  *,
  source_label_id: str,
  format_code: str | None,
  valid_from: date | None,
  valid_to: date | None,
) -> dict[str, Any]:
  by_split: Counter[str] = Counter({
    split_name: 0 for split_name in SPLIT_ORDER
  })
  by_window: dict[str, Counter[str]] = defaultdict(Counter)
  total = 0
  normalized_format = str(format_code).casefold() if format_code else None
  for row in rows:
    if row.label_id != source_label_id:
      continue
    if normalized_format is not None and row.format_code != normalized_format:
      continue
    if valid_from is not None and row.event_date < valid_from:
      continue
    if valid_to is not None and row.event_date > valid_to:
      continue
    by_split[row.split_name] += 1
    by_window[row.window_name][row.split_name] += 1
    total += 1

  return {
    "by_split": _ordered_split_counts(by_split),
    "by_window": [
      {
        "window": window_name,
        "splits": _ordered_split_counts(counter),
        "total": sum(counter.values()),
      }
      for window_name, counter in sorted(by_window.items())
    ],
    "total": total,
  }


def _ordered_split_counts(counter: Counter[str]) -> dict[str, int]:
  keys = list(SPLIT_ORDER) + sorted(set(counter) - set(SPLIT_ORDER))
  return {key: int(counter.get(key, 0)) for key in keys}


def _evidence_splits(edge: dict[str, Any]) -> list[dict[str, Any]]:
  return [
    {
      "rolling_window": scope.get("rolling_window"),
      "split_name": scope.get("split_name"),
      "valid_from": scope.get("valid_from"),
      "valid_to": scope.get("valid_to"),
      "confidence": scope.get("confidence"),
      "excluded_because_heldout": (
        scope.get("training_exclusion_reason") == "held_out_evaluation_split"
      ),
      "training_exclusion_reason": scope.get("training_exclusion_reason"),
    }
    for scope in edge.get("time_scopes", [])
  ]


def _aggregate_counts(preview_rows: list[dict[str, Any]]) -> dict[str, int]:
  counts: Counter[str] = Counter({split_name: 0 for split_name in SPLIT_ORDER})
  for row in preview_rows:
    for split_name, count in row["affected_rows_by_split"].items():
      counts[split_name] += int(count)
  return _ordered_split_counts(counts)


def _date_or_none(value: Any) -> date | None:
  if value is None:
    return None
  if isinstance(value, date):
    return value
  value = str(value)
  if not value or value == "unknown":
    return None
  return date.fromisoformat(value)

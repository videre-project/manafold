from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from manafold.data.validate import write_json
from manafold.models.weak_relations import (
  DEFAULT_EXCLUDED_TRAINING_SPLITS,
  SOFT_CANONICAL_RELATIONS,
)

HIGH_CONFIDENCE_THRESHOLD = 0.75
REVIEW_CONFIDENCE_THRESHOLD = 0.55


def run_weak_relation_state(
  *,
  alias_observations: Path,
  weak_relation_graph: Path,
  soft_target_suggestions: Path,
  weak_target_preview: Path,
  output: Path | None = None,
  high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
  review_confidence_threshold: float = REVIEW_CONFIDENCE_THRESHOLD,
  excluded_training_splits: tuple[str, ...] = DEFAULT_EXCLUDED_TRAINING_SPLITS,
) -> dict[str, Any]:
  alias_observations = alias_observations.resolve()
  weak_relation_graph = weak_relation_graph.resolve()
  soft_target_suggestions = soft_target_suggestions.resolve()
  weak_target_preview = weak_target_preview.resolve()
  output = output or weak_relation_graph.with_name("weak_relation_state.json")

  observations = _read_jsonl(alias_observations)
  graph = json.loads(weak_relation_graph.read_text(encoding="utf-8"))
  suggestions = _read_jsonl(soft_target_suggestions)
  preview = json.loads(weak_target_preview.read_text(encoding="utf-8"))
  preview_index = _preview_index(preview.get("preview_rows", []))

  usable_now = _usable_now(preview.get("preview_rows", []))
  review_queue = _review_queue(
    graph.get("edges", []),
    preview_index=preview_index,
    high_confidence_threshold=high_confidence_threshold,
    review_confidence_threshold=review_confidence_threshold,
  )
  deferred = _deferred_to_next_window(
    graph.get("edges", []),
    excluded_training_splits=excluded_training_splits,
  )
  deferred_soft_canonical_evidence = [
    row
    for row in deferred
    if row["deferred_training_eligible_if_future"]
  ]
  deferred_review_diagnostics = [
    row
    for row in deferred
    if not row["deferred_training_eligible_if_future"]
  ]
  blocked = _blocked(
    graph.get("edges", []),
    preview_index=preview_index,
    high_confidence_threshold=high_confidence_threshold,
    review_confidence_threshold=review_confidence_threshold,
  )
  bucket_rows = {
    "usable_now": usable_now,
    "review_queue": review_queue,
    "deferred_soft_canonical_evidence": deferred_soft_canonical_evidence,
    "deferred_review_diagnostics": deferred_review_diagnostics,
    "blocked": blocked,
  }

  result = {
    "run_id": "weak_relation_state_v0",
    "input_paths": {
      "alias_observations": str(alias_observations),
      "weak_relation_graph": str(weak_relation_graph),
      "soft_target_suggestions": str(soft_target_suggestions),
      "weak_target_preview": str(weak_target_preview),
    },
    "policy": {
      "excluded_training_splits": list(excluded_training_splits),
      "soft_canonical_relations": list(SOFT_CANONICAL_RELATIONS),
      "high_confidence_threshold": high_confidence_threshold,
      "review_confidence_threshold": review_confidence_threshold,
      "note": (
        "usable_now requires train-row coverage. Held-out evidence is "
        "deferred or reviewed, not converted into current training targets."
      ),
    },
    "input_counts": {
      "alias_observations": len(observations),
      "graph_edges": len(graph.get("edges", [])),
      "soft_target_suggestions": len(suggestions),
      "preview_rows": len(preview.get("preview_rows", [])),
    },
    "counts": {
      "usable_now": len(usable_now),
      "review_queue": len(review_queue),
      "deferred_soft_canonical_evidence": len(deferred_soft_canonical_evidence),
      "deferred_review_diagnostics": len(deferred_review_diagnostics),
      "blocked": len(blocked),
    },
    "bucket_summaries": _bucket_summaries(bucket_rows),
    "top_review_pairs": _top_review_pairs(review_queue),
    "usable_now": usable_now,
    "review_queue": review_queue,
    "deferred_soft_canonical_evidence": deferred_soft_canonical_evidence,
    "deferred_review_diagnostics": deferred_review_diagnostics,
    "blocked": blocked,
  }
  write_json(output, result)
  return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as input_file:
    for line in input_file:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def _preview_index(
  preview_rows: list[dict[str, Any]],
) -> dict[tuple[Any, str, str, str], dict[str, Any]]:
  rows: dict[tuple[Any, str, str, str], dict[str, Any]] = {}
  for row in preview_rows:
    source_label_id = str(row["source_label_id"])
    for target in row.get("candidate_canonical_labels", []):
      rows[
        (
          row.get("format"),
          str(target["label_id"]),
          source_label_id,
          str(target["relation"]),
        )
      ] = row
  return rows


def _usable_now(preview_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for row in preview_rows:
    train_rows = int(row.get("train_rows_affected") or 0)
    if train_rows <= 0:
      continue
    target = row["candidate_canonical_labels"][0]
    rows.append({
      "source_label": row["source_label"],
      "source_label_id": row["source_label_id"],
      "candidate_label": target["label"],
      "candidate_label_id": target["label_id"],
      "relation": target["relation"],
      "confidence": target["confidence"],
      "format": row.get("format"),
      "valid_from": row.get("valid_from"),
      "valid_to": row.get("valid_to"),
      "train_rows_affected": train_rows,
      "validation_rows_affected": row.get("validation_rows_affected", 0),
      "affected_rows_by_split": row.get("affected_rows_by_split", {}),
      "action": "eligible_for_soft_target_training",
    })
  return sorted(
    rows,
    key=lambda row: (
      -int(row["train_rows_affected"]),
      -float(row["confidence"]),
      row["source_label"],
    ),
  )


def _review_queue(
  edges: list[dict[str, Any]],
  *,
  preview_index: dict[tuple[Any, str, str, str], dict[str, Any]],
  high_confidence_threshold: float,
  review_confidence_threshold: float,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for edge in edges:
    confidence = _edge_confidence(edge)
    key = _edge_key(edge)
    preview = preview_index.get(key)
    has_heldout = int(edge.get("held_out_observation_count") or 0) > 0
    has_usable = int(edge.get("usable_observation_count") or 0) > 0
    train_rows = int((preview or {}).get("train_rows_affected") or 0)
    has_mixed_evidence = has_heldout and has_usable
    is_review_worthy = (
      confidence >= high_confidence_threshold
      or (
        has_mixed_evidence
        and confidence >= review_confidence_threshold
      )
    )
    if not is_review_worthy:
      continue
    reasons: list[str] = []
    if confidence >= high_confidence_threshold:
      reasons.append("high_confidence")
    if has_mixed_evidence:
      reasons.append("mixed_usable_and_heldout_evidence")
    elif has_heldout:
      reasons.append("heldout_evidence")
    if preview is not None and train_rows == 0:
      reasons.append("no_current_train_coverage")
    if edge["relation"] not in SOFT_CANONICAL_RELATIONS:
      reasons.append("non_canonical_relation")
    if not reasons:
      continue
    rows.append({
      **_edge_summary(edge),
      "train_rows_affected": train_rows,
      "review_reasons": sorted(set(reasons)),
      "action": _review_action(edge, confidence=confidence, train_rows=train_rows),
    })
  return sorted(
    rows,
    key=lambda row: (
      -float(row["confidence"]),
      row["label_a"],
      row["label_b"],
      row["relation"],
    ),
  )


def _deferred_to_next_window(
  edges: list[dict[str, Any]],
  *,
  excluded_training_splits: tuple[str, ...],
) -> list[dict[str, Any]]:
  excluded = set(excluded_training_splits)
  rows: list[dict[str, Any]] = []
  for edge in edges:
    heldout_scopes = [
      scope
      for scope in edge.get("time_scopes", [])
      if scope.get("split_name") in excluded
    ]
    if not heldout_scopes:
      continue
    rows.append({
      **_edge_summary(edge),
      "heldout_scopes": heldout_scopes,
      "deferred_training_eligible_if_future": (
        edge["relation"] in SOFT_CANONICAL_RELATIONS
      ),
      "deferred_reason": (
        "heldout_soft_canonical_evidence"
        if edge["relation"] in SOFT_CANONICAL_RELATIONS
        else "heldout_review_diagnostic"
      ),
      "action": (
        "defer_as_future_soft_canonical_evidence"
        if edge["relation"] in SOFT_CANONICAL_RELATIONS
        else "defer_as_future_review_diagnostic"
      ),
    })
  return sorted(
    rows,
    key=lambda row: (
      -float(row["confidence"]),
      row["label_a"],
      row["label_b"],
      row["relation"],
    ),
  )


def _blocked(
  edges: list[dict[str, Any]],
  *,
  preview_index: dict[tuple[Any, str, str, str], dict[str, Any]],
  high_confidence_threshold: float,
  review_confidence_threshold: float,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for edge in edges:
    confidence = _edge_confidence(edge)
    preview = preview_index.get(_edge_key(edge))
    reasons: list[str] = []
    if confidence < review_confidence_threshold:
      reasons.append("low_confidence")
    if edge["relation"] == "uncertain_boundary":
      reasons.append("uncertain_boundary")
    if (
      edge["relation"] not in SOFT_CANONICAL_RELATIONS
      and confidence < high_confidence_threshold
    ):
      reasons.append("not_a_soft_canonical_relation")
    if (
      preview is not None
      and int(preview.get("train_rows_affected") or 0) == 0
      and confidence < high_confidence_threshold
    ):
      reasons.append("no_train_coverage_below_high_confidence")
    if not reasons:
      continue
    rows.append({
      **_edge_summary(edge),
      "blocked_reasons": sorted(set(reasons)),
      "action": "diagnostics_only",
    })
  return sorted(
    rows,
    key=lambda row: (
      -float(row["confidence"]),
      row["label_a"],
      row["label_b"],
      row["relation"],
    ),
  )


def _review_action(edge: dict[str, Any], *, confidence: float, train_rows: int) -> str:
  if train_rows > 0 and edge["relation"] in SOFT_CANONICAL_RELATIONS:
    return "review_before_soft_target_training"
  if int(edge.get("held_out_observation_count") or 0) > 0:
    return "review_or_defer_to_next_window"
  if confidence >= HIGH_CONFIDENCE_THRESHOLD:
    return "review_for_seed_evidence"
  return "review_if_high_impact"


def _edge_key(edge: dict[str, Any]) -> tuple[Any, str, str, str]:
  return (
    edge.get("format"),
    str(edge["label_a_id"]),
    str(edge["label_b_id"]),
    str(edge["relation"]),
  )


def _edge_confidence(edge: dict[str, Any]) -> float:
  confidence = edge.get("confidence", {})
  if isinstance(confidence, dict):
    return float(confidence.get("max") or 0.0)
  return float(confidence or 0.0)


def _edge_summary(edge: dict[str, Any]) -> dict[str, Any]:
  return {
    "label_a": edge["label_a"],
    "label_b": edge["label_b"],
    "label_a_id": edge["label_a_id"],
    "label_b_id": edge["label_b_id"],
    "relation": edge["relation"],
    "format": edge.get("format"),
    "confidence": _edge_confidence(edge),
    "valid_from": edge.get("valid_from"),
    "valid_to": edge.get("valid_to"),
    "rolling_windows": edge.get("rolling_windows", []),
    "splits": edge.get("splits", []),
    "usable_observation_count": edge.get("usable_observation_count", 0),
    "held_out_observation_count": edge.get("held_out_observation_count", 0),
    "observation_count": edge.get("observation_count", 0),
    "aggregate_confusion_count": edge.get("aggregate_confusion_count", 0),
  }


def _bucket_summaries(
  buckets: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
  return {
    bucket_name: {
      "count": len(rows),
      "by_relation": _count_values(
        row.get("relation")
        for row in rows
      ),
      "by_split": _count_values(
        split_name
        for row in rows
        for split_name in _row_split_names(row)
      ),
      "by_rolling_window": _count_values(
        window_name
        for row in rows
        for window_name in _row_window_names(row)
      ),
    }
    for bucket_name, rows in buckets.items()
  }


def _top_review_pairs(review_queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
  grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
  for row in review_queue:
    key = tuple(sorted((str(row["label_a_id"]), str(row["label_b_id"]))))
    grouped.setdefault(key, []).append(row)

  rows: list[dict[str, Any]] = []
  for group in grouped.values():
    best = max(
      group,
      key=lambda row: (
        float(row["confidence"]),
        int(row.get("aggregate_confusion_count") or 0),
      ),
    )
    rows.append({
      "pair": [best["label_a"], best["label_b"]],
      "pair_ids": [best["label_a_id"], best["label_b_id"]],
      "item_count": len(group),
      "max_confidence": max(float(row["confidence"]) for row in group),
      "aggregate_confusion_count": sum(
        int(row.get("aggregate_confusion_count") or 0)
        for row in group
      ),
      "relations": _count_values(row.get("relation") for row in group),
      "review_reasons": _count_values(
        reason
        for row in group
        for reason in row.get("review_reasons", [])
      ),
      "actions": _count_values(row.get("action") for row in group),
      "rolling_windows": sorted({
        window_name
        for row in group
        for window_name in _row_window_names(row)
      }),
      "splits": sorted({
        split_name
        for row in group
        for split_name in _row_split_names(row)
      }),
    })

  rows.sort(
    key=lambda row: (
      -float(row["max_confidence"]),
      -int(row["aggregate_confusion_count"]),
      row["pair"][0],
      row["pair"][1],
    )
  )
  return rows


def _row_split_names(row: dict[str, Any]) -> list[str]:
  if row.get("heldout_scopes"):
    return [
      str(scope["split_name"])
      for scope in row["heldout_scopes"]
      if scope.get("split_name") is not None
    ]
  affected = row.get("affected_rows_by_split")
  if isinstance(affected, dict):
    return [
      str(split_name)
      for split_name, count in affected.items()
      if int(count or 0) > 0
    ]
  return [
    str(split_name)
    for split_name in row.get("splits", [])
  ]


def _row_window_names(row: dict[str, Any]) -> list[str]:
  if row.get("heldout_scopes"):
    return [
      str(scope["rolling_window"])
      for scope in row["heldout_scopes"]
      if scope.get("rolling_window") is not None
    ]
  return [
    str(window_name)
    for window_name in row.get("rolling_windows", [])
  ]


def _count_values(values: Any) -> dict[str, int]:
  counts: dict[str, int] = {}
  for value in values:
    if value is None:
      continue
    key = str(value)
    counts[key] = counts.get(key, 0) + 1
  return dict(sorted(counts.items()))

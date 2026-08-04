from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from manafold.serialization import read_jsonl, write_json, write_jsonl

RELATION_TYPES = (
  "alias_candidate",
  "same_family_candidate",
  "sibling_variant_candidate",
  "model_confusion_candidate",
  "uncertain_boundary",
)
DEFAULT_EXCLUDED_TRAINING_SPLITS = ("dev-test", "test", "final-test")
SOFT_CANONICAL_RELATIONS = ("alias_candidate", "same_family_candidate")


def run_weak_relation_graph(
  observations_path: Path,
  *,
  output: Path | None = None,
  suggestions_output: Path | None = None,
  excluded_training_splits: tuple[str, ...] = DEFAULT_EXCLUDED_TRAINING_SPLITS,
) -> dict[str, Any]:
  observations_path = observations_path.resolve()
  observations = read_jsonl(observations_path)
  output = output or observations_path.with_name("weak_relation_graph.json")
  suggestions_output = (
    suggestions_output
    or output.with_name("soft_canonical_target_suggestions.jsonl")
  )

  nodes = _nodes(observations)
  edges = _edges(
    observations,
    excluded_training_splits=excluded_training_splits,
  )
  suggestions = _soft_canonical_target_suggestions(edges)

  result = {
    "run_id": "weak_relation_graph_v0",
    "observation_path": str(observations_path),
    "soft_canonical_target_suggestions_path": str(suggestions_output),
    "relation_types": list(RELATION_TYPES),
    "training_evidence_policy": {
      "excluded_splits": list(excluded_training_splits),
      "included_relations": list(SOFT_CANONICAL_RELATIONS),
      "note": (
        "Held-out evidence is retained in graph edges but excluded from "
        "soft target suggestions."
      ),
    },
    "observation_count": len(observations),
    "node_count": len(nodes),
    "nodes": nodes,
    "edge_count": len(edges),
    "edges": edges,
    "soft_canonical_target_suggestion_count": len(suggestions),
  }
  write_json(output, result)
  write_jsonl(suggestions_output, suggestions)
  return result


def _nodes(observations: list[dict[str, Any]]) -> list[dict[str, str]]:
  labels: dict[str, str] = {}
  for observation in observations:
    labels[str(observation["label_a_id"])] = str(observation["label_a"])
    labels[str(observation["label_b_id"])] = str(observation["label_b"])
  return [
    {
      "label_id": label_id,
      "label": labels[label_id],
    }
    for label_id in sorted(labels, key=lambda value: labels[value])
  ]


def _edges(
  observations: list[dict[str, Any]],
  *,
  excluded_training_splits: tuple[str, ...],
) -> list[dict[str, Any]]:
  grouped: dict[tuple[str | None, str, str, str], list[dict[str, Any]]] = (
    defaultdict(list)
  )
  for observation in observations:
    key = (
      observation.get("format"),
      str(observation["label_a_id"]),
      str(observation["label_b_id"]),
      str(observation["relation"]),
    )
    grouped[key].append(observation)

  rows: list[dict[str, Any]] = []
  for (format_code, label_a_id, label_b_id, relation), group in grouped.items():
    usable = [
      observation
      for observation in group
      if _usable_for_training_suggestion(
        observation,
        excluded_training_splits=excluded_training_splits,
      )
    ]
    confidences = [float(row["confidence"]) for row in group]
    seed_consensus_values = [
      float(row.get("seed_consensus") or 0.0) for row in group
    ]
    seed_entropy_values = [
      float(row.get("seed_entropy_normalized") or 0.0) for row in group
    ]
    evidence_sources = sorted({
      source for row in group for source in row.get("evidence_sources", [])
    })
    rows.append({
      "format": format_code,
      "label_a": str(group[0]["label_a"]),
      "label_b": str(group[0]["label_b"]),
      "label_a_id": label_a_id,
      "label_b_id": label_b_id,
      "relation": relation,
      "confidence": {
        "max": max(confidences),
        "mean": sum(confidences) / len(confidences),
      },
      "valid_from": _min_present(row.get("valid_from") for row in group),
      "valid_to": _max_present(row.get("valid_to") for row in group),
      "rolling_windows": sorted({
        str(row["rolling_window"])
        for row in group
        if row.get("rolling_window") is not None
      }),
      "splits": sorted({
        str(row["split_name"])
        for row in group
        if row.get("split_name") is not None
      }),
      "evidence_sources": evidence_sources,
      "observation_count": len(group),
      "usable_observation_count": len(usable),
      "held_out_observation_count": len(group) - len(usable),
      "aggregate_confusion_count": sum(
        int(row.get("confusion_count") or 0) for row in group
      ),
      "seed_consensus": {
        "max": max(seed_consensus_values) if seed_consensus_values else 0.0,
        "mean": (
          sum(seed_consensus_values) / len(seed_consensus_values)
          if seed_consensus_values
          else 0.0
        ),
      },
      "seed_entropy_normalized": {
        "max": max(seed_entropy_values) if seed_entropy_values else 0.0,
        "mean": (
          sum(seed_entropy_values) / len(seed_entropy_values)
          if seed_entropy_values
          else 0.0
        ),
      },
      "time_scopes": [
        {
          "valid_from": row.get("valid_from"),
          "valid_to": row.get("valid_to"),
          "rolling_window": row.get("rolling_window"),
          "split_name": row.get("split_name"),
          "confidence": row.get("confidence"),
          "recommendation": row.get("recommendation"),
          "usable_for_training_suggestion": _usable_for_training_suggestion(
            row,
            excluded_training_splits=excluded_training_splits,
          ),
          "training_exclusion_reason": _training_exclusion_reason(
            row,
            excluded_training_splits=excluded_training_splits,
          ),
        }
        for row in sorted(
          group,
          key=lambda row: (
            str(row.get("valid_from")),
            str(row.get("rolling_window")),
            str(row.get("split_name")),
          ),
        )
      ],
    })

  rows.sort(
    key=lambda row: (
      -float(row["confidence"]["max"]),
      -int(row["usable_observation_count"]),
      row["label_a"],
      row["label_b"],
      row["relation"],
    )
  )
  return rows


def _usable_for_training_suggestion(
  observation: dict[str, Any],
  *,
  excluded_training_splits: tuple[str, ...],
) -> bool:
  if str(observation.get("relation")) not in SOFT_CANONICAL_RELATIONS:
    return False
  if not bool(observation.get("usable_for_training_suggestion", True)):
    return False
  if str(observation.get("split_name")) in set(excluded_training_splits):
    return False
  return True


def _training_exclusion_reason(
  observation: dict[str, Any],
  *,
  excluded_training_splits: tuple[str, ...],
) -> str | None:
  if str(observation.get("relation")) not in SOFT_CANONICAL_RELATIONS:
    return "non_canonical_relation"
  if str(observation.get("split_name")) in set(excluded_training_splits):
    return "held_out_evaluation_split"
  if not bool(observation.get("usable_for_training_suggestion", True)):
    return str(observation.get("training_exclusion_reason") or "not_usable")
  return None


def _soft_canonical_target_suggestions(
  edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for edge in edges:
    if edge["relation"] not in SOFT_CANONICAL_RELATIONS:
      continue
    usable_scopes = [
      scope
      for scope in edge["time_scopes"]
      if scope["usable_for_training_suggestion"]
    ]
    if not usable_scopes:
      continue
    rows.append({
      "source_label": edge["label_b"],
      "source_label_id": edge["label_b_id"],
      "candidate_targets": [
        {
          "label": edge["label_a"],
          "label_id": edge["label_a_id"],
          "relation": edge["relation"],
          "confidence": edge["confidence"]["max"],
          "soft_weight": edge["confidence"]["max"],
          "valid_from": _min_present(
            scope.get("valid_from") for scope in usable_scopes
          ),
          "valid_to": _max_present(
            scope.get("valid_to") for scope in usable_scopes
          ),
          "format": edge["format"],
          "evidence_observation_count": len(usable_scopes),
          "splits": sorted({
            str(scope["split_name"])
            for scope in usable_scopes
            if scope.get("split_name") is not None
          }),
          "rolling_windows": sorted({
            str(scope["rolling_window"])
            for scope in usable_scopes
            if scope.get("rolling_window") is not None
          }),
          "evidence_sources": edge["evidence_sources"],
        }
      ],
      "usable_for_training": True,
    })
  rows.sort(
    key=lambda row: (
      -float(row["candidate_targets"][0]["confidence"]),
      row["source_label"],
      row["candidate_targets"][0]["label"],
    )
  )
  return rows


def _min_present(values: Any) -> str | None:
  present = sorted(str(value) for value in values if value is not None)
  return present[0] if present else None


def _max_present(values: Any) -> str | None:
  present = sorted(str(value) for value in values if value is not None)
  return present[-1] if present else None

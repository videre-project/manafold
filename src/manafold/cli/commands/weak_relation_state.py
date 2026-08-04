from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from manafold.cli.common import SubparserRegistry, _artifact_path
from manafold.constants import PROJECT_ROOT
from manafold.taxonomy.weak_state import run_weak_relation_state


def register(subparsers: SubparserRegistry) -> None:
  parser = subparsers.add_parser(
    "weak-relation-state",
    help="Build leakage-safe weak relation state from weak evidence artifacts.",
  )
  parser.add_argument(
    "artifact_dir",
    type=Path,
    help="Directory containing weak relation artifacts.",
  )
  parser.add_argument("--alias-observations", type=Path)
  parser.add_argument("--weak-relation-graph", type=Path)
  parser.add_argument("--soft-target-suggestions", type=Path)
  parser.add_argument("--weak-target-preview", type=Path)
  parser.add_argument("--output", type=Path)
  parser.set_defaults(handler=run)


def run(args: Namespace) -> None:
  artifact_dir = args.artifact_dir
  if not artifact_dir.is_absolute():
    artifact_dir = PROJECT_ROOT / artifact_dir

  alias_observations = _artifact_path(
    args.alias_observations,
    artifact_dir,
    "alias_weak_label_observations.jsonl",
  )
  weak_relation_graph = _artifact_path(
    args.weak_relation_graph,
    artifact_dir,
    "weak_relation_graph.json",
  )
  soft_target_suggestions = _artifact_path(
    args.soft_target_suggestions,
    artifact_dir,
    "soft_canonical_target_suggestions.jsonl",
  )
  weak_target_preview = _artifact_path(
    args.weak_target_preview,
    artifact_dir,
    "weak_target_preview.json",
  )

  output = args.output
  if output is None:
    output = artifact_dir / "weak_relation_state.json"
  elif not output.is_absolute():
    output = PROJECT_ROOT / output

  result = run_weak_relation_state(
    alias_observations=alias_observations,
    weak_relation_graph=weak_relation_graph,
    soft_target_suggestions=soft_target_suggestions,
    weak_target_preview=weak_target_preview,
    output=output,
  )

  print(f"Wrote weak relation state to {output}")
  print(f"Usable now: {result['counts']['usable_now']}")
  print(f"Review queue: {result['counts']['review_queue']}")
  print(
    "Deferred soft canonical evidence: "
    f"{result['counts']['deferred_soft_canonical_evidence']}"
  )
  print(
    "Deferred review diagnostics: "
    f"{result['counts']['deferred_review_diagnostics']}"
  )
  print(f"Blocked: {result['counts']['blocked']}")

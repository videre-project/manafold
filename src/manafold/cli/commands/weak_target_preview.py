from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from manafold.cli.common import SubparserRegistry
from manafold.constants import PROJECT_ROOT
from manafold.taxonomy.weak_targets import run_weak_target_preview


def register(subparsers: SubparserRegistry) -> None:
  parser = subparsers.add_parser(
    "weak-target-preview",
    help="Preview training-row coverage for soft weak target suggestions.",
  )
  parser.add_argument(
    "rolling_result",
    type=Path,
    help=(
      "Path to rolling_evaluation_results.json, or a rolling-eval output directory."
    ),
  )
  parser.add_argument(
    "--weak-relation-graph",
    type=Path,
    required=True,
  )
  parser.add_argument(
    "--soft-target-suggestions",
    type=Path,
    required=True,
  )
  parser.add_argument("--output", type=Path)
  parser.add_argument(
    "--target-source",
    default="source_archetype_name_proxy",
  )
  parser.set_defaults(handler=run)


def run(args: Namespace) -> None:
  rolling_result = args.rolling_result
  if not rolling_result.is_absolute():
    rolling_result = PROJECT_ROOT / rolling_result

  weak_relation_graph = args.weak_relation_graph
  if not weak_relation_graph.is_absolute():
    weak_relation_graph = PROJECT_ROOT / weak_relation_graph

  soft_target_suggestions = args.soft_target_suggestions
  if not soft_target_suggestions.is_absolute():
    soft_target_suggestions = PROJECT_ROOT / soft_target_suggestions

  output = args.output
  if output is None:
    output = (
      rolling_result / "weak_target_preview.json"
      if rolling_result.is_dir()
      else rolling_result.parent / "weak_target_preview.json"
    )
  elif not output.is_absolute():
    output = PROJECT_ROOT / output

  result = run_weak_target_preview(
    rolling_result,
    weak_relation_graph=weak_relation_graph,
    soft_target_suggestions=soft_target_suggestions,
    output=output,
    target_source=args.target_source,
  )

  print(f"Wrote weak target preview to {output}")
  print(f"Suggestion count: {result['suggestion_count']}")
  print(f"Preview row count: {result['preview_row_count']}")
  print(
    "Rows with train coverage: "
    f"{result['rows_with_train_coverage_count']}"
  )
  print(f"Affected rows by split: {result['affected_rows_by_split']}")

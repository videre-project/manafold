from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from manafold.cli.common import SubparserRegistry
from manafold.constants import PROJECT_ROOT
from manafold.taxonomy.weak_relations import (
  DEFAULT_EXCLUDED_TRAINING_SPLITS,
  run_weak_relation_graph,
)


def register(subparsers: SubparserRegistry) -> None:
  parser = subparsers.add_parser(
    "weak-relation-graph",
    help="Build a time-scoped weak relation graph from alias observations.",
  )
  parser.add_argument(
    "observations",
    type=Path,
    help="Path to alias_weak_label_observations.jsonl.",
  )
  parser.add_argument("--output", type=Path)
  parser.add_argument("--suggestions-output", type=Path)
  parser.add_argument(
    "--exclude-training-split",
    action="append",
    dest="excluded_training_splits",
    default=None,
    help=(
      "Split whose evidence may appear in the graph but should not create "
      "soft target suggestions. May be repeated."
    ),
  )
  parser.set_defaults(handler=run)


def run(args: Namespace) -> None:
  observations = args.observations
  if not observations.is_absolute():
    observations = PROJECT_ROOT / observations

  output = args.output
  if output is None:
    output = observations.with_name("weak_relation_graph.json")
  elif not output.is_absolute():
    output = PROJECT_ROOT / output

  suggestions_output = args.suggestions_output
  if suggestions_output is not None and not suggestions_output.is_absolute():
    suggestions_output = PROJECT_ROOT / suggestions_output

  excluded_training_splits = tuple(
    args.excluded_training_splits
    if args.excluded_training_splits is not None
    else DEFAULT_EXCLUDED_TRAINING_SPLITS
  )

  result = run_weak_relation_graph(
    observations,
    output=output,
    suggestions_output=suggestions_output,
    excluded_training_splits=excluded_training_splits,
  )

  print(f"Wrote weak relation graph to {output}")
  print(
    "Wrote soft canonical target suggestions to "
    f"{result['soft_canonical_target_suggestions_path']}"
  )
  print(f"Observation count: {result['observation_count']}")
  print(f"Node count: {result['node_count']}")
  print(f"Edge count: {result['edge_count']}")
  print(
    "Soft canonical target suggestion count: "
    f"{result['soft_canonical_target_suggestion_count']}"
  )

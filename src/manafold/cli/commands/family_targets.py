from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from manafold.cli.common import SubparserRegistry
from manafold.constants import PROJECT_ROOT
from manafold.taxonomy.auto_ontology import (
  DEFAULT_CORE_CARD_FREQUENCY,
  DEFAULT_MIN_JACCARD_THRESHOLD,
  generate_auto_ontology,
)


def register(subparsers: SubparserRegistry) -> None:
  parser = subparsers.add_parser(
    "family-targets",
    help="Induce replayable A11 family targets from a training split.",
  )
  parser.add_argument("dataset", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--min-jaccard-threshold",
    type=float,
    default=DEFAULT_MIN_JACCARD_THRESHOLD,
    help="Minimum core-card Jaccard score for a proposed relation.",
  )
  parser.add_argument(
    "--core-card-frequency",
    type=float,
    default=DEFAULT_CORE_CARD_FREQUENCY,
    help="Exclusive within-label frequency threshold for core cards.",
  )
  parser.set_defaults(handler=run)


def run(args: Namespace) -> None:
  dataset_path = args.dataset
  if not dataset_path.is_absolute():
    dataset_path = PROJECT_ROOT / dataset_path

  output = args.output
  if not output.is_absolute():
    output = PROJECT_ROOT / output

  result = generate_auto_ontology(
    dataset_path,
    output,
    min_jaccard_threshold=args.min_jaccard_threshold,
    core_card_frequency=args.core_card_frequency,
  )

  summary = result["dataset_summary"]
  print(f"Wrote projected-family targets to {output}")
  print(
    f"Relations: {len(result['proposed_components'])}, "
    f"training examples: {summary['training_examples']}, "
    f"training labels: {summary['observed_training_labels']}"
  )

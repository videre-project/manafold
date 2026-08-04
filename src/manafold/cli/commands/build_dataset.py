from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from manafold.cli.common import SubparserRegistry, _date
from manafold.datasets.mtgo.build import DatasetBuildOptions, build_dataset


def register(subparsers: SubparserRegistry) -> None:
  parser = subparsers.add_parser(
    "dataset",
    help="Build a versioned model-training dataset from MTGO records.",
  )
  parser.add_argument("--format", default="modern", help="Format code to build.")
  parser.add_argument("--start", default="2023-01-01", type=_date)
  parser.add_argument("--end", default="2026-06-30", type=_date)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--dataset-version")
  parser.add_argument("--train-end", default="2025-06-30", type=_date)
  parser.add_argument("--validation-end", default="2025-12-31", type=_date)
  parser.add_argument(
    "--dev-test-end",
    type=_date,
    help=(
      "Optional final-holdout boundary. Events after validation and on or before "
      "this date are dev-test; later events are final-test."
    ),
  )
  parser.add_argument("--env-file", type=Path)
  parser.add_argument(
    "--limit-events",
    type=int,
    help="Use only the first N events in date order for a small sample.",
  )
  parser.add_argument(
    "--allow-empty",
    action="store_true",
    help=(
      "Write an empty dataset manifest instead of failing when a release "
      "format has no decks in the requested window."
    ),
  )
  parser.set_defaults(handler=run)


def run(args: Namespace) -> None:
  options = DatasetBuildOptions(
    format_code=args.format,
    start=args.start,
    end=args.end,
    output=args.output,
    dataset_version=args.dataset_version,
    train_end=args.train_end,
    validation_end=args.validation_end,
    dev_test_end=args.dev_test_end,
    env_file=args.env_file,
    limit_events=args.limit_events,
    allow_empty=args.allow_empty,
  )
  summary = build_dataset(options)
  print(f"Wrote {summary.dataset_version} to {summary.output}")
  print(
    "Rows: "
    f"{summary.deck_count} decks, "
    f"{summary.deck_card_count} deck-card entries, "
    f"{summary.card_count} canonical cards, "
    f"{summary.proxy_target_count} proxy targets"
  )

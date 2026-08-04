from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from manafold.cli.common import SubparserRegistry
from manafold.constants import PROJECT_ROOT
from manafold.datasets.verification import verify_dataset


def register(subparsers: SubparserRegistry) -> None:
  parser = subparsers.add_parser(
    "check",
    help="Verify dataset files, schemas, and cross-file consistency.",
  )
  parser.add_argument("dataset", type=Path)
  parser.set_defaults(handler=run)


def run(args: Namespace) -> None:
  dataset_path = args.dataset
  if not dataset_path.is_absolute():
    dataset_path = PROJECT_ROOT / dataset_path

  result = verify_dataset(dataset_path)
  print(f"Verified {result.dataset_version} at {result.dataset_path}")
  print(
    "Files: "
    f"{result.checked_files} total, "
    f"{result.checked_parquet_files} Parquet"
  )

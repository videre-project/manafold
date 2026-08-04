from __future__ import annotations

import argparse
from collections.abc import Sequence

from manafold.cli.commands import COMMAND_MODULES


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="manafold",
    description="Build Manafold datasets and model results.",
  )
  subparsers = parser.add_subparsers(dest="command", required=True)
  for command_module in COMMAND_MODULES:
    command_module.register(subparsers)
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  args = build_parser().parse_args(argv)
  args.handler(args)

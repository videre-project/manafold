from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from manafold.constants import PROJECT_ROOT
from manafold.models.training.rolling_evaluation import parse_rolling_window


class SubparserRegistry(Protocol):
  def add_parser(
    self,
    name: str,
    **kwargs: Any,
  ) -> argparse.ArgumentParser:
    ...


def _date(value: str) -> date:
  return date.fromisoformat(value)


def _seeds(value: str) -> tuple[int, ...]:
  try:
    seeds = tuple(
      int(seed.strip())
      for seed in value.split(",")
      if seed.strip()
    )
  except ValueError as error:
    raise argparse.ArgumentTypeError(
      "seeds must be comma-separated integers"
    ) from error
  if not seeds:
    raise argparse.ArgumentTypeError("at least one seed is required")
  return seeds


def _rolling_window(value: str):
  try:
    return parse_rolling_window(value)
  except ValueError as error:
    raise argparse.ArgumentTypeError(str(error)) from error


def _artifact_path(path: Path | None, artifact_dir: Path, filename: str) -> Path:
  if path is None:
    return artifact_dir / filename
  if path.is_absolute():
    return path
  return PROJECT_ROOT / path


def _repo_path_or_none(path: Path | None) -> Path | None:
  if path is None or path.is_absolute():
    return path
  return PROJECT_ROOT / path


def _mean_std(summary: dict[str, object], metric_name: str) -> str:
  metric = summary.get(metric_name)
  if not isinstance(metric, dict):
    return "n/a"
  mean = metric.get("mean")
  std = metric.get("std")
  if mean is None or std is None:
    return "n/a"
  return f"{mean:.4f} +/- {std:.4f}"

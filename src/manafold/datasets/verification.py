from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from manafold.datasets.schemas import (
  PARQUET_SCHEMAS,
  REQUIRED_DATASET_FILES,
  ZONE_VOCAB,
  parquet_schema,
)
from manafold.datasets.validation import (
  validate_model_inputs,
  validate_split_manifest,
)


@dataclass(frozen=True)
class DatasetCheckResult:
  dataset_path: Path
  dataset_version: str
  checked_files: int
  checked_parquet_files: int


def verify_dataset(dataset_path: Path) -> DatasetCheckResult:
  manifest = _read_json(
    dataset_path / "dataset_manifest.json", "dataset manifest"
  )
  artifacts = _mapping(manifest, "artifacts", "dataset manifest")
  row_counts = _mapping(manifest, "row_counts", "dataset manifest")
  issues: list[str] = []

  for file_stem, relative_path in sorted(artifacts.items()):
    if not (dataset_path / relative_path).exists():
      issues.append(f"Missing file {file_stem}: {relative_path}")

  for file_stem in sorted(REQUIRED_DATASET_FILES - set(artifacts)):
    issues.append(f"Manifest is missing required dataset file: {file_stem}")

  parquet_rows: dict[str, list[dict[str, Any]]] = {}
  for file_stem, relative_path in sorted(artifacts.items()):
    if file_stem not in PARQUET_SCHEMAS:
      continue

    path = dataset_path / relative_path
    if not path.exists():
      continue

    table = pq.read_table(path)
    if not table.schema.equals(
      parquet_schema(file_stem), check_metadata=False
    ):
      issues.append(f"{file_stem} schema does not match its declared schema.")

    expected_count = row_counts.get(file_stem)
    if expected_count is not None and table.num_rows != expected_count:
      issues.append(
        f"{file_stem} row count is {table.num_rows}, "
        f"but manifest records {expected_count}."
      )

    parquet_rows[file_stem] = table.to_pylist()

  _validate_zone_vocab(dataset_path, artifacts, issues)
  _validate_loaded_artifacts(parquet_rows, manifest, issues)
  _raise_if_any("Dataset verification failed", issues)

  return DatasetCheckResult(
    dataset_path=dataset_path,
    dataset_version=str(manifest["dataset_version"]),
    checked_files=len(artifacts),
    checked_parquet_files=len(parquet_rows),
  )


def _validate_loaded_artifacts(
  rows: dict[str, list[dict[str, Any]]],
  manifest: dict[str, Any],
  issues: list[str],
) -> None:
  required = {"card_vocab", "deck_tokens", "split_manifest"}
  missing = sorted(required - set(rows))
  if missing:
    issues.append(
      f"Cannot run cross-file validation; missing: {', '.join(missing)}"
    )
    return

  try:
    validate_split_manifest(
      rows["split_manifest"],
      require_all_splits=manifest.get("limit_events") is None,
    )
    validate_model_inputs(rows["card_vocab"], rows["deck_tokens"])
  except RuntimeError as error:
    issues.append(str(error))


def _validate_zone_vocab(
  dataset_path: Path,
  artifacts: dict[str, Any],
  issues: list[str],
) -> None:
  relative_path = artifacts.get("zone_vocab")
  if not relative_path:
    return

  try:
    zone_vocab = _read_json(dataset_path / relative_path, "zone vocabulary")
  except (json.JSONDecodeError, RuntimeError) as error:
    issues.append(str(error))
    return

  if {
    str(zone): int(index)
    for zone, index in zone_vocab.items()
    if isinstance(zone, str) and isinstance(index, int)
  } != ZONE_VOCAB:
    issues.append("zone_vocab.json does not match the expected zone vocabulary.")


def _read_json(path: Path, file_description: str) -> dict[str, Any]:
  if not path.exists():
    raise RuntimeError(f"Missing {file_description}: {path}")

  data = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(data, dict):
    raise RuntimeError(f"{path} must contain a JSON object.")
  return data


def _mapping(
  row: dict[str, Any],
  field: str,
  context: str,
) -> dict[str, Any]:
  value = row.get(field)
  if not isinstance(value, dict):
    raise RuntimeError(f"{context} is missing {field}.")
  return value


def _raise_if_any(title: str, issues: list[str]) -> None:
  if not issues:
    return

  sample = "\n".join(f"- {issue}" for issue in issues[:20])
  remaining = len(issues) - 20
  if remaining > 0:
    sample = f"{sample}\n- ... and {remaining} more"
  raise RuntimeError(f"{title}:\n{sample}")

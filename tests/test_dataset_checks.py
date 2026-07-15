from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from manafold.data.validate import parquet_schema, verify_dataset_export


class DatasetCheckTests(unittest.TestCase):
  def test_dataset_export_verifies(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      dataset_path = _write_dataset(Path(directory))

      result = verify_dataset_export(dataset_path)

      self.assertEqual("modern_2024_2024_v0", result.dataset_version)
      self.assertEqual(5, result.checked_artifacts)
      self.assertEqual(4, result.checked_parquet_artifacts)

  def test_row_count_mismatch_fails(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      dataset_path = _write_dataset(Path(directory))
      manifest_path = dataset_path / "dataset_manifest.json"
      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      manifest["row_counts"]["deck_tokens"] = 2
      manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

      with self.assertRaisesRegex(RuntimeError, "deck_tokens row count"):
        verify_dataset_export(dataset_path)

  def test_missing_required_dataset_file_fails(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      dataset_path = _write_dataset(Path(directory))
      manifest_path = dataset_path / "dataset_manifest.json"
      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      del manifest["artifacts"]["proxy_targets"]
      manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

      with self.assertRaisesRegex(RuntimeError, "proxy_targets"):
        verify_dataset_export(dataset_path)

  def test_missing_artifact_reports_path(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      dataset_path = _write_dataset(Path(directory))
      (dataset_path / "deck_tokens.parquet").unlink()

      with self.assertRaisesRegex(RuntimeError, "Missing artifact deck_tokens"):
        verify_dataset_export(dataset_path)


def _write_dataset(root: Path) -> Path:
  dataset_path = root / "dataset"
  dataset_path.mkdir()

  parquet_rows = _parquet_rows()
  for artifact_name, rows in parquet_rows.items():
    _write_parquet(dataset_path / f"{artifact_name}.parquet", artifact_name, rows)

  _write_json(
    dataset_path / "zone_vocab.json",
    {
      "main": 0,
      "side": 1,
      "companion": 2,
      "commander": 3,
      "other": 4,
    },
  )
  _write_json(
    dataset_path / "dataset_manifest.json",
    {
      "dataset_version": "modern_2024_2024_v0",
      "format": "modern",
      "source": "mtgo-db",
      "start": "2024-01-01",
      "end": "2024-01-01",
      "train_end": "2024-12-31",
      "validation_end": "2024-12-31",
      "limit_events": 1,
      "artifacts": {
        **{
          artifact_name: f"{artifact_name}.parquet"
          for artifact_name in parquet_rows
        },
        "zone_vocab": "zone_vocab.json",
      },
      "row_counts": {
        artifact_name: len(rows)
        for artifact_name, rows in parquet_rows.items()
      },
    },
  )

  return dataset_path


def _parquet_rows() -> dict[str, list[dict[str, Any]]]:
  return {
    "card_vocab": [
      {
        "dataset_version": "modern_2024_2024_v0",
        "card_idx": 0,
        "oracle_id": "oracle-1",
        "primary_name": "Lightning Bolt",
        "first_seen_at": None,
        "last_seen_at": None,
      }
    ],
    "deck_tokens": [
      {
        "dataset_version": "modern_2024_2024_v0",
        "deck_id": "deck-1",
        "card_idx": 0,
        "oracle_id": "oracle-1",
        "quantity": 60,
        "zone_idx": 0,
        "zone": "main",
      }
    ],
    "proxy_targets": [
      {
        "dataset_version": "modern_2024_2024_v0",
        "deck_id": "deck-1",
        "target_source": "source_archetype_name_proxy",
        "target_level": "family",
        "proxy_label_id": "proxy.source_archetype_name_proxy.burn",
        "display_label": "Burn",
        "normalized_label": "burn",
        "source_field": "source_archetype_name",
        "confidence": 0.35,
        "provenance": "mtgo-db:archetypes.archetype|proxy_target",
      }
    ],
    "split_manifest": [
      {
        "dataset_version": "modern_2024_2024_v0",
        "deck_id": "deck-1",
        "event_id": "event-1",
        "event_date": date(2024, 1, 1),
        "format": "modern",
        "source": "mtgo-db",
        "split_name": "train",
        "split_strategy": "event_forward_modern_2024_2024_v0",
      }
    ],
  }


def _write_parquet(
  path: Path,
  artifact_name: str,
  rows: list[dict[str, Any]],
) -> None:
  table = pa.Table.from_pylist(rows, schema=parquet_schema(artifact_name))
  pq.write_table(table, path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
  path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
  unittest.main()

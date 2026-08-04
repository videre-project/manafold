from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from manafold.datasets.model_inputs import load_training_dataset
from manafold.datasets.mtgo.build import DatasetBuildOptions, build_dataset
from manafold.datasets.mtgo.database import MtgoDatasetRecords
from manafold.datasets.verification import verify_dataset


class DatasetBuildTests(unittest.TestCase):
  def test_build_dataset_writes_a_verified_training_dataset(self) -> None:
    deck_records = [
      _deck("train", date(2024, 1, 15), "Alpha"),
      _deck("validation", date(2024, 4, 15), "Alpha"),
      _deck("dev-test", date(2024, 7, 15), "Beta"),
      _deck("final-test", date(2024, 10, 15), "Beta"),
    ]
    records = MtgoDatasetRecords(
      deck_records=deck_records,
      deck_cards=[_deck_card(deck["deck_id"]) for deck in deck_records],
      card_catalog=[
        {
          "oracle_id": "oracle-example",
          "primary_name": "Example Card",
          "first_seen_at": None,
          "last_seen_at": None,
        }
      ],
    )

    with tempfile.TemporaryDirectory() as temp_dir:
      output = Path(temp_dir) / "dataset"
      options = DatasetBuildOptions(
        format_code="modern",
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        output=output,
        dataset_version="modern_build_contract_v0",
        train_end=date(2024, 3, 31),
        validation_end=date(2024, 6, 30),
        dev_test_end=date(2024, 9, 30),
        env_file=None,
        limit_events=None,
      )

      with patch(
        "manafold.datasets.mtgo.build.load_mtgo_dataset_records",
        return_value=records,
      ) as load_records:
        summary = build_dataset(options)

      request = load_records.call_args.kwargs
      self.assertEqual("modern", request["scope"].format_code)
      self.assertEqual(date(2024, 1, 1), request["scope"].start)
      self.assertEqual(date(2024, 12, 31), request["scope"].end)
      self.assertEqual("modern_build_contract_v0", request["dataset_version"])
      self.assertEqual(output, summary.output)
      self.assertEqual(len(deck_records), summary.deck_count)
      self.assertEqual(len(deck_records), summary.proxy_target_count)

      verification = verify_dataset(output)
      dataset = load_training_dataset(output)
      self.assertEqual(summary.dataset_version, verification.dataset_version)
      self.assertEqual(
        {
          "train": "train",
          "validation": "validation",
          "dev-test": "dev-test",
          "final-test": "final-test",
        },
        {example.deck_id: example.split_name for example in dataset.examples},
      )
      self.assertEqual(
        {
          "proxy.source_archetype_name_proxy.alpha",
          "proxy.source_archetype_name_proxy.beta",
        },
        {example.target_label_id for example in dataset.examples},
      )


def _deck(deck_id: str, event_date: date, label: str) -> dict[str, object]:
  return {
    "dataset_version": "modern_build_contract_v0",
    "deck_id": deck_id,
    "format": "modern",
    "event_id": f"event-{deck_id}",
    "event_date": event_date,
    "source": "mtgo-db",
    "player_id": f"player-{deck_id}",
    "standing_rank": None,
    "reported_archetype": label,
    "source_archetype_name": label,
    "mtgo_db_archetype_id": None,
    "source_archetype_id": None,
  }


def _deck_card(deck_id: object) -> dict[str, object]:
  return {
    "dataset_version": "modern_build_contract_v0",
    "deck_id": deck_id,
    "resolved_card_id": 1,
    "oracle_id": "oracle-example",
    "source_card_id": 1,
    "name": "Example Card",
    "quantity": 60,
    "zone": "main",
  }


if __name__ == "__main__":
  unittest.main()

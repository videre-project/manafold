from __future__ import annotations

import unittest
from datetime import date

from manafold.datasets.validation import (
  validate_deck_cards,
  validate_deck_records,
  validate_model_inputs,
  validate_split_manifest,
)


class ValidationTests(unittest.TestCase):
  def test_deck_records_require_event_identity(self) -> None:
    with self.assertRaisesRegex(RuntimeError, "event_id"):
      validate_deck_records(
        [
          {
            "dataset_version": "modern_2023_2026_v0",
            "deck_id": "deck-1",
            "format": "modern",
            "event_id": None,
            "event_date": date(2024, 1, 1),
            "source": "mtgo-db",
          }
        ]
      )

  def test_split_manifest_requires_all_model_splits_for_full_exports(self) -> None:
    with self.assertRaisesRegex(RuntimeError, "test, validation"):
      validate_split_manifest(
        [
          _split_row("deck-1", "event-1", date(2024, 1, 1), "train"),
        ],
        require_all_splits=True,
      )

  def test_split_manifest_allows_sample_export_without_all_splits(self) -> None:
    validate_split_manifest(
      [
        _split_row("deck-1", "event-1", date(2024, 1, 1), "train"),
      ],
      require_all_splits=False,
    )

  def test_split_manifest_allows_fresh_final_holdout_splits(self) -> None:
    validate_split_manifest(
      [
        _split_row("deck-1", "event-1", date(2024, 1, 1), "train"),
        _split_row("deck-2", "event-2", date(2024, 7, 1), "validation"),
        _split_row("deck-3", "event-3", date(2025, 1, 1), "dev-test"),
        _split_row("deck-4", "event-4", date(2025, 7, 1), "final-test"),
      ],
      require_all_splits=True,
    )

  def test_split_manifest_rejects_mixed_test_and_final_holdout_splits(self) -> None:
    with self.assertRaisesRegex(RuntimeError, "cannot mix"):
      validate_split_manifest(
        [
          _split_row("deck-1", "event-1", date(2024, 1, 1), "train"),
          _split_row("deck-2", "event-2", date(2024, 7, 1), "validation"),
          _split_row("deck-3", "event-3", date(2025, 1, 1), "test"),
          _split_row("deck-4", "event-4", date(2025, 7, 1), "final-test"),
        ],
        require_all_splits=True,
      )

  def test_split_manifest_rejects_event_leakage(self) -> None:
    with self.assertRaisesRegex(RuntimeError, "multiple splits"):
      validate_split_manifest(
        [
          _split_row("deck-1", "event-1", date(2024, 1, 1), "train"),
          _split_row("deck-2", "event-1", date(2024, 7, 1), "validation"),
          _split_row("deck-3", "event-2", date(2025, 1, 1), "test"),
        ],
        require_all_splits=True,
      )

  def test_deck_cards_reject_invalid_model_inputs(self) -> None:
    deck_records = [
      {
        "dataset_version": "modern_2023_2026_v0",
        "deck_id": "deck-1",
        "format": "modern",
        "event_id": "event-1",
        "event_date": date(2024, 1, 1),
        "source": "mtgo-db",
      }
    ]

    with self.assertRaisesRegex(RuntimeError, "invalid quantity"):
      validate_deck_cards(
        deck_records,
        [
          _card_row("deck-1", quantity=0),
        ],
      )

  def test_model_inputs_reject_mismatched_zone_index(self) -> None:
    with self.assertRaisesRegex(RuntimeError, "zone_idx"):
      validate_model_inputs(
        [_vocab_row(0, "oracle-a")],
        [
          {
            "dataset_version": "modern_2023_2026_v0",
            "deck_id": "deck-1",
            "card_idx": 0,
            "oracle_id": "oracle-a",
            "quantity": 4,
            "zone_idx": 1,
            "zone": "main",
          }
        ],
      )


def _split_row(
  deck_id: str,
  event_id: str,
  event_date: date,
  split_name: str,
) -> dict[str, object]:
  return {
    "dataset_version": "modern_2023_2026_v0",
    "deck_id": deck_id,
    "event_id": event_id,
    "event_date": event_date,
    "format": "modern",
    "source": "mtgo-db",
    "split_name": split_name,
    "split_strategy": "event_forward_modern_2023_2026_v0",
  }


def _card_row(
  deck_id: str,
  *,
  source_card_id: int = 1,
  quantity: int = 60,
  zone: str = "main",
) -> dict[str, object]:
  return {
    "dataset_version": "modern_2023_2026_v0",
    "deck_id": deck_id,
    "resolved_card_id": source_card_id,
    "oracle_id": f"oracle-{source_card_id}",
    "source_card_id": source_card_id,
    "name": f"Card {source_card_id}",
    "quantity": quantity,
    "zone": zone,
  }


def _vocab_row(card_idx: int, oracle_id: str) -> dict[str, object]:
  return {
    "dataset_version": "modern_2023_2026_v0",
    "card_idx": card_idx,
    "oracle_id": oracle_id,
    "primary_name": "Card",
    "first_seen_at": None,
    "last_seen_at": None,
  }


if __name__ == "__main__":
  unittest.main()

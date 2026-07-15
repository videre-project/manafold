from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from manafold.data.export import (
  DatasetExportOptions,
  ZONE_VOCAB,
  _build_split_manifest,
  _validate_export_options,
  build_card_vocab,
  build_deck_tokens,
)


class ModelInputTests(unittest.TestCase):
  def test_card_vocab_is_stable_zero_based_and_deduplicated(self) -> None:
    catalog = [
      _catalog_row("oracle-b", "Zagoth Triome"),
      _catalog_row("oracle-a", "Lightning Bolt"),
      _catalog_row("oracle-a", "Lightning Bolt"),
    ]

    vocab = build_card_vocab(catalog, "modern_2023_2026_v0")

    self.assertEqual(
      [
        {
          "dataset_version": "modern_2023_2026_v0",
          "card_idx": 0,
          "oracle_id": "oracle-a",
          "primary_name": "Lightning Bolt",
          "first_seen_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
          "last_seen_at": datetime(2024, 1, 2, tzinfo=timezone.utc),
        },
        {
          "dataset_version": "modern_2023_2026_v0",
          "card_idx": 1,
          "oracle_id": "oracle-b",
          "primary_name": "Zagoth Triome",
          "first_seen_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
          "last_seen_at": datetime(2024, 1, 2, tzinfo=timezone.utc),
        },
      ],
      vocab,
    )

  def test_deck_tokens_use_card_and_zone_indexes(self) -> None:
    vocab = [
      _vocab_row(0, "oracle-a", "Lightning Bolt"),
      _vocab_row(1, "oracle-b", "Zagoth Triome"),
    ]
    deck_cards = [
      _deck_card("deck-1", "oracle-b", quantity=2, zone="side"),
      _deck_card("deck-1", "oracle-a", quantity=4, zone="main"),
    ]

    tokens = build_deck_tokens(deck_cards, vocab)

    self.assertEqual(
      [
        {
          "dataset_version": "modern_2023_2026_v0",
          "deck_id": "deck-1",
          "card_idx": 0,
          "oracle_id": "oracle-a",
          "quantity": 4,
          "zone_idx": ZONE_VOCAB["main"],
          "zone": "main",
        },
        {
          "dataset_version": "modern_2023_2026_v0",
          "deck_id": "deck-1",
          "card_idx": 1,
          "oracle_id": "oracle-b",
          "quantity": 2,
          "zone_idx": ZONE_VOCAB["side"],
          "zone": "side",
        },
      ],
      tokens,
    )

  def test_split_manifest_can_create_fresh_final_holdout(self) -> None:
    options = DatasetExportOptions(
      format_code="modern",
      start=date(2024, 1, 1),
      end=date(2024, 12, 31),
      output=None,
      dataset_version=None,
      train_end=date(2024, 3, 31),
      validation_end=date(2024, 6, 30),
      dev_test_end=date(2024, 9, 30),
      env_file=None,
      limit_events=None,
    )
    split_manifest = _build_split_manifest(
      [
        _deck_example("deck-1", "event-1", date(2024, 1, 1)),
        _deck_example("deck-2", "event-2", date(2024, 5, 1)),
        _deck_example("deck-3", "event-3", date(2024, 8, 1)),
        _deck_example("deck-4", "event-4", date(2024, 11, 1)),
      ],
      options,
      "modern_2024_2024_v0",
    )

    self.assertEqual(
      ["train", "validation", "dev-test", "final-test"],
      [row["split_name"] for row in split_manifest],
    )
    self.assertEqual(
      {
        "event_forward_final_holdout_modern_2024_2024_v0",
      },
      {row["split_strategy"] for row in split_manifest},
    )

  def test_export_options_require_strict_split_boundaries(self) -> None:
    options = DatasetExportOptions(
      format_code="modern",
      start=date(2024, 1, 1),
      end=date(2024, 12, 31),
      output=None,
      dataset_version=None,
      train_end=date(2024, 3, 31),
      validation_end=date(2024, 6, 30),
      dev_test_end=date(2024, 9, 30),
      env_file=None,
      limit_events=None,
    )
    _validate_export_options(options)

    with self.assertRaisesRegex(ValueError, "before --validation-end"):
      _validate_export_options(
        DatasetExportOptions(
          format_code="modern",
          start=date(2024, 1, 1),
          end=date(2024, 12, 31),
          output=None,
          dataset_version=None,
          train_end=date(2024, 6, 30),
          validation_end=date(2024, 6, 30),
          dev_test_end=date(2024, 9, 30),
          env_file=None,
          limit_events=None,
        )
      )

    with self.assertRaisesRegex(ValueError, "before --end"):
      _validate_export_options(
        DatasetExportOptions(
          format_code="modern",
          start=date(2024, 1, 1),
          end=date(2024, 12, 31),
          output=None,
          dataset_version=None,
          train_end=date(2024, 3, 31),
          validation_end=date(2024, 12, 31),
          dev_test_end=None,
          env_file=None,
          limit_events=None,
        )
      )


def _catalog_row(oracle_id: str, primary_name: str) -> dict[str, object]:
  return {
    "oracle_id": oracle_id,
    "primary_name": primary_name,
    "is_token": None,
    "first_seen_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
    "last_seen_at": datetime(2024, 1, 2, tzinfo=timezone.utc),
    "colors": None,
    "color_identity": None,
    "mana_value": None,
    "type_line": None,
    "oracle_text": None,
  }


def _vocab_row(
  card_idx: int,
  oracle_id: str,
  primary_name: str,
) -> dict[str, object]:
  return {
    "dataset_version": "modern_2023_2026_v0",
    "card_idx": card_idx,
    "oracle_id": oracle_id,
    "primary_name": primary_name,
    "first_seen_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
    "last_seen_at": datetime(2024, 1, 2, tzinfo=timezone.utc),
  }


def _deck_card(
  deck_id: str,
  oracle_id: str,
  *,
  quantity: int,
  zone: str,
) -> dict[str, object]:
  return {
    "dataset_version": "modern_2023_2026_v0",
    "deck_id": deck_id,
    "resolved_card_id": 1,
    "oracle_id": oracle_id,
    "source_card_id": 1,
    "name": "Card",
    "quantity": quantity,
    "zone": zone,
  }


def _deck_example(
  deck_id: str,
  event_id: str,
  event_date: date,
) -> dict[str, object]:
  return {
    "dataset_version": "modern_2024_2024_v0",
    "deck_id": deck_id,
    "event_id": event_id,
    "event_date": event_date,
    "format": "modern",
    "source": "mtgo-db",
  }

if __name__ == "__main__":
  unittest.main()

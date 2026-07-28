from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

from manafold.models.card_ranking import build_family_card_ranking
from manafold.models.data import DeckToken, ModelExample, TrainingDataset


class CardRankingTests(unittest.TestCase):
  def test_ranks_family_specific_training_cards(self) -> None:
    dataset = TrainingDataset(
      dataset_path=Path("/unused"),
      dataset_version="test",
      target_source="source",
      card_vocab=(
        {"card_idx": 0, "oracle_id": "shared", "primary_name": "Shared"},
        {"card_idx": 1, "oracle_id": "alpha", "primary_name": "Alpha"},
        {"card_idx": 2, "oracle_id": "beta", "primary_name": "Beta"},
      ),
      zone_vocab={"main": 0, "side": 1},
      examples=(
        self._example("a1", "alpha", (0, 1)),
        self._example("a2", "alpha", (0, 1)),
        self._example("b1", "beta", (0, 2)),
        self._example("heldout", "beta", (1,), split="final-test"),
      ),
      labels=("alpha", "beta"),
    )
    ranking = build_family_card_ranking(
      dataset,
      artifact_card_vocab=dataset.card_vocab,
      family_vocab={
        "families": [
          {"family_id": "family.alpha"},
          {"family_id": "family.beta"},
        ],
        "entries": [
          {"label_id": "alpha", "family_id": "family.alpha"},
          {"label_id": "beta", "family_id": "family.beta"},
        ],
      },
      shrinkage_decks=0,
    )

    self.assertEqual(3, ranking["training_deck_count"])
    self.assertEqual(1, ranking["families"]["family.alpha"][0]["card_idx"])
    self.assertGreater(ranking["families"]["family.alpha"][0]["score"], 0.6)
    self.assertNotIn(
      0,
      {
        row["card_idx"]
        for row in ranking["families"]["family.alpha"]
      },
    )

  @staticmethod
  def _example(
    deck_id: str,
    label: str,
    cards: tuple[int, ...],
    *,
    split: str = "train",
  ) -> ModelExample:
    return ModelExample(
      dataset_version="test",
      deck_id=deck_id,
      event_id=f"event-{deck_id}",
      event_date=date(2026, 1, 1),
      format_code="modern",
      split_name=split,
      target_label_id=label,
      source_label_id=label,
      target_source="source",
      tokens=tuple(
        DeckToken(card_idx=card_idx, quantity=4, zone_idx=0)
        for card_idx in cards
      ),
    )


if __name__ == "__main__":
  unittest.main()

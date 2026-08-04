from __future__ import annotations

import math
from collections import Counter
from typing import Any

from manafold.datasets.model_inputs import TrainingDataset

CARD_RANKING_VERSION = "manafold_family_card_ranking_v1"
CARD_RANKING_METHOD = "smoothed_family_adoption_log_lift"
DEFAULT_MAX_CARDS_PER_FAMILY = 32
DEFAULT_MIN_FAMILY_ADOPTION = 0.05
DEFAULT_MIN_FAMILY_CARD_DECKS = 2
DEFAULT_SHRINKAGE_DECKS = 20.0
DEFAULT_SMOOTHING = 1.0


def build_family_card_ranking(
  dataset: TrainingDataset,
  *,
  saved_card_vocab: tuple[dict[str, Any], ...],
  family_vocab: dict[str, Any],
  max_cards_per_family: int = DEFAULT_MAX_CARDS_PER_FAMILY,
  min_family_adoption: float = DEFAULT_MIN_FAMILY_ADOPTION,
  min_family_card_decks: int = DEFAULT_MIN_FAMILY_CARD_DECKS,
  smoothing: float = DEFAULT_SMOOTHING,
  shrinkage_decks: float = DEFAULT_SHRINKAGE_DECKS,
) -> dict[str, Any]:
  """Rank mainboard cards by training-only family adoption distinctiveness."""
  if max_cards_per_family <= 0:
    raise ValueError("max_cards_per_family must be positive.")
  if not 0 <= min_family_adoption <= 1:
    raise ValueError("min_family_adoption must be between zero and one.")
  if min_family_card_decks <= 0:
    raise ValueError("min_family_card_decks must be positive.")
  if smoothing <= 0:
    raise ValueError("smoothing must be positive.")
  if shrinkage_decks < 0:
    raise ValueError("shrinkage_decks cannot be negative.")

  family_by_label = {
    str(entry["label_id"]): str(entry["family_id"])
    for entry in family_vocab["entries"]
  }
  family_ids = {
    str(family["family_id"]) for family in family_vocab["families"]
  }
  saved_idx_by_oracle = {
    str(card["oracle_id"]): int(card["card_idx"]) for card in saved_card_vocab
  }
  saved_idx_by_dataset_idx = {
    int(card["card_idx"]): saved_idx_by_oracle.get(str(card["oracle_id"]))
    for card in dataset.card_vocab
  }
  main_zone_idx = dataset.zone_vocab.get("main")
  if main_zone_idx is None:
    raise ValueError("Ranking dataset does not define a mainboard zone.")

  family_decks: Counter[str] = Counter()
  global_card_decks: Counter[int] = Counter()
  family_card_decks: dict[str, Counter[int]] = {
    family_id: Counter() for family_id in family_ids
  }
  training_deck_count = 0

  for example in dataset.examples:
    if example.split_name != "train":
      continue
    family_id = family_by_label.get(example.target_label_id)
    if family_id is None:
      continue
    cards = {
      saved_idx
      for token in example.tokens
      if token.zone_idx == main_zone_idx
      for saved_idx in (saved_idx_by_dataset_idx.get(token.card_idx),)
      if saved_idx is not None
    }
    if not cards:
      continue
    training_deck_count += 1
    family_decks[family_id] += 1
    global_card_decks.update(cards)
    family_card_decks[family_id].update(cards)

  families: dict[str, list[dict[str, Any]]] = {}
  for family_id in sorted(family_ids):
    family_count = family_decks[family_id]
    if family_count == 0:
      families[family_id] = []
      continue
    outside_count = training_deck_count - family_count
    reliability = family_count / (family_count + shrinkage_decks)
    scored: list[tuple[float, int]] = []
    for card_idx, in_family_count in family_card_decks[family_id].items():
      empirical_family_adoption = in_family_count / family_count
      if (
        in_family_count < min_family_card_decks
        or empirical_family_adoption < min_family_adoption
      ):
        continue
      outside_card_count = global_card_decks[card_idx] - in_family_count
      global_adoption = global_card_decks[card_idx] / training_deck_count
      family_adoption = (in_family_count + smoothing * global_adoption) / (
        family_count + smoothing
      )
      outside_adoption = (outside_card_count + smoothing * global_adoption) / (
        outside_count + smoothing
      )
      raw_score = max(
        0.0,
        math.log(family_adoption / max(outside_adoption, 1e-12))
        * reliability
        * math.sqrt(empirical_family_adoption),
      )
      score = 1.0 - math.exp(-raw_score)
      if score > 0:
        scored.append((score, card_idx))

    scored.sort(key=lambda row: (-row[0], row[1]))
    scored = scored[:max_cards_per_family]
    families[family_id] = [
      {
        "card_idx": card_idx,
        "score": round(score, 6),
      }
      for score, card_idx in scored
    ]

  return {
    "version": CARD_RANKING_VERSION,
    "method": CARD_RANKING_METHOD,
    "scope": "train_mainboard_card_presence",
    "training_deck_count": training_deck_count,
    "parameters": {
      "max_cards_per_family": max_cards_per_family,
      "min_family_adoption": min_family_adoption,
      "min_family_card_decks": min_family_card_decks,
      "smoothing": smoothing,
      "shrinkage_decks": shrinkage_decks,
    },
    "families": families,
  }

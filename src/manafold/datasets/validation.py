from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from typing import Any

from manafold.datasets.schemas import (
  FINAL_HOLDOUT_REQUIRED_SPLITS,
  REQUIRED_SPLITS,
  VALID_SPLITS,
  VALID_ZONES,
  ZONE_VOCAB,
)


def validate_deck_records(deck_records: list[dict[str, Any]]) -> None:
  issues: list[str] = []
  if not deck_records:
    issues.append("The dataset contains no deck records.")

  deck_ids: set[str] = set()
  for row in deck_records:
    deck_id = row.get("deck_id")
    if not deck_id:
      issues.append("Deck example is missing deck_id.")
    elif deck_id in deck_ids:
      issues.append(f"Deck example appears more than once: {deck_id}")
    else:
      deck_ids.add(deck_id)

    _require(row, "format", "deck example", issues)
    _require(row, "event_id", f"deck {deck_id or '<missing>'}", issues)
    _require(row, "event_date", f"deck {deck_id or '<missing>'}", issues)
    _require(row, "source", f"deck {deck_id or '<missing>'}", issues)

  _raise_if_any("Deck example validation failed", issues)


def validate_deck_cards(
  deck_records: list[dict[str, Any]],
  deck_cards: list[dict[str, Any]],
) -> None:
  issues: list[str] = []
  deck_formats = {
    row["deck_id"]: str(row["format"]).lower()
    for row in deck_records
    if row.get("deck_id") and row.get("format")
  }
  deck_zone_counts: dict[str, Counter[str]] = defaultdict(Counter)

  if not deck_cards:
    issues.append("The dataset contains no deck-card records.")

  for row in deck_cards:
    deck_id = row.get("deck_id")
    if not deck_id:
      issues.append("Deck-card row is missing deck_id.")
      continue

    if deck_id not in deck_formats:
      issues.append(f"Deck-card row references unknown deck_id: {deck_id}")

    quantity = row.get("quantity")
    if not isinstance(quantity, int) or quantity <= 0:
      issues.append(f"Deck {deck_id} has invalid quantity: {quantity!r}")
    elif row.get("zone") in VALID_ZONES:
      deck_zone_counts[deck_id][row["zone"]] += quantity

    if not row.get("oracle_id"):
      issues.append(f"Deck {deck_id} has a card row without oracle_id.")

    zone = row.get("zone")
    if zone not in VALID_ZONES:
      issues.append(f"Deck {deck_id} has unknown zone: {zone!r}")

  _raise_if_any("Deck-card validation failed", issues)


def validate_split_manifest(
  split_manifest: list[dict[str, Any]],
  *,
  require_all_splits: bool,
) -> None:
  issues: list[str] = []
  if not split_manifest:
    issues.append("Split manifest is empty.")

  deck_ids: set[str] = set()
  splits_by_event: dict[str, set[str]] = defaultdict(set)
  dates_by_split: dict[str, list[date]] = defaultdict(list)
  split_counts: Counter[str] = Counter()

  for row in split_manifest:
    deck_id = row.get("deck_id")
    if not deck_id:
      issues.append("Split manifest row is missing deck_id.")
    elif deck_id in deck_ids:
      issues.append(f"Split manifest contains duplicate deck_id: {deck_id}")
    else:
      deck_ids.add(deck_id)

    event_id = row.get("event_id")
    if not event_id:
      issues.append(f"Split row for deck {deck_id or '<missing>'} is missing event_id.")

    event_date = row.get("event_date")
    if not event_date:
      issues.append(
        f"Split row for deck {deck_id or '<missing>'} is missing event_date."
      )

    split_name = row.get("split_name")
    if split_name not in VALID_SPLITS:
      issues.append(f"Deck {deck_id or '<missing>'} has invalid split: {split_name!r}")
      continue

    split_counts[split_name] += 1
    if event_id:
      splits_by_event[event_id].add(split_name)
    if event_date:
      dates_by_split[split_name].append(event_date)

  for event_id, split_names in sorted(splits_by_event.items()):
    if len(split_names) > 1:
      joined = ", ".join(sorted(split_names))
      issues.append(f"Event {event_id} appears in multiple splits: {joined}")

  if split_counts["test"] and (
    split_counts["dev-test"] or split_counts["final-test"]
  ):
    issues.append("Split manifest cannot mix test with dev-test/final-test.")

  if require_all_splits:
    required_splits = (
      FINAL_HOLDOUT_REQUIRED_SPLITS
      if split_counts["dev-test"] or split_counts["final-test"]
      else REQUIRED_SPLITS
    )
    missing_splits = sorted(
      split for split in required_splits if split_counts[split] == 0
    )
    if missing_splits:
      joined = ", ".join(missing_splits)
      issues.append(f"Split manifest has no rows for required splits: {joined}")

  _validate_forward_dates(dates_by_split, issues)
  _raise_if_any("Split validation failed", issues)


def validate_model_inputs(
  card_vocab: list[dict[str, Any]],
  deck_tokens: list[dict[str, Any]],
) -> None:
  issues: list[str] = []

  card_idxs = [row.get("card_idx") for row in card_vocab]
  if card_idxs != list(range(len(card_vocab))):
    issues.append("Card vocabulary indexes are not contiguous and zero-based.")

  oracle_ids = [row.get("oracle_id") for row in card_vocab]
  if len(oracle_ids) != len(set(oracle_ids)):
    issues.append("Card vocabulary contains duplicate oracle_id values.")

  card_idx_by_oracle_id = {
    row["oracle_id"]: row["card_idx"]
    for row in card_vocab
    if row.get("oracle_id") and row.get("card_idx") is not None
  }

  for token in deck_tokens:
    deck_id = token.get("deck_id") or "<missing>"
    oracle_id = token.get("oracle_id")
    if oracle_id not in card_idx_by_oracle_id:
      issues.append(f"Deck token for {deck_id} references unknown oracle_id.")
    elif token.get("card_idx") != card_idx_by_oracle_id[oracle_id]:
      issues.append(f"Deck token for {deck_id} has mismatched card_idx.")

    zone = token.get("zone")
    if zone not in ZONE_VOCAB:
      issues.append(f"Deck token for {deck_id} has unknown zone: {zone!r}")
    elif token.get("zone_idx") != ZONE_VOCAB[zone]:
      issues.append(f"Deck token for {deck_id} has mismatched zone_idx.")

    quantity = token.get("quantity")
    if not isinstance(quantity, int) or quantity <= 0:
      issues.append(f"Deck token for {deck_id} has invalid quantity: {quantity!r}")

  _raise_if_any("Model input validation failed", issues)


def _validate_forward_dates(
  dates_by_split: dict[str, list[date]],
  issues: list[str],
) -> None:
  if dates_by_split["dev-test"] or dates_by_split["final-test"]:
    ordered_splits = ("train", "validation", "dev-test", "final-test")
  else:
    ordered_splits = ("train", "validation", "test")

  for earlier, later in zip(
    ordered_splits,
    ordered_splits[1:],
  ):
    if dates_by_split[earlier] and dates_by_split[later]:
      if max(dates_by_split[earlier]) > min(dates_by_split[later]):
        if earlier == "train" and later == "validation":
          issues.append("Train dates overlap validation dates.")
        elif earlier == "validation" and later == "test":
          issues.append("Validation dates overlap test dates.")
        else:
          issues.append(f"{earlier} dates overlap {later} dates.")


def _require(
  row: dict[str, Any],
  key: str,
  context: str,
  issues: list[str],
) -> None:
  if not row.get(key):
    issues.append(f"{context} is missing {key}.")


def _raise_if_any(title: str, issues: list[str]) -> None:
  if not issues:
    return

  sample = "\n".join(f"- {issue}" for issue in issues[:20])
  remaining = len(issues) - 20
  if remaining > 0:
    sample = f"{sample}\n- ... and {remaining} more"
  raise RuntimeError(f"{title}:\n{sample}")

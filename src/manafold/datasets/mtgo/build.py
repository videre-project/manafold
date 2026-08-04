from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from manafold.constants import PROJECT_ROOT
from manafold.datasets.mtgo.database import (
  MTGO_SOURCE,
  MtgoDatasetScope,
  load_mtgo_dataset_records,
)
from manafold.datasets.parquet import write_parquet
from manafold.datasets.schemas import ZONE_VOCAB
from manafold.datasets.validation import (
  validate_deck_cards,
  validate_deck_records,
  validate_model_inputs,
  validate_split_manifest,
)
from manafold.serialization import write_json

SOURCE_ARCHETYPE_PROXY = "source_archetype_name_proxy"
REPORTED_ARCHETYPE_PROXY = "reported_archetype_proxy"
PROXY_TARGET_LEVEL = "family"

_LABEL_SOURCE_TO_TARGET_SOURCE = {
  "source_archetype_name": SOURCE_ARCHETYPE_PROXY,
  "reported_archetype": REPORTED_ARCHETYPE_PROXY,
}

_LABEL_SOURCE_CONFIDENCE = {
  "source_archetype_name": 0.35,
  "reported_archetype": 0.25,
}

_LABEL_SOURCE_PROVENANCE = {
  "reported_archetype": "mtgo-db:archetypes.name",
  "source_archetype_name": "mtgo-db:archetypes.archetype",
}


@dataclass(frozen=True)
class DatasetBuildOptions:
  format_code: str
  start: date
  end: date
  output: Path | None
  dataset_version: str | None
  train_end: date
  validation_end: date
  dev_test_end: date | None
  env_file: Path | None
  limit_events: int | None
  allow_empty: bool = False


@dataclass(frozen=True)
class DatasetBuildSummary:
  dataset_version: str
  output: Path
  deck_count: int
  deck_card_count: int
  card_count: int
  proxy_target_count: int


def build_dataset(options: DatasetBuildOptions) -> DatasetBuildSummary:
  root = PROJECT_ROOT
  _validate_build_options(options)
  format_code = options.format_code.lower()
  dataset_version = options.dataset_version or _dataset_version_for(
    format_code,
    options.start,
    options.end,
  )
  output = options.output or root / "data" / dataset_version
  if not output.is_absolute():
    output = root / output
  output.mkdir(parents=True, exist_ok=True)

  records = load_mtgo_dataset_records(
    scope=MtgoDatasetScope(
      format_code=options.format_code,
      start=options.start,
      end=options.end,
      limit_events=options.limit_events,
    ),
    dataset_version=dataset_version,
    project_root=root,
    env_file=options.env_file,
  )
  deck_records = records.deck_records
  deck_cards = records.deck_cards
  card_catalog = records.card_catalog

  if deck_records or not options.allow_empty:
    validate_deck_records(deck_records)
    validate_deck_cards(deck_records, deck_cards)

  split_manifest = _build_split_manifest(deck_records, options, dataset_version)
  if split_manifest:
    validate_split_manifest(
      split_manifest,
      require_all_splits=options.limit_events is None,
    )

  card_vocab = build_card_vocab(card_catalog, dataset_version)
  deck_tokens = build_deck_tokens(deck_cards, card_vocab)
  proxy_targets = build_proxy_targets(deck_records)
  validate_model_inputs(card_vocab, deck_tokens)

  write_parquet(output / "card_vocab.parquet", card_vocab, "card_vocab")
  write_parquet(output / "deck_tokens.parquet", deck_tokens, "deck_tokens")
  write_parquet(
    output / "proxy_targets.parquet",
    proxy_targets,
    "proxy_targets",
  )
  write_parquet(
    output / "split_manifest.parquet", split_manifest, "split_manifest"
  )
  write_json(output / "zone_vocab.json", ZONE_VOCAB, sort_keys=False)
  write_json(
    output / "dataset_manifest.json",
    _build_dataset_manifest(
      dataset_version=dataset_version,
      format_code=format_code,
      options=options,
      card_vocab=card_vocab,
      deck_tokens=deck_tokens,
      proxy_targets=proxy_targets,
      split_manifest=split_manifest,
    ),
  )

  return DatasetBuildSummary(
    dataset_version=dataset_version,
    output=output,
    deck_count=len(deck_records),
    deck_card_count=len(deck_cards),
    card_count=len(card_vocab),
    proxy_target_count=len(proxy_targets),
  )


def build_card_vocab(
  card_catalog: list[dict[str, Any]],
  dataset_version: str,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  seen_oracle_ids: set[str] = set()

  for card in sorted(card_catalog, key=_card_vocab_sort_key):
    oracle_id = card["oracle_id"]
    if oracle_id in seen_oracle_ids:
      continue

    seen_oracle_ids.add(oracle_id)
    rows.append({
      "dataset_version": dataset_version,
      "card_idx": len(rows),
      "oracle_id": oracle_id,
      "primary_name": card["primary_name"],
      "type_line": card.get("type_line"),
      "first_seen_at": card.get("first_seen_at"),
      "last_seen_at": card.get("last_seen_at"),
    })

  return rows


def build_deck_tokens(
  deck_cards: list[dict[str, Any]],
  card_vocab: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  card_index = {row["oracle_id"]: row["card_idx"] for row in card_vocab}
  rows: list[dict[str, Any]] = []

  for card in deck_cards:
    zone = card["zone"]
    rows.append({
      "dataset_version": card["dataset_version"],
      "deck_id": card["deck_id"],
      "card_idx": card_index[card["oracle_id"]],
      "oracle_id": card["oracle_id"],
      "quantity": card["quantity"],
      "zone_idx": ZONE_VOCAB[zone],
      "zone": zone,
    })

  return sorted(
    rows,
    key=lambda row: (
      row["deck_id"],
      row["zone_idx"],
      row["card_idx"],
      row["oracle_id"],
    ),
  )


def build_proxy_targets(
  deck_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []

  for deck in deck_records:
    label_source, raw_label = _select_proxy_label(deck)
    if label_source is None or raw_label is None:
      continue

    normalized_label = _normalize_label(str(raw_label))
    if not normalized_label:
      continue

    target_source = _LABEL_SOURCE_TO_TARGET_SOURCE[label_source]
    rows.append({
      "dataset_version": deck["dataset_version"],
      "deck_id": deck["deck_id"],
      "target_source": target_source,
      "target_level": PROXY_TARGET_LEVEL,
      "proxy_label_id": _proxy_label_id(target_source, normalized_label),
      "display_label": str(raw_label),
      "normalized_label": normalized_label,
      "source_field": label_source,
      "confidence": _LABEL_SOURCE_CONFIDENCE[label_source],
      "provenance": f"{_LABEL_SOURCE_PROVENANCE[label_source]}|proxy_target",
    })

  return sorted(
    rows,
    key=lambda row: (
      row["deck_id"],
      row["target_source"],
      row["proxy_label_id"],
    ),
  )


def _build_split_manifest(
  deck_records: list[dict[str, Any]],
  options: DatasetBuildOptions,
  dataset_version: str,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  strategy = (
    f"event_forward_final_holdout_{dataset_version}"
    if options.dev_test_end is not None
    else f"event_forward_{dataset_version}"
  )

  for deck in deck_records:
    event_date = deck["event_date"]
    if event_date <= options.train_end:
      split_name = "train"
    elif event_date <= options.validation_end:
      split_name = "validation"
    elif options.dev_test_end is not None and event_date <= options.dev_test_end:
      split_name = "dev-test"
    elif options.dev_test_end is not None:
      split_name = "final-test"
    else:
      split_name = "test"

    rows.append({
      "dataset_version": dataset_version,
      "deck_id": deck["deck_id"],
      "event_id": deck["event_id"],
      "event_date": event_date,
      "format": deck["format"],
      "source": deck["source"],
      "split_name": split_name,
      "split_strategy": strategy,
    })

  return sorted(
    rows,
    key=lambda row: (row["event_date"], row["event_id"], row["deck_id"]),
  )


def _build_dataset_manifest(
  *,
  dataset_version: str,
  format_code: str,
  options: DatasetBuildOptions,
  card_vocab: list[dict[str, Any]],
  deck_tokens: list[dict[str, Any]],
  proxy_targets: list[dict[str, Any]],
  split_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
  return {
    "dataset_version": dataset_version,
    "format": format_code,
    "source": MTGO_SOURCE,
    "start": options.start.isoformat(),
    "end": options.end.isoformat(),
    "train_end": options.train_end.isoformat(),
    "validation_end": options.validation_end.isoformat(),
    "dev_test_end": (
      options.dev_test_end.isoformat()
      if options.dev_test_end is not None
      else None
    ),
    "split_policy": (
      "event_forward_final_holdout"
      if options.dev_test_end is not None
      else "event_forward"
    ),
    "limit_events": options.limit_events,
    "empty": not split_manifest,
    "artifacts": {
      "card_vocab": "card_vocab.parquet",
      "deck_tokens": "deck_tokens.parquet",
      "proxy_targets": "proxy_targets.parquet",
      "split_manifest": "split_manifest.parquet",
      "zone_vocab": "zone_vocab.json",
    },
    "row_counts": {
      "card_vocab": len(card_vocab),
      "deck_tokens": len(deck_tokens),
      "proxy_targets": len(proxy_targets),
      "split_manifest": len(split_manifest),
    },
  }


def _validate_build_options(options: DatasetBuildOptions) -> None:
  if options.start > options.end:
    raise ValueError("--start must be on or before --end.")
  if options.train_end < options.start:
    raise ValueError("--train-end must be on or after --start.")
  if options.train_end >= options.validation_end:
    raise ValueError("--train-end must be before --validation-end.")
  if options.validation_end >= options.end:
    raise ValueError("--validation-end must be before --end.")
  if options.dev_test_end is None:
    return
  if options.dev_test_end <= options.validation_end:
    raise ValueError("--dev-test-end must be after --validation-end.")
  if options.dev_test_end >= options.end:
    raise ValueError("--dev-test-end must be before --end.")


def _select_proxy_label(deck: dict[str, Any]) -> tuple[str | None, Any | None]:
  for label_source in ("source_archetype_name", "reported_archetype"):
    raw_label = deck.get(label_source)
    if raw_label:
      return label_source, raw_label

  return None, None


def _proxy_label_id(target_source: str, normalized_label: str) -> str:
  slug = re.sub(r"[^a-z0-9]+", "_", normalized_label).strip("_")
  if not slug:
    slug = "unknown"
  return f"proxy.{target_source}.{slug}"


def _normalize_label(value: str) -> str:
  return " ".join(value.casefold().split())


def _card_vocab_sort_key(card: dict[str, Any]) -> tuple[str, str]:
  return (
    str(card.get("primary_name") or "").lower(),
    str(card["oracle_id"]),
  )


def _dataset_version_for(format_code: str, start: date, end: date) -> str:
  return f"{format_code}_{start.year}_{end.year}_v0"

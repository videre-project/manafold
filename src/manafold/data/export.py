from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from manafold.config import DatabaseSettings, repo_root
from manafold.data.validate import (
  validate_deck_cards,
  validate_deck_examples,
  validate_model_inputs,
  validate_split_manifest,
  write_json,
  write_parquet,
)
from manafold.db import connect

SOURCE = "mtgo-db"
SOURCE_ARCHETYPE_PROXY = "source_archetype_name_proxy"
REPORTED_ARCHETYPE_PROXY = "reported_archetype_proxy"
PROXY_TARGET_LEVEL = "family"

ZONE_VOCAB = {
  "main": 0,
  "side": 1,
  "companion": 2,
  "commander": 3,
  "other": 4,
}

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
class DatasetExportOptions:
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


@dataclass(frozen=True)
class DatasetExportSummary:
  dataset_version: str
  output: Path
  deck_count: int
  deck_card_count: int
  card_count: int
  proxy_target_count: int


def export_dataset(options: DatasetExportOptions) -> DatasetExportSummary:
  root = repo_root()
  _validate_export_options(options)
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

  settings = DatabaseSettings.from_environment(root, options.env_file)
  with connect(settings) as connection:
    deck_examples = _fetch_deck_examples(connection, options, dataset_version)
    deck_cards = _fetch_deck_cards(connection, options, dataset_version)
    card_catalog = _fetch_card_catalog(connection, options)

  validate_deck_examples(deck_examples)
  validate_deck_cards(deck_examples, deck_cards)

  split_manifest = _build_split_manifest(deck_examples, options, dataset_version)
  validate_split_manifest(
    split_manifest,
    require_all_splits=options.limit_events is None,
  )

  card_vocab = build_card_vocab(card_catalog, dataset_version)
  deck_tokens = build_deck_tokens(deck_cards, card_vocab)
  proxy_targets = build_proxy_targets(deck_examples)
  validate_model_inputs(card_vocab, deck_tokens)

  write_parquet(output / "card_vocab.parquet", card_vocab, "card_vocab")
  write_parquet(output / "deck_tokens.parquet", deck_tokens, "deck_tokens")
  write_parquet(
    output / "proxy_targets.parquet",
    proxy_targets,
    "proxy_targets",
  )
  write_parquet(output / "split_manifest.parquet", split_manifest, "split_manifest")
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

  return DatasetExportSummary(
    dataset_version=dataset_version,
    output=output,
    deck_count=len(deck_examples),
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
    rows.append(
      {
        "dataset_version": dataset_version,
        "card_idx": len(rows),
        "oracle_id": oracle_id,
        "primary_name": card["primary_name"],
        "first_seen_at": card.get("first_seen_at"),
        "last_seen_at": card.get("last_seen_at"),
      }
    )

  return rows


def build_deck_tokens(
  deck_cards: list[dict[str, Any]],
  card_vocab: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  card_index = {row["oracle_id"]: row["card_idx"] for row in card_vocab}
  rows: list[dict[str, Any]] = []

  for card in deck_cards:
    zone = card["zone"]
    rows.append(
      {
        "dataset_version": card["dataset_version"],
        "deck_id": card["deck_id"],
        "card_idx": card_index[card["oracle_id"]],
        "oracle_id": card["oracle_id"],
        "quantity": card["quantity"],
        "zone_idx": ZONE_VOCAB[zone],
        "zone": zone,
      }
    )

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
  deck_examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []

  for deck in deck_examples:
    label_source, raw_label = _select_proxy_label(deck)
    if label_source is None or raw_label is None:
      continue

    normalized_label = _normalize_label(str(raw_label))
    if not normalized_label:
      continue

    target_source = _LABEL_SOURCE_TO_TARGET_SOURCE[label_source]
    rows.append(
      {
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
      }
    )

  return sorted(
    rows,
    key=lambda row: (
      row["deck_id"],
      row["target_source"],
      row["proxy_label_id"],
    ),
  )


def _fetch_deck_examples(
  connection: Any,
  options: DatasetExportOptions,
  dataset_version: str,
) -> list[dict[str, Any]]:
  query = f"""
    WITH scoped_events AS (
      SELECT e.id, e.name, e.date, lower(e.format::text) AS format
      FROM events e
      WHERE lower(e.format::text) = %(format)s
       AND e.date >= %(start)s
       AND e.date <= %(end)s
      ORDER BY e.date, e.id
      {_limit_clause(options.limit_events)}
    )
    SELECT
      %(dataset_version)s AS dataset_version,
      d.id::text AS deck_id,
      se.format AS format,
      se.id::text AS event_id,
      se.date AS event_date,
      %(source)s AS source,
      d.player::text AS player_id,
      s.rank AS standing_rank,
      a.name AS reported_archetype,
      a.archetype AS source_archetype_name,
      a.id::text AS mtgo_db_archetype_id,
      a.archetype_id::text AS source_archetype_id
    FROM decks d
    JOIN scoped_events se ON se.id = d.event_id
    LEFT JOIN standings s
     ON s.event_id = d.event_id
    AND s.player = d.player
    LEFT JOIN archetypes a
     ON a.deck_id = d.id
    WHERE coalesce(cardinality(d.mainboard), 0) > 0
      AND {_fully_resolved_deck_predicate("d")}
    ORDER BY se.date, se.id, d.id
  """
  return _query(connection, query, options, dataset_version)


def _fetch_deck_cards(
  connection: Any,
  options: DatasetExportOptions,
  dataset_version: str,
) -> list[dict[str, Any]]:
  query = f"""
    WITH scoped_events AS (
      SELECT e.id, e.date
      FROM events e
      WHERE lower(e.format::text) = %(format)s
       AND e.date >= %(start)s
       AND e.date <= %(end)s
      ORDER BY e.date, e.id
      {_limit_clause(options.limit_events)}
    ),
    scoped_decks AS (
      SELECT d.id, d.mainboard, d.sideboard
      FROM decks d
      JOIN scoped_events se ON se.id = d.event_id
      WHERE coalesce(cardinality(d.mainboard), 0) > 0
        AND {_fully_resolved_deck_predicate("d")}
    ),
    deck_entries AS (
      SELECT sd.id, card_entries.zone, card_entries.entry
      FROM scoped_decks sd
      CROSS JOIN LATERAL (
        SELECT 'main'::text AS zone, unnest(sd.mainboard) AS entry
        UNION ALL
        SELECT 'side'::text AS zone, unnest(sd.sideboard) AS entry
      ) card_entries
    )
    SELECT
      %(dataset_version)s AS dataset_version,
      de.id::text AS deck_id,
      resolved_card.id AS resolved_card_id,
      resolved_card.oracle_id::text AS oracle_id,
      (de.entry).id AS source_card_id,
      (de.entry).name AS name,
      (de.entry).quantity AS quantity,
      de.zone AS zone
    FROM deck_entries de
    LEFT JOIN cards direct_card
     ON direct_card.id = (de.entry).id
    LEFT JOIN card_catalog_variants variant
     ON variant.catalog_id = (de.entry).id
    LEFT JOIN cards resolved_card
     ON resolved_card.id = coalesce(direct_card.id, variant.card_id)
    ORDER BY de.id, de.zone, name, source_card_id
  """
  return _query(connection, query, options, dataset_version)


def _fetch_card_catalog(
  connection: Any,
  options: DatasetExportOptions,
) -> list[dict[str, Any]]:
  query = f"""
    WITH scoped_events AS (
      SELECT e.id, e.date
      FROM events e
      WHERE lower(e.format::text) = %(format)s
       AND e.date >= %(start)s
       AND e.date <= %(end)s
      ORDER BY e.date, e.id
      {_limit_clause(options.limit_events)}
    ),
    scoped_decks AS (
      SELECT d.id, d.mainboard, d.sideboard
      FROM decks d
      JOIN scoped_events se ON se.id = d.event_id
      WHERE coalesce(cardinality(d.mainboard), 0) > 0
        AND {_fully_resolved_deck_predicate("d")}
    ),
    deck_entries AS (
      SELECT unnest(sd.mainboard) AS entry FROM scoped_decks sd
      UNION ALL
      SELECT unnest(sd.sideboard) AS entry FROM scoped_decks sd
    )
    SELECT DISTINCT
      oc.id::text AS oracle_id,
      oc.name AS primary_name,
      oc.first_seen_at AS first_seen_at,
      oc.last_seen_at AS last_seen_at
    FROM deck_entries de
    LEFT JOIN cards direct_card
     ON direct_card.id = (de.entry).id
    LEFT JOIN card_catalog_variants variant
     ON variant.catalog_id = (de.entry).id
    JOIN cards resolved_card
     ON resolved_card.id = coalesce(direct_card.id, variant.card_id)
    JOIN oracle_cards oc
     ON oc.id = resolved_card.oracle_id
    ORDER BY primary_name, oracle_id
  """
  return _query(connection, query, options, dataset_version=None)


def _query(
  connection: Any,
  query: str,
  options: DatasetExportOptions,
  dataset_version: str | None,
) -> list[dict[str, Any]]:
  parameters = {
    "format": options.format_code.lower(),
    "start": options.start,
    "end": options.end,
    "source": SOURCE,
    "dataset_version": dataset_version,
  }
  with connection.cursor() as cursor:
    cursor.execute(query, parameters)
    return [dict(row) for row in cursor.fetchall()]


def _fully_resolved_deck_predicate(deck_alias: str) -> str:
  return f"""
      NOT EXISTS (
        SELECT 1
        FROM (
          SELECT unnest({deck_alias}.mainboard) AS entry
          UNION ALL
          SELECT unnest({deck_alias}.sideboard) AS entry
        ) identity_check_entries
        LEFT JOIN cards identity_check_direct_card
         ON identity_check_direct_card.id = (identity_check_entries.entry).id
        LEFT JOIN card_catalog_variants identity_check_variant
         ON identity_check_variant.catalog_id = (identity_check_entries.entry).id
        LEFT JOIN cards identity_check_resolved_card
         ON identity_check_resolved_card.id = coalesce(
           identity_check_direct_card.id,
           identity_check_variant.card_id
         )
        WHERE identity_check_resolved_card.oracle_id IS NULL
      )
  """


def _build_split_manifest(
  deck_examples: list[dict[str, Any]],
  options: DatasetExportOptions,
  dataset_version: str,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  strategy = (
    f"event_forward_final_holdout_{dataset_version}"
    if options.dev_test_end is not None
    else f"event_forward_{dataset_version}"
  )

  for deck in deck_examples:
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

    rows.append(
      {
        "dataset_version": dataset_version,
        "deck_id": deck["deck_id"],
        "event_id": deck["event_id"],
        "event_date": event_date,
        "format": deck["format"],
        "source": deck["source"],
        "split_name": split_name,
        "split_strategy": strategy,
      }
    )

  return sorted(
    rows,
    key=lambda row: (row["event_date"], row["event_id"], row["deck_id"]),
  )


def _build_dataset_manifest(
  *,
  dataset_version: str,
  format_code: str,
  options: DatasetExportOptions,
  card_vocab: list[dict[str, Any]],
  deck_tokens: list[dict[str, Any]],
  proxy_targets: list[dict[str, Any]],
  split_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
  return {
    "dataset_version": dataset_version,
    "format": format_code,
    "source": SOURCE,
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


def _validate_export_options(options: DatasetExportOptions) -> None:
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


def _limit_clause(limit_events: int | None) -> str:
  if limit_events is None:
    return ""
  if limit_events <= 0:
    raise ValueError("--limit-events must be positive.")
  return f"LIMIT {limit_events}"

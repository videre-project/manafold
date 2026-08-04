from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row


MTGO_SOURCE = "mtgo-db"


@dataclass(frozen=True)
class MtgoDatasetScope:
  format_code: str
  start: date
  end: date
  limit_events: int | None = None


@dataclass(frozen=True)
class MtgoDatasetRecords:
  deck_records: list[dict[str, Any]]
  deck_cards: list[dict[str, Any]]
  card_catalog: list[dict[str, Any]]


def load_mtgo_dataset_records(
  *,
  scope: MtgoDatasetScope,
  dataset_version: str,
  project_root: Path,
  env_file: Path | None,
) -> MtgoDatasetRecords:
  settings = _DatabaseSettings.from_environment(project_root, env_file)
  with _connect(settings) as connection:
    return MtgoDatasetRecords(
      deck_records=_fetch_decks(connection, scope, dataset_version),
      deck_cards=_fetch_deck_cards(connection, scope, dataset_version),
      card_catalog=_fetch_card_catalog(connection, scope),
    )


@dataclass(frozen=True)
class _DatabaseSettings:
  dsn: str | None
  host: str | None
  port: str | None
  database: str | None
  user: str | None
  password: str | None

  @classmethod
  def from_environment(
    cls,
    project_root: Path,
    env_file: Path | None = None,
  ) -> _DatabaseSettings:
    values: dict[str, str] = {}
    for candidate in (
      project_root.parent / "mtgo-db" / ".env",
      project_root.parent / "MTGOBot" / ".env",
      project_root / ".env",
      env_file,
    ):
      if candidate is not None:
        values.update(_read_env_file(candidate))
    values.update({key: value for key, value in os.environ.items() if value})

    dsn = values.get("MANAFOLD_DATABASE_URL") or values.get("DATABASE_URL")
    host = (
      values.get("PGHOST")
      or values.get("POSTGRES_HOST")
      or values.get("TAILSCALE_IP")
    )
    if values.get("PGPORT"):
      port = values["PGPORT"]
    elif host and host == values.get("TAILSCALE_IP"):
      port = values.get("TAILSCALE_POSTGRES_PORT", "5433")
    else:
      port = values.get("POSTGRES_PORT", "5432")

    return cls(
      dsn=dsn,
      host=host,
      port=port,
      database=values.get("PGDATABASE") or values.get("POSTGRES_DB"),
      user=values.get("PGUSER") or values.get("POSTGRES_USER"),
      password=values.get("PGPASSWORD") or values.get("POSTGRES_PASSWORD"),
    )

  def connect_kwargs(self) -> dict[str, str]:
    if self.dsn:
      return {"conninfo": self.dsn}
    values = {
      "host": self.host,
      "port": self.port,
      "dbname": self.database,
      "user": self.user,
      "password": self.password,
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
      raise RuntimeError(
        f"Missing database connection settings: {', '.join(missing)}"
      )
    return {name: value or "" for name, value in values.items()}


@contextmanager
def _connect(settings: _DatabaseSettings) -> Iterator[psycopg.Connection]:
  connection = psycopg.connect(
    **settings.connect_kwargs(), row_factory=dict_row
  )
  try:
    yield connection
  finally:
    connection.close()


def _fetch_decks(
  connection: psycopg.Connection,
  scope: MtgoDatasetScope,
  dataset_version: str,
) -> list[dict[str, Any]]:
  query = f"""
    WITH scoped_events AS (
      SELECT e.id, e.name, e.date, lower(e.format::text) AS format
      FROM events e
      WHERE lower(e.format::text) = %(format)s
        AND e.date >= %(start)s
        AND e.date <= %(end)s
        AND EXISTS (
          SELECT 1 FROM decks d
          WHERE d.event_id = e.id
            AND coalesce(cardinality(d.mainboard), 0) > 0
            AND {_fully_resolved_deck_predicate("d")}
        )
      ORDER BY e.date, e.id
      {_limit_clause(scope.limit_events)}
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
  return _query(connection, query, scope, dataset_version)


def _fetch_deck_cards(
  connection: psycopg.Connection,
  scope: MtgoDatasetScope,
  dataset_version: str,
) -> list[dict[str, Any]]:
  query = f"""
    WITH scoped_events AS (
      SELECT e.id, e.date
      FROM events e
      WHERE lower(e.format::text) = %(format)s
        AND e.date >= %(start)s
        AND e.date <= %(end)s
        AND EXISTS (
          SELECT 1 FROM decks d
          WHERE d.event_id = e.id
            AND coalesce(cardinality(d.mainboard), 0) > 0
            AND {_fully_resolved_deck_predicate("d")}
        )
      ORDER BY e.date, e.id
      {_limit_clause(scope.limit_events)}
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
  return _query(connection, query, scope, dataset_version)


def _fetch_card_catalog(
  connection: psycopg.Connection,
  scope: MtgoDatasetScope,
) -> list[dict[str, Any]]:
  query = f"""
    WITH scoped_events AS (
      SELECT e.id, e.date
      FROM events e
      WHERE lower(e.format::text) = %(format)s
        AND e.date >= %(start)s
        AND e.date <= %(end)s
        AND EXISTS (
          SELECT 1 FROM decks d
          WHERE d.event_id = e.id
            AND coalesce(cardinality(d.mainboard), 0) > 0
            AND {_fully_resolved_deck_predicate("d")}
        )
      ORDER BY e.date, e.id
      {_limit_clause(scope.limit_events)}
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
      oc.type_line,
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
  return _query(connection, query, scope, dataset_version=None)


def _query(
  connection: psycopg.Connection,
  query: str,
  scope: MtgoDatasetScope,
  dataset_version: str | None,
) -> list[dict[str, Any]]:
  parameters = {
    "format": scope.format_code.lower(),
    "start": scope.start,
    "end": scope.end,
    "source": MTGO_SOURCE,
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


def _limit_clause(limit_events: int | None) -> str:
  if limit_events is None:
    return ""
  if limit_events <= 0:
    raise ValueError("--limit-events must be positive.")
  return f"LIMIT {limit_events}"


def _read_env_file(path: Path) -> dict[str, str]:
  if not path.exists():
    return {}
  values: dict[str, str] = {}
  for line in path.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
      continue
    key, value = stripped.split("=", 1)
    values[key.strip()] = value.strip().strip("'").strip('"')
  return values

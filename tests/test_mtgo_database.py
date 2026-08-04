from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from manafold.datasets.mtgo.database import (
  MtgoDatasetScope,
  _DatabaseSettings,
  _fetch_card_catalog,
  _fetch_deck_cards,
  _fetch_decks,
)


class MtgoDatabaseTests(unittest.TestCase):
  def test_explicit_environment_file_provides_connection_settings(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      env_file = root / "database.env"
      env_file.write_text(
        "PGHOST=db.internal\n"
        "PGPORT=5544\n"
        "PGDATABASE=manafold\n"
        "PGUSER=reader\n"
        "PGPASSWORD=secret\n",
        encoding="utf-8",
      )

      with patch.dict(os.environ, {}, clear=True):
        settings = _DatabaseSettings.from_environment(root, env_file)

      self.assertEqual(
        {
          "host": "db.internal",
          "port": "5544",
          "dbname": "manafold",
          "user": "reader",
          "password": "secret",
        },
        settings.connect_kwargs(),
      )

  def test_dataset_queries_bind_scope_without_live_row_counts(self) -> None:
    scope = MtgoDatasetScope(
      format_code="Modern",
      start=date(2024, 1, 1),
      end=date(2024, 12, 31),
      limit_events=25,
    )
    connection = _RecordingConnection([{"record_id": "one"}])

    self.assertEqual(
      [{"record_id": "one"}],
      _fetch_decks(connection, scope, "modern_contract_v0"),
    )
    self.assertEqual(
      [{"record_id": "one"}],
      _fetch_deck_cards(connection, scope, "modern_contract_v0"),
    )
    self.assertEqual(
      [{"record_id": "one"}],
      _fetch_card_catalog(connection, scope),
    )

    self.assertEqual(3, len(connection.executions))
    for _, parameters in connection.executions:
      self.assertEqual("modern", parameters["format"])
      self.assertEqual(date(2024, 1, 1), parameters["start"])
      self.assertEqual(date(2024, 12, 31), parameters["end"])
      self.assertEqual("mtgo-db", parameters["source"])
    self.assertEqual(
      ["modern_contract_v0", "modern_contract_v0", None],
      [parameters["dataset_version"] for _, parameters in connection.executions],
    )
    self.assertTrue(all("LIMIT 25" in query for query, _ in connection.executions))
    self.assertIn("FROM decks", connection.executions[0][0])
    self.assertIn("oracle_cards", connection.executions[2][0])


class _RecordingCursor:
  def __init__(
    self,
    rows: list[dict[str, object]],
    executions: list[tuple[str, dict[str, object]]],
  ) -> None:
    self._rows = rows
    self._executions = executions

  def __enter__(self) -> _RecordingCursor:
    return self

  def __exit__(self, *args: object) -> None:
    return None

  def execute(self, query: str, parameters: dict[str, object]) -> None:
    self._executions.append((query, parameters))

  def fetchall(self) -> list[dict[str, object]]:
    return self._rows


class _RecordingConnection:
  def __init__(self, rows: list[dict[str, object]]) -> None:
    self._rows = rows
    self.executions: list[tuple[str, dict[str, object]]] = []

  def cursor(self) -> _RecordingCursor:
    return _RecordingCursor(self._rows, self.executions)


if __name__ == "__main__":
  unittest.main()

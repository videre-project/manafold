from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from manafold.config import DatabaseSettings


@contextmanager
def connect(settings: DatabaseSettings) -> Iterator[psycopg.Connection]:
  connection = psycopg.connect(**settings.connect_kwargs(), row_factory=dict_row)
  try:
    yield connection
  finally:
    connection.close()

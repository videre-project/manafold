from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from manafold.datasets.schemas import parquet_schema


def write_parquet(
  path: Path,
  rows: list[dict[str, Any]],
  file_stem: str,
) -> None:
  table = pa.Table.from_pylist(rows, schema=parquet_schema(file_stem))
  pq.write_table(table, path)

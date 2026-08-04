from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable


def write_json(
  path: Path,
  data: dict[str, Any],
  *,
  sort_keys: bool = True,
) -> None:
  path.write_text(
    json.dumps(data, default=_json_default, indent=2, sort_keys=sort_keys) + "\n",
    encoding="utf-8",
  )


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as input_file:
    for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as input_file:
    for line in input_file:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as output_file:
    for row in rows:
      output_file.write(json.dumps(row, sort_keys=True) + "\n")


def _json_default(value: Any) -> str:
  if isinstance(value, date):
    return value.isoformat()
  return str(value)

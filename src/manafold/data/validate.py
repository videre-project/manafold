from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ZONE_VOCAB = {
  "main": 0,
  "side": 1,
  "companion": 2,
  "commander": 3,
  "other": 4,
}

CONSTRUCTED_FORMATS = {
  "alchemy",
  "brawl",
  "explorer",
  "historic",
  "legacy",
  "modern",
  "pauper",
  "pioneer",
  "premodern",
  "standard",
  "vintage",
}

REQUIRED_SPLITS = {"train", "validation", "test"}
FINAL_HOLDOUT_REQUIRED_SPLITS = {"train", "validation", "dev-test", "final-test"}
VALID_SPLITS = REQUIRED_SPLITS | FINAL_HOLDOUT_REQUIRED_SPLITS | {"novelty_holdout"}
VALID_ZONES = set(ZONE_VOCAB)
REQUIRED_DATASET_FILES = {
  "card_vocab",
  "deck_tokens",
  "proxy_targets",
  "split_manifest",
  "zone_vocab",
}

PROXY_TARGETS_SCHEMA = pa.schema(
  [
    pa.field("dataset_version", pa.string(), nullable=False),
    pa.field("deck_id", pa.string(), nullable=False),
    pa.field("target_source", pa.string(), nullable=False),
    pa.field("target_level", pa.string(), nullable=False),
    pa.field("proxy_label_id", pa.string(), nullable=False),
    pa.field("display_label", pa.string(), nullable=False),
    pa.field("normalized_label", pa.string(), nullable=False),
    pa.field("source_field", pa.string(), nullable=False),
    pa.field("confidence", pa.float64(), nullable=False),
    pa.field("provenance", pa.string(), nullable=False),
  ]
)

SPLIT_MANIFEST_SCHEMA = pa.schema(
  [
    pa.field("dataset_version", pa.string(), nullable=False),
    pa.field("deck_id", pa.string(), nullable=False),
    pa.field("event_id", pa.string(), nullable=False),
    pa.field("event_date", pa.date32(), nullable=False),
    pa.field("format", pa.string(), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("split_name", pa.string(), nullable=False),
    pa.field("split_strategy", pa.string(), nullable=False),
  ]
)

CARD_VOCAB_SCHEMA = pa.schema(
  [
    pa.field("dataset_version", pa.string(), nullable=False),
    pa.field("card_idx", pa.int64(), nullable=False),
    pa.field("oracle_id", pa.string(), nullable=False),
    pa.field("primary_name", pa.string(), nullable=False),
    pa.field("first_seen_at", pa.timestamp("us", tz="Etc/UTC")),
    pa.field("last_seen_at", pa.timestamp("us", tz="Etc/UTC")),
  ]
)

DECK_TOKENS_SCHEMA = pa.schema(
  [
    pa.field("dataset_version", pa.string(), nullable=False),
    pa.field("deck_id", pa.string(), nullable=False),
    pa.field("card_idx", pa.int64(), nullable=False),
    pa.field("oracle_id", pa.string(), nullable=False),
    pa.field("quantity", pa.int64(), nullable=False),
    pa.field("zone_idx", pa.int64(), nullable=False),
    pa.field("zone", pa.string(), nullable=False),
  ]
)

PARQUET_SCHEMAS = {
  "card_vocab": CARD_VOCAB_SCHEMA,
  "deck_tokens": DECK_TOKENS_SCHEMA,
  "proxy_targets": PROXY_TARGETS_SCHEMA,
  "split_manifest": SPLIT_MANIFEST_SCHEMA,
}


@dataclass(frozen=True)
class DatasetCheckResult:
  dataset_path: Path
  dataset_version: str
  checked_artifacts: int
  checked_parquet_artifacts: int


def parquet_schema(artifact_name: str) -> pa.Schema:
  try:
    return PARQUET_SCHEMAS[artifact_name]
  except KeyError as error:
    raise ValueError(f"Unknown Parquet artifact schema: {artifact_name}") from error


def write_parquet(
  path: Path,
  rows: list[dict[str, Any]],
  artifact_name: str,
) -> None:
  table = pa.Table.from_pylist(rows, schema=parquet_schema(artifact_name))
  pq.write_table(table, path)


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


def verify_dataset_export(dataset_path: Path) -> DatasetCheckResult:
  manifest = _read_json(dataset_path / "dataset_manifest.json", "dataset manifest")
  artifacts = _mapping(manifest, "artifacts", "dataset manifest")
  row_counts = _mapping(manifest, "row_counts", "dataset manifest")
  issues: list[str] = []

  for artifact_name, relative_path in sorted(artifacts.items()):
    if not (dataset_path / relative_path).exists():
      issues.append(f"Missing artifact {artifact_name}: {relative_path}")

  for artifact_name in sorted(REQUIRED_DATASET_FILES - set(artifacts)):
    issues.append(f"Manifest is missing required dataset file: {artifact_name}")

  parquet_rows: dict[str, list[dict[str, Any]]] = {}
  for artifact_name, relative_path in sorted(artifacts.items()):
    if artifact_name not in PARQUET_SCHEMAS:
      continue

    path = dataset_path / relative_path
    if not path.exists():
      continue

    table = pq.read_table(path)
    if not table.schema.equals(parquet_schema(artifact_name), check_metadata=False):
      issues.append(
        f"{artifact_name} schema does not match declared artifact schema."
      )

    expected_count = row_counts.get(artifact_name)
    if expected_count is not None and table.num_rows != expected_count:
      issues.append(
        f"{artifact_name} row count is {table.num_rows}, "
        f"but manifest records {expected_count}."
      )

    parquet_rows[artifact_name] = table.to_pylist()

  _validate_zone_vocab(dataset_path, artifacts, issues)
  _validate_loaded_artifacts(parquet_rows, manifest, issues)
  _raise_if_any("Dataset verification failed", issues)

  return DatasetCheckResult(
    dataset_path=dataset_path,
    dataset_version=str(manifest["dataset_version"]),
    checked_artifacts=len(artifacts),
    checked_parquet_artifacts=len(parquet_rows),
  )


def validate_deck_examples(deck_examples: list[dict[str, Any]]) -> None:
  issues: list[str] = []
  if not deck_examples:
    issues.append("Deck export produced zero deck examples.")

  deck_ids: set[str] = set()
  for row in deck_examples:
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
  deck_examples: list[dict[str, Any]],
  deck_cards: list[dict[str, Any]],
) -> None:
  issues: list[str] = []
  deck_formats = {
    row["deck_id"]: str(row["format"]).lower()
    for row in deck_examples
    if row.get("deck_id") and row.get("format")
  }
  deck_zone_counts: dict[str, Counter[str]] = defaultdict(Counter)

  if not deck_cards:
    issues.append("Deck-card export produced zero rows.")

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

  for deck_id, format_code in sorted(deck_formats.items()):
    if format_code not in CONSTRUCTED_FORMATS:
      continue

    mainboard_count = deck_zone_counts[deck_id]["main"]
    sideboard_count = deck_zone_counts[deck_id]["side"]
    if mainboard_count < 60:
      issues.append(
        f"Constructed deck {deck_id} has {mainboard_count} mainboard cards."
      )
    if sideboard_count > 15:
      issues.append(
        f"Constructed deck {deck_id} has {sideboard_count} sideboard cards."
      )

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


def _validate_loaded_artifacts(
  rows: dict[str, list[dict[str, Any]]],
  manifest: dict[str, Any],
  issues: list[str],
) -> None:
  required = {"card_vocab", "deck_tokens", "split_manifest"}
  missing = sorted(required - set(rows))
  if missing:
    issues.append(f"Cannot run cross-artifact validation; missing: {', '.join(missing)}")
    return

  try:
    validate_split_manifest(
      rows["split_manifest"],
      require_all_splits=manifest.get("limit_events") is None,
    )
    validate_model_inputs(rows["card_vocab"], rows["deck_tokens"])
  except RuntimeError as error:
    issues.append(str(error))


def _validate_zone_vocab(
  dataset_path: Path,
  artifacts: dict[str, Any],
  issues: list[str],
) -> None:
  relative_path = artifacts.get("zone_vocab")
  if not relative_path:
    return

  try:
    zone_vocab = _read_json(dataset_path / relative_path, "zone vocabulary")
  except (json.JSONDecodeError, RuntimeError) as error:
    issues.append(str(error))
    return

  if {
    str(zone): int(index)
    for zone, index in zone_vocab.items()
    if isinstance(zone, str) and isinstance(index, int)
  } != ZONE_VOCAB:
    issues.append("zone_vocab.json does not match the expected zone vocabulary.")


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


def _read_json(path: Path, artifact_description: str) -> dict[str, Any]:
  if not path.exists():
    raise RuntimeError(f"Missing {artifact_description}: {path}")

  data = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(data, dict):
    raise RuntimeError(f"{path} must contain a JSON object.")
  return data


def _mapping(
  row: dict[str, Any],
  field: str,
  context: str,
) -> dict[str, Any]:
  value = row.get(field)
  if not isinstance(value, dict):
    raise RuntimeError(f"{context} is missing {field}.")
  return value


def _require(
  row: dict[str, Any],
  key: str,
  context: str,
  issues: list[str],
) -> None:
  if not row.get(key):
    issues.append(f"{context} is missing {key}.")


def _json_default(value: Any) -> str:
  if isinstance(value, date):
    return value.isoformat()
  return str(value)


def _raise_if_any(title: str, issues: list[str]) -> None:
  if not issues:
    return

  sample = "\n".join(f"- {issue}" for issue in issues[:20])
  remaining = len(issues) - 20
  if remaining > 0:
    sample = f"{sample}\n- ... and {remaining} more"
  raise RuntimeError(f"{title}:\n{sample}")

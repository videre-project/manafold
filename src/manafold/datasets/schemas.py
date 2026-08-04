from __future__ import annotations

import pyarrow as pa


ZONE_VOCAB = {
  "main": 0,
  "side": 1,
  "companion": 2,
  "commander": 3,
  "other": 4,
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
    pa.field("type_line", pa.string(), nullable=True),
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


def parquet_schema(file_stem: str) -> pa.Schema:
  try:
    return PARQUET_SCHEMAS[file_stem]
  except KeyError as error:
    raise ValueError(f"Unknown Parquet file schema: {file_stem}") from error

from __future__ import annotations

import unittest

import pyarrow as pa

from manafold.datasets.schemas import parquet_schema


class SchemaTests(unittest.TestCase):
  def test_training_dataset_schemas_use_integer_indexes(self) -> None:
    self.assertEqual(pa.int64(), parquet_schema("card_vocab").field("card_idx").type)
    self.assertEqual(pa.int64(), parquet_schema("deck_tokens").field("card_idx").type)
    self.assertEqual(pa.int64(), parquet_schema("deck_tokens").field("zone_idx").type)

  def test_proxy_target_schema_has_proxy_label_identity(self) -> None:
    schema = parquet_schema("proxy_targets")

    self.assertEqual(pa.string(), schema.field("target_source").type)
    self.assertEqual(pa.string(), schema.field("proxy_label_id").type)
    self.assertEqual(pa.string(), schema.field("normalized_label").type)
    self.assertNotIn("node_id", schema.names)


if __name__ == "__main__":
  unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from manafold.datasets.model_inputs import build_training_inputs, load_training_dataset
from tests.model_test_support import _proxy_target, _split, _token, _write_dataset


class ModelDataTests(unittest.TestCase):
  def test_model_examples_group_variable_sized_decks(self) -> None:
    examples = build_training_inputs(
      deck_tokens=[
        _token("deck-1", 1, 4),
        _token("deck-1", 2, 2),
        _token("deck-2", 3, 4),
      ],
      split_manifest=[
        _split("deck-1", "train"),
        _split("deck-2", "validation"),
      ],
      proxy_targets=[
        _proxy_target("deck-1", "alpha"),
        _proxy_target("deck-2", "beta"),
      ],
    )

    self.assertEqual(["deck-1", "deck-2"], [example.deck_id for example in examples])
    self.assertEqual(2, len(examples[0].tokens))
    self.assertEqual(1, len(examples[1].tokens))

  def test_load_training_dataset_reads_training_files(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      dataset = load_training_dataset(dataset_path)

      self.assertEqual("modern_2024_2024_v0", dataset.dataset_version)
      self.assertEqual(3, dataset.card_count)
      self.assertEqual(5, dataset.zone_count)
      self.assertEqual(
        (
          "proxy.source_archetype_name_proxy.alpha",
          "proxy.source_archetype_name_proxy.beta",
        ),
        dataset.labels,
      )
      self.assertEqual(5, len(dataset.examples))



if __name__ == "__main__":
  unittest.main()

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from manafold.datasets.model_inputs import load_training_dataset
from manafold.models.training import training_pipeline
from manafold.taxonomy.auto_ontology import induce_auto_ontology
from tests.model_test_support import (
  _rewrite_dataset_with_local_card_indexes,
  _write_dataset,
)


class AutoOntologyTests(unittest.TestCase):
  def test_a11_rejects_a_second_target_projection(self) -> None:
    with self.assertRaisesRegex(ValueError, "cannot be combined"):
      training_pipeline.train_models(
        Path("unused"),
        model_names=(training_pipeline.MODEL_A11,),
        target_label_level=(
          training_pipeline.TARGET_LABEL_LEVEL_CANONICAL_FAMILY
        ),
      )

  def test_local_card_index_order_does_not_change_provenance(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      original_path = root / "original"
      remapped_path = root / "remapped"
      original_path.mkdir()
      remapped_path.mkdir()
      _write_dataset(original_path)
      _write_dataset(remapped_path)
      _rewrite_dataset_with_local_card_indexes(remapped_path)

      original = induce_auto_ontology(load_training_dataset(original_path))
      remapped = induce_auto_ontology(load_training_dataset(remapped_path))

      self.assertEqual(original, remapped)

  def test_held_out_changes_do_not_change_training_induction(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir) / "dataset"
      dataset_path.mkdir()
      _write_dataset(dataset_path)
      dataset = load_training_dataset(dataset_path)

      changed_examples = tuple(
        replace(example, target_label_id="proxy.changed-held-out-label")
        if example.split_name == "validation"
        else example
        for example in dataset.examples
      )
      changed_dataset = replace(dataset, examples=changed_examples)

      self.assertEqual(
        induce_auto_ontology(dataset),
        induce_auto_ontology(changed_dataset),
      )

  def test_training_change_updates_digest_and_family_relations(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir) / "dataset"
      dataset_path.mkdir()
      _write_dataset(dataset_path)
      dataset = load_training_dataset(dataset_path)
      beta_tokens = next(
        example.tokens
        for example in dataset.examples
        if example.deck_id == "train-b"
      )
      changed_examples = tuple(
        replace(example, tokens=beta_tokens)
        if example.deck_id == "train-a"
        else example
        for example in dataset.examples
      )

      before = induce_auto_ontology(dataset)
      after = induce_auto_ontology(replace(dataset, examples=changed_examples))

      self.assertNotEqual(
        before["dataset_provenance"]["training_split_sha256"],
        after["dataset_provenance"]["training_split_sha256"],
      )
      self.assertEqual([], before["proposed_components"])
      self.assertEqual(1, len(after["proposed_components"]))


if __name__ == "__main__":
  unittest.main()

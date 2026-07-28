from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import torch

from manafold.models.data import build_model_examples, load_training_dataset
from manafold.models.deepsets import (
  POOLING_QUANTITY_WEIGHTED,
  DeepSetsClassifier,
  SetTransformerClassifier,
  hypergeometric_quantity_weight,
  mixed_class_balanced_epoch_indexes,
  partial_observation_training_views,
)
from tests.model_test_support import _proxy_target, _split, _token, _write_dataset


class ModelArchitecturesTests(unittest.TestCase):
  def test_zero_extra_dims_preserve_init_is_functional_noop(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)
      dataset = load_training_dataset(dataset_path)

      base = DeepSetsClassifier(
        labels=dataset.labels,
        card_count=dataset.card_count,
        zone_count=dataset.zone_count,
        quantity_count=5,
        embedding_dim=8,
        hidden_dim=16,
        pooling=POOLING_QUANTITY_WEIGHTED,
        seed=37,
        device="cpu",
      )
      zero_extra = DeepSetsClassifier(
        labels=dataset.labels,
        card_count=dataset.card_count,
        zone_count=dataset.zone_count,
        quantity_count=5,
        embedding_dim=8,
        hidden_dim=16,
        pooling=POOLING_QUANTITY_WEIGHTED,
        seed=37,
        device="cpu",
        extra_feature_dim=4,
        extra_feature_value=0.0,
        preserve_base_rho_init=True,
      )

      self.assertTrue(
        torch.equal(
          base.logits_many(dataset.examples),
          zero_extra.logits_many(dataset.examples),
        )
      )

  def test_set_transformer_artifact_excludes_training_only_modules(self) -> None:
    classifier = SetTransformerClassifier(
      labels=("alpha", "beta"),
      card_count=8,
      zone_count=2,
      quantity_count=5,
      embedding_dim=4,
      hidden_dim=8,
      attention_heads=2,
      attention_layers=1,
      partial_observation_training=True,
      partial_contextual_weight=0.5,
      seed=13,
      device="cpu",
    )

    state = classifier.state_dict_for_artifact()

    self.assertTrue(state)
    self.assertFalse(any("teacher" in key or "predictor" in key for key in state))
    self.assertGreater(
      classifier.training_parameter_count(),
      classifier.inference_parameter_count(),
    )

  def test_hypergeometric_quantity_weight_is_nonlinear(self) -> None:
    weights = [
      hypergeometric_quantity_weight(
        quantity,
        population_size=60,
        draw_count=7,
      )
      for quantity in range(1, 5)
    ]

    self.assertAlmostEqual(1.0, weights[0])
    self.assertEqual(sorted(weights), weights)
    self.assertLess(weights[3], 4.0)
    self.assertLess(weights[3] - weights[2], weights[1] - weights[0])

  def test_partial_views_and_balanced_sampling_are_deterministic(self) -> None:
    deck_labels = [
      *[(f"common-{index}", "common") for index in range(20)],
      ("rare-1", "rare"),
      ("rare-2", "rare"),
    ]
    examples = build_model_examples(
      deck_tokens=[
        _token(deck_id, index + 1, 4)
        for index, (deck_id, _) in enumerate(deck_labels)
      ],
      split_manifest=[
        _split(deck_id, "train")
        for deck_id, _ in deck_labels
      ],
      proxy_targets=[
        _proxy_target(deck_id, label)
        for deck_id, label in deck_labels
      ],
    )

    first = mixed_class_balanced_epoch_indexes(
      examples,
      rng=random.Random(13),
    )
    second = mixed_class_balanced_epoch_indexes(
      examples,
      rng=random.Random(13),
    )
    partial_source = build_model_examples(
      deck_tokens=[
        _token("partial", 1, 4),
        _token("partial", 2, 4),
        _token("partial", 3, 4),
      ],
      split_manifest=[_split("partial", "train")],
      proxy_targets=[_proxy_target("partial", "common")],
    )
    partial = partial_observation_training_views(
      partial_source,
      rng=random.Random(13),
      identity_counts=(1,),
    )

    self.assertEqual(first, second)
    self.assertEqual(len(examples), len(first))
    self.assertFalse(partial[0].observation_complete)
    self.assertEqual(12, partial[0].expected_mainboard_size)



if __name__ == "__main__":
  unittest.main()

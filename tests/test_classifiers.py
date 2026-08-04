from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import torch

from manafold.datasets.model_inputs import build_training_inputs, load_training_dataset
from manafold.models.classifiers import (
  POOLING_QUANTITY_WEIGHTED,
  DeepSetsClassifier,
  SetTransformerClassifier,
  hypergeometric_quantity_weight,
  mixed_class_balanced_epoch_indexes,
  partial_observation_training_views,
)
from tests.model_test_support import _proxy_target, _split, _token, _write_dataset


class ClassifierTests(unittest.TestCase):
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

  def test_set_transformer_saved_state_excludes_training_only_modules(self) -> None:
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

    state = classifier.state_dict_for_saving()

    self.assertTrue(state)
    self.assertFalse(any("teacher" in key or "predictor" in key for key in state))
    self.assertGreater(
      classifier.training_parameter_count(),
      classifier.inference_parameter_count(),
    )

  def test_a8_objectives_are_active_and_exportable(self) -> None:
    alpha_label = "proxy.source_archetype_name_proxy.alpha"
    beta_label = "proxy.source_archetype_name_proxy.beta"
    examples = build_training_inputs(
      deck_tokens=[
        _token("alpha-a", 0, 4),
        _token("alpha-a", 1, 2),
        _token("alpha-b", 0, 4),
        _token("alpha-b", 1, 2),
        _token("beta-a", 1, 4),
        _token("beta-a", 2, 2),
        _token("beta-b", 1, 4),
        _token("beta-b", 2, 2),
      ],
      split_manifest=[
        _split(deck_id, "train")
        for deck_id in ("alpha-a", "alpha-b", "beta-a", "beta-b")
      ],
      proxy_targets=[
        _proxy_target("alpha-a", "alpha"),
        _proxy_target("alpha-b", "alpha"),
        _proxy_target("beta-a", "beta"),
        _proxy_target("beta-b", "beta"),
      ],
    )
    classifier = SetTransformerClassifier(
      labels=(alpha_label, beta_label),
      card_count=4,
      zone_count=2,
      quantity_count=5,
      embedding_dim=4,
      hidden_dim=8,
      attention_heads=2,
      attention_layers=1,
      partial_observation_training=True,
      partial_contextual_weight=0.5,
      supcon_loss_weight=0.05,
      anchor_gating=True,
      use_product_manifold=True,
      product_manifold_tree_weight=0.10,
      causal_target_alpha=0.10,
      family_by_label={alpha_label: "shared", beta_label: "shared"},
      seed=13,
      device="cpu",
    )

    summary = classifier.fit(examples, epochs=1, batch_size=4, shuffle=False)
    gate_scale = classifier._network.anchor_gate.gate_scale.detach().clone()
    with torch.no_grad():
      classifier._network.anchor_gate.gate_scale.zero_()
    unbiased_logits = classifier.logits_many(examples)
    with torch.no_grad():
      classifier._network.anchor_gate.gate_scale.copy_(gate_scale)
    biased_logits = classifier.logits_many(examples)
    state = classifier.state_dict_for_saving()

    self.assertEqual(8, classifier._network.output_dim)
    self.assertGreater(
      float(classifier._network.anchor_gate.anchor_scores.max().item()),
      0.0,
    )
    self.assertIsNotNone(classifier._causal_soft_target_matrix)
    self.assertGreater(classifier._causal_soft_target_matrix[0, 1].item(), 0.0)
    self.assertFalse(torch.equal(unbiased_logits, biased_logits))
    self.assertEqual(
      classifier._family_index_by_label_index[0].item(),
      classifier._family_index_by_label_index[1].item(),
    )
    self.assertIn("anchor_gate.anchor_scores", state)
    self.assertIn("product_manifold.proj_euc.weight", state)
    self.assertIsNotNone(summary["history"][0]["train_supcon_loss"])
    self.assertIsNotNone(summary["history"][0]["train_tree_loss"])

  def test_a9_objectives_are_active_and_exportable(self) -> None:
    alpha_label = "proxy.source_archetype_name_proxy.alpha"
    beta_label = "proxy.source_archetype_name_proxy.beta"
    examples = build_training_inputs(
      deck_tokens=[
        _token("alpha-a", 0, 4),
        _token("alpha-a", 1, 2),
        _token("alpha-b", 0, 4),
        _token("alpha-b", 1, 2),
        _token("beta-a", 1, 4),
        _token("beta-a", 2, 2),
        _token("beta-b", 1, 4),
        _token("beta-b", 2, 2),
      ],
      split_manifest=[
        _split(deck_id, "train")
        for deck_id in ("alpha-a", "alpha-b", "beta-a", "beta-b")
      ],
      proxy_targets=[
        _proxy_target("alpha-a", "alpha"),
        _proxy_target("alpha-b", "alpha"),
        _proxy_target("beta-a", "beta"),
        _proxy_target("beta-b", "beta"),
      ],
    )
    classifier = SetTransformerClassifier(
      labels=(alpha_label, beta_label),
      card_count=4,
      zone_count=2,
      quantity_count=5,
      embedding_dim=4,
      hidden_dim=8,
      attention_heads=2,
      attention_layers=1,
      partial_observation_training=True,
      partial_contextual_weight=0.5,
      supcon_loss_weight=0.05,
      anchor_gating=True,
      use_product_manifold=True,
      product_manifold_tree_weight=0.10,
      causal_target_alpha=0.10,
      adaptive_manifold_scaling=True,
      adaptive_manifold_ramp_cards=20.0,
      residual_anchor_logits=True,
      residual_anchor_gamma=0.25,
      family_by_label={alpha_label: "shared", beta_label: "shared"},
      seed=13,
      device="cpu",
    )

    summary = classifier.fit(examples, epochs=1, batch_size=4, shuffle=False)
    state = classifier.state_dict_for_saving()
    logits = classifier.logits_many(examples)

    self.assertIsNotNone(classifier._residual_anchor_pmi_matrix)
    self.assertEqual(classifier._residual_anchor_pmi_matrix.shape, (4, 2))
    self.assertIn("_residual_anchor_pmi_matrix", state)
    self.assertEqual(logits.shape, (4, 2))
    self.assertTrue(classifier.adaptive_manifold_scaling)
    self.assertTrue(classifier.residual_anchor_logits)
    self.assertIsNotNone(summary["history"][0]["train_tree_loss"])

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
    examples = build_training_inputs(
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
    partial_source = build_training_inputs(
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

  def test_a11_canonical_family_projection_and_objectives(self) -> None:
    from manafold.taxonomy.family_dataset import project_dataset_to_canonical_families
    from manafold.models.features.card_packages import PackageFeatureSet
    from manafold.models.training.training_pipeline import create_classifier, MODEL_SET_TRANSFORMER_A11

    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)
      dataset = load_training_dataset(dataset_path)
      from manafold.taxonomy.auto_ontology import generate_auto_ontology
      relation_path = dataset_path / "auto_ontology.json"
      generate_auto_ontology(dataset_path, relation_path)
      projected, mapping = project_dataset_to_canonical_families(
        dataset,
        family_relations_path=relation_path,
      )

      classifier, metadata = create_classifier(
        MODEL_SET_TRANSFORMER_A11,
        labels=projected.labels,
        card_count=projected.card_count,
        zone_count=projected.zone_count,
        quantity_count=5,
        package_features=PackageFeatureSet([]),
        package_feature_set_name=None,
        learning_rate=0.005,
        weight_decay=0.001,
        seed=13,
        embedding_dim=4,
        hidden_dim=8,
        head_v2_rho_hidden_dim=16,
        attention_heads=2,
        attention_layers=1,
        batch_size=2,
        shuffle=False,
        device="cpu",
        package_scale=1.0,
        package_projection_dim=4,
        architecture_package_count=0,
      )

      self.assertEqual(projected.target_source, "family_canonical_proxy")
      self.assertTrue(classifier.residual_anchor_logits)
      self.assertTrue(classifier.anchor_gating)
      self.assertEqual(metadata["embedding_dim"], 4)

  def test_a11_projection_preserves_unknown_scoring_sentinel(self) -> None:
    from dataclasses import replace
    from manafold.taxonomy.auto_ontology import generate_auto_ontology
    from manafold.datasets.model_inputs import UNKNOWN_LABEL_ID
    from manafold.taxonomy.family_dataset import project_dataset_to_canonical_families

    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)
      dataset = load_training_dataset(dataset_path)
      unknown = replace(dataset.examples[0], target_label_id=UNKNOWN_LABEL_ID)
      dataset = replace(
        dataset,
        examples=(*dataset.examples, unknown),
        labels=(*dataset.labels, UNKNOWN_LABEL_ID),
      )
      relation_path = dataset_path / "auto_ontology.json"
      generate_auto_ontology(dataset_path, relation_path)
      projected, mapping = project_dataset_to_canonical_families(
        dataset,
        family_relations_path=relation_path,
      )

      self.assertNotIn(UNKNOWN_LABEL_ID, mapping)
      self.assertEqual(UNKNOWN_LABEL_ID, projected.examples[-1].target_label_id)


if __name__ == "__main__":
  unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from manafold.models import train as model_training
from tests.model_test_support import _write_dataset


class ModelExperimentTests(unittest.TestCase):
  def test_default_head_v2_is_not_affected_by_package_flags(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      small_package_config = model_training.run_model_training(
        dataset_path,
        model_names=(model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2,),
        epochs=1,
        learning_rate=0.05,
        batch_size=2,
        device="cpu",
        package_max_count=8,
        package_projection_dim=4,
      )
      large_package_config = model_training.run_model_training(
        dataset_path,
        model_names=(model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2,),
        epochs=1,
        learning_rate=0.05,
        batch_size=2,
        device="cpu",
        package_max_count=4096,
        package_projection_dim=128,
      )

      small_model = small_package_config["models"][
        model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2
      ]
      large_model = large_package_config["models"][
        model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2
      ]
      self.assertEqual(model_training.DEFAULT_HEAD_V2_RHO_HIDDEN_DIM, small_model["rho_hidden_dim"])
      self.assertEqual(model_training.DEFAULT_HEAD_V2_RHO_HIDDEN_DIM, large_model["rho_hidden_dim"])
      self.assertEqual({"enabled": False}, small_package_config["package_mining"])
      self.assertEqual({"enabled": False}, large_package_config["package_mining"])

  def test_model_training_can_use_canonical_family_targets(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir) / "dataset"
      dataset_path.mkdir()
      _write_dataset(dataset_path)
      canonical_targets = Path(temp_dir) / "canonical_targets.yaml"
      canonical_targets.write_text(
        json.dumps(
          {
            "aliases": [
              {
                "source_label": "Beta",
                "canonical_family": "Alpha",
                "target_source": "source_archetype_name_proxy",
                "format": "modern",
                "valid_from": "2024-01-01",
                "valid_to": "2024-12-31",
                "confidence": "high",
                "evidence": {"source": "unit-test"},
              }
            ]
          }
        )
      )

      result = model_training.run_model_training(
        dataset_path,
        model_names=(model_training.MODEL_POOLED_LINEAR,),
        epochs=1,
        batch_size=2,
        device="cpu",
        target_label_level=model_training.TARGET_LABEL_LEVEL_CANONICAL_FAMILY,
        canonical_targets=canonical_targets,
        taxonomy_eval=canonical_targets,
      )

      model_result = result["models"][model_training.MODEL_POOLED_LINEAR]
      self.assertEqual(
        model_training.TARGET_LABEL_LEVEL_CANONICAL_FAMILY,
        result["training_target"]["label_level"],
      )
      self.assertEqual(1, result["train_label_count"])
      self.assertEqual(2, result["observed_label_count"])
      self.assertEqual(3, result["source_observed_label_count"])
      self.assertEqual(
        1.0,
        model_result["metrics"]["splits"]["test"]["accuracy"],
      )
      self.assertEqual(
        0.0,
        model_result["source_label_evaluation"]["metrics"]["splits"]["test"][
          "accuracy"
        ],
      )
      self.assertEqual(
        1.0,
        model_result["taxonomy_evaluation"]["metrics"]["splits"]["test"][
          "accuracy"
        ],
      )
      self.assertEqual(
        1,
        model_result["multi_seed_summary"]["primary_source_label_accuracy"][
          "count"
        ],
      )
      self.assertEqual(
        1,
        model_result["multi_seed_summary"]["primary_canonical_accuracy"]["count"],
      )

  def test_regularized_deepsets_models_use_model_specific_weight_decay(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = model_training.run_model_training(
        dataset_path,
        model_names=(
          model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
          model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2_REGULARIZED,
        ),
        epochs=1,
        learning_rate=0.05,
        weight_decay=0.0,
        deepsets_regularized_weight_decay=0.002,
        head_v2_weight_decay=0.003,
        batch_size=2,
        device="cpu",
      )

      deepsets = result["models"][model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED]
      head_v2 = result["models"][
        model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2_REGULARIZED
      ]
      self.assertEqual({"enabled": False}, result["package_mining"])
      self.assertEqual({}, result["package_mining_sets"])
      self.assertEqual(0.002, deepsets["weight_decay"])
      self.assertEqual(0.002, deepsets["model_config"]["weight_decay"])
      self.assertEqual(0.05, deepsets["model_config"]["learning_rate"])
      self.assertEqual(64, deepsets["rho_hidden_dim"])
      self.assertEqual("softmax", deepsets["model_config"]["head"])
      self.assertEqual(0.003, head_v2["weight_decay"])
      self.assertEqual(0.003, head_v2["model_config"]["weight_decay"])
      self.assertEqual(0.05, head_v2["model_config"]["learning_rate"])
      self.assertEqual(model_training.DEFAULT_HEAD_V2_RHO_HIDDEN_DIM, head_v2["rho_hidden_dim"])
      self.assertEqual("wide_rho", head_v2["model_config"]["head"])
      self.assertEqual(
        model_training.DEFAULT_HEAD_V2_RHO_HIDDEN_DIM,
        head_v2["head_v2_rho_hidden_dim"],
      )
      self.assertNotIn("architecture_reference_package_count", head_v2)

  def test_package_diagnostics_require_explicit_package_model(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = model_training.run_model_training(
        dataset_path,
        model_names=(model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES,),
        epochs=3,
        learning_rate=0.05,
        batch_size=2,
        device="cpu",
        package_min_support=1,
        package_min_event_support=1,
        package_max_size=2,
        package_max_count=8,
        package_scale=0.25,
        package_min_best_label_count=1,
        package_min_best_label_precision=0.0,
        package_max_train_activation_rate=1.0,
      )

      self.assertTrue(result["package_mining"]["enabled"])
      self.assertGreater(result["package_mining"]["count"], 0)
      self.assertIn("v2", result["package_mining_sets"])
      self.assertIn("zero", result["package_mining_sets"])
      self.assertIn("shuffled", result["package_mining_sets"])
      self.assertIn("synthetic", result["package_mining_sets"])
      self.assertIn("card_names", result["package_mining"]["diagnostics"][0])
      self.assertIn("package_type", result["package_mining"]["diagnostics"][0])
      self.assertIn("best_label_count", result["package_mining"]["diagnostics"][0])
      self.assertIn(
        "train_activation_rate",
        result["package_mining"]["diagnostics"][0],
      )

      model_result = result["models"][model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES]
      self.assertGreater(model_result["package_count"], 0)
      self.assertEqual("v2", model_result["package_feature_set"])
      self.assertIn("inference_package_ablations", model_result)
      self.assertIn("zeroed", model_result["inference_package_ablations"])
      self.assertIn(
        "paired_vs_normal",
        model_result["inference_package_ablations"]["zeroed"],
      )
      self.assertIn(
        "logit_delta_vs_normal",
        model_result["inference_package_ablations"]["zeroed"],
      )
      self.assertIn(
        "mean_abs_logit_delta",
        model_result["inference_package_ablations"]["zeroed"][
          "logit_delta_vs_normal"
        ]["test"],
      )
      self.assertEqual("completed", model_result["package_branch_usage"]["status"])
      self.assertIn(
        "package_projection_gradient_l2_norm",
        model_result["package_branch_usage"],
      )

  def test_package_signal_report_uses_primary_evaluation_split(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path, final_holdout=True)

      result = model_training.run_model_training(
        dataset_path,
        model_names=(
          model_training.MODEL_POOLED_LINEAR,
          model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED,
          model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES,
        ),
        epochs=1,
        learning_rate=0.05,
        batch_size=2,
        device="cpu",
        package_min_support=1,
        package_min_event_support=1,
        package_max_size=2,
        package_max_count=8,
        package_scale=0.25,
        package_min_best_label_count=1,
        package_min_best_label_precision=0.0,
        package_max_train_activation_rate=1.0,
      )

      self.assertEqual("final-test", result["primary_evaluation_split"])
      diagnostic = result["package_mining"]["diagnostics"][0]
      self.assertIn("dev-test_activation_rate", diagnostic)
      self.assertIn("final-test_activation_rate", diagnostic)
      self.assertGreater(len(result["package_signal_by_label"]), 0)
      first_row = result["package_signal_by_label"][0]
      self.assertEqual("final-test", first_row["evaluation_split"])
      self.assertIn("evaluation_support", first_row)
      self.assertNotIn("test_support", first_row)

  def test_model_training_keeps_explicit_package_controls_available(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = model_training.run_model_training(
        dataset_path,
        model_names=(
          model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES,
          model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO,
          model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES,
          model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES,
        ),
        epochs=1,
        learning_rate=0.05,
        batch_size=2,
        device="cpu",
        package_min_support=1,
        package_min_event_support=1,
        package_max_size=2,
        package_max_count=8,
        package_min_best_label_count=1,
        package_min_best_label_precision=0.0,
        package_max_train_activation_rate=1.0,
      )

      self.assertEqual(
        "zero",
        result["models"][model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES][
          "package_feature_set"
        ],
      )
      self.assertEqual(
        "synthetic",
        result["models"][model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES][
          "package_feature_set"
        ],
      )
      self.assertEqual(
        "label_shuffled_mined",
        result["models"][
          model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES
        ]["package_feature_set"],
      )

  def test_model_training_can_run_prototype_head(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = model_training.run_model_training(
        dataset_path,
        model_names=(model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE,),
        epochs=3,
        learning_rate=0.05,
        batch_size=2,
        device="cpu",
      )

      model_result = result["models"][model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE]
      self.assertEqual("completed", result["status"])
      self.assertIn(
        "nearest_prototype_distance",
        model_result["metrics"]["abstention"],
      )
      self.assertEqual(
        model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED,
        model_result["encoder_model_family"],
      )

  def test_model_training_can_run_set_transformer_quantity_weighted(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = model_training.run_model_training(
        dataset_path,
        model_names=(model_training.MODEL_SET_TRANSFORMER_QUANTITY_WEIGHTED,),
        epochs=1,
        learning_rate=0.05,
        batch_size=2,
        device="cpu",
      )

      model_result = result["models"][model_training.MODEL_SET_TRANSFORMER_QUANTITY_WEIGHTED]
      self.assertEqual("completed", result["status"])
      self.assertEqual(
        [model_training.MODEL_SET_TRANSFORMER_QUANTITY_WEIGHTED],
        result["model_names"],
      )
      self.assertEqual("quantity-weighted", model_result["pooling"])
      self.assertIn("metrics", model_result)



if __name__ == "__main__":
  unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from manafold.models.training import training_pipeline as model_training
from tests.model_test_support import _write_dataset


class ModelTrainingTests(unittest.TestCase):
  def test_a3_alias_resolves_to_the_selected_set_transformer(self) -> None:
    self.assertEqual(
      (model_training.MODEL_SET_TRANSFORMER_PARTIAL_BALANCED,),
      model_training._normalize_model_names(
        (model_training.MODEL_A3,),
        pooling=model_training.POOLING_SUM,
      ),
    )

  def test_a8_alias_resolves_to_product_anchor_set_transformer(self) -> None:
    self.assertEqual(
      (model_training.MODEL_SET_TRANSFORMER_A8,),
      model_training._normalize_model_names(
        (model_training.MODEL_A8,),
        pooling=model_training.POOLING_SUM,
      ),
    )

  def test_default_models_emit_predictions_metrics_and_embeddings(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)
      output = dataset_path / "model_runs" / "training_results.json"

      result = model_training.train_models(
        dataset_path,
        output=output,
        epochs=3,
        learning_rate=0.05,
        batch_size=2,
        device="cpu",
        export_embeddings=True,
        prediction_output=model_training.PREDICTION_OUTPUT_FULL,
      )

      expected_models = [
        model_training.MODEL_POOLED_LINEAR,
        model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED,
        model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
      ]
      self.assertEqual("completed", result["status"])
      self.assertEqual(expected_models, result["model_names"])
      self.assertEqual({"enabled": False}, result["package_mining"])
      self.assertEqual(
        {
          model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED,
          model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
        },
        set(result["comparison"]),
      )
      for model_name in expected_models:
        self.assertIn("metrics", result["models"][model_name])

      deepsets = result["models"][model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED]
      prediction_path = Path(deepsets["prediction_path"])
      prediction = json.loads(prediction_path.read_text().splitlines()[0])
      self.assertTrue({
        "deck_id",
        "event_id",
        "event_date",
        "entropy",
        "energy",
        "max_softmax_probability",
        "temperature_scaled_max_softmax_probability",
      } <= prediction.keys())
      self.assertTrue(Path(deepsets["card_embedding_path"]).exists())
      self.assertTrue(Path(deepsets["deck_embedding_path"]).exists())
      self.assertIn(
        "closed_set_seen_labels",
        deepsets["metrics"]["evaluation_views"]["test"],
      )
      self.assertIn("energy", deepsets["metrics"]["abstention"])
      self.assertIsNotNone(deepsets["best_validation_epoch"])
      self.assertIsNotNone(deepsets["best_validation_metric"])

      regularized = result["models"][
        model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED
      ]
      self.assertEqual(0.001, regularized["model_config"]["weight_decay"])
      self.assertEqual(result, json.loads(output.read_text()))

  def test_model_training_uses_safe_default_learning_rate(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = model_training.train_models(
        dataset_path,
        model_names=(model_training.MODEL_POOLED_LINEAR,),
        epochs=1,
        batch_size=2,
        device="cpu",
      )

      model_result = result["models"][model_training.MODEL_POOLED_LINEAR]
      self.assertEqual(model_training.DEFAULT_LEARNING_RATE, result["learning_rate"])
      self.assertEqual(model_training.DEFAULT_LEARNING_RATE, model_result["learning_rate"])
      self.assertEqual(
        model_training.DEFAULT_LEARNING_RATE,
        model_result["model_config"]["learning_rate"],
      )

  def test_model_training_uses_final_test_as_primary_evaluation_split(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path, final_holdout=True)

      result = model_training.train_models(
        dataset_path,
        model_names=(
          model_training.MODEL_POOLED_LINEAR,
          model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
        ),
        epochs=1,
        batch_size=2,
        device="cpu",
      )

      model_result = result["models"][model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED]
      self.assertEqual(
        ["validation", "dev-test", "final-test"],
        result["evaluation_splits"],
      )
      self.assertEqual("final-test", result["primary_evaluation_split"])
      self.assertEqual("final-test", model_result["primary_evaluation_split"])
      self.assertIn("primary_metrics_at_best_validation_epoch", model_result)
      self.assertEqual(
        1,
        model_result["multi_seed_summary"]["primary_accuracy"]["count"],
      )
      self.assertEqual(
        False,
        "test_accuracy" in model_result["multi_seed_summary"],
      )
      self.assertEqual(
        ["final-test-b"],
        [row["deck_id"] for row in model_result["prediction_pairing"]],
      )
      self.assertIn("dev-test", model_result["metrics"]["splits"])
      self.assertIn("final-test", model_result["metrics"]["splits"])
      self.assertNotIn("test", model_result["metrics"]["splits"])
      self.assertIn(
        "final-test",
        result["comparison"][model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED],
      )

  def test_model_training_can_run_only_pooled_linear(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = model_training.train_models(
        dataset_path,
        model_names=(model_training.MODEL_POOLED_LINEAR,),
        epochs=10,
      )

      self.assertEqual("completed", result["status"])
      self.assertEqual([model_training.MODEL_POOLED_LINEAR], result["model_names"])
      self.assertEqual([model_training.MODEL_POOLED_LINEAR], list(result["models"]))
      self.assertEqual({}, result["comparison"])
      self.assertIn("predictions", result["models"][model_training.MODEL_POOLED_LINEAR])

  def test_model_training_omits_embeddings_by_default(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      output = dataset_path / "model_runs" / "training_results.json"
      result = model_training.train_models(
        dataset_path,
        output=output,
        model_names=(model_training.MODEL_DEEPSETS_SUM,),
        epochs=1,
        learning_rate=0.05,
        batch_size=2,
        device="cpu",
      )

      self.assertIn("prediction_path", result["models"][model_training.MODEL_DEEPSETS_SUM])
      self.assertNotIn("card_embedding_path", result["models"][model_training.MODEL_DEEPSETS_SUM])
      self.assertNotIn("deck_embedding_path", result["models"][model_training.MODEL_DEEPSETS_SUM])

  def test_model_training_can_write_summary_without_prediction_jsonl(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      output = dataset_path / "model_runs" / "training_results.json"
      result = model_training.train_models(
        dataset_path,
        output=output,
        model_names=(model_training.MODEL_POOLED_LINEAR,),
        epochs=1,
        prediction_output=model_training.PREDICTION_OUTPUT_SUMMARY,
      )

      model_result = result["models"][model_training.MODEL_POOLED_LINEAR]
      self.assertEqual(model_training.PREDICTION_OUTPUT_SUMMARY, model_result["prediction_output"])
      self.assertEqual(0, model_result["prediction_export_count"])
      self.assertNotIn("prediction_path", model_result)

  def test_plusplus_models_record_supported_regularization(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = model_training.train_models(
        dataset_path,
        model_names=(
          model_training.MODEL_DEEPSETS_PLUSPLUS_REGULARIZED,
          model_training.MODEL_DEEPSETS_PLUSPLUS_REGULARIZED_LABEL_SMOOTHING_010,
        ),
        epochs=1,
        batch_size=2,
        device="cpu",
      )

      base = result["models"][model_training.MODEL_DEEPSETS_PLUSPLUS_REGULARIZED]
      smoothed = result["models"][
        model_training.MODEL_DEEPSETS_PLUSPLUS_REGULARIZED_LABEL_SMOOTHING_010
      ]
      self.assertEqual("plusplus", base["model_config"]["architecture"])
      self.assertEqual(0.1, base["model_config"]["dropout"])
      self.assertEqual(0.0, base["model_config"]["label_smoothing"])
      self.assertEqual(0.10, smoothed["model_config"]["label_smoothing"])
      self.assertEqual(model_training.DEFAULT_REGULARIZED_WEIGHT_DECAY, base["weight_decay"])

  def test_selected_set_transformer_records_production_policy(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = model_training.train_models(
        dataset_path,
        model_names=(model_training.MODEL_SET_TRANSFORMER_PARTIAL_BALANCED,),
        epochs=1,
        batch_size=2,
        embedding_dim=4,
        hidden_dim=8,
        attention_heads=2,
        attention_layers=1,
        device="cpu",
      )

      model = result["models"][model_training.MODEL_SET_TRANSFORMER_PARTIAL_BALANCED]
      partial = model["partial_observation"]
      self.assertEqual("mainboard", model["token_scope"])
      self.assertEqual("hypergeometric", model["pooling"])
      self.assertEqual(0.10, model["label_smoothing"])
      self.assertTrue(partial["enabled"])
      self.assertEqual(model_training.PARTIAL_CORRUPTION_MIXTURE, partial["corruption_policy"])
      self.assertEqual(
        "natural_sqrt_balanced",
        partial["training_sampling"]["policy"],
      )
      self.assertGreater(
        model["training_parameter_count"],
        model["inference_parameter_count"],
      )
      self.assertIsNotNone(
        model["training_history"][0]["train_partial_contextual_loss"]
      )

  def test_model_training_summarizes_multiple_seeds(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      output = dataset_path / "model_runs" / "training_results.json"
      result = model_training.train_models(
        dataset_path,
        output=output,
        model_names=(model_training.MODEL_POOLED_LINEAR,),
        epochs=2,
        learning_rate=0.05,
        batch_size=2,
        seeds=(13, 17),
      )

      model_result = result["models"][model_training.MODEL_POOLED_LINEAR]
      self.assertEqual([13, 17], result["seeds"])
      self.assertEqual([13, 17], model_result["seeds"])
      self.assertEqual(2, len(model_result["seed_runs"]))
      self.assertEqual(
        2,
        model_result["multi_seed_summary"]["primary_accuracy"]["count"],
      )
      self.assertEqual(
        2,
        model_result["multi_seed_summary"]["primary_temperature_scaled_nll"]["count"],
      )
      self.assertEqual(
        2,
        model_result["multi_seed_summary"]["primary_temperature_scaled_brier"]["count"],
      )
      self.assertEqual(
        2,
        model_result["multi_seed_summary"]["primary_temperature_scaled_ece"]["count"],
      )
      self.assertIn(
        "primary_temperature_scaled_msp_auroc",
        model_result["multi_seed_summary"],
      )
      self.assertIn(
        "primary_msp_auroc",
        model_result["multi_seed_summary"],
      )
      self.assertTrue(
        Path(model_result["seed_runs"][0]["prediction_path"]).name.endswith(
          "_seed_13.jsonl"
        )
      )

  def test_model_training_can_use_fixed_optimizer_steps(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = model_training.train_models(
        dataset_path,
        model_names=(model_training.MODEL_DEEPSETS_SUM,),
        epochs=1,
        max_steps=3,
        learning_rate=0.05,
        batch_size=1,
        device="cpu",
      )

      model_result = result["models"][model_training.MODEL_DEEPSETS_SUM]
      self.assertEqual(3, result["max_steps"])
      self.assertEqual(3, model_result["max_steps"])
      self.assertEqual(3, model_result["optimizer_steps"])
      self.assertEqual(2, model_result["completed_epochs"])
      self.assertEqual(3, model_result["training_history"][-1]["optimizer_steps"])



if __name__ == "__main__":
  unittest.main()

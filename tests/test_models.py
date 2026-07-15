from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from manafold.data.validate import write_parquet
from manafold.models.data import build_model_examples, load_training_dataset
from manafold.models.deepsets import (
  POOLING_QUANTITY_WEIGHTED,
  DeepSetsClassifier,
)
from manafold.models.aliases import run_alias_candidate_scoring
from manafold.models.metrics import classification_metrics
from manafold.models.model_artifacts import run_model_scoring
from manafold.models.rolling import (
  parse_rolling_window,
  summarize_across_window_diagnostics,
  summarize_rolling_window_results,
)
from manafold.models.taxonomy import (
  TaxonomyAliasRule,
  TaxonomyEvaluationConfig,
  label_id_for,
  taxonomy_metrics,
)
from manafold.models.weak_relations import run_weak_relation_graph
from manafold.models.weak_state import run_weak_relation_state
from manafold.models.weak_targets import run_weak_target_preview
from manafold.models.train import (
  DEFAULT_HEAD_V2_RHO_HIDDEN_DIM,
  DEFAULT_LEARNING_RATE,
  DEFAULT_REGULARIZED_WEIGHT_DECAY,
  MODEL_DEEPSETS_SUM,
  MODEL_DEEPSETS_MEAN,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_CONSTANT_FEATURES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2_REGULARIZED,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES_V1,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_SUPPORT_MATCHED_UNSCORED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_WIDE_RHO,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_EXTRA_DIMS_PRESERVE_INIT,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES,
  MODEL_POOLED_LINEAR,
  MODEL_POOLED_LINEAR_PACKAGE_ONLY,
  MODEL_POOLED_LINEAR_PACKAGES,
  MODEL_SET_TRANSFORMER_QUANTITY_WEIGHTED,
  PREDICTION_OUTPUT_FULL,
  PREDICTION_OUTPUT_SUMMARY,
  TARGET_LABEL_LEVEL_CANONICAL_FAMILY,
  run_model_training,
)


class ModelTests(unittest.TestCase):
  def test_model_examples_group_variable_sized_decks(self) -> None:
    examples = build_model_examples(
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

  def test_model_training_compares_deepsets_against_pooled_linear(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      output = dataset_path / "model_runs" / "training_results.json"
      result = run_model_training(
        dataset_path,
        output=output,
        epochs=30,
        learning_rate=0.05,
        package_weight_decay=0.001,
        batch_size=2,
        device="cpu",
        export_embeddings=True,
        prediction_output=PREDICTION_OUTPUT_FULL,
        package_min_support=1,
        package_min_event_support=1,
        package_max_size=2,
        package_max_count=8,
        package_scale=0.25,
        package_min_best_label_count=1,
        package_min_best_label_precision=0.0,
        package_max_train_activation_rate=1.0,
      )

      self.assertEqual("completed", result["status"])
      self.assertEqual("train_only", result["label_vocabulary"])
      self.assertEqual(2, result["train_label_count"])
      self.assertEqual(3, result["observed_label_count"])
      self.assertEqual(
        [
          MODEL_POOLED_LINEAR,
          MODEL_DEEPSETS_QUANTITY_WEIGHTED,
          MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
        ],
        result["model_names"],
      )
      self.assertFalse(result["package_mining"]["enabled"])
      self.assertEqual({"enabled": False}, result["package_mining"])
      self.assertEqual(0.001, result["package_weight_decay"])
      self.assertEqual({}, result["package_mining_sets"])
      self.assertEqual(
        1.0,
        result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED]["metrics"]["splits"]["test"]["top_3_accuracy"],
      )
      self.assertEqual(
        1.0,
        result["models"][MODEL_POOLED_LINEAR]["metrics"]["splits"]["test"]["accuracy"],
      )
      self.assertIsNotNone(
        result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED]["best_validation_epoch"]
      )
      self.assertIsNotNone(
        result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED]["best_validation_metric"]
      )
      self.assertIn(
        "primary_metrics_at_best_validation_epoch",
        result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED],
      )
      self.assertIn(MODEL_DEEPSETS_QUANTITY_WEIGHTED, result["comparison"])
      self.assertIn(MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED, result["comparison"])
      self.assertEqual(
        1,
        result["train_label_support"]["proxy.source_archetype_name_proxy.alpha"],
      )
      self.assertIn(
        "closed_set_seen_labels",
        result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED]["metrics"]["evaluation_views"]["test"],
      )
      self.assertIn(
        "train_support_1_9",
        result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED]["metrics"]["support_buckets"]["test"],
      )
      self.assertIn(
        "energy",
        result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED]["metrics"]["abstention"],
      )
      self.assertIn(
        "temperature_scaled_max_softmax_probability",
        result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED]["metrics"]["abstention"],
      )
      calibration = result["models"][MODEL_POOLED_LINEAR]["metrics"]["calibration"]
      self.assertEqual("completed", calibration["status"])
      self.assertEqual("closed_set_train_labels", calibration["target_scope"])
      self.assertEqual("validation", calibration["selection_split"])
      self.assertEqual(1, calibration["selection"]["known_count"])
      self.assertEqual(1, calibration["selection"]["excluded_unknown_count"])
      self.assertEqual(1, calibration["uncalibrated"]["splits"]["test"]["count"])
      self.assertIn("nll", calibration["uncalibrated"]["splits"]["test"])
      self.assertIn("brier", calibration["uncalibrated"]["splits"]["test"])
      self.assertIn("ece", calibration["uncalibrated"]["splits"]["test"])
      self.assertIn("temperature_scaled", calibration)
      self.assertIsNotNone(calibration["temperature_scaled"]["splits"]["test"]["nll"])
      self.assertNotIn("predictions", result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED])
      self.assertEqual(
        "predictions",
        Path(result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED]["prediction_path"]).parent.name,
      )
      self.assertTrue(
        Path(result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED]["prediction_path"]).exists()
      )
      prediction = json.loads(
        Path(result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED]["prediction_path"])
        .read_text()
        .splitlines()[0]
      )
      self.assertIn("entropy", prediction)
      self.assertIn("energy", prediction)
      self.assertIn("event_id", prediction)
      self.assertIn("event_date", prediction)
      self.assertEqual("modern", prediction["format"])
      self.assertEqual("source_archetype_name_proxy", prediction["target_source"])
      self.assertIn("max_softmax_probability", prediction)
      self.assertIn("temperature_scaled_max_softmax_probability", prediction)
      self.assertTrue(
        Path(result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED]["card_embedding_path"]).exists()
      )
      self.assertTrue(
        Path(result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED]["deck_embedding_path"]).exists()
      )
      self.assertEqual(
        0.001,
        result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED][
          "weight_decay"
        ],
      )
      model_config = result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED][
        "model_config"
      ]
      self.assertEqual("softmax", model_config["head"])
      self.assertEqual(64, model_config["rho_hidden_dim"])
      self.assertEqual(0.001, model_config["weight_decay"])
      self.assertEqual(0.05, model_config["learning_rate"])
      self.assertEqual(
        {"enabled": False},
        model_config["package_mining"],
      )
      self.assertIsNotNone(model_config["temperature"])
      self.assertIn("package_signal_by_label", result)
      self.assertEqual([], result["package_signal_by_label"])
      self.assertEqual(
        "embeddings",
        Path(result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED]["card_embedding_path"]).parent.name,
      )
      self.assertEqual(result, json.loads(output.read_text()))

  def test_default_head_v2_is_not_affected_by_package_flags(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      small_package_config = run_model_training(
        dataset_path,
        model_names=(MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2,),
        epochs=1,
        learning_rate=0.05,
        batch_size=2,
        device="cpu",
        package_max_count=8,
        package_projection_dim=4,
      )
      large_package_config = run_model_training(
        dataset_path,
        model_names=(MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2,),
        epochs=1,
        learning_rate=0.05,
        batch_size=2,
        device="cpu",
        package_max_count=4096,
        package_projection_dim=128,
      )

      small_model = small_package_config["models"][
        MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2
      ]
      large_model = large_package_config["models"][
        MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2
      ]
      self.assertEqual(DEFAULT_HEAD_V2_RHO_HIDDEN_DIM, small_model["rho_hidden_dim"])
      self.assertEqual(DEFAULT_HEAD_V2_RHO_HIDDEN_DIM, large_model["rho_hidden_dim"])
      self.assertEqual({"enabled": False}, small_package_config["package_mining"])
      self.assertEqual({"enabled": False}, large_package_config["package_mining"])

  def test_model_training_uses_safe_default_learning_rate(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = run_model_training(
        dataset_path,
        model_names=(MODEL_POOLED_LINEAR,),
        epochs=1,
        batch_size=2,
        device="cpu",
      )

      model_result = result["models"][MODEL_POOLED_LINEAR]
      self.assertEqual(DEFAULT_LEARNING_RATE, result["learning_rate"])
      self.assertEqual(DEFAULT_LEARNING_RATE, model_result["learning_rate"])
      self.assertEqual(
        DEFAULT_LEARNING_RATE,
        model_result["model_config"]["learning_rate"],
      )

  def test_model_artifact_scores_dataset(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir) / "dataset"
      dataset_path.mkdir()
      _write_dataset(dataset_path)
      artifact_dir = Path(temp_dir) / "model_artifact"
      training_output = dataset_path / "model_runs" / "training_results.json"

      result = run_model_training(
        dataset_path,
        output=training_output,
        model_names=(MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,),
        epochs=3,
        batch_size=2,
        device="cpu",
        model_artifact_output=artifact_dir,
        prediction_output=PREDICTION_OUTPUT_SUMMARY,
      )

      model_result = result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED]
      self.assertIn("model_artifact", model_result)
      for filename in (
        "model.pt",
        "model_config.json",
        "card_vocab.parquet",
        "label_vocab.json",
        "zone_vocab.json",
        "temperature.json",
        "training_manifest.json",
      ):
        self.assertTrue((artifact_dir / filename).exists())
      label_vocab = json.loads(
        (artifact_dir / "label_vocab.json").read_text(encoding="utf-8")
      )
      self.assertIn("entries", label_vocab)
      self.assertEqual("Alpha", label_vocab["entries"][0]["display_label"])

      output = Path(temp_dir) / "model_predictions.parquet"
      score_result = run_model_scoring(
        model_artifact=artifact_dir,
        dataset_path=dataset_path,
        output=output,
        batch_size=2,
        device="cpu",
      )

      rows = pq.read_table(output).to_pylist()
      self.assertEqual(len(load_training_dataset(dataset_path).examples), len(rows))
      self.assertEqual(len(rows), score_result["prediction_count"])
      first = rows[0]
      self.assertIn("deck_id", first)
      self.assertIn("event_id", first)
      self.assertIn("source_label", first)
      self.assertIn("source_label_id", first)
      self.assertIn("top1_label", first)
      self.assertIn("top1_label_id", first)
      self.assertIn("top3_labels", first)
      self.assertIn("top3_label_ids", first)
      self.assertIn("temperature_scaled_probability", first)
      self.assertIn("energy_score", first)
      self.assertIn("is_low_confidence", first)
      self.assertIn("model_version", first)
      self.assertTrue(str(first["top1_label_id"]).startswith("proxy."))
      self.assertIsInstance(first["top1_label"], str)
      self.assertTrue(Path(score_result["deck_embedding_output"]).exists())
      embedding_rows = pq.read_table(
        Path(score_result["deck_embedding_output"])
      ).to_pylist()
      self.assertEqual(len(rows), len(embedding_rows))

      remapped_dataset_path = Path(temp_dir) / "remapped_dataset"
      remapped_dataset_path.mkdir()
      _write_dataset(remapped_dataset_path)
      _rewrite_dataset_with_local_card_indexes(remapped_dataset_path)
      remapped_output = Path(temp_dir) / "remapped_predictions.parquet"
      remapped_score_result = run_model_scoring(
        model_artifact=artifact_dir,
        dataset_path=remapped_dataset_path,
        output=remapped_output,
        batch_size=2,
        device="cpu",
      )
      remapped_rows = pq.read_table(remapped_output).to_pylist()
      self.assertEqual(len(rows), len(remapped_rows))
      self.assertEqual(
        len(remapped_rows),
        remapped_score_result["prediction_count"],
      )

      unlabeled_dataset_path = Path(temp_dir) / "unlabeled_dataset"
      unlabeled_dataset_path.mkdir()
      _write_dataset(unlabeled_dataset_path)
      _remove_proxy_targets_artifact(unlabeled_dataset_path)
      unlabeled_output = Path(temp_dir) / "unlabeled_predictions.parquet"
      unlabeled_score_result = run_model_scoring(
        model_artifact=artifact_dir,
        dataset_path=unlabeled_dataset_path,
        output=unlabeled_output,
        batch_size=2,
        device="cpu",
      )
      unlabeled_rows = pq.read_table(unlabeled_output).to_pylist()
      self.assertEqual(len(rows), len(unlabeled_rows))
      self.assertEqual(
        len(unlabeled_rows),
        unlabeled_score_result["prediction_count"],
      )
      self.assertTrue(
        all(row["source_label_id"] is None for row in unlabeled_rows)
      )
      self.assertTrue(all(row["source_label"] is None for row in unlabeled_rows))

      pooled_artifact_dir = Path(temp_dir) / "pooled_model_artifact"
      pooled_training_output = (
        dataset_path / "model_runs" / "pooled_training_results.json"
      )
      run_model_training(
        dataset_path,
        output=pooled_training_output,
        model_names=(MODEL_POOLED_LINEAR,),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_artifact_output=pooled_artifact_dir,
        model_artifact_model_name=MODEL_POOLED_LINEAR,
        prediction_output=PREDICTION_OUTPUT_SUMMARY,
      )
      pooled_output = Path(temp_dir) / "pooled_predictions.parquet"
      pooled_score_result = run_model_scoring(
        model_artifact=pooled_artifact_dir,
        dataset_path=dataset_path,
        output=pooled_output,
        batch_size=2,
        device="cpu",
      )
      pooled_rows = pq.read_table(pooled_output).to_pylist()
      self.assertEqual(len(rows), len(pooled_rows))
      self.assertIsNone(pooled_score_result["deck_embedding_output"])

  def test_model_artifact_export_requires_explicit_seed_policy(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir) / "dataset"
      dataset_path.mkdir()
      _write_dataset(dataset_path)

      with self.assertRaisesRegex(ValueError, "seed policy 'single'"):
        run_model_training(
          dataset_path,
          model_names=(MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,),
          seeds=(13, 17),
          epochs=1,
          batch_size=2,
          device="cpu",
          model_artifact_output=Path(temp_dir) / "model_artifact",
          prediction_output=PREDICTION_OUTPUT_SUMMARY,
        )

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

      result = run_model_training(
        dataset_path,
        model_names=(MODEL_POOLED_LINEAR,),
        epochs=1,
        batch_size=2,
        device="cpu",
        target_label_level=TARGET_LABEL_LEVEL_CANONICAL_FAMILY,
        canonical_targets=canonical_targets,
        taxonomy_eval=canonical_targets,
      )

      model_result = result["models"][MODEL_POOLED_LINEAR]
      self.assertEqual(
        TARGET_LABEL_LEVEL_CANONICAL_FAMILY,
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

  def test_parse_rolling_window(self) -> None:
    window = parse_rolling_window(
      "window_1,2023-01-01,2025-06-30,2025-12-31,2026-06-30,2026-12-31"
    )

    self.assertEqual("window_1", window.name)
    self.assertEqual(date(2023, 1, 1), window.start)
    self.assertEqual(date(2025, 6, 30), window.train_end)
    self.assertEqual(date(2025, 12, 31), window.validation_end)
    self.assertEqual(date(2026, 6, 30), window.dev_test_end)
    self.assertEqual(date(2026, 12, 31), window.end)

  def test_rolling_summary_aggregates_window_means(self) -> None:
    summary = summarize_rolling_window_results([
      {
        "models": {
          MODEL_POOLED_LINEAR: {
            "multi_seed_summary": {
              "primary_accuracy": {"mean": 0.6},
              "primary_closed_set_accuracy": {"mean": 0.7},
            },
          },
        },
      },
      {
        "models": {
          MODEL_POOLED_LINEAR: {
            "multi_seed_summary": {
              "primary_accuracy": {"mean": 0.8},
              "primary_closed_set_accuracy": {"mean": 0.9},
            },
          },
        },
      },
    ])

    metrics = summary[MODEL_POOLED_LINEAR]["metrics"]
    self.assertEqual(2, metrics["primary_accuracy"]["count"])
    self.assertAlmostEqual(0.7, metrics["primary_accuracy"]["mean"])
    self.assertAlmostEqual(0.1, metrics["primary_accuracy"]["std"])
    self.assertEqual(
      [0.6, 0.8],
      metrics["primary_accuracy"]["values"],
    )
    self.assertAlmostEqual(0.8, metrics["primary_closed_set_accuracy"]["mean"])

  def test_rolling_diagnostics_rank_models_by_window(self) -> None:
    diagnostics = summarize_across_window_diagnostics([
      {
        "window": {"name": "window_1"},
        "models": {
          MODEL_POOLED_LINEAR: {
            "multi_seed_summary": {
              "primary_accuracy": {"mean": 0.6},
              "primary_temperature_scaled_nll": {"mean": 0.8},
            },
          },
          MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED: {
            "multi_seed_summary": {
              "primary_accuracy": {"mean": 0.7},
              "primary_temperature_scaled_nll": {"mean": 0.7},
            },
          },
        },
      },
      {
        "window": {"name": "window_2"},
        "models": {
          MODEL_POOLED_LINEAR: {
            "multi_seed_summary": {
              "primary_accuracy": {"mean": 0.8},
              "primary_temperature_scaled_nll": {"mean": 0.6},
            },
          },
          MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED: {
            "multi_seed_summary": {
              "primary_accuracy": {"mean": 0.7},
              "primary_temperature_scaled_nll": {"mean": 0.7},
            },
          },
        },
      },
    ])

    accuracy_ranking = diagnostics["ranking_by_metric"]["primary_accuracy"]
    self.assertEqual(
      MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
      accuracy_ranking["windows"][0]["ranking"][0]["model"],
    )
    self.assertEqual(
      MODEL_POOLED_LINEAR,
      accuracy_ranking["windows"][1]["ranking"][0]["model"],
    )
    self.assertEqual(
      1,
      diagnostics["vs_pooled_linear"][
        MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED
      ]["primary_accuracy"]["wins"],
    )
    self.assertEqual(
      1,
      diagnostics["vs_pooled_linear"][
        MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED
      ]["primary_accuracy"]["losses"],
    )
    self.assertEqual(
      1,
      diagnostics["vs_pooled_linear"][
        MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED
      ]["primary_temperature_scaled_nll"]["wins"],
    )

  def test_model_training_uses_final_test_as_primary_evaluation_split(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path, final_holdout=True)

      result = run_model_training(
        dataset_path,
        model_names=(
          MODEL_POOLED_LINEAR,
          MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
        ),
        epochs=1,
        batch_size=2,
        device="cpu",
      )

      model_result = result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED]
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
        result["comparison"][MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED],
      )

  def test_regularized_deepsets_models_use_model_specific_weight_decay(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = run_model_training(
        dataset_path,
        model_names=(
          MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
          MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2_REGULARIZED,
        ),
        epochs=1,
        learning_rate=0.05,
        weight_decay=0.0,
        deepsets_regularized_weight_decay=0.002,
        head_v2_weight_decay=0.003,
        batch_size=2,
        device="cpu",
      )

      deepsets = result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED]
      head_v2 = result["models"][
        MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2_REGULARIZED
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
      self.assertEqual(DEFAULT_HEAD_V2_RHO_HIDDEN_DIM, head_v2["rho_hidden_dim"])
      self.assertEqual("wide_rho", head_v2["model_config"]["head"])
      self.assertEqual(
        DEFAULT_HEAD_V2_RHO_HIDDEN_DIM,
        head_v2["head_v2_rho_hidden_dim"],
      )
      self.assertNotIn("architecture_reference_package_count", head_v2)

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

  def test_package_diagnostics_require_explicit_package_model(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = run_model_training(
        dataset_path,
        model_names=(MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES,),
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

      model_result = result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES]
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

      result = run_model_training(
        dataset_path,
        model_names=(
          MODEL_POOLED_LINEAR,
          MODEL_DEEPSETS_QUANTITY_WEIGHTED,
          MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES,
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

      result = run_model_training(
        dataset_path,
        model_names=(
          MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES,
          MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO,
          MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES,
          MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES,
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
        result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES][
          "package_feature_set"
        ],
      )
      self.assertEqual(
        "synthetic",
        result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES][
          "package_feature_set"
        ],
      )
      self.assertEqual(
        "label_shuffled_mined",
        result["models"][
          MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES
        ]["package_feature_set"],
      )

  def test_model_training_can_run_prototype_head(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = run_model_training(
        dataset_path,
        model_names=(MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE,),
        epochs=3,
        learning_rate=0.05,
        batch_size=2,
        device="cpu",
      )

      model_result = result["models"][MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE]
      self.assertEqual("completed", result["status"])
      self.assertIn(
        "nearest_prototype_distance",
        model_result["metrics"]["abstention"],
      )
      self.assertEqual(
        MODEL_DEEPSETS_QUANTITY_WEIGHTED,
        model_result["encoder_model_family"],
      )

  def test_model_training_can_run_only_pooled_linear(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = run_model_training(
        dataset_path,
        model_names=(MODEL_POOLED_LINEAR,),
        epochs=10,
      )

      self.assertEqual("completed", result["status"])
      self.assertEqual([MODEL_POOLED_LINEAR], result["model_names"])
      self.assertEqual([MODEL_POOLED_LINEAR], list(result["models"]))
      self.assertEqual({}, result["comparison"])
      self.assertIn("predictions", result["models"][MODEL_POOLED_LINEAR])

  def test_model_training_omits_embeddings_by_default(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      output = dataset_path / "model_runs" / "training_results.json"
      result = run_model_training(
        dataset_path,
        output=output,
        model_names=(MODEL_DEEPSETS_SUM,),
        epochs=1,
        learning_rate=0.05,
        batch_size=2,
        device="cpu",
      )

      self.assertIn("prediction_path", result["models"][MODEL_DEEPSETS_SUM])
      self.assertNotIn("card_embedding_path", result["models"][MODEL_DEEPSETS_SUM])
      self.assertNotIn("deck_embedding_path", result["models"][MODEL_DEEPSETS_SUM])

  def test_model_training_can_write_summary_without_prediction_jsonl(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      output = dataset_path / "model_runs" / "training_results.json"
      result = run_model_training(
        dataset_path,
        output=output,
        model_names=(MODEL_POOLED_LINEAR,),
        epochs=1,
        prediction_output=PREDICTION_OUTPUT_SUMMARY,
      )

      model_result = result["models"][MODEL_POOLED_LINEAR]
      self.assertEqual(PREDICTION_OUTPUT_SUMMARY, model_result["prediction_output"])
      self.assertEqual(0, model_result["prediction_export_count"])
      self.assertNotIn("prediction_path", model_result)

  def test_model_training_can_run_set_transformer_quantity_weighted(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      result = run_model_training(
        dataset_path,
        model_names=(MODEL_SET_TRANSFORMER_QUANTITY_WEIGHTED,),
        epochs=1,
        learning_rate=0.05,
        batch_size=2,
        device="cpu",
      )

      model_result = result["models"][MODEL_SET_TRANSFORMER_QUANTITY_WEIGHTED]
      self.assertEqual("completed", result["status"])
      self.assertEqual(
        [MODEL_SET_TRANSFORMER_QUANTITY_WEIGHTED],
        result["model_names"],
      )
      self.assertEqual("quantity-weighted", model_result["pooling"])
      self.assertIn("metrics", model_result)

  def test_model_training_summarizes_multiple_seeds(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir)
      _write_dataset(dataset_path)

      output = dataset_path / "model_runs" / "training_results.json"
      result = run_model_training(
        dataset_path,
        output=output,
        model_names=(MODEL_POOLED_LINEAR,),
        epochs=2,
        learning_rate=0.05,
        batch_size=2,
        seeds=(13, 17),
      )

      model_result = result["models"][MODEL_POOLED_LINEAR]
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

      result = run_model_training(
        dataset_path,
        model_names=(MODEL_DEEPSETS_SUM,),
        epochs=1,
        max_steps=3,
        learning_rate=0.05,
        batch_size=1,
        device="cpu",
      )

      model_result = result["models"][MODEL_DEEPSETS_SUM]
      self.assertEqual(3, result["max_steps"])
      self.assertEqual(3, model_result["max_steps"])
      self.assertEqual(3, model_result["optimizer_steps"])
      self.assertEqual(2, model_result["completed_epochs"])
      self.assertEqual(3, model_result["training_history"][-1]["optimizer_steps"])

  def test_metrics_separate_closed_and_open_set_labels(self) -> None:
    predictions = [
      _prediction("train-a", "train", "alpha", "alpha", 0.9, 2),
      _prediction("train-b", "train", "alpha", "alpha", 0.8, 2),
      _prediction("train-c", "train", "beta", "beta", 0.7, 20),
      _prediction("validation-a", "validation", "alpha", "alpha", 0.9, 2),
      _prediction("validation-b", "validation", "gamma", "alpha", 0.2, 0),
      _prediction("test-a", "test", "alpha", "alpha", 0.9, 2),
      _prediction("test-b", "test", "gamma", "alpha", 0.3, 0),
      _prediction("test-c", "test", "beta", "alpha", 0.4, 20),
    ]

    metrics = classification_metrics(
      predictions,
      train_label_support={
        "alpha": 2,
        "beta": 20,
      },
    )

    test_views = metrics["evaluation_views"]["test"]
    self.assertEqual(2, test_views["closed_set_seen_labels"]["count"])
    self.assertEqual(0.5, test_views["closed_set_seen_labels"]["accuracy"])
    self.assertEqual(1, test_views["open_set_unseen_labels"]["count"])
    self.assertEqual(0.0, test_views["open_set_unseen_labels"]["accuracy"])
    self.assertEqual(
      1,
      metrics["support_buckets"]["test"]["absent_from_train"]["count"],
    )
    self.assertEqual(
      1,
      metrics["support_buckets"]["test"]["train_support_1_9"]["count"],
    )
    self.assertEqual(
      1,
      metrics["support_buckets"]["test"]["train_support_10_49"]["count"],
    )
    abstention = metrics["abstention"]["max_softmax_probability"]
    self.assertIsNotNone(abstention["validation_selected_threshold"])
    test_abstention = abstention["splits"]["test"]
    self.assertEqual(1.0, test_abstention["unknown_recall"])
    self.assertEqual(0.5, test_abstention["known_false_abstention_rate"])
    self.assertEqual(1 / 3, test_abstention["coverage"])
    self.assertEqual(0.5, test_abstention["known_coverage"])
    self.assertEqual(0.0, test_abstention["unknown_acceptance_rate"])
    self.assertEqual(1.0, test_abstention["accepted_known_accuracy"])
    self.assertEqual(
      1.0,
      test_abstention["closed_set_accuracy_at_accepted_coverage"],
    )
    self.assertIn("fpr_at_95_unknown_recall", test_abstention)
    self.assertIn("energy", metrics["abstention"])
    self.assertIn("temperature_scaled_max_softmax_probability", metrics["abstention"])

  def test_taxonomy_metrics_score_time_scoped_aliases(self) -> None:
    target_source = "source_archetype_name_proxy"
    boros_energy = label_id_for("Boros Energy", target_source)
    boros_midrange = label_id_for("Boros Midrange", target_source)
    config = TaxonomyEvaluationConfig(
      path=None,
      aliases=(
        TaxonomyAliasRule(
          source_label="Boros Midrange",
          canonical_family="Boros Energy",
          target_source=target_source,
          format_code="modern",
          valid_from=date(2026, 1, 1),
          valid_to=date(2026, 6, 30),
          confidence="high",
          evidence={"source": "test"},
        ),
      ),
    )
    predictions = [
      _prediction(
        "train-energy",
        "train",
        boros_energy,
        boros_energy,
        0.9,
        1,
        event_date="2025-12-01",
      ),
      _prediction(
        "final-alias",
        "final-test",
        boros_energy,
        boros_midrange,
        0.8,
        1,
        event_date="2026-02-01",
      ),
      _prediction(
        "final-out-of-window",
        "final-test",
        boros_energy,
        boros_midrange,
        0.8,
        1,
        event_date="2025-12-31",
      ),
    ]

    result = taxonomy_metrics(predictions, config)

    self.assertTrue(result["enabled"])
    final_metrics = result["metrics"]["splits"]["final-test"]
    self.assertEqual(2, final_metrics["count"])
    self.assertEqual(0.5, final_metrics["accuracy"])
    self.assertEqual(
      1,
      result["label_noise_report"]["total_exact_errors_canonical_matches"],
    )
    report_row = result["label_noise_report"][
      "exact_errors_canonical_matches"
    ][0]
    self.assertEqual("final-test", report_row["split_name"])
    self.assertEqual("Boros Energy", report_row["actual_source_label"])
    self.assertEqual("Boros Midrange", report_row["predicted_source_label"])
    self.assertEqual("Boros Energy", report_row["canonical_actual"])
    self.assertEqual("Boros Energy", report_row["canonical_predicted"])
    self.assertEqual({"source": "test"}, report_row["evidence"])

  def test_alias_candidate_scorer_outputs_overlap_candidates(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      dataset_path = root / "window"
      dataset_path.mkdir()
      _write_alias_candidate_dataset(dataset_path)
      prediction_path = root / "predictions.jsonl"
      prediction_path.write_text(
        json.dumps(
          {
            "deck_id": "final-beta",
            "event_date": "2026-02-01",
            "format": "modern",
            "split_name": "final-test",
            "actual_label_id": "proxy.source_archetype_name_proxy.beta",
            "source_actual_label_id": "proxy.source_archetype_name_proxy.beta",
            "predicted_label_id": "proxy.source_archetype_name_proxy.alpha",
            "is_correct": False,
          }
        )
        + "\n",
        encoding="utf-8",
      )
      empty_prediction_path = root / "predictions_seed_17.jsonl"
      empty_prediction_path.write_text("", encoding="utf-8")
      training_result_path = root / "training_results.json"
      training_result_path.write_text(
        json.dumps(
          {
            "models": {
              MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED: {
                "seed_runs": [
                  {
                    "prediction_path": str(prediction_path),
                  },
                  {
                    "prediction_path": str(empty_prediction_path),
                  }
                ],
              }
            }
          }
        ),
        encoding="utf-8",
      )
      rolling_result_path = root / "rolling_evaluation_results.json"
      rolling_result_path.write_text(
        json.dumps(
          {
            "windows": [
              {
                "window": {"name": "window_a"},
                "dataset_path": str(dataset_path),
                "training_result_path": str(training_result_path),
              }
            ]
          }
        ),
        encoding="utf-8",
      )
      taxonomy_path = root / "taxonomy_eval.yaml"
      taxonomy_path.write_text(
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
        ),
        encoding="utf-8",
      )

      output_path = root / "alias_candidates.json"
      result = run_alias_candidate_scoring(
        rolling_result_path,
        output=output_path,
        taxonomy_eval=taxonomy_path,
        min_confusion_count=1,
      )

      self.assertEqual(1, result["candidate_count"])
      self.assertEqual(1, result["pair_summary_count"])
      self.assertEqual(1, result["weak_label_observation_count"])
      candidate = result["candidates"][0]
      self.assertEqual("window_a", candidate["rolling_window"])
      self.assertEqual("final-test", candidate["split_name"])
      self.assertEqual("Alpha", candidate["label_a"])
      self.assertEqual("Beta", candidate["label_b"])
      self.assertEqual("alias_candidate", candidate["suggested_relation"])
      self.assertTrue(candidate["evidence"]["seed_alias_match"])
      self.assertEqual(1, candidate["evidence"]["confusion_count"])
      self.assertEqual(2, candidate["evidence"]["confusion_seed_count"])
      self.assertEqual(1, candidate["evidence"]["confusion_active_seed_count"])
      self.assertEqual(0.5, candidate["evidence"]["confusion_seed_consensus"])
      self.assertGreater(
        candidate["evidence"]["mainboard_adoption_weighted_jaccard"],
        0.0,
      )
      summary = result["pair_summaries"][0]
      self.assertEqual(["Alpha", "Beta"], summary["pair"])
      self.assertEqual("reviewed_alias_candidate", summary["recommendation"])
      observations_path = root / "alias_weak_label_observations.jsonl"
      observations = [
        json.loads(line)
        for line in observations_path.read_text(encoding="utf-8").splitlines()
      ]
      self.assertEqual(1, len(observations))
      observation = observations[0]
      self.assertEqual("window_a", observation["rolling_window"])
      self.assertEqual("final-test", observation["split_name"])
      self.assertEqual("alias_candidate", observation["relation"])
      self.assertEqual("Alpha", observation["label_a"])
      self.assertEqual("Beta", observation["label_b"])
      self.assertEqual(0.5, observation["seed_consensus"])
      self.assertTrue(observation["seed_alias_match"])
      self.assertFalse(observation["usable_for_training_suggestion"])
      self.assertEqual(
        "held_out_evaluation_split",
        observation["training_exclusion_reason"],
      )

  def test_alias_candidate_scorer_reads_prediction_parquet(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      dataset_path = root / "dataset"
      dataset_path.mkdir()
      _write_alias_candidate_dataset(dataset_path)
      predictions_path = root / "model_predictions.parquet"
      prediction_rows = [
        {
          "deck_id": "final-beta",
          "event_id": "event-final-beta",
          "event_date": "2026-02-01",
          "format": "modern",
          "split_name": "final-test",
          "source_label": "Beta",
          "source_label_id": "proxy.source_archetype_name_proxy.beta",
          "top1_label": "Alpha",
          "top1_label_id": "proxy.source_archetype_name_proxy.alpha",
          "top1_probability": 0.72,
          "top3_labels": ["Alpha", "Beta"],
          "top3_label_ids": [
            "proxy.source_archetype_name_proxy.alpha",
            "proxy.source_archetype_name_proxy.beta",
          ],
          "top3_probabilities": [0.72, 0.2],
          "temperature_scaled_probability": 0.68,
          "energy_score": -4.5,
          "msp_score": 0.72,
          "entropy": 0.7,
          "normalized_entropy": 0.5,
          "is_low_confidence": False,
          "embedding_id": "final-beta",
          "embedding_path": None,
          "model_version": "model:v0",
          "taxonomy_eval_version": None,
          "target_source": "source_archetype_name_proxy",
        },
        {
          "deck_id": "unlabeled-low",
          "event_id": "event-unlabeled-low",
          "event_date": "2026-02-01",
          "format": "modern",
          "split_name": "final-test",
          "source_label": None,
          "source_label_id": None,
          "top1_label": "Alpha",
          "top1_label_id": "proxy.source_archetype_name_proxy.alpha",
          "top1_probability": 0.21,
          "top3_labels": ["Alpha", "Beta"],
          "top3_label_ids": [
            "proxy.source_archetype_name_proxy.alpha",
            "proxy.source_archetype_name_proxy.beta",
          ],
          "top3_probabilities": [0.21, 0.2],
          "temperature_scaled_probability": 0.19,
          "energy_score": -9.5,
          "msp_score": 0.21,
          "entropy": 1.2,
          "normalized_entropy": 0.9,
          "is_low_confidence": True,
          "embedding_id": "unlabeled-low",
          "embedding_path": None,
          "model_version": "model:v0",
          "taxonomy_eval_version": None,
          "target_source": "source_archetype_name_proxy",
        },
      ]
      pq.write_table(pa.Table.from_pylist(prediction_rows), predictions_path)
      manifest_path = root / "model_predictions.manifest.json"
      manifest_path.write_text(
        json.dumps(
          {
            "model_family": MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
            "model_version": "model:v0",
            "deck_embedding_count": 2,
            "model_artifact": "artifact",
            "dataset_path": str(dataset_path),
          }
        ),
        encoding="utf-8",
      )
      output_path = root / "alias_candidates.json"

      result = run_alias_candidate_scoring(
        None,
        output=output_path,
        predictions=predictions_path,
        predictions_manifest=manifest_path,
        dataset_path=dataset_path,
        min_confusion_count=1,
      )

      self.assertEqual("prediction_backed_alias_candidate_scoring", result["run_id"])
      self.assertEqual(1, result["candidate_count"])
      self.assertEqual(1, result["weak_label_observation_count"])
      self.assertEqual(1, result["unknown_review_candidate_count"])
      self.assertTrue((root / "backfill_report.json").exists())
      self.assertTrue((root / "alias_weak_label_observations.jsonl").exists())
      observations = [
        json.loads(line)
        for line in (root / "alias_weak_label_observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
      ]
      self.assertFalse(observations[0]["usable_for_training_suggestion"])
      self.assertEqual(
        "prediction_backfill_evidence_not_training_target",
        observations[0]["training_exclusion_reason"],
      )
      candidate = result["candidates"][0]
      self.assertEqual("final-test", candidate["split_name"])
      self.assertEqual("Alpha", candidate["label_a"])
      self.assertEqual("Beta", candidate["label_b"])
      self.assertEqual(["final-beta"], candidate["evidence"]["example_deck_ids"])
      report = result["backfill_report"]
      self.assertEqual(2, report["prediction_count"])
      self.assertEqual(2, report["embedding_count"])
      self.assertEqual(1, report["source_unlabeled_count"])
      self.assertEqual(1, report["low_confidence_count"])
      self.assertEqual(
        [{"top1_label": "Alpha", "count": 1}],
        report["top_predicted_labels_for_unlabeled_decks"],
      )

  def test_weak_relation_graph_excludes_final_test_from_suggestions(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      observations_path = root / "alias_weak_label_observations.jsonl"
      observations_path.write_text(
        json.dumps(
          {
            "label_a": "Alpha",
            "label_b": "Beta",
            "label_a_id": "proxy.source_archetype_name_proxy.alpha",
            "label_b_id": "proxy.source_archetype_name_proxy.beta",
            "relation": "alias_candidate",
            "confidence": 0.9,
            "format": "modern",
            "valid_from": "2026-02-01",
            "valid_to": "2026-02-01",
            "rolling_window": "window_a",
            "split_name": "final-test",
            "evidence_sources": ["model_confusion", "deck_overlap"],
            "seed_consensus": 1.0,
            "seed_entropy_normalized": 1.0,
            "usable_for_training_suggestion": False,
            "training_exclusion_reason": "held_out_evaluation_split",
          }
        )
        + "\n",
        encoding="utf-8",
      )

      result = run_weak_relation_graph(observations_path)

      self.assertEqual(1, result["edge_count"])
      self.assertEqual(0, result["soft_canonical_target_suggestion_count"])
      self.assertEqual(
        ["dev-test", "test", "final-test"],
        result["training_evidence_policy"]["excluded_splits"],
      )
      edge = result["edges"][0]
      self.assertEqual("alias_candidate", edge["relation"])
      self.assertEqual(1, edge["held_out_observation_count"])
      self.assertEqual(0, edge["usable_observation_count"])
      suggestions_path = root / "soft_canonical_target_suggestions.jsonl"
      self.assertEqual("", suggestions_path.read_text(encoding="utf-8"))

  def test_weak_relation_graph_emits_usable_soft_target_suggestions(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      observations_path = root / "alias_weak_label_observations.jsonl"
      observations_path.write_text(
        json.dumps(
          {
            "label_a": "Alpha",
            "label_b": "Beta",
            "label_a_id": "proxy.source_archetype_name_proxy.alpha",
            "label_b_id": "proxy.source_archetype_name_proxy.beta",
            "relation": "same_family_candidate",
            "confidence": 0.82,
            "format": "modern",
            "valid_from": "2026-01-15",
            "valid_to": "2026-01-31",
            "rolling_window": "window_a",
            "split_name": "validation",
            "evidence_sources": ["model_confusion", "deck_overlap"],
            "seed_consensus": 0.67,
            "seed_entropy_normalized": 0.8,
            "usable_for_training_suggestion": True,
            "training_exclusion_reason": None,
          }
        )
        + "\n",
        encoding="utf-8",
      )

      result = run_weak_relation_graph(observations_path)

      self.assertEqual(1, result["edge_count"])
      self.assertEqual(1, result["soft_canonical_target_suggestion_count"])
      suggestions_path = root / "soft_canonical_target_suggestions.jsonl"
      suggestions = [
        json.loads(line)
        for line in suggestions_path.read_text(encoding="utf-8").splitlines()
      ]
      self.assertEqual(1, len(suggestions))
      suggestion = suggestions[0]
      self.assertEqual("Beta", suggestion["source_label"])
      target = suggestion["candidate_targets"][0]
      self.assertEqual("Alpha", target["label"])
      self.assertEqual("same_family_candidate", target["relation"])
      self.assertEqual(["validation"], target["splits"])

  def test_weak_target_preview_counts_affected_rows_by_split(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      dataset_path = root / "window"
      dataset_path.mkdir()
      _write_dataset(dataset_path, final_holdout=True)
      rolling_result_path = root / "rolling_evaluation_results.json"
      rolling_result_path.write_text(
        json.dumps(
          {
            "windows": [
              {
                "window": {"name": "window_a"},
                "dataset_path": str(dataset_path),
              }
            ]
          }
        ),
        encoding="utf-8",
      )
      graph_path = root / "weak_relation_graph.json"
      graph_path.write_text(
        json.dumps(
          {
            "edges": [
              {
                "format": "modern",
                "label_a": "Alpha",
                "label_b": "Beta",
                "label_a_id": "proxy.source_archetype_name_proxy.alpha",
                "label_b_id": "proxy.source_archetype_name_proxy.beta",
                "relation": "same_family_candidate",
                "usable_observation_count": 1,
                "held_out_observation_count": 1,
                "time_scopes": [
                  {
                    "rolling_window": "window_a",
                    "split_name": "validation",
                    "valid_from": "2024-01-01",
                    "valid_to": "2024-01-01",
                    "confidence": 0.82,
                    "training_exclusion_reason": None,
                  },
                  {
                    "rolling_window": "window_a",
                    "split_name": "final-test",
                    "valid_from": "2024-01-01",
                    "valid_to": "2024-01-01",
                    "confidence": 0.82,
                    "training_exclusion_reason": "held_out_evaluation_split",
                  },
                ],
              }
            ],
          }
        ),
        encoding="utf-8",
      )
      suggestions_path = root / "soft_canonical_target_suggestions.jsonl"
      suggestions_path.write_text(
        json.dumps(
          {
            "source_label": "Beta",
            "source_label_id": "proxy.source_archetype_name_proxy.beta",
            "candidate_targets": [
              {
                "label": "Alpha",
                "label_id": "proxy.source_archetype_name_proxy.alpha",
                "relation": "same_family_candidate",
                "confidence": 0.82,
                "soft_weight": 0.82,
                "valid_from": "2023-01-01",
                "valid_to": "2024-12-31",
                "format": "modern",
                "splits": ["validation"],
                "rolling_windows": ["window_a"],
              }
            ],
            "usable_for_training": True,
          }
        )
        + "\n",
        encoding="utf-8",
      )

      result = run_weak_target_preview(
        rolling_result_path,
        weak_relation_graph=graph_path,
        soft_target_suggestions=suggestions_path,
      )

      self.assertEqual(1, result["preview_row_count"])
      self.assertEqual(1, result["rows_with_train_coverage_count"])
      row = result["preview_rows"][0]
      self.assertEqual("Beta", row["source_label"])
      self.assertEqual(1, row["train_rows_affected"])
      self.assertEqual(0, row["validation_rows_affected"])
      self.assertEqual(1, row["dev_test_rows_affected"])
      self.assertEqual(1, row["final_test_rows_affected"])
      self.assertTrue(row["has_heldout_evidence"])
      self.assertFalse(row["all_evidence_heldout"])
      self.assertEqual(
        [False, True],
        [
          scope["excluded_because_heldout"]
          for scope in row["evidence_splits"]
        ],
      )

  def test_weak_relation_state_separates_review_deferred_and_blocked(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      observations_path = root / "alias_weak_label_observations.jsonl"
      observations_path.write_text(
        json.dumps({"label_a": "Alpha", "label_b": "Beta"}) + "\n",
        encoding="utf-8",
      )
      graph_path = root / "weak_relation_graph.json"
      graph_path.write_text(
        json.dumps(
          {
            "edges": [
              {
                "format": "modern",
                "label_a": "Alpha",
                "label_b": "Beta",
                "label_a_id": "proxy.source_archetype_name_proxy.alpha",
                "label_b_id": "proxy.source_archetype_name_proxy.beta",
                "relation": "alias_candidate",
                "confidence": {"max": 0.82, "mean": 0.8},
                "valid_from": "2026-01-01",
                "valid_to": "2026-02-01",
                "rolling_windows": ["window_a"],
                "splits": ["validation", "final-test"],
                "usable_observation_count": 1,
                "held_out_observation_count": 1,
                "observation_count": 2,
                "aggregate_confusion_count": 12,
                "time_scopes": [
                  {
                    "rolling_window": "window_a",
                    "split_name": "validation",
                    "valid_from": "2026-01-01",
                    "valid_to": "2026-01-15",
                    "confidence": 0.8,
                    "training_exclusion_reason": None,
                  },
                  {
                    "rolling_window": "window_a",
                    "split_name": "final-test",
                    "valid_from": "2026-01-16",
                    "valid_to": "2026-02-01",
                    "confidence": 0.82,
                    "training_exclusion_reason": "held_out_evaluation_split",
                  },
                ],
              },
              {
                "format": "modern",
                "label_a": "Gamma",
                "label_b": "Delta",
                "label_a_id": "proxy.source_archetype_name_proxy.gamma",
                "label_b_id": "proxy.source_archetype_name_proxy.delta",
                "relation": "model_confusion_candidate",
                "confidence": {"max": 0.5, "mean": 0.5},
                "valid_from": "2026-01-01",
                "valid_to": "2026-02-01",
                "rolling_windows": ["window_a"],
                "splits": ["final-test"],
                "usable_observation_count": 0,
                "held_out_observation_count": 1,
                "observation_count": 1,
                "aggregate_confusion_count": 5,
                "time_scopes": [
                  {
                    "rolling_window": "window_a",
                    "split_name": "final-test",
                    "valid_from": "2026-01-01",
                    "valid_to": "2026-02-01",
                    "confidence": 0.5,
                    "training_exclusion_reason": "held_out_evaluation_split",
                  }
                ],
              },
            ],
          }
        ),
        encoding="utf-8",
      )
      suggestions_path = root / "soft_canonical_target_suggestions.jsonl"
      suggestions_path.write_text(
        json.dumps(
          {
            "source_label": "Beta",
            "source_label_id": "proxy.source_archetype_name_proxy.beta",
            "candidate_targets": [
              {
                "label": "Alpha",
                "label_id": "proxy.source_archetype_name_proxy.alpha",
                "relation": "alias_candidate",
                "confidence": 0.82,
                "soft_weight": 0.82,
                "valid_from": "2026-01-01",
                "valid_to": "2026-02-01",
                "format": "modern",
              }
            ],
          }
        )
        + "\n",
        encoding="utf-8",
      )
      preview_path = root / "weak_target_preview.json"
      preview_path.write_text(
        json.dumps(
          {
            "preview_rows": [
              {
                "source_label": "Beta",
                "source_label_id": "proxy.source_archetype_name_proxy.beta",
                "candidate_canonical_labels": [
                  {
                    "label": "Alpha",
                    "label_id": "proxy.source_archetype_name_proxy.alpha",
                    "relation": "alias_candidate",
                    "confidence": 0.82,
                  }
                ],
                "format": "modern",
                "valid_from": "2026-01-01",
                "valid_to": "2026-02-01",
                "train_rows_affected": 0,
                "validation_rows_affected": 3,
                "affected_rows_by_split": {
                  "train": 0,
                  "validation": 3,
                  "dev-test": 0,
                  "test": 0,
                  "final-test": 2,
                },
              }
            ],
          }
        ),
        encoding="utf-8",
      )

      result = run_weak_relation_state(
        alias_observations=observations_path,
        weak_relation_graph=graph_path,
        soft_target_suggestions=suggestions_path,
        weak_target_preview=preview_path,
      )

      self.assertEqual(0, result["counts"]["usable_now"])
      self.assertEqual(1, result["counts"]["review_queue"])
      self.assertEqual(1, result["counts"]["deferred_soft_canonical_evidence"])
      self.assertEqual(1, result["counts"]["deferred_review_diagnostics"])
      self.assertEqual(1, result["counts"]["blocked"])
      self.assertEqual(
        {"alias_candidate": 1},
        result["bucket_summaries"]["deferred_soft_canonical_evidence"][
          "by_relation"
        ],
      )
      self.assertEqual(
        {"model_confusion_candidate": 1},
        result["bucket_summaries"]["deferred_review_diagnostics"][
          "by_relation"
        ],
      )
      self.assertEqual(1, len(result["top_review_pairs"]))
      self.assertEqual(
        "review_or_defer_to_next_window",
        result["review_queue"][0]["action"],
      )
      self.assertEqual(
        "defer_as_future_soft_canonical_evidence",
        result["deferred_soft_canonical_evidence"][0]["action"],
      )
      self.assertTrue(
        result["deferred_soft_canonical_evidence"][0][
          "deferred_training_eligible_if_future"
        ],
      )
      self.assertEqual(
        "defer_as_future_review_diagnostic",
        result["deferred_review_diagnostics"][0]["action"],
      )
      self.assertFalse(
        result["deferred_review_diagnostics"][0][
          "deferred_training_eligible_if_future"
        ],
      )
      self.assertIn(
        "not_a_soft_canonical_relation",
        result["blocked"][0]["blocked_reasons"],
      )


def _write_dataset(dataset_path: Path, *, final_holdout: bool = False) -> None:
  artifacts = {
    "card_vocab": "card_vocab.parquet",
    "deck_tokens": "deck_tokens.parquet",
    "zone_vocab": "zone_vocab.json",
    "split_manifest": "split_manifest.parquet",
    "proxy_targets": "proxy_targets.parquet",
  }
  (dataset_path / "dataset_manifest.json").write_text(
    json.dumps(
      {
        "dataset_version": "modern_2024_2024_v0",
        "artifacts": artifacts,
      }
    )
  )
  write_parquet(
    dataset_path / artifacts["card_vocab"],
    [
      _card_vocab(0),
      _card_vocab(1),
      _card_vocab(2),
    ],
    "card_vocab",
  )
  (dataset_path / artifacts["zone_vocab"]).write_text(
    json.dumps(
      {
        "main": 0,
        "side": 1,
        "companion": 2,
        "commander": 3,
        "other": 4,
      }
    )
  )
  deck_tokens = [
      _token("train-a", 0, 4),
      _token("train-a", 1, 2),
      _token("train-b", 1, 4),
      _token("train-b", 2, 2),
      _token("validation-a", 0, 4),
      _token("validation-a", 1, 2),
      _token("validation-gamma", 1, 4),
      _token("validation-gamma", 2, 2),
  ]
  if final_holdout:
    deck_tokens.extend(
      [
        _token("dev-test-b", 1, 4),
        _token("dev-test-b", 2, 2),
        _token("final-test-b", 1, 4),
        _token("final-test-b", 2, 2),
      ]
    )
  else:
    deck_tokens.extend(
      [
        _token("test-b", 1, 4),
        _token("test-b", 2, 2),
      ]
    )
  write_parquet(
    dataset_path / artifacts["deck_tokens"],
    deck_tokens,
    "deck_tokens",
  )
  split_manifest = [
      _split("train-a", "train"),
      _split("train-b", "train"),
      _split("validation-a", "validation"),
      _split("validation-gamma", "validation"),
  ]
  if final_holdout:
    split_manifest.extend(
      [
        _split("dev-test-b", "dev-test"),
        _split("final-test-b", "final-test"),
      ]
    )
  else:
    split_manifest.append(_split("test-b", "test"))
  write_parquet(
    dataset_path / artifacts["split_manifest"],
    split_manifest,
    "split_manifest",
  )
  proxy_targets = [
      _proxy_target("train-a", "alpha"),
      _proxy_target("train-b", "beta"),
      _proxy_target("validation-a", "alpha"),
      _proxy_target("validation-gamma", "gamma"),
  ]
  if final_holdout:
    proxy_targets.extend(
      [
        _proxy_target("dev-test-b", "beta"),
        _proxy_target("final-test-b", "beta"),
      ]
    )
  else:
    proxy_targets.append(_proxy_target("test-b", "beta"))
  write_parquet(
    dataset_path / artifacts["proxy_targets"],
    proxy_targets,
    "proxy_targets",
  )


def _rewrite_dataset_with_local_card_indexes(dataset_path: Path) -> None:
  manifest = json.loads((dataset_path / "dataset_manifest.json").read_text())
  artifacts = manifest["artifacts"]
  card_vocab_path = dataset_path / artifacts["card_vocab"]
  deck_tokens_path = dataset_path / artifacts["deck_tokens"]
  card_vocab = pq.read_table(card_vocab_path).to_pylist()
  reordered_vocab = [card_vocab[1], card_vocab[0], card_vocab[2]]
  oracle_to_local_idx: dict[str, int] = {}
  remapped_vocab: list[dict[str, object]] = []
  for local_idx, card in enumerate(reordered_vocab):
    remapped_card = dict(card)
    remapped_card["card_idx"] = local_idx
    oracle_to_local_idx[str(remapped_card["oracle_id"])] = local_idx
    remapped_vocab.append(remapped_card)
  write_parquet(card_vocab_path, remapped_vocab, "card_vocab")

  remapped_tokens: list[dict[str, object]] = []
  for token in pq.read_table(deck_tokens_path).to_pylist():
    remapped_token = dict(token)
    remapped_token["card_idx"] = oracle_to_local_idx[str(token["oracle_id"])]
    remapped_tokens.append(remapped_token)
  write_parquet(deck_tokens_path, remapped_tokens, "deck_tokens")


def _remove_proxy_targets_artifact(dataset_path: Path) -> None:
  manifest_path = dataset_path / "dataset_manifest.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  proxy_targets = manifest["artifacts"].pop("proxy_targets")
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
  (dataset_path / proxy_targets).unlink()


def _write_alias_candidate_dataset(dataset_path: Path) -> None:
  artifacts = {
    "card_vocab": "card_vocab.parquet",
    "deck_tokens": "deck_tokens.parquet",
    "zone_vocab": "zone_vocab.json",
    "split_manifest": "split_manifest.parquet",
    "proxy_targets": "proxy_targets.parquet",
  }
  (dataset_path / "dataset_manifest.json").write_text(
    json.dumps(
      {
        "dataset_version": "modern_alias_candidate_test",
        "artifacts": artifacts,
      }
    )
  )
  write_parquet(
    dataset_path / artifacts["card_vocab"],
    [
      _card_vocab(0),
      _card_vocab(1),
      _card_vocab(2),
    ],
    "card_vocab",
  )
  (dataset_path / artifacts["zone_vocab"]).write_text(
    json.dumps(
      {
        "main": 0,
        "side": 1,
        "companion": 2,
        "commander": 3,
        "other": 4,
      }
    )
  )
  write_parquet(
    dataset_path / artifacts["deck_tokens"],
    [
      _token("final-alpha", 0, 4),
      _token("final-alpha", 1, 4),
      _token("final-beta", 0, 4),
      _token("final-beta", 1, 4),
      _token("final-beta", 2, 2),
    ],
    "deck_tokens",
  )
  write_parquet(
    dataset_path / artifacts["split_manifest"],
    [
      _split("final-alpha", "final-test"),
      _split("final-beta", "final-test"),
    ],
    "split_manifest",
  )
  write_parquet(
    dataset_path / artifacts["proxy_targets"],
    [
      _proxy_target("final-alpha", "alpha"),
      _proxy_target("final-beta", "beta"),
    ],
    "proxy_targets",
  )


def _token(deck_id: str, card_idx: int, quantity: int) -> dict[str, object]:
  return {
    "dataset_version": "modern_2024_2024_v0",
    "deck_id": deck_id,
    "card_idx": card_idx,
    "oracle_id": f"oracle-{card_idx}",
    "quantity": quantity,
    "zone_idx": 0,
    "zone": "main",
  }


def _card_vocab(card_idx: int) -> dict[str, object]:
  return {
    "dataset_version": "modern_2024_2024_v0",
    "card_idx": card_idx,
    "oracle_id": f"oracle-{card_idx}",
    "primary_name": f"Card {card_idx}",
    "first_seen_at": None,
    "last_seen_at": None,
  }


def _split(deck_id: str, split_name: str) -> dict[str, object]:
  return {
    "dataset_version": "modern_2024_2024_v0",
    "deck_id": deck_id,
    "event_id": f"event-{deck_id}",
    "event_date": date(2024, 1, 1),
    "format": "modern",
    "source": "mtgo-db",
    "split_name": split_name,
    "split_strategy": "event_forward_modern_2024_2024_v0",
  }


def _proxy_target(deck_id: str, label: str) -> dict[str, object]:
  return {
    "dataset_version": "modern_2024_2024_v0",
    "deck_id": deck_id,
    "target_source": "source_archetype_name_proxy",
    "target_level": "family",
    "proxy_label_id": f"proxy.source_archetype_name_proxy.{label}",
    "display_label": label.title(),
    "normalized_label": label,
    "source_field": "source_archetype_name",
    "confidence": 0.35,
    "provenance": "mtgo-db:archetypes.archetype|proxy_target",
  }


def _prediction(
  deck_id: str,
  split_name: str,
  actual_label_id: str,
  predicted_label_id: str,
  score: float,
  actual_label_train_support: int,
  *,
  event_date: str = "2026-02-01",
  format_code: str = "modern",
) -> dict[str, object]:
  top_label_ids = [predicted_label_id]
  return {
    "deck_id": deck_id,
    "event_id": f"event-{deck_id}",
    "event_date": event_date,
    "format": format_code,
    "split_name": split_name,
    "target_source": "source_archetype_name_proxy",
    "actual_label_id": actual_label_id,
    "actual_label_train_support": actual_label_train_support,
    "is_train_unseen_label": actual_label_train_support == 0,
    "predicted_label_id": predicted_label_id,
    "score": score,
    "max_softmax_probability": score,
    "temperature_scaled_max_softmax_probability": score,
    "entropy": 1.0 - score,
    "normalized_entropy": 1.0 - score,
    "energy": 1.0 - score,
    "top_label_ids": top_label_ids,
    "top_scores": [score],
    "is_correct": actual_label_id == predicted_label_id,
    "is_top_3_correct": actual_label_id in top_label_ids,
  }


if __name__ == "__main__":
  unittest.main()

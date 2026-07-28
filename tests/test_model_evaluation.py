from __future__ import annotations

import unittest
from datetime import date

from manafold.models import train as model_training
from manafold.models.metrics import classification_metrics
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
from tests.model_test_support import _prediction


class ModelEvaluationTests(unittest.TestCase):
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
          model_training.MODEL_POOLED_LINEAR: {
            "multi_seed_summary": {
              "primary_accuracy": {"mean": 0.6},
              "primary_closed_set_accuracy": {"mean": 0.7},
            },
          },
        },
      },
      {
        "models": {
          model_training.MODEL_POOLED_LINEAR: {
            "multi_seed_summary": {
              "primary_accuracy": {"mean": 0.8},
              "primary_closed_set_accuracy": {"mean": 0.9},
            },
          },
        },
      },
    ])

    metrics = summary[model_training.MODEL_POOLED_LINEAR]["metrics"]
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
          model_training.MODEL_POOLED_LINEAR: {
            "multi_seed_summary": {
              "primary_accuracy": {"mean": 0.6},
              "primary_temperature_scaled_nll": {"mean": 0.8},
            },
          },
          model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED: {
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
          model_training.MODEL_POOLED_LINEAR: {
            "multi_seed_summary": {
              "primary_accuracy": {"mean": 0.8},
              "primary_temperature_scaled_nll": {"mean": 0.6},
            },
          },
          model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED: {
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
      model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
      accuracy_ranking["windows"][0]["ranking"][0]["model"],
    )
    self.assertEqual(
      model_training.MODEL_POOLED_LINEAR,
      accuracy_ranking["windows"][1]["ranking"][0]["model"],
    )
    self.assertEqual(
      1,
      diagnostics["vs_pooled_linear"][
        model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED
      ]["primary_accuracy"]["wins"],
    )
    self.assertEqual(
      1,
      diagnostics["vs_pooled_linear"][
        model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED
      ]["primary_accuracy"]["losses"],
    )
    self.assertEqual(
      1,
      diagnostics["vs_pooled_linear"][
        model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED
      ]["primary_temperature_scaled_nll"]["wins"],
    )

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
    canonical_label = label_id_for("Example Engine", target_source)
    alias_label = label_id_for("Legacy Engine Name", target_source)
    config = TaxonomyEvaluationConfig(
      path=None,
      aliases=(
        TaxonomyAliasRule(
          source_label="Legacy Engine Name",
          canonical_family="Example Engine",
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
        "train-canonical",
        "train",
        canonical_label,
        canonical_label,
        0.9,
        1,
        event_date="2025-12-01",
      ),
      _prediction(
        "final-alias",
        "final-test",
        canonical_label,
        alias_label,
        0.8,
        1,
        event_date="2026-02-01",
      ),
      _prediction(
        "final-out-of-window",
        "final-test",
        canonical_label,
        alias_label,
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
    self.assertEqual("Example Engine", report_row["actual_source_label"])
    self.assertEqual("Legacy Engine Name", report_row["predicted_source_label"])
    self.assertEqual("Example Engine", report_row["canonical_actual"])
    self.assertEqual("Example Engine", report_row["canonical_predicted"])
    self.assertEqual({"source": "test"}, report_row["evidence"])



if __name__ == "__main__":
  unittest.main()

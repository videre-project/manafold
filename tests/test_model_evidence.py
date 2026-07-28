from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from manafold.models import train as model_training
from manafold.models.aliases import run_alias_candidate_scoring
from manafold.models.weak_relations import run_weak_relation_graph
from manafold.models.weak_state import run_weak_relation_state
from manafold.models.weak_targets import run_weak_target_preview
from tests.model_test_support import _write_alias_candidate_dataset, _write_dataset


class ModelEvidenceTests(unittest.TestCase):
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
              model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED: {
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
            "model_family": model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
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



if __name__ == "__main__":
  unittest.main()

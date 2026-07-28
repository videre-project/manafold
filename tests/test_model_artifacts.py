from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from manafold.models import train as model_training
from manafold.models.data import load_training_dataset
from manafold.models.model_artifacts import run_model_scoring
from tests.model_test_support import (
  _remove_proxy_targets_artifact,
  _rewrite_dataset_with_local_card_indexes,
  _write_dataset,
)


class ModelArtifactsTests(unittest.TestCase):
  def test_model_artifact_scores_dataset(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir) / "dataset"
      dataset_path.mkdir()
      _write_dataset(dataset_path)
      artifact_dir = Path(temp_dir) / "model_artifact"
      training_output = dataset_path / "model_runs" / "training_results.json"

      result = model_training.run_model_training(
        dataset_path,
        output=training_output,
        model_names=(model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,),
        epochs=3,
        batch_size=2,
        device="cpu",
        model_artifact_output=artifact_dir,
        prediction_output=model_training.PREDICTION_OUTPUT_SUMMARY,
      )

      model_result = result["models"][model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED]
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
      model_training.run_model_training(
        dataset_path,
        output=pooled_training_output,
        model_names=(model_training.MODEL_POOLED_LINEAR,),
        epochs=1,
        batch_size=2,
        device="cpu",
        model_artifact_output=pooled_artifact_dir,
        model_artifact_model_name=model_training.MODEL_POOLED_LINEAR,
        prediction_output=model_training.PREDICTION_OUTPUT_SUMMARY,
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
        model_training.run_model_training(
          dataset_path,
          model_names=(model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,),
          seeds=(13, 17),
          epochs=1,
          batch_size=2,
          device="cpu",
          model_artifact_output=Path(temp_dir) / "model_artifact",
          prediction_output=model_training.PREDICTION_OUTPUT_SUMMARY,
        )



if __name__ == "__main__":
  unittest.main()

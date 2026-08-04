from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from manafold.models.training import training_pipeline
from scripts import export_onnx
from tests.model_test_support import _write_dataset


class OnnxExportTests(unittest.TestCase):
  def test_a11_saved_model_exports_with_generated_family_relations(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      dataset_path = root / "dataset"
      saved_model = root / "saved_model"
      output = root / "onnx"
      dataset_path.mkdir()
      _write_dataset(dataset_path)

      training_pipeline.train_models(
        dataset_path,
        output=root / "training_results.json",
        model_names=(training_pipeline.MODEL_A11,),
        epochs=1,
        batch_size=2,
        embedding_dim=4,
        hidden_dim=8,
        attention_heads=2,
        attention_layers=1,
        device="cpu",
        saved_model_output=saved_model,
        saved_model_name=training_pipeline.MODEL_A11,
        prediction_output=training_pipeline.PREDICTION_OUTPUT_SUMMARY,
      )

      status = export_onnx.main([
        "--saved-model", str(saved_model),
        "--ranking-dataset", str(dataset_path),
        "--output-dir", str(output),
        "--format", "modern",
        "--sample-token-count", "4",
      ])

      self.assertEqual(0, status)
      self.assertTrue((output / "model.onnx").is_file())
      manifest = json.loads(
        (output / "worker_manifest.json").read_text(encoding="utf-8")
      )
      self.assertEqual("manafold-a11", manifest["serving"]["model"])
      self.assertEqual(
        "family_canonical_proxy",
        manifest["target_source"],
      )
      family_vocab = json.loads(
        (output / "family_vocab.json").read_text(encoding="utf-8")
      )
      self.assertEqual(
        "family_relations.json",
        family_vocab["relation_source"]["filename"],
      )
      ranking = json.loads(
        (output / "family_card_ranking.json").read_text(encoding="utf-8")
      )
      self.assertGreater(ranking["training_deck_count"], 0)


if __name__ == "__main__":
  unittest.main()

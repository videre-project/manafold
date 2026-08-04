from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq
import torch

from manafold.models.training import training_pipeline as model_training
from manafold.models.classifiers import (
  SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
  SetTransformerClassifier,
)
from manafold.datasets.model_inputs import load_training_dataset
from manafold.models.dataset_scoring import score_dataset
from manafold.models.evaluation.family_classification import (
  evaluate_family_classification,
)
from manafold.models.saved_model import load_saved_model, save_trained_model
from manafold.serialization import sha256_file
from tests.model_test_support import (
  _remove_proxy_targets_file,
  _rewrite_dataset_with_local_card_indexes,
  _write_dataset,
)


class SavedModelTests(unittest.TestCase):
  def test_production_refit_uses_all_splits_and_evaluation_calibration(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      dataset_path = root / "dataset"
      dataset_path.mkdir()
      _write_dataset(dataset_path)
      evaluation_model = root / "evaluation_model"

      model_training.train_models(
        dataset_path,
        output=root / "evaluation_results.json",
        model_names=(model_training.MODEL_A11,),
        epochs=1,
        batch_size=2,
        embedding_dim=4,
        hidden_dim=8,
        attention_heads=2,
        attention_layers=1,
        device="cpu",
        saved_model_output=evaluation_model,
        saved_model_name=model_training.MODEL_A11,
        prediction_output=model_training.PREDICTION_OUTPUT_SUMMARY,
      )

      production_model = root / "production_model"
      result = model_training.train_models(
        dataset_path,
        output=root / "production_results.json",
        model_names=(model_training.MODEL_A11,),
        epochs=1,
        batch_size=2,
        embedding_dim=4,
        hidden_dim=8,
        attention_heads=2,
        attention_layers=1,
        device="cpu",
        saved_model_output=production_model,
        saved_model_name=model_training.MODEL_A11,
        prediction_output=model_training.PREDICTION_OUTPUT_SUMMARY,
        production_refit=True,
        calibration_model=evaluation_model,
      )

      self.assertEqual("completed", result["status"])
      self.assertEqual("all_available_splits", result["fit_scope"]["mode"])
      self.assertEqual(5, result["fit_scope"]["fit_example_count"])
      self.assertEqual(
        {"train": 2, "validation": 2, "test": 1},
        result["fit_scope"]["source_split_counts"],
      )
      self.assertEqual("all_available_splits", result["label_vocabulary"])
      ontology = json.loads(
        (production_model / "family_relations.json").read_text(encoding="utf-8")
      )
      self.assertEqual(5, ontology["dataset_summary"]["training_examples"])
      self.assertEqual(3, ontology["dataset_summary"]["observed_training_labels"])
      self.assertEqual(
        "generated_from_all_available_examples",
        result["training_target"]["auto_ontology"]["generation_mode"],
      )
      temperature = json.loads(
        (production_model / "temperature.json").read_text(encoding="utf-8")
      )
      self.assertEqual("validation", temperature["selection_split"])
      self.assertEqual(
        sha256_file(evaluation_model / "temperature.json"),
        temperature["source_temperature_sha256"],
      )
      training_manifest = json.loads(
        (production_model / "training_manifest.json").read_text(encoding="utf-8")
      )
      self.assertEqual(
        "all_available_splits",
        training_manifest["fit_scope"]["mode"],
      )

  def test_a8_saved_model_restores_anchor_and_product_streams(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir) / "dataset"
      dataset_path.mkdir()
      _write_dataset(dataset_path)
      saved_model_dir = Path(temp_dir) / "a8_saved_model"
      family_relations = Path(temp_dir) / "family_targets.json"
      family_relations.write_text(
        '{"proposal_policy": {"method": "test"}}\n',
        encoding="utf-8",
      )
      dataset = load_training_dataset(dataset_path)
      model = SetTransformerClassifier(
        labels=dataset.labels,
        card_count=dataset.card_count,
        zone_count=dataset.zone_count,
        quantity_count=5,
        embedding_dim=4,
        hidden_dim=8,
        attention_heads=2,
        attention_layers=1,
        pooling=SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
        partial_observation_training=True,
        partial_contextual_weight=0.5,
        supcon_loss_weight=0.05,
        anchor_gating=True,
        use_product_manifold=True,
        product_manifold_tree_weight=0.10,
        causal_target_alpha=0.10,
        seed=13,
        device="cpu",
      )
      model.fit(
        [example for example in dataset.examples if example.split_name == "train"],
        epochs=1,
        batch_size=2,
      )
      logits_before_saving = model.logits_many(dataset.examples)
      saved = save_trained_model(
        saved_model_dir,
        model_name=model_training.MODEL_SET_TRANSFORMER_A8,
        model=model,
        dataset=dataset,
        model_result={
          "seed": 13,
          "model_config": {
            "pooling": SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
            "token_scope": "mainboard",
            "embedding_dim": 4,
            "hidden_dim": 8,
            "attention_heads": 2,
            "attention_layers": 1,
            "a8_objectives": model.a8_objective_config(),
          },
        },
        training_result={"run_id": "a8-saved-model-test"},
        seed=13,
        family_relations=family_relations,
      )
      copied_relations = saved_model_dir / "family_relations.json"
      self.assertEqual(family_relations.read_bytes(), copied_relations.read_bytes())
      self.assertEqual(
        str(copied_relations),
        saved["files"]["family_relations"],
      )
      saved_model = load_saved_model(saved_model_dir, device="cpu")
      objectives = saved_model["model_config"]["a8_objectives"]
      state = saved_model["model"].state_dict_for_saving()
      logits_after_loading = saved_model["model"].logits_many(dataset.examples)

      self.assertTrue(objectives["anchor_gating"])
      self.assertTrue(objectives["product_manifold"])
      self.assertEqual(0.10, objectives["causal_target_alpha"])
      self.assertIn("anchor_gate.anchor_scores", state)
      self.assertIn("product_manifold.proj_hyp.weight", state)
      self.assertGreater(float(state["anchor_gate.anchor_scores"].max()), 0.0)
      self.assertTrue(
        torch.allclose(logits_before_saving, logits_after_loading, atol=1e-7)
      )

  def test_saved_model_scores_dataset(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir) / "dataset"
      dataset_path.mkdir()
      _write_dataset(dataset_path)
      saved_model_dir = Path(temp_dir) / "saved_model"
      training_output = dataset_path / "model_runs" / "training_results.json"

      result = model_training.train_models(
        dataset_path,
        output=training_output,
        model_names=(model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,),
        epochs=3,
        batch_size=2,
        device="cpu",
        saved_model_output=saved_model_dir,
        prediction_output=model_training.PREDICTION_OUTPUT_SUMMARY,
      )

      model_result = result["models"][model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED]
      self.assertIn("saved_model", model_result)
      for filename in (
        "model.pt",
        "model_config.json",
        "card_vocab.parquet",
        "label_vocab.json",
        "zone_vocab.json",
        "temperature.json",
        "training_manifest.json",
      ):
        self.assertTrue((saved_model_dir / filename).exists())
      label_vocab = json.loads(
        (saved_model_dir / "label_vocab.json").read_text(encoding="utf-8")
      )
      self.assertIn("entries", label_vocab)
      self.assertEqual("Alpha", label_vocab["entries"][0]["display_label"])

      output = Path(temp_dir) / "model_predictions.parquet"
      score_result = score_dataset(
        saved_model_dir=saved_model_dir,
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
      remapped_score_result = score_dataset(
        saved_model_dir=saved_model_dir,
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
      rows_by_deck = {row["deck_id"]: row for row in rows}
      remapped_rows_by_deck = {row["deck_id"]: row for row in remapped_rows}
      self.assertEqual(rows_by_deck.keys(), remapped_rows_by_deck.keys())
      for deck_id, baseline in rows_by_deck.items():
        remapped = remapped_rows_by_deck[deck_id]
        self.assertEqual(baseline["top1_label_id"], remapped["top1_label_id"])
        self.assertEqual(baseline["top3_label_ids"], remapped["top3_label_ids"])
        self.assertAlmostEqual(
          baseline["temperature_scaled_probability"],
          remapped["temperature_scaled_probability"],
          places=7,
        )
        self.assertAlmostEqual(
          baseline["energy_score"],
          remapped["energy_score"],
          places=7,
        )

      unlabeled_dataset_path = Path(temp_dir) / "unlabeled_dataset"
      unlabeled_dataset_path.mkdir()
      _write_dataset(unlabeled_dataset_path)
      _remove_proxy_targets_file(unlabeled_dataset_path)
      unlabeled_output = Path(temp_dir) / "unlabeled_predictions.parquet"
      unlabeled_score_result = score_dataset(
        saved_model_dir=saved_model_dir,
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

      pooled_saved_model_dir = Path(temp_dir) / "pooled_saved_model"
      pooled_training_output = (
        dataset_path / "model_runs" / "pooled_training_results.json"
      )
      model_training.train_models(
        dataset_path,
        output=pooled_training_output,
        model_names=(model_training.MODEL_POOLED_LINEAR,),
        epochs=1,
        batch_size=2,
        device="cpu",
        saved_model_output=pooled_saved_model_dir,
        saved_model_name=model_training.MODEL_POOLED_LINEAR,
        prediction_output=model_training.PREDICTION_OUTPUT_SUMMARY,
      )
      pooled_output = Path(temp_dir) / "pooled_predictions.parquet"
      pooled_score_result = score_dataset(
        saved_model_dir=pooled_saved_model_dir,
        dataset_path=dataset_path,
        output=pooled_output,
        batch_size=2,
        device="cpu",
      )
      pooled_rows = pq.read_table(pooled_output).to_pylist()
      self.assertEqual(len(rows), len(pooled_rows))
      self.assertIsNone(pooled_score_result["deck_embedding_output"])

  def test_a11_trains_saves_scores_and_evaluates_partial_decks(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      dataset_path = root / "dataset"
      dataset_path.mkdir()
      _write_dataset(dataset_path)
      training_output = root / "run" / "training_results.json"
      training_output.parent.mkdir()
      family_relations = training_output.parent / "family_relations.json"
      family_relations.write_text('{"stale": true}\n', encoding="utf-8")
      saved_model_dir = root / "a11_saved_model"

      result = model_training.train_models(
        dataset_path,
        output=training_output,
        model_names=(model_training.MODEL_A11,),
        epochs=1,
        batch_size=2,
        embedding_dim=4,
        hidden_dim=8,
        attention_heads=2,
        attention_layers=1,
        device="cpu",
        saved_model_output=saved_model_dir,
        saved_model_name=model_training.MODEL_A11,
        prediction_output=model_training.PREDICTION_OUTPUT_SUMMARY,
      )

      model_result = result["models"][model_training.MODEL_SET_TRANSFORMER_A11]
      self.assertEqual("completed", result["status"])
      self.assertEqual("family_canonical_proxy", result["target_source"])
      self.assertEqual(
        "auto-ontology-family",
        result["training_target"]["label_level"],
      )
      self.assertIn("saved_model", model_result)
      packaged_relations = saved_model_dir / "family_relations.json"
      self.assertTrue(packaged_relations.exists())
      self.assertEqual(
        family_relations.read_bytes(),
        packaged_relations.read_bytes(),
      )
      ontology = json.loads(family_relations.read_text(encoding="utf-8"))
      self.assertNotIn("stale", ontology)
      self.assertEqual(
        "training_split_core_card_jaccard",
        ontology["proposal_policy"]["method"],
      )
      provenance = ontology["dataset_provenance"]
      self.assertEqual("train", provenance["induction_split"])
      self.assertEqual(64, len(provenance["training_split_sha256"]))
      training_target = result["training_target"]["auto_ontology"]
      self.assertEqual(
        "generated_from_training_split",
        training_target["generation_mode"],
      )
      self.assertEqual(
        sha256_file(family_relations),
        training_target["file_sha256"],
      )
      training_manifest = json.loads(
        (saved_model_dir / "training_manifest.json").read_text(encoding="utf-8")
      )
      self.assertEqual(
        sha256_file(packaged_relations),
        training_manifest["family_relations"]["sha256"],
      )

      score_output = root / "a11_predictions.parquet"
      score_result = score_dataset(
        saved_model_dir=saved_model_dir,
        dataset_path=dataset_path,
        output=score_output,
        batch_size=2,
        device="cpu",
      )
      score_rows = pq.read_table(score_output).to_pylist()
      self.assertEqual(len(load_training_dataset(dataset_path).examples), len(score_rows))
      self.assertEqual(len(score_rows), score_result["prediction_count"])
      self.assertTrue(all(row["top1_label_id"] for row in score_rows))
      self.assertTrue(all(row["energy_score"] is not None for row in score_rows))

      complete_report = evaluate_family_classification(
        saved_model_dir=saved_model_dir,
        dataset_path=dataset_path,
        output=root / "a11_complete_evaluation.json",
        split_name="test",
        batch_size=2,
        device="cpu",
      )
      partial_report = evaluate_family_classification(
        saved_model_dir=saved_model_dir,
        dataset_path=dataset_path,
        output=root / "a11_partial_evaluation.json",
        split_name="test",
        partial_identity_count=1,
        partial_seed=13,
        batch_size=2,
        device="cpu",
      )

      self.assertEqual(1, complete_report["metrics"]["count"])
      self.assertEqual(1, partial_report["metrics"]["count"])
      self.assertIsNone(complete_report["observation"]["identity_count"])
      self.assertEqual(1, partial_report["observation"]["identity_count"])
      self.assertEqual(
        complete_report["model_version"],
        partial_report["model_version"],
      )

  def test_saved_model_export_requires_explicit_seed_policy(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      dataset_path = Path(temp_dir) / "dataset"
      dataset_path.mkdir()
      _write_dataset(dataset_path)

      with self.assertRaisesRegex(ValueError, "seed policy 'single'"):
        model_training.train_models(
          dataset_path,
          model_names=(model_training.MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,),
          seeds=(13, 17),
          epochs=1,
          batch_size=2,
          device="cpu",
          saved_model_output=Path(temp_dir) / "saved_model",
          prediction_output=model_training.PREDICTION_OUTPUT_SUMMARY,
        )



if __name__ == "__main__":
  unittest.main()

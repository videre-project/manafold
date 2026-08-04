from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from manafold.cli import main


class CliTests(unittest.TestCase):
  def test_model_train_routes_production_refit(self) -> None:
    with (
      patch.object(
        sys,
        "argv",
        [
          "manafold",
          "model-train",
          "dataset",
          "--model",
          "a11",
          "--saved-model-output",
          "release/model",
          "--saved-model-name",
          "a11",
          "--production-refit",
          "--calibration-model",
          "release/evaluation_model",
        ],
      ),
      patch(
        "manafold.cli.commands.train_models.PROJECT_ROOT",
        Path("/repo"),
      ),
      patch("manafold.cli.common.PROJECT_ROOT", Path("/repo")),
      patch(
        "manafold.cli.commands.train_models.train_models",
        return_value={"status": "skipped"},
      ) as train,
    ):
      main()

    options = train.call_args.kwargs
    self.assertTrue(options["production_refit"])
    self.assertEqual(
      Path("/repo/release/evaluation_model"),
      options["calibration_model"],
    )

  def test_model_train_routes_a11_ontology_generation_options(self) -> None:
    with (
      patch.object(
        sys,
        "argv",
        [
          "manafold",
          "model-train",
          "dataset",
          "--model",
          "a11",
          "--auto-ontology-output",
          "generated/family_relations.json",
          "--auto-ontology-min-jaccard-threshold",
          "0.6",
          "--auto-ontology-core-card-frequency",
          "0.3",
        ],
      ),
      patch(
        "manafold.cli.commands.train_models.PROJECT_ROOT",
        Path("/repo"),
      ),
      patch("manafold.cli.common.PROJECT_ROOT", Path("/repo")),
      patch(
        "manafold.cli.commands.train_models.train_models",
        return_value={"status": "skipped"},
      ) as train,
    ):
      main()

    dataset_path, = train.call_args.args
    options = train.call_args.kwargs
    self.assertEqual(Path("/repo/dataset"), dataset_path)
    self.assertEqual(("a11",), options["model_names"])
    self.assertEqual(
      Path("/repo/generated/family_relations.json"),
      options["auto_ontology_output"],
    )
    self.assertEqual(0.6, options["auto_ontology_min_jaccard_threshold"])
    self.assertEqual(0.3, options["auto_ontology_core_card_frequency"])
    self.assertNotIn("family_relations", options)

  def test_family_targets_routes_paths_and_thresholds(self) -> None:
    result = {
      "proposed_components": [{"component_id": "one"}],
      "dataset_summary": {
        "training_examples": 10,
        "observed_training_labels": 2,
      },
    }
    with (
      patch.object(
        sys,
        "argv",
        [
          "manafold",
          "family-targets",
          "dataset",
          "--output",
          "targets.json",
          "--min-jaccard-threshold",
          "0.6",
          "--core-card-frequency",
          "0.3",
        ],
      ),
      patch(
        "manafold.cli.commands.family_targets.PROJECT_ROOT",
        Path("/repo"),
      ),
      patch(
        "manafold.cli.commands.family_targets.generate_auto_ontology",
        return_value=result,
      ) as generate,
    ):
      main()

    generate.assert_called_once_with(
      Path("/repo/dataset"),
      Path("/repo/targets.json"),
      min_jaccard_threshold=0.6,
      core_card_frequency=0.3,
    )

  def test_family_eval_routes_a11_inputs(self) -> None:
    result = {
      "metrics": {
        "accuracy": 0.8,
        "macro_f1": 0.7,
        "top3_accuracy": 0.9,
      },
    }
    with (
      patch.object(
        sys,
        "argv",
        [
          "manafold",
          "family-eval",
          "--saved-model",
          "model",
          "--dataset",
          "dataset",
          "--family-relations",
          "targets.json",
          "--output",
          "evaluation.json",
          "--partial-identity-count",
          "5",
        ],
      ),
      patch("manafold.cli.common.PROJECT_ROOT", Path("/repo")),
      patch(
        "manafold.cli.commands.evaluate_families.evaluate_family_classification",
        return_value=result,
      ) as evaluate,
    ):
      main()

    evaluate.assert_called_once_with(
      saved_model_dir=Path("/repo/model"),
      dataset_path=Path("/repo/dataset"),
      output=Path("/repo/evaluation.json"),
      family_relations=Path("/repo/targets.json"),
      split_name="final-test",
      partial_identity_count=5,
      partial_seed=13,
      batch_size=1024,
      device="auto",
    )


if __name__ == "__main__":
  unittest.main()

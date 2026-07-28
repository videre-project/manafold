from __future__ import annotations

import unittest

import torch

from manafold.models.family_backoff import (
  aggregate_family_probabilities,
  build_family_mapping,
  build_family_vocab,
  extract_proposed_edges,
  relaxed_archetype_name,
)
from manafold.models.family_evaluation import _classification_metrics


class FamilyBackoffTests(unittest.TestCase):
  def test_family_backoff_preserves_macro_color_boundaries(self) -> None:
    metadata = {
      "red_engine": {"display_label": "Red Example Engine"},
      "green_engine": {"display_label": "Green Example Engine"},
      "azorius": {"display_label": "Azorius Control"},
      "dimir": {"display_label": "Dimir Control"},
      "ur_artifact": {"display_label": "UR Artifact Engine"},
      "mono_g_artifact": {"display_label": "Mono-G Artifact Engine"},
      "gr_big_mana": {"display_label": "GR Big Mana Engine"},
      "mono_g_big_mana": {"display_label": "Mono-G Big Mana Engine"},
      "jeskai_blink": {"display_label": "Jeskai Blink"},
      "esper_blink": {"display_label": "Esper Blink"},
      "generic": {"display_label": "GenericBlink"},
      "spaced_generic": {"display_label": "Generic Blink"},
    }
    mapping = build_family_mapping(metadata)
    families, probabilities = aggregate_family_probabilities(
      torch.tensor([[0.25, 0.30, 0.20, 0.25]]),
      labels=("red_engine", "green_engine", "azorius", "dimir"),
      family_by_label=mapping,
    )

    scores = dict(zip(families, probabilities[0].tolist(), strict=True))
    self.assertEqual(mapping["red_engine"], mapping["green_engine"])
    self.assertNotEqual(mapping["azorius"], mapping["dimir"])
    self.assertEqual(mapping["ur_artifact"], mapping["mono_g_artifact"])
    self.assertEqual(mapping["gr_big_mana"], mapping["mono_g_big_mana"])
    self.assertNotEqual(mapping["jeskai_blink"], mapping["esper_blink"])
    self.assertEqual("Control", relaxed_archetype_name("Control"))
    self.assertEqual("Blink", mapping["generic"])
    self.assertEqual(mapping["generic"], mapping["spaced_generic"])
    self.assertEqual("Genericity", relaxed_archetype_name("Genericity"))
    self.assertAlmostEqual(0.55, scores["Example Engine"], places=6)

  def test_family_vocab_serializes_relaxation_and_semantic_edges(self) -> None:
    metadata = {
      "red_engine": {"display_label": "Red Example Engine"},
      "green_engine": {"display_label": "Green Example Engine"},
      "primary_name": {"display_label": "Example Engine"},
      "legacy_name": {"display_label": "Legacy Engine Name"},
      "azorius": {"display_label": "Azorius Control"},
      "dimir": {"display_label": "Dimir Control"},
    }
    edges = extract_proposed_edges({
      "proposed_edges": [
        {
          "label_a": "Example Engine",
          "label_b": "Legacy Engine Name",
          "proposed_relation": "same_family",
        },
        {
          "label_a": "Azorius Control",
          "label_b": "Dimir Control",
          "proposed_relation": "model_confusion",
        },
      ],
    })
    payload = build_family_vocab(metadata, proposed_edges=edges)
    families = {
      entry["label_id"]: entry["family_id"]
      for entry in payload["entries"]
    }

    self.assertEqual(families["red_engine"], families["green_engine"])
    self.assertEqual(families["primary_name"], families["legacy_name"])
    self.assertNotEqual(families["azorius"], families["dimir"])
    self.assertEqual(2, payload["policy"]["relation_edge_count"])
    self.assertIn("same_family", payload["policy"]["accepted_relations"])

  def test_extract_proposed_edges_rejects_invalid_payload(self) -> None:
    with self.assertRaisesRegex(ValueError, "edge array"):
      extract_proposed_edges({"proposed_edges": "not-an-array"})

  def test_family_metrics_include_open_and_closed_family_views(self) -> None:
    metrics = _classification_metrics(
      ["Family Alpha", "Family Alpha", "Family Beta"],
      ["Family Alpha", "Family Beta", "Family Beta"],
      top_families=[
        {"Family Alpha"},
        {"Family Alpha", "Family Beta"},
        {"Family Beta"},
      ],
      represented_families={"Family Alpha"},
    )

    self.assertEqual(3, metrics["count"])
    self.assertAlmostEqual(2 / 3, metrics["accuracy"])
    self.assertAlmostEqual(0.5, metrics["closed_set_accuracy"])
    self.assertAlmostEqual(1.0, metrics["top3_accuracy"])
    self.assertAlmostEqual(2 / 3, metrics["macro_f1"])
    self.assertEqual(1, metrics["unrepresented_family_count"])



if __name__ == "__main__":
  unittest.main()

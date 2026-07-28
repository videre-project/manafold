from __future__ import annotations

import unittest

from manafold.data.export import build_proxy_targets


class ModelTargetTests(unittest.TestCase):
  def test_source_archetype_name_proxy_is_preferred(self) -> None:
    targets = build_proxy_targets(
      [
        {
          "dataset_version": "modern_2024_2024_v0",
          "deck_id": "deck-1",
          "source_archetype_name": "Example Engine",
          "reported_archetype": "WR",
        }
      ]
    )

    self.assertEqual(1, len(targets))
    self.assertEqual("source_archetype_name_proxy", targets[0]["target_source"])
    self.assertEqual("family", targets[0]["target_level"])
    self.assertEqual(
      "proxy.source_archetype_name_proxy.example_engine",
      targets[0]["proxy_label_id"],
    )
    self.assertEqual("Example Engine", targets[0]["display_label"])
    self.assertEqual("example engine", targets[0]["normalized_label"])

  def test_reported_archetype_is_used_as_fallback(self) -> None:
    targets = build_proxy_targets(
      [
        {
          "dataset_version": "modern_2024_2024_v0",
          "deck_id": "deck-1",
          "source_archetype_name": None,
          "reported_archetype": "Five Color Example",
        }
      ]
    )

    self.assertEqual("reported_archetype_proxy", targets[0]["target_source"])
    self.assertEqual(
      "proxy.reported_archetype_proxy.five_color_example",
      targets[0]["proxy_label_id"],
    )


if __name__ == "__main__":
  unittest.main()

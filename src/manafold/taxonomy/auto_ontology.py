"""Unsupervised family-relation induction for canonical-target training."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from manafold.datasets.mtgo.build import SOURCE_ARCHETYPE_PROXY
from manafold.datasets.model_inputs import TrainingDataset, load_training_dataset
from manafold.taxonomy.family_backoff import display_label_from_id


AUTO_ONTOLOGY_FORMAT_VERSION = "manafold_auto_ontology_v1"
AUTO_ONTOLOGY_TARGET_LABEL_LEVEL = "auto-ontology-family"
DEFAULT_MIN_JACCARD_THRESHOLD = 0.50
DEFAULT_CORE_CARD_FREQUENCY = 0.25


def induce_auto_ontology(
  dataset: TrainingDataset,
  *,
  min_jaccard_threshold: float = DEFAULT_MIN_JACCARD_THRESHOLD,
  core_card_frequency: float = DEFAULT_CORE_CARD_FREQUENCY,
) -> dict[str, Any]:
  """Induce a relation graph using only card occurrence in the training split."""
  if not 0.0 <= min_jaccard_threshold <= 1.0:
    raise ValueError("min_jaccard_threshold must be in [0, 1].")
  if not 0.0 <= core_card_frequency <= 1.0:
    raise ValueError("core_card_frequency must be in [0, 1].")

  train_examples = [
    example for example in dataset.examples if example.split_name == "train"
  ]
  examples_by_label: dict[str, list[Any]] = {}
  for example in train_examples:
    examples_by_label.setdefault(example.target_label_id, []).append(example)

  core_cards: dict[str, frozenset[int]] = {}
  for label, examples in examples_by_label.items():
    occurrences: Counter[int] = Counter()
    for example in examples:
      occurrences.update({token.card_idx for token in example.tokens})
    core_cards[label] = frozenset(
      card_idx
      for card_idx, count in occurrences.items()
      if count / len(examples) > core_card_frequency
    )

  components: list[dict[str, Any]] = []
  labels = sorted(core_cards)
  for left_index, left in enumerate(labels):
    left_cards = core_cards[left]
    if not left_cards:
      continue
    for right in labels[left_index + 1 :]:
      right_cards = core_cards[right]
      if not right_cards:
        continue
      score = len(left_cards & right_cards) / len(left_cards | right_cards)
      if score < min_jaccard_threshold:
        continue
      relation = "same_family" if score >= 0.70 else "sibling_variant"
      edge = {
        "label_a": display_label_from_id(left),
        "label_a_id": left,
        "label_b": display_label_from_id(right),
        "label_b_id": right,
        "proposed_relation": relation,
        "score": round(score, 6),
      }
      components.append({
        "component_id": f"auto_component_{len(components) + 1:04d}",
        "component_score": round(score, 6),
        "edge_count": 1,
        "max_edge_score": round(score, 6),
        "accepted_proposal_edges": [edge],
        "labels": [
          {
            "label": display_label_from_id(left),
            "label_id": left,
            "support": float(len(examples_by_label[left])),
          },
          {
            "label": display_label_from_id(right),
            "label_id": right,
            "support": float(len(examples_by_label[right])),
          },
        ],
      })

  return {
    "format_version": AUTO_ONTOLOGY_FORMAT_VERSION,
    "proposal_policy": {
      "method": "training_split_core_card_jaccard",
      "min_edge_score": min_jaccard_threshold,
      "core_card_frequency_exclusive": core_card_frequency,
      "same_family_threshold": 0.70,
      "reviewed_inputs_used_for_induction": False,
    },
    "dataset_provenance": {
      "dataset_version": dataset.dataset_version,
      "target_source": dataset.target_source,
      "induction_split": "train",
      "training_split_sha256": training_split_digest(dataset),
    },
    "dataset_summary": {
      "training_examples": len(train_examples),
      "observed_training_labels": len(examples_by_label),
      "exported_labels": len(dataset.labels),
    },
    "proposed_components": components,
  }


def generate_auto_ontology(
  dataset_path: Path,
  output_path: Path,
  *,
  min_jaccard_threshold: float = DEFAULT_MIN_JACCARD_THRESHOLD,
  core_card_frequency: float = DEFAULT_CORE_CARD_FREQUENCY,
) -> dict[str, Any]:
  """Load a generated dataset, induce its graph, and write the relation file."""
  started = time.perf_counter()
  dataset = load_training_dataset(
    dataset_path,
    target_source=SOURCE_ARCHETYPE_PROXY,
  )
  return write_auto_ontology(
    dataset,
    output_path,
    min_jaccard_threshold=min_jaccard_threshold,
    core_card_frequency=core_card_frequency,
    started=started,
  )


def write_auto_ontology(
  dataset: TrainingDataset,
  output_path: Path,
  *,
  min_jaccard_threshold: float = DEFAULT_MIN_JACCARD_THRESHOLD,
  core_card_frequency: float = DEFAULT_CORE_CARD_FREQUENCY,
  started: float | None = None,
) -> dict[str, Any]:
  """Induce and atomically replace an ontology file for one loaded dataset."""
  started = time.perf_counter() if started is None else started
  payload = induce_auto_ontology(
    dataset,
    min_jaccard_threshold=min_jaccard_threshold,
    core_card_frequency=core_card_frequency,
  )
  elapsed_seconds = time.perf_counter() - started
  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path = output_path.with_name(f".{output_path.name}.tmp")
  temporary_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  temporary_path.replace(output_path)
  return {**payload, "generation": {"elapsed_seconds": elapsed_seconds}}


def training_split_digest(dataset: TrainingDataset) -> str:
  """Identify the semantic training rows independently of local vocabulary order."""
  oracle_id_by_card_idx = {
    int(card["card_idx"]): str(card["oracle_id"])
    for card in dataset.card_vocab
  }
  zone_by_idx = {
    int(zone_idx): str(zone) for zone, zone_idx in dataset.zone_vocab.items()
  }
  rows = []
  for example in dataset.examples:
    if example.split_name != "train":
      continue
    rows.append({
      "deck_id": example.deck_id,
      "event_id": example.event_id,
      "event_date": (
        example.event_date.isoformat()
        if example.event_date is not None
        else None
      ),
      "format_code": example.format_code,
      "target_label_id": example.target_label_id,
      "source_label_id": example.source_label_id,
      "tokens": sorted(
        (
          oracle_id_by_card_idx[token.card_idx],
          token.quantity,
          zone_by_idx[token.zone_idx],
        )
        for token in example.tokens
      ),
    })
  rows.sort(
    key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"))
  )
  encoded = json.dumps(
    rows,
    ensure_ascii=True,
    sort_keys=True,
    separators=(",", ":"),
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()

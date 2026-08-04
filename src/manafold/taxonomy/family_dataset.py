"""Dataset projection utility for Direct Canonical Family-Target Training (Manafold A11)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from manafold.datasets.model_inputs import TrainingDataset, UNKNOWN_LABEL_ID
from manafold.taxonomy.family_backoff import (
  build_family_mapping,
  display_label_from_id,
  extract_proposed_edges,
)


def project_dataset_to_canonical_families(
  dataset: TrainingDataset,
  *,
  family_relations_path: Path | None = None,
) -> tuple[TrainingDataset, dict[str, str]]:
  """Project raw proxy targets through an explicit generated auto-ontology.

  Returns:
    tuple: (projected_dataset, family_by_raw_label_map)
  """
  if family_relations_path is None:
    raise ValueError(
      "Canonical-family projection requires an explicit auto-ontology artifact."
    )
  proposed_edges = extract_proposed_edges(
    json.loads(family_relations_path.read_text(encoding="utf-8"))
  )

  all_label_ids = set(dataset.labels) - {UNKNOWN_LABEL_ID}
  label_metadata = {
    label_id: {
      "label_id": label_id,
      "display_label": display_label_from_id(label_id),
      "target_source": dataset.target_source or "",
    }
    for label_id in all_label_ids
  }

  family_by_label = build_family_mapping(
    label_metadata,
    proposed_edges=proposed_edges,
  )

  # Project all dataset examples
  projected_examples = []
  for example in dataset.examples:
    fam_label = (
      UNKNOWN_LABEL_ID
      if example.target_label_id == UNKNOWN_LABEL_ID
      else family_by_label.get(example.target_label_id, example.target_label_id)
    )
    projected_examples.append(replace(example, target_label_id=fam_label))

  canonical_family_labels = sorted(list(set(family_by_label.values())))

  projected_dataset = replace(
    dataset,
    examples=tuple(projected_examples),
    labels=tuple(canonical_family_labels),
    target_source="family_canonical_proxy",
  )

  return projected_dataset, family_by_label

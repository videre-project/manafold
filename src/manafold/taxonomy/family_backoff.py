from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

import torch

FAMILY_POLICY_VERSION = "manafold_family_backoff_v1"
COLOR_PREFIXES = (
  "Mono-White",
  "Mono-Blue",
  "Mono-Black",
  "Mono-Red",
  "Mono-Green",
  "Mono White",
  "Mono Blue",
  "Mono Black",
  "Mono Red",
  "Mono Green",
  "Mono-W",
  "Mono-U",
  "Mono-B",
  "Mono-R",
  "Mono-G",
  "Azorius",
  "Orzhov",
  "Boros",
  "Selesnya",
  "Dimir",
  "Izzet",
  "Rakdos",
  "Golgari",
  "Gruul",
  "Simic",
  "Jeskai",
  "Grixis",
  "Jund",
  "Naya",
  "Bant",
  "Abzan",
  "Sultai",
  "Mardu",
  "Temur",
  "Esper",
  "White",
  "Blue",
  "Black",
  "Red",
  "Green",
  "Colorless",
  "Snow",
  "WUBRG",
  "WBRG",
  "WURG",
  "WUBG",
  "WUBR",
  "UBRG",
  "4/5c",
  "4c",
  "5c",
  "WUR",
  "UBR",
  "BRG",
  "WRG",
  "GWU",
  "WBG",
  "UBG",
  "WBR",
  "URG",
  "WUB",
  "WUG",
  "WU",
  "WB",
  "WR",
  "WG",
  "UB",
  "UR",
  "BR",
  "BG",
  "RG",
  "GR",
  "UG",
  "W",
  "U",
  "B",
  "R",
  "G",
  "C",
  "S",
)
MACRO_ARCHETYPES = frozenset({
  "Aggro",
  "Blink",
  "Combo",
  "Control",
  "Midrange",
  "Prison",
  "Ramp",
  "Tempo",
})
SEMANTIC_AUTO_RELATIONS = frozenset({
  "alias",
  "same_family",
  "sibling_variant",
})


class _DisjointSet:

  def __init__(self, values: Iterable[str]) -> None:
    self.parent = {value: value for value in values}

  def find(self, value: str) -> str:
    parent = self.parent[value]
    if parent != value:
      self.parent[value] = self.find(parent)
    return self.parent[value]

  def union(self, left: str, right: str) -> None:
    left_root = self.find(left)
    right_root = self.find(right)
    if left_root != right_root:
      self.parent[right_root] = left_root


def relaxed_archetype_name(label: str) -> str:
  """Normalize generic markers and relax colors on descriptive family names."""
  normalized = re.sub(
    r"^Generic(?=[A-Z])",
    "",
    label,
    count=1,
  )
  normalized = re.sub(
    r"^generic[\s_/-]+",
    "",
    normalized,
    count=1,
    flags=re.IGNORECASE,
  ).strip()
  if normalized:
    label = normalized

  for prefix in sorted(COLOR_PREFIXES, key=len, reverse=True):
    relaxed = re.sub(
      rf"^{re.escape(prefix)}(?:\s+|[-_/]+\s*)",
      "",
      label,
      count=1,
      flags=re.IGNORECASE,
    ).strip()
    if relaxed != label and relaxed not in MACRO_ARCHETYPES:
      return relaxed
  return label


def display_label_from_id(label_id: str) -> str:
  """Recover a readable fallback label when no exported metadata is available."""
  slug = label_id.rsplit(".", 1)[-1]
  return " ".join(word.capitalize() for word in slug.split("_") if word)


def build_family_mapping(
  label_metadata: dict[str, dict[str, Any]],
  *,
  proposed_edges: Iterable[dict[str, Any]] = (),
  accepted_relations: frozenset[str] = SEMANTIC_AUTO_RELATIONS,
) -> dict[str, str]:
  """Build a serving-time family map from labels and optional induced edges."""
  label_by_id = {
    label_id: str(metadata["display_label"])
    for label_id, metadata in label_metadata.items()
  }
  ids_by_display: dict[str, list[str]] = defaultdict(list)
  ids_by_relaxed_name: dict[str, list[str]] = defaultdict(list)
  for label_id, display_label in label_by_id.items():
    ids_by_display[display_label.casefold()].append(label_id)
    ids_by_relaxed_name[
      relaxed_archetype_name(display_label).casefold()
    ].append(label_id)

  groups = _DisjointSet(label_by_id)
  for label_ids in ids_by_relaxed_name.values():
    for label_id in label_ids[1:]:
      groups.union(label_ids[0], label_id)

  for edge in proposed_edges:
    relation = str(
      edge.get("proposed_relation") or edge.get("relation") or ""
    )
    if relation not in accepted_relations:
      continue
    left_ids = ids_by_display.get(str(edge.get("label_a", "")).casefold(), ())
    right_ids = ids_by_display.get(str(edge.get("label_b", "")).casefold(), ())
    for left_id in left_ids:
      for right_id in right_ids:
        groups.union(left_id, right_id)

  members_by_root: dict[str, list[str]] = defaultdict(list)
  for label_id in label_by_id:
    members_by_root[groups.find(label_id)].append(label_id)

  family_by_label: dict[str, str] = {}
  for members in members_by_root.values():
    names = Counter(
      relaxed_archetype_name(label_by_id[label_id]) for label_id in members
    )
    family_name = min(
      names,
      key=lambda name: (-names[name], len(name), name.casefold()),
    )
    for label_id in members:
      family_by_label[label_id] = family_name
  return family_by_label


def extract_proposed_edges(payload: Any) -> tuple[dict[str, Any], ...]:
  """Read relation edges from an auto-ontology or compact relation artifact."""
  if isinstance(payload, list):
    edges = payload
  elif isinstance(payload, dict):
    if "proposed_components" in payload:
      edges = [
        edge
        for component in payload["proposed_components"]
        for edge in component.get("accepted_proposal_edges", ())
      ]
    else:
      edges = payload.get("proposed_edges", payload.get("edges", ()))
  else:
    raise ValueError("Family relation evidence must be a JSON object or array.")
  if not isinstance(edges, (list, tuple)):
    raise ValueError("Family relation evidence does not contain an edge array.")
  if not all(isinstance(edge, dict) for edge in edges):
    raise ValueError("Every family relation edge must be a JSON object.")
  return tuple(edges)


def build_family_vocab(
  label_metadata: dict[str, dict[str, Any]],
  *,
  proposed_edges: Iterable[dict[str, Any]] = (),
  accepted_relations: frozenset[str] = SEMANTIC_AUTO_RELATIONS,
) -> dict[str, Any]:
  """Create the compact family vocabulary consumed by serving runtimes."""
  edges = tuple(proposed_edges)
  family_by_label = build_family_mapping(
    label_metadata,
    proposed_edges=edges,
    accepted_relations=accepted_relations,
  )
  family_names = sorted(set(family_by_label.values()), key=str.casefold)
  family_ids: dict[str, str] = {}
  used_ids: set[str] = set()
  for name in family_names:
    base = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_") or "unnamed"
    family_id = f"family.{base}"
    if family_id in used_ids:
      digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
      family_id = f"{family_id}.{digest}"
    used_ids.add(family_id)
    family_ids[name] = family_id

  return {
    "version": FAMILY_POLICY_VERSION,
    "policy": {
      "color_relaxation": True,
      "generic_marker_normalization": True,
      "accepted_relations": sorted(accepted_relations),
      "relation_edge_count": len(edges),
    },
    "families": [
      {
        "family_id": family_ids[name],
        "display_label": name,
      }
      for name in family_names
    ],
    "entries": [
      {
        "label_id": label_id,
        "family_id": family_ids[family_by_label[label_id]],
      }
      for label_id in sorted(family_by_label)
    ],
  }


def aggregate_family_probabilities(
  probabilities: torch.Tensor,
  *,
  labels: tuple[str, ...],
  family_by_label: dict[str, str],
) -> tuple[tuple[str, ...], torch.Tensor]:
  """Sum label probabilities into relaxed or induced archetype families."""
  families = tuple(sorted({family_by_label[label] for label in labels}))
  family_index = {family: index for index, family in enumerate(families)}
  indexes = torch.tensor(
    [family_index[family_by_label[label]] for label in labels],
    dtype=torch.int64,
    device=probabilities.device,
  )
  aggregated = probabilities.new_zeros((probabilities.shape[0], len(families)))
  aggregated.scatter_add_(
    1,
    indexes.unsqueeze(0).expand(probabilities.shape[0], -1),
    probabilities,
  )
  return families, aggregated

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def _logit_stats(
  logits: torch.Tensor,
  labels: tuple[str, ...],
  k: int,
) -> list[dict[str, Any]]:
  if not labels:
    return [_empty_prediction_stats() for _ in range(logits.shape[0])]

  probabilities = torch.softmax(logits, dim=1)
  entropy = _probability_entropy(probabilities)
  energy = -torch.logsumexp(logits, dim=1)
  if len(labels) > 1:
    normalizer = torch.log(
      torch.tensor(
        float(len(labels)),
        dtype=torch.float32,
        device=probabilities.device,
      )
    )
    normalized_entropy = entropy / normalizer
  else:
    normalized_entropy = torch.zeros_like(entropy)

  values, indexes = torch.topk(
    probabilities,
    k=min(k, len(labels)),
    dim=1,
  )
  rows: list[dict[str, Any]] = []
  for (
    row_scores,
    row_indexes,
    row_entropy,
    row_normalized_entropy,
    row_energy,
  ) in zip(
    values.tolist(),
    indexes.tolist(),
    entropy.tolist(),
    normalized_entropy.tolist(),
    energy.tolist(),
    strict=True,
  ):
    top_predictions = [
      (labels[int(label_idx)], float(score))
      for score, label_idx in zip(row_scores, row_indexes, strict=True)
    ]
    rows.append({
      "top_predictions": top_predictions,
      "max_probability": (
        float(top_predictions[0][1]) if top_predictions else 0.0
      ),
      "entropy": float(row_entropy),
      "normalized_entropy": float(row_normalized_entropy),
      "energy": float(row_energy),
    })

  return rows


def _probability_entropy(probabilities: torch.Tensor) -> torch.Tensor:
  safe_probabilities = probabilities.clamp_min(1e-12)
  return -(safe_probabilities * safe_probabilities.log()).sum(dim=1)


def _prototype_stats(
  *,
  top_probabilities: torch.Tensor,
  top_indexes: torch.Tensor,
  top_distances: torch.Tensor,
  entropy: torch.Tensor,
  energy: torch.Tensor,
  nearest_distances: torch.Tensor,
  margins: torch.Tensor,
  labels: tuple[str, ...],
) -> list[dict[str, Any]]:
  if len(labels) > 1:
    normalizer = torch.log(
      torch.tensor(
        float(len(labels)),
        dtype=torch.float32,
        device=top_probabilities.device,
      )
    )
    normalized_entropy = entropy / normalizer
  else:
    normalized_entropy = torch.zeros_like(entropy)

  rows: list[dict[str, Any]] = []
  for (
    row_scores,
    row_indexes,
    row_distances,
    row_entropy,
    row_normalized_entropy,
    row_energy,
    row_nearest_distance,
    row_margin,
  ) in zip(
    top_probabilities.tolist(),
    top_indexes.tolist(),
    top_distances.tolist(),
    entropy.tolist(),
    normalized_entropy.tolist(),
    energy.tolist(),
    nearest_distances.tolist(),
    margins.tolist(),
    strict=True,
  ):
    top_predictions = [
      (labels[int(label_idx)], float(score))
      for score, label_idx in zip(row_scores, row_indexes, strict=True)
    ]
    rows.append({
      "top_predictions": top_predictions,
      "max_probability": (
        float(top_predictions[0][1]) if top_predictions else 0.0
      ),
      "entropy": float(row_entropy),
      "normalized_entropy": float(row_normalized_entropy),
      "energy": float(row_energy),
      "nearest_prototype_distance": float(row_nearest_distance),
      "prototype_margin": float(row_margin),
      "top_prototype_distances": [
        float(distance) for distance in row_distances
      ],
    })

  return rows


def _empty_prediction_stats() -> dict[str, Any]:
  return {
    "top_predictions": [],
    "max_probability": 0.0,
    "entropy": None,
    "normalized_entropy": None,
    "energy": None,
    "nearest_prototype_distance": None,
    "prototype_margin": None,
    "top_prototype_distances": [],
  }


def _card_nearest_neighbors(
  embeddings: torch.Tensor,
  card_vocab: tuple[dict[str, Any], ...],
  *,
  anchor_card_idxs: list[int] | None = None,
  card_support: dict[int, int] | None = None,
  anchor_limit: int = 20,
  neighbor_count: int = 5,
) -> list[dict[str, Any]]:
  if anchor_limit <= 0 or neighbor_count <= 0:
    return []

  vocab = sorted(card_vocab, key=lambda row: int(row["card_idx"]))
  vocab_by_idx = {int(card["card_idx"]): card for card in vocab}
  normalized = F.normalize(embeddings, p=2, dim=1)
  if anchor_card_idxs is None:
    anchor_idxs = [int(card["card_idx"]) for card in vocab[:anchor_limit]]
  else:
    anchor_idxs = [
      card_idx
      for card_idx in anchor_card_idxs[:anchor_limit]
      if card_idx in vocab_by_idx
    ]
  rows: list[dict[str, Any]] = []
  for card_idx in anchor_idxs:
    card = vocab_by_idx[card_idx]
    similarities = torch.mv(normalized, normalized[card_idx])
    similarities[card_idx] = -2.0
    scores, indexes = torch.topk(
      similarities,
      k=min(neighbor_count, len(vocab) - 1),
    )
    rows.append({
      "card_idx": card_idx,
      "primary_name": card.get("primary_name"),
      "train_support": (
        None if card_support is None else card_support.get(card_idx, 0)
      ),
      "neighbors": [
        {
          "card_idx": int(neighbor_idx),
          "primary_name": vocab_by_idx[int(neighbor_idx)].get("primary_name"),
          "train_support": (
            None
            if card_support is None
            else card_support.get(int(neighbor_idx), 0)
          ),
          "score": float(score),
        }
        for score, neighbor_idx in zip(
          scores.tolist(),
          indexes.tolist(),
          strict=True,
        )
      ],
    })
  return rows

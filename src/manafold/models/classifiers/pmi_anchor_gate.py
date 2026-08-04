"""Train-only card-information bias for Set Transformer attention."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn


class PMIAnchorGate(nn.Module):
  """Store train-derived PMI scores and return a learnable key bias."""

  def __init__(
    self,
    card_count: int,
    labels: Sequence[str],
    examples: Sequence[Any],
    alpha: float = 1.0,
    scale: float = 0.5,
    support_prior: float = 5.0,
    device: str | torch.device = "cpu",
  ) -> None:
    super().__init__()
    self.card_count = card_count
    self.labels = tuple(labels)
    self.num_classes = len(self.labels)
    self.scale = scale
    self.alpha = alpha
    self.support_prior = support_prior

    self.register_buffer(
      "anchor_scores",
      torch.zeros(card_count, dtype=torch.float32, device=device),
    )
    self.gate_scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
    self.fit(examples)

  @torch.no_grad()
  def fit(self, examples: Sequence[Any]) -> None:
    """Replace scores using only the examples supplied by the caller."""
    pmi_matrix, card_support = self._compute_pmi_matrix_with_support(
      self.card_count,
      self.labels,
      examples,
      alpha=self.alpha,
    )
    anchor_scores = np.max(pmi_matrix, axis=1)
    if self.support_prior > 0:
      shrinkage = card_support / (card_support + self.support_prior)
      anchor_scores *= shrinkage
    max_score = float(np.max(anchor_scores)) if anchor_scores.size else 0.0
    if max_score > 0:
      anchor_scores = anchor_scores / max_score * 5.0
    self.anchor_scores.copy_(
      torch.as_tensor(
        anchor_scores,
        dtype=self.anchor_scores.dtype,
        device=self.anchor_scores.device,
      )
    )

  def _compute_pmi_matrix(
    self,
    card_count: int,
    labels: Sequence[str],
    examples: Sequence[Any],
    alpha: float = 1.0,
  ) -> np.ndarray:
    """Return the card-by-class positive PMI matrix."""
    matrix, _ = self._compute_pmi_matrix_with_support(
      card_count,
      labels,
      examples,
      alpha=alpha,
    )
    return matrix

  def _compute_pmi_matrix_with_support(
    self,
    card_count: int,
    labels: Sequence[str],
    examples: Sequence[Any],
    alpha: float = 1.0,
  ) -> tuple[np.ndarray, np.ndarray]:
    label_to_idx = {l: i for i, l in enumerate(labels)}
    class_card_counts: dict[str, dict[int, int]] = defaultdict(
      lambda: defaultdict(int)
    )
    class_total_decks: Counter[str] = Counter()
    card_global_decks: Counter[int] = Counter()
    total_train_decks = 0

    for ex in examples:
      lbl = getattr(ex, "target_label_id", None)
      if lbl not in label_to_idx:
        continue
      total_train_decks += 1
      class_total_decks[lbl] += 1
      tokens = getattr(ex, "tokens", ())
      seen_cards = {t.card_idx for t in tokens}
      for c_idx in seen_cards:
        if c_idx < card_count:
          class_card_counts[lbl][c_idx] += 1
          card_global_decks[c_idx] += 1

    pmi_matrix = np.zeros((card_count, len(labels)), dtype=np.float32)

    for card_idx in range(card_count):
      p_card = (card_global_decks[card_idx] + alpha) / (
        total_train_decks + alpha * 2
      )
      for lbl, class_idx in label_to_idx.items():
        n_class_decks = class_total_decks[lbl]
        if n_class_decks == 0:
          continue
        c_count = class_card_counts[lbl][card_idx]
        p_card_given_class = (c_count + alpha) / (n_class_decks + alpha * 2)
        pmi = math.log(max(p_card_given_class / p_card, 1e-6))
        pmi_matrix[card_idx, class_idx] = max(0.0, pmi)

    return pmi_matrix, np.asarray(
      [card_global_decks[index] for index in range(card_count)],
      dtype=np.float32,
    )

  def forward(self, card_indices: torch.Tensor) -> torch.Tensor:
    """Returns PMI anchor attention bias for given card indices.

    Args:
          card_indices: Tensor of shape (batch_size, set_size) with card indices.

    Returns:
          A tensor with the same shape as ``card_indices``.
    """
    clamped_indices = card_indices.clamp(0, self.card_count - 1)
    return self.gate_scale * self.anchor_scores[clamped_indices]

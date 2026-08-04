"""Causal soft target generation and loss functions for Candidate C.

Implements non-uniform similarity-weighted label smoothing using historical
co-confusion and mainboard Jaccard card overlaps prior to t_train_end.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CausalSoftTargetGenerator:
  """Generates non-uniform soft target distributions from causal past-window metrics."""

  def __init__(
    self,
    num_classes: int,
    alpha: float = 0.15,
    confusion_weight: float = 0.5,
    jaccard_weight: float = 0.5,
    eps: float = 1e-8,
  ) -> None:
    self.num_classes = num_classes
    self.alpha = alpha
    self.confusion_weight = confusion_weight
    self.jaccard_weight = jaccard_weight
    self.eps = eps
    self.soft_matrix: Optional[torch.Tensor] = None

  def fit_from_historical_data(
    self,
    target_indices: List[int],
    card_features_by_deck: List[List[int]],
    historical_confusion_matrix: Optional[np.ndarray] = None,
  ) -> torch.Tensor:
    """Builds causal K x K soft target transition matrix from train data up to t_train_end.

    Args:
          target_indices: List of integer target class indices for training decks.
          card_features_by_deck: List of card ID sets per deck.
          historical_confusion_matrix: Optional (K, K) co-confusion matrix.

    Returns:
          torch.Tensor of shape (K, K) where row k is the soft target vector for class k.
    """
    K = self.num_classes
    soft_matrix = np.eye(K, dtype=np.float32)

    # 1. Build mainboard Jaccard card similarity between classes
    jaccard_matrix = np.zeros((K, K), dtype=np.float32)
    class_card_sets: Dict[int, set] = {k: set() for k in range(K)}
    for target, cards in zip(target_indices, card_features_by_deck):
      if 0 <= target < K:
        class_card_sets[target].update(cards)

    for i in range(K):
      set_i = class_card_sets[i]
      if not set_i:
        continue
      for j in range(i + 1, K):
        set_j = class_card_sets[j]
        if not set_j:
          continue
        intersection = len(set_i.intersection(set_j))
        union = len(set_i.union(set_j))
        if union > 0:
          sim = intersection / union
          jaccard_matrix[i, j] = sim
          jaccard_matrix[j, i] = sim

    # 2. Process confusion matrix if available
    conf_matrix = np.zeros((K, K), dtype=np.float32)
    if (
      historical_confusion_matrix is not None
      and historical_confusion_matrix.shape == (K, K)
    ):
      conf_matrix = historical_confusion_matrix.astype(np.float32)
      # Remove self-loops for normalization
      np.fill_diagonal(conf_matrix, 0.0)

    # 3. Combine similarity signals
    combined_sim = (
      self.jaccard_weight * jaccard_matrix + self.confusion_weight * conf_matrix
    )

    # Zero out diagonal
    np.fill_diagonal(combined_sim, 0.0)

    # Normalize non-target similarity rows
    row_sums = combined_sim.sum(axis=1, keepdims=True)
    normalized_sim = np.zeros_like(combined_sim)
    np.divide(
      combined_sim,
      row_sums,
      out=normalized_sim,
      where=row_sums > self.eps,
    )

    # Mix one-hot identity with non-uniform soft similarity
    soft_matrix = (1.0 - self.alpha) * np.eye(
      K, dtype=np.float32
    ) + self.alpha * normalized_sim

    # Ensure valid probability distribution per row
    row_totals = soft_matrix.sum(axis=1, keepdims=True)
    soft_matrix = soft_matrix / row_totals

    self.soft_matrix = torch.from_numpy(soft_matrix).float()
    return self.soft_matrix

  def get_soft_targets(self, targets: torch.Tensor) -> torch.Tensor:
    """Looks up soft target vectors for batch of class targets.

    Args:
          targets: Tensor of integer class indices (batch_size,).

    Returns:
          Tensor of shape (batch_size, K) containing soft target probabilities.
    """
    if self.soft_matrix is None:
      # Fallback to standard one-hot if not fit
      K = self.num_classes
      return F.one_hot(targets, num_classes=K).float()

    return self.soft_matrix[targets].to(targets.device)


class SoftTargetCrossEntropyLoss(nn.Module):
  """Computes cross-entropy loss against continuous soft target probability distributions."""

  def __init__(self, reduction: str = "mean") -> None:
    super().__init__()
    self.reduction = reduction

  def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Computes -sum(y_soft * log_softmax(logits)).

    Args:
          logits: Unnormalized logit tensor of shape (batch_size, num_classes).
          soft_targets: Target probability distributions of shape (batch_size, num_classes).

    Returns:
          Loss scalar or tensor based on reduction mode.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -torch.sum(soft_targets * log_probs, dim=-1)

    if self.reduction == "mean":
      return torch.mean(loss)
    elif self.reduction == "sum":
      return torch.sum(loss)
    return loss

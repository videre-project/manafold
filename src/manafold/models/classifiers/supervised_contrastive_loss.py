"""Supervised Set-Level Contrastive Loss (SupConLoss) for Manafold A6.

Pulls full-deck (60 cards) and partial-deck (5, 10, 20 cards) views of the same archetype family
together in normalized Euclidean feature space while pushing distinct archetype clusters apart.
Ref: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
  """Supervised Contrastive Loss over set feature embeddings."""

  def __init__(
    self, temperature: float = 0.07, base_temperature: float = 0.07
  ) -> None:
    super().__init__()
    self.temperature = temperature
    self.base_temperature = base_temperature

  def forward(
    self,
    features: torch.Tensor,
    labels: torch.Tensor,
    partial_features: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Args:
          features: (batch_size, dim) normalized full-deck embeddings.
          labels: (batch_size,) integer class label IDs.
          partial_features: Optional (batch_size, dim) normalized partial-deck embeddings.

    Returns:
          Scalar loss tensor.
    """
    device = features.device
    batch_size = features.shape[0]

    # L2 normalize features
    features = F.normalize(features, dim=-1)

    if partial_features is not None:
      partial_features = F.normalize(partial_features, dim=-1)
      # Concatenate full and partial views: (2 * batch_size, dim)
      combined_features = torch.cat([features, partial_features], dim=0)
      combined_labels = torch.cat([labels, labels], dim=0)
    else:
      combined_features = features
      combined_labels = labels

    total_samples = combined_features.shape[0]

    # Compute label mask: 1 where labels match, 0 elsewhere
    labels_mask = (
      torch.eq(combined_labels.unsqueeze(0), combined_labels.unsqueeze(1))
      .float()
      .to(device)
    )

    # Self-contrastive mask: zero out self-comparisons on diagonal
    self_mask = torch.eye(total_samples, device=device).bool()
    mask = labels_mask.masked_fill(self_mask, 0.0)

    # Compute dot product similarity logits
    similarity_matrix = (
      torch.matmul(combined_features, combined_features.T) / self.temperature
    )

    # Subtract max for numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    # Mask out self-contrastive diagonal in denominator
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(
      exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-8)
    )

    # Compute mean of log-likelihood over positive pairs
    pos_per_sample = mask.sum(dim=1)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_per_sample.clamp(min=1.0)

    # Loss is only evaluated on samples with at least 1 positive pair
    valid_samples = pos_per_sample > 0
    if not valid_samples.any():
      return torch.tensor(0.0, device=device, requires_grad=True)

    loss = (
      -(self.temperature / self.base_temperature)
      * mean_log_prob_pos[valid_samples]
    )
    return loss.mean()

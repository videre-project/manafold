"""Hyperbolic geometry operations and Poincaré Prototypical Head for A5.

Implements Poincaré ball manifold operations, Riemannian tangent-space set pooling,
Poincaré distance metrics, radius clipping, and Prototypical Head classification.

References:
- Müller et al. (2025) arXiv:2506.10146: Balanced Hyperbolic Embeddings Are Natural OOD Detectors
- Gerek et al. (2022) arXiv:2211.04462: Hyperbolic Centroid Calculations for Text Classification
- Bera et al. (2023) arXiv:2309.10013: Hyperbolic vs Euclidean Embeddings in Few-Shot Learning
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def clip_to_poincare_ball(
  x: torch.Tensor, c: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
  """Clips vectors to stay within the Poincaré ball ||x|| < 1/sqrt(c) - eps."""
  max_norm = (1.0 / math.sqrt(c)) - eps
  norm = torch.norm(x, dim=-1, keepdim=True)
  cond = norm >= max_norm
  clipped = (x / (norm + 1e-12)) * max_norm
  return torch.where(cond, clipped, x)


def exp_map_zero(v: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
  """Projects Euclidean tangent vector v at origin into the Poincaré ball B^n."""
  sqrt_c = math.sqrt(c)
  v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-12)
  gamma = torch.tanh(sqrt_c * v_norm) * (v / (sqrt_c * v_norm))
  return clip_to_poincare_ball(gamma, c=c, eps=eps)


def log_map_zero(y: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
  """Projects Poincaré point y back to Euclidean tangent space at origin."""
  y = clip_to_poincare_ball(y, c=c, eps=eps)
  sqrt_c = math.sqrt(c)
  y_norm = torch.norm(y, dim=-1, keepdim=True).clamp(min=1e-12, max=1.0 - eps)
  # artanh(x) = 0.5 * log((1+x)/(1-x))
  artanh_val = 0.5 * torch.log(
    (1.0 + sqrt_c * y_norm) / (1.0 - sqrt_c * y_norm + 1e-12)
  )
  return artanh_val * (y / (sqrt_c * y_norm))


def mobius_addition(
  u: torch.Tensor, v: torch.Tensor, c: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
  """Computes Mobius addition (-u) (+) v in the Poincare ball."""
  u = clip_to_poincare_ball(u, c=c, eps=eps)
  v = clip_to_poincare_ball(v, c=c, eps=eps)

  u_sq = torch.sum(u * u, dim=-1, keepdim=True)
  v_sq = torch.sum(v * v, dim=-1, keepdim=True)
  uv_dot = torch.sum(u * v, dim=-1, keepdim=True)

  num = (1.0 + 2.0 * c * uv_dot + c * v_sq) * u + (1.0 - c * u_sq) * v
  denom = 1.0 + 2.0 * c * uv_dot + (c**2) * u_sq * v_sq

  result = num / (denom + 1e-12)
  return clip_to_poincare_ball(result, c=c, eps=eps)


def poincare_distance(
  u: torch.Tensor, v: torch.Tensor, c: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
  """Computes Poincare distance d_B(u, v) between pairs of points in B^n."""
  u = clip_to_poincare_ball(u, c=c, eps=eps)
  v = clip_to_poincare_ball(v, c=c, eps=eps)

  diff_norm_sq = torch.sum((u - v) ** 2, dim=-1)
  u_norm_sq = torch.sum(u**2, dim=-1)
  v_norm_sq = torch.sum(v**2, dim=-1)

  arg = 1.0 + 2.0 * c * diff_norm_sq / (
    ((1.0 - c * u_norm_sq) * (1.0 - c * v_norm_sq)).clamp(min=1e-12)
  )
  arg = arg.clamp(min=1.0 + 1e-7)

  # arcosh(x) = log(x + sqrt(x^2 - 1))
  dist = (1.0 / math.sqrt(c)) * torch.log(
    arg + torch.sqrt((arg**2 - 1.0).clamp(min=0.0))
  )
  dist = torch.where(diff_norm_sq < 1e-12, torch.zeros_like(dist), dist)
  return dist


class TangentSpaceSetPooler(nn.Module):
  """Aggregates card embeddings in hyperbolic space via Euclidean tangent-space projection.

  Implements: exp_0( sum_i q_i * log_0( phi(c_i) ) )
  """

  def __init__(self, curvature: float = 1.0) -> None:
    super().__init__()
    self.curvature = curvature

  def forward(
    self,
    card_embeddings: torch.Tensor,
    card_weights: torch.Tensor,
    mask: torch.Tensor,
  ) -> torch.Tensor:
    """Args:
    card_embeddings: (batch_size, seq_len, dim)
    card_weights: (batch_size, seq_len)
    mask: (batch_size, seq_len) boolean mask where True = valid card
    """
    # Map card embeddings to tangent space at origin
    tangent_cards = log_map_zero(card_embeddings, c=self.curvature)

    # Weighted sum in tangent space
    weights = card_weights * mask.float()
    weighted_tangent = tangent_cards * weights.unsqueeze(-1)
    sum_tangent = torch.sum(weighted_tangent, dim=1)

    # Normalize by weight sum to maintain centroid scaling
    weight_sum = torch.sum(weights, dim=1, keepdim=True).clamp(min=1e-8)
    mean_tangent = sum_tangent / weight_sum

    # Map aggregated vector back to Poincare ball
    return exp_map_zero(mean_tangent, c=self.curvature)


class HyperbolicPrototypicalHead(nn.Module):
  """Classification & OOD detection head using Poincare prototypical distance."""

  def __init__(
    self,
    embedding_dim: int,
    num_classes: int,
    temperature: float = 1.0,
    curvature: float = 1.0,
  ) -> None:
    super().__init__()
    self.embedding_dim = embedding_dim
    self.num_classes = num_classes
    self.temperature = temperature
    self.curvature = curvature

    # Initialize archetype prototypes near origin in B^n
    prototypes_init = torch.randn(num_classes, embedding_dim) * 0.01
    self.prototypes = nn.Parameter(exp_map_zero(prototypes_init, c=curvature))

  def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes logits and OOD margin scores.

    Args:
          embeddings: Deck embeddings in Poincare ball (batch_size, dim).

    Returns:
          Tuple of (logits, ood_margin_scores)
          - logits: (batch_size, num_classes) negative scaled distances
          - ood_margin_scores: (batch_size,) minimum Poincare distance to any prototype
    """
    clipped_embeds = clip_to_poincare_ball(embeddings, c=self.curvature)
    clipped_protos = clip_to_poincare_ball(self.prototypes, c=self.curvature)

    # Compute (batch_size, num_classes) Poincare distance matrix
    B = clipped_embeds.size(0)
    K = self.num_classes

    # Expand for pairwise distance
    embeds_exp = clipped_embeds.unsqueeze(1).expand(B, K, self.embedding_dim)
    protos_exp = clipped_protos.unsqueeze(0).expand(B, K, self.embedding_dim)

    dists = poincare_distance(embeds_exp, protos_exp, c=self.curvature)

    logits = -dists / self.temperature
    ood_margins, _ = torch.min(dists, dim=-1)

    return logits, ood_margins

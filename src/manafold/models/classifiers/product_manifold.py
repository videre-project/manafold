"""Euclidean and Poincare product representation used by Manafold A8."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoincaréHyperbolicSpace:
  """Operations in the Poincaré Ball Model of Hyperbolic Geometry."""

  @staticmethod
  def exp_map_zero(
    v: torch.Tensor, c: float = 1.0, eps: float = 1e-5
  ) -> torch.Tensor:
    """Exponential map at zero in Poincaré ball.

    Maps Euclidean vectors v into Poincaré ball of curvature c.
    """
    sqrt_c = math.sqrt(c)
    v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp(min=eps)
    gamma = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
    y = gamma * v
    # Projection to ensure strict norm < 1 / sqrt(c)
    max_norm = (1.0 - 1e-4) / sqrt_c
    y_norm = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    cond = y_norm >= max_norm
    y = torch.where(cond, y / y_norm.clamp_min(eps) * max_norm, y)
    return y

  @staticmethod
  def distance(
    u: torch.Tensor, v: torch.Tensor, c: float = 1.0, eps: float = 1e-5
  ) -> torch.Tensor:
    sqrt_c = math.sqrt(c)
    u_sqnorm = torch.sum(u * u, dim=-1).clamp(max=1.0 - eps)
    v_sqnorm = torch.sum(v * v, dim=-1).clamp(max=1.0 - eps)
    diff_sqnorm = torch.sum((u - v) * (u - v), dim=-1)

    alpha = 1.0 - c * u_sqnorm
    beta = 1.0 - c * v_sqnorm

    denom = (alpha * beta).clamp(min=eps)
    x = 1.0 + 2.0 * c * diff_sqnorm / denom
    x = x.clamp(min=1.0)

    # arcosh(x) = log(x + sqrt(x^2 - 1))
    # Clamp term inside sqrt to >= 0
    sqrt_arg = (torch.square(x) - 1.0).clamp(min=eps)
    dist = torch.log(x + torch.sqrt(sqrt_arg)) / sqrt_c
    # Keep exact coincident-point distances at zero while the clamped
    # square root above keeps their backward pass finite.
    dist = torch.where(
      diff_sqnorm <= eps,
      torch.zeros_like(dist),
      dist,
    )
    return dist


class ProductManifoldEmbedding(nn.Module):
  """Maps feature representations to Product Manifold R^n x H^m."""

  def __init__(
    self,
    in_dim: int,
    euc_dim: int = 32,
    hyp_dim: int = 32,
    curvature: float = 1.0,
  ) -> None:
    super().__init__()
    self.in_dim = in_dim
    self.euc_dim = euc_dim
    self.hyp_dim = hyp_dim
    self.curvature = curvature

    # Linear projections for Euclidean and Hyperbolic streams
    self.proj_euc = nn.Linear(in_dim, euc_dim)
    self.proj_hyp = nn.Linear(in_dim, hyp_dim)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Maps input tensor x to (z_euc, z_hyp).

    Args:
          x: Input representation of shape (batch_size, in_dim).

    Returns:
          Tuple (z_euc, z_hyp) where z_euc in R^euc_dim and z_hyp in B^hyp_dim.
    """
    z_euc = self.proj_euc(x)
    h_raw = self.proj_hyp(x)
    z_hyp = PoincaréHyperbolicSpace.exp_map_zero(h_raw, c=self.curvature)
    return z_euc, z_hyp

  def compute_tree_loss(
    self,
    z_hyp: torch.Tensor,
    targets: torch.Tensor,
    family_targets: torch.Tensor,
    *,
    family_margin: float = 0.5,
    unrelated_margin: float = 2.0,
  ) -> torch.Tensor:
    """Keep relaxed-family variants close and unrelated labels separated.

    Args:
          z_hyp: Hyperbolic embeddings of shape (batch_size, hyp_dim).
          targets: Exact source-label indices of shape ``(batch_size,)``.
          family_targets: Serving-family indices of shape ``(batch_size,)``.

    Returns:
          Scalar loss tensor.
    """
    if z_hyp.shape[0] <= 1:
      return torch.zeros((), dtype=torch.float32, device=z_hyp.device)

    # Pairwise hyperbolic distance matrix
    u = z_hyp.unsqueeze(1)  # (batch, 1, hyp_dim)
    v = z_hyp.unsqueeze(0)  # (1, batch, hyp_dim)
    dist_matrix = PoincaréHyperbolicSpace.distance(u, v, c=self.curvature)

    diagonal = torch.eye(
      z_hyp.shape[0],
      dtype=torch.bool,
      device=z_hyp.device,
    )
    same_label = (targets.unsqueeze(1) == targets.unsqueeze(0)) & ~diagonal
    same_family = (
      (family_targets.unsqueeze(1) == family_targets.unsqueeze(0))
      & ~same_label
      & ~diagonal
    )
    unrelated = family_targets.unsqueeze(1) != family_targets.unsqueeze(0)

    losses: list[torch.Tensor] = []
    if same_label.any():
      losses.append(dist_matrix[same_label].mean())
    if same_family.any():
      losses.append(F.relu(dist_matrix[same_family] - family_margin).mean())
    if unrelated.any():
      losses.append(F.relu(unrelated_margin - dist_matrix[unrelated]).mean())
    if not losses:
      return z_hyp.sum() * 0.0
    return torch.stack(losses).mean()

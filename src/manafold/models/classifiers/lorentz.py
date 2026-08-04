"""Lorentz (Hyperboloid) manifold operations, Transformer blocks, and Prototypical Head.

Implements the Lorentz model L^d = {x in R^{d+1} : <x, x>_L = -1/c, x_0 > 0}.
Lorentz distance avoids division singularities (1 - ||x||^2 -> 0) of the Poincare ball,
providing smooth gradients across high-dimensional set representations.

Key design: residual connections and layer norms operate in the tangent space at the
origin (T_o L^d ≅ R^d), preserving standard Transformer gradient flow properties.
Points are mapped back to L^d via exp_0 only when geodesic distance computation is needed.
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def minkowski_inner_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Computes Minkowski inner product <u, v>_L = -u_0 v_0 + sum_{i=1}^d u_i v_i."""
  u_0 = u[..., 0]
  v_0 = v[..., 0]
  u_spatial = u[..., 1:]
  v_spatial = v[..., 1:]
  return -u_0 * v_0 + torch.sum(u_spatial * v_spatial, dim=-1)


def project_to_lorentz(x_spatial: torch.Tensor, c: float = 1.0) -> torch.Tensor:
  """Projects spatial vectors R^d onto the Lorentz hyperboloid L^d in R^{d+1}.

  x_0 = sqrt(1/c + ||x_{1:d}||^2)
  """
  spatial_norm_sq = torch.sum(x_spatial**2, dim=-1, keepdim=True)
  x_0 = torch.sqrt(1.0 / c + spatial_norm_sq)
  return torch.cat([x_0, x_spatial], dim=-1)


def exp_map_lorentz_zero(v_spatial: torch.Tensor, c: float = 1.0) -> torch.Tensor:
  """Projects tangent vector at origin e_0 into Lorentz space L^d.

  Includes tangent norm clamping to prevent exponential saturation.
  """
  sqrt_c = math.sqrt(c)
  v_norm_raw = torch.norm(v_spatial, dim=-1, keepdim=True)
  v_norm = v_norm_raw.clamp(min=1e-12, max=4.0 / sqrt_c)
  scale = torch.where(
    v_norm_raw > 1e-12,
    v_norm / v_norm_raw.clamp(min=1e-12),
    torch.ones_like(v_norm_raw),
  )
  v_spatial_scaled = v_spatial * scale

  x_0 = (1.0 / sqrt_c) * torch.cosh(sqrt_c * v_norm)
  x_spatial = torch.sinh(sqrt_c * v_norm) * (v_spatial_scaled / (sqrt_c * v_norm))
  return torch.cat([x_0, x_spatial], dim=-1)


def log_map_lorentz_zero(x_lorentz: torch.Tensor, c: float = 1.0) -> torch.Tensor:
  """Projects Lorentz point x in L^d to Euclidean tangent space at origin e_0."""
  sqrt_c = math.sqrt(c)
  x_0 = x_lorentz[..., :1]
  x_spatial = x_lorentz[..., 1:]
  x_spatial_norm = torch.norm(x_spatial, dim=-1, keepdim=True).clamp(min=1e-12)

  # arcosh(sqrt(c) * x_0)
  arg = (sqrt_c * x_0).clamp(min=1.0 + 1e-7)
  arcosh_val = torch.log(arg + torch.sqrt((arg**2 - 1.0).clamp(min=0.0)))

  return (arcosh_val / (sqrt_c * x_spatial_norm)) * x_spatial


def lorentz_distance(u: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
  """Computes Lorentz distance d_L^c(u, v) = (1/sqrt(c)) * arcosh(-c * <u, v>_L)."""
  inner = minkowski_inner_product(u, v)
  arg = (-c * inner).clamp(min=1.0 + 1e-7)
  # arcosh(x) = log(x + sqrt(x^2 - 1))
  dist = (1.0 / math.sqrt(c)) * torch.log(
    arg + torch.sqrt((arg**2 - 1.0).clamp(min=0.0))
  )
  # Clamp exact self-distance zero
  spatial_diff = torch.sum((u[..., 1:] - v[..., 1:]) ** 2, dim=-1)
  dist = torch.where(spatial_diff < 1e-12, torch.zeros_like(dist), dist)
  return dist


class HybridLorentzAttention(nn.Module):
  """Hybrid attention combining dot-product discrimination with Lorentz distance.

  Attention score: α_ij = q_i^T k_j / √d  -  λ · d_L(q_i, k_j)

  The dot-product term provides directional discrimination (positive AND negative
  correlations between cards), while the Lorentz distance term provides geometric
  distance awareness for partial-observation robustness. λ is a learned scalar.
  """

  def __init__(
    self,
    spatial_dim: int,
    num_heads: int = 4,
    curvature: float = 1.0,
    initial_lambda: float = 0.1,
  ) -> None:
    super().__init__()
    self.spatial_dim = spatial_dim
    self.num_heads = num_heads
    self.head_dim = spatial_dim // num_heads
    self.curvature = curvature
    self.scale = self.head_dim**-0.5

    self.q_proj = nn.Linear(spatial_dim, spatial_dim)
    self.k_proj = nn.Linear(spatial_dim, spatial_dim)
    self.v_proj = nn.Linear(spatial_dim, spatial_dim)
    self.o_proj = nn.Linear(spatial_dim, spatial_dim)

    # Learned mixing coefficient for Lorentz distance term
    self.log_lambda = nn.Parameter(torch.tensor(math.log(initial_lambda)))

  def forward(self, x_lorentz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Returns attention output in tangent space (R^d).

    Args:
          x_lorentz: (batch_size, seq_len, spatial_dim + 1) in L^d
          mask: (batch_size, seq_len) boolean mask

    Returns:
          (batch_size, seq_len, spatial_dim) tangent space output
    """
    B, N, _ = x_lorentz.shape
    H, D = self.num_heads, self.head_dim

    # Map to tangent space for linear projections
    tangent = log_map_lorentz_zero(x_lorentz, c=self.curvature)

    q = self.q_proj(tangent).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
    k = self.k_proj(tangent).view(B, N, H, D).transpose(1, 2)
    v = self.v_proj(tangent).view(B, N, H, D).transpose(1, 2)

    # --- Dot-product term: q·k / √d ---
    dot_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

    # --- Lorentz distance term: -λ · d_L(q, k) ---
    # Compute distances in full (non-head-split) Lorentz space for geometric fidelity
    q_full = self.q_proj(tangent)  # (B, N, D_full)
    k_full = self.k_proj(tangent)
    q_lorentz = exp_map_lorentz_zero(q_full, c=self.curvature)
    k_lorentz = exp_map_lorentz_zero(k_full, c=self.curvature)
    q_exp = q_lorentz.unsqueeze(2).expand(B, N, N, -1)
    k_exp = k_lorentz.unsqueeze(1).expand(B, N, N, -1)
    dists = lorentz_distance(q_exp, k_exp, c=self.curvature)  # (B, N, N)

    lam = self.log_lambda.exp()
    dist_scores = -lam * dists  # (B, N, N)

    # --- Combine: broadcast dist_scores across heads ---
    attn_logits = dot_scores + dist_scores.unsqueeze(1)  # (B, H, N, N)

    # Mask
    mask_exp = mask.unsqueeze(1).unsqueeze(2).expand(B, H, N, N)
    attn_logits = attn_logits.masked_fill(~mask_exp, -1e9)
    attn_weights = F.softmax(attn_logits, dim=-1)

    # Aggregate
    out = torch.matmul(attn_weights, v)  # (B, H, N, D)
    out = out.transpose(1, 2).contiguous().view(B, N, self.spatial_dim)
    return self.o_proj(out)


class LorentzTransformerBlock(nn.Module):
  """Full Transformer block operating on the Lorentz manifold.

  Architecture (per block):
      1. LayerNorm(tangent) → Lorentz Attention → residual add (in tangent space)
      2. LayerNorm(tangent) → FFN → residual add (in tangent space)
      3. exp_0 back to L^d

  Residual connections and layer norms live in T_o(L^d) ≅ R^d.
  This preserves standard Transformer gradient flow while keeping
  the geometric distance computation on the hyperboloid.
  """

  def __init__(
    self,
    spatial_dim: int,
    ffn_dim: int | None = None,
    num_heads: int = 4,
    temperature: float = 0.1,
    curvature: float = 1.0,
    dropout: float = 0.1,
  ) -> None:
    super().__init__()
    self.spatial_dim = spatial_dim
    self.curvature = curvature
    if ffn_dim is None:
      ffn_dim = spatial_dim * 4

    # Pre-norm attention sublayer
    self.norm1 = nn.LayerNorm(spatial_dim)
    self.attn = HybridLorentzAttention(
      spatial_dim=spatial_dim,
      num_heads=num_heads,
      curvature=curvature,
    )
    self.dropout1 = nn.Dropout(dropout)

    # Pre-norm FFN sublayer
    self.norm2 = nn.LayerNorm(spatial_dim)
    self.ffn = nn.Sequential(
      nn.Linear(spatial_dim, ffn_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(ffn_dim, spatial_dim),
      nn.Dropout(dropout),
    )

  def forward(self, x_lorentz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Input and output are both on L^d (batch_size, seq_len, spatial_dim + 1)."""
    # Map input to tangent space for residual path
    tangent = log_map_lorentz_zero(x_lorentz, c=self.curvature)

    # Sublayer 1: Attention with residual
    normed = self.norm1(tangent)
    normed_lorentz = exp_map_lorentz_zero(normed, c=self.curvature)
    attn_out = self.attn(normed_lorentz, mask)  # returns tangent space
    tangent = tangent + self.dropout1(attn_out)

    # Sublayer 2: FFN with residual
    normed = self.norm2(tangent)
    tangent = tangent + self.ffn(normed)

    # Map back to L^d
    return exp_map_lorentz_zero(tangent, c=self.curvature)


class LorentzPrototypicalHead(nn.Module):
  """Classification and OOD detection head using Lorentz prototypical distance."""

  def __init__(
    self,
    spatial_dim: int,
    num_classes: int,
    temperature: float = 0.1,
    curvature: float = 1.0,
  ) -> None:
    super().__init__()
    self.spatial_dim = spatial_dim
    self.num_classes = num_classes
    self.temperature = temperature
    self.curvature = curvature

    # Initialize archetype prototypes on L^d with wider spread
    proto_spatial = torch.randn(num_classes, spatial_dim) * 0.1
    self.prototypes = nn.Parameter(exp_map_lorentz_zero(proto_spatial, c=curvature))

  def forward(
    self, embeddings_lorentz: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Args:
          embeddings_lorentz: (batch_size, spatial_dim + 1) in L^d

    Returns:
          Tuple of (logits, ood_margin_scores)
    """
    B = embeddings_lorentz.size(0)
    K = self.num_classes

    embeds_exp = embeddings_lorentz.unsqueeze(1).expand(B, K, -1)
    protos_exp = self.prototypes.unsqueeze(0).expand(B, K, -1)

    dists = lorentz_distance(embeds_exp, protos_exp, c=self.curvature)  # (B, K)

    logits = -dists / self.temperature
    ood_margins, _ = torch.min(dists, dim=-1)

    return logits, ood_margins

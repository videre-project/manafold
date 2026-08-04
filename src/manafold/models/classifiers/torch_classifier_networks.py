from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from manafold.models.classifiers.config import (
  POOLING_MEAN,
  POOLING_QUANTITY_WEIGHTED,
  SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
  SET_TRANSFORMER_POOLING_PMA,
  SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED,
)


def _padded_tokens(
  token_embeddings: torch.Tensor,
  deck_idx: torch.Tensor,
  deck_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  if token_embeddings.numel() == 0:
    padded = token_embeddings.new_zeros(
      (deck_count, 1, token_embeddings.shape[-1])
    )
    padding_mask = torch.ones(
      (deck_count, 1),
      dtype=torch.bool,
      device=token_embeddings.device,
    )
    return padded, padding_mask

  token_counts = torch.bincount(deck_idx, minlength=deck_count)
  max_tokens = int(token_counts.max().item())
  padded = token_embeddings.new_zeros(
    (deck_count, max_tokens, token_embeddings.shape[-1])
  )
  padding_mask = torch.ones(
    (deck_count, max_tokens),
    dtype=torch.bool,
    device=token_embeddings.device,
  )
  token_offsets = torch.cumsum(token_counts, dim=0) - token_counts
  token_positions = (
    torch.arange(
      len(deck_idx),
      dtype=torch.long,
      device=token_embeddings.device,
    )
    - token_offsets[deck_idx]
  )
  padded[deck_idx, token_positions] = token_embeddings
  padding_mask[deck_idx, token_positions] = False

  return padded, padding_mask


class _ResidualBlock(nn.Module):

  def __init__(self, hidden_dim: int, *, dropout: float) -> None:
    super().__init__()
    self.norm = nn.LayerNorm(hidden_dim)
    self.layers = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, hidden_dim),
      nn.Dropout(dropout),
    )

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    return values + self.layers(self.norm(values))


class _DeepSetsNetwork(nn.Module):

  def __init__(
    self,
    *,
    card_count: int,
    zone_count: int,
    quantity_count: int,
    label_count: int,
    embedding_dim: int,
    hidden_dim: int,
    pooling: str,
    package_count: int,
    package_projection_dim: int,
    package_projection_bias: bool,
    rho_hidden_dim: int,
    extra_feature_dim: int,
    extra_feature_value: float,
    preserve_base_rho_init: bool,
    dropout: float = 0.0,
  ) -> None:
    super().__init__()
    self.pooling = pooling
    self.package_count = package_count
    self.hidden_dim = hidden_dim
    self.package_projection_dim = package_projection_dim
    self.extra_feature_dim = extra_feature_dim
    self.extra_feature_value = extra_feature_value
    self.card_embedding = nn.Embedding(card_count, embedding_dim)
    self.zone_embedding = nn.Embedding(zone_count, embedding_dim)
    self.quantity_embedding = nn.Embedding(quantity_count, embedding_dim)
    self.phi = nn.Sequential(
      nn.Linear(embedding_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
    )
    if package_count > 0 and package_projection_dim > 0:
      self.package_projection = nn.Sequential(
        nn.Linear(
          package_count,
          package_projection_dim,
          bias=package_projection_bias,
        ),
        nn.ReLU(),
      )
      rho_input_dim = hidden_dim + package_projection_dim
    else:
      self.package_projection = None
      rho_input_dim = hidden_dim
    rho_input_dim += extra_feature_dim

    rho_first = _make_rho_first_layer(
      hidden_dim=hidden_dim,
      rho_input_dim=rho_input_dim,
      rho_hidden_dim=rho_hidden_dim,
      preserve_base_rho_init=preserve_base_rho_init,
    )
    self.rho = nn.Sequential(
      rho_first,
      nn.ReLU(),
      nn.Linear(rho_hidden_dim, label_count),
    )

  def forward(self, batch: _Batch) -> torch.Tensor:
    deck_embedding = self.encode(batch)
    parts = [deck_embedding]
    if self.package_projection is not None:
      parts.append(self.package_projection(batch.package_features))
    if self.extra_feature_dim > 0:
      parts.append(
        deck_embedding.new_full(
          (batch.deck_count, self.extra_feature_dim),
          float(self.extra_feature_value),
        )
      )
    return self.rho(
      torch.cat(parts, dim=1) if len(parts) > 1 else deck_embedding
    )

  def encode(self, batch: _Batch) -> torch.Tensor:
    token_embeddings = (
      self.card_embedding(batch.card_idx)
      + self.zone_embedding(batch.zone_idx)
      + self.quantity_embedding(batch.quantity_idx)
    )
    token_latents = self.phi(token_embeddings)
    if self.pooling == POOLING_QUANTITY_WEIGHTED:
      token_latents = token_latents * batch.quantity_weight.unsqueeze(1)

    pooled = token_latents.new_zeros(
      (batch.deck_count, token_latents.shape[-1])
    )
    pooled.index_add_(0, batch.deck_idx, token_latents)
    if self.pooling == POOLING_MEAN:
      counts = token_latents.new_zeros((batch.deck_count, 1))
      counts.index_add_(
        0,
        batch.deck_idx,
        torch.ones((len(batch.deck_idx), 1), device=token_latents.device),
      )
      pooled = pooled / counts.clamp_min(1.0)

    return pooled

  def forward_onnx(
    self,
    card_idx: torch.Tensor,
    zone_idx: torch.Tensor,
    quantity_idx: torch.Tensor,
    quantity_weight: torch.Tensor,
    deck_idx: torch.Tensor,
    deck_count: torch.Tensor,
    package_features: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """ONNX-exportable forward pass taking raw tensors instead of _Batch."""
    token_embeddings = (
      self.card_embedding(card_idx)
      + self.zone_embedding(zone_idx)
      + self.quantity_embedding(quantity_idx)
    )
    token_latents = self.phi(token_embeddings)
    if self.pooling == POOLING_QUANTITY_WEIGHTED:
      token_latents = token_latents * quantity_weight.unsqueeze(1)

    dc = int(deck_count.item()) if deck_count.numel() == 1 else deck_count
    pooled = token_latents.new_zeros((dc, token_latents.shape[-1]))
    pooled.scatter_reduce_(
      0,
      deck_idx.unsqueeze(-1).expand_as(token_latents),
      token_latents,
      reduce="sum",
    )
    if self.pooling == POOLING_MEAN:
      counts = token_latents.new_zeros((dc, 1))
      counts.scatter_reduce_(
        0,
        deck_idx.unsqueeze(-1),
        torch.ones_like(deck_idx, dtype=token_latents.dtype).unsqueeze(-1),
        reduce="sum",
      )
      pooled = pooled / counts.clamp_min(1.0)

    parts = [pooled]
    if self.package_projection is not None and package_features is not None:
      parts.append(self.package_projection(package_features))
    if self.extra_feature_dim > 0:
      parts.append(
        pooled.new_full(
          (dc, self.extra_feature_dim),
          float(self.extra_feature_value),
        )
      )
    return self.rho(torch.cat(parts, dim=1) if len(parts) > 1 else pooled)


class _DeepSetsPlusPlusNetwork(nn.Module):

  def __init__(
    self,
    *,
    card_count: int,
    zone_count: int,
    quantity_count: int,
    label_count: int,
    embedding_dim: int,
    hidden_dim: int,
    pooling: str,
    package_count: int,
    package_projection_dim: int,
    package_projection_bias: bool,
    rho_hidden_dim: int,
    extra_feature_dim: int,
    extra_feature_value: float,
    preserve_base_rho_init: bool,
    dropout: float = 0.1,
  ) -> None:
    super().__init__()
    if package_count > 0:
      raise ValueError("Deep Sets++ does not support package features.")
    if extra_feature_dim > 0 or preserve_base_rho_init:
      raise ValueError(
        "Deep Sets++ does not support head attribution controls."
      )

    self.pooling = pooling
    self.package_count = 0
    self.hidden_dim = hidden_dim
    self.package_projection_dim = package_projection_dim
    self.extra_feature_dim = extra_feature_dim
    self.extra_feature_value = extra_feature_value
    self.package_projection = None
    self.card_embedding = nn.Embedding(card_count, embedding_dim)
    self.zone_embedding = nn.Embedding(zone_count, embedding_dim)
    self.quantity_embedding = nn.Embedding(quantity_count, embedding_dim)
    self.input_projection = nn.Linear(embedding_dim, hidden_dim)
    self.phi = nn.Sequential(
      _ResidualBlock(hidden_dim, dropout=dropout),
      _ResidualBlock(hidden_dim, dropout=dropout),
    )
    self.rho = nn.Sequential(
      _ResidualBlock(hidden_dim, dropout=dropout),
      _ResidualBlock(hidden_dim, dropout=dropout),
      nn.LayerNorm(hidden_dim),
      nn.Linear(hidden_dim, label_count),
    )

  def forward(self, batch: _Batch) -> torch.Tensor:
    return self.rho(self.encode(batch))

  def encode(self, batch: _Batch) -> torch.Tensor:
    token_embeddings = (
      self.card_embedding(batch.card_idx)
      + self.zone_embedding(batch.zone_idx)
      + self.quantity_embedding(batch.quantity_idx)
    )
    token_latents = self.phi(self.input_projection(token_embeddings))
    if self.pooling == POOLING_QUANTITY_WEIGHTED:
      token_latents = token_latents * batch.quantity_weight.unsqueeze(1)

    pooled = token_latents.new_zeros(
      (batch.deck_count, token_latents.shape[-1])
    )
    pooled.index_add_(0, batch.deck_idx, token_latents)
    if self.pooling == POOLING_MEAN:
      counts = token_latents.new_zeros((batch.deck_count, 1))
      counts.index_add_(
        0,
        batch.deck_idx,
        torch.ones((len(batch.deck_idx), 1), device=token_latents.device),
      )
      pooled = pooled / counts.clamp_min(1.0)

    return pooled

  def forward_onnx(
    self,
    card_idx: torch.Tensor,
    zone_idx: torch.Tensor,
    quantity_idx: torch.Tensor,
    quantity_weight: torch.Tensor,
    deck_idx: torch.Tensor,
    deck_count: torch.Tensor,
    package_features: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """ONNX-exportable forward pass taking raw tensors instead of _Batch."""
    token_embeddings = (
      self.card_embedding(card_idx)
      + self.zone_embedding(zone_idx)
      + self.quantity_embedding(quantity_idx)
    )
    token_latents = self.phi(self.input_projection(token_embeddings))
    if self.pooling == POOLING_QUANTITY_WEIGHTED:
      token_latents = token_latents * quantity_weight.unsqueeze(1)

    dc = int(deck_count.item()) if deck_count.numel() == 1 else deck_count
    pooled = token_latents.new_zeros((dc, token_latents.shape[-1]))
    pooled.scatter_reduce_(
      0,
      deck_idx.unsqueeze(-1).expand_as(token_latents),
      token_latents,
      reduce="sum",
    )
    if self.pooling == POOLING_MEAN:
      counts = token_latents.new_zeros((dc, 1))
      counts.scatter_reduce_(
        0,
        deck_idx.unsqueeze(-1),
        torch.ones_like(deck_idx, dtype=token_latents.dtype).unsqueeze(-1),
        reduce="sum",
      )
      pooled = pooled / counts.clamp_min(1.0)

    return self.rho(pooled)


def _make_rho_first_layer(
  *,
  hidden_dim: int,
  rho_input_dim: int,
  rho_hidden_dim: int,
  preserve_base_rho_init: bool,
) -> nn.Linear:
  if not preserve_base_rho_init or rho_input_dim == hidden_dim:
    return nn.Linear(rho_input_dim, rho_hidden_dim)

  base_layer = nn.Linear(hidden_dim, rho_hidden_dim)
  rng_state = torch.get_rng_state()
  rho_first = nn.Linear(rho_input_dim, rho_hidden_dim)
  torch.set_rng_state(rng_state)
  with torch.no_grad():
    rho_first.weight.zero_()
    rho_first.weight[:, :hidden_dim].copy_(base_layer.weight)
    rho_first.bias.copy_(base_layer.bias)
  return rho_first


def _module_parameter_norm(module: nn.Module) -> float:
  values = [
    parameter.detach().flatten() for parameter in module.parameters()
  ]
  if not values:
    return 0.0
  return float(torch.cat(values).norm().item())


def _module_gradient_norm(module: nn.Module) -> float | None:
  values = [
    parameter.grad.detach().flatten()
    for parameter in module.parameters()
    if parameter.grad is not None
  ]
  if not values:
    return None
  return float(torch.cat(values).norm().item())


class _SetTransformerNetwork(nn.Module):

  def __init__(
    self,
    *,
    card_count: int,
    zone_count: int,
    quantity_count: int,
    label_count: int,
    embedding_dim: int,
    hidden_dim: int,
    attention_heads: int,
    attention_layers: int,
    pooling: str,
    observation_conditioning: bool = False,
    anchor_gating: bool = False,
    labels: tuple[str, ...] = (),
    use_product_manifold: bool = False,
  ) -> None:
    super().__init__()
    self.pooling = pooling
    self.observation_conditioning = observation_conditioning
    self.attention_heads = attention_heads
    self.use_product_manifold = use_product_manifold
    self.card_embedding = nn.Embedding(card_count, embedding_dim)
    self.zone_embedding = nn.Embedding(zone_count, embedding_dim)
    self.quantity_embedding = nn.Embedding(quantity_count, embedding_dim)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=embedding_dim,
      nhead=attention_heads,
      dim_feedforward=hidden_dim,
      dropout=0.0,
      activation="relu",
      batch_first=True,
      norm_first=True,
    )
    self.encoder = nn.TransformerEncoder(
      encoder_layer,
      num_layers=attention_layers,
      enable_nested_tensor=False,
    )
    if pooling == SET_TRANSFORMER_POOLING_PMA:
      self.seed_vector = nn.Parameter(torch.empty(1, 1, embedding_dim))
      self.pooling_attention = nn.MultiheadAttention(
        embed_dim=embedding_dim,
        num_heads=attention_heads,
        dropout=0.0,
        batch_first=True,
      )
      nn.init.normal_(self.seed_vector, mean=0.0, std=0.02)
    else:
      self.seed_vector = None
      self.pooling_attention = None
    if anchor_gating:
      from manafold.models.classifiers.pmi_anchor_gate import PMIAnchorGate

      self.anchor_gate: nn.Module | None = PMIAnchorGate(
        card_count=card_count,
        labels=labels,
        examples=(),
      )
    else:
      self.anchor_gate = None
    if use_product_manifold:
      from manafold.models.classifiers.product_manifold import (
        ProductManifoldEmbedding,
      )

      self.product_manifold: nn.Module | None = ProductManifoldEmbedding(
        in_dim=embedding_dim,
        euc_dim=embedding_dim,
        hyp_dim=embedding_dim,
      )
      output_dim = embedding_dim * 2
    else:
      self.product_manifold = None
      output_dim = embedding_dim
    self.output_dim = output_dim
    self.rho = nn.Sequential(
      nn.LayerNorm(output_dim),
      nn.Linear(output_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, label_count),
    )
    self.observation_projection = (
      nn.Sequential(
        nn.Linear(3, embedding_dim),
        nn.Tanh(),
      )
      if observation_conditioning
      else None
    )

  def forward(self, batch: _Batch) -> torch.Tensor:
    return self.rho(self.encode(batch))

  def forward_onnx(
    self,
    card_idx: torch.Tensor,
    zone_idx: torch.Tensor,
    quantity_idx: torch.Tensor,
    quantity_weight: torch.Tensor,
    deck_idx: torch.Tensor,
    deck_count: torch.Tensor,
    package_features: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """ONNX-exportable single-deck forward pass for Worker inference."""
    del deck_idx, deck_count, package_features
    if self.observation_projection is not None:
      raise ValueError(
        "ONNX export does not support observation-conditioned Set"
        " Transformers."
      )
    token_embeddings = (
      self.card_embedding(card_idx)
      + self.zone_embedding(zone_idx)
      + self.quantity_embedding(quantity_idx)
    )
    anchor_bias = (
      self.anchor_gate(card_idx).unsqueeze(-1)
      if self.anchor_gate is not None
      else None
    )
    if anchor_bias is not None:
      token_embeddings = token_embeddings + anchor_bias
    anchor_bias = None
    encoded = token_embeddings.unsqueeze(0)
    for layer in self.encoder.layers:
      normed = layer.norm1(encoded)
      encoded = encoded + _onnx_multihead_attention(
        normed,
        normed,
        normed,
        layer.self_attn,
        key_bias=anchor_bias,
      )
      encoded = encoded + layer.linear2(
        F.relu(layer.linear1(layer.norm2(encoded)))
      )
    if self.pooling in (
      SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED,
      SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
    ):
      pooled = (encoded * quantity_weight.reshape(1, -1, 1)).sum(dim=1)
    else:
      if self.seed_vector is None or self.pooling_attention is None:
        raise AssertionError(
          f"Unhandled Set Transformer pooling mode: {self.pooling}"
        )
      pooled = _onnx_multihead_attention(
        self.seed_vector,
        encoded,
        encoded,
        self.pooling_attention,
        key_bias=anchor_bias,
      )
      pooled = pooled.squeeze(1)
    return self.rho(self._product_embedding(pooled))

  def encode(self, batch: _Batch) -> torch.Tensor:
    pooled, _ = self.encode_with_contextual_tokens(batch)
    return pooled

  def encode_contextual_tokens(
    self,
    batch: _Batch,
    *,
    use_anchor_gating: bool = True,
  ) -> torch.Tensor:
    _, contextual_tokens = self.encode_with_contextual_tokens(
      batch,
      use_anchor_gating=use_anchor_gating,
    )
    return contextual_tokens

  def encode_with_contextual_tokens(
    self,
    batch: _Batch,
    *,
    use_anchor_gating: bool = True,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    token_embeddings = (
      self.card_embedding(batch.card_idx)
      + self.zone_embedding(batch.zone_idx)
      + self.quantity_embedding(batch.quantity_idx)
    )
    if self.anchor_gate is not None and use_anchor_gating:
      token_embeddings = token_embeddings + self.anchor_gate(
        batch.card_idx
      ).unsqueeze(-1)
    padded, padding_mask = _padded_tokens(
      token_embeddings,
      batch.deck_idx,
      batch.deck_count,
    )
    encoded = self.encoder(
      padded,
      src_key_padding_mask=padding_mask,
    )
    token_counts = torch.bincount(batch.deck_idx, minlength=batch.deck_count)
    token_offsets = torch.cumsum(token_counts, dim=0) - token_counts
    token_positions = (
      torch.arange(
        len(batch.deck_idx),
        dtype=torch.long,
        device=batch.deck_idx.device,
      )
      - token_offsets[batch.deck_idx]
    )
    contextual_tokens = encoded[batch.deck_idx, token_positions]
    if self.pooling in (
      SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED,
      SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
    ):
      quantity_weights, _ = _padded_tokens(
        batch.quantity_weight.unsqueeze(1),
        batch.deck_idx,
        batch.deck_count,
      )
      pooled = (encoded * quantity_weights).sum(dim=1)
      pooled = self._condition_pooled(pooled, batch)
      return self._product_embedding(pooled), contextual_tokens

    if self.seed_vector is None or self.pooling_attention is None:
      raise AssertionError(
        f"Unhandled Set Transformer pooling mode: {self.pooling}"
      )
    seed = self.seed_vector.expand(batch.deck_count, -1, -1)
    pooled, _ = self.pooling_attention(
      seed,
      encoded,
      encoded,
      key_padding_mask=padding_mask,
      need_weights=False,
    )
    pooled = self._condition_pooled(pooled.squeeze(1), batch)
    return self._product_embedding(pooled), contextual_tokens

  def _product_embedding(self, pooled: torch.Tensor) -> torch.Tensor:
    if self.product_manifold is None:
      return pooled
    euclidean, hyperbolic = self.product_manifold(pooled)
    return torch.cat((euclidean, hyperbolic), dim=1)

  def hyperbolic_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
    if self.product_manifold is None:
      raise ValueError("This Set Transformer has no product-manifold stream.")
    return embedding[:, -self.product_manifold.hyp_dim :]

  def _condition_pooled(
    self,
    pooled: torch.Tensor,
    batch: _Batch,
  ) -> torch.Tensor:
    if self.observation_projection is None:
      return pooled
    identity_counts = torch.bincount(
      batch.deck_idx,
      minlength=batch.deck_count,
    ).to(dtype=pooled.dtype)
    identity_scale = math.log1p(60.0)
    features = torch.stack(
      (
        batch.observation_coverage.to(dtype=pooled.dtype),
        torch.log1p(identity_counts) / identity_scale,
        batch.observation_complete.to(dtype=pooled.dtype),
      ),
      dim=1,
    )
    return pooled + self.observation_projection(features)


def _onnx_multihead_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attention: nn.MultiheadAttention,
  *,
  key_bias: torch.Tensor | None = None,
) -> torch.Tensor:
  """Single-deck attention expressed with ONNX operators and dynamic lengths."""
  embedding_dim = attention.embed_dim
  head_count = attention.num_heads
  head_dim = embedding_dim // head_count
  projection_weight = attention.in_proj_weight
  projection_bias = attention.in_proj_bias
  if projection_weight is None:
    raise ValueError(
      "ONNX export requires packed attention projection weights."
    )

  query_projection = F.linear(
    query,
    projection_weight[:embedding_dim],
    projection_bias[:embedding_dim] if projection_bias is not None else None,
  )
  key_projection = F.linear(
    key,
    projection_weight[embedding_dim : 2 * embedding_dim],
    (
      projection_bias[embedding_dim : 2 * embedding_dim]
      if projection_bias is not None
      else None
    ),
  )
  value_projection = F.linear(
    value,
    projection_weight[2 * embedding_dim :],
    (
      projection_bias[2 * embedding_dim :]
      if projection_bias is not None
      else None
    ),
  )
  query_heads = query_projection.reshape(
    1,
    -1,
    head_count,
    head_dim,
  ).transpose(1, 2)
  key_heads = key_projection.reshape(
    1,
    -1,
    head_count,
    head_dim,
  ).transpose(1, 2)
  value_heads = value_projection.reshape(
    1,
    -1,
    head_count,
    head_dim,
  ).transpose(1, 2)
  scores = torch.matmul(
    query_heads,
    key_heads.transpose(-2, -1),
  ) / math.sqrt(float(head_dim))
  if key_bias is not None:
    scores = scores + key_bias.reshape(1, 1, 1, -1)
  attended = torch.matmul(torch.softmax(scores, dim=-1), value_heads)
  merged = attended.transpose(1, 2).reshape(1, -1, embedding_dim)
  return attention.out_proj(merged)


class _PartialLatentPredictor(nn.Module):

  def __init__(self, *, embedding_dim: int, hidden_dim: int) -> None:
    super().__init__()
    self.layers = nn.Sequential(
      nn.LayerNorm(embedding_dim),
      nn.Linear(embedding_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, embedding_dim),
    )

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    return self.layers(values)

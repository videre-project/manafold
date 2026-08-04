from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from manafold.models.classifiers.input_batches import _Batch


def _partial_latent_prediction_loss(
  *,
  predicted: torch.Tensor,
  target: torch.Tensor,
) -> torch.Tensor:
  return (1.0 - F.cosine_similarity(predicted, target.detach(), dim=1)).mean()


def _partial_contextual_prediction_loss(
  *,
  predicted: torch.Tensor,
  target: torch.Tensor,
  deck_idx: torch.Tensor,
  deck_count: int,
) -> torch.Tensor:
  token_losses = 1.0 - F.cosine_similarity(
    predicted,
    target.detach(),
    dim=1,
  )
  deck_losses = token_losses.new_zeros(deck_count)
  deck_losses.index_add_(0, deck_idx, token_losses)
  token_counts = torch.bincount(deck_idx, minlength=deck_count).clamp_min(1)
  return (deck_losses / token_counts).mean()


def _matching_full_token_indexes(
  *,
  full_batch: _Batch,
  partial_batch: _Batch,
  card_count: int,
  zone_count: int,
) -> torch.Tensor:
  full_keys = (
    (full_batch.deck_idx * card_count + full_batch.card_idx) * zone_count
    + full_batch.zone_idx
  )
  partial_keys = (
    (partial_batch.deck_idx * card_count + partial_batch.card_idx) * zone_count
    + partial_batch.zone_idx
  )
  sorted_keys, sorted_indexes = torch.sort(full_keys)
  positions = torch.searchsorted(sorted_keys, partial_keys)
  if bool((positions >= len(sorted_keys)).any().item()):
    raise ValueError("Partial view contains a token absent from its full deck.")
  matching_indexes = sorted_indexes.index_select(0, positions)
  if not torch.equal(
    full_keys.index_select(0, matching_indexes),
    partial_keys,
  ):
    raise ValueError("Partial view token identity does not match its full deck.")
  return matching_indexes


def _update_ema_module(
  teacher: nn.Module,
  student: nn.Module,
  *,
  decay: float,
) -> None:
  teacher_parameters = dict(teacher.named_parameters())
  for name, student_parameter in student.named_parameters():
    teacher_parameters[name].mul_(decay).add_(
      student_parameter.detach(),
      alpha=1.0 - decay,
    )
  teacher_buffers = dict(teacher.named_buffers())
  for name, student_buffer in student.named_buffers():
    teacher_buffers[name].copy_(student_buffer.detach())


def _partial_observation_losses(
  *,
  full_logits: torch.Tensor,
  partial_logits: torch.Tensor,
  target_idx: torch.Tensor,
  coverage: torch.Tensor,
  label_smoothing: float,
  min_coverage_weight: float,
  soft_target_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  coverage_weights = (
    min_coverage_weight
    + (1.0 - min_coverage_weight) * coverage.clamp(0.0, 1.0)
  )
  weight_sum = coverage_weights.sum().clamp_min(1e-8)

  if soft_target_matrix is None:
    partial_ce_rows = F.cross_entropy(
      partial_logits,
      target_idx,
      label_smoothing=label_smoothing,
      reduction="none",
    )
  else:
    target_distribution = soft_target_matrix.index_select(0, target_idx)
    partial_ce_rows = -(
      target_distribution * F.log_softmax(partial_logits, dim=1)
    ).sum(dim=1)

  partial_ce = (partial_ce_rows * coverage_weights).sum() / weight_sum
  teacher_probabilities = F.softmax(full_logits.detach(), dim=1)
  consistency_rows = F.kl_div(
    F.log_softmax(partial_logits, dim=1),
    teacher_probabilities,
    reduction="none",
  ).sum(dim=1)
  consistency = (consistency_rows * coverage_weights).sum() / weight_sum
  
  return partial_ce, consistency, coverage.mean()

from __future__ import annotations

from typing import Any

import torch

from manafold.datasets.model_inputs import DeckModelInput


def _example_batches(
  examples: list[DeckModelInput],
  batch_size: int,
) -> list[list[DeckModelInput]]:
  return [
    examples[index:index + batch_size]
    for index in range(0, len(examples), batch_size)
  ]


def _index_batches(
  indexes: list[int],
  batch_size: int,
) -> list[list[int]]:
  return [
    indexes[index:index + batch_size]
    for index in range(0, len(indexes), batch_size)
  ]


def _sequential_index_batches(
  item_count: int,
  batch_size: int,
) -> list[list[int]]:
  return _index_batches(list(range(item_count)), max(1, batch_size))


def _should_continue_training(
  completed_epochs: int,
  epochs: int,
  optimizer_steps: int,
  max_steps: int | None,
) -> bool:
  if max_steps is not None:
    return optimizer_steps < max_steps
  return completed_epochs < epochs


def _resolve_device(device: str) -> torch.device:
  if device == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return torch.device(device)


def _empty_training_summary() -> dict[str, Any]:
  return {
    "history": [],
    "best_validation_epoch": None,
    "best_validation_metric": None,
    "best_validation_loss": None,
    "optimizer_steps": 0,
    "completed_epochs": 0,
  }


def _is_better_validation(
  best_metric: float | None,
  best_loss: float | None,
  validation_accuracy: float | None,
  validation_loss: float | None,
) -> bool:
  if validation_accuracy is None:
    return False
  if best_metric is None or validation_accuracy > best_metric:
    return True
  if validation_accuracy < best_metric:
    return False
  if validation_loss is None:
    return False
  return best_loss is None or validation_loss < best_loss

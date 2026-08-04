from __future__ import annotations

import random
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from manafold.datasets.model_inputs import DeckModelInput
from manafold.models.classifiers.input_batches import (
  _DenseLinearExamples,
  _PreparedLinearExamples,
  _dense_linear_batch,
  _dense_linear_examples,
  _prepare_linear_examples,
  _prepared_linear_batch,
)
from manafold.models.classifiers.prediction_statistics import (
  _empty_prediction_stats,
  _logit_stats,
)
from manafold.models.classifiers.training_control import (
  _empty_training_summary,
  _index_batches,
  _is_better_validation,
  _resolve_device,
  _sequential_index_batches,
  _should_continue_training,
)
from manafold.models.features.card_packages import PackageFeatureSet


class PooledLinearClassifier:
  """Linear softmax baseline over normalized card-zone count features."""

  def __init__(
    self,
    *,
    labels: tuple[str, ...],
    card_count: int,
    zone_count: int,
    learning_rate: float = 0.5,
    weight_decay: float = 0.0,
    seed: int = 13,
    device: str = "auto",
    package_features: PackageFeatureSet | None = None,
    package_only: bool = False,
    package_scale: float = 1.0,
  ) -> None:
    if not labels:
      raise ValueError("PooledLinearClassifier requires at least one label.")
    if card_count <= 0:
      raise ValueError("card_count must be positive.")
    if zone_count <= 0:
      raise ValueError("zone_count must be positive.")

    self.labels = labels
    self.card_count = card_count
    self.zone_count = zone_count
    if package_scale < 0:
      raise ValueError("package_scale must be non-negative.")
    if weight_decay < 0:
      raise ValueError("weight_decay must be non-negative.")

    self.card_feature_count = card_count * zone_count
    self.package_only = package_only
    self.base_feature_count = 0 if package_only else self.card_feature_count
    self.package_features = package_features
    self.package_count = (
      0 if package_features is None else len(package_features)
    )
    self.feature_count = self.base_feature_count + self.package_count
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.package_scale = package_scale
    self.device = _resolve_device(device)
    self._label_to_idx = {
      label: index for index, label in enumerate(labels)
    }
    self._rng = random.Random(seed)
    self._package_index_by_deck: dict[str, tuple[int, ...]] = {}
    torch.manual_seed(seed)
    self._linear = nn.Linear(self.feature_count, len(labels)).to(self.device)

  def fit(
    self,
    examples: list[DeckModelInput],
    *,
    validation_examples: list[DeckModelInput] | None = None,
    epochs: int = 40,
    batch_size: int = 32,
    shuffle: bool = True,
    max_steps: int | None = None,
  ) -> dict[str, Any]:
    if batch_size <= 0:
      raise ValueError("batch_size must be positive.")
    if max_steps is not None and max_steps <= 0:
      raise ValueError("max_steps must be positive.")

    trainable = [
      example
      for example in examples
      if example.target_label_id in self._label_to_idx
    ]
    validation = [
      example
      for example in (validation_examples or [])
      if example.target_label_id in self._label_to_idx
    ]
    if not trainable:
      return _empty_training_summary()
    self._ensure_package_cache([*trainable, *validation])
    trainable_prepared = _prepare_linear_examples(
      trainable,
      label_to_idx=self._label_to_idx,
      zone_count=self.zone_count,
      package_index_by_deck=self._package_index_by_deck,
      package_feature_offset=self.base_feature_count,
      package_scale=self.package_scale,
      include_card_features=not self.package_only,
    )
    validation_prepared = _prepare_linear_examples(
      validation,
      label_to_idx=self._label_to_idx,
      zone_count=self.zone_count,
      package_index_by_deck=self._package_index_by_deck,
      package_feature_offset=self.base_feature_count,
      package_scale=self.package_scale,
      include_card_features=not self.package_only,
    )
    trainable_dense = _dense_linear_examples(
      trainable_prepared,
      feature_count=self.feature_count,
      device=self.device,
    )
    validation_dense = _dense_linear_examples(
      validation_prepared,
      feature_count=self.feature_count,
      device=self.device,
    )

    history: list[dict[str, float | int | None]] = []
    best_state = self._state()
    best_epoch: int | None = None
    best_metric: float | None = None
    best_loss: float | None = None
    optimizer = torch.optim.Adam(
      self._linear.parameters(),
      lr=self.learning_rate,
      weight_decay=self.weight_decay,
    )
    optimizer_steps = 0

    epoch = 0
    while _should_continue_training(epoch, epochs, optimizer_steps, max_steps):
      epoch += 1
      epoch_indexes = list(range(len(trainable)))
      if shuffle:
        self._rng.shuffle(epoch_indexes)

      total_loss = 0.0
      total_correct = 0
      total_count = 0
      self._linear.train()
      for batch_indexes in _index_batches(epoch_indexes, max(1, batch_size)):
        if max_steps is not None and optimizer_steps >= max_steps:
          break
        batch = _dense_linear_batch(
          trainable_dense,
          batch_indexes,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = self._linear(batch.features)
        loss = F.cross_entropy(logits, batch.target_idx)
        loss.backward()
        optimizer.step()
        optimizer_steps += 1

        batch_count = len(batch_indexes)
        total_loss += float(loss.item()) * batch_count
        total_correct += int(
          (logits.argmax(dim=1) == batch.target_idx).sum().item()
        )
        total_count += batch_count

      train_loss, train_accuracy = self._loss_accuracy_dense(trainable_dense)
      validation_loss, validation_accuracy = self._loss_accuracy_dense(
        validation_dense
      )
      if validation and _is_better_validation(
        best_metric,
        best_loss,
        validation_accuracy,
        validation_loss,
      ):
        best_epoch = epoch
        best_metric = validation_accuracy
        best_loss = validation_loss
        best_state = self._state()

      history.append({
        "epoch": epoch,
        "loss": total_loss / total_count if total_count else None,
        "accuracy": total_correct / total_count if total_count else None,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "optimizer_steps": optimizer_steps,
      })

    if validation:
      self._restore_state(best_state)

    return {
      "history": history,
      "best_validation_epoch": best_epoch,
      "best_validation_metric": best_metric,
      "best_validation_loss": best_loss,
      "optimizer_steps": optimizer_steps,
      "completed_epochs": epoch,
    }

  def predict(self, example: DeckModelInput) -> tuple[str | None, float]:
    top = self.predict_top_k(example, k=1)
    if not top:
      return None, 0.0
    return top[0]

  def predict_top_k(
    self,
    example: DeckModelInput,
    *,
    k: int,
  ) -> list[tuple[str, float]]:
    if k <= 0 or not self.labels:
      return []

    return self.predict_top_k_many([example], k=k)[0]

  def predict_top_k_many(
    self,
    examples: list[DeckModelInput],
    *,
    k: int,
    batch_size: int = 32,
  ) -> list[list[tuple[str, float]]]:
    return [
      row["top_predictions"]
      for row in self.predict_top_k_with_stats_many(
        examples,
        k=k,
        batch_size=batch_size,
      )
    ]

  def predict_top_k_with_stats_many(
    self,
    examples: list[DeckModelInput],
    *,
    k: int,
    batch_size: int = 32,
  ) -> list[dict[str, Any]]:
    if k <= 0:
      return [_empty_prediction_stats() for _ in examples]

    predictions: list[dict[str, Any]] = []
    self._ensure_package_cache(examples)
    prepared = _prepare_linear_examples(
      examples,
      label_to_idx=self._label_to_idx,
      zone_count=self.zone_count,
      package_index_by_deck=self._package_index_by_deck,
      package_feature_offset=self.base_feature_count,
      package_scale=self.package_scale,
      include_card_features=not self.package_only,
      require_targets=False,
    )
    self._linear.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(
        len(examples), batch_size
      ):
        batch = _prepared_linear_batch(
          prepared,
          batch_indexes,
          feature_count=self.feature_count,
          device=self.device,
        )
        logits = self._linear(batch.features)
        predictions.extend(_logit_stats(logits, self.labels, k))

    return predictions

  def logits_many(
    self,
    examples: list[DeckModelInput],
    *,
    batch_size: int = 32,
  ) -> torch.Tensor:
    self._ensure_package_cache(examples)
    prepared = _prepare_linear_examples(
      examples,
      label_to_idx=self._label_to_idx,
      zone_count=self.zone_count,
      package_index_by_deck=self._package_index_by_deck,
      package_feature_offset=self.base_feature_count,
      package_scale=self.package_scale,
      include_card_features=not self.package_only,
      require_targets=False,
    )
    rows: list[torch.Tensor] = []
    self._linear.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(
        len(examples), batch_size
      ):
        batch = _prepared_linear_batch(
          prepared,
          batch_indexes,
          feature_count=self.feature_count,
          device=self.device,
        )
        rows.append(self._linear(batch.features).detach().cpu())

    if not rows:
      return torch.empty((0, len(self.labels)), dtype=torch.float32)
    return torch.cat(rows, dim=0)

  def _loss_accuracy(
    self,
    examples: list[DeckModelInput],
  ) -> tuple[float | None, float | None]:
    if not examples:
      return None, None

    self._ensure_package_cache(examples)
    prepared = _prepare_linear_examples(
      examples,
      label_to_idx=self._label_to_idx,
      zone_count=self.zone_count,
      package_index_by_deck=self._package_index_by_deck,
      package_feature_offset=self.base_feature_count,
      package_scale=self.package_scale,
      include_card_features=not self.package_only,
    )
    return self._loss_accuracy_prepared(prepared)

  def _loss_accuracy_prepared(
    self,
    prepared: _PreparedLinearExamples,
  ) -> tuple[float | None, float | None]:
    if not prepared.examples:
      return None, None

    total_loss = 0.0
    correct = 0
    count = 0
    self._linear.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(
        len(prepared.examples), 512
      ):
        batch = _prepared_linear_batch(
          prepared,
          batch_indexes,
          feature_count=self.feature_count,
          device=self.device,
        )
        logits = self._linear(batch.features)
        loss = F.cross_entropy(logits, batch.target_idx)
        batch_count = len(batch_indexes)
        total_loss += float(loss.item()) * batch_count
        correct += int((logits.argmax(dim=1) == batch.target_idx).sum().item())
        count += batch_count

    return total_loss / count, correct / count

  def _loss_accuracy_dense(
    self,
    prepared: _DenseLinearExamples,
  ) -> tuple[float | None, float | None]:
    if prepared.features.shape[0] == 0:
      return None, None

    total_loss = 0.0
    correct = 0
    count = 0
    self._linear.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(
        prepared.features.shape[0], 512
      ):
        batch = _dense_linear_batch(prepared, batch_indexes)
        logits = self._linear(batch.features)
        loss = F.cross_entropy(logits, batch.target_idx)
        batch_count = len(batch_indexes)
        total_loss += float(loss.item()) * batch_count
        correct += int((logits.argmax(dim=1) == batch.target_idx).sum().item())
        count += batch_count

    return total_loss / count, correct / count

  def _state(self) -> dict[str, torch.Tensor]:
    return {
      key: value.detach().cpu().clone()
      for key, value in self._linear.state_dict().items()
    }

  def _restore_state(self, state: dict[str, torch.Tensor]) -> None:
    self._linear.load_state_dict({
      key: value.to(self.device) for key, value in state.items()
    })

  def state_dict_for_saving(self) -> dict[str, torch.Tensor]:
    return self._state()

  def load_saved_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    self._restore_state(state)

  def _ensure_package_cache(self, examples: list[DeckModelInput]) -> None:
    if self.package_features is None:
      return
    for example in examples:
      if example.deck_id not in self._package_index_by_deck:
        self._package_index_by_deck[example.deck_id] = (
          self.package_features.activation_indexes(example)
        )

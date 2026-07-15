from __future__ import annotations

import random
from typing import Any, Protocol

import torch
from torch import nn
from torch.nn import functional as F

from manafold.models.data import ModelExample
from manafold.models.packages import PackageFeatureSet

POOLING_SUM = "sum"
POOLING_MEAN = "mean"
POOLING_QUANTITY_WEIGHTED = "quantity-weighted"
POOLING_MODES = (POOLING_SUM, POOLING_MEAN, POOLING_QUANTITY_WEIGHTED)
SET_TRANSFORMER_POOLING_PMA = "pma"
SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED = "quantity-weighted"
SET_TRANSFORMER_POOLING_MODES = (
  SET_TRANSFORMER_POOLING_PMA,
  SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED,
)
PROTOTYPE_DISTANCE_EUCLIDEAN = "euclidean"


class Classifier(Protocol):
  labels: tuple[str, ...]

  def fit(
    self,
    examples: list[ModelExample],
    *,
    validation_examples: list[ModelExample] | None = None,
    epochs: int = 40,
    batch_size: int = 32,
    shuffle: bool = True,
    max_steps: int | None = None,
  ) -> dict[str, Any]:
    ...

  def predict(self, example: ModelExample) -> tuple[str | None, float]:
    ...

  def predict_top_k(
    self,
    example: ModelExample,
    *,
    k: int,
  ) -> list[tuple[str, float]]:
    ...


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
    self.package_count = 0 if package_features is None else len(package_features)
    self.feature_count = self.base_feature_count + self.package_count
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.package_scale = package_scale
    self.device = _resolve_device(device)
    self._label_to_idx = {
      label: index
      for index, label in enumerate(labels)
    }
    self._rng = random.Random(seed)
    self._package_index_by_deck: dict[str, tuple[int, ...]] = {}
    torch.manual_seed(seed)
    self._linear = nn.Linear(self.feature_count, len(labels)).to(self.device)

  def fit(
    self,
    examples: list[ModelExample],
    *,
    validation_examples: list[ModelExample] | None = None,
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
        total_correct += int((logits.argmax(dim=1) == batch.target_idx).sum().item())
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

      history.append(
        {
          "epoch": epoch,
          "loss": total_loss / total_count if total_count else None,
          "accuracy": total_correct / total_count if total_count else None,
          "train_loss": train_loss,
          "train_accuracy": train_accuracy,
          "validation_loss": validation_loss,
          "validation_accuracy": validation_accuracy,
          "optimizer_steps": optimizer_steps,
        }
      )

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

  def predict(self, example: ModelExample) -> tuple[str | None, float]:
    top = self.predict_top_k(example, k=1)
    if not top:
      return None, 0.0
    return top[0]

  def predict_top_k(
    self,
    example: ModelExample,
    *,
    k: int,
  ) -> list[tuple[str, float]]:
    if k <= 0 or not self.labels:
      return []

    return self.predict_top_k_many([example], k=k)[0]

  def predict_top_k_many(
    self,
    examples: list[ModelExample],
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
    examples: list[ModelExample],
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
      for batch_indexes in _sequential_index_batches(len(examples), batch_size):
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
    examples: list[ModelExample],
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
      for batch_indexes in _sequential_index_batches(len(examples), batch_size):
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
    examples: list[ModelExample],
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
    prepared: "_PreparedLinearExamples",
  ) -> tuple[float | None, float | None]:
    if not prepared.examples:
      return None, None

    total_loss = 0.0
    correct = 0
    count = 0
    self._linear.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(len(prepared.examples), 512):
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
    prepared: "_DenseLinearExamples",
  ) -> tuple[float | None, float | None]:
    if prepared.features.shape[0] == 0:
      return None, None

    total_loss = 0.0
    correct = 0
    count = 0
    self._linear.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(prepared.features.shape[0], 512):
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
      key: value.to(self.device)
      for key, value in state.items()
    })

  def state_dict_for_artifact(self) -> dict[str, torch.Tensor]:
    return self._state()

  def load_state_dict_from_artifact(self, state: dict[str, torch.Tensor]) -> None:
    self._restore_state(state)

  def _ensure_package_cache(self, examples: list[ModelExample]) -> None:
    if self.package_features is None:
      return
    for example in examples:
      if example.deck_id not in self._package_index_by_deck:
        self._package_index_by_deck[example.deck_id] = (
          self.package_features.activation_indexes(example)
        )


class DeepSetsClassifier:
  """Learned card, zone, and count embeddings with Deep Sets pooling."""

  def __init__(
    self,
    *,
    labels: tuple[str, ...],
    card_count: int,
    zone_count: int,
    quantity_count: int,
    embedding_dim: int = 32,
    hidden_dim: int = 64,
    pooling: str = POOLING_SUM,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0,
    seed: int = 13,
    device: str = "auto",
    package_features: PackageFeatureSet | None = None,
    package_projection_dim: int = 64,
    package_projection_bias: bool = True,
    package_scale: float = 1.0,
    rho_hidden_dim: int | None = None,
    extra_feature_dim: int = 0,
    extra_feature_value: float = 0.0,
    preserve_base_rho_init: bool = False,
  ) -> None:
    if not labels:
      raise ValueError("DeepSetsClassifier requires at least one label.")
    if card_count <= 0:
      raise ValueError("card_count must be positive.")
    if zone_count <= 0:
      raise ValueError("zone_count must be positive.")
    if quantity_count <= 0:
      raise ValueError("quantity_count must be positive.")
    if pooling not in POOLING_MODES:
      raise ValueError(f"Unsupported pooling mode: {pooling}")
    if package_scale < 0:
      raise ValueError("package_scale must be non-negative.")
    if package_projection_dim < 0:
      raise ValueError("package_projection_dim must be non-negative.")
    if extra_feature_dim < 0:
      raise ValueError("extra_feature_dim must be non-negative.")
    if rho_hidden_dim is not None and rho_hidden_dim <= 0:
      raise ValueError("rho_hidden_dim must be positive.")
    if weight_decay < 0:
      raise ValueError("weight_decay must be non-negative.")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)

    self.labels = labels
    self.card_count = card_count
    self.zone_count = zone_count
    self.quantity_count = quantity_count
    self.pooling = pooling
    self.package_features = package_features
    self.package_count = 0 if package_features is None else len(package_features)
    self.package_projection_dim = package_projection_dim
    self.package_projection_bias = package_projection_bias
    self.rho_hidden_dim = rho_hidden_dim or hidden_dim
    self.extra_feature_dim = extra_feature_dim
    self.extra_feature_value = extra_feature_value
    self.preserve_base_rho_init = preserve_base_rho_init
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.package_scale = package_scale
    self.device = _resolve_device(device)
    self._label_to_idx = {
      label: index
      for index, label in enumerate(labels)
    }
    self._rng = random.Random(seed)
    self._network = _DeepSetsNetwork(
      card_count=card_count,
      zone_count=zone_count,
      quantity_count=quantity_count,
      label_count=len(labels),
      embedding_dim=embedding_dim,
      hidden_dim=hidden_dim,
      pooling=pooling,
      package_count=self.package_count,
      package_projection_dim=package_projection_dim,
      package_projection_bias=package_projection_bias,
      rho_hidden_dim=self.rho_hidden_dim,
      extra_feature_dim=extra_feature_dim,
      extra_feature_value=extra_feature_value,
      preserve_base_rho_init=preserve_base_rho_init,
    ).to(self.device)

  def fit(
    self,
    examples: list[ModelExample],
    *,
    validation_examples: list[ModelExample] | None = None,
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

    trainable_prepared = _prepare_set_examples(
      trainable,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
      package_features=self.package_features,
    )
    validation_prepared = _prepare_set_examples(
      validation,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
      package_features=self.package_features,
    )
    optimizer = torch.optim.Adam(
      self._network.parameters(),
      lr=self.learning_rate,
      weight_decay=self.weight_decay,
    )
    history: list[dict[str, float | int | None]] = []
    best_state = self._state()
    best_epoch: int | None = None
    best_metric: float | None = None
    best_loss: float | None = None
    optimizer_steps = 0

    epoch = 0
    while _should_continue_training(epoch, epochs, optimizer_steps, max_steps):
      epoch += 1
      epoch_indexes = list(range(len(trainable)))
      if shuffle:
        self._rng.shuffle(epoch_indexes)

      self._network.train()
      total_loss = 0.0
      total_correct = 0
      total_count = 0
      for batch_indexes in _index_batches(epoch_indexes, batch_size):
        if max_steps is not None and optimizer_steps >= max_steps:
          break
        batch = _prepared_set_batch(
          trainable_prepared,
          batch_indexes,
          device=self.device,
          package_scale=self.package_scale,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = self._network(batch)
        loss = F.cross_entropy(logits, batch.target_idx)
        loss.backward()
        optimizer.step()
        optimizer_steps += 1

        batch_count = len(batch_indexes)
        total_loss += float(loss.item()) * batch_count
        total_correct += int((logits.argmax(dim=1) == batch.target_idx).sum().item())
        total_count += batch_count

      train_loss = total_loss / total_count if total_count else None
      train_accuracy = total_correct / total_count if total_count else None
      validation_loss, validation_accuracy = self._loss_accuracy_prepared(
        validation_prepared,
        batch_size,
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

      history.append(
        {
          "epoch": epoch,
          "train_loss": train_loss,
          "train_accuracy": train_accuracy,
          "validation_loss": validation_loss,
          "validation_accuracy": validation_accuracy,
          "optimizer_steps": optimizer_steps,
        }
      )

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

  def predict(self, example: ModelExample) -> tuple[str | None, float]:
    top = self.predict_top_k(example, k=1)
    if not top:
      return None, 0.0
    return top[0]

  def predict_top_k(
    self,
    example: ModelExample,
    *,
    k: int,
  ) -> list[tuple[str, float]]:
    if k <= 0 or not self.labels or not example.tokens:
      return []
    return self.predict_top_k_many([example], k=k)[0]

  def predict_top_k_many(
    self,
    examples: list[ModelExample],
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
    examples: list[ModelExample],
    *,
    k: int,
    batch_size: int = 32,
  ) -> list[dict[str, Any]]:
    if k <= 0:
      return [_empty_prediction_stats() for _ in examples]

    predictions: list[dict[str, Any]] = []
    prepared = _prepare_set_examples(
      examples,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
      require_targets=False,
      package_features=self.package_features,
    )
    self._network.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(len(examples), batch_size):
        batch = _prepared_set_batch(
          prepared,
          batch_indexes,
          device=self.device,
          package_scale=self.package_scale,
        )
        logits = self._network(batch)
        predictions.extend(_logit_stats(logits, self.labels, k))

    return predictions

  def logits_many(
    self,
    examples: list[ModelExample],
    *,
    batch_size: int = 32,
  ) -> torch.Tensor:
    prepared = _prepare_set_examples(
      examples,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
      require_targets=False,
      package_features=self.package_features,
    )
    rows: list[torch.Tensor] = []
    self._network.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(len(examples), batch_size):
        batch = _prepared_set_batch(
          prepared,
          batch_indexes,
          device=self.device,
          package_scale=self.package_scale,
        )
        rows.append(self._network(batch).detach().cpu())

    if not rows:
      return torch.empty((0, len(self.labels)), dtype=torch.float32)
    return torch.cat(rows, dim=0)

  def parameter_count(self) -> int:
    return sum(parameter.numel() for parameter in self._network.parameters())

  def package_branch_usage_summary(
    self,
    examples: list[ModelExample],
    *,
    batch_size: int = 32,
    gradient_example_limit: int = 4096,
  ) -> dict[str, Any]:
    if self.package_count <= 0 or self._network.package_projection is None:
      return {
        "status": "skipped",
        "reason": "Model has no package projection branch.",
      }

    trainable = [
      example
      for example in examples
      if example.target_label_id in self._label_to_idx
    ][:gradient_example_limit]
    if not trainable:
      return {
        "status": "skipped",
        "reason": "No labeled examples are available for package branch diagnostics.",
      }

    prepared = _prepare_set_examples(
      trainable,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
      package_features=self.package_features,
    )
    batch = _prepared_set_batch(
      prepared,
      list(range(len(trainable))),
      device=self.device,
      package_scale=self.package_scale,
    )
    package_projection = self._network.package_projection
    if package_projection is None:
      return {
        "status": "skipped",
        "reason": "Model has no package projection branch.",
      }

    self._network.zero_grad(set_to_none=True)
    self._network.train()
    projected = package_projection(batch.package_features)
    logits = self._network(batch)
    loss = F.cross_entropy(logits, batch.target_idx)
    loss.backward()

    rho_first = self._network.rho[0]
    package_start = self._network.hidden_dim
    package_end = package_start + self.package_projection_dim
    rho_package_weights = rho_first.weight[:, package_start:package_end]
    rho_package_grad = (
      None
      if rho_first.weight.grad is None
      else rho_first.weight.grad[:, package_start:package_end]
    )
    activation_count = int(torch.count_nonzero(batch.package_features).item())
    activation_total = int(batch.package_features.numel())

    summary = {
      "status": "completed",
      "example_count": len(trainable),
      "gradient_example_limit": gradient_example_limit,
      "package_activation_count": activation_count,
      "package_activation_density": (
        activation_count / activation_total
        if activation_total
        else None
      ),
      "package_projection_output_l2_mean": float(
        projected.norm(dim=1).mean().item()
      ),
      "package_projection_output_l2_max": float(
        projected.norm(dim=1).max().item()
      ),
      "package_projection_parameter_l2_norm": _module_parameter_norm(
        package_projection
      ),
      "package_projection_gradient_l2_norm": _module_gradient_norm(
        package_projection
      ),
      "rho_package_column_weight_l2_norm": float(
        rho_package_weights.norm().item()
      ),
      "rho_package_column_gradient_l2_norm": (
        float(rho_package_grad.norm().item())
        if rho_package_grad is not None
        else None
      ),
      "diagnostic_loss": float(loss.item()),
    }
    self._network.zero_grad(set_to_none=True)
    self._network.eval()
    return summary

  def card_embedding_rows(
    self,
    card_vocab: tuple[dict[str, Any], ...],
  ) -> list[dict[str, Any]]:
    weights = self._network.card_embedding.weight.detach().cpu().tolist()
    rows: list[dict[str, Any]] = []
    for card in sorted(card_vocab, key=lambda row: int(row["card_idx"])):
      card_idx = int(card["card_idx"])
      rows.append(
        {
          "card_idx": card_idx,
          "oracle_id": card.get("oracle_id"),
          "primary_name": card.get("primary_name"),
          "embedding": weights[card_idx],
        }
      )
    return rows

  def deck_embedding_rows(
    self,
    examples: list[ModelExample],
    *,
    batch_size: int = 32,
  ) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prepared = _prepare_set_examples(
      examples,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
      require_targets=False,
      package_features=self.package_features,
    )
    self._network.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(len(examples), batch_size):
        batch = _prepared_set_batch(
          prepared,
          batch_indexes,
          device=self.device,
          package_scale=self.package_scale,
        )
        embeddings = self._network.encode(batch).detach().cpu().tolist()
        batch_examples = [
          examples[index]
          for index in batch_indexes
        ]
        for example, embedding in zip(batch_examples, embeddings, strict=True):
          rows.append(
            {
              "deck_id": example.deck_id,
              "split_name": example.split_name,
              "target_label_id": example.target_label_id,
              "embedding": embedding,
            }
          )
    return rows

  def deck_embedding_tensor(
    self,
    examples: list[ModelExample],
    *,
    batch_size: int = 32,
  ) -> torch.Tensor:
    embeddings: list[torch.Tensor] = []
    prepared = _prepare_set_examples(
      examples,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
      require_targets=False,
      package_features=self.package_features,
    )
    self._network.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(len(examples), batch_size):
        batch = _prepared_set_batch(
          prepared,
          batch_indexes,
          device=self.device,
          package_scale=self.package_scale,
        )
        embeddings.append(self._network.encode(batch).detach())

    if not embeddings:
      return torch.empty((0, 0), dtype=torch.float32, device=self.device)
    return torch.cat(embeddings, dim=0)

  def card_nearest_neighbors(
    self,
    card_vocab: tuple[dict[str, Any], ...],
    *,
    anchor_card_idxs: list[int] | None = None,
    card_support: dict[int, int] | None = None,
    anchor_limit: int = 20,
    neighbor_count: int = 5,
  ) -> list[dict[str, Any]]:
    if anchor_limit <= 0 or neighbor_count <= 0:
      return []

    vocab = sorted(card_vocab, key=lambda row: int(row["card_idx"]))
    vocab_by_idx = {
      int(card["card_idx"]): card
      for card in vocab
    }
    embeddings = self._network.card_embedding.weight.detach().cpu()
    normalized = F.normalize(embeddings, p=2, dim=1)
    if anchor_card_idxs is None:
      anchor_idxs = [int(card["card_idx"]) for card in vocab[:anchor_limit]]
    else:
      anchor_idxs = [
        card_idx
        for card_idx in anchor_card_idxs[:anchor_limit]
        if card_idx in vocab_by_idx
      ]
    rows: list[dict[str, Any]] = []
    for card_idx in anchor_idxs:
      card = vocab_by_idx[card_idx]
      similarities = torch.mv(normalized, normalized[card_idx])
      similarities[card_idx] = -2.0
      scores, indexes = torch.topk(
        similarities,
        k=min(neighbor_count, len(vocab) - 1),
      )
      rows.append(
        {
          "card_idx": card_idx,
          "primary_name": card.get("primary_name"),
          "train_support": (
            None
            if card_support is None
            else card_support.get(card_idx, 0)
          ),
          "neighbors": [
            {
              "card_idx": int(neighbor_idx),
              "primary_name": vocab_by_idx[int(neighbor_idx)].get("primary_name"),
              "train_support": (
                None
                if card_support is None
                else card_support.get(int(neighbor_idx), 0)
              ),
              "score": float(score),
            }
            for score, neighbor_idx in zip(
              scores.tolist(),
              indexes.tolist(),
              strict=True,
            )
          ],
        }
      )
    return rows

  def _loss_accuracy(
    self,
    examples: list[ModelExample],
    batch_size: int,
  ) -> tuple[float | None, float | None]:
    prepared = _prepare_set_examples(
      examples,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
      package_features=self.package_features,
    )
    return self._loss_accuracy_prepared(prepared, batch_size)

  def _loss_accuracy_prepared(
    self,
    prepared: "_PreparedSetExamples",
    batch_size: int,
  ) -> tuple[float | None, float | None]:
    if not prepared.examples:
      return None, None

    self._network.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(
        len(prepared.examples),
        batch_size,
      ):
        batch = _prepared_set_batch(
          prepared,
          batch_indexes,
          device=self.device,
          package_scale=self.package_scale,
        )
        logits = self._network(batch)
        loss = F.cross_entropy(logits, batch.target_idx)
        batch_count = len(batch_indexes)
        total_loss += float(loss.item()) * batch_count
        total_correct += int((logits.argmax(dim=1) == batch.target_idx).sum().item())
        total_count += batch_count

    return total_loss / total_count, total_correct / total_count

  def _state(self) -> dict[str, torch.Tensor]:
    return {
      key: value.detach().cpu().clone()
      for key, value in self._network.state_dict().items()
    }

  def _restore_state(self, state: dict[str, torch.Tensor]) -> None:
    self._network.load_state_dict({
      key: value.to(self.device)
      for key, value in state.items()
    })

  def state_dict_for_artifact(self) -> dict[str, torch.Tensor]:
    return self._state()

  def load_state_dict_from_artifact(self, state: dict[str, torch.Tensor]) -> None:
    self._restore_state(state)


class PrototypeClassifier:
  """Nearest-prototype classifier over a fitted Deep Sets deck encoder."""

  def __init__(
    self,
    *,
    encoder: DeepSetsClassifier,
    prototypes: torch.Tensor,
    prototype_counts: dict[str, int],
    distance: str = PROTOTYPE_DISTANCE_EUCLIDEAN,
  ) -> None:
    if distance != PROTOTYPE_DISTANCE_EUCLIDEAN:
      raise ValueError(f"Unsupported prototype distance: {distance}")
    if prototypes.shape[0] != len(encoder.labels):
      raise ValueError("Prototype count must match label count.")

    self.labels = encoder.labels
    self.encoder = encoder
    self.distance = distance
    self.prototype_counts = prototype_counts
    self.device = encoder.device
    self._label_to_idx = {
      label: index
      for index, label in enumerate(self.labels)
    }
    self._prototypes = prototypes.detach().to(self.device)

  @classmethod
  def from_deepsets(
    cls,
    encoder: DeepSetsClassifier,
    train_examples: list[ModelExample],
    *,
    batch_size: int = 32,
  ) -> "PrototypeClassifier":
    trainable = [
      example
      for example in train_examples
      if example.target_label_id in encoder._label_to_idx
    ]
    if not trainable:
      raise ValueError("PrototypeClassifier requires at least one train example.")

    embeddings = encoder.deck_embedding_tensor(trainable, batch_size=batch_size)
    if embeddings.numel() == 0:
      raise ValueError("PrototypeClassifier requires non-empty deck embeddings.")

    label_indexes = torch.tensor(
      [
        encoder._label_to_idx[example.target_label_id]
        for example in trainable
      ],
      dtype=torch.long,
      device=encoder.device,
    )
    sums = embeddings.new_zeros((len(encoder.labels), embeddings.shape[1]))
    counts = embeddings.new_zeros((len(encoder.labels), 1))
    sums.index_add_(0, label_indexes, embeddings)
    counts.index_add_(
      0,
      label_indexes,
      torch.ones((len(trainable), 1), dtype=embeddings.dtype, device=encoder.device),
    )
    prototypes = sums / counts.clamp_min(1.0)
    prototype_counts = {
      label: int(counts[index].item())
      for index, label in enumerate(encoder.labels)
    }
    return cls(
      encoder=encoder,
      prototypes=prototypes,
      prototype_counts=prototype_counts,
    )

  def fit(
    self,
    examples: list[ModelExample],
    *,
    validation_examples: list[ModelExample] | None = None,
    epochs: int = 40,
    batch_size: int = 32,
    shuffle: bool = True,
    max_steps: int | None = None,
  ) -> dict[str, Any]:
    del examples, validation_examples, epochs, batch_size, shuffle, max_steps
    return _empty_training_summary()

  def predict(self, example: ModelExample) -> tuple[str | None, float]:
    top = self.predict_top_k(example, k=1)
    if not top:
      return None, 0.0
    return top[0]

  def predict_top_k(
    self,
    example: ModelExample,
    *,
    k: int,
  ) -> list[tuple[str, float]]:
    return self.predict_top_k_many([example], k=k)[0]

  def predict_top_k_many(
    self,
    examples: list[ModelExample],
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
    examples: list[ModelExample],
    *,
    k: int,
    batch_size: int = 32,
  ) -> list[dict[str, Any]]:
    if k <= 0:
      return [_empty_prediction_stats() for _ in examples]

    rows: list[dict[str, Any]] = []
    prepared = _prepare_set_examples(
      examples,
      label_to_idx=self._label_to_idx,
      quantity_count=self.encoder.quantity_count,
      require_targets=False,
    )
    self.encoder._network.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(len(examples), batch_size):
        batch = _prepared_set_batch(
          prepared,
          batch_indexes,
          device=self.device,
        )
        embeddings = self.encoder._network.encode(batch)
        distances = torch.cdist(embeddings, self._prototypes, p=2)
        prototype_logits = -distances
        probabilities = torch.softmax(prototype_logits, dim=1)
        top_distances, indexes = torch.topk(
          distances,
          k=min(k, len(self.labels)),
          largest=False,
          dim=1,
        )
        top_probabilities = probabilities.gather(1, indexes)
        entropy = _probability_entropy(probabilities)
        energy = -torch.logsumexp(prototype_logits, dim=1)
        if len(self.labels) > 1:
          sorted_distances, _ = torch.sort(distances, dim=1)
          margins = sorted_distances[:, 1] - sorted_distances[:, 0]
        else:
          margins = distances.new_zeros((distances.shape[0],))
        rows.extend(
          _prototype_stats(
            top_probabilities=top_probabilities,
            top_indexes=indexes,
            top_distances=top_distances,
            entropy=entropy,
            energy=energy,
            nearest_distances=top_distances[:, 0],
            margins=margins,
            labels=self.labels,
          )
        )

    return rows

  def prototype_rows(self) -> list[dict[str, Any]]:
    prototypes = self._prototypes.detach().cpu().tolist()
    return [
      {
        "label_id": label,
        "train_support": self.prototype_counts.get(label, 0),
        "prototype": prototypes[index],
      }
      for index, label in enumerate(self.labels)
    ]


class SetTransformerClassifier:
  """Self-attention set encoder with pooling by multihead attention."""

  def __init__(
    self,
    *,
    labels: tuple[str, ...],
    card_count: int,
    zone_count: int,
    quantity_count: int,
    embedding_dim: int = 32,
    hidden_dim: int = 64,
    attention_heads: int = 4,
    attention_layers: int = 2,
    pooling: str = SET_TRANSFORMER_POOLING_PMA,
    learning_rate: float = 0.01,
    seed: int = 13,
    device: str = "auto",
  ) -> None:
    if not labels:
      raise ValueError("SetTransformerClassifier requires at least one label.")
    if card_count <= 0:
      raise ValueError("card_count must be positive.")
    if zone_count <= 0:
      raise ValueError("zone_count must be positive.")
    if quantity_count <= 0:
      raise ValueError("quantity_count must be positive.")
    if attention_heads <= 0:
      raise ValueError("attention_heads must be positive.")
    if embedding_dim % attention_heads:
      raise ValueError("embedding_dim must be divisible by attention_heads.")
    if attention_layers <= 0:
      raise ValueError("attention_layers must be positive.")
    if pooling not in SET_TRANSFORMER_POOLING_MODES:
      raise ValueError(f"Unsupported Set Transformer pooling mode: {pooling}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)

    self.labels = labels
    self.card_count = card_count
    self.zone_count = zone_count
    self.quantity_count = quantity_count
    self.pooling = pooling
    self.learning_rate = learning_rate
    self.device = _resolve_device(device)
    self._label_to_idx = {
      label: index
      for index, label in enumerate(labels)
    }
    self._rng = random.Random(seed)
    self._network = _SetTransformerNetwork(
      card_count=card_count,
      zone_count=zone_count,
      quantity_count=quantity_count,
      label_count=len(labels),
      embedding_dim=embedding_dim,
      hidden_dim=hidden_dim,
      attention_heads=attention_heads,
      attention_layers=attention_layers,
      pooling=pooling,
    ).to(self.device)

  def fit(
    self,
    examples: list[ModelExample],
    *,
    validation_examples: list[ModelExample] | None = None,
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

    trainable_prepared = _prepare_set_examples(
      trainable,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
    )
    validation_prepared = _prepare_set_examples(
      validation,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
    )
    optimizer = torch.optim.Adam(self._network.parameters(), lr=self.learning_rate)
    history: list[dict[str, float | int | None]] = []
    best_state = self._state()
    best_epoch: int | None = None
    best_metric: float | None = None
    best_loss: float | None = None
    optimizer_steps = 0

    epoch = 0
    while _should_continue_training(epoch, epochs, optimizer_steps, max_steps):
      epoch += 1
      epoch_indexes = list(range(len(trainable)))
      if shuffle:
        self._rng.shuffle(epoch_indexes)

      self._network.train()
      total_loss = 0.0
      total_correct = 0
      total_count = 0
      for batch_indexes in _index_batches(epoch_indexes, batch_size):
        if max_steps is not None and optimizer_steps >= max_steps:
          break
        batch = _prepared_set_batch(
          trainable_prepared,
          batch_indexes,
          device=self.device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = self._network(batch)
        loss = F.cross_entropy(logits, batch.target_idx)
        loss.backward()
        optimizer.step()
        optimizer_steps += 1

        batch_count = len(batch_indexes)
        total_loss += float(loss.item()) * batch_count
        total_correct += int((logits.argmax(dim=1) == batch.target_idx).sum().item())
        total_count += batch_count

      train_loss = total_loss / total_count if total_count else None
      train_accuracy = total_correct / total_count if total_count else None
      validation_loss, validation_accuracy = self._loss_accuracy_prepared(
        validation_prepared,
        batch_size,
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

      history.append(
        {
          "epoch": epoch,
          "train_loss": train_loss,
          "train_accuracy": train_accuracy,
          "validation_loss": validation_loss,
          "validation_accuracy": validation_accuracy,
          "optimizer_steps": optimizer_steps,
        }
      )

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

  def predict(self, example: ModelExample) -> tuple[str | None, float]:
    top = self.predict_top_k(example, k=1)
    if not top:
      return None, 0.0
    return top[0]

  def predict_top_k(
    self,
    example: ModelExample,
    *,
    k: int,
  ) -> list[tuple[str, float]]:
    return self.predict_top_k_many([example], k=k)[0]

  def predict_top_k_many(
    self,
    examples: list[ModelExample],
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
    examples: list[ModelExample],
    *,
    k: int,
    batch_size: int = 32,
  ) -> list[dict[str, Any]]:
    if k <= 0:
      return [_empty_prediction_stats() for _ in examples]

    predictions: list[dict[str, Any]] = []
    prepared = _prepare_set_examples(
      examples,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
      require_targets=False,
    )
    self._network.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(len(examples), batch_size):
        batch = _prepared_set_batch(
          prepared,
          batch_indexes,
          device=self.device,
        )
        logits = self._network(batch)
        predictions.extend(_logit_stats(logits, self.labels, k))

    return predictions

  def card_embedding_rows(
    self,
    card_vocab: tuple[dict[str, Any], ...],
  ) -> list[dict[str, Any]]:
    weights = self._network.card_embedding.weight.detach().cpu().tolist()
    rows: list[dict[str, Any]] = []
    for card in sorted(card_vocab, key=lambda row: int(row["card_idx"])):
      card_idx = int(card["card_idx"])
      rows.append(
        {
          "card_idx": card_idx,
          "oracle_id": card.get("oracle_id"),
          "primary_name": card.get("primary_name"),
          "embedding": weights[card_idx],
        }
      )
    return rows

  def deck_embedding_rows(
    self,
    examples: list[ModelExample],
    *,
    batch_size: int = 32,
  ) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prepared = _prepare_set_examples(
      examples,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
      require_targets=False,
    )
    self._network.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(len(examples), batch_size):
        batch = _prepared_set_batch(
          prepared,
          batch_indexes,
          device=self.device,
        )
        embeddings = self._network.encode(batch).detach().cpu().tolist()
        batch_examples = [
          examples[index]
          for index in batch_indexes
        ]
        for example, embedding in zip(batch_examples, embeddings, strict=True):
          rows.append(
            {
              "deck_id": example.deck_id,
              "split_name": example.split_name,
              "target_label_id": example.target_label_id,
              "embedding": embedding,
            }
          )
    return rows

  def card_nearest_neighbors(
    self,
    card_vocab: tuple[dict[str, Any], ...],
    *,
    anchor_card_idxs: list[int] | None = None,
    card_support: dict[int, int] | None = None,
    anchor_limit: int = 20,
    neighbor_count: int = 5,
  ) -> list[dict[str, Any]]:
    return _card_nearest_neighbors(
      self._network.card_embedding.weight.detach().cpu(),
      card_vocab,
      anchor_card_idxs=anchor_card_idxs,
      card_support=card_support,
      anchor_limit=anchor_limit,
      neighbor_count=neighbor_count,
    )

  def _loss_accuracy(
    self,
    examples: list[ModelExample],
    batch_size: int,
  ) -> tuple[float | None, float | None]:
    prepared = _prepare_set_examples(
      examples,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
    )
    return self._loss_accuracy_prepared(prepared, batch_size)

  def _loss_accuracy_prepared(
    self,
    prepared: "_PreparedSetExamples",
    batch_size: int,
  ) -> tuple[float | None, float | None]:
    if not prepared.examples:
      return None, None

    self._network.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(
        len(prepared.examples),
        batch_size,
      ):
        batch = _prepared_set_batch(
          prepared,
          batch_indexes,
          device=self.device,
        )
        logits = self._network(batch)
        loss = F.cross_entropy(logits, batch.target_idx)
        batch_count = len(batch_indexes)
        total_loss += float(loss.item()) * batch_count
        total_correct += int((logits.argmax(dim=1) == batch.target_idx).sum().item())
        total_count += batch_count

    return total_loss / total_count, total_correct / total_count

  def _state(self) -> dict[str, torch.Tensor]:
    return {
      key: value.detach().cpu().clone()
      for key, value in self._network.state_dict().items()
    }

  def _restore_state(self, state: dict[str, torch.Tensor]) -> None:
    self._network.load_state_dict({
      key: value.to(self.device)
      for key, value in state.items()
    })


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

  def forward(self, batch: "_Batch") -> torch.Tensor:
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
    return self.rho(torch.cat(parts, dim=1) if len(parts) > 1 else deck_embedding)

  def encode(self, batch: "_Batch") -> torch.Tensor:
    token_embeddings = (
      self.card_embedding(batch.card_idx)
      + self.zone_embedding(batch.zone_idx)
      + self.quantity_embedding(batch.quantity_idx)
    )
    token_latents = self.phi(token_embeddings)
    if self.pooling == POOLING_QUANTITY_WEIGHTED:
      token_latents = token_latents * batch.quantity_weight.unsqueeze(1)

    pooled = token_latents.new_zeros((batch.deck_count, token_latents.shape[-1]))
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
    parameter.detach().flatten()
    for parameter in module.parameters()
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
  ) -> None:
    super().__init__()
    self.pooling = pooling
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
    self.rho = nn.Sequential(
      nn.LayerNorm(embedding_dim),
      nn.Linear(embedding_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, label_count),
    )

  def forward(self, batch: "_Batch") -> torch.Tensor:
    return self.rho(self.encode(batch))

  def encode(self, batch: "_Batch") -> torch.Tensor:
    token_embeddings = (
      self.card_embedding(batch.card_idx)
      + self.zone_embedding(batch.zone_idx)
      + self.quantity_embedding(batch.quantity_idx)
    )
    padded, padding_mask = _padded_tokens(
      token_embeddings,
      batch.deck_idx,
      batch.deck_count,
    )
    encoded = self.encoder(padded, src_key_padding_mask=padding_mask)
    if self.pooling == SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED:
      quantity_weights, _ = _padded_tokens(
        batch.quantity_weight.unsqueeze(1),
        batch.deck_idx,
        batch.deck_count,
      )
      return (encoded * quantity_weights).sum(dim=1)

    if self.seed_vector is None or self.pooling_attention is None:
      raise AssertionError(f"Unhandled Set Transformer pooling mode: {self.pooling}")
    seed = self.seed_vector.expand(batch.deck_count, -1, -1)
    pooled, _ = self.pooling_attention(
      seed,
      encoded,
      encoded,
      key_padding_mask=padding_mask,
      need_weights=False,
    )
    return pooled.squeeze(1)


class _Batch:
  def __init__(
    self,
    *,
    card_idx: torch.Tensor,
    zone_idx: torch.Tensor,
    quantity_idx: torch.Tensor,
    quantity_weight: torch.Tensor,
    deck_idx: torch.Tensor,
    deck_count: int,
    target_idx: torch.Tensor,
    package_features: torch.Tensor,
  ) -> None:
    self.card_idx = card_idx
    self.zone_idx = zone_idx
    self.quantity_idx = quantity_idx
    self.quantity_weight = quantity_weight
    self.deck_idx = deck_idx
    self.deck_count = deck_count
    self.target_idx = target_idx
    self.package_features = package_features


class _LinearBatch:
  def __init__(
    self,
    *,
    features: torch.Tensor,
    target_idx: torch.Tensor,
  ) -> None:
    self.features = features
    self.target_idx = target_idx


class _PreparedLinearExamples:
  def __init__(
    self,
    *,
    examples: list[ModelExample],
    target_idx: torch.Tensor,
    feature_idx_by_deck: tuple[tuple[int, ...], ...],
    feature_value_by_deck: tuple[tuple[float, ...], ...],
  ) -> None:
    self.examples = examples
    self.target_idx = target_idx
    self.feature_idx_by_deck = feature_idx_by_deck
    self.feature_value_by_deck = feature_value_by_deck


class _DenseLinearExamples:
  def __init__(
    self,
    *,
    features: torch.Tensor,
    target_idx: torch.Tensor,
  ) -> None:
    self.features = features
    self.target_idx = target_idx


class _PreparedSetExamples:
  def __init__(
    self,
    *,
    examples: list[ModelExample],
    target_idx: torch.Tensor,
    card_idx: torch.Tensor,
    zone_idx: torch.Tensor,
    quantity_idx: torch.Tensor,
    quantity_weight: torch.Tensor,
    token_start: torch.Tensor,
    token_end: torch.Tensor,
    package_idx_by_deck: tuple[tuple[int, ...], ...],
    package_count: int,
  ) -> None:
    self.examples = examples
    self.target_idx = target_idx
    self.card_idx = card_idx
    self.zone_idx = zone_idx
    self.quantity_idx = quantity_idx
    self.quantity_weight = quantity_weight
    self.token_start = token_start
    self.token_end = token_end
    self.package_idx_by_deck = package_idx_by_deck
    self.package_count = package_count


def max_quantity(examples: list[ModelExample]) -> int:
  values = [
    token.quantity
    for example in examples
    for token in example.tokens
  ]
  return max(values, default=0)


def _prepare_set_examples(
  examples: list[ModelExample],
  *,
  label_to_idx: dict[str, int],
  quantity_count: int,
  require_targets: bool = True,
  package_features: PackageFeatureSet | None = None,
) -> _PreparedSetExamples:
  card_idx: list[int] = []
  zone_idx: list[int] = []
  quantity_idx: list[int] = []
  quantity_weight: list[float] = []
  token_start: list[int] = []
  token_end: list[int] = []
  target_idx: list[int] = []
  package_idx_by_deck: list[tuple[int, ...]] = []

  for example in examples:
    if require_targets:
      target_idx.append(label_to_idx[example.target_label_id])
    else:
      target_idx.append(0)
    token_start.append(len(card_idx))
    for token in example.tokens:
      card_idx.append(token.card_idx)
      zone_idx.append(token.zone_idx)
      quantity_idx.append(max(0, min(token.quantity, quantity_count - 1)))
      quantity_weight.append(float(max(token.quantity, 0)))
    token_end.append(len(card_idx))
    if package_features is None:
      package_idx_by_deck.append(())
    else:
      package_idx_by_deck.append(package_features.activation_indexes(example))

  return _PreparedSetExamples(
    examples=examples,
    target_idx=torch.tensor(target_idx, dtype=torch.long),
    card_idx=torch.tensor(card_idx, dtype=torch.long),
    zone_idx=torch.tensor(zone_idx, dtype=torch.long),
    quantity_idx=torch.tensor(quantity_idx, dtype=torch.long),
    quantity_weight=torch.tensor(quantity_weight, dtype=torch.float32),
    token_start=torch.tensor(token_start, dtype=torch.long),
    token_end=torch.tensor(token_end, dtype=torch.long),
    package_idx_by_deck=tuple(package_idx_by_deck),
    package_count=0 if package_features is None else len(package_features),
  )


def _prepared_set_batch(
  prepared: _PreparedSetExamples,
  deck_numbers: list[int],
  *,
  device: torch.device,
  package_scale: float = 1.0,
) -> _Batch:
  deck_number_tensor = torch.tensor(deck_numbers, dtype=torch.long)
  starts = prepared.token_start.index_select(0, deck_number_tensor)
  ends = prepared.token_end.index_select(0, deck_number_tensor)
  lengths = ends - starts
  token_parts = [
    torch.arange(int(start), int(end), dtype=torch.long)
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True)
    if end > start
  ]
  if token_parts:
    token_indexes = torch.cat(token_parts)
    deck_idx = torch.repeat_interleave(
      torch.arange(len(deck_numbers), dtype=torch.long),
      lengths,
    )
  else:
    token_indexes = torch.empty(0, dtype=torch.long)
    deck_idx = torch.empty(0, dtype=torch.long)

  package_features = torch.zeros(
    (len(deck_numbers), prepared.package_count),
    dtype=torch.float32,
    device=device,
  )
  package_row_idx: list[int] = []
  package_col_idx: list[int] = []
  for row_number, deck_number in enumerate(deck_numbers):
    for package_idx in prepared.package_idx_by_deck[deck_number]:
      package_row_idx.append(row_number)
      package_col_idx.append(package_idx)
  if package_row_idx:
    package_features.index_put_(
      (
        torch.tensor(package_row_idx, dtype=torch.long, device=device),
        torch.tensor(package_col_idx, dtype=torch.long, device=device),
      ),
      torch.full(
        (len(package_row_idx),),
        float(package_scale),
        dtype=torch.float32,
        device=device,
      ),
    )

  return _Batch(
    card_idx=prepared.card_idx.index_select(0, token_indexes).to(device),
    zone_idx=prepared.zone_idx.index_select(0, token_indexes).to(device),
    quantity_idx=prepared.quantity_idx.index_select(0, token_indexes).to(device),
    quantity_weight=prepared.quantity_weight.index_select(0, token_indexes).to(device),
    deck_idx=deck_idx.to(device),
    deck_count=len(deck_numbers),
    target_idx=prepared.target_idx.index_select(0, deck_number_tensor).to(device),
    package_features=package_features,
  )


def _tensor_batch(
  examples: list[ModelExample],
  *,
  label_to_idx: dict[str, int],
  quantity_count: int,
  device: torch.device,
  require_targets: bool = True,
) -> _Batch:
  card_idx: list[int] = []
  zone_idx: list[int] = []
  quantity_idx: list[int] = []
  quantity_weight: list[float] = []
  deck_idx: list[int] = []
  target_idx: list[int] = []

  for deck_number, example in enumerate(examples):
    if require_targets:
      target_idx.append(label_to_idx[example.target_label_id])
    else:
      target_idx.append(0)

    for token in example.tokens:
      card_idx.append(token.card_idx)
      zone_idx.append(token.zone_idx)
      quantity_idx.append(max(0, min(token.quantity, quantity_count - 1)))
      quantity_weight.append(float(max(token.quantity, 0)))
      deck_idx.append(deck_number)

  return _Batch(
    card_idx=torch.tensor(card_idx, dtype=torch.long, device=device),
    zone_idx=torch.tensor(zone_idx, dtype=torch.long, device=device),
    quantity_idx=torch.tensor(quantity_idx, dtype=torch.long, device=device),
    quantity_weight=torch.tensor(quantity_weight, dtype=torch.float32, device=device),
    deck_idx=torch.tensor(deck_idx, dtype=torch.long, device=device),
    deck_count=len(examples),
    target_idx=torch.tensor(target_idx, dtype=torch.long, device=device),
    package_features=torch.empty((len(examples), 0), dtype=torch.float32, device=device),
  )


def _prepare_linear_examples(
  examples: list[ModelExample],
  *,
  label_to_idx: dict[str, int],
  zone_count: int,
  package_index_by_deck: dict[str, tuple[int, ...]],
  package_feature_offset: int,
  package_scale: float,
  include_card_features: bool,
  require_targets: bool = True,
) -> _PreparedLinearExamples:
  target_idx: list[int] = []
  feature_idx_by_deck: list[tuple[int, ...]] = []
  feature_value_by_deck: list[tuple[float, ...]] = []

  for example in examples:
    target_idx.append(label_to_idx[example.target_label_id] if require_targets else 0)
    feature_idx: list[int] = []
    values: list[float] = []
    total_quantity = sum(token.quantity for token in example.tokens)
    if include_card_features and total_quantity > 0:
      for token in example.tokens:
        feature_idx.append(token.card_idx * zone_count + token.zone_idx)
        values.append(token.quantity / total_quantity)

    for package_idx in package_index_by_deck.get(example.deck_id, ()):
      feature_idx.append(package_feature_offset + package_idx)
      values.append(package_scale)

    feature_idx_by_deck.append(tuple(feature_idx))
    feature_value_by_deck.append(tuple(values))

  return _PreparedLinearExamples(
    examples=examples,
    target_idx=torch.tensor(target_idx, dtype=torch.long),
    feature_idx_by_deck=tuple(feature_idx_by_deck),
    feature_value_by_deck=tuple(feature_value_by_deck),
  )


def _prepared_linear_batch(
  prepared: _PreparedLinearExamples,
  deck_numbers: list[int],
  *,
  feature_count: int,
  device: torch.device,
) -> _LinearBatch:
  row_idx: list[int] = []
  feature_idx: list[int] = []
  values: list[float] = []

  for row_number, deck_number in enumerate(deck_numbers):
    row_features = prepared.feature_idx_by_deck[deck_number]
    row_values = prepared.feature_value_by_deck[deck_number]
    row_idx.extend([row_number] * len(row_features))
    feature_idx.extend(row_features)
    values.extend(row_values)

  features = torch.zeros(
    (len(deck_numbers), feature_count),
    dtype=torch.float32,
    device=device,
  )
  if row_idx:
    features.index_put_(
      (
        torch.tensor(row_idx, dtype=torch.long, device=device),
        torch.tensor(feature_idx, dtype=torch.long, device=device),
      ),
      torch.tensor(values, dtype=torch.float32, device=device),
      accumulate=True,
    )

  deck_number_tensor = torch.tensor(deck_numbers, dtype=torch.long)
  return _LinearBatch(
    features=features,
    target_idx=prepared.target_idx.index_select(0, deck_number_tensor).to(device),
  )


def _dense_linear_examples(
  prepared: _PreparedLinearExamples,
  *,
  feature_count: int,
  device: torch.device,
) -> _DenseLinearExamples:
  row_idx: list[int] = []
  feature_idx: list[int] = []
  values: list[float] = []
  for row_number, row_features in enumerate(prepared.feature_idx_by_deck):
    row_values = prepared.feature_value_by_deck[row_number]
    row_idx.extend([row_number] * len(row_features))
    feature_idx.extend(row_features)
    values.extend(row_values)

  features = torch.zeros(
    (len(prepared.examples), feature_count),
    dtype=torch.float32,
    device=device,
  )
  if row_idx:
    features.index_put_(
      (
        torch.tensor(row_idx, dtype=torch.long, device=device),
        torch.tensor(feature_idx, dtype=torch.long, device=device),
      ),
      torch.tensor(values, dtype=torch.float32, device=device),
      accumulate=True,
    )

  return _DenseLinearExamples(
    features=features,
    target_idx=prepared.target_idx.to(device),
  )


def _dense_linear_batch(
  prepared: _DenseLinearExamples,
  deck_numbers: list[int],
) -> _LinearBatch:
  deck_number_tensor = torch.tensor(
    deck_numbers,
    dtype=torch.long,
    device=prepared.features.device,
  )
  return _LinearBatch(
    features=prepared.features.index_select(0, deck_number_tensor),
    target_idx=prepared.target_idx.index_select(0, deck_number_tensor),
  )


def _linear_batch(
  examples: list[ModelExample],
  *,
  label_to_idx: dict[str, int],
  feature_count: int,
  zone_count: int,
  device: torch.device,
  require_targets: bool = True,
  package_index_by_deck: dict[str, tuple[int, ...]] | None = None,
  base_feature_count: int | None = None,
  package_scale: float = 1.0,
) -> _LinearBatch:
  row_idx: list[int] = []
  feature_idx: list[int] = []
  values: list[float] = []
  target_idx: list[int] = []

  for row_number, example in enumerate(examples):
    target_idx.append(label_to_idx[example.target_label_id] if require_targets else 0)
    total_quantity = sum(token.quantity for token in example.tokens)
    if total_quantity <= 0:
      continue

    for token in example.tokens:
      row_idx.append(row_number)
      feature_idx.append(token.card_idx * zone_count + token.zone_idx)
      values.append(token.quantity / total_quantity)

    if package_index_by_deck is not None and base_feature_count is not None:
      for package_idx in package_index_by_deck.get(example.deck_id, ()):
        row_idx.append(row_number)
        feature_idx.append(base_feature_count + package_idx)
        values.append(package_scale)

  features = torch.zeros(
    (len(examples), feature_count),
    dtype=torch.float32,
    device=device,
  )
  if row_idx:
    features.index_put_(
      (
        torch.tensor(row_idx, dtype=torch.long, device=device),
        torch.tensor(feature_idx, dtype=torch.long, device=device),
      ),
      torch.tensor(values, dtype=torch.float32, device=device),
      accumulate=True,
    )

  return _LinearBatch(
    features=features,
    target_idx=torch.tensor(target_idx, dtype=torch.long, device=device),
  )


def _padded_tokens(
  token_embeddings: torch.Tensor,
  deck_idx: torch.Tensor,
  deck_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  if token_embeddings.numel() == 0:
    padded = token_embeddings.new_zeros((deck_count, 1, token_embeddings.shape[-1]))
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
    torch.arange(len(deck_idx), dtype=torch.long, device=token_embeddings.device)
    - token_offsets[deck_idx]
  )
  padded[deck_idx, token_positions] = token_embeddings
  padding_mask[deck_idx, token_positions] = False

  return padded, padding_mask


def _logit_stats(
  logits: torch.Tensor,
  labels: tuple[str, ...],
  k: int,
) -> list[dict[str, Any]]:
  if not labels:
    return [_empty_prediction_stats() for _ in range(logits.shape[0])]

  probabilities = torch.softmax(logits, dim=1)
  entropy = _probability_entropy(probabilities)
  energy = -torch.logsumexp(logits, dim=1)
  if len(labels) > 1:
    normalizer = torch.log(
      torch.tensor(float(len(labels)), dtype=torch.float32, device=probabilities.device)
    )
    normalized_entropy = entropy / normalizer
  else:
    normalized_entropy = torch.zeros_like(entropy)

  values, indexes = torch.topk(
    probabilities,
    k=min(k, len(labels)),
    dim=1,
  )
  rows: list[dict[str, Any]] = []
  for row_scores, row_indexes, row_entropy, row_normalized_entropy, row_energy in zip(
    values.tolist(),
    indexes.tolist(),
    entropy.tolist(),
    normalized_entropy.tolist(),
    energy.tolist(),
    strict=True,
  ):
    top_predictions = [
      (labels[int(label_idx)], float(score))
      for score, label_idx in zip(row_scores, row_indexes, strict=True)
    ]
    rows.append(
      {
        "top_predictions": top_predictions,
        "max_probability": (
          float(top_predictions[0][1])
          if top_predictions
          else 0.0
        ),
        "entropy": float(row_entropy),
        "normalized_entropy": float(row_normalized_entropy),
        "energy": float(row_energy),
      }
    )

  return rows


def _probability_entropy(probabilities: torch.Tensor) -> torch.Tensor:
  safe_probabilities = probabilities.clamp_min(1e-12)
  return -(safe_probabilities * safe_probabilities.log()).sum(dim=1)


def _prototype_stats(
  *,
  top_probabilities: torch.Tensor,
  top_indexes: torch.Tensor,
  top_distances: torch.Tensor,
  entropy: torch.Tensor,
  energy: torch.Tensor,
  nearest_distances: torch.Tensor,
  margins: torch.Tensor,
  labels: tuple[str, ...],
) -> list[dict[str, Any]]:
  if len(labels) > 1:
    normalizer = torch.log(
      torch.tensor(
        float(len(labels)),
        dtype=torch.float32,
        device=top_probabilities.device,
      )
    )
    normalized_entropy = entropy / normalizer
  else:
    normalized_entropy = torch.zeros_like(entropy)

  rows: list[dict[str, Any]] = []
  for (
    row_scores,
    row_indexes,
    row_distances,
    row_entropy,
    row_normalized_entropy,
    row_energy,
    row_nearest_distance,
    row_margin,
  ) in zip(
    top_probabilities.tolist(),
    top_indexes.tolist(),
    top_distances.tolist(),
    entropy.tolist(),
    normalized_entropy.tolist(),
    energy.tolist(),
    nearest_distances.tolist(),
    margins.tolist(),
    strict=True,
  ):
    top_predictions = [
      (labels[int(label_idx)], float(score))
      for score, label_idx in zip(row_scores, row_indexes, strict=True)
    ]
    rows.append(
      {
        "top_predictions": top_predictions,
        "max_probability": (
          float(top_predictions[0][1])
          if top_predictions
          else 0.0
        ),
        "entropy": float(row_entropy),
        "normalized_entropy": float(row_normalized_entropy),
        "energy": float(row_energy),
        "nearest_prototype_distance": float(row_nearest_distance),
        "prototype_margin": float(row_margin),
        "top_prototype_distances": [
          float(distance)
          for distance in row_distances
        ],
      }
    )

  return rows


def _empty_prediction_stats() -> dict[str, Any]:
  return {
    "top_predictions": [],
    "max_probability": 0.0,
    "entropy": None,
    "normalized_entropy": None,
    "energy": None,
    "nearest_prototype_distance": None,
    "prototype_margin": None,
    "top_prototype_distances": [],
  }


def _card_nearest_neighbors(
  embeddings: torch.Tensor,
  card_vocab: tuple[dict[str, Any], ...],
  *,
  anchor_card_idxs: list[int] | None = None,
  card_support: dict[int, int] | None = None,
  anchor_limit: int = 20,
  neighbor_count: int = 5,
) -> list[dict[str, Any]]:
  if anchor_limit <= 0 or neighbor_count <= 0:
    return []

  vocab = sorted(card_vocab, key=lambda row: int(row["card_idx"]))
  vocab_by_idx = {
    int(card["card_idx"]): card
    for card in vocab
  }
  normalized = F.normalize(embeddings, p=2, dim=1)
  if anchor_card_idxs is None:
    anchor_idxs = [int(card["card_idx"]) for card in vocab[:anchor_limit]]
  else:
    anchor_idxs = [
      card_idx
      for card_idx in anchor_card_idxs[:anchor_limit]
      if card_idx in vocab_by_idx
    ]
  rows: list[dict[str, Any]] = []
  for card_idx in anchor_idxs:
    card = vocab_by_idx[card_idx]
    similarities = torch.mv(normalized, normalized[card_idx])
    similarities[card_idx] = -2.0
    scores, indexes = torch.topk(
      similarities,
      k=min(neighbor_count, len(vocab) - 1),
    )
    rows.append(
      {
        "card_idx": card_idx,
        "primary_name": card.get("primary_name"),
        "train_support": (
          None
          if card_support is None
          else card_support.get(card_idx, 0)
        ),
        "neighbors": [
          {
            "card_idx": int(neighbor_idx),
            "primary_name": vocab_by_idx[int(neighbor_idx)].get("primary_name"),
            "train_support": (
              None
              if card_support is None
              else card_support.get(int(neighbor_idx), 0)
            ),
            "score": float(score),
          }
          for score, neighbor_idx in zip(
            scores.tolist(),
            indexes.tolist(),
            strict=True,
          )
        ],
      }
    )
  return rows


def _example_batches(
  examples: list[ModelExample],
  batch_size: int,
) -> list[list[ModelExample]]:
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

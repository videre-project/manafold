from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
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
SET_TRANSFORMER_POOLING_HYPERGEOMETRIC = "hypergeometric"
SET_TRANSFORMER_POOLING_MODES = (
  SET_TRANSFORMER_POOLING_PMA,
  SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED,
  SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
)
DEFAULT_HYPERGEOMETRIC_DRAW_COUNT = 7
DEFAULT_PARTIAL_OBSERVATION_IDENTITY_COUNTS = (5, 10, 20)
DEFAULT_PARTIAL_CLASSIFICATION_WEIGHT = 0.5
DEFAULT_PARTIAL_CONSISTENCY_WEIGHT = 0.5
DEFAULT_PARTIAL_MIN_COVERAGE_WEIGHT = 0.25
DEFAULT_PARTIAL_LATENT_WEIGHT = 0.5
DEFAULT_PARTIAL_TEACHER_DECAY = 0.99
PARTIAL_CORRUPTION_FIXED = "fixed_identity_counts"
PARTIAL_CORRUPTION_MIXTURE = "regular_extreme_evidence"
PARTIAL_CORRUPTION_POLICIES = (
  PARTIAL_CORRUPTION_FIXED,
  PARTIAL_CORRUPTION_MIXTURE,
)
TRAINING_SAMPLING_NATURAL = "natural"
TRAINING_SAMPLING_NATURAL_SQRT_BALANCED = "natural_sqrt_balanced"
TRAINING_SAMPLING_POLICIES = (
  TRAINING_SAMPLING_NATURAL,
  TRAINING_SAMPLING_NATURAL_SQRT_BALANCED,
)
DEFAULT_BALANCED_SAMPLING_FRACTION = 0.5
DEFAULT_BALANCED_SAMPLING_MAX_MULTIPLIER = 4.0
PROTOTYPE_DISTANCE_EUCLIDEAN = "euclidean"
DEEPSETS_ARCHITECTURE_BASE = "base"
DEEPSETS_ARCHITECTURE_PLUSPLUS = "plusplus"
DEEPSETS_ARCHITECTURES = (
  DEEPSETS_ARCHITECTURE_BASE,
  DEEPSETS_ARCHITECTURE_PLUSPLUS,
)


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
    card_embedding_init: torch.Tensor | None = None,
    card_embedding_init_card_idxs: tuple[int, ...] | None = None,
    freeze_card_embedding_steps: int = 0,
    architecture: str = DEEPSETS_ARCHITECTURE_BASE,
    dropout: float = 0.0,
    label_smoothing: float = 0.0,
    relation_smoothing_neighbors: dict[str, dict[str, float]] | None = None,
    metric_loss_weight: float = 0.0,
    metric_loss_min_label_support: int = 10,
    metric_loss_temperature: float = 0.1,
    partial_observation_training: bool = False,
    partial_observation_identity_counts: tuple[int, ...] = (
      DEFAULT_PARTIAL_OBSERVATION_IDENTITY_COUNTS
    ),
    partial_classification_weight: float = DEFAULT_PARTIAL_CLASSIFICATION_WEIGHT,
    partial_consistency_weight: float = DEFAULT_PARTIAL_CONSISTENCY_WEIGHT,
    partial_min_coverage_weight: float = DEFAULT_PARTIAL_MIN_COVERAGE_WEIGHT,
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
    if freeze_card_embedding_steps < 0:
      raise ValueError("freeze_card_embedding_steps must be non-negative.")
    if architecture not in DEEPSETS_ARCHITECTURES:
      raise ValueError(f"Unsupported Deep Sets architecture: {architecture}")
    if dropout < 0.0 or dropout >= 1.0:
      raise ValueError("dropout must be in [0.0, 1.0).")
    if label_smoothing < 0.0 or label_smoothing >= 1.0:
      raise ValueError("label_smoothing must be in [0.0, 1.0).")
    if metric_loss_weight < 0.0:
      raise ValueError("metric_loss_weight must be non-negative.")
    if metric_loss_min_label_support <= 0:
      raise ValueError("metric_loss_min_label_support must be positive.")
    if metric_loss_temperature <= 0.0:
      raise ValueError("metric_loss_temperature must be positive.")
    _validate_partial_observation_config(
      enabled=partial_observation_training,
      identity_counts=partial_observation_identity_counts,
      classification_weight=partial_classification_weight,
      consistency_weight=partial_consistency_weight,
      min_coverage_weight=partial_min_coverage_weight,
    )

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
    self.freeze_card_embedding_steps = freeze_card_embedding_steps
    self.architecture = architecture
    self.dropout = dropout
    self.label_smoothing = label_smoothing
    self.metric_loss_weight = metric_loss_weight
    self.metric_loss_min_label_support = metric_loss_min_label_support
    self.metric_loss_temperature = metric_loss_temperature
    self.partial_observation_training = partial_observation_training
    self.partial_observation_identity_counts = partial_observation_identity_counts
    self.partial_classification_weight = partial_classification_weight
    self.partial_consistency_weight = partial_consistency_weight
    self.partial_min_coverage_weight = partial_min_coverage_weight
    self.device = _resolve_device(device)
    self._label_to_idx = {
      label: index
      for index, label in enumerate(labels)
    }
    self._rng = random.Random(seed)
    network_class = (
      _DeepSetsPlusPlusNetwork
      if architecture == DEEPSETS_ARCHITECTURE_PLUSPLUS
      else _DeepSetsNetwork
    )
    self._network = network_class(
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
      dropout=dropout,
    ).to(self.device)
    self.card_embedding_init_count = self._copy_card_embedding_init(
      card_embedding_init,
      card_embedding_init_card_idxs,
    )
    self._relation_smoothing_targets: torch.Tensor | None = None
    self._relation_smoothing_has_neighbors: torch.Tensor | None = None
    self._relation_smoothing_config = self._build_relation_smoothing_targets(
      relation_smoothing_neighbors or {},
    )
    self._metric_prototypes: nn.Parameter | None = None
    if metric_loss_weight > 0.0:
      self._metric_prototypes = nn.Parameter(
        torch.empty(len(labels), hidden_dim, device=self.device)
      )
      nn.init.normal_(
        self._metric_prototypes,
        mean=0.0,
        std=hidden_dim ** -0.5,
      )

  def _copy_card_embedding_init(
    self,
    card_embedding_init: torch.Tensor | None,
    card_embedding_init_card_idxs: tuple[int, ...] | None,
  ) -> int:
    if card_embedding_init is None:
      return 0

    embedding_weight = self._network.card_embedding.weight
    init = card_embedding_init.detach().to(
      device=self.device,
      dtype=embedding_weight.dtype,
    )
    if tuple(init.shape) != tuple(embedding_weight.shape):
      raise ValueError(
        "card_embedding_init must have shape "
        f"{tuple(embedding_weight.shape)}, got {tuple(init.shape)}."
      )

    with torch.no_grad():
      if card_embedding_init_card_idxs is None:
        embedding_weight.copy_(init)
        return int(init.shape[0])
      if not card_embedding_init_card_idxs:
        return 0
      card_idxs = torch.tensor(
        card_embedding_init_card_idxs,
        dtype=torch.long,
        device=self.device,
      )
      embedding_weight.index_copy_(0, card_idxs, init.index_select(0, card_idxs))
      return int(card_idxs.numel())

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
    metric_support = self._metric_support_tensors(trainable)
    card_embedding_frozen = self.freeze_card_embedding_steps > 0
    if card_embedding_frozen:
      self._network.card_embedding.weight.requires_grad_(False)

    parameters = list(self._network.parameters())
    if self._metric_prototypes is not None:
      parameters.append(self._metric_prototypes)
    optimizer = torch.optim.Adam(
      parameters,
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
      partial_prepared = None
      if getattr(self, "partial_observation_training", False):
        partial_prepared = _prepare_set_examples(
          partial_observation_training_views(
            trainable,
            rng=self._rng,
            identity_counts=self.partial_observation_identity_counts,
          ),
          label_to_idx=self._label_to_idx,
          quantity_count=self.quantity_count,
          package_features=self.package_features,
        )

      self._network.train()
      total_loss = 0.0
      total_ce_loss = 0.0
      total_metric_loss = 0.0
      total_partial_ce_loss = 0.0
      total_partial_consistency_loss = 0.0
      total_partial_coverage = 0.0
      total_correct = 0
      total_count = 0
      metric_batch_count = 0
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
        embeddings = None
        if self._metric_prototypes is not None:
          embeddings = self._network.encode(batch)
          logits = self._logits_from_embeddings(batch, embeddings)
        else:
          logits = self._network(batch)
        ce_loss = self._classification_loss(logits, batch.target_idx)
        loss = ce_loss
        partial_ce_loss = None
        partial_consistency_loss = None
        partial_coverage = None
        if partial_prepared is not None:
          partial_batch = _prepared_set_batch(
            partial_prepared,
            batch_indexes,
            device=self.device,
            package_scale=self.package_scale,
          )
          partial_logits = self._network(partial_batch)
          (
            partial_ce_loss,
            partial_consistency_loss,
            partial_coverage,
          ) = _partial_observation_losses(
            full_logits=logits,
            partial_logits=partial_logits,
            target_idx=partial_batch.target_idx,
            coverage=partial_batch.observation_coverage,
            label_smoothing=self.label_smoothing,
            min_coverage_weight=self.partial_min_coverage_weight,
          )
          loss = (
            loss
            + self.partial_classification_weight * partial_ce_loss
            + self.partial_consistency_weight * partial_consistency_loss
          ) / (
            1.0
            + self.partial_classification_weight
            + self.partial_consistency_weight
          )
        metric_loss = None
        if embeddings is not None and metric_support is not None:
          metric_loss = self._metric_auxiliary_loss(
            embeddings,
            batch.target_idx,
            eligible_label_idx=metric_support["eligible_label_idx"],
            target_remap=metric_support["target_remap"],
          )
          if metric_loss is not None:
            loss = loss + self.metric_loss_weight * metric_loss
        loss.backward()
        optimizer.step()
        optimizer_steps += 1
        if (
          card_embedding_frozen
          and optimizer_steps >= self.freeze_card_embedding_steps
        ):
          self._network.card_embedding.weight.requires_grad_(True)
          card_embedding_frozen = False

        batch_count = len(batch_indexes)
        total_loss += float(loss.item()) * batch_count
        total_ce_loss += float(ce_loss.item()) * batch_count
        if metric_loss is not None:
          total_metric_loss += float(metric_loss.item()) * batch_count
          metric_batch_count += batch_count
        if partial_ce_loss is not None:
          total_partial_ce_loss += float(partial_ce_loss.item()) * batch_count
          total_partial_consistency_loss += (
            float(partial_consistency_loss.item()) * batch_count
          )
          total_partial_coverage += float(partial_coverage.item()) * batch_count
        total_correct += int((logits.argmax(dim=1) == batch.target_idx).sum().item())
        total_count += batch_count

      train_loss = total_loss / total_count if total_count else None
      train_ce_loss = total_ce_loss / total_count if total_count else None
      train_metric_loss = (
        total_metric_loss / metric_batch_count
        if metric_batch_count
        else None
      )
      train_partial_ce_loss = (
        total_partial_ce_loss / total_count
        if partial_prepared is not None and total_count
        else None
      )
      train_partial_consistency_loss = (
        total_partial_consistency_loss / total_count
        if partial_prepared is not None and total_count
        else None
      )
      train_partial_coverage = (
        total_partial_coverage / total_count
        if partial_prepared is not None and total_count
        else None
      )
      train_relation_smoothed_fraction = (
        self._relation_smoothed_fraction(trainable_prepared, epoch_indexes)
        if self._relation_smoothing_has_neighbors is not None
        else None
      )
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
          "train_ce_loss": train_ce_loss,
          "train_metric_loss": train_metric_loss,
          "train_partial_ce_loss": train_partial_ce_loss,
          "train_partial_consistency_loss": train_partial_consistency_loss,
          "train_partial_coverage": train_partial_coverage,
          "train_relation_smoothed_fraction": train_relation_smoothed_fraction,
          "train_accuracy": train_accuracy,
          "validation_loss": validation_loss,
          "validation_accuracy": validation_accuracy,
          "optimizer_steps": optimizer_steps,
        }
      )

    if validation:
      self._restore_state(best_state)
    if card_embedding_frozen:
      self._network.card_embedding.weight.requires_grad_(True)

    return {
      "history": history,
      "best_validation_epoch": best_epoch,
      "best_validation_metric": best_metric,
      "best_validation_loss": best_loss,
      "optimizer_steps": optimizer_steps,
      "completed_epochs": epoch,
      "metric_auxiliary": self.metric_auxiliary_config(),
      "relation_smoothing": self.relation_smoothing_config(),
      "partial_observation": self.partial_observation_config(),
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

  def metric_auxiliary_config(self) -> dict[str, Any]:
    return {
      "enabled": self._metric_prototypes is not None,
      "loss": "cosine_prototype_cross_entropy",
      "weight": self.metric_loss_weight,
      "min_label_support": self.metric_loss_min_label_support,
      "temperature": self.metric_loss_temperature,
    }

  def relation_smoothing_config(self) -> dict[str, Any]:
    return dict(self._relation_smoothing_config)

  def partial_observation_config(self) -> dict[str, Any]:
    return {
      "enabled": getattr(self, "partial_observation_training", False),
      "identity_counts": list(
        getattr(
          self,
          "partial_observation_identity_counts",
          DEFAULT_PARTIAL_OBSERVATION_IDENTITY_COUNTS,
        )
      ),
      "classification_weight": getattr(
        self,
        "partial_classification_weight",
        DEFAULT_PARTIAL_CLASSIFICATION_WEIGHT,
      ),
      "consistency_weight": getattr(
        self,
        "partial_consistency_weight",
        DEFAULT_PARTIAL_CONSISTENCY_WEIGHT,
      ),
      "min_coverage_weight": getattr(
        self,
        "partial_min_coverage_weight",
        DEFAULT_PARTIAL_MIN_COVERAGE_WEIGHT,
      ),
    }

  def _build_relation_smoothing_targets(
    self,
    neighbors_by_label: dict[str, dict[str, float]],
  ) -> dict[str, Any]:
    label_count = len(self.labels)
    if not neighbors_by_label or self.label_smoothing <= 0.0:
      return {
        "enabled": False,
        "smoothing_mass": self.label_smoothing,
        "relation_label_count": 0,
        "relation_edge_count": 0,
      }

    targets = torch.full(
      (label_count, label_count),
      self.label_smoothing / label_count,
      dtype=torch.float32,
      device=self.device,
    )
    targets[
      torch.arange(label_count, dtype=torch.long, device=self.device),
      torch.arange(label_count, dtype=torch.long, device=self.device),
    ] += 1.0 - self.label_smoothing
    has_neighbors = torch.zeros(label_count, dtype=torch.bool, device=self.device)

    relation_edge_count = 0
    for label_id, neighbor_weights in neighbors_by_label.items():
      label_idx = self._label_to_idx.get(label_id)
      if label_idx is None:
        continue
      filtered: list[tuple[int, float]] = []
      for neighbor_label_id, raw_weight in neighbor_weights.items():
        neighbor_idx = self._label_to_idx.get(neighbor_label_id)
        if neighbor_idx is None or neighbor_idx == label_idx:
          continue
        weight = float(raw_weight)
        if weight > 0.0:
          filtered.append((neighbor_idx, weight))
      weight_sum = sum(weight for _, weight in filtered)
      if weight_sum <= 0.0:
        continue

      targets[label_idx].zero_()
      targets[label_idx, label_idx] = 1.0 - self.label_smoothing
      for neighbor_idx, weight in filtered:
        targets[label_idx, neighbor_idx] += self.label_smoothing * weight / weight_sum
      has_neighbors[label_idx] = True
      relation_edge_count += len(filtered)

    relation_label_count = int(has_neighbors.sum().item())
    if relation_label_count <= 0:
      return {
        "enabled": False,
        "smoothing_mass": self.label_smoothing,
        "relation_label_count": 0,
        "relation_edge_count": 0,
      }

    self._relation_smoothing_targets = targets
    self._relation_smoothing_has_neighbors = has_neighbors
    return {
      "enabled": True,
      "smoothing_mass": self.label_smoothing,
      "relation_label_count": relation_label_count,
      "relation_edge_count": relation_edge_count,
    }

  def _classification_loss(
    self,
    logits: torch.Tensor,
    target_idx: torch.Tensor,
  ) -> torch.Tensor:
    if self._relation_smoothing_targets is None:
      return F.cross_entropy(
        logits,
        target_idx,
        label_smoothing=self.label_smoothing,
      )
    target_distribution = self._relation_smoothing_targets.index_select(
      0,
      target_idx,
    )
    log_probabilities = F.log_softmax(logits, dim=1)
    return -(target_distribution * log_probabilities).sum(dim=1).mean()

  def _relation_smoothed_fraction(
    self,
    prepared: "_PreparedSetExamples",
    epoch_indexes: list[int],
  ) -> float | None:
    if self._relation_smoothing_has_neighbors is None or not epoch_indexes:
      return None
    example_indexes = torch.tensor(
      epoch_indexes,
      dtype=torch.long,
    )
    target_indexes = torch.tensor(
      prepared.target_idx.index_select(0, example_indexes).tolist(),
      dtype=torch.long,
      device=self.device,
    )
    smoothed_count = int(
      self._relation_smoothing_has_neighbors.index_select(
        0,
        target_indexes,
      ).sum().item()
    )
    return smoothed_count / len(epoch_indexes)

  def _metric_support_tensors(
    self,
    trainable: list[ModelExample],
  ) -> dict[str, torch.Tensor] | None:
    if self._metric_prototypes is None:
      return None
    support = torch.zeros(len(self.labels), dtype=torch.long, device=self.device)
    for example in trainable:
      label_idx = self._label_to_idx.get(example.target_label_id)
      if label_idx is not None:
        support[label_idx] += 1
    eligible_label_idx = torch.nonzero(
      support >= self.metric_loss_min_label_support,
      as_tuple=False,
    ).flatten()
    if int(eligible_label_idx.numel()) < 2:
      return None
    target_remap = torch.full(
      (len(self.labels),),
      -1,
      dtype=torch.long,
      device=self.device,
    )
    target_remap.index_copy_(
      0,
      eligible_label_idx,
      torch.arange(
        int(eligible_label_idx.numel()),
        dtype=torch.long,
        device=self.device,
      ),
    )
    return {
      "eligible_label_idx": eligible_label_idx,
      "target_remap": target_remap,
    }

  def _metric_auxiliary_loss(
    self,
    embeddings: torch.Tensor,
    target_idx: torch.Tensor,
    *,
    eligible_label_idx: torch.Tensor,
    target_remap: torch.Tensor,
  ) -> torch.Tensor | None:
    if self._metric_prototypes is None:
      return None
    remapped_targets = target_remap.index_select(0, target_idx)
    mask = remapped_targets >= 0
    if int(mask.sum().item()) <= 0:
      return None
    selected_embeddings = F.normalize(embeddings[mask], p=2, dim=1)
    selected_prototypes = F.normalize(
      self._metric_prototypes.index_select(0, eligible_label_idx),
      p=2,
      dim=1,
    )
    logits = (
      torch.matmul(selected_embeddings, selected_prototypes.transpose(0, 1))
      / self.metric_loss_temperature
    )
    return F.cross_entropy(logits, remapped_targets[mask])

  def _logits_from_embeddings(
    self,
    batch: "_Batch",
    embeddings: torch.Tensor,
  ) -> torch.Tensor:
    if (
      self._network.package_projection is None
      and self.extra_feature_dim == 0
    ):
      return self._network.rho(embeddings)
    return self._network(batch)

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
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    hypergeometric_draw_count: int = DEFAULT_HYPERGEOMETRIC_DRAW_COUNT,
    partial_observation_training: bool = False,
    partial_observation_identity_counts: tuple[int, ...] = (
      DEFAULT_PARTIAL_OBSERVATION_IDENTITY_COUNTS
    ),
    partial_classification_weight: float = DEFAULT_PARTIAL_CLASSIFICATION_WEIGHT,
    partial_consistency_weight: float = DEFAULT_PARTIAL_CONSISTENCY_WEIGHT,
    partial_min_coverage_weight: float = DEFAULT_PARTIAL_MIN_COVERAGE_WEIGHT,
    partial_latent_weight: float = 0.0,
    partial_contextual_weight: float = 0.0,
    partial_teacher_decay: float = DEFAULT_PARTIAL_TEACHER_DECAY,
    partial_corruption_policy: str = PARTIAL_CORRUPTION_FIXED,
    observation_conditioning: bool = False,
    training_sampling_policy: str = TRAINING_SAMPLING_NATURAL,
    balanced_sampling_fraction: float = DEFAULT_BALANCED_SAMPLING_FRACTION,
    balanced_sampling_max_multiplier: float = (
      DEFAULT_BALANCED_SAMPLING_MAX_MULTIPLIER
    ),
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
    if weight_decay < 0.0:
      raise ValueError("weight_decay must be non-negative.")
    if label_smoothing < 0.0 or label_smoothing >= 1.0:
      raise ValueError("label_smoothing must be in [0.0, 1.0).")
    if hypergeometric_draw_count <= 0:
      raise ValueError("hypergeometric_draw_count must be positive.")
    if partial_corruption_policy not in PARTIAL_CORRUPTION_POLICIES:
      raise ValueError(
        f"Unsupported partial corruption policy: {partial_corruption_policy}"
      )
    if training_sampling_policy not in TRAINING_SAMPLING_POLICIES:
      raise ValueError(
        f"Unsupported training sampling policy: {training_sampling_policy}"
      )
    if not 0.0 <= balanced_sampling_fraction <= 1.0:
      raise ValueError("balanced_sampling_fraction must be in [0.0, 1.0].")
    if balanced_sampling_max_multiplier < 1.0:
      raise ValueError("balanced_sampling_max_multiplier must be at least 1.0.")
    _validate_partial_observation_config(
      enabled=partial_observation_training,
      identity_counts=partial_observation_identity_counts,
      classification_weight=partial_classification_weight,
      consistency_weight=partial_consistency_weight,
      min_coverage_weight=partial_min_coverage_weight,
      latent_weight=partial_latent_weight,
      contextual_weight=partial_contextual_weight,
      teacher_decay=partial_teacher_decay,
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)

    self.labels = labels
    self.card_count = card_count
    self.zone_count = zone_count
    self.quantity_count = quantity_count
    self.pooling = pooling
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.label_smoothing = label_smoothing
    self.hypergeometric_draw_count = hypergeometric_draw_count
    self.partial_observation_training = partial_observation_training
    self.partial_observation_identity_counts = partial_observation_identity_counts
    self.partial_classification_weight = partial_classification_weight
    self.partial_consistency_weight = partial_consistency_weight
    self.partial_min_coverage_weight = partial_min_coverage_weight
    self.partial_latent_weight = partial_latent_weight
    self.partial_contextual_weight = partial_contextual_weight
    self.partial_teacher_decay = partial_teacher_decay
    self.partial_corruption_policy = partial_corruption_policy
    self.observation_conditioning = observation_conditioning
    self.training_sampling_policy = training_sampling_policy
    self.balanced_sampling_fraction = balanced_sampling_fraction
    self.balanced_sampling_max_multiplier = balanced_sampling_max_multiplier
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
      observation_conditioning=observation_conditioning,
    ).to(self.device)
    self._partial_predictor: nn.Module | None = None
    self._partial_teacher: _SetTransformerNetwork | None = None
    if self.partial_latent_weight > 0.0 or self.partial_contextual_weight > 0.0:
      self._partial_predictor = _PartialLatentPredictor(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
      ).to(self.device)
      self._partial_teacher = deepcopy(self._network).to(self.device)
      self._partial_teacher.requires_grad_(False)
      self._partial_teacher.eval()

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
      quantity_weighting=self.pooling,
      hypergeometric_draw_count=self.hypergeometric_draw_count,
    )
    validation_prepared = _prepare_set_examples(
      validation,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
      quantity_weighting=self.pooling,
      hypergeometric_draw_count=self.hypergeometric_draw_count,
    )
    card_information = (
      card_identity_idf(trainable)
      if self.partial_corruption_policy == PARTIAL_CORRUPTION_MIXTURE
      else None
    )
    optimizer_parameters = list(self._network.parameters())
    if self._partial_predictor is not None:
      optimizer_parameters.extend(self._partial_predictor.parameters())
    optimizer = torch.optim.Adam(
      optimizer_parameters,
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
      if self.training_sampling_policy == TRAINING_SAMPLING_NATURAL_SQRT_BALANCED:
        epoch_indexes = mixed_class_balanced_epoch_indexes(
          trainable,
          rng=self._rng,
          balanced_fraction=self.balanced_sampling_fraction,
          max_multiplier=self.balanced_sampling_max_multiplier,
        )
      else:
        epoch_indexes = list(range(len(trainable)))
        if shuffle:
          self._rng.shuffle(epoch_indexes)
      partial_prepared = None
      if self.partial_observation_training:
        partial_prepared = _prepare_set_examples(
          partial_observation_training_views(
            trainable,
            rng=self._rng,
            identity_counts=self.partial_observation_identity_counts,
            corruption_policy=self.partial_corruption_policy,
            card_information=card_information,
          ),
          label_to_idx=self._label_to_idx,
          quantity_count=self.quantity_count,
          quantity_weighting=self.pooling,
          hypergeometric_draw_count=self.hypergeometric_draw_count,
        )

      self._network.train()
      if self._partial_predictor is not None:
        self._partial_predictor.train()
      if self._partial_teacher is not None:
        self._partial_teacher.eval()
      total_loss = 0.0
      total_full_ce_loss = 0.0
      total_partial_ce_loss = 0.0
      total_partial_consistency_loss = 0.0
      total_partial_latent_loss = 0.0
      total_partial_contextual_loss = 0.0
      total_partial_coverage = 0.0
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
        full_embedding = self._network.encode(batch)
        logits = self._network.rho(full_embedding)
        loss = F.cross_entropy(
          logits,
          batch.target_idx,
          label_smoothing=self.label_smoothing,
        )
        full_ce_loss = loss
        partial_ce_loss = None
        partial_consistency_loss = None
        partial_latent_loss = None
        partial_contextual_loss = None
        partial_coverage = None
        if partial_prepared is not None:
          partial_batch = _prepared_set_batch(
            partial_prepared,
            batch_indexes,
            device=self.device,
          )
          if self.partial_contextual_weight > 0.0:
            (
              partial_embedding,
              partial_contextual_tokens,
            ) = self._network.encode_with_contextual_tokens(partial_batch)
          else:
            partial_embedding = self._network.encode(partial_batch)
            partial_contextual_tokens = None
          partial_logits = self._network.rho(partial_embedding)
          (
            partial_ce_loss,
            partial_consistency_loss,
            partial_coverage,
          ) = _partial_observation_losses(
            full_logits=logits,
            partial_logits=partial_logits,
            target_idx=partial_batch.target_idx,
            coverage=partial_batch.observation_coverage,
            label_smoothing=self.label_smoothing,
            min_coverage_weight=self.partial_min_coverage_weight,
          )
          objective_weight = 1.0 + self.partial_classification_weight
          loss = loss + self.partial_classification_weight * partial_ce_loss
          if self.partial_consistency_weight > 0.0:
            loss = loss + (
              self.partial_consistency_weight * partial_consistency_loss
            )
            objective_weight += self.partial_consistency_weight
          if self.partial_latent_weight > 0.0:
            if self._partial_predictor is None or self._partial_teacher is None:
              raise AssertionError("Partial latent modules are not initialized.")
            with torch.no_grad():
              teacher_embedding = self._partial_teacher.encode(batch)
            partial_latent_loss = _partial_latent_prediction_loss(
              predicted=self._partial_predictor(partial_embedding),
              target=teacher_embedding,
            )
            loss = loss + self.partial_latent_weight * partial_latent_loss
            objective_weight += self.partial_latent_weight
          if self.partial_contextual_weight > 0.0:
            if (
              self._partial_predictor is None
              or self._partial_teacher is None
              or partial_contextual_tokens is None
            ):
              raise AssertionError("Partial contextual modules are not initialized.")
            with torch.no_grad():
              teacher_contextual_tokens = (
                self._partial_teacher.encode_contextual_tokens(batch)
              )
              teacher_token_indexes = _matching_full_token_indexes(
                full_batch=batch,
                partial_batch=partial_batch,
                card_count=self.card_count,
                zone_count=self.zone_count,
              )
              contextual_targets = teacher_contextual_tokens.index_select(
                0,
                teacher_token_indexes,
              )
            partial_contextual_loss = _partial_contextual_prediction_loss(
              predicted=self._partial_predictor(partial_contextual_tokens),
              target=contextual_targets,
              deck_idx=partial_batch.deck_idx,
              deck_count=partial_batch.deck_count,
            )
            loss = loss + (
              self.partial_contextual_weight * partial_contextual_loss
            )
            objective_weight += self.partial_contextual_weight
          loss = loss / objective_weight
        loss.backward()
        optimizer.step()
        if self._partial_teacher is not None:
          _update_ema_module(
            self._partial_teacher,
            self._network,
            decay=self.partial_teacher_decay,
          )
        optimizer_steps += 1

        batch_count = len(batch_indexes)
        total_loss += float(loss.item()) * batch_count
        total_full_ce_loss += float(full_ce_loss.item()) * batch_count
        if partial_ce_loss is not None:
          total_partial_ce_loss += float(partial_ce_loss.item()) * batch_count
          total_partial_consistency_loss += (
            float(partial_consistency_loss.item()) * batch_count
          )
          if partial_latent_loss is not None:
            total_partial_latent_loss += (
              float(partial_latent_loss.item()) * batch_count
            )
          if partial_contextual_loss is not None:
            total_partial_contextual_loss += (
              float(partial_contextual_loss.item()) * batch_count
            )
          total_partial_coverage += float(partial_coverage.item()) * batch_count
        total_correct += int((logits.argmax(dim=1) == batch.target_idx).sum().item())
        total_count += batch_count

      train_loss = total_loss / total_count if total_count else None
      train_full_ce_loss = total_full_ce_loss / total_count if total_count else None
      train_partial_ce_loss = (
        total_partial_ce_loss / total_count
        if partial_prepared is not None and total_count
        else None
      )
      train_partial_consistency_loss = (
        total_partial_consistency_loss / total_count
        if partial_prepared is not None and total_count
        else None
      )
      train_partial_latent_loss = (
        total_partial_latent_loss / total_count
        if self._partial_teacher is not None and total_count
        else None
      )
      train_partial_contextual_loss = (
        total_partial_contextual_loss / total_count
        if self.partial_contextual_weight > 0.0 and total_count
        else None
      )
      train_partial_coverage = (
        total_partial_coverage / total_count
        if partial_prepared is not None and total_count
        else None
      )
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
          "train_full_ce_loss": train_full_ce_loss,
          "train_partial_ce_loss": train_partial_ce_loss,
          "train_partial_consistency_loss": train_partial_consistency_loss,
          "train_partial_latent_loss": train_partial_latent_loss,
          "train_partial_contextual_loss": train_partial_contextual_loss,
          "train_partial_coverage": train_partial_coverage,
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
      "partial_observation": self.partial_observation_config(),
    }

  def partial_observation_config(self) -> dict[str, Any]:
    return {
      "enabled": self.partial_observation_training,
      "identity_counts": list(self.partial_observation_identity_counts),
      "classification_weight": self.partial_classification_weight,
      "consistency_weight": self.partial_consistency_weight,
      "min_coverage_weight": self.partial_min_coverage_weight,
      "latent_weight": self.partial_latent_weight,
      "contextual_weight": self.partial_contextual_weight,
      "teacher_decay": (
        self.partial_teacher_decay
        if self.partial_latent_weight > 0.0
        or self.partial_contextual_weight > 0.0
        else None
      ),
      "objective": (
        "ce_kl_latent"
        if self.partial_latent_weight > 0.0
        and self.partial_consistency_weight > 0.0
        else "ce_latent"
        if self.partial_latent_weight > 0.0
        else "ce_kl_contextual"
        if self.partial_contextual_weight > 0.0
        and self.partial_consistency_weight > 0.0
        else "ce_contextual"
        if self.partial_contextual_weight > 0.0
        else "ce_kl"
      ),
      "corruption_policy": self.partial_corruption_policy,
      "corruption_mixture": (
        {
          "regular_probability": 0.25,
          "regular_retention_range": [0.5, 0.8],
          "extreme_probability": 0.5,
          "extreme_identity_counts": [5, 10],
          "evidence_probability": 0.25,
          "evidence_identity_counts": list(
            self.partial_observation_identity_counts
          ),
          "evidence_weighting": "train_deck_idf",
        }
        if self.partial_corruption_policy == PARTIAL_CORRUPTION_MIXTURE
        else None
      ),
      "observation_conditioning": self.observation_conditioning,
      "training_sampling": {
        "policy": self.training_sampling_policy,
        "natural_fraction": (
          1.0 - self.balanced_sampling_fraction
          if self.training_sampling_policy
          == TRAINING_SAMPLING_NATURAL_SQRT_BALANCED
          else 1.0
        ),
        "balanced_fraction": (
          self.balanced_sampling_fraction
          if self.training_sampling_policy
          == TRAINING_SAMPLING_NATURAL_SQRT_BALANCED
          else 0.0
        ),
        "balance_power": 0.5,
        "max_multiplier": self.balanced_sampling_max_multiplier,
        "epoch_size_policy": "preserve_train_example_count",
      },
    }

  def inference_parameter_count(self) -> int:
    return sum(parameter.numel() for parameter in self._network.parameters())

  def training_parameter_count(self) -> int:
    predictor_count = (
      sum(parameter.numel() for parameter in self._partial_predictor.parameters())
      if self._partial_predictor is not None
      else 0
    )
    return self.inference_parameter_count() + predictor_count

  def state_dict_for_artifact(self) -> dict[str, torch.Tensor]:
    return self._state()

  def load_state_dict_from_artifact(self, state: dict[str, torch.Tensor]) -> None:
    self._restore_state(state)

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
      quantity_weighting=self.pooling,
      hypergeometric_draw_count=self.hypergeometric_draw_count,
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
      quantity_weighting=self.pooling,
      hypergeometric_draw_count=self.hypergeometric_draw_count,
    )
    rows: list[torch.Tensor] = []
    self._network.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(len(examples), batch_size):
        batch = _prepared_set_batch(
          prepared,
          batch_indexes,
          device=self.device,
        )
        rows.append(self._network(batch).detach().cpu())
    if not rows:
      return torch.empty((0, len(self.labels)), dtype=torch.float32)
    return torch.cat(rows, dim=0)

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
      quantity_weighting=self.pooling,
      hypergeometric_draw_count=self.hypergeometric_draw_count,
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

  def deck_embedding_tensor(
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
      quantity_weighting=self.pooling,
      hypergeometric_draw_count=self.hypergeometric_draw_count,
    )
    embeddings: list[torch.Tensor] = []
    self._network.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(len(examples), batch_size):
        batch = _prepared_set_batch(
          prepared,
          batch_indexes,
          device=self.device,
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
      quantity_weighting=self.pooling,
      hypergeometric_draw_count=self.hypergeometric_draw_count,
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
        loss = F.cross_entropy(
          logits,
          batch.target_idx,
          label_smoothing=self.label_smoothing,
        )
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
      raise ValueError("Deep Sets++ does not support head attribution controls.")

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

  def forward(self, batch: "_Batch") -> torch.Tensor:
    return self.rho(self.encode(batch))

  def encode(self, batch: "_Batch") -> torch.Tensor:
    token_embeddings = (
      self.card_embedding(batch.card_idx)
      + self.zone_embedding(batch.zone_idx)
      + self.quantity_embedding(batch.quantity_idx)
    )
    token_latents = self.phi(self.input_projection(token_embeddings))
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
    observation_conditioning: bool = False,
  ) -> None:
    super().__init__()
    self.pooling = pooling
    self.observation_conditioning = observation_conditioning
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
    self.observation_projection = (
      nn.Sequential(
        nn.Linear(3, embedding_dim),
        nn.Tanh(),
      )
      if observation_conditioning
      else None
    )

  def forward(self, batch: "_Batch") -> torch.Tensor:
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
        "ONNX export does not support observation-conditioned Set Transformers."
      )
    token_embeddings = (
      self.card_embedding(card_idx)
      + self.zone_embedding(zone_idx)
      + self.quantity_embedding(quantity_idx)
    )
    encoded = token_embeddings.unsqueeze(0)
    for layer in self.encoder.layers:
      normed = layer.norm1(encoded)
      encoded = encoded + _onnx_multihead_attention(
        normed,
        normed,
        normed,
        layer.self_attn,
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
        raise AssertionError(f"Unhandled Set Transformer pooling mode: {self.pooling}")
      pooled = _onnx_multihead_attention(
        self.seed_vector,
        encoded,
        encoded,
        self.pooling_attention,
      )
      pooled = pooled.squeeze(1)
    return self.rho(pooled)

  def encode(self, batch: "_Batch") -> torch.Tensor:
    pooled, _ = self.encode_with_contextual_tokens(batch)
    return pooled

  def encode_contextual_tokens(self, batch: "_Batch") -> torch.Tensor:
    _, contextual_tokens = self.encode_with_contextual_tokens(batch)
    return contextual_tokens

  def encode_with_contextual_tokens(
    self,
    batch: "_Batch",
  ) -> tuple[torch.Tensor, torch.Tensor]:
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
      return self._condition_pooled(pooled, batch), contextual_tokens

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
    return self._condition_pooled(pooled.squeeze(1), batch), contextual_tokens

  def _condition_pooled(
    self,
    pooled: torch.Tensor,
    batch: "_Batch",
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
) -> torch.Tensor:
  """Single-deck attention expressed with ONNX operators and dynamic lengths."""
  embedding_dim = attention.embed_dim
  head_count = attention.num_heads
  head_dim = embedding_dim // head_count
  projection_weight = attention.in_proj_weight
  projection_bias = attention.in_proj_bias
  if projection_weight is None:
    raise ValueError("ONNX export requires packed attention projection weights.")

  query_projection = F.linear(
    query,
    projection_weight[:embedding_dim],
    projection_bias[:embedding_dim] if projection_bias is not None else None,
  )
  key_projection = F.linear(
    key,
    projection_weight[embedding_dim:2 * embedding_dim],
    (
      projection_bias[embedding_dim:2 * embedding_dim]
      if projection_bias is not None
      else None
    ),
  )
  value_projection = F.linear(
    value,
    projection_weight[2 * embedding_dim:],
    projection_bias[2 * embedding_dim:] if projection_bias is not None else None,
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
    observation_coverage: torch.Tensor,
    observation_complete: torch.Tensor,
  ) -> None:
    self.card_idx = card_idx
    self.zone_idx = zone_idx
    self.quantity_idx = quantity_idx
    self.quantity_weight = quantity_weight
    self.deck_idx = deck_idx
    self.deck_count = deck_count
    self.target_idx = target_idx
    self.package_features = package_features
    self.observation_coverage = observation_coverage
    self.observation_complete = observation_complete


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
    observation_coverage: torch.Tensor,
    observation_complete: torch.Tensor,
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
    self.observation_coverage = observation_coverage
    self.observation_complete = observation_complete


def max_quantity(examples: list[ModelExample]) -> int:
  values = [
    token.quantity
    for example in examples
    for token in example.tokens
  ]
  return max(values, default=0)


def partial_observation_training_views(
  examples: list[ModelExample],
  *,
  rng: random.Random,
  identity_counts: tuple[int, ...] = DEFAULT_PARTIAL_OBSERVATION_IDENTITY_COUNTS,
  corruption_policy: str = PARTIAL_CORRUPTION_FIXED,
  card_information: dict[int, float] | None = None,
) -> list[ModelExample]:
  """Sample one token-identity subset per full example for paired training."""
  views: list[ModelExample] = []
  for example in examples:
    token_count = len(example.tokens)
    if token_count <= 1:
      views.append(example)
      continue
    if corruption_policy == PARTIAL_CORRUPTION_FIXED:
      observed_count = _partial_identity_count(
        token_count,
        identity_counts=identity_counts,
        rng=rng,
      )
      selected = set(rng.sample(range(token_count), observed_count))
    elif corruption_policy == PARTIAL_CORRUPTION_MIXTURE:
      mode = rng.random()
      if mode < 0.25:
        observed_count = max(
          1,
          min(
            token_count - 1,
            round(token_count * rng.uniform(0.5, 0.8)),
          ),
        )
        selected = set(rng.sample(range(token_count), observed_count))
      elif mode < 0.75:
        observed_count = _partial_identity_count(
          token_count,
          identity_counts=(5, 10),
          rng=rng,
        )
        selected = set(rng.sample(range(token_count), observed_count))
      else:
        observed_count = _partial_identity_count(
          token_count,
          identity_counts=identity_counts,
          rng=rng,
        )
        selected = _weighted_token_sample(
          example,
          observed_count=observed_count,
          card_information=card_information or {},
          rng=rng,
        )
    else:
      raise ValueError(f"Unsupported partial corruption policy: {corruption_policy}")
    expected_size = example.expected_mainboard_size or sum(
      max(token.quantity, 0)
      for token in example.tokens
    )
    views.append(
      replace(
        example,
        tokens=tuple(
          token
          for index, token in enumerate(example.tokens)
          if index in selected
        ),
        expected_mainboard_size=expected_size,
        observation_complete=False,
      )
    )
  return views


def card_identity_idf(examples: list[ModelExample]) -> dict[int, float]:
  deck_count = len(examples)
  support: dict[int, int] = {}
  for example in examples:
    for card_idx in {token.card_idx for token in example.tokens}:
      support[card_idx] = support.get(card_idx, 0) + 1
  return {
    card_idx: math.log((deck_count + 1.0) / (count + 1.0)) + 1.0
    for card_idx, count in support.items()
  }


def mixed_class_balanced_epoch_indexes(
  examples: list[ModelExample],
  *,
  rng: random.Random,
  balanced_fraction: float = DEFAULT_BALANCED_SAMPLING_FRACTION,
  max_multiplier: float = DEFAULT_BALANCED_SAMPLING_MAX_MULTIPLIER,
) -> list[int]:
  """Mix natural examples with capped square-root class-balanced resampling."""
  if not 0.0 <= balanced_fraction <= 1.0:
    raise ValueError("balanced_fraction must be in [0.0, 1.0].")
  if max_multiplier < 1.0:
    raise ValueError("max_multiplier must be at least 1.0.")
  if not examples:
    return []

  support_by_label: dict[str, int] = {}
  for example in examples:
    support_by_label[example.target_label_id] = (
      support_by_label.get(example.target_label_id, 0) + 1
    )
  max_support = max(support_by_label.values())
  weights = [
    min(
      max_multiplier,
      math.sqrt(max_support / support_by_label[example.target_label_id]),
    )
    for example in examples
  ]

  balanced_count = round(len(examples) * balanced_fraction)
  natural_count = len(examples) - balanced_count
  natural_indexes = list(range(len(examples)))
  rng.shuffle(natural_indexes)
  epoch_indexes = natural_indexes[:natural_count]
  epoch_indexes.extend(
    rng.choices(
      range(len(examples)),
      weights=weights,
      k=balanced_count,
    )
  )
  rng.shuffle(epoch_indexes)
  return epoch_indexes


def _partial_identity_count(
  token_count: int,
  *,
  identity_counts: tuple[int, ...],
  rng: random.Random,
) -> int:
  eligible_counts = [
    count
    for count in identity_counts
    if count < token_count
  ]
  return rng.choice(eligible_counts) if eligible_counts else token_count - 1


def _weighted_token_sample(
  example: ModelExample,
  *,
  observed_count: int,
  card_information: dict[int, float],
  rng: random.Random,
) -> set[int]:
  ranked = sorted(
    range(len(example.tokens)),
    key=lambda index: (
      rng.random() ** (
        1.0 / max(card_information.get(example.tokens[index].card_idx, 1.0), 1e-8)
      )
    ),
    reverse=True,
  )
  return set(ranked[:observed_count])


def partial_observation_views_at_count(
  examples: list[ModelExample],
  *,
  identity_count: int,
  seed: int,
) -> list[ModelExample]:
  """Create deterministic partial views for evaluation at one identity count."""
  if identity_count <= 0:
    raise ValueError("identity_count must be positive.")
  views: list[ModelExample] = []
  for example in examples:
    if len(example.tokens) <= identity_count:
      continue
    example_rng = random.Random(f"{seed}:{example.deck_id}:{identity_count}")
    selected = set(example_rng.sample(range(len(example.tokens)), identity_count))
    expected_size = example.expected_mainboard_size or sum(
      max(token.quantity, 0)
      for token in example.tokens
    )
    views.append(
      replace(
        example,
        tokens=tuple(
          token
          for index, token in enumerate(example.tokens)
          if index in selected
        ),
        expected_mainboard_size=expected_size,
        observation_complete=False,
      )
    )
  return views


def _validate_partial_observation_config(
  *,
  enabled: bool,
  identity_counts: tuple[int, ...],
  classification_weight: float,
  consistency_weight: float,
  min_coverage_weight: float,
  latent_weight: float = 0.0,
  contextual_weight: float = 0.0,
  teacher_decay: float = DEFAULT_PARTIAL_TEACHER_DECAY,
) -> None:
  if any(count <= 0 for count in identity_counts):
    raise ValueError("Partial-observation identity counts must be positive.")
  if enabled and not identity_counts:
    raise ValueError("Partial-observation training requires identity counts.")
  if classification_weight < 0.0:
    raise ValueError("partial_classification_weight must be non-negative.")
  if consistency_weight < 0.0:
    raise ValueError("partial_consistency_weight must be non-negative.")
  if min_coverage_weight < 0.0 or min_coverage_weight > 1.0:
    raise ValueError("partial_min_coverage_weight must be in [0.0, 1.0].")
  if latent_weight < 0.0:
    raise ValueError("partial_latent_weight must be non-negative.")
  if contextual_weight < 0.0:
    raise ValueError("partial_contextual_weight must be non-negative.")
  if teacher_decay < 0.0 or teacher_decay >= 1.0:
    raise ValueError("partial_teacher_decay must be in [0.0, 1.0).")


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


@torch.no_grad()
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  coverage_weights = (
    min_coverage_weight
    + (1.0 - min_coverage_weight) * coverage.clamp(0.0, 1.0)
  )
  weight_sum = coverage_weights.sum().clamp_min(1e-8)
  partial_ce_rows = F.cross_entropy(
    partial_logits,
    target_idx,
    label_smoothing=label_smoothing,
    reduction="none",
  )
  partial_ce = (partial_ce_rows * coverage_weights).sum() / weight_sum
  teacher_probabilities = F.softmax(full_logits.detach(), dim=1)
  consistency_rows = F.kl_div(
    F.log_softmax(partial_logits, dim=1),
    teacher_probabilities,
    reduction="none",
  ).sum(dim=1)
  consistency = (consistency_rows * coverage_weights).sum() / weight_sum
  return partial_ce, consistency, coverage.mean()


def _prepare_set_examples(
  examples: list[ModelExample],
  *,
  label_to_idx: dict[str, int],
  quantity_count: int,
  require_targets: bool = True,
  package_features: PackageFeatureSet | None = None,
  quantity_weighting: str = SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED,
  hypergeometric_draw_count: int = DEFAULT_HYPERGEOMETRIC_DRAW_COUNT,
) -> _PreparedSetExamples:
  card_idx: list[int] = []
  zone_idx: list[int] = []
  quantity_idx: list[int] = []
  quantity_weight: list[float] = []
  token_start: list[int] = []
  token_end: list[int] = []
  target_idx: list[int] = []
  package_idx_by_deck: list[tuple[int, ...]] = []
  observation_coverage: list[float] = []
  observation_complete: list[bool] = []

  for example in examples:
    if require_targets:
      target_idx.append(label_to_idx[example.target_label_id])
    else:
      target_idx.append(0)
    token_start.append(len(card_idx))
    observed_size = sum(max(token.quantity, 0) for token in example.tokens)
    population_size = example.expected_mainboard_size or observed_size
    observation_coverage.append(
      min(float(observed_size) / float(population_size), 1.0)
      if population_size > 0
      else 0.0
    )
    observation_complete.append(example.observation_complete)
    for token in example.tokens:
      card_idx.append(token.card_idx)
      zone_idx.append(token.zone_idx)
      quantity_idx.append(max(0, min(token.quantity, quantity_count - 1)))
      if quantity_weighting == SET_TRANSFORMER_POOLING_HYPERGEOMETRIC:
        quantity_weight.append(
          hypergeometric_quantity_weight(
            token.quantity,
            population_size=population_size,
            draw_count=hypergeometric_draw_count,
          )
        )
      else:
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
    observation_coverage=torch.tensor(
      observation_coverage,
      dtype=torch.float32,
    ),
    observation_complete=torch.tensor(
      observation_complete,
      dtype=torch.bool,
    ),
  )


def hypergeometric_quantity_weight(
  quantity: int,
  *,
  population_size: int,
  draw_count: int = DEFAULT_HYPERGEOMETRIC_DRAW_COUNT,
) -> float:
  """Probability of seeing a card, normalized so one copy has weight one."""
  quantity = max(0, min(int(quantity), int(population_size)))
  population_size = int(population_size)
  if quantity == 0 or population_size <= 0 or draw_count <= 0:
    return 0.0
  draws = min(int(draw_count), population_size)
  miss_count = population_size - quantity
  miss_probability = (
    math.comb(miss_count, draws) / math.comb(population_size, draws)
    if miss_count >= draws
    else 0.0
  )
  singleton_probability = draws / population_size
  return (1.0 - miss_probability) / singleton_probability


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
    observation_coverage=prepared.observation_coverage.index_select(
      0,
      deck_number_tensor,
    ).to(device),
    observation_complete=prepared.observation_complete.index_select(
      0,
      deck_number_tensor,
    ).to(device),
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
    observation_coverage=torch.tensor(
      [
        min(
          sum(max(token.quantity, 0) for token in example.tokens)
          / float(
            example.expected_mainboard_size
            or sum(max(token.quantity, 0) for token in example.tokens)
            or 1
          ),
          1.0,
        )
        for example in examples
      ],
      dtype=torch.float32,
      device=device,
    ),
    observation_complete=torch.tensor(
      [example.observation_complete for example in examples],
      dtype=torch.bool,
      device=device,
    ),
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

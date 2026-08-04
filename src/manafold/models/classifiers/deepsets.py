from __future__ import annotations

import random
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from manafold.models.classifiers.partial_view_learning import (
  _partial_observation_losses,
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
from manafold.datasets.model_inputs import DeckModelInput
from manafold.models.classifiers.config import (
  DEEPSETS_ARCHITECTURES,
  DEEPSETS_ARCHITECTURE_BASE,
  DEEPSETS_ARCHITECTURE_PLUSPLUS,
  DEFAULT_PARTIAL_CLASSIFICATION_WEIGHT,
  DEFAULT_PARTIAL_CONSISTENCY_WEIGHT,
  DEFAULT_PARTIAL_MIN_COVERAGE_WEIGHT,
  DEFAULT_PARTIAL_OBSERVATION_IDENTITY_COUNTS,
  POOLING_MODES,
  POOLING_SUM,
)
from manafold.models.classifiers.input_batches import (
  _PreparedSetExamples,
  _prepare_set_examples,
  _prepared_set_batch,
)
from manafold.models.classifiers.torch_classifier_networks import (
  _DeepSetsNetwork,
  _DeepSetsPlusPlusNetwork,
  _module_gradient_norm,
  _module_parameter_norm,
)
from manafold.models.features.card_packages import PackageFeatureSet
from manafold.models.classifiers.partial_decks import (
  _validate_partial_observation_config,
  partial_observation_training_views,
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
    self.package_count = (
      0 if package_features is None else len(package_features)
    )
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
    self.partial_observation_identity_counts = (
      partial_observation_identity_counts
    )
    self.partial_classification_weight = partial_classification_weight
    self.partial_consistency_weight = partial_consistency_weight
    self.partial_min_coverage_weight = partial_min_coverage_weight
    self.device = _resolve_device(device)
    self._label_to_idx = {
      label: index for index, label in enumerate(labels)
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
    self._causal_soft_target_matrix: torch.Tensor | None = None
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
        std=hidden_dim**-0.5,
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
      embedding_weight.index_copy_(
        0, card_idxs, init.index_select(0, card_idxs)
      )
      return int(card_idxs.numel())

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
            soft_target_matrix=self._causal_soft_target_matrix,
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
          total_partial_coverage += (
            float(partial_coverage.item()) * batch_count
          )
        total_correct += int(
          (logits.argmax(dim=1) == batch.target_idx).sum().item()
        )
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

      history.append({
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
      })

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
    if k <= 0 or not self.labels or not example.tokens:
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
    prepared = _prepare_set_examples(
      examples,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
      require_targets=False,
      package_features=self.package_features,
    )
    self._network.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(
        len(examples), batch_size
      ):
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
    examples: list[DeckModelInput],
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
      for batch_indexes in _sequential_index_batches(
        len(examples), batch_size
      ):
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
    examples: list[DeckModelInput],
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
        "reason": (
          "No labeled examples are available for package branch diagnostics."
        ),
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
        activation_count / activation_total if activation_total else None
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
      rows.append({
        "card_idx": card_idx,
        "oracle_id": card.get("oracle_id"),
        "primary_name": card.get("primary_name"),
        "embedding": weights[card_idx],
      })
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
    has_neighbors = torch.zeros(
      label_count, dtype=torch.bool, device=self.device
    )

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
        targets[label_idx, neighbor_idx] += (
          self.label_smoothing * weight / weight_sum
        )
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

  def set_causal_soft_targets(self, soft_matrix: torch.Tensor) -> None:
    """Sets continuous causal soft target matrix for Candidate C training."""
    self._causal_soft_target_matrix = soft_matrix.to(self.device)

  def _classification_loss(
    self,
    logits: torch.Tensor,
    target_idx: torch.Tensor,
  ) -> torch.Tensor:
    if getattr(self, "_causal_soft_target_matrix", None) is not None:
      target_distribution = self._causal_soft_target_matrix.index_select(
        0,
        target_idx,
      ).to(logits.device)
      log_probabilities = F.log_softmax(logits, dim=1)
      return -(target_distribution * log_probabilities).sum(dim=1).mean()
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
    prepared: _PreparedSetExamples,
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
      )
      .sum()
      .item()
    )
    return smoothed_count / len(epoch_indexes)

  def _metric_support_tensors(
    self,
    trainable: list[DeckModelInput],
  ) -> dict[str, torch.Tensor] | None:
    if self._metric_prototypes is None:
      return None
    support = torch.zeros(
      len(self.labels), dtype=torch.long, device=self.device
    )
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
    batch: _Batch,
    embeddings: torch.Tensor,
  ) -> torch.Tensor:
    if self._network.package_projection is None and self.extra_feature_dim == 0:
      return self._network.rho(embeddings)
    return self._network(batch)

  def deck_embedding_rows(
    self,
    examples: list[DeckModelInput],
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
      for batch_indexes in _sequential_index_batches(
        len(examples), batch_size
      ):
        batch = _prepared_set_batch(
          prepared,
          batch_indexes,
          device=self.device,
          package_scale=self.package_scale,
        )
        embeddings = self._network.encode(batch).detach().cpu().tolist()
        batch_examples = [examples[index] for index in batch_indexes]
        for example, embedding in zip(batch_examples, embeddings, strict=True):
          rows.append({
            "deck_id": example.deck_id,
            "split_name": example.split_name,
            "target_label_id": example.target_label_id,
            "embedding": embedding,
          })
    return rows

  def deck_embedding_tensor(
    self,
    examples: list[DeckModelInput],
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
      for batch_indexes in _sequential_index_batches(
        len(examples), batch_size
      ):
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
    vocab_by_idx = {int(card["card_idx"]): card for card in vocab}
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
      rows.append({
        "card_idx": card_idx,
        "primary_name": card.get("primary_name"),
        "train_support": (
          None if card_support is None else card_support.get(card_idx, 0)
        ),
        "neighbors": [
          {
            "card_idx": int(neighbor_idx),
            "primary_name": vocab_by_idx[int(neighbor_idx)].get(
              "primary_name"
            ),
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
      })
    return rows

  def _loss_accuracy(
    self,
    examples: list[DeckModelInput],
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
    prepared: _PreparedSetExamples,
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
        total_correct += int(
          (logits.argmax(dim=1) == batch.target_idx).sum().item()
        )
        total_count += batch_count

    return total_loss / total_count, total_correct / total_count

  def _state(self) -> dict[str, torch.Tensor]:
    return {
      key: value.detach().cpu().clone()
      for key, value in self._network.state_dict().items()
    }

  def _restore_state(self, state: dict[str, torch.Tensor]) -> None:
    self._network.load_state_dict({
      key: value.to(self.device) for key, value in state.items()
    })

  def state_dict_for_saving(self) -> dict[str, torch.Tensor]:
    return self._state()

  def load_saved_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    self._restore_state(state)

from __future__ import annotations

import contextlib
from copy import deepcopy
import random
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from manafold.models.classifiers.partial_view_learning import (
  _matching_full_token_indexes,
  _partial_contextual_prediction_loss,
  _partial_latent_prediction_loss,
  _partial_observation_losses,
  _update_ema_module,
)
from manafold.models.classifiers.prediction_statistics import (
  _card_nearest_neighbors,
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
  DEFAULT_BALANCED_SAMPLING_FRACTION,
  DEFAULT_BALANCED_SAMPLING_MAX_MULTIPLIER,
  DEFAULT_HYPERGEOMETRIC_DRAW_COUNT,
  DEFAULT_PARTIAL_CLASSIFICATION_WEIGHT,
  DEFAULT_PARTIAL_CONSISTENCY_WEIGHT,
  DEFAULT_PARTIAL_MIN_COVERAGE_WEIGHT,
  DEFAULT_PARTIAL_OBSERVATION_IDENTITY_COUNTS,
  DEFAULT_PARTIAL_TEACHER_DECAY,
  PARTIAL_CORRUPTION_FIXED,
  PARTIAL_CORRUPTION_MIXTURE,
  PARTIAL_CORRUPTION_POLICIES,
  SET_TRANSFORMER_POOLING_MODES,
  SET_TRANSFORMER_POOLING_PMA,
  TRAINING_SAMPLING_NATURAL,
  TRAINING_SAMPLING_NATURAL_SQRT_BALANCED,
  TRAINING_SAMPLING_POLICIES,
)
from manafold.models.classifiers.input_batches import (
  _Batch,
  _PreparedSetExamples,
  _prepare_set_examples,
  _prepared_set_batch,
)
from manafold.models.classifiers.torch_classifier_networks import (
  _PartialLatentPredictor,
  _SetTransformerNetwork,
)
from manafold.models.classifiers.partial_decks import (
  _validate_partial_observation_config,
  card_identity_idf,
  mixed_class_balanced_epoch_indexes,
  partial_observation_training_views,
)


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
    supcon_loss_weight: float = 0.0,
    anchor_gating: bool = False,
    use_product_manifold: bool = False,
    product_manifold_tree_weight: float = 0.10,
    causal_target_alpha: float = 0.0,
    adaptive_manifold_scaling: bool = False,
    adaptive_manifold_ramp_cards: float = 20.0,
    residual_anchor_logits: bool = False,
    residual_anchor_gamma: float = 0.25,
    family_by_label: dict[str, str] | None = None,
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
    if supcon_loss_weight < 0.0:
      raise ValueError("supcon_loss_weight must be non-negative.")
    if product_manifold_tree_weight < 0.0:
      raise ValueError("product_manifold_tree_weight must be non-negative.")
    if not 0.0 <= causal_target_alpha < 1.0:
      raise ValueError("causal_target_alpha must be in [0.0, 1.0).")
    if adaptive_manifold_ramp_cards <= 0.0:
      raise ValueError("adaptive_manifold_ramp_cards must be positive.")
    if residual_anchor_gamma < 0.0:
      raise ValueError("residual_anchor_gamma must be non-negative.")
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
    self.supcon_loss_weight = supcon_loss_weight
    self.anchor_gating = anchor_gating
    self.use_product_manifold = use_product_manifold
    self.product_manifold_tree_weight = product_manifold_tree_weight
    self.causal_target_alpha = causal_target_alpha
    self.adaptive_manifold_scaling = adaptive_manifold_scaling
    self.adaptive_manifold_ramp_cards = adaptive_manifold_ramp_cards
    self.residual_anchor_logits = residual_anchor_logits
    self.residual_anchor_gamma = residual_anchor_gamma
    self._causal_soft_target_matrix: torch.Tensor | None = None
    self._residual_anchor_pmi_matrix: torch.Tensor | None = None
    self.device = _resolve_device(device)
    self._label_to_idx = {label: index for index, label in enumerate(labels)}
    family_names = {
      label: (family_by_label or {}).get(label, label) for label in labels
    }
    family_index = {
      family: index
      for index, family in enumerate(sorted(set(family_names.values())))
    }
    self._family_index_by_label_index = torch.tensor(
      [family_index[family_names[label]] for label in labels],
      dtype=torch.long,
      device=self.device,
    )
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
      anchor_gating=anchor_gating,
      labels=labels,
      use_product_manifold=use_product_manifold,
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

    if self._network.anchor_gate is not None:
      self._network.anchor_gate.fit(trainable)
      if (
        self._partial_teacher is not None
        and self._partial_teacher.anchor_gate is not None
      ):
        self._partial_teacher.anchor_gate.fit(trainable)
    if self.causal_target_alpha > 0.0:
      from manafold.models.classifiers.causal_soft_targets import (
        CausalSoftTargetGenerator,
      )

      generator = CausalSoftTargetGenerator(
        num_classes=len(self.labels),
        alpha=self.causal_target_alpha,
        confusion_weight=0.0,
        jaccard_weight=1.0,
      )
      self._causal_soft_target_matrix = generator.fit_from_historical_data(
        [self._label_to_idx[example.target_label_id] for example in trainable],
        [
          list({token.card_idx for token in example.tokens})
          for example in trainable
        ],
      ).to(self.device)

    if self.residual_anchor_logits and self._network.anchor_gate is not None:
      pmi_matrix = self._network.anchor_gate._compute_pmi_matrix(
        self.card_count,
        self.labels,
        trainable,
        alpha=1.0,
      )
      self._residual_anchor_pmi_matrix = torch.as_tensor(
        pmi_matrix,
        dtype=torch.float32,
        device=self.device,
      )

    trainable_prepared = _prepare_set_examples(
      trainable,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
      quantity_weighting=self.pooling,
      hypergeometric_draw_count=self.hypergeometric_draw_count,
    ).to(self.device)
    validation_prepared = _prepare_set_examples(
      validation,
      label_to_idx=self._label_to_idx,
      quantity_count=self.quantity_count,
      quantity_weighting=self.pooling,
      hypergeometric_draw_count=self.hypergeometric_draw_count,
    ).to(self.device)
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
        ).to(self.device)

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
      total_supcon_loss = 0.0
      total_tree_loss = 0.0
      total_partial_coverage = 0.0
      total_correct = 0
      total_count = 0
      amp_context = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if self.device.type == "cuda"
        else contextlib.nullcontext()
      )
      for batch_indexes in _index_batches(epoch_indexes, batch_size):
        if max_steps is not None and optimizer_steps >= max_steps:
          break
        batch = _prepared_set_batch(
          trainable_prepared,
          batch_indexes,
          device=self.device,
        )
        optimizer.zero_grad(set_to_none=True)
        with amp_context:
          full_embedding = self._network.encode(batch)
          logits = self._network.rho(full_embedding)
        if (
          self.residual_anchor_logits
          and self._residual_anchor_pmi_matrix is not None
        ):
          logits = logits + self._residual_logits_for_batch(batch)
        loss = self._classification_loss(logits, batch.target_idx)
        full_ce_loss = loss
        objective_weight = 1.0
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
            partial_embedding = self._network.encode(partial_batch)
            partial_contextual_tokens = (
              self._network.encode_contextual_tokens(
                partial_batch,
                use_anchor_gating=False,
              )
            )
          else:
            partial_embedding = self._network.encode(partial_batch)
            partial_contextual_tokens = None
          partial_logits = self._network.rho(partial_embedding)
          if (
            self.residual_anchor_logits
            and self._residual_anchor_pmi_matrix is not None
          ):
            partial_logits = (
              partial_logits
              + self._residual_logits_for_batch(partial_batch)
            )
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
          loss = loss + self.partial_classification_weight * partial_ce_loss
          objective_weight += self.partial_classification_weight
          if self.partial_consistency_weight > 0.0:
            loss = loss + (
              self.partial_consistency_weight * partial_consistency_loss
            )
            objective_weight += self.partial_consistency_weight
          if self.partial_latent_weight > 0.0:
            if (
              self._partial_predictor is None
              or self._partial_teacher is None
            ):
              raise AssertionError(
                "Partial latent modules are not initialized."
              )
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
              raise AssertionError(
                "Partial contextual modules are not initialized."
              )
            with torch.no_grad():
              teacher_contextual_tokens = (
                self._partial_teacher.encode_contextual_tokens(
                  batch,
                  use_anchor_gating=False,
                )
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
              predicted=self._partial_predictor(
                partial_contextual_tokens
              ),
              target=contextual_targets,
              deck_idx=partial_batch.deck_idx,
              deck_count=partial_batch.deck_count,
            )
            loss = loss + (
              self.partial_contextual_weight * partial_contextual_loss
            )
            objective_weight += self.partial_contextual_weight

        supcon_loss = None
        if self.supcon_loss_weight > 0.0:
          if not hasattr(self, "_supcon_loss_fn"):
            from manafold.models.classifiers.supervised_contrastive_loss import (
              SupConLoss,
            )

            self._supcon_loss_fn = SupConLoss(temperature=0.07).to(
              self.device
            )
          contrastive_targets = (
            self._family_index_by_label_index.index_select(
              0,
              batch.target_idx,
            )
            if self.use_product_manifold
            else batch.target_idx
          )
          supcon_loss = self._supcon_loss_fn(
            full_embedding,
            contrastive_targets,
            partial_features=(
              partial_embedding if partial_prepared is not None else None
            ),
          )
          loss = loss + self.supcon_loss_weight * supcon_loss
          objective_weight += self.supcon_loss_weight

        tree_loss = None
        if (
          self.product_manifold_tree_weight > 0.0
          and self._network.product_manifold is not None
        ):
          family_targets = self._family_index_by_label_index.index_select(
            0,
            batch.target_idx,
          )
          tree_loss = self._network.product_manifold.compute_tree_loss(
            self._network.hyperbolic_embedding(full_embedding),
            batch.target_idx,
            family_targets,
          )
          effective_tree_weight = self.product_manifold_tree_weight
          if self.adaptive_manifold_scaling:
            token_counts = torch.bincount(
              batch.deck_idx, minlength=batch.deck_count
            )
            avg_set_size = token_counts.float().mean().item()
            scale_factor = min(
              1.0, avg_set_size / self.adaptive_manifold_ramp_cards
            )
            effective_tree_weight = (
              self.product_manifold_tree_weight * scale_factor
            )
          loss = loss + effective_tree_weight * tree_loss
          objective_weight += effective_tree_weight

        loss = loss / objective_weight

        loss.backward()
        nonfinite_gradients = [
          name
          for name, parameter in self._network.named_parameters()
          if parameter.grad is not None
          and not torch.isfinite(parameter.grad).all()
        ]
        if self._partial_predictor is not None:
          nonfinite_gradients.extend(
            f"partial_predictor.{name}"
            for name, parameter in self._partial_predictor.named_parameters()
            if parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
          )
        if nonfinite_gradients:
          raise FloatingPointError(
            "Non-finite A8 training gradients: "
            + ", ".join(nonfinite_gradients[:8])
          )
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
        if supcon_loss is not None:
          total_supcon_loss += float(supcon_loss.item()) * batch_count
        if tree_loss is not None:
          total_tree_loss += float(tree_loss.item()) * batch_count
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
          total_partial_coverage += (
            float(partial_coverage.item()) * batch_count
          )
        total_correct += int(
          (logits.argmax(dim=1) == batch.target_idx).sum().item()
        )
        total_count += batch_count

      train_loss = total_loss / total_count if total_count else None
      train_full_ce_loss = (
        total_full_ce_loss / total_count if total_count else None
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
      train_supcon_loss = (
        total_supcon_loss / total_count
        if self.supcon_loss_weight > 0.0 and total_count
        else None
      )
      train_tree_loss = (
        total_tree_loss / total_count
        if self.product_manifold_tree_weight > 0.0 and total_count
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
          "train_supcon_loss": train_supcon_loss,
          "train_tree_loss": train_tree_loss,
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
      "a8_objectives": self.a8_objective_config(),
    }

  def _classification_loss(
    self,
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    *,
    reduction: str = "mean",
  ) -> torch.Tensor:
    if self._causal_soft_target_matrix is None:
      return F.cross_entropy(
        logits,
        target_idx,
        label_smoothing=self.label_smoothing,
        reduction=reduction,
      )
    targets = self._causal_soft_target_matrix.index_select(0, target_idx)
    rows = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)
    if reduction == "none":
      return rows
    if reduction == "sum":
      return rows.sum()
    return rows.mean()

  def a8_objective_config(self) -> dict[str, Any]:
    product_manifold = self._network.product_manifold
    return {
      "anchor_gating": self.anchor_gating,
      "anchor_source": "train_split_positive_pmi",
      "product_manifold": self.use_product_manifold,
      "product_space": (
        {
          "euclidean_dim": product_manifold.euc_dim,
          "hyperbolic_dim": product_manifold.hyp_dim,
          "curvature": product_manifold.curvature,
          "hierarchy_source": "deterministic_family_relaxation",
        }
        if product_manifold is not None
        else None
      ),
      "hyperbolic_family_weight": self.product_manifold_tree_weight,
      "causal_target_alpha": self.causal_target_alpha,
      "causal_target_source": "train_split_class_card_jaccard",
      "supcon_weight": self.supcon_loss_weight,
      "supcon_target": (
        "relaxed_family" if self.use_product_manifold else "source_label"
      ),
      "family_count": len(set(self._family_index_by_label_index.tolist())),
      "adaptive_manifold_scaling": self.adaptive_manifold_scaling,
      "adaptive_manifold_ramp_cards": self.adaptive_manifold_ramp_cards,
      "residual_anchor_logits": self.residual_anchor_logits,
      "residual_anchor_gamma": self.residual_anchor_gamma,
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

  def _residual_logits_for_batch(
    self,
    batch: "_Batch",
  ) -> torch.Tensor:
    """Compute residual anchor logits from PMI matrix for a batch.

    For each deck, averages the PMI row vectors of all observed card indices
    and scales by gamma. Returns a (deck_count, label_count) tensor that can
    be added directly to neural logits.
    """
    if self._residual_anchor_pmi_matrix is None:
      raise AssertionError("Residual anchor PMI matrix is not initialized.")
    card_pmi = self._residual_anchor_pmi_matrix[
      batch.card_idx.clamp(0, self.card_count - 1)
    ]
    deck_pmi_sums = torch.zeros(
      batch.deck_count,
      card_pmi.shape[1],
      device=card_pmi.device,
      dtype=card_pmi.dtype,
    )
    deck_pmi_sums.scatter_add_(
      0,
      batch.deck_idx.unsqueeze(1).expand_as(card_pmi),
      card_pmi,
    )
    token_counts = (
      torch.bincount(batch.deck_idx, minlength=batch.deck_count)
      .float()
      .clamp(min=1.0)
    )
    deck_pmi_means = deck_pmi_sums / token_counts.unsqueeze(1)
    return self.residual_anchor_gamma * deck_pmi_means

  def inference_parameter_count(self) -> int:
    return sum(parameter.numel() for parameter in self._network.parameters())

  def training_parameter_count(self) -> int:
    predictor_count = (
      sum(parameter.numel() for parameter in self._partial_predictor.parameters())
      if self._partial_predictor is not None
      else 0
    )
    return self.inference_parameter_count() + predictor_count

  def state_dict_for_saving(self) -> dict[str, torch.Tensor]:
    return self._state()

  def load_saved_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    self._restore_state(state)

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
        if (
          self.residual_anchor_logits
          and self._residual_anchor_pmi_matrix is not None
        ):
          logits = logits + self._residual_logits_for_batch(batch)
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
        logits = self._network(batch)
        if (
          self.residual_anchor_logits
          and self._residual_anchor_pmi_matrix is not None
        ):
          logits = logits + self._residual_logits_for_batch(batch)
        rows.append(logits.detach().cpu())
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
        batch_examples = [examples[index] for index in batch_indexes]
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
    examples: list[DeckModelInput],
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
    examples: list[DeckModelInput],
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
        if (
          self.residual_anchor_logits
          and self._residual_anchor_pmi_matrix is not None
        ):
          logits = logits + self._residual_logits_for_batch(batch)
        loss = self._classification_loss(logits, batch.target_idx)
        batch_count = len(batch_indexes)
        total_loss += float(loss.item()) * batch_count
        total_correct += int(
          (logits.argmax(dim=1) == batch.target_idx).sum().item()
        )
        total_count += batch_count

    return total_loss / total_count, total_correct / total_count

  def _state(self) -> dict[str, torch.Tensor]:
    state = {
      key: value.detach().cpu().clone()
      for key, value in self._network.state_dict().items()
    }
    if self._residual_anchor_pmi_matrix is not None:
      state["_residual_anchor_pmi_matrix"] = (
        self._residual_anchor_pmi_matrix.detach().cpu().clone()
      )
    return state

  def _restore_state(self, state: dict[str, torch.Tensor]) -> None:
    if "_residual_anchor_pmi_matrix" in state:
      self._residual_anchor_pmi_matrix = state["_residual_anchor_pmi_matrix"].to(
        self.device
      )
      net_state = {
        k: v for k, v in state.items() if k != "_residual_anchor_pmi_matrix"
      }
    else:
      net_state = state
    self._network.load_state_dict(
      {key: value.to(self.device) for key, value in net_state.items()}
    )

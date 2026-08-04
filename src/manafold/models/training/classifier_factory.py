from __future__ import annotations

import math
from typing import Any

from manafold.models.classifiers import (
  DEEPSETS_ARCHITECTURE_PLUSPLUS,
  PARTIAL_CORRUPTION_FIXED,
  PARTIAL_CORRUPTION_MIXTURE,
  POOLING_QUANTITY_WEIGHTED,
  TRAINING_SAMPLING_NATURAL,
  TRAINING_SAMPLING_NATURAL_SQRT_BALANCED,
  Classifier,
  DeepSetsClassifier,
  PooledLinearClassifier,
  SetTransformerClassifier,
)
from manafold.taxonomy.family_backoff import build_family_mapping, display_label_from_id
from manafold.models.training.model_names import (
  DEFAULT_DEEPSETS_PLUSPLUS_DROPOUT,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_CONSTANT_FEATURES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2_REGULARIZED,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES_V1,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_SHUFFLED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_SUPPORT_MATCHED_UNSCORED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_WIDE_RHO,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_EXTRA_DIMS_PRESERVE_INIT,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES,
  MODEL_POOLED_LINEAR,
  MODEL_POOLED_LINEAR_PACKAGES,
  MODEL_POOLED_LINEAR_PACKAGE_ONLY,
  MODEL_SET_TRANSFORMER_A10,
  MODEL_SET_TRANSFORMER_A11,
  MODEL_SET_TRANSFORMER_A6,
  MODEL_SET_TRANSFORMER_A8,
  MODEL_SET_TRANSFORMER_A9,
  MODEL_SET_TRANSFORMER_PARTIAL_BALANCED,
  _DEEPSETS_HEAD_ATTRIBUTION_MODELS,
  _DEEPSETS_PLUSPLUS_LABEL_SMOOTHING,
  _MODEL_POOLING,
  _SET_TRANSFORMER_POOLING,
)
from manafold.models.features.card_packages import PackageFeatureSet


def _package_control_for_model(model_name: str) -> str | None:
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES:
    return "v2_package_identity"
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES_V1:
    return "v1_package_identity"
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_SUPPORT_MATCHED_UNSCORED_PACKAGES:
    return "support_matched_unscored_identity"
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_SHUFFLED_PACKAGES:
    return "shuffled_package_activations"
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES:
    return "true_zero_package_activations"
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO:
    return "capacity_matched_no_package_input"
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES:
    return "synthetic_activation_rates"
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES:
    return "label_shuffled_mined_identity"
  return None


def _deepsets_head_control_config(
  model_name: str,
  *,
  hidden_dim: int,
  label_count: int,
  architecture_package_count: int,
  package_projection_dim: int,
  head_v2_rho_hidden_dim: int,
) -> dict[str, Any]:
  if model_name in (
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2_REGULARIZED,
  ):
    return {
      "head_control": "head_v2_wide_rho",
      "rho_hidden_dim": head_v2_rho_hidden_dim,
      "extra_feature_dim": 0,
      "extra_feature_value": 0.0,
      "preserve_base_rho_init": False,
    }
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_WIDE_RHO:
    return {
      "head_control": "wide_rho",
      "rho_hidden_dim": _matched_wide_rho_hidden_dim(
        hidden_dim=hidden_dim,
        label_count=label_count,
        architecture_package_count=architecture_package_count,
        package_projection_dim=package_projection_dim,
      ),
      "extra_feature_dim": 0,
      "extra_feature_value": 0.0,
      "preserve_base_rho_init": False,
    }
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_CONSTANT_FEATURES:
    return {
      "head_control": "extra_constant_features",
      "rho_hidden_dim": hidden_dim,
      "extra_feature_dim": package_projection_dim,
      "extra_feature_value": 1.0,
      "preserve_base_rho_init": False,
    }
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_EXTRA_DIMS_PRESERVE_INIT:
    return {
      "head_control": "zero_extra_dims_preserve_init",
      "rho_hidden_dim": hidden_dim,
      "extra_feature_dim": package_projection_dim,
      "extra_feature_value": 0.0,
      "preserve_base_rho_init": True,
    }
  raise AssertionError(f"Unhandled Deep Sets head control: {model_name}")


def _matched_wide_rho_hidden_dim(
  *,
  hidden_dim: int,
  label_count: int,
  architecture_package_count: int,
  package_projection_dim: int,
) -> int:
  if package_projection_dim <= 0 or architecture_package_count <= 0:
    return hidden_dim
  package_extra_parameters = (
    architecture_package_count * package_projection_dim
    + package_projection_dim
    + package_projection_dim * hidden_dim
  )
  parameters_per_extra_rho_unit = hidden_dim + label_count + 1
  extra_units = math.ceil(package_extra_parameters / parameters_per_extra_rho_unit)
  return hidden_dim + max(1, extra_units)


def create_classifier(
  model_name: str,
  *,
  labels: tuple[str, ...],
  card_count: int,
  zone_count: int,
  quantity_count: int,
  package_features: PackageFeatureSet,
  package_feature_set_name: str | None,
  learning_rate: float,
  weight_decay: float,
  seed: int,
  embedding_dim: int,
  hidden_dim: int,
  head_v2_rho_hidden_dim: int,
  attention_heads: int,
  attention_layers: int,
  batch_size: int,
  shuffle: bool,
  device: str,
  package_scale: float,
  package_projection_dim: int,
  architecture_package_count: int,
) -> tuple[Classifier, dict[str, Any]]:
  if model_name == MODEL_POOLED_LINEAR:
    classifier = PooledLinearClassifier(
      labels=labels,
      card_count=card_count,
      zone_count=zone_count,
      learning_rate=learning_rate,
      weight_decay=weight_decay,
      seed=seed,
      device=device,
    )
    return (
      classifier,
      {
        "feature_count": classifier.feature_count,
        "card_feature_count": classifier.card_feature_count,
        "package_count": classifier.package_count,
        "package_only": classifier.package_only,
        "package_scale": classifier.package_scale,
        "weight_decay": classifier.weight_decay,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "device": str(classifier.device),
      },
    )
  if model_name == MODEL_POOLED_LINEAR_PACKAGE_ONLY:
    classifier = PooledLinearClassifier(
      labels=labels,
      card_count=card_count,
      zone_count=zone_count,
      learning_rate=learning_rate,
      weight_decay=weight_decay,
      seed=seed,
      device=device,
      package_features=package_features,
      package_only=True,
      package_scale=package_scale,
    )
    return (
      classifier,
      {
        "feature_count": classifier.feature_count,
        "card_feature_count": classifier.card_feature_count,
        "base_feature_count": classifier.base_feature_count,
        "package_count": classifier.package_count,
        "package_only": classifier.package_only,
        "package_feature_set": package_feature_set_name,
        "package_scale": classifier.package_scale,
        "weight_decay": classifier.weight_decay,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "device": str(classifier.device),
      },
    )
  if model_name == MODEL_POOLED_LINEAR_PACKAGES:
    classifier = PooledLinearClassifier(
      labels=labels,
      card_count=card_count,
      zone_count=zone_count,
      learning_rate=learning_rate,
      weight_decay=weight_decay,
      seed=seed,
      device=device,
      package_features=package_features,
      package_scale=package_scale,
    )
    return (
      classifier,
      {
        "feature_count": classifier.feature_count,
        "card_feature_count": classifier.card_feature_count,
        "base_feature_count": classifier.base_feature_count,
        "package_count": classifier.package_count,
        "package_only": classifier.package_only,
        "package_feature_set": package_feature_set_name,
        "package_scale": classifier.package_scale,
        "weight_decay": classifier.weight_decay,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "device": str(classifier.device),
      },
    )
  if model_name in _DEEPSETS_HEAD_ATTRIBUTION_MODELS:
    control = _deepsets_head_control_config(
      model_name,
      hidden_dim=hidden_dim,
      label_count=len(labels),
      architecture_package_count=architecture_package_count,
      package_projection_dim=package_projection_dim,
      head_v2_rho_hidden_dim=head_v2_rho_hidden_dim,
    )
    classifier = DeepSetsClassifier(
      labels=labels,
      card_count=card_count,
      zone_count=zone_count,
      quantity_count=quantity_count,
      embedding_dim=embedding_dim,
      hidden_dim=hidden_dim,
      pooling=POOLING_QUANTITY_WEIGHTED,
      learning_rate=learning_rate,
      weight_decay=weight_decay,
      seed=seed,
      device=device,
      rho_hidden_dim=control["rho_hidden_dim"],
      extra_feature_dim=control["extra_feature_dim"],
      extra_feature_value=control["extra_feature_value"],
      preserve_base_rho_init=control["preserve_base_rho_init"],
    )
    metadata = {
      "embedding_dim": embedding_dim,
      "hidden_dim": hidden_dim,
      "head": "softmax",
      "head_control": control["head_control"],
      "pooling": POOLING_QUANTITY_WEIGHTED,
      "quantity_count": quantity_count,
      "package_count": classifier.package_count,
      "package_scale": classifier.package_scale,
      "rho_hidden_dim": classifier.rho_hidden_dim,
      "extra_feature_dim": classifier.extra_feature_dim,
      "extra_feature_value": classifier.extra_feature_value,
      "preserve_base_rho_init": classifier.preserve_base_rho_init,
      "parameter_count": classifier.parameter_count(),
      "weight_decay": classifier.weight_decay,
      "batch_size": batch_size,
      "shuffle": shuffle,
      "device": str(classifier.device),
      "model_config": {
        "head": "wide_rho",
        "rho_hidden_dim": classifier.rho_hidden_dim,
        "weight_decay": classifier.weight_decay,
        "pooling": POOLING_QUANTITY_WEIGHTED,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
      },
    }
    if model_name in (
      MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2,
      MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2_REGULARIZED,
    ):
      metadata["head_v2_rho_hidden_dim"] = head_v2_rho_hidden_dim
    else:
      metadata["architecture_reference_package_count"] = (
        architecture_package_count
      )
      metadata["architecture_reference_projection_dim"] = package_projection_dim
    return (classifier, metadata)
  if model_name in _DEEPSETS_PLUSPLUS_LABEL_SMOOTHING:
    classifier = DeepSetsClassifier(
      labels=labels,
      card_count=card_count,
      zone_count=zone_count,
      quantity_count=quantity_count,
      embedding_dim=embedding_dim,
      hidden_dim=hidden_dim,
      pooling=POOLING_QUANTITY_WEIGHTED,
      learning_rate=learning_rate,
      weight_decay=weight_decay,
      seed=seed,
      device=device,
      architecture=DEEPSETS_ARCHITECTURE_PLUSPLUS,
      dropout=DEFAULT_DEEPSETS_PLUSPLUS_DROPOUT,
      label_smoothing=_DEEPSETS_PLUSPLUS_LABEL_SMOOTHING[model_name],
    )
    return (
      classifier,
      {
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "head": "softmax",
        "architecture": classifier.architecture,
        "dropout": classifier.dropout,
        "label_smoothing": classifier.label_smoothing,
        "pooling": POOLING_QUANTITY_WEIGHTED,
        "quantity_count": quantity_count,
        "package_count": classifier.package_count,
        "rho_hidden_dim": classifier.rho_hidden_dim,
        "parameter_count": classifier.parameter_count(),
        "weight_decay": classifier.weight_decay,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "device": str(classifier.device),
        "model_config": {
          "head": "softmax",
          "architecture": classifier.architecture,
          "dropout": classifier.dropout,
          "label_smoothing": classifier.label_smoothing,
          "rho_hidden_dim": classifier.rho_hidden_dim,
          "weight_decay": classifier.weight_decay,
          "pooling": POOLING_QUANTITY_WEIGHTED,
          "embedding_dim": embedding_dim,
          "hidden_dim": hidden_dim,
        },
      },
    )
  if model_name in (
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES_V1,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_SUPPORT_MATCHED_UNSCORED_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_SHUFFLED_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES,
  ):
    package_projection_bias = (
      model_name != MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES
    )
    classifier = DeepSetsClassifier(
      labels=labels,
      card_count=card_count,
      zone_count=zone_count,
      quantity_count=quantity_count,
      embedding_dim=embedding_dim,
      hidden_dim=hidden_dim,
      pooling=POOLING_QUANTITY_WEIGHTED,
      learning_rate=learning_rate,
      weight_decay=weight_decay,
      seed=seed,
      device=device,
      package_features=package_features,
      package_projection_dim=package_projection_dim,
      package_projection_bias=package_projection_bias,
      package_scale=package_scale,
    )
    return (
      classifier,
      {
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "head": "softmax",
        "pooling": POOLING_QUANTITY_WEIGHTED,
        "quantity_count": quantity_count,
        "package_count": classifier.package_count,
        "package_feature_set": package_feature_set_name,
        "package_projection_dim": classifier.package_projection_dim,
        "package_projection_bias": classifier.package_projection_bias,
        "package_control": _package_control_for_model(model_name),
        "rho_hidden_dim": classifier.rho_hidden_dim,
        "extra_feature_dim": classifier.extra_feature_dim,
        "extra_feature_value": classifier.extra_feature_value,
        "preserve_base_rho_init": classifier.preserve_base_rho_init,
        "parameter_count": classifier.parameter_count(),
        "package_scale": classifier.package_scale,
        "weight_decay": classifier.weight_decay,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "device": str(classifier.device),
        "model_config": {
          "head": "softmax",
          "rho_hidden_dim": classifier.rho_hidden_dim,
          "weight_decay": classifier.weight_decay,
          "pooling": POOLING_QUANTITY_WEIGHTED,
          "embedding_dim": embedding_dim,
          "hidden_dim": hidden_dim,
        },
      },
    )
  if model_name in _MODEL_POOLING:
    pooling = _MODEL_POOLING[model_name]
    classifier = DeepSetsClassifier(
      labels=labels,
      card_count=card_count,
      zone_count=zone_count,
      quantity_count=quantity_count,
      embedding_dim=embedding_dim,
      hidden_dim=hidden_dim,
      pooling=pooling,
      learning_rate=learning_rate,
      weight_decay=weight_decay,
      seed=seed,
      device=device,
    )
    return (
      classifier,
      {
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "head": "softmax",
        "pooling": pooling,
        "quantity_count": quantity_count,
        "package_count": classifier.package_count,
        "rho_hidden_dim": classifier.rho_hidden_dim,
        "extra_feature_dim": classifier.extra_feature_dim,
        "extra_feature_value": classifier.extra_feature_value,
        "preserve_base_rho_init": classifier.preserve_base_rho_init,
        "parameter_count": classifier.parameter_count(),
        "package_scale": classifier.package_scale,
        "weight_decay": classifier.weight_decay,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "device": str(classifier.device),
        "model_config": {
          "head": "softmax",
          "rho_hidden_dim": classifier.rho_hidden_dim,
          "weight_decay": classifier.weight_decay,
          "pooling": pooling,
          "embedding_dim": embedding_dim,
          "hidden_dim": hidden_dim,
        },
      },
    )
  if model_name in _SET_TRANSFORMER_POOLING:
    pooling = _SET_TRANSFORMER_POOLING[model_name]
    selected_partial_model = model_name in (
      MODEL_SET_TRANSFORMER_PARTIAL_BALANCED,
      MODEL_SET_TRANSFORMER_A6,
      MODEL_SET_TRANSFORMER_A8,
      MODEL_SET_TRANSFORMER_A9,
      MODEL_SET_TRANSFORMER_A10,
      MODEL_SET_TRANSFORMER_A11,
    )
    is_a6 = model_name == MODEL_SET_TRANSFORMER_A6
    is_a8 = model_name == MODEL_SET_TRANSFORMER_A8
    is_a9 = model_name == MODEL_SET_TRANSFORMER_A9
    is_a10 = model_name == MODEL_SET_TRANSFORMER_A10
    is_a11 = model_name == MODEL_SET_TRANSFORMER_A11
    is_manifold_model = is_a8 or is_a9
    is_anchor_model = is_manifold_model or is_a10 or is_a11
    family_by_label = build_family_mapping(
      {
        label: {
          "label_id": label,
          "display_label": display_label_from_id(label),
        }
        for label in labels
      }
    )
    classifier = SetTransformerClassifier(
      labels=labels,
      card_count=card_count,
      zone_count=zone_count,
      quantity_count=quantity_count,
      embedding_dim=embedding_dim,
      hidden_dim=hidden_dim,
      attention_heads=attention_heads,
      attention_layers=attention_layers,
      pooling=pooling,
      learning_rate=learning_rate,
      weight_decay=weight_decay,
      label_smoothing=(
        0.0 if is_manifold_model else 0.10 if selected_partial_model else 0.0
      ),
      partial_observation_training=selected_partial_model,
      partial_consistency_weight=0.5 if selected_partial_model else 0.0,
      partial_contextual_weight=0.5 if selected_partial_model else 0.0,
      partial_corruption_policy=(
        PARTIAL_CORRUPTION_MIXTURE
        if selected_partial_model
        else PARTIAL_CORRUPTION_FIXED
      ),
      training_sampling_policy=(
        TRAINING_SAMPLING_NATURAL_SQRT_BALANCED
        if selected_partial_model
        else TRAINING_SAMPLING_NATURAL
      ),
      supcon_loss_weight=0.25 if is_a6 else (0.05 if is_manifold_model else 0.0),
      anchor_gating=is_anchor_model,
      use_product_manifold=is_manifold_model,
      product_manifold_tree_weight=0.10 if is_manifold_model else 0.0,
      causal_target_alpha=0.10 if is_manifold_model else 0.0,
      adaptive_manifold_scaling=is_a9,
      residual_anchor_logits=is_a9 or is_a10 or is_a11,
      family_by_label=family_by_label if is_manifold_model else None,
      seed=seed,
      device=device,
    )
    return (
      classifier,
      {
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "attention_heads": attention_heads,
        "attention_layers": attention_layers,
        "pooling": pooling,
        "hypergeometric_draw_count": classifier.hypergeometric_draw_count,
        "label_smoothing": classifier.label_smoothing,
        "weight_decay": classifier.weight_decay,
        "partial_observation": classifier.partial_observation_config(),
        "token_scope": "mainboard" if selected_partial_model else "all",
        "parameter_count": classifier.inference_parameter_count(),
        "inference_parameter_count": classifier.inference_parameter_count(),
        "training_parameter_count": classifier.training_parameter_count(),
        "a8_objectives": classifier.a8_objective_config(),
        "quantity_count": quantity_count,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "device": str(classifier.device),
        "model_config": {
          "pooling": pooling,
          "hypergeometric_draw_count": classifier.hypergeometric_draw_count,
          "label_smoothing": classifier.label_smoothing,
          "weight_decay": classifier.weight_decay,
          "partial_observation": classifier.partial_observation_config(),
          "token_scope": "mainboard" if selected_partial_model else "all",
          "inference_parameter_count": classifier.inference_parameter_count(),
          "training_parameter_count": classifier.training_parameter_count(),
          "a8_objectives": classifier.a8_objective_config(),
          "embedding_dim": embedding_dim,
          "hidden_dim": hidden_dim,
          "attention_heads": attention_heads,
          "attention_layers": attention_layers,
        },
      },
    )
  raise AssertionError(f"Unhandled model: {model_name}")

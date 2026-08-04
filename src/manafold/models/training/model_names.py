"""Model name constants and aliases.

This module defines all model identifiers used throughout the training
pipeline, CLI, and experiment tracking. Model names follow a consistent
naming convention: <architecture>-<variant>-<key_features>.
"""

from manafold.models.classifiers.config import (
  POOLING_MEAN,
  POOLING_QUANTITY_WEIGHTED,
  POOLING_SUM,
  SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
  SET_TRANSFORMER_POOLING_PMA,
  SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED,
)


# Pooled Linear models
MODEL_POOLED_LINEAR = "pooled-linear"
"""Simple linear classifier on pooled card embeddings."""

MODEL_POOLED_LINEAR_PACKAGE_ONLY = "pooled-linear-package-only"
"""Pooled linear using only package features (no card embeddings)."""

MODEL_POOLED_LINEAR_PACKAGES = "pooled-linear-packages"
"""Pooled linear with both card embeddings and package features."""


# Deep Sets base models
MODEL_DEEPSETS = "deepsets"
"""Base Deep Sets model (uses default pooling)."""

MODEL_DEEPSETS_SUM = "deepsets-sum"
"""Deep Sets with sum pooling."""

MODEL_DEEPSETS_MEAN = "deepsets-mean"
"""Deep Sets with mean pooling."""

MODEL_DEEPSETS_QUANTITY_WEIGHTED = "deepsets-quantity-weighted"
"""Deep Sets with quantity-weighted pooling (default for Deep Sets)."""


# Deep Sets regularized models
MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED = "deepsets-quantity-weighted-regularized"
"""Deep Sets quantity-weighted with weight decay regularization.

This is the recommended baseline Deep Sets model (A1.5).
"""

MODEL_DEEPSETS_PLUSPLUS_REGULARIZED = "deepsets-plusplus-regularized"
"""Deep Sets++ with regularization (residual connections, layer norm)."""

MODEL_DEEPSETS_PLUSPLUS_REGULARIZED_LABEL_SMOOTHING_005 = (
  "deepsets-plusplus-regularized-label-smoothing-005"
)
"""Deep Sets++ regularized with 0.05 label smoothing."""

MODEL_DEEPSETS_PLUSPLUS_REGULARIZED_LABEL_SMOOTHING_010 = (
  "deepsets-plusplus-regularized-label-smoothing-010"
)
"""Deep Sets++ regularized with 0.10 label smoothing (A2++)."""


# Deep Sets experimental variants
MODEL_DEEPSETS_QUANTITY_WEIGHTED_WIDE_RHO = "deepsets-quantity-weighted-wide-rho"
"""Deep Sets with wider rho projection head."""

MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_CONSTANT_FEATURES = (
  "deepsets-quantity-weighted-extra-constant-features"
)
"""Deep Sets with additional constant features."""

MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_EXTRA_DIMS_PRESERVE_INIT = (
  "deepsets-quantity-weighted-zero-extra-dims-preserve-init"
)
"""Deep Sets with zero extra dimensions, preserving initialization."""

MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2 = "deepsets-quantity-weighted-head-v2"
"""Deep Sets with head v2 architecture."""

MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2_REGULARIZED = (
  "deepsets-quantity-weighted-head-v2-regularized"
)
"""Deep Sets head v2 with regularization."""


# Deep Sets package feature variants
MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES = "deepsets-quantity-weighted-packages"
"""Deep Sets with package features (current feature set)."""

MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES_V1 = "deepsets-quantity-weighted-packages-v1"
"""Deep Sets with package features (v1 feature set)."""

MODEL_DEEPSETS_QUANTITY_WEIGHTED_SUPPORT_MATCHED_UNSCORED_PACKAGES = (
  "deepsets-quantity-weighted-support-matched-unscored-packages"
)
"""Deep Sets with support-matched unscored package features."""

MODEL_DEEPSETS_QUANTITY_WEIGHTED_SHUFFLED_PACKAGES = (
  "deepsets-quantity-weighted-shuffled-packages"
)
"""Deep Sets with shuffled package features (ablation)."""

MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES = (
  "deepsets-quantity-weighted-zero-packages"
)
"""Deep Sets with zeroed package features (ablation)."""

MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO = "deepsets-quantity-weighted-extra-rho"
"""Deep Sets with extra rho dimensions."""

MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES = (
  "deepsets-quantity-weighted-synthetic-packages"
)
"""Deep Sets with synthetic package features."""

MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES = (
  "deepsets-quantity-weighted-label-shuffled-mined-packages"
)
"""Deep Sets with label-shuffled mined packages (ablation)."""

MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE = "deepsets-quantity-weighted-prototype"
"""Deep Sets with prototype-based classification head."""


# Set Transformer models
MODEL_SET_TRANSFORMER = "set-transformer"
"""Base Set Transformer model."""

MODEL_SET_TRANSFORMER_PMA = "set-transformer-pma"
"""Set Transformer with PMA pooling."""

MODEL_SET_TRANSFORMER_QUANTITY_WEIGHTED = "set-transformer-quantity-weighted"
"""Set Transformer with quantity-weighted pooling."""

MODEL_SET_TRANSFORMER_PARTIAL_BALANCED = (
  "set-transformer-mainboard-hypergeometric-partial-contextual-mixture-"
  "balanced-regularized-label-smoothing-010"
)
"""Set Transformer with hypergeometric pooling, partial observation,
balanced sampling, and label smoothing (A3)."""

MODEL_SET_TRANSFORMER_A6 = "set-transformer-a6-supcon"
"""Set Transformer A6: supervised contrastive loss."""

MODEL_SET_TRANSFORMER_A8 = "set-transformer-a8-product-anchor"
"""Set Transformer A8: product manifold with anchor loss."""

MODEL_SET_TRANSFORMER_A9 = "set-transformer-a9-adaptive-manifold"
"""Set Transformer A9: adaptive manifold geometry."""

MODEL_SET_TRANSFORMER_A10 = "set-transformer-a10-residual-anchor"
"""Set Transformer A10: residual anchor connections."""

MODEL_SET_TRANSFORMER_A11 = "set-transformer-a11-canonical-family"
"""Set Transformer A11: canonical family targets."""


# Short aliases for CLI convenience
MODEL_A1_5 = "a1.5"
MODEL_A15 = "a15"
MODEL_A2PP = "a2++"
MODEL_A2_PP = "a2pp"
MODEL_A2 = "a2"
MODEL_A3 = "a3"
MODEL_A6 = "a6"
MODEL_A8 = "a8"
MODEL_A9 = "a9"
MODEL_A10 = "a10"
MODEL_A11 = "a11"


# Alias resolution map
MODEL_ALIASES = {
  MODEL_A1_5: MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
  MODEL_A15: MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
  MODEL_A2PP: MODEL_DEEPSETS_PLUSPLUS_REGULARIZED_LABEL_SMOOTHING_010,
  MODEL_A2_PP: MODEL_DEEPSETS_PLUSPLUS_REGULARIZED_LABEL_SMOOTHING_010,
  MODEL_A2: MODEL_DEEPSETS_PLUSPLUS_REGULARIZED_LABEL_SMOOTHING_010,
  MODEL_A3: MODEL_SET_TRANSFORMER_PARTIAL_BALANCED,
  MODEL_A6: MODEL_SET_TRANSFORMER_A6,
  MODEL_A8: MODEL_SET_TRANSFORMER_A8,
  MODEL_A9: MODEL_SET_TRANSFORMER_A9,
  MODEL_A10: MODEL_SET_TRANSFORMER_A10,
  MODEL_A11: MODEL_SET_TRANSFORMER_A11,
}
"""Short alias to canonical model name mapping."""


MODEL_ALL = "all"
"""Special value meaning 'train all supported models'."""


# Default model set for quick runs
DEFAULT_MODEL_SET = (
  MODEL_POOLED_LINEAR,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
)
"""Default models trained when --model is not specified."""


# All supported models (explicit list for validation)
SUPPORTED_MODELS = (
  MODEL_POOLED_LINEAR,
  MODEL_POOLED_LINEAR_PACKAGE_ONLY,
  MODEL_POOLED_LINEAR_PACKAGES,
  MODEL_DEEPSETS,
  MODEL_DEEPSETS_SUM,
  MODEL_DEEPSETS_MEAN,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
  MODEL_DEEPSETS_PLUSPLUS_REGULARIZED,
  MODEL_DEEPSETS_PLUSPLUS_REGULARIZED_LABEL_SMOOTHING_005,
  MODEL_DEEPSETS_PLUSPLUS_REGULARIZED_LABEL_SMOOTHING_010,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_WIDE_RHO,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_CONSTANT_FEATURES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_EXTRA_DIMS_PRESERVE_INIT,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2_REGULARIZED,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES_V1,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_SUPPORT_MATCHED_UNSCORED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_SHUFFLED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE,
  MODEL_SET_TRANSFORMER,
  MODEL_SET_TRANSFORMER_PMA,
  MODEL_SET_TRANSFORMER_QUANTITY_WEIGHTED,
  MODEL_SET_TRANSFORMER_PARTIAL_BALANCED,
  MODEL_SET_TRANSFORMER_A6,
  MODEL_SET_TRANSFORMER_A8,
  MODEL_SET_TRANSFORMER_A9,
  MODEL_SET_TRANSFORMER_A10,
  MODEL_SET_TRANSFORMER_A11,
)
"""Complete list of all supported model identifiers."""


# Internal: model -> pooling mode mapping
_MODEL_POOLING = {
  MODEL_DEEPSETS_SUM: POOLING_SUM,
  MODEL_DEEPSETS_MEAN: POOLING_MEAN,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED: POOLING_QUANTITY_WEIGHTED,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED: POOLING_QUANTITY_WEIGHTED,
}

# Internal: model -> label smoothing mapping
_DEEPSETS_PLUSPLUS_LABEL_SMOOTHING = {
  MODEL_DEEPSETS_PLUSPLUS_REGULARIZED: 0.0,
  MODEL_DEEPSETS_PLUSPLUS_REGULARIZED_LABEL_SMOOTHING_005: 0.05,
  MODEL_DEEPSETS_PLUSPLUS_REGULARIZED_LABEL_SMOOTHING_010: 0.10,
}

# Internal: models using head v2 attribution
_DEEPSETS_HEAD_ATTRIBUTION_MODELS = (
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_WIDE_RHO,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_CONSTANT_FEATURES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_EXTRA_DIMS_PRESERVE_INIT,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2_REGULARIZED,
)

# Internal: Set Transformer model -> pooling mode mapping
_SET_TRANSFORMER_POOLING = {
  MODEL_SET_TRANSFORMER_PMA: SET_TRANSFORMER_POOLING_PMA,
  MODEL_SET_TRANSFORMER_QUANTITY_WEIGHTED: SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED,
  MODEL_SET_TRANSFORMER_PARTIAL_BALANCED: SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
  MODEL_SET_TRANSFORMER_A6: SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
  MODEL_SET_TRANSFORMER_A8: SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
  MODEL_SET_TRANSFORMER_A9: SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
  MODEL_SET_TRANSFORMER_A10: SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
  MODEL_SET_TRANSFORMER_A11: SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
}


# Training hyperparameter defaults
DEFAULT_LEARNING_RATE = 0.005
"""Default learning rate for neural models (Deep Sets, Set Transformer)."""

DEFAULT_REGULARIZED_WEIGHT_DECAY = 1e-3
"""Default weight decay for regularized models."""

DEFAULT_HEAD_V2_RHO_HIDDEN_DIM = 443
"""Default hidden dimension for head v2 rho projection."""

DEFAULT_DEEPSETS_PLUSPLUS_DROPOUT = 0.1
"""Default dropout rate for Deep Sets++ architecture."""


# Saved model seed policies
SAVED_MODEL_SEED_POLICY_SINGLE = "single"
"""Use exactly one seed for saved model export."""

SAVED_MODEL_SEED_POLICY_FIRST = "first"
"""Use the first seed from a multi-seed run for saved model export."""

SAVED_MODEL_SEED_POLICIES = (
  SAVED_MODEL_SEED_POLICY_SINGLE,
  SAVED_MODEL_SEED_POLICY_FIRST,
)
"""Valid saved model seed policies."""


# Target label levels
TARGET_LABEL_LEVEL_SOURCE = "source"
"""Train on source archetype labels (e.g., 'Azorius Control')."""

TARGET_LABEL_LEVEL_CANONICAL_FAMILY = "canonical-family"
"""Train on canonical family labels (e.g., 'Control')."""

TARGET_LABEL_LEVELS = (
  TARGET_LABEL_LEVEL_SOURCE,
  TARGET_LABEL_LEVEL_CANONICAL_FAMILY,
)
"""Valid target label levels."""


# Split ordering for evaluation
PRIMARY_EVALUATION_SPLITS = ("final-test", "test", "dev-test", "validation")
"""Evaluation splits in priority order (first available is used as primary)."""

SPLIT_ORDER = (
  "train",
  "validation",
  "dev-test",
  "test",
  "final-test",
  "novelty_holdout",
)
"""Canonical split order for reporting and iteration."""

SPLIT_ORDER_INDEX = {split_name: index for index, split_name in enumerate(SPLIT_ORDER)}
"""Split name to ordinal index mapping."""


# Prediction output modes
PREDICTION_OUTPUT_SUMMARY = "summary"
"""Output only summary metrics per split."""

PREDICTION_OUTPUT_ERRORS = "errors"
"""Output summary metrics plus misclassified examples."""

PREDICTION_OUTPUT_FULL = "full"
"""Output all predictions with full detail."""

PREDICTION_OUTPUT_MODES = (
  PREDICTION_OUTPUT_SUMMARY,
  PREDICTION_OUTPUT_ERRORS,
  PREDICTION_OUTPUT_FULL,
)
"""Valid prediction output modes."""


# Package feature set identifiers
PACKAGE_FEATURE_SET_CURRENT = "current"
"""Current production package feature set."""

PACKAGE_FEATURE_SET_V2 = "v2"
"""Version 2 package feature set."""

PACKAGE_FEATURE_SET_SUPPORT_MATCHED_UNSCORED = "support_matched_unscored"
"""Support-matched unscored package features."""

PACKAGE_FEATURE_SET_SHUFFLED = "shuffled"
"""Shuffled package features (ablation control)."""

PACKAGE_FEATURE_SET_ZERO = "zero"
"""Zeroed package features (ablation control)."""

PACKAGE_FEATURE_SET_SYNTHETIC = "synthetic"
"""Synthetic package features."""

PACKAGE_FEATURE_SET_LABEL_SHUFFLED_MINED = "label_shuffled_mined"
"""Label-shuffled mined package features (ablation control)."""

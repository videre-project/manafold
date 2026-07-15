from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from manafold.data.export import SOURCE_ARCHETYPE_PROXY
from manafold.models.data import (
  ModelExample,
  TrainingDataset,
  load_training_dataset,
  split_examples,
)
from manafold.models.deepsets import (
  POOLING_MEAN,
  POOLING_MODES,
  POOLING_QUANTITY_WEIGHTED,
  POOLING_SUM,
  SET_TRANSFORMER_POOLING_PMA,
  SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED,
  Classifier,
  DeepSetsClassifier,
  PooledLinearClassifier,
  PrototypeClassifier,
  SetTransformerClassifier,
  max_quantity,
)
from manafold.models.metrics import (
  annotate_temperature_scaled_confidence,
  calibration_metrics,
  classification_metrics,
)
from manafold.models.model_artifacts import save_model_artifact
from manafold.models.packages import (
  DEFAULT_PACKAGE_TYPES,
  PACKAGE_SCORING_BAYESIAN_LOG_ODDS,
  PACKAGE_SCORING_MODES,
  PACKAGE_SCORING_SUPPORT_LIFT,
  PACKAGE_TYPES,
  PackageFeatureSet,
  mine_package_features,
  mine_support_matched_unscored_package_features,
)
from manafold.models.taxonomy import (
  TaxonomyEvaluationConfig,
  canonical_label_id,
  load_taxonomy_evaluation_config,
  taxonomy_metrics,
)

MODEL_POOLED_LINEAR = "pooled-linear"
MODEL_POOLED_LINEAR_PACKAGE_ONLY = "pooled-linear-package-only"
MODEL_POOLED_LINEAR_PACKAGES = "pooled-linear-packages"
MODEL_DEEPSETS = "deepsets"
MODEL_DEEPSETS_SUM = "deepsets-sum"
MODEL_DEEPSETS_MEAN = "deepsets-mean"
MODEL_DEEPSETS_QUANTITY_WEIGHTED = "deepsets-quantity-weighted"
MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED = (
  "deepsets-quantity-weighted-regularized"
)
MODEL_ARTIFACT_SEED_POLICY_SINGLE = "single"
MODEL_ARTIFACT_SEED_POLICY_FIRST = "first"
MODEL_ARTIFACT_SEED_POLICIES = (
  MODEL_ARTIFACT_SEED_POLICY_SINGLE,
  MODEL_ARTIFACT_SEED_POLICY_FIRST,
)
MODEL_DEEPSETS_QUANTITY_WEIGHTED_WIDE_RHO = "deepsets-quantity-weighted-wide-rho"
MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_CONSTANT_FEATURES = (
  "deepsets-quantity-weighted-extra-constant-features"
)
MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_EXTRA_DIMS_PRESERVE_INIT = (
  "deepsets-quantity-weighted-zero-extra-dims-preserve-init"
)
MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2 = "deepsets-quantity-weighted-head-v2"
MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2_REGULARIZED = (
  "deepsets-quantity-weighted-head-v2-regularized"
)
MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES = "deepsets-quantity-weighted-packages"
MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES_V1 = (
  "deepsets-quantity-weighted-packages-v1"
)
MODEL_DEEPSETS_QUANTITY_WEIGHTED_SUPPORT_MATCHED_UNSCORED_PACKAGES = (
  "deepsets-quantity-weighted-support-matched-unscored-packages"
)
MODEL_DEEPSETS_QUANTITY_WEIGHTED_SHUFFLED_PACKAGES = (
  "deepsets-quantity-weighted-shuffled-packages"
)
MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES = (
  "deepsets-quantity-weighted-zero-packages"
)
MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO = (
  "deepsets-quantity-weighted-extra-rho"
)
MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES = (
  "deepsets-quantity-weighted-synthetic-packages"
)
MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES = (
  "deepsets-quantity-weighted-label-shuffled-mined-packages"
)
MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE = "deepsets-quantity-weighted-prototype"
MODEL_SET_TRANSFORMER = "set-transformer"
MODEL_SET_TRANSFORMER_PMA = "set-transformer-pma"
MODEL_SET_TRANSFORMER_QUANTITY_WEIGHTED = "set-transformer-quantity-weighted"
MODEL_ALL = "all"
DEFAULT_LEARNING_RATE = 0.005
DEFAULT_REGULARIZED_WEIGHT_DECAY = 1e-3
DEFAULT_HEAD_V2_RHO_HIDDEN_DIM = 443
TARGET_LABEL_LEVEL_SOURCE = "source"
TARGET_LABEL_LEVEL_CANONICAL_FAMILY = "canonical-family"
TARGET_LABEL_LEVELS = (
  TARGET_LABEL_LEVEL_SOURCE,
  TARGET_LABEL_LEVEL_CANONICAL_FAMILY,
)
PRIMARY_EVALUATION_SPLITS = ("final-test", "test", "dev-test", "validation")
SPLIT_ORDER = (
  "train",
  "validation",
  "dev-test",
  "test",
  "final-test",
  "novelty_holdout",
)
SPLIT_ORDER_INDEX = {
  split_name: index
  for index, split_name in enumerate(SPLIT_ORDER)
}
PREDICTION_OUTPUT_SUMMARY = "summary"
PREDICTION_OUTPUT_ERRORS = "errors"
PREDICTION_OUTPUT_FULL = "full"
PREDICTION_OUTPUT_MODES = (
  PREDICTION_OUTPUT_SUMMARY,
  PREDICTION_OUTPUT_ERRORS,
  PREDICTION_OUTPUT_FULL,
)
SUPPORTED_MODELS = (
  MODEL_POOLED_LINEAR,
  MODEL_POOLED_LINEAR_PACKAGE_ONLY,
  MODEL_POOLED_LINEAR_PACKAGES,
  MODEL_DEEPSETS,
  MODEL_DEEPSETS_SUM,
  MODEL_DEEPSETS_MEAN,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
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
)
DEFAULT_MODEL_SET = (
  MODEL_POOLED_LINEAR,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
)

PACKAGE_FEATURE_SET_CURRENT = "current"
PACKAGE_FEATURE_SET_V2 = "v2"
PACKAGE_FEATURE_SET_SUPPORT_MATCHED_UNSCORED = "support_matched_unscored"
PACKAGE_FEATURE_SET_SHUFFLED = "shuffled"
PACKAGE_FEATURE_SET_ZERO = "zero"
PACKAGE_FEATURE_SET_SYNTHETIC = "synthetic"
PACKAGE_FEATURE_SET_LABEL_SHUFFLED_MINED = "label_shuffled_mined"

_MODEL_POOLING = {
  MODEL_DEEPSETS_SUM: POOLING_SUM,
  MODEL_DEEPSETS_MEAN: POOLING_MEAN,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED: POOLING_QUANTITY_WEIGHTED,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED: POOLING_QUANTITY_WEIGHTED,
}

_DEEPSETS_HEAD_ATTRIBUTION_MODELS = (
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_WIDE_RHO,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_CONSTANT_FEATURES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_EXTRA_DIMS_PRESERVE_INIT,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2_REGULARIZED,
)

_SET_TRANSFORMER_POOLING = {
  MODEL_SET_TRANSFORMER_PMA: SET_TRANSFORMER_POOLING_PMA,
  MODEL_SET_TRANSFORMER_QUANTITY_WEIGHTED: SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED,
}


def run_model_training(
  dataset_path: Path,
  *,
  output: Path | None = None,
  model_names: tuple[str, ...] = DEFAULT_MODEL_SET,
  pooling: str = POOLING_SUM,
  target_source: str = SOURCE_ARCHETYPE_PROXY,
  epochs: int = 80,
  learning_rate: float = DEFAULT_LEARNING_RATE,
  pooled_learning_rate: float | None = None,
  neural_learning_rate: float | None = None,
  weight_decay: float = 0.0,
  deepsets_regularized_weight_decay: float = DEFAULT_REGULARIZED_WEIGHT_DECAY,
  head_v2_weight_decay: float = DEFAULT_REGULARIZED_WEIGHT_DECAY,
  package_weight_decay: float | None = None,
  seed: int = 13,
  seeds: tuple[int, ...] | None = None,
  embedding_dim: int = 32,
  hidden_dim: int = 64,
  head_v2_rho_hidden_dim: int = DEFAULT_HEAD_V2_RHO_HIDDEN_DIM,
  attention_heads: int = 4,
  attention_layers: int = 2,
  batch_size: int = 32,
  shuffle: bool = True,
  max_steps: int | None = None,
  device: str = "auto",
  export_embeddings: bool = False,
  prediction_output: str = PREDICTION_OUTPUT_ERRORS,
  package_min_support: int = 25,
  package_min_event_support: int = 8,
  package_max_size: int = 3,
  package_max_count: int = 2048,
  package_scale: float = 1.0,
  package_projection_dim: int = 64,
  package_scoring: str = PACKAGE_SCORING_BAYESIAN_LOG_ODDS,
  package_min_best_label_count: int = 10,
  package_min_best_label_precision: float = 0.05,
  package_max_label_entropy: float | None = None,
  package_max_train_activation_rate: float | None = 0.08,
  package_types: tuple[str, ...] = DEFAULT_PACKAGE_TYPES,
  package_random_seed: int = 13,
  taxonomy_eval: Path | None = None,
  target_label_level: str = TARGET_LABEL_LEVEL_SOURCE,
  canonical_targets: Path | None = None,
  model_artifact_output: Path | None = None,
  model_artifact_model_name: str = MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
  model_artifact_seed_policy: str = MODEL_ARTIFACT_SEED_POLICY_SINGLE,
) -> dict[str, Any]:
  if prediction_output not in PREDICTION_OUTPUT_MODES:
    supported = ", ".join(PREDICTION_OUTPUT_MODES)
    raise ValueError(
      f"Unsupported prediction output {prediction_output!r}; choose one of: {supported}."
    )
  if head_v2_rho_hidden_dim <= 0:
    raise ValueError("head_v2_rho_hidden_dim must be positive.")
  if deepsets_regularized_weight_decay < 0:
    raise ValueError("deepsets_regularized_weight_decay must be non-negative.")
  if head_v2_weight_decay < 0:
    raise ValueError("head_v2_weight_decay must be non-negative.")
  if target_label_level not in TARGET_LABEL_LEVELS:
    supported = ", ".join(TARGET_LABEL_LEVELS)
    raise ValueError(
      f"Unsupported target label level {target_label_level!r}; "
      f"choose one of: {supported}."
    )
  if model_artifact_seed_policy not in MODEL_ARTIFACT_SEED_POLICIES:
    supported = ", ".join(MODEL_ARTIFACT_SEED_POLICIES)
    raise ValueError(
      f"Unsupported model artifact seed policy {model_artifact_seed_policy!r}; "
      f"choose one of: {supported}."
    )

  selected_models = _normalize_model_names(model_names, pooling=pooling)
  if model_artifact_output is not None and model_artifact_model_name not in selected_models:
    raise ValueError(
      f"Artifact model {model_artifact_model_name!r} must be one of the selected models."
    )
  if model_artifact_output is not None and model_artifact_model_name in (
    MODEL_DEEPSETS_QUANTITY_WEIGHTED,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE,
  ):
    raise ValueError(
      "Artifact export v0 supports directly trained model families. "
      f"Use {MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED!r} for the A1.5 artifact."
    )
  run_seeds = _normalize_seeds(seed, seeds)
  if (
    model_artifact_output is not None
    and model_artifact_seed_policy == MODEL_ARTIFACT_SEED_POLICY_SINGLE
    and len(run_seeds) != 1
  ):
    raise ValueError(
      "Artifact export uses seed policy 'single' by default; pass exactly one "
      "seed, or set --model-artifact-seed-policy first for an explicit "
      "research artifact from the first seed."
    )
  model_artifact_seed = run_seeds[0] if model_artifact_output is not None else None
  dataset = load_training_dataset(dataset_path, target_source=target_source)
  taxonomy_config = load_taxonomy_evaluation_config(
    _taxonomy_eval_path(taxonomy_eval),
    target_source=target_source,
  )
  canonical_target_config = load_taxonomy_evaluation_config(
    _canonical_targets_path(canonical_targets),
    target_source=target_source,
  )
  if (
    target_label_level == TARGET_LABEL_LEVEL_CANONICAL_FAMILY
    and not canonical_target_config.enabled
  ):
    raise ValueError(
      "canonical-family training requires a non-empty reviewed target map. "
      "Pass --canonical-targets explicitly."
    )
  source_examples = list(dataset.examples)
  examples = _training_target_examples(
    source_examples,
    target_label_level=target_label_level,
    canonical_target_config=canonical_target_config,
  )
  dataset = replace(
    dataset,
    examples=tuple(examples),
    labels=_train_labels(examples),
  )
  grouped = split_examples(examples)
  split_plan = _split_plan(grouped)
  train_examples = grouped.get("train", [])
  validation_examples = grouped.get("validation", [])
  evaluation_examples = [
    example
    for split_name in split_plan["evaluation_splits"]
    for example in grouped.get(split_name, [])
  ]
  quantity_count = max(1, max_quantity(examples) + 1)
  train_label_support = _label_support(train_examples)
  train_source_label_support = _source_label_support(train_examples)
  observed_label_count = len({example.target_label_id for example in examples})
  source_observed_label_count = len({
    example.source_label_id
    for example in examples
  })
  package_feature_set_names = _selected_package_feature_sets(selected_models)
  package_feature_set_names = _with_inference_ablation_feature_sets(
    selected_models,
    package_feature_set_names,
  )
  package_model_selected = bool(package_feature_set_names)
  package_feature_sets = _build_package_feature_sets(
    package_feature_set_names,
    train_examples=train_examples,
    examples=examples,
    card_vocab=dataset.card_vocab,
    package_min_support=package_min_support,
    package_min_event_support=package_min_event_support,
    package_max_size=package_max_size,
    package_max_count=package_max_count,
    package_scoring=package_scoring,
    package_min_best_label_count=package_min_best_label_count,
    package_min_best_label_precision=package_min_best_label_precision,
    package_max_label_entropy=package_max_label_entropy,
    package_max_train_activation_rate=package_max_train_activation_rate,
    package_types=package_types,
    package_random_seed=package_random_seed,
  )
  primary_package_features = (
    package_feature_sets.get(PACKAGE_FEATURE_SET_V2)
    or package_feature_sets.get(PACKAGE_FEATURE_SET_CURRENT)
    or next(iter(package_feature_sets.values()), PackageFeatureSet([]))
  )
  architecture_package_count = (
    len(primary_package_features)
    if package_feature_sets
    else package_max_count
  )
  package_mining: dict[str, Any] = {"enabled": False}
  if package_model_selected:
    package_mining = {
      "enabled": True,
      "method": "train_only_supervised",
      "split": "train",
      "min_support": package_min_support,
      "min_event_support": package_min_event_support,
      "max_size": package_max_size,
      "max_count": package_max_count,
      "package_scale": package_scale,
      "package_projection_dim": package_projection_dim,
      "scoring": package_scoring,
      "min_best_label_count": package_min_best_label_count,
      "min_best_label_precision": package_min_best_label_precision,
      "max_label_entropy": package_max_label_entropy,
      "max_train_activation_rate": package_max_train_activation_rate,
      "package_types": list(package_types),
      "random_seed": package_random_seed,
      **primary_package_features.summary(),
      "diagnostics": primary_package_features.diagnostics(
        examples_by_split=grouped,
        card_vocab=dataset.card_vocab,
        zone_vocab=dataset.zone_vocab,
      ),
    }

  result: dict[str, Any] = {
    "run_id": "model_training",
    "dataset_version": dataset.dataset_version,
    "target_source": target_source,
    "training_target": {
      "label_level": target_label_level,
      "canonical_targets": (
        canonical_target_config.to_dict()
        if target_label_level == TARGET_LABEL_LEVEL_CANONICAL_FAMILY
        else {"enabled": False}
      ),
    },
    "model_names": list(selected_models),
    "seeds": list(run_seeds),
    "example_count": len(examples),
    "card_count": dataset.card_count,
    "zone_count": dataset.zone_count,
    "label_count": len(dataset.labels),
    "train_label_count": len(dataset.labels),
    "observed_label_count": observed_label_count,
    "source_observed_label_count": source_observed_label_count,
    "label_vocabulary": "train_only",
    "quantity_count": quantity_count,
    "split_counts": split_plan["counts"],
    "evaluation_splits": list(split_plan["evaluation_splits"]),
    "primary_evaluation_split": split_plan["primary_evaluation_split"],
    "train_label_support": train_label_support,
    "learning_rate": learning_rate,
    "pooled_learning_rate": pooled_learning_rate,
    "neural_learning_rate": neural_learning_rate,
    "weight_decay": weight_decay,
    "deepsets_regularized_weight_decay": deepsets_regularized_weight_decay,
    "head_v2_weight_decay": head_v2_weight_decay,
    "head_v2_rho_hidden_dim": head_v2_rho_hidden_dim,
    "package_weight_decay": package_weight_decay,
    "package_projection_dim": package_projection_dim,
    "max_steps": max_steps,
    "export_embeddings": export_embeddings,
    "prediction_output": prediction_output,
    "taxonomy_evaluation": taxonomy_config.to_dict(),
    "package_mining": package_mining,
    "package_mining_sets": _package_set_summaries(
      package_feature_sets,
      examples_by_split=grouped,
      card_vocab=dataset.card_vocab,
      zone_vocab=dataset.zone_vocab,
    ),
    "model_artifact_export": {
      "enabled": model_artifact_output is not None,
      "model_name": model_artifact_model_name if model_artifact_output else None,
      "seed_policy": (
        model_artifact_seed_policy
        if model_artifact_output is not None
        else None
      ),
      "export_seed": model_artifact_seed,
      "output": str(model_artifact_output) if model_artifact_output else None,
    },
    "status": "completed",
    "models": {},
    "comparison": {},
  }

  skip_reason = _training_skip_reason(
    train_examples,
    evaluation_examples,
    dataset.labels,
  )
  if skip_reason is not None:
    result["status"] = "skipped"
    result["reason"] = skip_reason
    return _write_result(result, output)

  shared_deepsets_runs: list[dict[str, dict[str, Any]]] | None = None
  include_prototype = MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE in selected_models
  for model_name in selected_models:
    if model_name in (
      MODEL_DEEPSETS_QUANTITY_WEIGHTED,
      MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE,
    ):
      if shared_deepsets_runs is None:
        shared_deepsets_runs = [
          _run_quantity_weighted_deepsets_training(
            seed=run_seed,
            dataset=dataset,
            examples=examples,
            train_examples=train_examples,
            validation_examples=validation_examples,
            train_label_support=train_label_support,
            train_source_label_support=train_source_label_support,
            quantity_count=quantity_count,
            package_features=PackageFeatureSet([]),
            package_feature_set_name=None,
            output=output,
            epochs=epochs,
            learning_rate=_model_learning_rate(
              MODEL_DEEPSETS_QUANTITY_WEIGHTED,
              learning_rate=learning_rate,
              pooled_learning_rate=pooled_learning_rate,
              neural_learning_rate=neural_learning_rate,
            ),
            weight_decay=_model_weight_decay(
              MODEL_DEEPSETS_QUANTITY_WEIGHTED,
              weight_decay=weight_decay,
              deepsets_regularized_weight_decay=deepsets_regularized_weight_decay,
              head_v2_weight_decay=head_v2_weight_decay,
              package_weight_decay=package_weight_decay,
            ),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            head_v2_rho_hidden_dim=head_v2_rho_hidden_dim,
            batch_size=batch_size,
            shuffle=shuffle,
            max_steps=max_steps,
            device=device,
            export_embeddings=export_embeddings,
            prediction_output=prediction_output,
            include_prototype=include_prototype,
            include_seed_in_artifact_name=len(run_seeds) > 1,
            package_scale=package_scale,
            package_projection_dim=package_projection_dim,
            architecture_package_count=architecture_package_count,
            taxonomy_config=taxonomy_config,
          )
          for run_seed in run_seeds
        ]
      seed_runs = [
        run[model_name]
        for run in shared_deepsets_runs
      ]
    else:
      package_feature_set_name = _package_feature_set_for_model(model_name)
      package_features = (
        package_feature_sets.get(package_feature_set_name, PackageFeatureSet([]))
        if package_feature_set_name is not None
        else PackageFeatureSet([])
      )
      seed_runs = [
        _run_single_model_training(
          model_name,
          seed=run_seed,
          dataset=dataset,
          examples=examples,
          train_examples=train_examples,
          validation_examples=validation_examples,
          train_label_support=train_label_support,
          train_source_label_support=train_source_label_support,
          quantity_count=quantity_count,
          package_features=package_features,
          package_feature_set_name=package_feature_set_name,
          inference_package_feature_sets=_inference_package_feature_sets_for_model(
            model_name,
            package_feature_sets,
          ),
          output=output,
          epochs=epochs,
          learning_rate=_model_learning_rate(
            model_name,
            learning_rate=learning_rate,
            pooled_learning_rate=pooled_learning_rate,
            neural_learning_rate=neural_learning_rate,
          ),
          weight_decay=_model_weight_decay(
            model_name,
            weight_decay=weight_decay,
            deepsets_regularized_weight_decay=deepsets_regularized_weight_decay,
            head_v2_weight_decay=head_v2_weight_decay,
            package_weight_decay=package_weight_decay,
          ),
          embedding_dim=embedding_dim,
          hidden_dim=hidden_dim,
          head_v2_rho_hidden_dim=head_v2_rho_hidden_dim,
          attention_heads=attention_heads,
          attention_layers=attention_layers,
          batch_size=batch_size,
          shuffle=shuffle,
          max_steps=max_steps,
          device=device,
          export_embeddings=export_embeddings,
          prediction_output=prediction_output,
          include_seed_in_artifact_name=len(run_seeds) > 1,
          package_scale=package_scale,
          package_projection_dim=package_projection_dim,
          architecture_package_count=architecture_package_count,
          taxonomy_config=taxonomy_config,
          model_artifact_output=(
            model_artifact_output
            if (
              model_name == model_artifact_model_name
              and run_seed == model_artifact_seed
            )
            else None
          ),
          training_result=result,
        )
        for run_seed in run_seeds
      ]
    result["models"][model_name] = _model_result_from_seed_runs(
      model_name,
      seed_runs,
      seeds=run_seeds,
      primary_evaluation_split=str(split_plan["primary_evaluation_split"]),
    )

  if MODEL_POOLED_LINEAR in result["models"]:
    result["comparison"] = {
      model_name: _accuracy_comparison(
        result["models"][MODEL_POOLED_LINEAR]["metrics"],
        result["models"][model_name]["metrics"],
      )
      for model_name in selected_models
      if model_name != MODEL_POOLED_LINEAR
    }

  result["paired_model_tests"] = _paired_model_tests(result)

  result["package_signal_by_label"] = _package_signal_by_label(
    result,
    train_label_support=train_label_support,
    evaluation_examples=grouped.get(str(split_plan["primary_evaluation_split"]), []),
    evaluation_split=str(split_plan["primary_evaluation_split"]),
    package_features=package_feature_sets.get(
      PACKAGE_FEATURE_SET_V2,
      PackageFeatureSet([]),
    ),
    card_vocab=dataset.card_vocab,
    zone_vocab=dataset.zone_vocab,
  )

  return _write_result(result, output)


def _normalize_model_names(
  model_names: tuple[str, ...],
  *,
  pooling: str,
) -> tuple[str, ...]:
  if pooling not in POOLING_MODES:
    supported = ", ".join(POOLING_MODES)
    raise ValueError(f"Unsupported pooling mode {pooling!r}; choose one of: {supported}.")
  if not model_names:
    raise ValueError("At least one model must be selected.")
  if model_names == (MODEL_ALL,):
    return DEFAULT_MODEL_SET

  normalized: list[str] = []
  for model_name in model_names:
    if model_name == MODEL_ALL:
      normalized.extend(DEFAULT_MODEL_SET)
      continue
    if model_name == MODEL_DEEPSETS:
      normalized.append(_deepsets_model_name(pooling))
      continue
    if model_name == MODEL_SET_TRANSFORMER:
      normalized.append(MODEL_SET_TRANSFORMER_PMA)
      continue
    if model_name not in SUPPORTED_MODELS:
      supported = ", ".join((MODEL_ALL, *SUPPORTED_MODELS))
      raise ValueError(f"Unsupported model {model_name!r}; choose one of: {supported}.")
    normalized.append(model_name)

  return tuple(dict.fromkeys(normalized))


def _taxonomy_eval_path(path: Path | None) -> Path | None:
  return path


def _canonical_targets_path(path: Path | None) -> Path | None:
  return path


def _training_target_examples(
  examples: list[ModelExample],
  *,
  target_label_level: str,
  canonical_target_config: TaxonomyEvaluationConfig,
) -> list[ModelExample]:
  if target_label_level == TARGET_LABEL_LEVEL_SOURCE:
    return examples
  if target_label_level != TARGET_LABEL_LEVEL_CANONICAL_FAMILY:
    raise AssertionError(f"Unhandled target label level: {target_label_level}")

  return [
    replace(
      example,
      target_label_id=canonical_label_id(
        example.source_label_id,
        _example_taxonomy_context(example),
        canonical_target_config,
      ),
    )
    for example in examples
  ]


def _example_taxonomy_context(example: ModelExample) -> dict[str, Any]:
  return {
    "event_date": (
      example.event_date.isoformat()
      if example.event_date is not None
      else None
    ),
    "format": example.format_code,
  }


def _train_labels(examples: list[ModelExample]) -> tuple[str, ...]:
  return tuple(
    sorted({
      example.target_label_id
      for example in examples
      if example.split_name == "train"
    })
  )


def _deepsets_model_name(pooling: str) -> str:
  if pooling == POOLING_SUM:
    return MODEL_DEEPSETS_SUM
  if pooling == POOLING_MEAN:
    return MODEL_DEEPSETS_MEAN
  if pooling == POOLING_QUANTITY_WEIGHTED:
    return MODEL_DEEPSETS_QUANTITY_WEIGHTED
  raise AssertionError(f"Unhandled pooling mode: {pooling}")


def _model_learning_rate(
  model_name: str,
  *,
  learning_rate: float,
  pooled_learning_rate: float | None,
  neural_learning_rate: float | None,
) -> float:
  if model_name in (
    MODEL_POOLED_LINEAR,
    MODEL_POOLED_LINEAR_PACKAGE_ONLY,
    MODEL_POOLED_LINEAR_PACKAGES,
  ):
    return pooled_learning_rate if pooled_learning_rate is not None else learning_rate
  return neural_learning_rate if neural_learning_rate is not None else learning_rate


def _model_weight_decay(
  model_name: str,
  *,
  weight_decay: float,
  deepsets_regularized_weight_decay: float,
  head_v2_weight_decay: float,
  package_weight_decay: float | None,
) -> float:
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED:
    return deepsets_regularized_weight_decay
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2_REGULARIZED:
    return head_v2_weight_decay
  if model_name in (
    MODEL_POOLED_LINEAR_PACKAGE_ONLY,
    MODEL_POOLED_LINEAR_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES_V1,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_SUPPORT_MATCHED_UNSCORED_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_SHUFFLED_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES,
  ):
    return (
      package_weight_decay
      if package_weight_decay is not None
      else weight_decay
    )
  return weight_decay


def _selected_package_feature_sets(
  selected_models: tuple[str, ...],
) -> tuple[str, ...]:
  names = [
    package_feature_set_name
    for model_name in selected_models
    if (package_feature_set_name := _package_feature_set_for_model(model_name))
    is not None
  ]
  return tuple(dict.fromkeys(names))


def _with_inference_ablation_feature_sets(
  selected_models: tuple[str, ...],
  package_feature_set_names: tuple[str, ...],
) -> tuple[str, ...]:
  if MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES not in selected_models:
    return package_feature_set_names
  return tuple(
    dict.fromkeys(
      (
        *package_feature_set_names,
        PACKAGE_FEATURE_SET_ZERO,
        PACKAGE_FEATURE_SET_SHUFFLED,
        PACKAGE_FEATURE_SET_SYNTHETIC,
      )
    )
  )


def _inference_package_feature_sets_for_model(
  model_name: str,
  package_feature_sets: dict[str, PackageFeatureSet],
) -> dict[str, PackageFeatureSet]:
  if model_name != MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES:
    return {}
  requested = {
    "zeroed": PACKAGE_FEATURE_SET_ZERO,
    "shuffled": PACKAGE_FEATURE_SET_SHUFFLED,
    "synthetic": PACKAGE_FEATURE_SET_SYNTHETIC,
  }
  return {
    ablation_name: package_feature_sets[feature_set_name]
    for ablation_name, feature_set_name in requested.items()
    if feature_set_name in package_feature_sets
  }


def _package_feature_set_for_model(model_name: str) -> str | None:
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES_V1:
    return PACKAGE_FEATURE_SET_CURRENT
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_SUPPORT_MATCHED_UNSCORED_PACKAGES:
    return PACKAGE_FEATURE_SET_SUPPORT_MATCHED_UNSCORED
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_SHUFFLED_PACKAGES:
    return PACKAGE_FEATURE_SET_SHUFFLED
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES:
    return PACKAGE_FEATURE_SET_ZERO
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO:
    return PACKAGE_FEATURE_SET_ZERO
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES:
    return PACKAGE_FEATURE_SET_SYNTHETIC
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES:
    return PACKAGE_FEATURE_SET_LABEL_SHUFFLED_MINED
  if model_name in (
    MODEL_POOLED_LINEAR_PACKAGE_ONLY,
    MODEL_POOLED_LINEAR_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES,
  ):
    return PACKAGE_FEATURE_SET_V2
  return None


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


def _build_package_feature_sets(
  package_feature_set_names: tuple[str, ...],
  *,
  train_examples: list[ModelExample],
  examples: list[ModelExample],
  card_vocab: tuple[dict[str, Any], ...],
  package_min_support: int,
  package_min_event_support: int,
  package_max_size: int,
  package_max_count: int,
  package_scoring: str,
  package_min_best_label_count: int,
  package_min_best_label_precision: float,
  package_max_label_entropy: float | None,
  package_max_train_activation_rate: float | None,
  package_types: tuple[str, ...],
  package_random_seed: int,
) -> dict[str, PackageFeatureSet]:
  if not package_feature_set_names:
    return {}

  rows: dict[str, PackageFeatureSet] = {}
  if PACKAGE_FEATURE_SET_CURRENT in package_feature_set_names:
    rows[PACKAGE_FEATURE_SET_CURRENT] = mine_package_features(
      train_examples,
      min_support=package_min_support,
      min_event_support=package_min_event_support,
      max_size=package_max_size,
      max_packages=package_max_count,
      card_vocab=card_vocab,
      package_types=PACKAGE_TYPES,
      scoring=PACKAGE_SCORING_SUPPORT_LIFT,
      min_best_label_count=1,
      min_best_label_precision=0.0,
      max_label_entropy=None,
      max_train_activation_rate=None,
    )

  needs_v2 = any(
    name in package_feature_set_names
    for name in (
      PACKAGE_FEATURE_SET_V2,
      PACKAGE_FEATURE_SET_SUPPORT_MATCHED_UNSCORED,
      PACKAGE_FEATURE_SET_SHUFFLED,
      PACKAGE_FEATURE_SET_ZERO,
      PACKAGE_FEATURE_SET_SYNTHETIC,
    )
  )
  if needs_v2:
    rows[PACKAGE_FEATURE_SET_V2] = mine_package_features(
      train_examples,
      min_support=package_min_support,
      min_event_support=package_min_event_support,
      max_size=package_max_size,
      max_packages=package_max_count,
      card_vocab=card_vocab,
      package_types=package_types,
      scoring=package_scoring,
      min_best_label_count=package_min_best_label_count,
      min_best_label_precision=package_min_best_label_precision,
      max_label_entropy=package_max_label_entropy,
      max_train_activation_rate=package_max_train_activation_rate,
    )

  if PACKAGE_FEATURE_SET_SUPPORT_MATCHED_UNSCORED in package_feature_set_names:
    rows[PACKAGE_FEATURE_SET_SUPPORT_MATCHED_UNSCORED] = (
      mine_support_matched_unscored_package_features(
        train_examples,
        reference_features=rows[PACKAGE_FEATURE_SET_V2].features,
        min_support=package_min_support,
        min_event_support=package_min_event_support,
        max_size=package_max_size,
        max_packages=package_max_count,
        card_vocab=card_vocab,
        package_types=package_types,
        random_seed=package_random_seed,
      )
    )

  if PACKAGE_FEATURE_SET_ZERO in package_feature_set_names:
    rows[PACKAGE_FEATURE_SET_ZERO] = rows[PACKAGE_FEATURE_SET_V2].with_zero_activations(
      examples
    )

  if PACKAGE_FEATURE_SET_SYNTHETIC in package_feature_set_names:
    rows[PACKAGE_FEATURE_SET_SYNTHETIC] = rows[
      PACKAGE_FEATURE_SET_V2
    ].with_synthetic_activations(
      train_examples=train_examples,
      examples=examples,
      seed=package_random_seed,
    )

  if PACKAGE_FEATURE_SET_SHUFFLED in package_feature_set_names:
    rows[PACKAGE_FEATURE_SET_SHUFFLED] = rows[
      PACKAGE_FEATURE_SET_V2
    ].with_shuffled_activations(
      examples,
      seed=package_random_seed,
      within_split=True,
    )

  if PACKAGE_FEATURE_SET_LABEL_SHUFFLED_MINED in package_feature_set_names:
    rows[PACKAGE_FEATURE_SET_LABEL_SHUFFLED_MINED] = mine_package_features(
      _label_shuffled_examples(train_examples, seed=package_random_seed),
      min_support=package_min_support,
      min_event_support=package_min_event_support,
      max_size=package_max_size,
      max_packages=package_max_count,
      card_vocab=card_vocab,
      package_types=package_types,
      scoring=package_scoring,
      min_best_label_count=package_min_best_label_count,
      min_best_label_precision=package_min_best_label_precision,
      max_label_entropy=package_max_label_entropy,
      max_train_activation_rate=package_max_train_activation_rate,
    )

  return {
    name: rows[name]
    for name in package_feature_set_names
    if name in rows
  }


def _package_set_summaries(
  package_feature_sets: dict[str, PackageFeatureSet],
  *,
  examples_by_split: dict[str, list[ModelExample]],
  card_vocab: tuple[dict[str, Any], ...],
  zone_vocab: dict[str, int],
) -> dict[str, dict[str, Any]]:
  return {
    name: {
      **feature_set.summary(),
      "diagnostics": feature_set.diagnostics(
        examples_by_split=examples_by_split,
        card_vocab=card_vocab,
        zone_vocab=zone_vocab,
      ),
    }
    for name, feature_set in package_feature_sets.items()
  }


def _label_shuffled_examples(
  train_examples: list[ModelExample],
  *,
  seed: int,
) -> list[ModelExample]:
  rng = random.Random(seed)
  labels = [example.target_label_id for example in train_examples]
  rng.shuffle(labels)
  return [
    replace(example, target_label_id=label)
    for example, label in zip(train_examples, labels, strict=True)
  ]


def _run_single_model_training(
  model_name: str,
  *,
  seed: int,
  dataset: TrainingDataset,
  examples: list[ModelExample],
  train_examples: list[ModelExample],
  validation_examples: list[ModelExample],
  train_label_support: dict[str, int],
  train_source_label_support: dict[str, int],
  quantity_count: int,
  package_features: PackageFeatureSet,
  package_feature_set_name: str | None,
  inference_package_feature_sets: dict[str, PackageFeatureSet],
  output: Path | None,
  epochs: int,
  learning_rate: float,
  weight_decay: float,
  embedding_dim: int,
  hidden_dim: int,
  head_v2_rho_hidden_dim: int,
  attention_heads: int,
  attention_layers: int,
  batch_size: int,
  shuffle: bool,
  max_steps: int | None,
  device: str,
  export_embeddings: bool,
  prediction_output: str,
  include_seed_in_artifact_name: bool,
  package_scale: float,
  package_projection_dim: int,
  architecture_package_count: int,
  taxonomy_config: TaxonomyEvaluationConfig,
  model_artifact_output: Path | None = None,
  training_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
  model, metadata = _build_classifier(
    model_name,
    labels=dataset.labels,
    card_count=dataset.card_count,
    zone_count=dataset.zone_count,
    quantity_count=quantity_count,
    package_features=package_features,
    package_feature_set_name=package_feature_set_name,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    seed=seed,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    head_v2_rho_hidden_dim=head_v2_rho_hidden_dim,
    attention_heads=attention_heads,
    attention_layers=attention_layers,
    batch_size=batch_size,
    shuffle=shuffle,
    device=device,
    package_scale=package_scale,
    package_projection_dim=package_projection_dim,
    architecture_package_count=architecture_package_count,
  )
  training_summary = model.fit(
    train_examples,
    validation_examples=validation_examples,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=shuffle,
    max_steps=max_steps,
  )
  model_result = _evaluate_trained_model(
    model_name,
    model=model,
    metadata=metadata,
    training_summary=training_summary,
    seed=seed,
    dataset=dataset,
    examples=examples,
    train_examples=train_examples,
    train_label_support=train_label_support,
    train_source_label_support=train_source_label_support,
    inference_package_feature_sets=inference_package_feature_sets,
    output=output,
    epochs=epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    shuffle=shuffle,
    max_steps=max_steps,
    export_embeddings=export_embeddings,
    prediction_output=prediction_output,
    include_seed_in_artifact_name=include_seed_in_artifact_name,
    taxonomy_config=taxonomy_config,
  )
  if model_artifact_output is not None:
    if training_result is None:
      raise ValueError("training_result is required for model artifact export.")
    model_result["model_artifact"] = save_model_artifact(
      model_artifact_output,
      model_name=model_name,
      model=model,
      dataset=dataset,
      model_result=model_result,
      training_result=training_result,
      seed=seed,
    )
  return model_result


def _evaluate_trained_model(
  model_name: str,
  *,
  model: Classifier,
  metadata: dict[str, Any],
  training_summary: dict[str, Any],
  seed: int,
  dataset: TrainingDataset,
  examples: list[ModelExample],
  train_examples: list[ModelExample],
  train_label_support: dict[str, int],
  train_source_label_support: dict[str, int],
  output: Path | None,
  epochs: int,
  learning_rate: float,
  batch_size: int,
  shuffle: bool,
  max_steps: int | None,
  export_embeddings: bool,
  prediction_output: str,
  include_seed_in_artifact_name: bool,
  inference_package_feature_sets: dict[str, PackageFeatureSet] | None = None,
  taxonomy_config: TaxonomyEvaluationConfig | None = None,
) -> dict[str, Any]:
  primary_evaluation_split = str(
    _split_plan(split_examples(examples))["primary_evaluation_split"]
  )
  predictions = _predict(
    model,
    examples,
    batch_size=batch_size,
    train_label_support=train_label_support,
  )
  logits = _logits_for_metrics(model, examples, batch_size=batch_size)
  calibration = calibration_metrics(
    logits=logits,
    labels=tuple(getattr(model, "labels", ())),
    examples=examples,
  )
  annotate_temperature_scaled_confidence(
    predictions,
    logits=logits,
    calibration=calibration,
  )
  metrics = classification_metrics(
    predictions,
    train_label_support=train_label_support,
  )
  source_label_evaluation = _source_label_evaluation(
    predictions,
    train_source_label_support=train_source_label_support,
  )
  taxonomy_evaluation = taxonomy_metrics(
    predictions,
    taxonomy_config or TaxonomyEvaluationConfig(path=None, aliases=()),
  )
  if calibration:
    metrics["calibration"] = calibration
  metadata = dict(metadata)
  model_config = dict(metadata.get("model_config", {}))
  model_config.setdefault("learning_rate", learning_rate)
  if "weight_decay" in metadata:
    model_config.setdefault("weight_decay", metadata["weight_decay"])
  if "temperature" not in model_config and calibration.get("status") == "completed":
    model_config["temperature"] = calibration.get("temperature")
  model_config.setdefault(
    "package_mining",
    {"enabled": metadata.get("package_feature_set") is not None},
  )
  metadata["model_config"] = model_config
  prediction_exports = _write_prediction_exports(
    output,
    model_name,
    predictions,
    seed=seed,
    prediction_output=prediction_output,
    include_seed_in_artifact_name=include_seed_in_artifact_name,
  )
  embedding_exports = _write_embedding_exports(
    output,
    model_name,
    model,
    dataset,
    examples,
    train_examples,
    seed=seed,
    batch_size=batch_size,
    export_embeddings=export_embeddings,
    include_seed_in_artifact_name=include_seed_in_artifact_name,
  )
  model_result = {
    "model_family": model_name,
    "epochs": epochs,
    "learning_rate": learning_rate,
    "seed": seed,
    "batch_size": batch_size,
    "shuffle": shuffle,
    "max_steps": max_steps,
    "prediction_output": prediction_output,
    "optimizer_steps": training_summary["optimizer_steps"],
    "completed_epochs": training_summary["completed_epochs"],
    "training_history": training_summary["history"],
    "best_validation_epoch": training_summary["best_validation_epoch"],
    "best_validation_metric": training_summary["best_validation_metric"],
    "best_validation_loss": training_summary["best_validation_loss"],
    "primary_evaluation_split": primary_evaluation_split,
    "primary_metrics_at_best_validation_epoch": metrics["splits"].get(
      primary_evaluation_split
    ),
    "metrics": metrics,
    "source_label_evaluation": source_label_evaluation,
    "prediction_pairing": _prediction_pairing(
      predictions,
      split_name=primary_evaluation_split,
    ),
    **metadata,
    **prediction_exports,
    **embedding_exports,
  }
  if taxonomy_evaluation.get("enabled"):
    model_result["taxonomy_evaluation"] = taxonomy_evaluation
  package_ablations = _evaluate_inference_package_ablations(
    model,
    normal_predictions=predictions,
    normal_metrics=metrics,
    examples=examples,
    train_label_support=train_label_support,
    batch_size=batch_size,
    package_feature_sets=inference_package_feature_sets or {},
    seed=seed,
    primary_evaluation_split=primary_evaluation_split,
  )
  if package_ablations:
    model_result["inference_package_ablations"] = package_ablations
  package_usage = _package_branch_usage_summary(
    model,
    train_examples=train_examples,
    batch_size=batch_size,
  )
  if package_usage:
    model_result["package_branch_usage"] = package_usage
  if output is None:
    model_result["predictions"] = _filter_predictions_for_output(
      predictions,
      prediction_output=prediction_output,
    )
  return model_result


def _logits_for_metrics(
  model: Classifier,
  examples: list[ModelExample],
  *,
  batch_size: int,
) -> torch.Tensor | None:
  logits_many = getattr(model, "logits_many", None)
  if logits_many is None:
    return None
  return logits_many(examples, batch_size=batch_size)


def _package_branch_usage_summary(
  model: Classifier,
  *,
  train_examples: list[ModelExample],
  batch_size: int,
) -> dict[str, Any]:
  if not isinstance(model, DeepSetsClassifier) or model.package_count <= 0:
    return {}
  return model.package_branch_usage_summary(
    train_examples,
    batch_size=batch_size,
  )


def _evaluate_inference_package_ablations(
  model: Classifier,
  *,
  normal_predictions: list[dict[str, Any]],
  normal_metrics: dict[str, Any],
  examples: list[ModelExample],
  train_label_support: dict[str, int],
  batch_size: int,
  package_feature_sets: dict[str, PackageFeatureSet],
  seed: int,
  primary_evaluation_split: str,
) -> dict[str, Any]:
  if not isinstance(model, DeepSetsClassifier) or not package_feature_sets:
    return {}
  if model.package_count <= 0:
    return {}

  original_package_features = model.package_features
  normal_pairing = _prediction_pairing(
    normal_predictions,
    split_name=primary_evaluation_split,
  )
  normal_logits = model.logits_many(examples, batch_size=batch_size)
  rows: dict[str, Any] = {
    "normal": {
      "package_feature_set": "v2",
      "split_name": primary_evaluation_split,
      "metrics": normal_metrics,
      "prediction_pairing": normal_pairing,
    }
  }
  try:
    for ablation_name, package_features in package_feature_sets.items():
      if len(package_features) != model.package_count:
        rows[ablation_name] = {
          "status": "skipped",
          "reason": (
            "Package feature count does not match the trained projection "
            f"({len(package_features)} != {model.package_count})."
          ),
        }
        continue
      model.package_features = package_features
      predictions = _predict(
        model,
        examples,
        batch_size=batch_size,
        train_label_support=train_label_support,
      )
      ablation_logits = model.logits_many(examples, batch_size=batch_size)
      pairing = _prediction_pairing(
        predictions,
        split_name=primary_evaluation_split,
      )
      rows[ablation_name] = {
        "status": "completed",
        "split_name": primary_evaluation_split,
        "metrics": classification_metrics(
          predictions,
          train_label_support=train_label_support,
        ),
        "prediction_pairing": pairing,
        "paired_vs_normal": _paired_correctness_report(
          normal_pairing,
          pairing,
          seed=seed,
        ),
        "logit_delta_vs_normal": _logit_delta_summary(
          normal_logits,
          ablation_logits,
          examples,
        ),
      }
  finally:
    model.package_features = original_package_features

  return rows


def _prediction_pairing(
  predictions: list[dict[str, Any]],
  *,
  split_name: str,
) -> list[dict[str, Any]]:
  return [
    {
      "deck_id": str(prediction["deck_id"]),
      "is_correct": bool(prediction.get("is_correct")),
    }
    for prediction in predictions
    if prediction.get("split_name") == split_name
  ]


def _logit_delta_summary(
  reference_logits: torch.Tensor,
  comparison_logits: torch.Tensor,
  examples: list[ModelExample],
) -> dict[str, Any]:
  if reference_logits.shape != comparison_logits.shape:
    return {
      "status": "skipped",
      "reason": (
        "Logit shapes differ: "
        f"{tuple(reference_logits.shape)} != {tuple(comparison_logits.shape)}."
      ),
    }
  indexes_by_split: dict[str, list[int]] = {"overall": list(range(len(examples)))}
  for index, example in enumerate(examples):
    indexes_by_split.setdefault(example.split_name, []).append(index)

  return {
    split_name: _logit_delta_block(
      reference_logits.index_select(0, torch.tensor(indexes, dtype=torch.long)),
      comparison_logits.index_select(0, torch.tensor(indexes, dtype=torch.long)),
    )
    for split_name, indexes in sorted(indexes_by_split.items())
    if indexes
  }


def _logit_delta_block(
  reference_logits: torch.Tensor,
  comparison_logits: torch.Tensor,
) -> dict[str, Any]:
  if reference_logits.numel() == 0:
    return {
      "count": 0,
      "mean_abs_logit_delta": None,
      "max_abs_logit_delta": None,
      "mean_abs_probability_delta": None,
      "max_abs_probability_delta": None,
      "top_1_changed_count": 0,
      "top_1_changed_rate": None,
      "mean_reference_top_rank_under_comparison": None,
      "max_reference_top_rank_under_comparison": None,
    }

  logit_delta = (reference_logits - comparison_logits).abs()
  reference_probabilities = torch.softmax(reference_logits, dim=1)
  comparison_probabilities = torch.softmax(comparison_logits, dim=1)
  probability_delta = (reference_probabilities - comparison_probabilities).abs()
  reference_top = reference_logits.argmax(dim=1)
  comparison_top = comparison_logits.argmax(dim=1)
  reference_top_comparison_logits = comparison_logits.gather(
    1,
    reference_top.unsqueeze(1),
  )
  reference_top_rank_under_comparison = (
    (comparison_logits > reference_top_comparison_logits).sum(dim=1) + 1
  )
  top_1_changed = reference_top != comparison_top
  count = int(reference_logits.shape[0])

  return {
    "count": count,
    "mean_abs_logit_delta": float(logit_delta.mean().item()),
    "max_abs_logit_delta": float(logit_delta.max().item()),
    "mean_abs_probability_delta": float(probability_delta.mean().item()),
    "max_abs_probability_delta": float(probability_delta.max().item()),
    "top_1_changed_count": int(top_1_changed.sum().item()),
    "top_1_changed_rate": float(top_1_changed.float().mean().item()),
    "mean_reference_top_rank_under_comparison": float(
      reference_top_rank_under_comparison.float().mean().item()
    ),
    "max_reference_top_rank_under_comparison": int(
      reference_top_rank_under_comparison.max().item()
    ),
  }


def _paired_correctness_report(
  reference_pairing: list[dict[str, Any]],
  comparison_pairing: list[dict[str, Any]],
  *,
  seed: int,
  bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
  reference_by_deck = {
    str(row["deck_id"]): bool(row["is_correct"])
    for row in reference_pairing
  }
  comparison_by_deck = {
    str(row["deck_id"]): bool(row["is_correct"])
    for row in comparison_pairing
  }
  deck_ids = sorted(set(reference_by_deck) & set(comparison_by_deck))
  if not deck_ids:
    return {
      "count": 0,
      "reference_accuracy": None,
      "comparison_accuracy": None,
      "accuracy_delta": None,
      "bootstrap_accuracy_delta_ci_95": None,
      "mcnemar_chi_square": None,
      "mcnemar_p_value_approx": None,
      "reference_only_correct": 0,
      "comparison_only_correct": 0,
      "both_correct": 0,
      "both_wrong": 0,
    }

  reference_correct = [reference_by_deck[deck_id] for deck_id in deck_ids]
  comparison_correct = [comparison_by_deck[deck_id] for deck_id in deck_ids]
  deltas = [
    (1 if reference else 0) - (1 if comparison else 0)
    for reference, comparison in zip(
      reference_correct,
      comparison_correct,
      strict=True,
    )
  ]
  reference_only = sum(
    1
    for reference, comparison in zip(
      reference_correct,
      comparison_correct,
      strict=True,
    )
    if reference and not comparison
  )
  comparison_only = sum(
    1
    for reference, comparison in zip(
      reference_correct,
      comparison_correct,
      strict=True,
    )
    if comparison and not reference
  )
  both_correct = sum(
    1
    for reference, comparison in zip(
      reference_correct,
      comparison_correct,
      strict=True,
    )
    if reference and comparison
  )
  both_wrong = sum(
    1
    for reference, comparison in zip(
      reference_correct,
      comparison_correct,
      strict=True,
    )
    if not reference and not comparison
  )
  delta = sum(deltas) / len(deltas)
  bootstrap_ci = _bootstrap_mean_ci(
    deltas,
    seed=seed,
    iterations=bootstrap_iterations,
  )
  discordant = reference_only + comparison_only
  if discordant:
    chi_square = ((abs(reference_only - comparison_only) - 1) ** 2) / discordant
    p_value = math.erfc(math.sqrt(chi_square / 2))
  else:
    chi_square = None
    p_value = None

  return {
    "count": len(deck_ids),
    "reference_accuracy": sum(reference_correct) / len(deck_ids),
    "comparison_accuracy": sum(comparison_correct) / len(deck_ids),
    "accuracy_delta": delta,
    "bootstrap_accuracy_delta_ci_95": bootstrap_ci,
    "mcnemar_chi_square": chi_square,
    "mcnemar_p_value_approx": p_value,
    "reference_only_correct": reference_only,
    "comparison_only_correct": comparison_only,
    "both_correct": both_correct,
    "both_wrong": both_wrong,
  }


def _bootstrap_mean_ci(
  values: list[int],
  *,
  seed: int,
  iterations: int,
) -> list[float] | None:
  if not values or iterations <= 0:
    return None

  rng = random.Random(seed)
  bootstrapped: list[float] = []
  for _ in range(iterations):
    total = 0
    for _ in values:
      total += values[rng.randrange(len(values))]
    bootstrapped.append(total / len(values))
  bootstrapped.sort()
  lower_idx = int(0.025 * (len(bootstrapped) - 1))
  upper_idx = int(0.975 * (len(bootstrapped) - 1))
  return [bootstrapped[lower_idx], bootstrapped[upper_idx]]


def _run_quantity_weighted_deepsets_training(
  *,
  seed: int,
  dataset: TrainingDataset,
  examples: list[ModelExample],
  train_examples: list[ModelExample],
  validation_examples: list[ModelExample],
  train_label_support: dict[str, int],
  train_source_label_support: dict[str, int],
  quantity_count: int,
  package_features: PackageFeatureSet,
  package_feature_set_name: str | None,
  output: Path | None,
  epochs: int,
  learning_rate: float,
  weight_decay: float,
  embedding_dim: int,
  hidden_dim: int,
  head_v2_rho_hidden_dim: int,
  batch_size: int,
  shuffle: bool,
  max_steps: int | None,
  device: str,
  export_embeddings: bool,
  prediction_output: str,
  include_prototype: bool,
  include_seed_in_artifact_name: bool,
  package_scale: float,
  package_projection_dim: int,
  architecture_package_count: int,
  taxonomy_config: TaxonomyEvaluationConfig,
) -> dict[str, dict[str, Any]]:
  model, metadata = _build_classifier(
    MODEL_DEEPSETS_QUANTITY_WEIGHTED,
    labels=dataset.labels,
    card_count=dataset.card_count,
    zone_count=dataset.zone_count,
    quantity_count=quantity_count,
    package_features=package_features,
    package_feature_set_name=package_feature_set_name,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    seed=seed,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    head_v2_rho_hidden_dim=head_v2_rho_hidden_dim,
    attention_heads=4,
    attention_layers=2,
    batch_size=batch_size,
    shuffle=shuffle,
    device=device,
    package_scale=package_scale,
    package_projection_dim=package_projection_dim,
    architecture_package_count=architecture_package_count,
  )
  if not isinstance(model, DeepSetsClassifier):
    raise AssertionError("Expected quantity-weighted Deep Sets classifier.")

  training_summary = model.fit(
    train_examples,
    validation_examples=validation_examples,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=shuffle,
    max_steps=max_steps,
  )
  softmax_result = _evaluate_trained_model(
    MODEL_DEEPSETS_QUANTITY_WEIGHTED,
    model=model,
    metadata=metadata,
    training_summary=training_summary,
    seed=seed,
    dataset=dataset,
    examples=examples,
    train_examples=train_examples,
    train_label_support=train_label_support,
    train_source_label_support=train_source_label_support,
    output=output,
    epochs=epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    shuffle=shuffle,
    max_steps=max_steps,
    export_embeddings=export_embeddings,
    prediction_output=prediction_output,
    include_seed_in_artifact_name=include_seed_in_artifact_name,
    taxonomy_config=taxonomy_config,
  )
  results = {MODEL_DEEPSETS_QUANTITY_WEIGHTED: softmax_result}
  if not include_prototype:
    return results

  prototype_model = PrototypeClassifier.from_deepsets(
    model,
    train_examples,
    batch_size=batch_size,
  )
  prototype_metadata = {
    **metadata,
    "head": "prototype",
    "encoder_model_family": MODEL_DEEPSETS_QUANTITY_WEIGHTED,
    "prototype_distance": prototype_model.distance,
    "prototype_count": len(prototype_model.labels),
    "prototype_counts": prototype_model.prototype_counts,
  }
  prototype_result = _evaluate_trained_model(
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE,
    model=prototype_model,
    metadata=prototype_metadata,
    training_summary=training_summary,
    seed=seed,
    dataset=dataset,
    examples=examples,
    train_examples=train_examples,
    train_label_support=train_label_support,
    train_source_label_support=train_source_label_support,
    output=output,
    epochs=epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    shuffle=shuffle,
    max_steps=max_steps,
    export_embeddings=False,
    prediction_output=prediction_output,
    include_seed_in_artifact_name=include_seed_in_artifact_name,
    taxonomy_config=taxonomy_config,
  )
  results[MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE] = prototype_result
  return results


def _model_result_from_seed_runs(
  model_name: str,
  seed_runs: list[dict[str, Any]],
  *,
  seeds: tuple[int, ...],
  primary_evaluation_split: str,
) -> dict[str, Any]:
  if not seed_runs:
    return {
      "model_family": model_name,
      "seeds": list(seeds),
      "seed_runs": [],
      "primary_evaluation_split": primary_evaluation_split,
      "multi_seed_summary": {},
    }

  result = dict(seed_runs[0])
  result["seeds"] = list(seeds)
  result["primary_seed"] = seeds[0]
  result["primary_evaluation_split"] = primary_evaluation_split
  result["seed_runs"] = seed_runs
  result["multi_seed_summary"] = _multi_seed_summary(
    seed_runs,
    primary_evaluation_split=primary_evaluation_split,
  )
  return result


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


def _build_classifier(
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
      metadata["architecture_reference_package_count"] = architecture_package_count
      metadata["architecture_reference_projection_dim"] = package_projection_dim
    return (classifier, metadata)
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
        "quantity_count": quantity_count,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "device": str(classifier.device),
      },
    )
  raise AssertionError(f"Unhandled model: {model_name}")


def _predict(
  model: Classifier,
  examples: list[ModelExample],
  *,
  batch_size: int,
  train_label_support: dict[str, int],
) -> list[dict[str, Any]]:
  if hasattr(model, "predict_top_k_with_stats_many"):
    prediction_stats_by_example = model.predict_top_k_with_stats_many(
      examples,
      k=3,
      batch_size=batch_size,
    )
  else:
    prediction_stats_by_example = [
      {
        "top_predictions": model.predict_top_k(example, k=3),
        "max_probability": None,
        "entropy": None,
        "normalized_entropy": None,
        "energy": None,
      }
      for example in examples
    ]

  rows: list[dict[str, Any]] = []
  for example, prediction_stats in zip(
    examples,
    prediction_stats_by_example,
    strict=True,
  ):
    top_predictions = prediction_stats["top_predictions"]
    predicted_label_id = top_predictions[0][0] if top_predictions else None
    score = top_predictions[0][1] if top_predictions else 0.0
    top_label_ids = [label_id for label_id, _ in top_predictions]
    actual_label_train_support = train_label_support.get(example.target_label_id, 0)
    rows.append(
      {
        "deck_id": example.deck_id,
        "event_id": example.event_id,
        "event_date": (
          example.event_date.isoformat()
          if example.event_date is not None
          else None
        ),
        "format": example.format_code,
        "split_name": example.split_name,
        "target_source": example.target_source,
        "actual_label_id": example.target_label_id,
        "source_actual_label_id": example.source_label_id,
        "actual_label_train_support": actual_label_train_support,
        "is_train_unseen_label": actual_label_train_support == 0,
        "predicted_label_id": predicted_label_id,
        "score": score,
        "max_softmax_probability": prediction_stats.get(
          "max_probability",
          score,
        ),
        "entropy": prediction_stats.get("entropy"),
        "normalized_entropy": prediction_stats.get("normalized_entropy"),
        "energy": prediction_stats.get("energy"),
        "nearest_prototype_distance": prediction_stats.get(
          "nearest_prototype_distance"
        ),
        "prototype_margin": prediction_stats.get("prototype_margin"),
        "top_prototype_distances": prediction_stats.get(
          "top_prototype_distances",
          [],
        ),
        "top_label_ids": top_label_ids,
        "top_scores": [score for _, score in top_predictions],
        "is_correct": predicted_label_id == example.target_label_id,
        "is_top_3_correct": example.target_label_id in top_label_ids,
      }
    )

  return rows


def _source_label_evaluation(
  predictions: list[dict[str, Any]],
  *,
  train_source_label_support: dict[str, int],
) -> dict[str, Any]:
  source_predictions: list[dict[str, Any]] = []
  for prediction in predictions:
    source_actual = str(
      prediction.get("source_actual_label_id")
      or prediction.get("actual_label_id")
    )
    predicted = prediction.get("predicted_label_id")
    top_label_ids = [
      str(label_id)
      for label_id in prediction.get("top_label_ids", [])
      if label_id is not None
    ]
    support = train_source_label_support.get(source_actual, 0)
    row = dict(prediction)
    row["training_actual_label_id"] = prediction.get("actual_label_id")
    row["actual_label_id"] = source_actual
    row["actual_label_train_support"] = support
    row["is_train_unseen_label"] = support == 0
    row["is_correct"] = predicted == source_actual
    row["is_top_3_correct"] = source_actual in top_label_ids
    source_predictions.append(row)

  return {
    "enabled": True,
    "label_level": "source_label",
    "train_label_support": train_source_label_support,
    "metrics": classification_metrics(
      source_predictions,
      train_label_support=train_source_label_support,
    ),
  }


def _write_prediction_exports(
  output: Path | None,
  model_name: str,
  predictions: list[dict[str, Any]],
  *,
  seed: int,
  prediction_output: str,
  include_seed_in_artifact_name: bool,
) -> dict[str, Any]:
  if output is None or prediction_output == PREDICTION_OUTPUT_SUMMARY:
    return {
      "prediction_output": prediction_output,
      "prediction_export_count": 0,
    }

  artifact_name = _artifact_stem(
    model_name,
    seed=seed,
    include_seed=include_seed_in_artifact_name,
  )
  prediction_path = output.parent / "predictions" / f"{artifact_name}.jsonl"
  exported_predictions = _filter_predictions_for_output(
    predictions,
    prediction_output=prediction_output,
  )
  _write_jsonl(prediction_path, exported_predictions)
  return {
    "prediction_output": prediction_output,
    "prediction_path": str(prediction_path),
    "prediction_export_count": len(exported_predictions),
  }


def _filter_predictions_for_output(
  predictions: list[dict[str, Any]],
  *,
  prediction_output: str,
) -> list[dict[str, Any]]:
  if prediction_output == PREDICTION_OUTPUT_SUMMARY:
    return []
  if prediction_output == PREDICTION_OUTPUT_FULL:
    return predictions
  if prediction_output == PREDICTION_OUTPUT_ERRORS:
    return [
      prediction
      for prediction in predictions
      if (
        not prediction.get("is_correct")
        or prediction.get("is_train_unseen_label")
      )
    ]
  raise AssertionError(f"Unhandled prediction output mode: {prediction_output}")


def _write_embedding_exports(
  output: Path | None,
  model_name: str,
  model: Classifier,
  dataset: TrainingDataset,
  examples: list[ModelExample],
  train_examples: list[ModelExample],
  *,
  seed: int,
  batch_size: int,
  export_embeddings: bool,
  include_seed_in_artifact_name: bool,
) -> dict[str, Any]:
  if (
    output is None
    or not export_embeddings
    or not isinstance(model, (DeepSetsClassifier, SetTransformerClassifier))
  ):
    return {}

  embedding_dir = output.parent / "embeddings"
  artifact_name = _artifact_stem(
    model_name,
    seed=seed,
    include_seed=include_seed_in_artifact_name,
  )
  card_embedding_path = embedding_dir / f"{artifact_name}_cards.jsonl"
  deck_embedding_path = embedding_dir / f"{artifact_name}_decks.jsonl"

  _write_jsonl(card_embedding_path, model.card_embedding_rows(dataset.card_vocab))
  _write_jsonl(
    deck_embedding_path,
    model.deck_embedding_rows(examples, batch_size=batch_size),
  )
  card_support = _card_support(train_examples)
  return {
    "card_embedding_path": str(card_embedding_path),
    "deck_embedding_path": str(deck_embedding_path),
    "learned_card_nearest_neighbors": model.card_nearest_neighbors(
      dataset.card_vocab,
      anchor_card_idxs=list(card_support)[:20],
      card_support=card_support,
    ),
  }


def _artifact_stem(
  model_name: str,
  *,
  seed: int,
  include_seed: bool,
) -> str:
  if not include_seed:
    return model_name
  return f"{model_name}_seed_{seed}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as output_file:
    for row in rows:
      output_file.write(json.dumps(row, sort_keys=True) + "\n")


def _training_skip_reason(
  train_examples: list[ModelExample],
  evaluation_examples: list[ModelExample],
  labels: tuple[str, ...],
) -> str | None:
  if not labels:
    return "No proxy-target labels are available."
  if not train_examples:
    return "No train split proxy-target examples are available."
  if not evaluation_examples:
    return "No evaluation split proxy-target examples are available."
  return None


def _split_plan(grouped: dict[str, list[ModelExample]]) -> dict[str, Any]:
  names = sorted(
    grouped,
    key=lambda split_name: (SPLIT_ORDER_INDEX.get(split_name, 10_000), split_name),
  )
  evaluation_splits = tuple(
    split_name
    for split_name in names
    if split_name != "train"
  )
  primary_evaluation_split = next(
    (
      split_name
      for split_name in PRIMARY_EVALUATION_SPLITS
      if grouped.get(split_name)
    ),
    evaluation_splits[0] if evaluation_splits else "test",
  )
  return {
    "counts": {
      split_name: len(grouped[split_name])
      for split_name in names
    },
    "evaluation_splits": evaluation_splits,
    "primary_evaluation_split": primary_evaluation_split,
  }


def _normalize_seeds(seed: int, seeds: tuple[int, ...] | None) -> tuple[int, ...]:
  if seeds is None:
    return (seed,)
  normalized = tuple(dict.fromkeys(seeds))
  if not normalized:
    raise ValueError("At least one seed must be provided.")
  return normalized


def _label_support(examples: list[ModelExample]) -> dict[str, int]:
  rows: dict[str, int] = {}
  for example in examples:
    rows[example.target_label_id] = rows.get(example.target_label_id, 0) + 1
  return dict(sorted(rows.items()))


def _source_label_support(examples: list[ModelExample]) -> dict[str, int]:
  rows: dict[str, int] = {}
  for example in examples:
    rows[example.source_label_id] = rows.get(example.source_label_id, 0) + 1
  return dict(sorted(rows.items()))


def _card_support(examples: list[ModelExample]) -> dict[int, int]:
  rows: dict[int, int] = {}
  for example in examples:
    for card_idx in {token.card_idx for token in example.tokens}:
      rows[card_idx] = rows.get(card_idx, 0) + 1

  return dict(
    sorted(
      rows.items(),
      key=lambda item: (-item[1], item[0]),
    )
  )


def _multi_seed_summary(
  seed_runs: list[dict[str, Any]],
  *,
  primary_evaluation_split: str,
) -> dict[str, Any]:
  metric_paths = {
    "primary_accuracy": (
      "metrics",
      "splits",
      primary_evaluation_split,
      "accuracy",
    ),
    "primary_closed_set_accuracy": (
      "metrics",
      "evaluation_views",
      primary_evaluation_split,
      "closed_set_seen_labels",
      "accuracy",
    ),
    "primary_top_3_accuracy": (
      "metrics",
      "splits",
      primary_evaluation_split,
      "top_3_accuracy",
    ),
    "primary_macro_f1": (
      "metrics",
      "splits",
      primary_evaluation_split,
      "macro_f1",
    ),
    "primary_source_label_accuracy": (
      "source_label_evaluation",
      "metrics",
      "splits",
      primary_evaluation_split,
      "accuracy",
    ),
    "primary_source_label_closed_set_accuracy": (
      "source_label_evaluation",
      "metrics",
      "evaluation_views",
      primary_evaluation_split,
      "closed_set_seen_labels",
      "accuracy",
    ),
    "primary_source_label_top_3_accuracy": (
      "source_label_evaluation",
      "metrics",
      "splits",
      primary_evaluation_split,
      "top_3_accuracy",
    ),
    "primary_source_label_macro_f1": (
      "source_label_evaluation",
      "metrics",
      "splits",
      primary_evaluation_split,
      "macro_f1",
    ),
    "primary_canonical_accuracy": (
      "taxonomy_evaluation",
      "metrics",
      "splits",
      primary_evaluation_split,
      "accuracy",
    ),
    "primary_canonical_closed_set_accuracy": (
      "taxonomy_evaluation",
      "metrics",
      "evaluation_views",
      primary_evaluation_split,
      "closed_set_seen_labels",
      "accuracy",
    ),
    "primary_canonical_top_3_accuracy": (
      "taxonomy_evaluation",
      "metrics",
      "splits",
      primary_evaluation_split,
      "top_3_accuracy",
    ),
    "primary_canonical_macro_f1": (
      "taxonomy_evaluation",
      "metrics",
      "splits",
      primary_evaluation_split,
      "macro_f1",
    ),
    "primary_temperature_scaled_nll": (
      "metrics",
      "calibration",
      "temperature_scaled",
      "splits",
      primary_evaluation_split,
      "nll",
    ),
    "primary_temperature_scaled_brier": (
      "metrics",
      "calibration",
      "temperature_scaled",
      "splits",
      primary_evaluation_split,
      "brier",
    ),
    "primary_temperature_scaled_ece": (
      "metrics",
      "calibration",
      "temperature_scaled",
      "splits",
      primary_evaluation_split,
      "ece",
    ),
    "primary_msp_auroc": (
      "metrics",
      "abstention",
      "max_softmax_probability",
      "splits",
      primary_evaluation_split,
      "auroc",
    ),
    "primary_temperature_scaled_msp_auroc": (
      "metrics",
      "abstention",
      "temperature_scaled_max_softmax_probability",
      "splits",
      primary_evaluation_split,
      "auroc",
    ),
    "primary_energy_auroc": (
      "metrics",
      "abstention",
      "energy",
      "splits",
      primary_evaluation_split,
      "auroc",
    ),
    "primary_known_false_abstention_rate": (
      "metrics",
      "abstention",
      "max_softmax_probability",
      "splits",
      primary_evaluation_split,
      "known_false_abstention_rate",
    ),
    "primary_unknown_recall": (
      "metrics",
      "abstention",
      "max_softmax_probability",
      "splits",
      primary_evaluation_split,
      "unknown_recall",
    ),
    "primary_energy_known_false_abstention_rate": (
      "metrics",
      "abstention",
      "energy",
      "splits",
      primary_evaluation_split,
      "known_false_abstention_rate",
    ),
    "primary_energy_unknown_recall": (
      "metrics",
      "abstention",
      "energy",
      "splits",
      primary_evaluation_split,
      "unknown_recall",
    ),
    "primary_nearest_prototype_distance_auroc": (
      "metrics",
      "abstention",
      "nearest_prototype_distance",
      "splits",
      primary_evaluation_split,
      "auroc",
    ),
    "primary_nearest_prototype_distance_known_false_abstention_rate": (
      "metrics",
      "abstention",
      "nearest_prototype_distance",
      "splits",
      primary_evaluation_split,
      "known_false_abstention_rate",
    ),
    "primary_prototype_margin_auroc": (
      "metrics",
      "abstention",
      "prototype_margin",
      "splits",
      primary_evaluation_split,
      "auroc",
    ),
    "primary_prototype_margin_known_false_abstention_rate": (
      "metrics",
      "abstention",
      "prototype_margin",
      "splits",
      primary_evaluation_split,
      "known_false_abstention_rate",
    ),
    "temperature": ("metrics", "calibration", "temperature"),
  }
  summary: dict[str, Any] = {}
  for metric_name, path in metric_paths.items():
    summary[metric_name] = _stats(
      [
        _nested_value(seed_run, path)
        for seed_run in seed_runs
      ]
    )
  return summary


def _nested_value(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
  current: Any = row
  for key in path:
    if not isinstance(current, dict) or key not in current:
      return None
    current = current[key]
  if current is None:
    return None
  return float(current)


def _stats(values: list[float | None]) -> dict[str, Any]:
  numeric_values = [
    value
    for value in values
    if value is not None
  ]
  if not numeric_values:
    return {
      "count": 0,
      "mean": None,
      "std": None,
      "values": values,
    }

  mean = sum(numeric_values) / len(numeric_values)
  variance = (
    sum((value - mean) ** 2 for value in numeric_values) / len(numeric_values)
  )
  return {
    "count": len(numeric_values),
    "mean": mean,
    "std": math.sqrt(variance),
    "values": values,
  }


def _accuracy_comparison(
  pooled_linear_metrics: dict[str, Any],
  model_metrics: dict[str, Any],
) -> dict[str, Any]:
  rows: dict[str, Any] = {}
  split_names = sorted(
    set(pooled_linear_metrics.get("splits", {}))
    | set(model_metrics.get("splits", {})),
    key=lambda split_name: (SPLIT_ORDER_INDEX.get(split_name, 10_000), split_name),
  )
  for split_name in split_names:
    pooled_linear_accuracy = _split_accuracy(pooled_linear_metrics, split_name)
    model_accuracy = _split_accuracy(model_metrics, split_name)
    rows[split_name] = {
      "pooled_linear_accuracy": pooled_linear_accuracy,
      "model_accuracy": model_accuracy,
      "accuracy_delta": (
        model_accuracy - pooled_linear_accuracy
        if model_accuracy is not None and pooled_linear_accuracy is not None
        else None
      ),
      "closed_set_seen_labels": _view_accuracy_comparison(
        pooled_linear_metrics,
        model_metrics,
        split_name,
        "closed_set_seen_labels",
      ),
      "open_set_unseen_labels": _view_accuracy_comparison(
        pooled_linear_metrics,
        model_metrics,
        split_name,
        "open_set_unseen_labels",
      ),
    }

  return rows


def _paired_model_tests(result: dict[str, Any]) -> dict[str, Any]:
  reference_name = MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES
  if reference_name not in result["models"]:
    return {}

  comparisons = (
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES,
  )
  reference = result["models"][reference_name]
  reference_pairing = reference.get("prediction_pairing", [])
  split_name = str(result["primary_evaluation_split"])
  rows: dict[str, Any] = {}
  for comparison_name in comparisons:
    comparison = result["models"].get(comparison_name)
    if comparison is None:
      continue
    rows[f"{reference_name}__vs__{comparison_name}"] = {
      "split_name": split_name,
      "seed": result.get("primary_seed", result.get("seeds", [None])[0]),
      "reference_model": reference_name,
      "comparison_model": comparison_name,
      "paired_correctness": _paired_correctness_report(
        reference_pairing,
        comparison.get("prediction_pairing", []),
        seed=int(reference.get("primary_seed", reference.get("seed", 13))),
      ),
    }
  return rows


def _package_signal_by_label(
  result: dict[str, Any],
  *,
  train_label_support: dict[str, int],
  evaluation_examples: list[ModelExample],
  evaluation_split: str,
  package_features: PackageFeatureSet,
  card_vocab: tuple[dict[str, Any], ...],
  zone_vocab: dict[str, int],
) -> list[dict[str, Any]]:
  required_models = (
    MODEL_POOLED_LINEAR,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES,
  )
  if any(model_name not in result["models"] for model_name in required_models):
    return []

  per_label = {
    model_name: _per_label_accuracy_by_id(
      result["models"][model_name]["metrics"],
      split_name=evaluation_split,
    )
    for model_name in required_models
  }
  top_packages = _top_packages_by_label(
    evaluation_examples,
    package_features=package_features,
    card_vocab=card_vocab,
    zone_vocab=zone_vocab,
  )
  labels = sorted(
    set(per_label[MODEL_POOLED_LINEAR])
    | set(per_label[MODEL_DEEPSETS_QUANTITY_WEIGHTED])
    | set(per_label[MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES])
  )
  rows: list[dict[str, Any]] = []
  for label in labels:
    pooled = per_label[MODEL_POOLED_LINEAR].get(label, {})
    deepsets = per_label[MODEL_DEEPSETS_QUANTITY_WEIGHTED].get(label, {})
    packages = per_label[MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES].get(label, {})
    deepsets_accuracy = deepsets.get("accuracy")
    package_accuracy = packages.get("accuracy")
    pooled_accuracy = pooled.get("accuracy")
    rows.append(
      {
        "label_id": label,
        "evaluation_split": evaluation_split,
        "train_support": train_label_support.get(label, 0),
        "evaluation_support": packages.get(
          "count",
          deepsets.get("count", pooled.get("count", 0)),
        ),
        "pooled_linear_accuracy": pooled_accuracy,
        "deepsets_quantity_weighted_accuracy": deepsets_accuracy,
        "deepsets_quantity_weighted_packages_accuracy": package_accuracy,
        "package_delta_vs_deepsets": (
          package_accuracy - deepsets_accuracy
          if package_accuracy is not None and deepsets_accuracy is not None
          else None
        ),
        "package_delta_vs_pooled_linear": (
          package_accuracy - pooled_accuracy
          if package_accuracy is not None and pooled_accuracy is not None
          else None
        ),
        "top_active_packages": top_packages.get(label, []),
      }
    )

  rows.sort(
    key=lambda row: (
      -_none_to_negative_infinity(row["package_delta_vs_deepsets"]),
      -int(row["evaluation_support"] or 0),
      row["label_id"],
    )
  )
  return rows


def _per_label_accuracy_by_id(
  metrics: dict[str, Any],
  *,
  split_name: str,
) -> dict[str, dict[str, Any]]:
  rows = metrics.get("splits", {}).get(split_name, {}).get("per_label_accuracy", [])
  return {
    str(row["label_id"]): row
    for row in rows
  }


def _top_packages_by_label(
  test_examples: list[ModelExample],
  *,
  package_features: PackageFeatureSet,
  card_vocab: tuple[dict[str, Any], ...],
  zone_vocab: dict[str, int],
  limit: int = 5,
) -> dict[str, list[dict[str, Any]]]:
  if not package_features:
    return {}

  counts_by_label: dict[str, dict[int, int]] = {}
  support_by_label: dict[str, int] = {}
  for example in test_examples:
    label = example.target_label_id
    support_by_label[label] = support_by_label.get(label, 0) + 1
    label_counts = counts_by_label.setdefault(label, {})
    for package_idx in package_features.activation_indexes(example):
      label_counts[package_idx] = label_counts.get(package_idx, 0) + 1

  card_name_by_idx = {
    int(card["card_idx"]): str(card.get("primary_name") or card["card_idx"])
    for card in card_vocab
  }
  zone_by_idx = {
    int(zone_idx): str(zone_name)
    for zone_name, zone_idx in zone_vocab.items()
  }
  features_by_idx = {
    feature.package_idx: feature
    for feature in package_features.features
  }
  rows: dict[str, list[dict[str, Any]]] = {}
  for label, package_counts in counts_by_label.items():
    label_support = support_by_label[label]
    rows[label] = [
      _package_label_row(
        features_by_idx[package_idx],
        activation_count=count,
        label_support=label_support,
        card_name_by_idx=card_name_by_idx,
        zone_by_idx=zone_by_idx,
      )
      for package_idx, count in sorted(
        package_counts.items(),
        key=lambda item: (-item[1], item[0]),
      )[:limit]
      if package_idx in features_by_idx
    ]
  return rows


def _package_label_row(
  feature: Any,
  *,
  activation_count: int,
  label_support: int,
  card_name_by_idx: dict[int, str],
  zone_by_idx: dict[int, str],
) -> dict[str, Any]:
  row = feature.to_dict()
  row["zone"] = zone_by_idx.get(feature.zone_idx, str(feature.zone_idx))
  row["card_names"] = [
    card_name_by_idx.get(card_idx, str(card_idx))
    for card_idx in feature.card_idxs
  ]
  row["label_activation_count"] = activation_count
  row["label_activation_rate"] = (
    activation_count / label_support
    if label_support
    else None
  )
  return row


def _none_to_negative_infinity(value: float | None) -> float:
  return float("-inf") if value is None else float(value)


def _split_accuracy(
  metrics: dict[str, Any],
  split_name: str,
) -> float | None:
  split_metrics = metrics.get("splits", {}).get(split_name)
  if split_metrics is None:
    return None
  return split_metrics.get("accuracy")


def _view_accuracy_comparison(
  pooled_linear_metrics: dict[str, Any],
  model_metrics: dict[str, Any],
  split_name: str,
  view_name: str,
) -> dict[str, float | None]:
  pooled_linear_accuracy = _view_accuracy(
    pooled_linear_metrics,
    split_name,
    view_name,
  )
  model_accuracy = _view_accuracy(
    model_metrics,
    split_name,
    view_name,
  )
  return {
    "pooled_linear_accuracy": pooled_linear_accuracy,
    "model_accuracy": model_accuracy,
    "accuracy_delta": (
      model_accuracy - pooled_linear_accuracy
      if model_accuracy is not None and pooled_linear_accuracy is not None
      else None
    ),
  }


def _view_accuracy(
  metrics: dict[str, Any],
  split_name: str,
  view_name: str,
) -> float | None:
  split_views = metrics.get("evaluation_views", {}).get(split_name, {})
  view_metrics = split_views.get(view_name)
  if view_metrics is None:
    return None
  return view_metrics.get("accuracy")


def _write_result(
  result: dict[str, Any],
  output: Path | None,
) -> dict[str, Any]:
  if output is None:
    return result

  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return result

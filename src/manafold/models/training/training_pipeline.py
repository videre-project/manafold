from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from manafold.datasets.mtgo.build import SOURCE_ARCHETYPE_PROXY
from manafold.datasets.model_inputs import (
  DeckModelInput,
  TrainingDataset,
  load_training_dataset,
  group_inputs_by_split,
)
from manafold.models.classifiers import (
  DEEPSETS_ARCHITECTURE_PLUSPLUS,
  PARTIAL_CORRUPTION_FIXED,
  PARTIAL_CORRUPTION_MIXTURE,
  POOLING_MEAN,
  POOLING_MODES,
  POOLING_QUANTITY_WEIGHTED,
  POOLING_SUM,
  SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
  SET_TRANSFORMER_POOLING_PMA,
  SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED,
  TRAINING_SAMPLING_NATURAL,
  TRAINING_SAMPLING_NATURAL_SQRT_BALANCED,
  Classifier,
  DeepSetsClassifier,
  PooledLinearClassifier,
  PrototypeClassifier,
  SetTransformerClassifier,
  max_quantity,
)
from manafold.models.evaluation.prediction_metrics import (
  annotate_temperature_scaled_confidence,
  calibration_metrics,
  classification_metrics,
)
from manafold.models.evaluation.taxonomy_metrics import taxonomy_metrics
from manafold.taxonomy.family_backoff import (
  build_family_mapping,
  display_label_from_id,
)
from manafold.models.saved_model import save_trained_model
from manafold.models.training.classifier_factory import create_classifier
from manafold.models.evaluation.summary_statistics import summary_statistics
from manafold.models.training.experiment_reports import (
  _paired_correctness_report,
  _bootstrap_mean_ci,
  _multi_seed_summary,
  _nested_value,
  _accuracy_comparison,
  _paired_model_tests,
  _package_signal_by_label,
  _per_label_accuracy_by_id,
  _top_packages_by_label,
  _package_label_row,
  _none_to_negative_infinity,
  _split_accuracy,
  _view_accuracy_comparison,
  _view_accuracy,
  _write_result,
)
from manafold.models.features.card_packages import (
  DEFAULT_PACKAGE_TYPES,
  PACKAGE_SCORING_BAYESIAN_LOG_ODDS,
  PACKAGE_SCORING_MODES,
  PACKAGE_SCORING_SUPPORT_LIFT,
  PACKAGE_TYPES,
  PackageFeatureSet,
  mine_package_features,
  mine_support_matched_unscored_package_features,
)
from manafold.taxonomy import (
  TaxonomyEvaluationConfig,
  canonical_label_id,
  load_taxonomy_evaluation_config,
)

from manafold.models.training.model_names import (
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
  MODEL_A1_5,
  MODEL_A15,
  MODEL_A2PP,
  MODEL_A2_PP,
  MODEL_A2,
  MODEL_A3,
  MODEL_A6,
  MODEL_A8,
  MODEL_A9,
  MODEL_A10,
  MODEL_A11,
  MODEL_ALIASES,
  MODEL_ALL,
  DEFAULT_LEARNING_RATE,
  DEFAULT_REGULARIZED_WEIGHT_DECAY,
  DEFAULT_HEAD_V2_RHO_HIDDEN_DIM,
  DEFAULT_DEEPSETS_PLUSPLUS_DROPOUT,
  SAVED_MODEL_SEED_POLICY_SINGLE,
  SAVED_MODEL_SEED_POLICY_FIRST,
  SAVED_MODEL_SEED_POLICIES,
  TARGET_LABEL_LEVEL_SOURCE,
  TARGET_LABEL_LEVEL_CANONICAL_FAMILY,
  TARGET_LABEL_LEVELS,
  PRIMARY_EVALUATION_SPLITS,
  SPLIT_ORDER,
  SPLIT_ORDER_INDEX,
  PREDICTION_OUTPUT_SUMMARY,
  PREDICTION_OUTPUT_ERRORS,
  PREDICTION_OUTPUT_FULL,
  PREDICTION_OUTPUT_MODES,
  SUPPORTED_MODELS,
  DEFAULT_MODEL_SET,
  PACKAGE_FEATURE_SET_CURRENT,
  PACKAGE_FEATURE_SET_V2,
  PACKAGE_FEATURE_SET_SUPPORT_MATCHED_UNSCORED,
  PACKAGE_FEATURE_SET_SHUFFLED,
  PACKAGE_FEATURE_SET_ZERO,
  PACKAGE_FEATURE_SET_SYNTHETIC,
  PACKAGE_FEATURE_SET_LABEL_SHUFFLED_MINED,
  _MODEL_POOLING,
  _DEEPSETS_PLUSPLUS_LABEL_SMOOTHING,
  _DEEPSETS_HEAD_ATTRIBUTION_MODELS,
  _SET_TRANSFORMER_POOLING,
)
from manafold.serialization import sha256_file
from manafold.taxonomy.auto_ontology import (
  AUTO_ONTOLOGY_TARGET_LABEL_LEVEL,
  DEFAULT_CORE_CARD_FREQUENCY,
  DEFAULT_MIN_JACCARD_THRESHOLD,
  write_auto_ontology,
)


def train_models(
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
  auto_ontology_output: Path | None = None,
  auto_ontology_min_jaccard_threshold: float = DEFAULT_MIN_JACCARD_THRESHOLD,
  auto_ontology_core_card_frequency: float = DEFAULT_CORE_CARD_FREQUENCY,
  saved_model_output: Path | None = None,
  saved_model_name: str = MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
  saved_model_seed_policy: str = SAVED_MODEL_SEED_POLICY_SINGLE,
  production_refit: bool = False,
  calibration_model: Path | None = None,
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
  if saved_model_seed_policy not in SAVED_MODEL_SEED_POLICIES:
    supported = ", ".join(SAVED_MODEL_SEED_POLICIES)
    raise ValueError(
      f"Unsupported saved model seed policy {saved_model_seed_policy!r}; "
      f"choose one of: {supported}."
    )
  if production_refit and saved_model_output is None:
    raise ValueError("--production-refit requires a saved-model output.")
  if calibration_model is not None and not production_refit:
    raise ValueError(
      "A calibration model is only valid for a production refit."
    )

  selected_models = _normalize_model_names(model_names, pooling=pooling)
  if (
    MODEL_SET_TRANSFORMER_A11 in selected_models
    and target_label_level != TARGET_LABEL_LEVEL_SOURCE
  ):
    raise ValueError(
      "A11 generates its family targets during training and cannot be combined "
      "with --target-label-level canonical-family."
    )
  selected_saved_model_name = _resolve_model_alias(saved_model_name)
  if (
    saved_model_output is not None
    and selected_saved_model_name not in selected_models
  ):
    raise ValueError(
      f"Saved model {selected_saved_model_name!r} must be one of the selected models."
    )
  if saved_model_output is not None and selected_saved_model_name in (
    MODEL_DEEPSETS_QUANTITY_WEIGHTED,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE,
  ):
    raise ValueError(
      "Saved-model export v0 supports directly trained model families. "
      f"Use {MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED!r} for the A1.5 saved model."
    )
  run_seeds = _normalize_seeds(seed, seeds)
  if (
    saved_model_output is not None
    and saved_model_seed_policy == SAVED_MODEL_SEED_POLICY_SINGLE
    and len(run_seeds) != 1
  ):
    raise ValueError(
      "Saved-model export uses seed policy 'single' by default; pass exactly one "
      "seed, or set --saved-model-seed-policy first for an explicit "
      "saved model from the first seed."
    )
  saved_model_seed = run_seeds[0] if saved_model_output is not None else None
  dataset = load_training_dataset(dataset_path, target_source=target_source)
  source_split_counts = {
    split_name: len(split_examples)
    for split_name, split_examples in group_inputs_by_split(
      dataset.examples
    ).items()
  }
  if production_refit:
    refit_examples = tuple(
      replace(example, split_name="train") for example in dataset.examples
    )
    dataset = replace(
      dataset,
      examples=refit_examples,
      labels=tuple(
        sorted({example.target_label_id for example in refit_examples})
      ),
    )
  family_relations: Path | None = None
  auto_ontology: dict[str, Any] | None = None
  if MODEL_SET_TRANSFORMER_A11 in selected_models:
    family_relations = auto_ontology_output or _default_auto_ontology_output(
      dataset_path=dataset.dataset_path,
      training_output=output,
      saved_model_output=saved_model_output,
    )
    auto_ontology = write_auto_ontology(
      dataset,
      family_relations,
      min_jaccard_threshold=auto_ontology_min_jaccard_threshold,
      core_card_frequency=auto_ontology_core_card_frequency,
    )
    from manafold.taxonomy.family_dataset import (
      project_dataset_to_canonical_families,
    )

    dataset, _ = project_dataset_to_canonical_families(
      dataset,
      family_relations_path=family_relations,
    )
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
  grouped = group_inputs_by_split(examples)
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
    example.source_label_id for example in examples
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
    len(primary_package_features) if package_feature_sets else package_max_count
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
    "dataset_path": str(dataset.dataset_path),
    "dataset_version": dataset.dataset_version,
    "formats": sorted({
      example.format_code for example in examples if example.format_code
    }),
    "target_source": dataset.target_source,
    "training_target": {
      "label_level": (
        AUTO_ONTOLOGY_TARGET_LABEL_LEVEL
        if auto_ontology is not None
        else target_label_level
      ),
      "requested_label_level": target_label_level,
      "family_relations": str(family_relations) if family_relations else None,
      "auto_ontology": (
        {
          "generation_mode": (
            "generated_from_all_available_examples"
            if production_refit
            else "generated_from_training_split"
          ),
          "format_version": auto_ontology["format_version"],
          "file_sha256": sha256_file(family_relations),
          "proposal_policy": auto_ontology["proposal_policy"],
          "dataset_provenance": auto_ontology["dataset_provenance"],
          "dataset_summary": auto_ontology["dataset_summary"],
        }
        if auto_ontology is not None and family_relations is not None
        else None
      ),
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
    "label_vocabulary": (
      "all_available_splits" if production_refit else "train_only"
    ),
    "fit_scope": {
      "mode": "all_available_splits" if production_refit else "train_split",
      "source_split_counts": source_split_counts,
      "fit_example_count": len(train_examples),
      "latest_event_date": max(
        (
          example.event_date.isoformat()
          for example in train_examples
          if example.event_date is not None
        ),
        default=None,
      ),
    },
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
    "saved_model_export": {
      "enabled": saved_model_output is not None,
      "model_name": selected_saved_model_name if saved_model_output else None,
      "seed_policy": (
        saved_model_seed_policy if saved_model_output is not None else None
      ),
      "export_seed": saved_model_seed,
      "output": str(saved_model_output) if saved_model_output else None,
    },
    "status": "completed",
    "models": {},
    "comparison": {},
  }

  skip_reason = _training_skip_reason(
    train_examples,
    evaluation_examples,
    dataset.labels,
    require_evaluation=not production_refit,
  )
  if skip_reason is not None:
    result["status"] = "skipped"
    result["reason"] = skip_reason
    return _write_result(result, output)

  shared_deepsets_runs: list[dict[str, dict[str, Any]]] | None = None
  include_prototype = (
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE in selected_models
  )
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
            include_seed_in_filename=len(run_seeds) > 1,
            package_scale=package_scale,
            package_projection_dim=package_projection_dim,
            architecture_package_count=architecture_package_count,
            taxonomy_config=taxonomy_config,
          )
          for run_seed in run_seeds
        ]
      seed_runs = [run[model_name] for run in shared_deepsets_runs]
    else:
      package_feature_set_name = _package_feature_set_for_model(model_name)
      package_features = (
        package_feature_sets.get(
          package_feature_set_name, PackageFeatureSet([])
        )
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
          include_seed_in_filename=len(run_seeds) > 1,
          package_scale=package_scale,
          package_projection_dim=package_projection_dim,
          architecture_package_count=architecture_package_count,
          taxonomy_config=taxonomy_config,
          saved_model_output=(
            saved_model_output
            if (
              model_name == selected_saved_model_name
              and run_seed == saved_model_seed
            )
            else None
          ),
          training_result=result,
          family_relations=family_relations,
          calibration_model=(
            calibration_model
            if (
              model_name == selected_saved_model_name
              and run_seed == saved_model_seed
            )
            else None
          ),
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
    evaluation_examples=grouped.get(
      str(split_plan["primary_evaluation_split"]), []
    ),
    evaluation_split=str(split_plan["primary_evaluation_split"]),
    package_features=package_feature_sets.get(
      PACKAGE_FEATURE_SET_V2,
      PackageFeatureSet([]),
    ),
    card_vocab=dataset.card_vocab,
    zone_vocab=dataset.zone_vocab,
  )

  return _write_result(result, output)


def _default_auto_ontology_output(
  *,
  dataset_path: Path,
  training_output: Path | None,
  saved_model_output: Path | None,
) -> Path:
  if training_output is not None:
    return training_output.parent / "family_relations.json"
  if saved_model_output is not None:
    return saved_model_output / "family_relations.json"
  return dataset_path / "model_runs" / "family_relations.json"


def _normalize_model_names(
  model_names: tuple[str, ...],
  *,
  pooling: str,
) -> tuple[str, ...]:
  if pooling not in POOLING_MODES:
    supported = ", ".join(POOLING_MODES)
    raise ValueError(
      f"Unsupported pooling mode {pooling!r}; choose one of: {supported}."
    )
  if not model_names:
    raise ValueError("At least one model must be selected.")
  if model_names == (MODEL_ALL,):
    return DEFAULT_MODEL_SET

  normalized: list[str] = []
  for model_name in model_names:
    model_name = _resolve_model_alias(model_name)
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
      raise ValueError(
        f"Unsupported model {model_name!r}; choose one of: {supported}."
      )
    normalized.append(model_name)

  return tuple(dict.fromkeys(normalized))


def _resolve_model_alias(model_name: str) -> str:
  return MODEL_ALIASES.get(model_name, model_name)


def _taxonomy_eval_path(path: Path | None) -> Path | None:
  return path


def _canonical_targets_path(path: Path | None) -> Path | None:
  return path


def _training_target_examples(
  examples: list[DeckModelInput],
  *,
  target_label_level: str,
  canonical_target_config: TaxonomyEvaluationConfig,
) -> list[DeckModelInput]:
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


def _example_taxonomy_context(example: DeckModelInput) -> dict[str, Any]:
  return {
    "event_date": (
      example.event_date.isoformat() if example.event_date is not None else None
    ),
    "format": example.format_code,
  }


def _train_labels(examples: list[DeckModelInput]) -> tuple[str, ...]:
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
    return (
      pooled_learning_rate
      if pooled_learning_rate is not None
      else learning_rate
    )
  return (
    neural_learning_rate if neural_learning_rate is not None else learning_rate
  )


def _model_weight_decay(
  model_name: str,
  *,
  weight_decay: float,
  deepsets_regularized_weight_decay: float,
  head_v2_weight_decay: float,
  package_weight_decay: float | None,
) -> float:
  if model_name in (
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
    *_DEEPSETS_PLUSPLUS_LABEL_SMOOTHING,
    MODEL_SET_TRANSFORMER_PARTIAL_BALANCED,
    MODEL_SET_TRANSFORMER_A6,
    MODEL_SET_TRANSFORMER_A8,
    MODEL_SET_TRANSFORMER_A9,
    MODEL_SET_TRANSFORMER_A10,
    MODEL_SET_TRANSFORMER_A11,
  ):
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
    dict.fromkeys((
      *package_feature_set_names,
      PACKAGE_FEATURE_SET_ZERO,
      PACKAGE_FEATURE_SET_SHUFFLED,
      PACKAGE_FEATURE_SET_SYNTHETIC,
    ))
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
  if (
    model_name
    == MODEL_DEEPSETS_QUANTITY_WEIGHTED_SUPPORT_MATCHED_UNSCORED_PACKAGES
  ):
    return PACKAGE_FEATURE_SET_SUPPORT_MATCHED_UNSCORED
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_SHUFFLED_PACKAGES:
    return PACKAGE_FEATURE_SET_SHUFFLED
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES:
    return PACKAGE_FEATURE_SET_ZERO
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO:
    return PACKAGE_FEATURE_SET_ZERO
  if model_name == MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES:
    return PACKAGE_FEATURE_SET_SYNTHETIC
  if (
    model_name
    == MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES
  ):
    return PACKAGE_FEATURE_SET_LABEL_SHUFFLED_MINED
  if model_name in (
    MODEL_POOLED_LINEAR_PACKAGE_ONLY,
    MODEL_POOLED_LINEAR_PACKAGES,
    MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES,
  ):
    return PACKAGE_FEATURE_SET_V2
  return None


def _build_package_feature_sets(
  package_feature_set_names: tuple[str, ...],
  *,
  train_examples: list[DeckModelInput],
  examples: list[DeckModelInput],
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
    rows[PACKAGE_FEATURE_SET_ZERO] = rows[
      PACKAGE_FEATURE_SET_V2
    ].with_zero_activations(examples)

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
    name: rows[name] for name in package_feature_set_names if name in rows
  }


def _package_set_summaries(
  package_feature_sets: dict[str, PackageFeatureSet],
  *,
  examples_by_split: dict[str, list[DeckModelInput]],
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
  train_examples: list[DeckModelInput],
  *,
  seed: int,
) -> list[DeckModelInput]:
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
  examples: list[DeckModelInput],
  train_examples: list[DeckModelInput],
  validation_examples: list[DeckModelInput],
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
  include_seed_in_filename: bool,
  package_scale: float,
  package_projection_dim: int,
  architecture_package_count: int,
  taxonomy_config: TaxonomyEvaluationConfig,
  saved_model_output: Path | None = None,
  training_result: dict[str, Any] | None = None,
  family_relations: Path | None = None,
  calibration_model: Path | None = None,
) -> dict[str, Any]:
  model_examples = examples
  model_train_examples = train_examples
  model_validation_examples = validation_examples
  if model_name in (
    MODEL_SET_TRANSFORMER_PARTIAL_BALANCED,
    MODEL_SET_TRANSFORMER_A6,
    MODEL_SET_TRANSFORMER_A8,
    MODEL_SET_TRANSFORMER_A9,
    MODEL_SET_TRANSFORMER_A10,
    MODEL_SET_TRANSFORMER_A11,
  ):
    main_zone_idx = dataset.zone_vocab.get("main")
    if main_zone_idx is None:
      raise ValueError("The selected Set Transformer requires a 'main' zone.")
    model_examples = _examples_for_zone(examples, zone_idx=main_zone_idx)
    model_train_examples = _examples_for_zone(
      train_examples,
      zone_idx=main_zone_idx,
    )
    model_validation_examples = _examples_for_zone(
      validation_examples,
      zone_idx=main_zone_idx,
    )

  model, metadata = create_classifier(
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
    model_train_examples,
    validation_examples=model_validation_examples,
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
    examples=model_examples,
    train_examples=model_train_examples,
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
    include_seed_in_filename=include_seed_in_filename,
    taxonomy_config=taxonomy_config,
  )
  if calibration_model is not None:
    _apply_saved_model_calibration(model_result, calibration_model)
  if saved_model_output is not None:
    if training_result is None:
      raise ValueError("training_result is required for saved model export.")
    model_result["saved_model"] = save_trained_model(
      saved_model_output,
      model_name=model_name,
      model=model,
      dataset=dataset,
      model_result=model_result,
      training_result=training_result,
      seed=seed,
      family_relations=family_relations,
    )
  return model_result


def _apply_saved_model_calibration(
  model_result: dict[str, Any],
  calibration_model: Path,
) -> None:
  temperature_path = calibration_model / "temperature.json"
  model_config_path = calibration_model / "model_config.json"
  try:
    temperature_payload = json.loads(
      temperature_path.read_text(encoding="utf-8")
    )
    source_model_config = json.loads(
      model_config_path.read_text(encoding="utf-8")
    )
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(
      f"Cannot load calibration model from {calibration_model}: {error}"
    ) from error
  temperature = temperature_payload.get("temperature")
  if not isinstance(temperature, int | float) or temperature <= 0:
    raise ValueError("Calibration model temperature must be positive.")
  model_result.setdefault("model_config", {})["temperature"] = float(
    temperature
  )
  model_result["saved_model_calibration"] = {
    "temperature": float(temperature),
    "selection_split": temperature_payload.get("selection_split"),
    "calibration": temperature_payload.get("calibration"),
    "source_model_version": source_model_config.get("model_version"),
    "source_temperature_sha256": sha256_file(temperature_path),
  }


def _examples_for_zone(
  examples: list[DeckModelInput],
  *,
  zone_idx: int,
) -> list[DeckModelInput]:
  projected: list[DeckModelInput] = []
  for example in examples:
    tokens = tuple(
      token for token in example.tokens if token.zone_idx == zone_idx
    )
    if tokens:
      projected.append(replace(example, tokens=tokens))
  return projected


def _evaluate_trained_model(
  model_name: str,
  *,
  model: Classifier,
  metadata: dict[str, Any],
  training_summary: dict[str, Any],
  seed: int,
  dataset: TrainingDataset,
  examples: list[DeckModelInput],
  train_examples: list[DeckModelInput],
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
  include_seed_in_filename: bool,
  inference_package_feature_sets: dict[str, PackageFeatureSet] | None = None,
  taxonomy_config: TaxonomyEvaluationConfig | None = None,
) -> dict[str, Any]:
  primary_evaluation_split = str(
    _split_plan(group_inputs_by_split(examples))["primary_evaluation_split"]
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
  if (
    "temperature" not in model_config
    and calibration.get("status") == "completed"
  ):
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
    include_seed_in_filename=include_seed_in_filename,
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
    include_seed_in_filename=include_seed_in_filename,
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
  examples: list[DeckModelInput],
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
  train_examples: list[DeckModelInput],
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
  examples: list[DeckModelInput],
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
  examples: list[DeckModelInput],
) -> dict[str, Any]:
  if reference_logits.shape != comparison_logits.shape:
    return {
      "status": "skipped",
      "reason": (
        "Logit shapes differ: "
        f"{tuple(reference_logits.shape)} != {tuple(comparison_logits.shape)}."
      ),
    }
  indexes_by_split: dict[str, list[int]] = {
    "overall": list(range(len(examples)))
  }
  for index, example in enumerate(examples):
    indexes_by_split.setdefault(example.split_name, []).append(index)

  return {
    split_name: _logit_delta_block(
      reference_logits.index_select(
        0, torch.tensor(indexes, dtype=torch.long)
      ),
      comparison_logits.index_select(
        0, torch.tensor(indexes, dtype=torch.long)
      ),
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
    comparison_logits > reference_top_comparison_logits
  ).sum(dim=1) + 1
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


def _run_quantity_weighted_deepsets_training(
  *,
  seed: int,
  dataset: TrainingDataset,
  examples: list[DeckModelInput],
  train_examples: list[DeckModelInput],
  validation_examples: list[DeckModelInput],
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
  include_seed_in_filename: bool,
  package_scale: float,
  package_projection_dim: int,
  architecture_package_count: int,
  taxonomy_config: TaxonomyEvaluationConfig,
) -> dict[str, dict[str, Any]]:
  model, metadata = create_classifier(
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
    include_seed_in_filename=include_seed_in_filename,
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
    include_seed_in_filename=include_seed_in_filename,
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


def _predict(
  model: Classifier,
  examples: list[DeckModelInput],
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
    actual_label_train_support = train_label_support.get(
      example.target_label_id, 0
    )
    rows.append({
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
    })

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
  include_seed_in_filename: bool,
) -> dict[str, Any]:
  if output is None or prediction_output == PREDICTION_OUTPUT_SUMMARY:
    return {
      "prediction_output": prediction_output,
      "prediction_export_count": 0,
    }

  file_stem = _prediction_file_stem(
    model_name,
    seed=seed,
    include_seed=include_seed_in_filename,
  )
  prediction_path = output.parent / "predictions" / f"{file_stem}.jsonl"
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
  raise AssertionError(
    f"Unhandled prediction output mode: {prediction_output}"
  )


def _write_embedding_exports(
  output: Path | None,
  model_name: str,
  model: Classifier,
  dataset: TrainingDataset,
  examples: list[DeckModelInput],
  train_examples: list[DeckModelInput],
  *,
  seed: int,
  batch_size: int,
  export_embeddings: bool,
  include_seed_in_filename: bool,
) -> dict[str, Any]:
  if (
    output is None
    or not export_embeddings
    or not isinstance(model, (DeepSetsClassifier, SetTransformerClassifier))
  ):
    return {}

  embedding_dir = output.parent / "embeddings"
  file_stem = _prediction_file_stem(
    model_name,
    seed=seed,
    include_seed=include_seed_in_filename,
  )
  card_embedding_path = embedding_dir / f"{file_stem}_cards.jsonl"
  deck_embedding_path = embedding_dir / f"{file_stem}_decks.jsonl"

  _write_jsonl(
    card_embedding_path, model.card_embedding_rows(dataset.card_vocab)
  )
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


def _prediction_file_stem(
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
  train_examples: list[DeckModelInput],
  evaluation_examples: list[DeckModelInput],
  labels: tuple[str, ...],
  *,
  require_evaluation: bool = True,
) -> str | None:
  if not labels:
    return "No proxy-target labels are available."
  if not train_examples:
    return "No train split proxy-target examples are available."
  if require_evaluation and not evaluation_examples:
    return "No evaluation split proxy-target examples are available."
  return None


def _split_plan(grouped: dict[str, list[DeckModelInput]]) -> dict[str, Any]:
  names = sorted(
    grouped,
    key=lambda split_name: (
      SPLIT_ORDER_INDEX.get(split_name, 10_000),
      split_name,
    ),
  )
  evaluation_splits = tuple(
    split_name for split_name in names if split_name != "train"
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
    "counts": {split_name: len(grouped[split_name]) for split_name in names},
    "evaluation_splits": evaluation_splits,
    "primary_evaluation_split": primary_evaluation_split,
  }


def _normalize_seeds(
  seed: int, seeds: tuple[int, ...] | None
) -> tuple[int, ...]:
  if seeds is None:
    return (seed,)
  normalized = tuple(dict.fromkeys(seeds))
  if not normalized:
    raise ValueError("At least one seed must be provided.")
  return normalized


def _label_support(examples: list[DeckModelInput]) -> dict[str, int]:
  rows: dict[str, int] = {}
  for example in examples:
    rows[example.target_label_id] = rows.get(example.target_label_id, 0) + 1
  return dict(sorted(rows.items()))


def _source_label_support(examples: list[DeckModelInput]) -> dict[str, int]:
  rows: dict[str, int] = {}
  for example in examples:
    rows[example.source_label_id] = rows.get(example.source_label_id, 0) + 1
  return dict(sorted(rows.items()))


def _card_support(examples: list[DeckModelInput]) -> dict[int, int]:
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

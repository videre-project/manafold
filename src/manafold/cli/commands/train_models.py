from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from manafold.cli.common import (
  SubparserRegistry,
  _mean_std,
  _repo_path_or_none,
  _seeds,
)
from manafold.constants import PROJECT_ROOT
from manafold.models.classifiers import POOLING_MODES, POOLING_SUM
from manafold.models.features.card_packages import (
  DEFAULT_PACKAGE_TYPES,
  PACKAGE_SCORING_MODES,
  PACKAGE_TYPES,
)
from manafold.models.training.training_pipeline import (
  DEFAULT_HEAD_V2_RHO_HIDDEN_DIM,
  DEFAULT_LEARNING_RATE,
  DEFAULT_REGULARIZED_WEIGHT_DECAY,
  MODEL_ALIASES,
  MODEL_ALL,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
  PREDICTION_OUTPUT_ERRORS,
  PREDICTION_OUTPUT_MODES,
  SAVED_MODEL_SEED_POLICIES,
  SAVED_MODEL_SEED_POLICY_SINGLE,
  SUPPORTED_MODELS,
  TARGET_LABEL_LEVEL_SOURCE,
  TARGET_LABEL_LEVELS,
  train_models,
)
from manafold.taxonomy.auto_ontology import (
  DEFAULT_CORE_CARD_FREQUENCY,
  DEFAULT_MIN_JACCARD_THRESHOLD,
)


def register(subparsers: SubparserRegistry) -> None:
  parser = subparsers.add_parser(
    "model-train",
    help="Train the reference baselines or an explicit selected model.",
  )
  parser.add_argument("dataset", type=Path)
  parser.add_argument(
    "--model",
    choices=(MODEL_ALL, *MODEL_ALIASES, *SUPPORTED_MODELS),
    action="append",
    help=(
      "Model to train. Repeat to run a focused set. "
      "Omit for pooled-linear, Deep Sets quantity-weighted, and regularized Deep Sets."
    ),
  )
  parser.add_argument(
    "--pooling",
    choices=POOLING_MODES,
    default=POOLING_SUM,
    help="Pooling mode used when --model deepsets is selected.",
  )
  parser.add_argument("--output", type=Path)
  parser.add_argument(
    "--target-source",
    default="source_archetype_name_proxy",
    help="Proxy target source to train against.",
  )
  parser.add_argument(
    "--target-label-level",
    choices=TARGET_LABEL_LEVELS,
    default=TARGET_LABEL_LEVEL_SOURCE,
    help=(
      "Training label level. Use canonical-family only for reviewed target maps."
    ),
  )
  parser.add_argument(
    "--canonical-targets",
    type=Path,
    help=(
      "Reviewed source-label to canonical-family target map. Required for "
      "canonical-family training."
    ),
  )
  parser.add_argument(
    "--auto-ontology-output",
    type=Path,
    help=(
      "A11 family-relation output. It is regenerated from the training split "
      "on every A11 training run."
    ),
  )
  parser.add_argument(
    "--auto-ontology-min-jaccard-threshold",
    type=float,
    default=DEFAULT_MIN_JACCARD_THRESHOLD,
    help="Minimum core-card Jaccard score for an A11 family relation.",
  )
  parser.add_argument(
    "--auto-ontology-core-card-frequency",
    type=float,
    default=DEFAULT_CORE_CARD_FREQUENCY,
    help="Exclusive within-label frequency threshold for A11 core cards.",
  )
  parser.add_argument("--epochs", type=int, default=40)
  parser.add_argument(
    "--learning-rate",
    type=float,
    default=DEFAULT_LEARNING_RATE,
  )
  parser.add_argument("--pooled-learning-rate", type=float)
  parser.add_argument("--neural-learning-rate", type=float)
  parser.add_argument("--weight-decay", type=float, default=0.0)
  parser.add_argument(
    "--deepsets-regularized-weight-decay",
    type=float,
    default=DEFAULT_REGULARIZED_WEIGHT_DECAY,
    help="Weight decay for deepsets-quantity-weighted-regularized.",
  )
  parser.add_argument(
    "--head-v2-weight-decay",
    type=float,
    default=DEFAULT_REGULARIZED_WEIGHT_DECAY,
    help="Weight decay for deepsets-quantity-weighted-head-v2-regularized.",
  )
  parser.add_argument("--package-weight-decay", type=float)
  parser.add_argument("--seed", type=int, default=13)
  parser.add_argument(
    "--seeds",
    type=_seeds,
    help="Comma-separated seeds, such as 13,17,23. Overrides --seed.",
  )
  parser.add_argument("--embedding-dim", type=int, default=32)
  parser.add_argument("--hidden-dim", type=int, default=64)
  parser.add_argument(
    "--head-v2-rho-hidden-dim",
    type=int,
    default=DEFAULT_HEAD_V2_RHO_HIDDEN_DIM,
    help="Rho hidden width for deepsets-quantity-weighted-head-v2.",
  )
  parser.add_argument("--attention-heads", type=int, default=4)
  parser.add_argument("--attention-layers", type=int, default=2)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--package-min-support", type=int, default=25)
  parser.add_argument("--package-min-event-support", type=int, default=8)
  parser.add_argument("--package-max-size", type=int, default=3)
  parser.add_argument("--package-max-count", type=int, default=2048)
  parser.add_argument("--package-scale", type=float, default=1.0)
  parser.add_argument("--package-projection-dim", type=int, default=64)
  parser.add_argument(
    "--package-scoring",
    choices=PACKAGE_SCORING_MODES,
    default="bayesian-log-odds",
  )
  parser.add_argument("--package-min-best-label-count", type=int, default=10)
  parser.add_argument(
    "--package-min-best-label-precision",
    type=float,
    default=0.05,
  )
  parser.add_argument("--package-max-label-entropy", type=float)
  parser.add_argument(
    "--package-max-train-activation-rate",
    type=float,
    default=0.08,
  )
  parser.add_argument(
    "--package-type",
    choices=PACKAGE_TYPES,
    action="append",
    dest="package_types",
    help=(
      "Package type to include. Repeat to include several. "
      f"Defaults to {', '.join(DEFAULT_PACKAGE_TYPES)}."
    ),
  )
  parser.add_argument("--package-random-seed", type=int, default=13)
  parser.add_argument(
    "--max-steps",
    type=int,
    help="Train for this many optimizer updates, independent of epoch count.",
  )
  parser.add_argument(
    "--shuffle",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Shuffle training deck batches each epoch.",
  )
  parser.add_argument(
    "--device",
    default="auto",
    help="Torch device for deepsets, such as auto, cpu, cuda, or cuda:0.",
  )
  parser.add_argument(
    "--export-embeddings",
    action="store_true",
    help="Write Deep Sets card and deck embedding JSONL files.",
  )
  parser.add_argument(
    "--prediction-output",
    choices=PREDICTION_OUTPUT_MODES,
    default=PREDICTION_OUTPUT_ERRORS,
    help="Amount of per-deck prediction detail to write.",
  )
  parser.add_argument(
    "--taxonomy-eval",
    type=Path,
    help="Optional taxonomy evaluation file for canonical-family reporting.",
  )
  parser.add_argument(
    "--saved-model-output",
    type=Path,
    help="Directory where the selected fitted saved model should be written.",
  )
  parser.add_argument(
    "--saved-model-name",
    default=MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
    help="Model family to save for later scoring.",
  )
  parser.add_argument(
    "--saved-model-seed-policy",
    choices=SAVED_MODEL_SEED_POLICIES,
    default=SAVED_MODEL_SEED_POLICY_SINGLE,
    help=(
      "Seed policy for saved-model export. The default requires exactly one seed; "
      "use first only to save the first model from a multi-seed research run."
    ),
  )
  parser.add_argument(
    "--production-refit",
    action="store_true",
    help=(
      "Fit the saved model on every available dataset split. Use this only after "
      "a chronological evaluation has selected the epoch count."
    ),
  )
  parser.add_argument(
    "--calibration-model",
    type=Path,
    help=(
      "Saved chronological-evaluation model whose validation calibration should "
      "be carried into a production refit."
    ),
  )
  parser.set_defaults(handler=run)


def run(args: Namespace) -> None:
  dataset_path = args.dataset
  if not dataset_path.is_absolute():
    dataset_path = PROJECT_ROOT / dataset_path

  output = args.output
  if output is None:
    output = dataset_path / "model_runs" / "training_results.json"
  elif not output.is_absolute():
    output = PROJECT_ROOT / output
  taxonomy_eval = _repo_path_or_none(args.taxonomy_eval)
  canonical_targets = _repo_path_or_none(args.canonical_targets)
  auto_ontology_output = _repo_path_or_none(args.auto_ontology_output)

  result = train_models(
    dataset_path,
    output=output,
    model_names=tuple(args.model or (MODEL_ALL,)),
    pooling=args.pooling,
    target_source=args.target_source,
    target_label_level=args.target_label_level,
    canonical_targets=canonical_targets,
    auto_ontology_output=auto_ontology_output,
    auto_ontology_min_jaccard_threshold=(
      args.auto_ontology_min_jaccard_threshold
    ),
    auto_ontology_core_card_frequency=(
      args.auto_ontology_core_card_frequency
    ),
    epochs=args.epochs,
    learning_rate=args.learning_rate,
    pooled_learning_rate=args.pooled_learning_rate,
    neural_learning_rate=args.neural_learning_rate,
    weight_decay=args.weight_decay,
    deepsets_regularized_weight_decay=args.deepsets_regularized_weight_decay,
    head_v2_weight_decay=args.head_v2_weight_decay,
    package_weight_decay=args.package_weight_decay,
    seed=args.seed,
    seeds=args.seeds,
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    head_v2_rho_hidden_dim=args.head_v2_rho_hidden_dim,
    attention_heads=args.attention_heads,
    attention_layers=args.attention_layers,
    batch_size=args.batch_size,
    shuffle=args.shuffle,
    max_steps=args.max_steps,
    device=args.device,
    export_embeddings=args.export_embeddings,
    prediction_output=args.prediction_output,
    package_min_support=args.package_min_support,
    package_min_event_support=args.package_min_event_support,
    package_max_size=args.package_max_size,
    package_max_count=args.package_max_count,
    package_scale=args.package_scale,
    package_projection_dim=args.package_projection_dim,
    package_scoring=args.package_scoring,
    package_min_best_label_count=args.package_min_best_label_count,
    package_min_best_label_precision=args.package_min_best_label_precision,
    package_max_label_entropy=args.package_max_label_entropy,
    package_max_train_activation_rate=args.package_max_train_activation_rate,
    package_types=tuple(args.package_types or DEFAULT_PACKAGE_TYPES),
    package_random_seed=args.package_random_seed,
    taxonomy_eval=taxonomy_eval,
    saved_model_output=(
      _repo_path_or_none(args.saved_model_output)
      if args.saved_model_output is not None
      else None
    ),
    saved_model_name=args.saved_model_name,
    saved_model_seed_policy=args.saved_model_seed_policy,
    production_refit=args.production_refit,
    calibration_model=(
      _repo_path_or_none(args.calibration_model)
      if args.calibration_model is not None
      else None
    ),
  )
  print(f"Wrote model training result to {output}")
  print(f"Status: {result['status']}")
  if result["status"] == "completed":
    for model_name, model_result in result["models"].items():
      if result.get("fit_scope", {}).get("mode") == "all_available_splits":
        calibration = model_result.get("saved_model_calibration", {})
        print(
          f"{model_name}: production refit on "
          f"{result['fit_scope']['fit_example_count']} trainable examples through "
          f"{result['fit_scope']['latest_event_date']}, "
          f"{model_result['completed_epochs']} epochs, "
          f"validation temperature {calibration.get('temperature')}"
        )
        continue
      summary = model_result.get("multi_seed_summary", {})
      evaluation_split = model_result["primary_evaluation_split"]
      evaluation_metrics = model_result["metrics"]["splits"].get(
        evaluation_split,
        {},
      )
      evaluation_views = model_result["metrics"]["evaluation_views"].get(
        evaluation_split,
        {},
      )
      closed_seen = evaluation_views.get("closed_set_seen_labels", {})
      open_unseen = evaluation_views.get("open_set_unseen_labels", {})
      calibration = model_result["metrics"].get("calibration", {})
      evaluation_calibration = (
        calibration.get("temperature_scaled", {})
        .get("splits", {})
        .get(evaluation_split, {})
      )
      taxonomy = model_result.get("taxonomy_evaluation", {})
      canonical_metrics = (
        taxonomy.get("metrics", {})
        .get("splits", {})
        .get(evaluation_split, {})
      )
      canonical_views = (
        taxonomy.get("metrics", {})
        .get("evaluation_views", {})
        .get(evaluation_split, {})
      )
      canonical_closed_seen = canonical_views.get(
        "closed_set_seen_labels",
        {},
      )
      source_label_metrics = (
        model_result.get("source_label_evaluation", {})
        .get("metrics", {})
        .get("splits", {})
        .get(evaluation_split, {})
      )
      print(
        f"{model_name}: "
        f"target {result['training_target']['label_level']}, "
        f"{evaluation_split} accuracy {evaluation_metrics.get('accuracy')}, "
        f"top-3 {evaluation_metrics.get('top_3_accuracy')}, "
        f"macro-F1 {evaluation_metrics.get('macro_f1')}, "
        f"closed-set {closed_seen.get('accuracy')}, "
        f"open-set {open_unseen.get('accuracy')}, "
        f"temp {calibration.get('temperature')}, "
        f"temp NLL {evaluation_calibration.get('nll')}, "
        f"temp Brier {evaluation_calibration.get('brier')}, "
        f"temp ECE {evaluation_calibration.get('ece')}, "
        f"source-label accuracy {source_label_metrics.get('accuracy')}, "
        f"canonical accuracy {canonical_metrics.get('accuracy')}, "
        f"canonical closed-set {canonical_closed_seen.get('accuracy')}"
      )
      if len(model_result.get("seeds", [])) > 1 and summary:
        print(
          f"{model_name} means: "
          f"{evaluation_split} accuracy {_mean_std(summary, 'primary_accuracy')}, "
          f"closed-set {_mean_std(summary, 'primary_closed_set_accuracy')}, "
          f"top-3 {_mean_std(summary, 'primary_top_3_accuracy')}, "
          f"macro-F1 {_mean_std(summary, 'primary_macro_f1')}, "
          f"source-label accuracy "
          f"{_mean_std(summary, 'primary_source_label_accuracy')}, "
          f"source-label closed-set "
          f"{_mean_std(summary, 'primary_source_label_closed_set_accuracy')}, "
          f"canonical accuracy "
          f"{_mean_std(summary, 'primary_canonical_accuracy')}, "
          f"canonical closed-set "
          f"{_mean_std(summary, 'primary_canonical_closed_set_accuracy')}, "
          f"canonical macro-F1 "
          f"{_mean_std(summary, 'primary_canonical_macro_f1')}, "
          f"temp NLL {_mean_std(summary, 'primary_temperature_scaled_nll')}, "
          f"temp Brier {_mean_std(summary, 'primary_temperature_scaled_brier')}, "
          f"temp ECE {_mean_std(summary, 'primary_temperature_scaled_ece')}, "
          f"MSP AUROC {_mean_std(summary, 'primary_msp_auroc')}, "
          f"temp MSP AUROC "
          f"{_mean_std(summary, 'primary_temperature_scaled_msp_auroc')}, "
          f"energy AUROC {_mean_std(summary, 'primary_energy_auroc')}, "
          f"prototype distance AUROC "
          f"{_mean_std(summary, 'primary_nearest_prototype_distance_auroc')}, "
          f"known false abstain "
          f"{_mean_std(summary, 'primary_known_false_abstention_rate')}, "
          f"unknown recall {_mean_std(summary, 'primary_unknown_recall')}, "
          f"energy known false abstain "
          f"{_mean_std(summary, 'primary_energy_known_false_abstention_rate')}, "
          f"energy unknown recall {_mean_std(summary, 'primary_energy_unknown_recall')}"
        )
    if result["comparison"]:
      for model_name, comparison in result["comparison"].items():
        for split_name, split in comparison.items():
          print(
            f"{model_name} {split_name}: "
            f"model {split['model_accuracy']}, "
            f"pooled-linear {split['pooled_linear_accuracy']}, "
            f"delta {split['accuracy_delta']}"
          )

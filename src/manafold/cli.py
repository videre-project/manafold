from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from manafold.config import repo_root
from manafold.data.export import DatasetExportOptions, export_dataset
from manafold.data.validate import verify_dataset_export
from manafold.models.aliases import run_alias_candidate_scoring
from manafold.models.weak_relations import (
  DEFAULT_EXCLUDED_TRAINING_SPLITS,
  run_weak_relation_graph,
)
from manafold.models.weak_state import run_weak_relation_state
from manafold.models.weak_targets import run_weak_target_preview
from manafold.models.model_artifacts import run_model_scoring
from manafold.models.train import (
  DEFAULT_HEAD_V2_RHO_HIDDEN_DIM,
  DEFAULT_LEARNING_RATE,
  DEFAULT_REGULARIZED_WEIGHT_DECAY,
  MODEL_ARTIFACT_SEED_POLICIES,
  MODEL_ARTIFACT_SEED_POLICY_SINGLE,
  MODEL_ALL,
  MODEL_DEEPSETS,
  MODEL_DEEPSETS_MEAN,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_CONSTANT_FEATURES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_HEAD_V2_REGULARIZED,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_PACKAGES_V1,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_SHUFFLED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES,
  MODEL_DEEPSETS_SUM,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_SUPPORT_MATCHED_UNSCORED_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_WIDE_RHO,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_EXTRA_DIMS_PRESERVE_INIT,
  MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES,
  MODEL_POOLED_LINEAR,
  MODEL_POOLED_LINEAR_PACKAGE_ONLY,
  MODEL_POOLED_LINEAR_PACKAGES,
  MODEL_SET_TRANSFORMER,
  MODEL_SET_TRANSFORMER_PMA,
  MODEL_SET_TRANSFORMER_QUANTITY_WEIGHTED,
  PREDICTION_OUTPUT_ERRORS,
  PREDICTION_OUTPUT_MODES,
  TARGET_LABEL_LEVEL_SOURCE,
  TARGET_LABEL_LEVELS,
  run_model_training,
)
from manafold.models.deepsets import POOLING_MODES, POOLING_SUM
from manafold.models.packages import (
  DEFAULT_PACKAGE_TYPES,
  PACKAGE_SCORING_MODES,
  PACKAGE_TYPES,
)
from manafold.models.rolling import (
  parse_rolling_window,
  run_rolling_evaluation,
)


def main() -> None:
  parser = argparse.ArgumentParser(
    prog="manafold",
    description="Build Manafold datasets and model results.",
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  dataset = subparsers.add_parser(
    "dataset",
    help="Export dataset files for model training.",
  )
  dataset.add_argument("--format", default="modern", help="Format code to export.")
  dataset.add_argument("--start", default="2023-01-01", type=_date)
  dataset.add_argument("--end", default="2026-06-30", type=_date)
  dataset.add_argument("--output", type=Path)
  dataset.add_argument("--dataset-version")
  dataset.add_argument("--train-end", default="2025-06-30", type=_date)
  dataset.add_argument("--validation-end", default="2025-12-31", type=_date)
  dataset.add_argument(
    "--dev-test-end",
    type=_date,
    help=(
      "Optional final-holdout boundary. Events after validation and on or before "
      "this date are dev-test; later events are final-test."
    ),
  )
  dataset.add_argument("--env-file", type=Path)
  dataset.add_argument(
    "--limit-events",
    type=int,
    help="Export only the first N events in date order for a small sample.",
  )
  check = subparsers.add_parser(
    "check",
    help="Verify exported dataset files and schemas.",
  )
  check.add_argument("dataset", type=Path)
  model_train = subparsers.add_parser(
    "model-train",
    help="Train the reference baselines or an explicit selected model.",
  )
  model_train.add_argument("dataset", type=Path)
  model_train.add_argument(
    "--model",
    choices=(
      MODEL_ALL,
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
      MODEL_DEEPSETS_QUANTITY_WEIGHTED_SHUFFLED_PACKAGES,
      MODEL_DEEPSETS_QUANTITY_WEIGHTED_ZERO_PACKAGES,
      MODEL_DEEPSETS_QUANTITY_WEIGHTED_EXTRA_RHO,
      MODEL_DEEPSETS_QUANTITY_WEIGHTED_SYNTHETIC_PACKAGES,
      MODEL_DEEPSETS_QUANTITY_WEIGHTED_LABEL_SHUFFLED_MINED_PACKAGES,
      MODEL_DEEPSETS_QUANTITY_WEIGHTED_SUPPORT_MATCHED_UNSCORED_PACKAGES,
      MODEL_DEEPSETS_QUANTITY_WEIGHTED_PROTOTYPE,
      MODEL_SET_TRANSFORMER,
      MODEL_SET_TRANSFORMER_PMA,
      MODEL_SET_TRANSFORMER_QUANTITY_WEIGHTED,
    ),
    action="append",
    help=(
      "Model to train. Repeat to run a focused set. "
      "Omit for pooled-linear, Deep Sets quantity-weighted, and regularized Deep Sets."
    ),
  )
  model_train.add_argument(
    "--pooling",
    choices=POOLING_MODES,
    default=POOLING_SUM,
    help="Pooling mode used when --model deepsets is selected.",
  )
  model_train.add_argument("--output", type=Path)
  model_train.add_argument(
    "--target-source",
    default="source_archetype_name_proxy",
    help="Proxy target source to train against.",
  )
  model_train.add_argument(
    "--target-label-level",
    choices=TARGET_LABEL_LEVELS,
    default=TARGET_LABEL_LEVEL_SOURCE,
    help=(
      "Training label level. Use canonical-family only for reviewed target maps."
    ),
  )
  model_train.add_argument(
    "--canonical-targets",
    type=Path,
    help=(
      "Reviewed source-label to canonical-family target map. Required for "
      "canonical-family training."
    ),
  )
  model_train.add_argument("--epochs", type=int, default=40)
  model_train.add_argument(
    "--learning-rate",
    type=float,
    default=DEFAULT_LEARNING_RATE,
  )
  model_train.add_argument("--pooled-learning-rate", type=float)
  model_train.add_argument("--neural-learning-rate", type=float)
  model_train.add_argument("--weight-decay", type=float, default=0.0)
  model_train.add_argument(
    "--deepsets-regularized-weight-decay",
    type=float,
    default=DEFAULT_REGULARIZED_WEIGHT_DECAY,
    help="Weight decay for deepsets-quantity-weighted-regularized.",
  )
  model_train.add_argument(
    "--head-v2-weight-decay",
    type=float,
    default=DEFAULT_REGULARIZED_WEIGHT_DECAY,
    help="Weight decay for deepsets-quantity-weighted-head-v2-regularized.",
  )
  model_train.add_argument("--package-weight-decay", type=float)
  model_train.add_argument("--seed", type=int, default=13)
  model_train.add_argument(
    "--seeds",
    type=_seeds,
    help="Comma-separated seeds, such as 13,17,23. Overrides --seed.",
  )
  model_train.add_argument("--embedding-dim", type=int, default=32)
  model_train.add_argument("--hidden-dim", type=int, default=64)
  model_train.add_argument(
    "--head-v2-rho-hidden-dim",
    type=int,
    default=DEFAULT_HEAD_V2_RHO_HIDDEN_DIM,
    help="Rho hidden width for deepsets-quantity-weighted-head-v2.",
  )
  model_train.add_argument("--attention-heads", type=int, default=4)
  model_train.add_argument("--attention-layers", type=int, default=2)
  model_train.add_argument("--batch-size", type=int, default=32)
  model_train.add_argument("--package-min-support", type=int, default=25)
  model_train.add_argument("--package-min-event-support", type=int, default=8)
  model_train.add_argument("--package-max-size", type=int, default=3)
  model_train.add_argument("--package-max-count", type=int, default=2048)
  model_train.add_argument("--package-scale", type=float, default=1.0)
  model_train.add_argument("--package-projection-dim", type=int, default=64)
  model_train.add_argument(
    "--package-scoring",
    choices=PACKAGE_SCORING_MODES,
    default="bayesian-log-odds",
  )
  model_train.add_argument("--package-min-best-label-count", type=int, default=10)
  model_train.add_argument(
    "--package-min-best-label-precision",
    type=float,
    default=0.05,
  )
  model_train.add_argument("--package-max-label-entropy", type=float)
  model_train.add_argument(
    "--package-max-train-activation-rate",
    type=float,
    default=0.08,
  )
  model_train.add_argument(
    "--package-type",
    choices=PACKAGE_TYPES,
    action="append",
    dest="package_types",
    help=(
      "Package type to include. Repeat to include several. "
      f"Defaults to {', '.join(DEFAULT_PACKAGE_TYPES)}."
    ),
  )
  model_train.add_argument("--package-random-seed", type=int, default=13)
  model_train.add_argument(
    "--max-steps",
    type=int,
    help="Train for this many optimizer updates, independent of epoch count.",
  )
  model_train.add_argument(
    "--shuffle",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Shuffle training deck batches each epoch.",
  )
  model_train.add_argument(
    "--device",
    default="auto",
    help="Torch device for deepsets, such as auto, cpu, cuda, or cuda:0.",
  )
  model_train.add_argument(
    "--export-embeddings",
    action="store_true",
    help="Write Deep Sets card and deck embedding JSONL files.",
  )
  model_train.add_argument(
    "--prediction-output",
    choices=PREDICTION_OUTPUT_MODES,
    default=PREDICTION_OUTPUT_ERRORS,
    help="Per-deck prediction artifact detail.",
  )
  model_train.add_argument(
    "--taxonomy-eval",
    type=Path,
    help="Optional taxonomy evaluation file for canonical-family reporting.",
  )
  model_train.add_argument(
    "--model-artifact-output",
    type=Path,
    help="Directory where the selected fitted model artifact should be written.",
  )
  model_train.add_argument(
    "--model-artifact-model",
    default=MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
    help="Selected model family to export as a reusable scoring artifact.",
  )
  model_train.add_argument(
    "--model-artifact-seed-policy",
    choices=MODEL_ARTIFACT_SEED_POLICIES,
    default=MODEL_ARTIFACT_SEED_POLICY_SINGLE,
    help=(
      "Seed policy for artifact export. The default requires exactly one seed; "
      "use first only for an explicit research artifact from a multi-seed run."
    ),
  )
  model_score = subparsers.add_parser(
    "model-score",
    help="Score a dataset with a saved Manafold model artifact.",
  )
  model_score.add_argument("--model-artifact", type=Path, required=True)
  model_score.add_argument("--dataset", type=Path, required=True)
  model_score.add_argument("--output", type=Path, required=True)
  model_score.add_argument(
    "--target-source",
    default="source_archetype_name_proxy",
  )
  model_score.add_argument("--batch-size", type=int, default=1024)
  model_score.add_argument("--top-k", type=int, default=3)
  model_score.add_argument(
    "--low-confidence-threshold",
    type=float,
    default=0.5,
  )
  model_score.add_argument(
    "--device",
    default="auto",
    help="Torch device such as auto, cpu, cuda, or cuda:0.",
  )
  model_score.add_argument("--deck-embedding-output", type=Path)
  model_score.add_argument("--taxonomy-eval-version")
  rolling_eval = subparsers.add_parser(
    "rolling-eval",
    help="Run the reference baselines across time windows.",
  )
  rolling_eval.add_argument(
    "--window",
    action="append",
    required=True,
    type=_rolling_window,
    help=(
      "Window as name,start,train_end,validation_end,dev_test_end,end. "
      "Repeat for multiple time slices."
    ),
  )
  rolling_eval.add_argument("--format", default="modern", help="Format code to export.")
  rolling_eval.add_argument(
    "--output",
    type=Path,
    default=Path("data/rolling_eval"),
    help="Directory for rolling summary and per-window artifacts.",
  )
  rolling_eval.add_argument(
    "--target-source",
    default="source_archetype_name_proxy",
    help="Proxy target source to train against.",
  )
  rolling_eval.add_argument(
    "--target-label-level",
    choices=TARGET_LABEL_LEVELS,
    default=TARGET_LABEL_LEVEL_SOURCE,
    help=(
      "Training label level. Use canonical-family only for reviewed target maps."
    ),
  )
  rolling_eval.add_argument(
    "--canonical-targets",
    type=Path,
    help=(
      "Reviewed source-label to canonical-family target map. Required for "
      "canonical-family training."
    ),
  )
  rolling_eval.add_argument("--env-file", type=Path)
  rolling_eval.add_argument(
    "--limit-events",
    type=int,
    help="Export only the first N events per window for a small sample.",
  )
  rolling_eval.add_argument("--epochs", type=int, default=40)
  rolling_eval.add_argument(
    "--learning-rate",
    type=float,
    default=DEFAULT_LEARNING_RATE,
  )
  rolling_eval.add_argument("--pooled-learning-rate", type=float)
  rolling_eval.add_argument("--neural-learning-rate", type=float)
  rolling_eval.add_argument("--weight-decay", type=float, default=0.0)
  rolling_eval.add_argument(
    "--deepsets-regularized-weight-decay",
    type=float,
    default=DEFAULT_REGULARIZED_WEIGHT_DECAY,
  )
  rolling_eval.add_argument("--seed", type=int, default=13)
  rolling_eval.add_argument(
    "--seeds",
    type=_seeds,
    help="Comma-separated seeds, such as 13,17,23. Overrides --seed.",
  )
  rolling_eval.add_argument("--embedding-dim", type=int, default=32)
  rolling_eval.add_argument("--hidden-dim", type=int, default=64)
  rolling_eval.add_argument("--batch-size", type=int, default=1024)
  rolling_eval.add_argument(
    "--max-steps",
    type=int,
    help="Train for this many optimizer updates per model/seed/window.",
  )
  rolling_eval.add_argument(
    "--shuffle",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Shuffle training deck batches each epoch.",
  )
  rolling_eval.add_argument(
    "--device",
    default="auto",
    help="Torch device for deepsets, such as auto, cpu, cuda, or cuda:0.",
  )
  rolling_eval.add_argument(
    "--prediction-output",
    choices=PREDICTION_OUTPUT_MODES,
    default=PREDICTION_OUTPUT_ERRORS,
    help="Per-deck prediction artifact detail.",
  )
  rolling_eval.add_argument(
    "--taxonomy-eval",
    type=Path,
    help="Optional taxonomy evaluation file for canonical-family reporting.",
  )
  alias_candidates = subparsers.add_parser(
    "alias-candidates",
    help="Score source-label alias and review candidates from predictions.",
  )
  alias_candidates.add_argument(
    "rolling_result",
    nargs="?",
    type=Path,
    help=(
      "Path to rolling_evaluation_results.json, or a rolling-eval output directory."
    ),
  )
  alias_candidates.add_argument(
    "--predictions",
    type=Path,
    help="Prediction parquet from model-score for production-backed evidence.",
  )
  alias_candidates.add_argument(
    "--predictions-manifest",
    type=Path,
    help="Optional model-score manifest JSON. Defaults next to --predictions.",
  )
  alias_candidates.add_argument(
    "--deck-embeddings",
    type=Path,
    help="Optional deck embeddings parquet from model-score.",
  )
  alias_candidates.add_argument(
    "--dataset",
    type=Path,
    help="Dataset export used by the prediction parquet.",
  )
  alias_candidates.add_argument("--output", type=Path)
  alias_candidates.add_argument(
    "--model",
    default=MODEL_DEEPSETS_QUANTITY_WEIGHTED_REGULARIZED,
    help="Model family whose prediction errors should seed confusion evidence.",
  )
  alias_candidates.add_argument(
    "--target-source",
    default="source_archetype_name_proxy",
  )
  alias_candidates.add_argument(
    "--taxonomy-eval",
    type=Path,
    help="Optional seed alias evidence file.",
  )
  alias_candidates.add_argument(
    "--min-confusion-count",
    type=int,
    default=5,
  )
  alias_candidates.add_argument(
    "--min-seed-confusion-count",
    type=int,
    default=1,
  )
  alias_candidates.add_argument(
    "--max-candidates",
    type=int,
    default=200,
  )
  weak_relation_graph = subparsers.add_parser(
    "weak-relation-graph",
    help="Build a time-scoped weak relation graph from alias observations.",
  )
  weak_relation_graph.add_argument(
    "observations",
    type=Path,
    help="Path to alias_weak_label_observations.jsonl.",
  )
  weak_relation_graph.add_argument("--output", type=Path)
  weak_relation_graph.add_argument("--suggestions-output", type=Path)
  weak_relation_graph.add_argument(
    "--exclude-training-split",
    action="append",
    dest="excluded_training_splits",
    default=None,
    help=(
      "Split whose evidence may appear in the graph but should not create "
      "soft target suggestions. May be repeated."
    ),
  )
  weak_target_preview = subparsers.add_parser(
    "weak-target-preview",
    help="Preview training-row coverage for soft weak target suggestions.",
  )
  weak_target_preview.add_argument(
    "rolling_result",
    type=Path,
    help=(
      "Path to rolling_evaluation_results.json, or a rolling-eval output directory."
    ),
  )
  weak_target_preview.add_argument(
    "--weak-relation-graph",
    type=Path,
    required=True,
  )
  weak_target_preview.add_argument(
    "--soft-target-suggestions",
    type=Path,
    required=True,
  )
  weak_target_preview.add_argument("--output", type=Path)
  weak_target_preview.add_argument(
    "--target-source",
    default="source_archetype_name_proxy",
  )
  weak_relation_state = subparsers.add_parser(
    "weak-relation-state",
    help="Build leakage-safe weak relation state from weak evidence artifacts.",
  )
  weak_relation_state.add_argument(
    "artifact_dir",
    type=Path,
    help="Directory containing weak relation artifacts.",
  )
  weak_relation_state.add_argument("--alias-observations", type=Path)
  weak_relation_state.add_argument("--weak-relation-graph", type=Path)
  weak_relation_state.add_argument("--soft-target-suggestions", type=Path)
  weak_relation_state.add_argument("--weak-target-preview", type=Path)
  weak_relation_state.add_argument("--output", type=Path)

  args = parser.parse_args()

  if args.command == "dataset":
    options = DatasetExportOptions(
      format_code=args.format,
      start=args.start,
      end=args.end,
      output=args.output,
      dataset_version=args.dataset_version,
      train_end=args.train_end,
      validation_end=args.validation_end,
      dev_test_end=args.dev_test_end,
      env_file=args.env_file,
      limit_events=args.limit_events,
    )
    summary = export_dataset(options)
    print(f"Wrote {summary.dataset_version} to {summary.output}")
    print(
      "Rows: "
      f"{summary.deck_count} decks, "
      f"{summary.deck_card_count} deck-card entries, "
      f"{summary.card_count} canonical cards, "
      f"{summary.proxy_target_count} proxy targets"
    )
  elif args.command == "check":
    dataset_path = args.dataset
    if not dataset_path.is_absolute():
      dataset_path = repo_root() / dataset_path

    result = verify_dataset_export(dataset_path)
    print(f"Verified {result.dataset_version} at {result.dataset_path}")
    print(
      "Files: "
      f"{result.checked_artifacts} total, "
      f"{result.checked_parquet_artifacts} Parquet"
    )
  elif args.command == "model-train":
    dataset_path = args.dataset
    if not dataset_path.is_absolute():
      dataset_path = repo_root() / dataset_path

    output = args.output
    if output is None:
      output = dataset_path / "model_runs" / "training_results.json"
    elif not output.is_absolute():
      output = repo_root() / output
    taxonomy_eval = _repo_path_or_none(args.taxonomy_eval)
    canonical_targets = _repo_path_or_none(args.canonical_targets)

    result = run_model_training(
      dataset_path,
      output=output,
      model_names=tuple(args.model or (MODEL_ALL,)),
      pooling=args.pooling,
      target_source=args.target_source,
      target_label_level=args.target_label_level,
      canonical_targets=canonical_targets,
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
      model_artifact_output=(
        _repo_path_or_none(args.model_artifact_output)
        if args.model_artifact_output is not None
        else None
      ),
      model_artifact_model_name=args.model_artifact_model,
      model_artifact_seed_policy=args.model_artifact_seed_policy,
    )
    print(f"Wrote model training result to {output}")
    print(f"Status: {result['status']}")
    if result["status"] == "completed":
      for model_name, model_result in result["models"].items():
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
  elif args.command == "model-score":
    model_artifact = args.model_artifact
    if not model_artifact.is_absolute():
      model_artifact = repo_root() / model_artifact
    dataset_path = args.dataset
    if not dataset_path.is_absolute():
      dataset_path = repo_root() / dataset_path
    output = args.output
    if not output.is_absolute():
      output = repo_root() / output
    deck_embedding_output = args.deck_embedding_output
    if deck_embedding_output is not None and not deck_embedding_output.is_absolute():
      deck_embedding_output = repo_root() / deck_embedding_output
    result = run_model_scoring(
      model_artifact=model_artifact,
      dataset_path=dataset_path,
      output=output,
      target_source=args.target_source,
      batch_size=args.batch_size,
      top_k=args.top_k,
      low_confidence_threshold=args.low_confidence_threshold,
      device=args.device,
      deck_embedding_output=deck_embedding_output,
      taxonomy_eval_version=args.taxonomy_eval_version,
    )
    print(f"Wrote model predictions to {output}")
    if result.get("deck_embedding_output"):
      print(f"Wrote deck embeddings to {result['deck_embedding_output']}")
    print(f"Prediction count: {result['prediction_count']}")
    print(f"Model version: {result['model_version']}")
  elif args.command == "rolling-eval":
    output = args.output
    if not output.is_absolute():
      output = repo_root() / output
    taxonomy_eval = _repo_path_or_none(args.taxonomy_eval)
    canonical_targets = _repo_path_or_none(args.canonical_targets)

    result = run_rolling_evaluation(
      windows=tuple(args.window),
      output=output,
      format_code=args.format,
      target_source=args.target_source,
      target_label_level=args.target_label_level,
      canonical_targets=canonical_targets,
      env_file=args.env_file,
      limit_events=args.limit_events,
      epochs=args.epochs,
      learning_rate=args.learning_rate,
      pooled_learning_rate=args.pooled_learning_rate,
      neural_learning_rate=args.neural_learning_rate,
      weight_decay=args.weight_decay,
      deepsets_regularized_weight_decay=args.deepsets_regularized_weight_decay,
      seed=args.seed,
      seeds=args.seeds,
      embedding_dim=args.embedding_dim,
      hidden_dim=args.hidden_dim,
      batch_size=args.batch_size,
      shuffle=args.shuffle,
      max_steps=args.max_steps,
      device=args.device,
      prediction_output=args.prediction_output,
      taxonomy_eval=taxonomy_eval,
    )
    print(f"Wrote rolling evaluation result to {output / 'rolling_evaluation_results.json'}")
    print(f"Status: {result['status']}")
    for window in result["windows"]:
      print(
        f"{window['window']['name']}: "
        f"{window['primary_evaluation_split']} "
        f"split counts {window['split_counts']}"
      )
    for model_name, model_summary in result["summary_by_model"].items():
      metrics = model_summary["metrics"]
      print(
        f"{model_name} across windows: "
        f"primary accuracy {_mean_std(metrics, 'primary_accuracy')}, "
        f"closed-set {_mean_std(metrics, 'primary_closed_set_accuracy')}, "
        f"top-3 {_mean_std(metrics, 'primary_top_3_accuracy')}, "
        f"macro-F1 {_mean_std(metrics, 'primary_macro_f1')}, "
        f"source-label accuracy "
        f"{_mean_std(metrics, 'primary_source_label_accuracy')}, "
        f"canonical accuracy {_mean_std(metrics, 'primary_canonical_accuracy')}, "
        f"canonical closed-set "
        f"{_mean_std(metrics, 'primary_canonical_closed_set_accuracy')}, "
        f"temp NLL {_mean_std(metrics, 'primary_temperature_scaled_nll')}, "
        f"temp Brier {_mean_std(metrics, 'primary_temperature_scaled_brier')}, "
        f"temp ECE {_mean_std(metrics, 'primary_temperature_scaled_ece')}, "
        f"MSP AUROC {_mean_std(metrics, 'primary_msp_auroc')}, "
        f"energy AUROC {_mean_std(metrics, 'primary_energy_auroc')}"
      )
  elif args.command == "alias-candidates":
    rolling_result = args.rolling_result
    if rolling_result is not None and not rolling_result.is_absolute():
      rolling_result = repo_root() / rolling_result
    predictions = args.predictions
    if predictions is not None and not predictions.is_absolute():
      predictions = repo_root() / predictions
    predictions_manifest = args.predictions_manifest
    if predictions_manifest is not None and not predictions_manifest.is_absolute():
      predictions_manifest = repo_root() / predictions_manifest
    if predictions_manifest is None and predictions is not None:
      default_manifest = predictions.with_suffix(".manifest.json")
      if default_manifest.exists():
        predictions_manifest = default_manifest
    deck_embeddings = args.deck_embeddings
    if deck_embeddings is not None and not deck_embeddings.is_absolute():
      deck_embeddings = repo_root() / deck_embeddings
    dataset_path = args.dataset
    if dataset_path is not None and not dataset_path.is_absolute():
      dataset_path = repo_root() / dataset_path
    if predictions is not None and dataset_path is None:
      raise SystemExit("--dataset is required when --predictions is passed.")
    if predictions is None and rolling_result is None:
      raise SystemExit("Pass a rolling result path or --predictions with --dataset.")
    output = args.output
    if output is None:
      if predictions is not None:
        output = predictions.parent / "alias_candidates.json"
      else:
        output = (
          rolling_result / "alias_candidates.json"
          if rolling_result.is_dir()
          else rolling_result.parent / "alias_candidates.json"
        )
    elif not output.is_absolute():
      output = repo_root() / output
    taxonomy_eval = _repo_path_or_none(args.taxonomy_eval)
    result = run_alias_candidate_scoring(
      rolling_result,
      output=output,
      predictions=predictions,
      predictions_manifest=predictions_manifest,
      deck_embeddings=deck_embeddings,
      dataset_path=dataset_path,
      model_name=args.model,
      target_source=args.target_source,
      taxonomy_eval=taxonomy_eval,
      min_confusion_count=args.min_confusion_count,
      min_seed_confusion_count=args.min_seed_confusion_count,
      max_candidates=args.max_candidates,
    )
    print(f"Wrote alias candidates to {output}")
    if result.get("weak_label_observations_path"):
      print(
        "Wrote weak-label observations to "
        f"{result['weak_label_observations_path']}"
      )
    if result.get("backfill_report_path"):
      print(f"Wrote backfill report to {result['backfill_report_path']}")
    print(f"Candidate count: {result['candidate_count']}")
    print(f"Pair summary count: {result['pair_summary_count']}")
    print(f"Weak-label observation count: {result['weak_label_observation_count']}")
    if "unknown_review_candidate_count" in result:
      print(f"Unknown review candidate count: {result['unknown_review_candidate_count']}")
    for row in result["candidates"][:10]:
      evidence = row["evidence"]
      print(
        f"{row['rolling_window']} {row['split_name']}: "
        f"{row['label_a']} / {row['label_b']} "
        f"confidence {row['confidence']}, "
        f"relation {row['suggested_relation']}, "
        f"confusions {evidence['confusion_count']}, "
        f"seed consensus {evidence['confusion_seed_consensus']}, "
        f"main WJ {evidence['mainboard_adoption_weighted_jaccard']}"
      )
  elif args.command == "weak-relation-graph":
    observations = args.observations
    if not observations.is_absolute():
      observations = repo_root() / observations
    output = args.output
    if output is None:
      output = observations.with_name("weak_relation_graph.json")
    elif not output.is_absolute():
      output = repo_root() / output
    suggestions_output = args.suggestions_output
    if suggestions_output is not None and not suggestions_output.is_absolute():
      suggestions_output = repo_root() / suggestions_output
    excluded_training_splits = tuple(
      args.excluded_training_splits
      if args.excluded_training_splits is not None
      else DEFAULT_EXCLUDED_TRAINING_SPLITS
    )
    result = run_weak_relation_graph(
      observations,
      output=output,
      suggestions_output=suggestions_output,
      excluded_training_splits=excluded_training_splits,
    )
    print(f"Wrote weak relation graph to {output}")
    print(
      "Wrote soft canonical target suggestions to "
      f"{result['soft_canonical_target_suggestions_path']}"
    )
    print(f"Observation count: {result['observation_count']}")
    print(f"Node count: {result['node_count']}")
    print(f"Edge count: {result['edge_count']}")
    print(
      "Soft canonical target suggestion count: "
      f"{result['soft_canonical_target_suggestion_count']}"
    )
  elif args.command == "weak-target-preview":
    rolling_result = args.rolling_result
    if not rolling_result.is_absolute():
      rolling_result = repo_root() / rolling_result
    weak_relation_graph = args.weak_relation_graph
    if not weak_relation_graph.is_absolute():
      weak_relation_graph = repo_root() / weak_relation_graph
    soft_target_suggestions = args.soft_target_suggestions
    if not soft_target_suggestions.is_absolute():
      soft_target_suggestions = repo_root() / soft_target_suggestions
    output = args.output
    if output is None:
      output = (
        rolling_result / "weak_target_preview.json"
        if rolling_result.is_dir()
        else rolling_result.parent / "weak_target_preview.json"
      )
    elif not output.is_absolute():
      output = repo_root() / output
    result = run_weak_target_preview(
      rolling_result,
      weak_relation_graph=weak_relation_graph,
      soft_target_suggestions=soft_target_suggestions,
      output=output,
      target_source=args.target_source,
    )
    print(f"Wrote weak target preview to {output}")
    print(f"Suggestion count: {result['suggestion_count']}")
    print(f"Preview row count: {result['preview_row_count']}")
    print(
      "Rows with train coverage: "
      f"{result['rows_with_train_coverage_count']}"
    )
    print(f"Affected rows by split: {result['affected_rows_by_split']}")
  elif args.command == "weak-relation-state":
    artifact_dir = args.artifact_dir
    if not artifact_dir.is_absolute():
      artifact_dir = repo_root() / artifact_dir
    alias_observations = _artifact_path(
      args.alias_observations,
      artifact_dir,
      "alias_weak_label_observations.jsonl",
    )
    weak_relation_graph = _artifact_path(
      args.weak_relation_graph,
      artifact_dir,
      "weak_relation_graph.json",
    )
    soft_target_suggestions = _artifact_path(
      args.soft_target_suggestions,
      artifact_dir,
      "soft_canonical_target_suggestions.jsonl",
    )
    weak_target_preview = _artifact_path(
      args.weak_target_preview,
      artifact_dir,
      "weak_target_preview.json",
    )
    output = args.output
    if output is None:
      output = artifact_dir / "weak_relation_state.json"
    elif not output.is_absolute():
      output = repo_root() / output
    result = run_weak_relation_state(
      alias_observations=alias_observations,
      weak_relation_graph=weak_relation_graph,
      soft_target_suggestions=soft_target_suggestions,
      weak_target_preview=weak_target_preview,
      output=output,
    )
    print(f"Wrote weak relation state to {output}")
    print(f"Usable now: {result['counts']['usable_now']}")
    print(f"Review queue: {result['counts']['review_queue']}")
    print(
      "Deferred soft canonical evidence: "
      f"{result['counts']['deferred_soft_canonical_evidence']}"
    )
    print(
      "Deferred review diagnostics: "
      f"{result['counts']['deferred_review_diagnostics']}"
    )
    print(f"Blocked: {result['counts']['blocked']}")


def _date(value: str) -> date:
  return date.fromisoformat(value)


def _artifact_path(path: Path | None, artifact_dir: Path, filename: str) -> Path:
  if path is None:
    return artifact_dir / filename
  if path.is_absolute():
    return path
  return repo_root() / path


def _seeds(value: str) -> tuple[int, ...]:
  try:
    seeds = tuple(
      int(seed.strip())
      for seed in value.split(",")
      if seed.strip()
    )
  except ValueError as error:
    raise argparse.ArgumentTypeError(
      "seeds must be comma-separated integers"
    ) from error
  if not seeds:
    raise argparse.ArgumentTypeError("at least one seed is required")
  return seeds


def _rolling_window(value: str):
  try:
    return parse_rolling_window(value)
  except ValueError as error:
    raise argparse.ArgumentTypeError(str(error)) from error


def _repo_path_or_none(path: Path | None) -> Path | None:
  if path is None or path.is_absolute():
    return path
  return repo_root() / path


def _mean_std(summary: dict[str, object], metric_name: str) -> str:
  metric = summary.get(metric_name)
  if not isinstance(metric, dict):
    return "n/a"
  mean = metric.get("mean")
  std = metric.get("std")
  if mean is None or std is None:
    return "n/a"
  return f"{mean:.4f} +/- {std:.4f}"


if __name__ == "__main__":
  main()

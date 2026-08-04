from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from manafold.cli.common import (
  SubparserRegistry,
  _mean_std,
  _repo_path_or_none,
  _rolling_window,
  _seeds,
)
from manafold.constants import PROJECT_ROOT
from manafold.models.training.rolling_evaluation import evaluate_rolling_windows
from manafold.models.training.training_pipeline import (
  DEFAULT_LEARNING_RATE,
  DEFAULT_REGULARIZED_WEIGHT_DECAY,
  PREDICTION_OUTPUT_ERRORS,
  PREDICTION_OUTPUT_MODES,
  TARGET_LABEL_LEVEL_SOURCE,
  TARGET_LABEL_LEVELS,
)


def register(subparsers: SubparserRegistry) -> None:
  parser = subparsers.add_parser(
    "rolling-eval",
    help="Run the reference baselines across time windows.",
  )
  parser.add_argument(
    "--window",
    action="append",
    required=True,
    type=_rolling_window,
    help=(
      "Window as name,start,train_end,validation_end,dev_test_end,end. "
      "Repeat for multiple time slices."
    ),
  )
  parser.add_argument(
    "--format", default="modern", help="Format code to export."
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("data/rolling_eval"),
    help="Directory for the rolling summary and per-window results.",
  )
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
      "Training label level. Use canonical-family only for reviewed target"
      " maps."
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
  parser.add_argument("--env-file", type=Path)
  parser.add_argument(
    "--limit-events",
    type=int,
    help="Export only the first N events per window for a small sample.",
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
  )
  parser.add_argument("--seed", type=int, default=13)
  parser.add_argument(
    "--seeds",
    type=_seeds,
    help="Comma-separated seeds, such as 13,17,23. Overrides --seed.",
  )
  parser.add_argument("--embedding-dim", type=int, default=32)
  parser.add_argument("--hidden-dim", type=int, default=64)
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument(
    "--max-steps",
    type=int,
    help="Train for this many optimizer updates per model/seed/window.",
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
  parser.set_defaults(handler=run)


def run(args: Namespace) -> None:
  output = args.output
  if not output.is_absolute():
    output = PROJECT_ROOT / output

  taxonomy_eval = _repo_path_or_none(args.taxonomy_eval)
  canonical_targets = _repo_path_or_none(args.canonical_targets)

  result = evaluate_rolling_windows(
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
  print(
    "Wrote rolling evaluation result to"
    f" {output / 'rolling_evaluation_results.json'}"
  )
  print(f"Status: {result['status']}")

  for window in result["windows"]:
    print(
      f"{window['window']['name']}: {window['primary_evaluation_split']}"
      f" split counts {window['split_counts']}"
    )

  for model_name, model_summary in result["summary_by_model"].items():
    metrics = model_summary["metrics"]
    print(
      f"{model_name} across windows: "
      f"primary accuracy {_mean_std(metrics, 'primary_accuracy')}, "
      f"closed-set {_mean_std(metrics, 'primary_closed_set_accuracy')}, "
      f"top-3 {_mean_std(metrics, 'primary_top_3_accuracy')}, "
      f"macro-F1 {_mean_std(metrics, 'primary_macro_f1')}, "
      "source-label accuracy "
      f"{_mean_std(metrics, 'primary_source_label_accuracy')}, "
      f"canonical accuracy {_mean_std(metrics, 'primary_canonical_accuracy')}, "
      "canonical closed-set "
      f"{_mean_std(metrics, 'primary_canonical_closed_set_accuracy')}, "
      f"temp NLL {_mean_std(metrics, 'primary_temperature_scaled_nll')}, "
      f"temp Brier {_mean_std(metrics, 'primary_temperature_scaled_brier')}, "
      f"temp ECE {_mean_std(metrics, 'primary_temperature_scaled_ece')}, "
      f"MSP AUROC {_mean_std(metrics, 'primary_msp_auroc')}, "
      f"energy AUROC {_mean_std(metrics, 'primary_energy_auroc')}"
    )

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from manafold.cli.common import SubparserRegistry, _repo_path_or_none
from manafold.constants import PROJECT_ROOT
from manafold.taxonomy.aliases import (
  DEFAULT_ALIAS_MODEL,
  run_alias_candidate_scoring,
)


def register(subparsers: SubparserRegistry) -> None:
  parser = subparsers.add_parser(
    "alias-candidates",
    help="Score source-label alias and review candidates from predictions.",
  )
  parser.add_argument(
    "rolling_result",
    nargs="?",
    type=Path,
    help=(
      "Path to rolling_evaluation_results.json, or a rolling-eval output"
      " directory."
    ),
  )
  parser.add_argument(
    "--predictions",
    type=Path,
    help="Prediction parquet from model-score for production-backed evidence.",
  )
  parser.add_argument(
    "--predictions-manifest",
    type=Path,
    help="Optional model-score manifest JSON. Defaults next to --predictions.",
  )
  parser.add_argument(
    "--deck-embeddings",
    type=Path,
    help="Optional deck embeddings parquet from model-score.",
  )
  parser.add_argument(
    "--dataset",
    type=Path,
    help="Dataset export used by the prediction parquet.",
  )
  parser.add_argument("--output", type=Path)
  parser.add_argument(
    "--model",
    default=DEFAULT_ALIAS_MODEL,
    help="Model family whose prediction errors should seed confusion evidence.",
  )
  parser.add_argument(
    "--target-source",
    default="source_archetype_name_proxy",
  )
  parser.add_argument(
    "--taxonomy-eval",
    type=Path,
    help="Optional seed alias evidence file.",
  )
  parser.add_argument(
    "--min-confusion-count",
    type=int,
    default=5,
  )
  parser.add_argument(
    "--min-seed-confusion-count",
    type=int,
    default=1,
  )
  parser.add_argument(
    "--max-candidates",
    type=int,
    default=200,
  )
  parser.set_defaults(handler=run)


def run(args: Namespace) -> None:
  rolling_result = args.rolling_result
  if rolling_result is not None and not rolling_result.is_absolute():
    rolling_result = PROJECT_ROOT / rolling_result

  predictions = args.predictions
  if predictions is not None and not predictions.is_absolute():
    predictions = PROJECT_ROOT / predictions

  predictions_manifest = args.predictions_manifest
  if (
    predictions_manifest is not None
    and not predictions_manifest.is_absolute()
  ):
    predictions_manifest = PROJECT_ROOT / predictions_manifest
  if predictions_manifest is None and predictions is not None:
    default_manifest = predictions.with_suffix(".manifest.json")
    if default_manifest.exists():
      predictions_manifest = default_manifest

  deck_embeddings = args.deck_embeddings
  if deck_embeddings is not None and not deck_embeddings.is_absolute():
    deck_embeddings = PROJECT_ROOT / deck_embeddings

  dataset_path = args.dataset
  if dataset_path is not None and not dataset_path.is_absolute():
    dataset_path = PROJECT_ROOT / dataset_path
  if predictions is not None and dataset_path is None:
    raise SystemExit("--dataset is required when --predictions is passed.")
  if predictions is None and rolling_result is None:
    raise SystemExit(
      "Pass a rolling result path or --predictions with --dataset."
    )

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
    output = PROJECT_ROOT / output

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
  print(
    f"Weak-label observation count: {result['weak_label_observation_count']}"
  )

  if "unknown_review_candidate_count" in result:
    print(
      "Unknown review candidate count: "
      f"{result['unknown_review_candidate_count']}"
    )
    print(
      "Unknown review candidate count: "
      f"{result['unknown_review_candidate_count']}"
    )

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

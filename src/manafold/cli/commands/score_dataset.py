from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from manafold.cli.common import SubparserRegistry, _repo_path_or_none
from manafold.models.dataset_scoring import score_dataset


def register(subparsers: SubparserRegistry) -> None:
  parser = subparsers.add_parser(
    "model-score",
    help="Score every deck in a dataset with a saved Manafold model.",
  )
  parser.add_argument("--saved-model", type=Path, required=True)
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--target-source",
    default="source_archetype_name_proxy",
  )
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument("--top-k", type=int, default=3)
  parser.add_argument(
    "--low-confidence-threshold",
    type=float,
    default=0.5,
  )
  parser.add_argument(
    "--device",
    default="auto",
    help="Torch device such as auto, cpu, cuda, or cuda:0.",
  )
  parser.add_argument("--deck-embedding-output", type=Path)
  parser.add_argument("--taxonomy-eval-version")
  parser.set_defaults(handler=run)


def run(args: Namespace) -> None:
  saved_model_dir = _repo_path_or_none(args.saved_model)
  dataset_path = _repo_path_or_none(args.dataset)
  output = _repo_path_or_none(args.output)
  deck_embedding_output = _repo_path_or_none(args.deck_embedding_output)

  result = score_dataset(
    saved_model_dir=saved_model_dir,
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

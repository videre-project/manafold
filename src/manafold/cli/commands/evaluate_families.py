from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from manafold.cli.common import SubparserRegistry, _repo_path_or_none
from manafold.models.evaluation.family_classification import (
  evaluate_family_classification,
)


def register(subparsers: SubparserRegistry) -> None:
  parser = subparsers.add_parser(
    "family-eval",
    help="Evaluate a saved model under the A11 family policy.",
  )
  parser.add_argument("--saved-model", type=Path, required=True)
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--family-relations", type=Path)
  parser.add_argument("--split-name", default="final-test")
  parser.add_argument("--partial-identity-count", type=int)
  parser.add_argument("--partial-seed", type=int, default=13)
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument(
    "--device",
    default="auto",
    help="Torch device such as auto, cpu, cuda, or cuda:0.",
  )
  parser.set_defaults(handler=run)


def run(args: Namespace) -> None:
  saved_model_dir = _repo_path_or_none(args.saved_model)
  dataset_path = _repo_path_or_none(args.dataset)
  output = _repo_path_or_none(args.output)
  family_relations = _repo_path_or_none(args.family_relations)
  report = evaluate_family_classification(
    saved_model_dir=saved_model_dir,
    dataset_path=dataset_path,
    output=output,
    family_relations=family_relations,
    split_name=args.split_name,
    partial_identity_count=args.partial_identity_count,
    partial_seed=args.partial_seed,
    batch_size=args.batch_size,
    device=args.device,
  )
  metrics = report["metrics"]
  print(f"Wrote family evaluation to {output}")
  print(
    f"Family accuracy: {metrics['accuracy']}, "
    f"macro-F1: {metrics['macro_f1']}, "
    f"top-3: {metrics['top3_accuracy']}"
  )

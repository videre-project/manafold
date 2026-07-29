from __future__ import annotations

import argparse
import os
from pathlib import Path

from manafold.models.family_evaluation import evaluate_family_backoff


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Evaluate a model artifact through serving-time family backoff.",
  )
  parser.add_argument("--model-artifact", type=Path, required=True)
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--family-relations", type=Path)
  parser.add_argument("--split-name", default="test")
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument("--device", default="auto")
  args = parser.parse_args()
  workspace = Path(
    os.environ.get("BUILD_WORKSPACE_DIRECTORY", Path.cwd())
  ).resolve()

  def resolve_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
      return path
    return (workspace / path).resolve()

  report = evaluate_family_backoff(
    model_artifact=resolve_path(args.model_artifact),
    dataset_path=resolve_path(args.dataset),
    output=resolve_path(args.output),
    family_relations=resolve_path(args.family_relations),
    split_name=args.split_name,
    batch_size=args.batch_size,
    device=args.device,
  )
  metrics = report["metrics"]
  print(
    f"{report['format']}: "
    f"family accuracy={metrics['accuracy']:.4f}, "
    f"family macro-F1={metrics['macro_f1']:.4f}, "
    f"family top-3={metrics['top3_accuracy']:.4f}"
  )


if __name__ == "__main__":
  main()

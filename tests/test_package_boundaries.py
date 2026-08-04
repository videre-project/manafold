from __future__ import annotations

import ast
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "manafold"

# Datasets and taxonomy are independent inputs to the model lifecycle. Within
# models/, dependencies flow from features to classifiers, saved-model scoring,
# evaluation, and finally training.
FORBIDDEN_IMPORTS = {
  ("datasets",): (
    "manafold.models",
    "manafold.taxonomy",
  ),
  ("taxonomy",): ("manafold.models",),
  ("models", "features"): (
    "manafold.models.classifiers",
    "manafold.models.saved_model",
    "manafold.models.dataset_scoring",
    "manafold.models.evaluation",
    "manafold.models.training",
    "manafold.taxonomy",
  ),
  ("models", "classifiers"): (
    "manafold.models.saved_model",
    "manafold.models.dataset_scoring",
    "manafold.models.evaluation",
    "manafold.models.training",
    "manafold.taxonomy",
  ),
  ("models", "evaluation"): ("manafold.models.training",),
}


class PackageBoundaryTests(unittest.TestCase):
  def test_domain_imports_follow_dependency_direction(self) -> None:
    violations: list[str] = []
    for package_path, forbidden_prefixes in FORBIDDEN_IMPORTS.items():
      for path in sorted(PACKAGE_ROOT.joinpath(*package_path).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
          imported_module = _imported_module(node)
          if imported_module is None:
            continue
          if imported_module.startswith(forbidden_prefixes):
            relative_path = path.relative_to(PACKAGE_ROOT.parent)
            violations.append(
              f"{relative_path}:{node.lineno} imports {imported_module}"
            )

    self.assertEqual([], violations, "\n".join(violations))


def _imported_module(node: ast.AST) -> str | None:
  if isinstance(node, ast.ImportFrom):
    return node.module
  if isinstance(node, ast.Import) and node.names:
    return node.names[0].name
  return None


if __name__ == "__main__":
  unittest.main()

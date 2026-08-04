"""Command modules registered by the Manafold CLI."""

from manafold.cli.commands import (
  alias_candidates,
  build_dataset as build_dataset_command,
  check,
  evaluate_families,
  evaluate_rolling_windows,
  family_targets,
  score_dataset,
  train_models,
  weak_relation_graph,
  weak_relation_state,
  weak_target_preview,
)


COMMAND_MODULES = (
  build_dataset_command,
  check,
  family_targets,
  train_models,
  score_dataset,
  evaluate_families,
  evaluate_rolling_windows,
  alias_candidates,
  weak_relation_graph,
  weak_target_preview,
  weak_relation_state,
)

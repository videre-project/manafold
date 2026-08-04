from __future__ import annotations

from dataclasses import replace
import math
import random

from manafold.datasets.model_inputs import DeckModelInput
from manafold.models.classifiers.config import (
  DEFAULT_BALANCED_SAMPLING_FRACTION,
  DEFAULT_BALANCED_SAMPLING_MAX_MULTIPLIER,
  DEFAULT_PARTIAL_OBSERVATION_IDENTITY_COUNTS,
  DEFAULT_PARTIAL_TEACHER_DECAY,
  PARTIAL_CORRUPTION_FIXED,
  PARTIAL_CORRUPTION_MIXTURE,
)


def max_quantity(examples: list[DeckModelInput]) -> int:
  values = [token.quantity for example in examples for token in example.tokens]
  return max(values, default=0)


def partial_observation_training_views(
  examples: list[DeckModelInput],
  *,
  rng: random.Random,
  identity_counts: tuple[int, ...] = DEFAULT_PARTIAL_OBSERVATION_IDENTITY_COUNTS,
  corruption_policy: str = PARTIAL_CORRUPTION_FIXED,
  card_information: dict[int, float] | None = None,
) -> list[DeckModelInput]:
  """Sample one token-identity subset per full example for paired training."""
  views: list[DeckModelInput] = []
  for example in examples:
    token_count = len(example.tokens)
    if token_count <= 1:
      views.append(example)
      continue
    if corruption_policy == PARTIAL_CORRUPTION_FIXED:
      observed_count = _partial_identity_count(
        token_count,
        identity_counts=identity_counts,
        rng=rng,
      )
      selected = set(rng.sample(range(token_count), observed_count))
    elif corruption_policy == PARTIAL_CORRUPTION_MIXTURE:
      mode = rng.random()
      if mode < 0.25:
        observed_count = max(
          1,
          min(
            token_count - 1,
            round(token_count * rng.uniform(0.5, 0.8)),
          ),
        )
        selected = set(rng.sample(range(token_count), observed_count))
      elif mode < 0.75:
        observed_count = _partial_identity_count(
          token_count,
          identity_counts=(5, 10),
          rng=rng,
        )
        selected = set(rng.sample(range(token_count), observed_count))
      else:
        observed_count = _partial_identity_count(
          token_count,
          identity_counts=identity_counts,
          rng=rng,
        )
        selected = _weighted_token_sample(
          example,
          observed_count=observed_count,
          card_information=card_information or {},
          rng=rng,
        )
    else:
      raise ValueError(
        f"Unsupported partial corruption policy: {corruption_policy}"
      )
    expected_size = example.expected_mainboard_size or sum(
      max(token.quantity, 0) for token in example.tokens
    )
    views.append(
      replace(
        example,
        tokens=tuple(
          token
          for index, token in enumerate(example.tokens)
          if index in selected
        ),
        expected_mainboard_size=expected_size,
        observation_complete=False,
      )
    )
  return views


def card_identity_idf(examples: list[DeckModelInput]) -> dict[int, float]:
  deck_count = len(examples)
  support: dict[int, int] = {}
  for example in examples:
    for card_idx in {token.card_idx for token in example.tokens}:
      support[card_idx] = support.get(card_idx, 0) + 1
  return {
    card_idx: math.log((deck_count + 1.0) / (count + 1.0)) + 1.0
    for card_idx, count in support.items()
  }


def mixed_class_balanced_epoch_indexes(
  examples: list[DeckModelInput],
  *,
  rng: random.Random,
  balanced_fraction: float = DEFAULT_BALANCED_SAMPLING_FRACTION,
  max_multiplier: float = DEFAULT_BALANCED_SAMPLING_MAX_MULTIPLIER,
) -> list[int]:
  """Mix natural examples with capped square-root class-balanced resampling."""
  if not 0.0 <= balanced_fraction <= 1.0:
    raise ValueError("balanced_fraction must be in [0.0, 1.0].")
  if max_multiplier < 1.0:
    raise ValueError("max_multiplier must be at least 1.0.")
  if not examples:
    return []

  support_by_label: dict[str, int] = {}
  for example in examples:
    support_by_label[example.target_label_id] = (
      support_by_label.get(example.target_label_id, 0) + 1
    )
  max_support = max(support_by_label.values())
  weights = [
    min(
      max_multiplier,
      math.sqrt(max_support / support_by_label[example.target_label_id]),
    )
    for example in examples
  ]

  balanced_count = round(len(examples) * balanced_fraction)
  natural_count = len(examples) - balanced_count
  natural_indexes = list(range(len(examples)))
  rng.shuffle(natural_indexes)
  epoch_indexes = natural_indexes[:natural_count]
  epoch_indexes.extend(
    rng.choices(
      range(len(examples)),
      weights=weights,
      k=balanced_count,
    )
  )
  rng.shuffle(epoch_indexes)
  return epoch_indexes


def _partial_identity_count(
  token_count: int,
  *,
  identity_counts: tuple[int, ...],
  rng: random.Random,
) -> int:
  eligible_counts = [count for count in identity_counts if count < token_count]
  return rng.choice(eligible_counts) if eligible_counts else token_count - 1


def _weighted_token_sample(
  example: DeckModelInput,
  *,
  observed_count: int,
  card_information: dict[int, float],
  rng: random.Random,
) -> set[int]:
  ranked = sorted(
    range(len(example.tokens)),
    key=lambda index: (
      rng.random()
      ** (
        1.0
        / max(card_information.get(example.tokens[index].card_idx, 1.0), 1e-8)
      )
    ),
    reverse=True,
  )
  return set(ranked[:observed_count])


def partial_observation_views_at_count(
  examples: list[DeckModelInput],
  *,
  identity_count: int,
  seed: int,
) -> list[DeckModelInput]:
  """Create deterministic partial views for evaluation at one identity count."""
  if identity_count <= 0:
    raise ValueError("identity_count must be positive.")
  views: list[DeckModelInput] = []
  for example in examples:
    if len(example.tokens) <= identity_count:
      continue
    example_rng = random.Random(f"{seed}:{example.deck_id}:{identity_count}")
    selected = set(example_rng.sample(range(len(example.tokens)), identity_count))
    expected_size = example.expected_mainboard_size or sum(
      max(token.quantity, 0) for token in example.tokens
    )
    views.append(
      replace(
        example,
        tokens=tuple(
          token
          for index, token in enumerate(example.tokens)
          if index in selected
        ),
        expected_mainboard_size=expected_size,
        observation_complete=False,
      )
    )
  return views


def _validate_partial_observation_config(
  *,
  enabled: bool,
  identity_counts: tuple[int, ...],
  classification_weight: float,
  consistency_weight: float,
  min_coverage_weight: float,
  latent_weight: float = 0.0,
  contextual_weight: float = 0.0,
  teacher_decay: float = DEFAULT_PARTIAL_TEACHER_DECAY,
) -> None:
  if any(count <= 0 for count in identity_counts):
    raise ValueError("Partial-observation identity counts must be positive.")
  if enabled and not identity_counts:
    raise ValueError("Partial-observation training requires identity counts.")
  if classification_weight < 0.0:
    raise ValueError("partial_classification_weight must be non-negative.")
  if consistency_weight < 0.0:
    raise ValueError("partial_consistency_weight must be non-negative.")
  if min_coverage_weight < 0.0 or min_coverage_weight > 1.0:
    raise ValueError("partial_min_coverage_weight must be in [0.0, 1.0].")
  if latent_weight < 0.0:
    raise ValueError("partial_latent_weight must be non-negative.")
  if contextual_weight < 0.0:
    raise ValueError("partial_contextual_weight must be non-negative.")
  if teacher_decay < 0.0 or teacher_decay >= 1.0:
    raise ValueError("partial_teacher_decay must be in [0.0, 1.0).")

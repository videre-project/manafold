from __future__ import annotations

from typing import Any, Protocol

from manafold.datasets.model_inputs import DeckModelInput
from manafold.models.classifiers.deepsets import DeepSetsClassifier
from manafold.models.classifiers.pooled_linear import PooledLinearClassifier
from manafold.models.classifiers.prototype import PrototypeClassifier
from manafold.models.classifiers.set_transformer import SetTransformerClassifier
from manafold.models.classifiers.partial_decks import (
  card_identity_idf,
  max_quantity,
  mixed_class_balanced_epoch_indexes,
  partial_observation_training_views,
  partial_observation_views_at_count,
)
from manafold.models.classifiers.input_batches import hypergeometric_quantity_weight

from manafold.models.classifiers.config import (
  POOLING_SUM,
  POOLING_MEAN,
  POOLING_QUANTITY_WEIGHTED,
  POOLING_MODES,
  SET_TRANSFORMER_POOLING_PMA,
  SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED,
  SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
  SET_TRANSFORMER_POOLING_MODES,
  DEFAULT_HYPERGEOMETRIC_DRAW_COUNT,
  DEFAULT_PARTIAL_OBSERVATION_IDENTITY_COUNTS,
  DEFAULT_PARTIAL_CLASSIFICATION_WEIGHT,
  DEFAULT_PARTIAL_CONSISTENCY_WEIGHT,
  DEFAULT_PARTIAL_MIN_COVERAGE_WEIGHT,
  DEFAULT_PARTIAL_LATENT_WEIGHT,
  DEFAULT_PARTIAL_TEACHER_DECAY,
  PARTIAL_CORRUPTION_FIXED,
  PARTIAL_CORRUPTION_MIXTURE,
  PARTIAL_CORRUPTION_POLICIES,
  TRAINING_SAMPLING_NATURAL,
  TRAINING_SAMPLING_NATURAL_SQRT_BALANCED,
  TRAINING_SAMPLING_POLICIES,
  DEFAULT_BALANCED_SAMPLING_FRACTION,
  DEFAULT_BALANCED_SAMPLING_MAX_MULTIPLIER,
  PROTOTYPE_DISTANCE_EUCLIDEAN,
  DEEPSETS_ARCHITECTURE_BASE,
  DEEPSETS_ARCHITECTURE_PLUSPLUS,
  DEEPSETS_ARCHITECTURES,
)


class Classifier(Protocol):
  labels: tuple[str, ...]

  def fit(
    self,
    examples: list[DeckModelInput],
    *,
    validation_examples: list[DeckModelInput] | None = None,
    epochs: int = 40,
    batch_size: int = 32,
    shuffle: bool = True,
    max_steps: int | None = None,
  ) -> dict[str, Any]: ...

  def predict(self, example: DeckModelInput) -> tuple[str | None, float]: ...

  def predict_top_k(
    self,
    example: DeckModelInput,
    *,
    k: int,
  ) -> list[tuple[str, float]]: ...

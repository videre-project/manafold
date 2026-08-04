from __future__ import annotations

from typing import Any

import torch

from manafold.models.classifiers.prediction_statistics import (
  _empty_prediction_stats,
  _probability_entropy,
  _prototype_stats,
)
from manafold.models.classifiers.training_control import (
  _empty_training_summary,
  _sequential_index_batches,
)
from manafold.datasets.model_inputs import DeckModelInput
from manafold.models.classifiers.deepsets import DeepSetsClassifier
from manafold.models.classifiers.config import PROTOTYPE_DISTANCE_EUCLIDEAN
from manafold.models.classifiers.input_batches import (
  _prepare_set_examples,
  _prepared_set_batch,
)


class PrototypeClassifier:
  """Nearest-prototype classifier over a fitted Deep Sets deck encoder."""

  def __init__(
    self,
    *,
    encoder: DeepSetsClassifier,
    prototypes: torch.Tensor,
    prototype_counts: dict[str, int],
    distance: str = PROTOTYPE_DISTANCE_EUCLIDEAN,
  ) -> None:
    if distance != PROTOTYPE_DISTANCE_EUCLIDEAN:
      raise ValueError(f"Unsupported prototype distance: {distance}")
    if prototypes.shape[0] != len(encoder.labels):
      raise ValueError("Prototype count must match label count.")

    self.labels = encoder.labels
    self.encoder = encoder
    self.distance = distance
    self.prototype_counts = prototype_counts
    self.device = encoder.device
    self._label_to_idx = {
      label: index for index, label in enumerate(self.labels)
    }
    self._prototypes = prototypes.detach().to(self.device)

  @classmethod
  def from_deepsets(
    cls,
    encoder: DeepSetsClassifier,
    train_examples: list[DeckModelInput],
    *,
    batch_size: int = 32,
  ) -> "PrototypeClassifier":
    trainable = [
      example
      for example in train_examples
      if example.target_label_id in encoder._label_to_idx
    ]
    if not trainable:
      raise ValueError(
        "PrototypeClassifier requires at least one train example."
      )

    embeddings = encoder.deck_embedding_tensor(
      trainable, batch_size=batch_size
    )
    if embeddings.numel() == 0:
      raise ValueError(
        "PrototypeClassifier requires non-empty deck embeddings."
      )

    label_indexes = torch.tensor(
      [
        encoder._label_to_idx[example.target_label_id]
        for example in trainable
      ],
      dtype=torch.long,
      device=encoder.device,
    )
    sums = embeddings.new_zeros((len(encoder.labels), embeddings.shape[1]))
    counts = embeddings.new_zeros((len(encoder.labels), 1))
    sums.index_add_(0, label_indexes, embeddings)
    counts.index_add_(
      0,
      label_indexes,
      torch.ones(
        (len(trainable), 1),
        dtype=embeddings.dtype,
        device=encoder.device,
      ),
    )
    prototypes = sums / counts.clamp_min(1.0)
    prototype_counts = {
      label: int(counts[index].item())
      for index, label in enumerate(encoder.labels)
    }
    return cls(
      encoder=encoder,
      prototypes=prototypes,
      prototype_counts=prototype_counts,
    )

  def fit(
    self,
    examples: list[DeckModelInput],
    *,
    validation_examples: list[DeckModelInput] | None = None,
    epochs: int = 40,
    batch_size: int = 32,
    shuffle: bool = True,
    max_steps: int | None = None,
  ) -> dict[str, Any]:
    del examples, validation_examples, epochs, batch_size, shuffle, max_steps
    return _empty_training_summary()

  def predict(self, example: DeckModelInput) -> tuple[str | None, float]:
    top = self.predict_top_k(example, k=1)
    if not top:
      return None, 0.0
    return top[0]

  def predict_top_k(
    self,
    example: DeckModelInput,
    *,
    k: int,
  ) -> list[tuple[str, float]]:
    return self.predict_top_k_many([example], k=k)[0]

  def predict_top_k_many(
    self,
    examples: list[DeckModelInput],
    *,
    k: int,
    batch_size: int = 32,
  ) -> list[list[tuple[str, float]]]:
    return [
      row["top_predictions"]
      for row in self.predict_top_k_with_stats_many(
        examples,
        k=k,
        batch_size=batch_size,
      )
    ]

  def predict_top_k_with_stats_many(
    self,
    examples: list[DeckModelInput],
    *,
    k: int,
    batch_size: int = 32,
  ) -> list[dict[str, Any]]:
    if k <= 0:
      return [_empty_prediction_stats() for _ in examples]

    rows: list[dict[str, Any]] = []
    prepared = _prepare_set_examples(
      examples,
      label_to_idx=self._label_to_idx,
      quantity_count=self.encoder.quantity_count,
      require_targets=False,
    )
    self.encoder._network.eval()
    with torch.no_grad():
      for batch_indexes in _sequential_index_batches(
        len(examples), batch_size
      ):
        batch = _prepared_set_batch(
          prepared,
          batch_indexes,
          device=self.device,
        )
        embeddings = self.encoder._network.encode(batch)
        distances = torch.cdist(embeddings, self._prototypes, p=2)
        prototype_logits = -distances
        probabilities = torch.softmax(prototype_logits, dim=1)
        top_distances, indexes = torch.topk(
          distances,
          k=min(k, len(self.labels)),
          largest=False,
          dim=1,
        )
        top_probabilities = probabilities.gather(1, indexes)
        entropy = _probability_entropy(probabilities)
        energy = -torch.logsumexp(prototype_logits, dim=1)
        if len(self.labels) > 1:
          sorted_distances, _ = torch.sort(distances, dim=1)
          margins = sorted_distances[:, 1] - sorted_distances[:, 0]
        else:
          margins = distances.new_zeros((distances.shape[0],))
        rows.extend(
          _prototype_stats(
            top_probabilities=top_probabilities,
            top_indexes=indexes,
            top_distances=top_distances,
            entropy=entropy,
            energy=energy,
            nearest_distances=top_distances[:, 0],
            margins=margins,
            labels=self.labels,
          )
        )

    return rows

  def prototype_rows(self) -> list[dict[str, Any]]:
    prototypes = self._prototypes.detach().cpu().tolist()
    return [
      {
        "label_id": label,
        "train_support": self.prototype_counts.get(label, 0),
        "prototype": prototypes[index],
      }
      for index, label in enumerate(self.labels)
    ]

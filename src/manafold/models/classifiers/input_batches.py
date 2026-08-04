from __future__ import annotations

import math

import torch

from manafold.datasets.model_inputs import DeckModelInput
from manafold.models.classifiers.config import (
    DEFAULT_HYPERGEOMETRIC_DRAW_COUNT,
    SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
    SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED,
)
from manafold.models.features.card_packages import PackageFeatureSet


class _Batch:
    def __init__(
        self,
        *,
        card_idx: torch.Tensor,
        zone_idx: torch.Tensor,
        quantity_idx: torch.Tensor,
        quantity_weight: torch.Tensor,
        deck_idx: torch.Tensor,
        deck_count: int,
        target_idx: torch.Tensor,
        package_features: torch.Tensor,
        observation_coverage: torch.Tensor,
        observation_complete: torch.Tensor,
    ) -> None:
        self.card_idx = card_idx
        self.zone_idx = zone_idx
        self.quantity_idx = quantity_idx
        self.quantity_weight = quantity_weight
        self.deck_idx = deck_idx
        self.deck_count = deck_count
        self.target_idx = target_idx
        self.package_features = package_features
        self.observation_coverage = observation_coverage
        self.observation_complete = observation_complete


class _LinearBatch:
    def __init__(
        self,
        *,
        features: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> None:
        self.features = features
        self.target_idx = target_idx


class _PreparedLinearExamples:
    def __init__(
        self,
        *,
        examples: list[DeckModelInput],
        target_idx: torch.Tensor,
        feature_idx_by_deck: tuple[tuple[int, ...], ...],
        feature_value_by_deck: tuple[tuple[float, ...], ...],
    ) -> None:
        self.examples = examples
        self.target_idx = target_idx
        self.feature_idx_by_deck = feature_idx_by_deck
        self.feature_value_by_deck = feature_value_by_deck


class _DenseLinearExamples:
    def __init__(
        self,
        *,
        features: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> None:
        self.features = features
        self.target_idx = target_idx


class _PreparedSetExamples:
    def __init__(
        self,
        *,
        examples: list[DeckModelInput],
        target_idx: torch.Tensor,
        card_idx: torch.Tensor,
        zone_idx: torch.Tensor,
        quantity_idx: torch.Tensor,
        quantity_weight: torch.Tensor,
        token_start: torch.Tensor,
        token_end: torch.Tensor,
        package_idx_by_deck: tuple[tuple[int, ...], ...],
        package_count: int,
        observation_coverage: torch.Tensor,
        observation_complete: torch.Tensor,
    ) -> None:
        self.examples = examples
        self.target_idx = target_idx
        self.card_idx = card_idx
        self.zone_idx = zone_idx
        self.quantity_idx = quantity_idx
        self.quantity_weight = quantity_weight
        self.token_start = token_start
        self.token_end = token_end
        self.package_idx_by_deck = package_idx_by_deck
        self.package_count = package_count
        self.observation_coverage = observation_coverage
        self.observation_complete = observation_complete

    def to(self, device: torch.device | str) -> _PreparedSetExamples:
        device_obj = torch.device(device)
        if self.target_idx.device == device_obj:
            return self
        return _PreparedSetExamples(
            examples=self.examples,
            target_idx=self.target_idx.to(device_obj),
            card_idx=self.card_idx.to(device_obj),
            zone_idx=self.zone_idx.to(device_obj),
            quantity_idx=self.quantity_idx.to(device_obj),
            quantity_weight=self.quantity_weight.to(device_obj),
            token_start=self.token_start.to(device_obj),
            token_end=self.token_end.to(device_obj),
            package_idx_by_deck=self.package_idx_by_deck,
            package_count=self.package_count,
            observation_coverage=self.observation_coverage.to(device_obj),
            observation_complete=self.observation_complete.to(device_obj),
        )


def _prepare_set_examples(
    examples: list[DeckModelInput],
    *,
    label_to_idx: dict[str, int],
    quantity_count: int,
    require_targets: bool = True,
    package_features: PackageFeatureSet | None = None,
    quantity_weighting: str = SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED,
    hypergeometric_draw_count: int = DEFAULT_HYPERGEOMETRIC_DRAW_COUNT,
) -> _PreparedSetExamples:
    card_idx: list[int] = []
    zone_idx: list[int] = []
    quantity_idx: list[int] = []
    quantity_weight: list[float] = []
    token_start: list[int] = []
    token_end: list[int] = []
    target_idx: list[int] = []
    package_idx_by_deck: list[tuple[int, ...]] = []
    observation_coverage: list[float] = []
    observation_complete: list[bool] = []

    for example in examples:
        if require_targets:
            target_idx.append(label_to_idx[example.target_label_id])
        else:
            target_idx.append(0)
        token_start.append(len(card_idx))
        observed_size = sum(max(token.quantity, 0) for token in example.tokens)
        population_size = example.expected_mainboard_size or observed_size
        observation_coverage.append(
            min(float(observed_size) / float(population_size), 1.0)
            if population_size > 0
            else 0.0
        )
        observation_complete.append(example.observation_complete)
        for token in example.tokens:
            card_idx.append(token.card_idx)
            zone_idx.append(token.zone_idx)
            quantity_idx.append(max(0, min(token.quantity, quantity_count - 1)))
            if quantity_weighting == SET_TRANSFORMER_POOLING_HYPERGEOMETRIC:
                quantity_weight.append(
                    hypergeometric_quantity_weight(
                        token.quantity,
                        population_size=population_size,
                        draw_count=hypergeometric_draw_count,
                    )
                )
            else:
                quantity_weight.append(float(max(token.quantity, 0)))
        token_end.append(len(card_idx))
        if package_features is None:
            package_idx_by_deck.append(())
        else:
            package_idx_by_deck.append(package_features.activation_indexes(example))

    return _PreparedSetExamples(
        examples=examples,
        target_idx=torch.tensor(target_idx, dtype=torch.long),
        card_idx=torch.tensor(card_idx, dtype=torch.long),
        zone_idx=torch.tensor(zone_idx, dtype=torch.long),
        quantity_idx=torch.tensor(quantity_idx, dtype=torch.long),
        quantity_weight=torch.tensor(quantity_weight, dtype=torch.float32),
        token_start=torch.tensor(token_start, dtype=torch.long),
        token_end=torch.tensor(token_end, dtype=torch.long),
        package_idx_by_deck=tuple(package_idx_by_deck),
        package_count=0 if package_features is None else len(package_features),
        observation_coverage=torch.tensor(
            observation_coverage,
            dtype=torch.float32,
        ),
        observation_complete=torch.tensor(
            observation_complete,
            dtype=torch.bool,
        ),
    )


def hypergeometric_quantity_weight(
    quantity: int,
    *,
    population_size: int,
    draw_count: int = DEFAULT_HYPERGEOMETRIC_DRAW_COUNT,
) -> float:
    """Probability of seeing a card, normalized so one copy has weight one."""
    quantity = max(0, min(int(quantity), int(population_size)))
    population_size = int(population_size)
    if quantity == 0 or population_size <= 0 or draw_count <= 0:
        return 0.0
    draws = min(int(draw_count), population_size)
    miss_count = population_size - quantity
    miss_probability = (
        math.comb(miss_count, draws) / math.comb(population_size, draws)
        if miss_count >= draws
        else 0.0
    )
    singleton_probability = draws / population_size
    return (1.0 - miss_probability) / singleton_probability


def _prepared_set_batch(
    prepared: _PreparedSetExamples,
    deck_numbers: list[int],
    *,
    device: torch.device,
    package_scale: float = 1.0,
) -> _Batch:
    prep_device = prepared.target_idx.device
    deck_number_tensor = torch.tensor(
        deck_numbers, dtype=torch.long, device=prep_device)
    starts = prepared.token_start.index_select(0, deck_number_tensor)
    ends = prepared.token_end.index_select(0, deck_number_tensor)
    lengths = ends - starts
    total_tokens = int(lengths.sum().item())

    if total_tokens > 0:
        deck_idx = torch.repeat_interleave(
            torch.arange(len(deck_numbers), dtype=torch.long, device=prep_device),
            lengths,
        )
        offsets = torch.zeros(
            len(deck_numbers) + 1,
            dtype=torch.long,
            device=prep_device,
        )
        torch.cumsum(lengths, dim=0, out=offsets[1:])
        starts_per_token = torch.repeat_interleave(starts, lengths)
        deck_offsets = torch.repeat_interleave(offsets[:-1], lengths)
        token_positions = (
            torch.arange(total_tokens, dtype=torch.long, device=prep_device)
            - deck_offsets
        )
        token_indexes = starts_per_token + token_positions
    else:
        token_indexes = torch.empty(0, dtype=torch.long, device=prep_device)
        deck_idx = torch.empty(0, dtype=torch.long, device=prep_device)

    if prepared.package_count > 0:
        package_features = torch.zeros(
            (len(deck_numbers), prepared.package_count),
            dtype=torch.float32,
            device=device,
        )
        package_row_idx: list[int] = []
        package_col_idx: list[int] = []
        for row_number, deck_number in enumerate(deck_numbers):
            for package_idx in prepared.package_idx_by_deck[deck_number]:
                package_row_idx.append(row_number)
                package_col_idx.append(package_idx)
        if package_row_idx:
            package_features.index_put_(
                (
                    torch.tensor(package_row_idx, dtype=torch.long, device=device),
                    torch.tensor(package_col_idx, dtype=torch.long, device=device),
                ),
                torch.full(
                    (len(package_row_idx),),
                    float(package_scale),
                    dtype=torch.float32,
                    device=device,
                ),
            )
    else:
        package_features = None

    return _Batch(
        card_idx=prepared.card_idx.index_select(0, token_indexes).to(device),
        zone_idx=prepared.zone_idx.index_select(0, token_indexes).to(device),
        quantity_idx=prepared.quantity_idx.index_select(0, token_indexes).to(device),
        quantity_weight=prepared.quantity_weight.index_select(
            0, token_indexes,
        ).to(device),
        deck_idx=deck_idx.to(device),
        deck_count=len(deck_numbers),
        target_idx=prepared.target_idx.index_select(0, deck_number_tensor).to(device),
        package_features=package_features,
        observation_coverage=prepared.observation_coverage.index_select(
            0,
            deck_number_tensor,
        ).to(device),
        observation_complete=prepared.observation_complete.index_select(
            0,
            deck_number_tensor,
        ).to(device),
    )


def _tensor_batch(
    examples: list[DeckModelInput],
    *,
    label_to_idx: dict[str, int],
    quantity_count: int,
    device: torch.device,
    require_targets: bool = True,
) -> _Batch:
    card_idx: list[int] = []
    zone_idx: list[int] = []
    quantity_idx: list[int] = []
    quantity_weight: list[float] = []
    deck_idx: list[int] = []
    target_idx: list[int] = []

    for deck_number, example in enumerate(examples):
        if require_targets:
            target_idx.append(label_to_idx[example.target_label_id])
        else:
            target_idx.append(0)

        for token in example.tokens:
            card_idx.append(token.card_idx)
            zone_idx.append(token.zone_idx)
            quantity_idx.append(max(0, min(token.quantity, quantity_count - 1)))
            quantity_weight.append(float(max(token.quantity, 0)))
            deck_idx.append(deck_number)

    return _Batch(
        card_idx=torch.tensor(card_idx, dtype=torch.long, device=device),
        zone_idx=torch.tensor(zone_idx, dtype=torch.long, device=device),
        quantity_idx=torch.tensor(quantity_idx, dtype=torch.long, device=device),
        quantity_weight=torch.tensor(quantity_weight, dtype=torch.float32, device=device),
        deck_idx=torch.tensor(deck_idx, dtype=torch.long, device=device),
        deck_count=len(examples),
        target_idx=torch.tensor(target_idx, dtype=torch.long, device=device),
        package_features=torch.empty((len(examples), 0), dtype=torch.float32, device=device),
        observation_coverage=torch.tensor(
            [
                min(
                    sum(max(token.quantity, 0) for token in example.tokens)
                    / float(
                        example.expected_mainboard_size
                        or sum(max(token.quantity, 0) for token in example.tokens)
                        or 1
                    ),
                    1.0,
                )
                for example in examples
            ],
            dtype=torch.float32,
            device=device,
        ),
        observation_complete=torch.tensor(
            [example.observation_complete for example in examples],
            dtype=torch.bool,
            device=device,
        ),
    )


def _prepare_linear_examples(
    examples: list[DeckModelInput],
    *,
    label_to_idx: dict[str, int],
    zone_count: int,
    package_index_by_deck: dict[str, tuple[int, ...]],
    package_feature_offset: int,
    package_scale: float,
    include_card_features: bool,
    require_targets: bool = True,
) -> _PreparedLinearExamples:
    target_idx: list[int] = []
    feature_idx_by_deck: list[tuple[int, ...]] = []
    feature_value_by_deck: list[tuple[float, ...]] = []

    for example in examples:
        target_idx.append(
            label_to_idx[example.target_label_id] if require_targets else 0
        )
        feature_idx: list[int] = []
        values: list[float] = []
        total_quantity = sum(token.quantity for token in example.tokens)
        if include_card_features and total_quantity > 0:
            for token in example.tokens:
                feature_idx.append(token.card_idx * zone_count + token.zone_idx)
                values.append(token.quantity / total_quantity)

        for package_idx in package_index_by_deck.get(example.deck_id, ()):
            feature_idx.append(package_feature_offset + package_idx)
            values.append(package_scale)

        feature_idx_by_deck.append(tuple(feature_idx))
        feature_value_by_deck.append(tuple(values))

    return _PreparedLinearExamples(
        examples=examples,
        target_idx=torch.tensor(target_idx, dtype=torch.long),
        feature_idx_by_deck=tuple(feature_idx_by_deck),
        feature_value_by_deck=tuple(feature_value_by_deck),
    )


def _prepared_linear_batch(
    prepared: _PreparedLinearExamples,
    deck_numbers: list[int],
    *,
    feature_count: int,
    device: torch.device,
) -> _LinearBatch:
    row_idx: list[int] = []
    feature_idx: list[int] = []
    values: list[float] = []

    for row_number, deck_number in enumerate(deck_numbers):
        row_features = prepared.feature_idx_by_deck[deck_number]
        row_values = prepared.feature_value_by_deck[deck_number]
        row_idx.extend([row_number] * len(row_features))
        feature_idx.extend(row_features)
        values.extend(row_values)

    features = torch.zeros(
        (len(deck_numbers), feature_count),
        dtype=torch.float32,
        device=device,
    )
    if row_idx:
        features.index_put_(
            (
                torch.tensor(row_idx, dtype=torch.long, device=device),
                torch.tensor(feature_idx, dtype=torch.long, device=device),
            ),
            torch.tensor(values, dtype=torch.float32, device=device),
            accumulate=True,
        )

    deck_number_tensor = torch.tensor(deck_numbers, dtype=torch.long)
    return _LinearBatch(
        features=features,
        target_idx=prepared.target_idx.index_select(0, deck_number_tensor).to(device),
    )


def _dense_linear_examples(
    prepared: _PreparedLinearExamples,
    *,
    feature_count: int,
    device: torch.device,
) -> _DenseLinearExamples:
    row_idx: list[int] = []
    feature_idx: list[int] = []
    values: list[float] = []
    for row_number, row_features in enumerate(prepared.feature_idx_by_deck):
        row_values = prepared.feature_value_by_deck[row_number]
        row_idx.extend([row_number] * len(row_features))
        feature_idx.extend(row_features)
        values.extend(row_values)

    features = torch.zeros(
        (len(prepared.examples), feature_count),
        dtype=torch.float32,
        device=device,
    )
    if row_idx:
        features.index_put_(
            (
                torch.tensor(row_idx, dtype=torch.long, device=device),
                torch.tensor(feature_idx, dtype=torch.long, device=device),
            ),
            torch.tensor(values, dtype=torch.float32, device=device),
            accumulate=True,
        )

    return _DenseLinearExamples(
        features=features,
        target_idx=prepared.target_idx.to(device),
    )


def _dense_linear_batch(
    prepared: _DenseLinearExamples,
    deck_numbers: list[int],
) -> _LinearBatch:
    deck_number_tensor = torch.tensor(
        deck_numbers,
        dtype=torch.long,
        device=prepared.features.device,
    )
    return _LinearBatch(
        features=prepared.features.index_select(0, deck_number_tensor),
        target_idx=prepared.target_idx.index_select(0, deck_number_tensor),
    )


def _linear_batch(
    examples: list[DeckModelInput],
    *,
    label_to_idx: dict[str, int],
    feature_count: int,
    zone_count: int,
    device: torch.device,
    require_targets: bool = True,
    package_index_by_deck: dict[str, tuple[int, ...]] | None = None,
    base_feature_count: int | None = None,
    package_scale: float = 1.0,
) -> _LinearBatch:
    row_idx: list[int] = []
    feature_idx: list[int] = []
    values: list[float] = []
    target_idx: list[int] = []

    for row_number, example in enumerate(examples):
        target_idx.append(
            label_to_idx[example.target_label_id] if require_targets else 0
        )
        total_quantity = sum(token.quantity for token in example.tokens)
        if total_quantity <= 0:
            continue

        for token in example.tokens:
            row_idx.append(row_number)
            feature_idx.append(token.card_idx * zone_count + token.zone_idx)
            values.append(token.quantity / total_quantity)

        if package_index_by_deck is not None and base_feature_count is not None:
            for package_idx in package_index_by_deck.get(example.deck_id, ()):
                row_idx.append(row_number)
                feature_idx.append(base_feature_count + package_idx)
                values.append(package_scale)

    features = torch.zeros(
        (len(examples), feature_count),
        dtype=torch.float32,
        device=device,
    )
    if row_idx:
        features.index_put_(
            (
                torch.tensor(row_idx, dtype=torch.long, device=device),
                torch.tensor(feature_idx, dtype=torch.long, device=device),
            ),
            torch.tensor(values, dtype=torch.float32, device=device),
            accumulate=True,
        )

    return _LinearBatch(
        features=features,
        target_idx=torch.tensor(target_idx, dtype=torch.long, device=device),
    )

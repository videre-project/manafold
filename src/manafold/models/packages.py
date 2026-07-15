from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from manafold.models.data import ModelExample

PACKAGE_SCORING_SUPPORT_LIFT = "support-lift"
PACKAGE_SCORING_PRECISION_FILTERED_LIFT = "precision-filtered-lift"
PACKAGE_SCORING_ENTROPY_PENALIZED_LIFT = "entropy-penalized-lift"
PACKAGE_SCORING_BAYESIAN_LOG_ODDS = "bayesian-log-odds"
PACKAGE_SCORING_MODES = (
  PACKAGE_SCORING_SUPPORT_LIFT,
  PACKAGE_SCORING_PRECISION_FILTERED_LIFT,
  PACKAGE_SCORING_ENTROPY_PENALIZED_LIFT,
  PACKAGE_SCORING_BAYESIAN_LOG_ODDS,
)
PACKAGE_TYPE_MANABASE_PAIR = "manabase_pair"
PACKAGE_TYPE_SPELL_PAIR = "spell_pair"
PACKAGE_TYPE_MIXED_LAND_SPELL_PAIR = "mixed_land_spell_pair"
PACKAGE_TYPE_SIDEBOARD_PAIR = "sideboard_pair"
PACKAGE_TYPES = (
  PACKAGE_TYPE_MANABASE_PAIR,
  PACKAGE_TYPE_SPELL_PAIR,
  PACKAGE_TYPE_MIXED_LAND_SPELL_PAIR,
  PACKAGE_TYPE_SIDEBOARD_PAIR,
)
DEFAULT_PACKAGE_TYPES = (
  PACKAGE_TYPE_SPELL_PAIR,
  PACKAGE_TYPE_MIXED_LAND_SPELL_PAIR,
  PACKAGE_TYPE_SIDEBOARD_PAIR,
)
_DIAGNOSTIC_SPLIT_ORDER = (
  "train",
  "validation",
  "dev-test",
  "test",
  "final-test",
  "novelty_holdout",
)


@dataclass(frozen=True)
class PackageFeature:
  package_idx: int
  zone_idx: int
  package_type: str
  card_idxs: tuple[int, ...]
  support: int
  event_support: int
  train_activation_rate: float | None
  score: float
  scoring: str
  best_label_id: str | None
  best_label_count: int | None
  best_label_precision: float | None
  best_label_lift: float | None
  label_entropy: float | None

  def to_dict(self) -> dict[str, Any]:
    return {
      "package_idx": self.package_idx,
      "zone_idx": self.zone_idx,
      "package_type": self.package_type,
      "card_idxs": list(self.card_idxs),
      "size": len(self.card_idxs),
      "support": self.support,
      "event_support": self.event_support,
      "train_activation_rate": self.train_activation_rate,
      "score": self.score,
      "scoring": self.scoring,
      "best_label_id": self.best_label_id,
      "best_label_count": self.best_label_count,
      "best_label_precision": self.best_label_precision,
      "best_label_lift": self.best_label_lift,
      "label_entropy": self.label_entropy,
    }


class PackageFeatureSet:
  def __init__(
    self,
    features: list[PackageFeature],
    *,
    activation_override_by_deck_id: dict[str, tuple[int, ...]] | None = None,
  ) -> None:
    self.features = tuple(features)
    self._activation_override_by_deck_id = activation_override_by_deck_id or {}
    self._index_by_zone_itemset: dict[int, dict[tuple[int, ...], int]] = {}
    self._sizes_by_zone: dict[int, tuple[int, ...]] = {}
    sizes_by_zone: dict[int, set[int]] = {}
    for feature in self.features:
      self._index_by_zone_itemset.setdefault(feature.zone_idx, {})[
        feature.card_idxs
      ] = feature.package_idx
      sizes_by_zone.setdefault(feature.zone_idx, set()).add(len(feature.card_idxs))
    self._sizes_by_zone = {
      zone_idx: tuple(sorted(sizes))
      for zone_idx, sizes in sizes_by_zone.items()
    }

  def __len__(self) -> int:
    return len(self.features)

  def activation_indexes(self, example: ModelExample) -> tuple[int, ...]:
    if example.deck_id in self._activation_override_by_deck_id:
      return self._activation_override_by_deck_id[example.deck_id]

    cards_by_zone: dict[int, set[int]] = {}
    for token in example.tokens:
      cards_by_zone.setdefault(token.zone_idx, set()).add(token.card_idx)

    active: list[int] = []
    for zone_idx, cards in cards_by_zone.items():
      index_by_itemset = self._index_by_zone_itemset.get(zone_idx)
      if not index_by_itemset:
        continue
      sorted_cards = tuple(sorted(cards))
      for size in self._sizes_by_zone.get(zone_idx, ()):
        if len(sorted_cards) < size:
          continue
        for itemset in itertools.combinations(sorted_cards, size):
          package_idx = index_by_itemset.get(itemset)
          if package_idx is not None:
            active.append(package_idx)

    return tuple(sorted(active))

  def with_shuffled_activations(
    self,
    examples: list[ModelExample],
    *,
    seed: int,
    within_split: bool = True,
  ) -> "PackageFeatureSet":
    rng = random.Random(seed)
    override_by_deck_id: dict[str, tuple[int, ...]] = {}
    groups: dict[str, list[ModelExample]] = defaultdict(list)
    for example in examples:
      key = example.split_name if within_split else "all"
      groups[key].append(example)

    for grouped_examples in groups.values():
      deck_ids = [example.deck_id for example in grouped_examples]
      activations = [self.activation_indexes(example) for example in grouped_examples]
      rng.shuffle(activations)
      override_by_deck_id.update(dict(zip(deck_ids, activations, strict=True)))

    return PackageFeatureSet(
      list(self.features),
      activation_override_by_deck_id=override_by_deck_id,
    )

  def with_zero_activations(
    self,
    examples: list[ModelExample],
  ) -> "PackageFeatureSet":
    return PackageFeatureSet(
      list(self.features),
      activation_override_by_deck_id={
        example.deck_id: ()
        for example in examples
      },
    )

  def with_synthetic_activations(
    self,
    *,
    train_examples: list[ModelExample],
    examples: list[ModelExample],
    seed: int,
  ) -> "PackageFeatureSet":
    rng = random.Random(seed)
    rates = self._train_activation_rates(train_examples)
    activation_override_by_deck_id: dict[str, tuple[int, ...]] = {}
    for example in examples:
      activation_override_by_deck_id[example.deck_id] = tuple(
        package_idx
        for package_idx, rate in enumerate(rates)
        if rate > 0 and rng.random() < rate
      )

    return PackageFeatureSet(
      list(self.features),
      activation_override_by_deck_id=activation_override_by_deck_id,
    )

  def _train_activation_rates(
    self,
    train_examples: list[ModelExample],
  ) -> list[float]:
    if not train_examples:
      return [0.0 for _ in self.features]
    counts = [0 for _ in self.features]
    for example in train_examples:
      for package_idx in self.activation_indexes(example):
        counts[package_idx] += 1
    return [
      count / len(train_examples)
      for count in counts
    ]

  def summary(self, *, limit: int = 50) -> dict[str, Any]:
    return {
      "count": len(self.features),
      "features": [
        feature.to_dict()
        for feature in self.features[:limit]
      ],
    }

  def diagnostics(
    self,
    *,
    examples_by_split: dict[str, list[ModelExample]],
    card_vocab: tuple[dict[str, Any], ...],
    zone_vocab: dict[str, int],
    limit: int = 50,
  ) -> list[dict[str, Any]]:
    card_name_by_idx = {
      int(card["card_idx"]): str(card.get("primary_name") or card["card_idx"])
      for card in card_vocab
    }
    zone_by_idx = {
      int(zone_idx): str(zone_name)
      for zone_name, zone_idx in zone_vocab.items()
    }
    split_counts = self._activation_counts_by_split(
      examples_by_split,
      limit=limit,
    )

    rows: list[dict[str, Any]] = []
    for feature in self.features[:limit]:
      row = feature.to_dict()
      row["zone"] = zone_by_idx.get(feature.zone_idx, str(feature.zone_idx))
      row["card_names"] = [
        card_name_by_idx.get(card_idx, str(card_idx))
        for card_idx in feature.card_idxs
      ]
      for split_name in _ordered_diagnostic_splits(examples_by_split):
        split_examples = examples_by_split.get(split_name, [])
        active_count = split_counts.get(split_name, {}).get(feature.package_idx, 0)
        row[f"{split_name}_activation_count"] = active_count
        row[f"{split_name}_activation_rate"] = (
          active_count / len(split_examples)
          if split_examples
          else None
        )
      rows.append(row)

    return rows

  def _activation_counts_by_split(
    self,
    examples_by_split: dict[str, list[ModelExample]],
    *,
    limit: int,
  ) -> dict[str, dict[int, int]]:
    features = self.features[:limit]
    counts: dict[str, dict[int, int]] = {}
    for split_name, examples in examples_by_split.items():
      split_counts: dict[int, int] = {
        feature.package_idx: 0
        for feature in features
      }
      for example in examples:
        active_packages = set(self.activation_indexes(example))
        for feature in features:
          if feature.package_idx in active_packages:
            split_counts[feature.package_idx] += 1
      counts[split_name] = split_counts
    return counts


def _ordered_diagnostic_splits(
  examples_by_split: dict[str, list[ModelExample]],
) -> tuple[str, ...]:
  order_by_name = {
    split_name: index
    for index, split_name in enumerate(_DIAGNOSTIC_SPLIT_ORDER)
  }
  return tuple(
    sorted(
      examples_by_split,
      key=lambda split_name: (
        order_by_name.get(split_name, len(order_by_name)),
        split_name,
      ),
    )
  )


def mine_package_features(
  train_examples: list[ModelExample],
  *,
  min_support: int = 25,
  min_event_support: int = 8,
  max_size: int = 3,
  max_packages: int = 2048,
  zone_idxs: tuple[int, ...] = (0, 1),
  card_vocab: tuple[dict[str, Any], ...] = (),
  package_types: tuple[str, ...] = DEFAULT_PACKAGE_TYPES,
  scoring: str = PACKAGE_SCORING_BAYESIAN_LOG_ODDS,
  min_best_label_count: int = 10,
  min_best_label_precision: float = 0.05,
  max_label_entropy: float | None = None,
  max_train_activation_rate: float | None = 0.08,
  bayesian_prior_strength: float = 10.0,
) -> PackageFeatureSet:
  if min_support <= 0:
    raise ValueError("min_support must be positive.")
  if min_event_support <= 0:
    raise ValueError("min_event_support must be positive.")
  if max_size < 2:
    raise ValueError("max_size must be at least 2.")
  if max_packages <= 0:
    raise ValueError("max_packages must be positive.")
  if scoring not in PACKAGE_SCORING_MODES:
    supported = ", ".join(PACKAGE_SCORING_MODES)
    raise ValueError(f"Unsupported package scoring {scoring!r}; choose one of: {supported}.")
  if min_best_label_count < 0:
    raise ValueError("min_best_label_count must be non-negative.")
  if min_best_label_precision < 0 or min_best_label_precision > 1:
    raise ValueError("min_best_label_precision must be between 0 and 1.")
  if max_label_entropy is not None and max_label_entropy < 0:
    raise ValueError("max_label_entropy must be non-negative.")
  if max_train_activation_rate is not None and (
    max_train_activation_rate <= 0 or max_train_activation_rate > 1
  ):
    raise ValueError("max_train_activation_rate must be in (0, 1].")
  _validate_package_types(package_types)

  label_counts = Counter(example.target_label_id for example in train_examples)
  card_name_by_idx = _card_name_by_idx(card_vocab)
  transactions_by_zone = _transactions_by_zone(train_examples, zone_idxs=zone_idxs)
  candidates: list[_Candidate] = []
  for zone_idx in zone_idxs:
    candidates.extend(
      _mine_zone_candidates(
        transactions_by_zone.get(zone_idx, []),
        label_counts=label_counts,
        train_count=len(train_examples),
        zone_idx=zone_idx,
        min_support=min_support,
        min_event_support=min_event_support,
        max_size=max_size,
        card_name_by_idx=card_name_by_idx,
        package_types=package_types,
        scoring=scoring,
        min_best_label_count=min_best_label_count,
        min_best_label_precision=min_best_label_precision,
        max_label_entropy=max_label_entropy,
        max_train_activation_rate=max_train_activation_rate,
        bayesian_prior_strength=bayesian_prior_strength,
      )
    )

  closed_candidates = _closed_candidates(candidates)
  closed_candidates.sort(
    key=lambda candidate: (
      -candidate.score,
      -candidate.support,
      -candidate.event_support,
      candidate.zone_idx,
      candidate.card_idxs,
    )
  )

  features = [
    PackageFeature(
      package_idx=index,
      zone_idx=candidate.zone_idx,
      package_type=candidate.package_type,
      card_idxs=candidate.card_idxs,
      support=candidate.support,
      event_support=candidate.event_support,
      train_activation_rate=candidate.train_activation_rate,
      score=candidate.score,
      scoring=scoring,
      best_label_id=candidate.best_label_id,
      best_label_count=candidate.best_label_count,
      best_label_precision=candidate.best_label_precision,
      best_label_lift=candidate.best_label_lift,
      label_entropy=candidate.label_entropy,
    )
    for index, candidate in enumerate(closed_candidates[:max_packages])
  ]
  return PackageFeatureSet(features)


def mine_support_matched_unscored_package_features(
  train_examples: list[ModelExample],
  *,
  reference_features: tuple[PackageFeature, ...],
  min_support: int = 25,
  min_event_support: int = 8,
  max_size: int = 3,
  max_packages: int = 2048,
  zone_idxs: tuple[int, ...] = (0, 1),
  card_vocab: tuple[dict[str, Any], ...] = (),
  package_types: tuple[str, ...] = DEFAULT_PACKAGE_TYPES,
  random_seed: int = 13,
  bayesian_prior_strength: float = 10.0,
) -> PackageFeatureSet:
  _validate_package_types(package_types)
  label_counts = Counter(example.target_label_id for example in train_examples)
  card_name_by_idx = _card_name_by_idx(card_vocab)
  transactions_by_zone = _transactions_by_zone(train_examples, zone_idxs=zone_idxs)
  candidates: list[_Candidate] = []
  for zone_idx in zone_idxs:
    candidates.extend(
      _mine_zone_candidates(
        transactions_by_zone.get(zone_idx, []),
        label_counts=label_counts,
        train_count=len(train_examples),
        zone_idx=zone_idx,
        min_support=min_support,
        min_event_support=min_event_support,
        max_size=max_size,
        card_name_by_idx=card_name_by_idx,
        package_types=package_types,
        scoring=PACKAGE_SCORING_BAYESIAN_LOG_ODDS,
        min_best_label_count=0,
        min_best_label_precision=0.0,
        max_label_entropy=None,
        max_train_activation_rate=None,
        bayesian_prior_strength=bayesian_prior_strength,
      )
    )

  rng = random.Random(random_seed)
  reference_keys = {
    (feature.zone_idx, feature.card_idxs)
    for feature in reference_features
  }
  candidates_by_key: dict[tuple[int, int, str], list[_Candidate]] = defaultdict(list)
  for candidate in candidates:
    if (candidate.zone_idx, candidate.card_idxs) in reference_keys:
      continue
    key = (candidate.zone_idx, len(candidate.card_idxs), candidate.package_type)
    candidates_by_key[key].append(candidate)

  for rows in candidates_by_key.values():
    rng.shuffle(rows)
    rows.sort(key=lambda candidate: candidate.support)

  selected: list[_Candidate] = []
  used: set[tuple[int, tuple[int, ...]]] = set()
  for reference in reference_features[:max_packages]:
    key = (reference.zone_idx, len(reference.card_idxs), reference.package_type)
    candidates_for_key = candidates_by_key.get(key, [])
    if not candidates_for_key:
      continue
    candidate = _nearest_unused_candidate(
      candidates_for_key,
      target_support=reference.support,
      used=used,
    )
    if candidate is None:
      continue
    used.add((candidate.zone_idx, candidate.card_idxs))
    selected.append(candidate)

  return PackageFeatureSet([
    PackageFeature(
      package_idx=index,
      zone_idx=candidate.zone_idx,
      package_type=candidate.package_type,
      card_idxs=candidate.card_idxs,
      support=candidate.support,
      event_support=candidate.event_support,
      train_activation_rate=candidate.train_activation_rate,
      score=candidate.score,
      scoring="support-matched-unscored",
      best_label_id=candidate.best_label_id,
      best_label_count=candidate.best_label_count,
      best_label_precision=candidate.best_label_precision,
      best_label_lift=candidate.best_label_lift,
      label_entropy=candidate.label_entropy,
    )
    for index, candidate in enumerate(selected)
  ])


@dataclass(frozen=True)
class _Transaction:
  event_id: str
  label_id: str
  card_idxs: tuple[int, ...]


@dataclass(frozen=True)
class _Candidate:
  zone_idx: int
  package_type: str
  card_idxs: tuple[int, ...]
  support: int
  event_support: int
  train_activation_rate: float | None
  score: float
  best_label_id: str | None
  best_label_count: int | None
  best_label_precision: float | None
  best_label_lift: float | None
  label_entropy: float | None


def _transactions_by_zone(
  examples: list[ModelExample],
  *,
  zone_idxs: tuple[int, ...],
) -> dict[int, list[_Transaction]]:
  rows: dict[int, list[_Transaction]] = {zone_idx: [] for zone_idx in zone_idxs}
  for example in examples:
    cards_by_zone: dict[int, set[int]] = {}
    for token in example.tokens:
      if token.zone_idx in rows:
        cards_by_zone.setdefault(token.zone_idx, set()).add(token.card_idx)
    for zone_idx, cards in cards_by_zone.items():
      if len(cards) >= 2:
        rows[zone_idx].append(
          _Transaction(
            event_id=example.event_id,
            label_id=example.target_label_id,
            card_idxs=tuple(sorted(cards)),
          )
        )
  return rows


def _mine_zone_candidates(
  transactions: list[_Transaction],
  *,
  label_counts: Counter[str],
  train_count: int,
  zone_idx: int,
  min_support: int,
  min_event_support: int,
  max_size: int,
  card_name_by_idx: dict[int, str],
  package_types: tuple[str, ...],
  scoring: str,
  min_best_label_count: int,
  min_best_label_precision: float,
  max_label_entropy: float | None,
  max_train_activation_rate: float | None,
  bayesian_prior_strength: float,
) -> list[_Candidate]:
  candidates: list[_Candidate] = []
  previous_frequent: set[tuple[int, ...]] | None = None
  for size in range(2, max_size + 1):
    counts: Counter[tuple[int, ...]] = Counter()
    events: dict[tuple[int, ...], set[str]] = defaultdict(set)
    labels: dict[tuple[int, ...], Counter[str]] = defaultdict(Counter)

    for transaction in transactions:
      if len(transaction.card_idxs) < size:
        continue
      for itemset in itertools.combinations(transaction.card_idxs, size):
        if previous_frequent is not None and not _all_subsets_frequent(
          itemset,
          previous_frequent,
        ):
          continue
        counts[itemset] += 1
        events[itemset].add(transaction.event_id)
        labels[itemset][transaction.label_id] += 1

    current_frequent: set[tuple[int, ...]] = set()
    for itemset, support in counts.items():
      event_support = len(events[itemset])
      if support < min_support or event_support < min_event_support:
        continue
      package_type = _package_type(
        zone_idx,
        itemset,
        card_name_by_idx=card_name_by_idx,
      )
      if package_type not in package_types:
        continue
      train_activation_rate = support / train_count if train_count else None
      if (
        max_train_activation_rate is not None
        and train_activation_rate is not None
        and train_activation_rate > max_train_activation_rate
      ):
        continue
      (
        score,
        best_label_id,
        best_label_count,
        best_label_precision,
        best_label_lift,
        label_entropy,
      ) = (
        _discriminative_stats(
          labels[itemset],
          support=support,
          label_counts=label_counts,
          train_count=train_count,
          scoring=scoring,
          bayesian_prior_strength=bayesian_prior_strength,
        )
      )
      if best_label_count is None or best_label_count < min_best_label_count:
        continue
      if (
        best_label_precision is None
        or best_label_precision < min_best_label_precision
      ):
        continue
      if (
        max_label_entropy is not None
        and label_entropy is not None
        and label_entropy > max_label_entropy
      ):
        continue
      current_frequent.add(itemset)
      candidates.append(
        _Candidate(
          zone_idx=zone_idx,
          package_type=package_type,
          card_idxs=itemset,
          support=support,
          event_support=event_support,
          train_activation_rate=train_activation_rate,
          score=score,
          best_label_id=best_label_id,
          best_label_count=best_label_count,
          best_label_precision=best_label_precision,
          best_label_lift=best_label_lift,
          label_entropy=label_entropy,
        )
      )

    if not current_frequent:
      break
    previous_frequent = current_frequent

  return candidates


def _all_subsets_frequent(
  itemset: tuple[int, ...],
  previous_frequent: set[tuple[int, ...]],
) -> bool:
  return all(
    subset in previous_frequent
    for subset in itertools.combinations(itemset, len(itemset) - 1)
  )


def _discriminative_stats(
  label_counts_for_itemset: Counter[str],
  *,
  support: int,
  label_counts: Counter[str],
  train_count: int,
  scoring: str,
  bayesian_prior_strength: float,
) -> tuple[float, str | None, int | None, float | None, float | None, float | None]:
  if support <= 0 or train_count <= 0 or not label_counts_for_itemset:
    return 0.0, None, None, None, None, None

  best_label_id: str | None = None
  best_score = 0.0
  best_count: int | None = None
  best_precision: float | None = None
  best_lift: float | None = None
  entropy = _label_entropy(label_counts_for_itemset, support=support)
  for label_id, label_support in label_counts_for_itemset.items():
    package_label_rate = label_support / support
    base_label_rate = label_counts[label_id] / train_count
    if base_label_rate <= 0:
      continue
    lift = package_label_rate / base_label_rate
    if scoring in (
      PACKAGE_SCORING_SUPPORT_LIFT,
      PACKAGE_SCORING_PRECISION_FILTERED_LIFT,
    ):
      score = math.log(lift) * math.sqrt(support)
    elif scoring == PACKAGE_SCORING_ENTROPY_PENALIZED_LIFT:
      score = math.log(lift) * math.sqrt(support) / (1.0 + entropy)
    elif scoring == PACKAGE_SCORING_BAYESIAN_LOG_ODDS:
      score = _bayesian_log_odds_z(
        label_support=label_support,
        support=support,
        total_label_support=label_counts[label_id],
        train_count=train_count,
        prior_strength=bayesian_prior_strength,
      )
    else:
      raise AssertionError(f"Unhandled package scoring mode: {scoring}")
    if best_label_id is None or score > best_score:
      best_label_id = label_id
      best_score = score
      best_count = label_support
      best_precision = package_label_rate
      best_lift = lift

  return (
    best_score,
    best_label_id,
    best_count,
    best_precision,
    best_lift,
    entropy,
  )


def _closed_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
  rows: list[_Candidate] = []
  by_zone_support: dict[tuple[int, int], list[_Candidate]] = defaultdict(list)
  for candidate in sorted(candidates, key=lambda row: -len(row.card_idxs)):
    supersets = by_zone_support[(candidate.zone_idx, candidate.support)]
    if any(
      set(candidate.card_idxs) < set(superset.card_idxs)
      for superset in supersets
    ):
      continue
    rows.append(candidate)
    supersets.append(candidate)

  return rows


def _label_entropy(
  label_counts_for_itemset: Counter[str],
  *,
  support: int,
) -> float:
  entropy = 0.0
  for label_support in label_counts_for_itemset.values():
    if label_support <= 0:
      continue
    probability = label_support / support
    entropy -= probability * math.log(probability)
  return entropy


def _bayesian_log_odds_z(
  *,
  label_support: int,
  support: int,
  total_label_support: int,
  train_count: int,
  prior_strength: float,
) -> float:
  outside_count = train_count - support
  outside_label_support = total_label_support - label_support
  if support <= 0 or outside_count <= 0 or train_count <= 0:
    return 0.0

  base_label_rate = total_label_support / train_count
  alpha_label = max(base_label_rate * prior_strength, 1e-6)
  alpha_other = max((1.0 - base_label_rate) * prior_strength, 1e-6)
  package_other = support - label_support
  outside_other = outside_count - outside_label_support
  package_log_odds = math.log(
    (label_support + alpha_label) / (package_other + alpha_other)
  )
  outside_log_odds = math.log(
    (outside_label_support + alpha_label) / (outside_other + alpha_other)
  )
  variance = (
    1.0 / (label_support + alpha_label)
    + 1.0 / (package_other + alpha_other)
    + 1.0 / (outside_label_support + alpha_label)
    + 1.0 / (outside_other + alpha_other)
  )
  return (package_log_odds - outside_log_odds) / math.sqrt(variance)


def _nearest_unused_candidate(
  candidates: list[_Candidate],
  *,
  target_support: int,
  used: set[tuple[int, tuple[int, ...]]],
) -> _Candidate | None:
  best: _Candidate | None = None
  best_distance: int | None = None
  for candidate in candidates:
    key = (candidate.zone_idx, candidate.card_idxs)
    if key in used:
      continue
    distance = abs(candidate.support - target_support)
    if best_distance is None or distance < best_distance:
      best = candidate
      best_distance = distance
    if best_distance == 0:
      break
  return best


def _package_type(
  zone_idx: int,
  card_idxs: tuple[int, ...],
  *,
  card_name_by_idx: dict[int, str],
) -> str:
  if zone_idx == 1:
    return PACKAGE_TYPE_SIDEBOARD_PAIR
  land_count = sum(
    1
    for card_idx in card_idxs
    if _is_probable_land(card_name_by_idx.get(card_idx, ""))
  )
  if land_count == len(card_idxs):
    return PACKAGE_TYPE_MANABASE_PAIR
  if land_count:
    return PACKAGE_TYPE_MIXED_LAND_SPELL_PAIR
  return PACKAGE_TYPE_SPELL_PAIR


def _card_name_by_idx(card_vocab: tuple[dict[str, Any], ...]) -> dict[int, str]:
  return {
    int(card["card_idx"]): str(card.get("primary_name") or "")
    for card in card_vocab
  }


def _validate_package_types(package_types: tuple[str, ...]) -> None:
  unsupported = sorted(set(package_types) - set(PACKAGE_TYPES))
  if unsupported:
    supported = ", ".join(PACKAGE_TYPES)
    raise ValueError(
      f"Unsupported package types {unsupported}; choose from: {supported}."
    )


_LAND_NAMES = {
  "Adarkar Wastes",
  "Arid Mesa",
  "Blood Crypt",
  "Bloodstained Mire",
  "Blooming Marsh",
  "Botanical Sanctum",
  "Breeding Pool",
  "Cavern of Souls",
  "Concealed Courtyard",
  "Copperline Gorge",
  "Elegant Parlor",
  "Flooded Strand",
  "Forest",
  "Godless Shrine",
  "Hallowed Fountain",
  "Hedge Maze",
  "Island",
  "Lush Portico",
  "Marsh Flats",
  "Meticulous Archive",
  "Misty Rainforest",
  "Mountain",
  "Overgrown Tomb",
  "Plains",
  "Polluted Delta",
  "Raucous Theater",
  "Sacred Foundry",
  "Scalding Tarn",
  "Shadowy Backstreet",
  "Spara's Headquarters",
  "Steam Vents",
  "Stomping Ground",
  "Swamp",
  "Temple Garden",
  "Thundering Falls",
  "Undercity Sewers",
  "Verdant Catacombs",
  "Watery Grave",
  "Windswept Heath",
  "Wooded Foothills",
  "Xander's Lounge",
  "Zagoth Triome",
}

_LAND_SUFFIXES = (
  "Canal",
  "Courtyard",
  "Falls",
  "Fountain",
  "Grave",
  "Ground",
  "Heath",
  "Mesa",
  "Mire",
  "Pool",
  "Rainforest",
  "Sanctum",
  "Shrine",
  "Tarn",
  "Tomb",
  "Triome",
  "Vents",
)


def _is_probable_land(card_name: str) -> bool:
  if card_name in _LAND_NAMES:
    return True
  return any(card_name.endswith(suffix) for suffix in _LAND_SUFFIXES)

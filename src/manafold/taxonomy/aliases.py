from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from manafold.datasets.mtgo.build import SOURCE_ARCHETYPE_PROXY
from manafold.serialization import write_json
from manafold.taxonomy import (
  TaxonomyEvaluationConfig,
  display_label_from_id,
  load_taxonomy_evaluation_config,
)

ZONE_MAIN = "main"
ZONE_SIDE = "side"
DEFAULT_ALIAS_MODEL = "deepsets-quantity-weighted-regularized"
DEFAULT_PROFILE_SPLITS = ("validation", "dev-test", "test", "final-test")
HELD_OUT_EVIDENCE_SPLITS = ("dev-test", "test", "final-test")


@dataclass(frozen=True)
class _DeckRecord:
  deck_id: str
  split_name: str
  event_date: date
  format_code: str
  label_id: str
  cards_by_zone: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class _LabelProfile:
  label_id: str
  split_name: str
  deck_count: int
  adoption_by_zone: dict[str, dict[int, float]]


def run_alias_candidate_scoring(
  rolling_result: Path | None,
  *,
  output: Path | None = None,
  predictions: Path | None = None,
  predictions_manifest: Path | None = None,
  deck_embeddings: Path | None = None,
  dataset_path: Path | None = None,
  model_name: str = DEFAULT_ALIAS_MODEL,
  target_source: str = SOURCE_ARCHETYPE_PROXY,
  taxonomy_eval: Path | None = None,
  min_confusion_count: int = 5,
  min_seed_confusion_count: int = 1,
  max_candidates: int = 200,
) -> dict[str, Any]:
  taxonomy_config = load_taxonomy_evaluation_config(
    taxonomy_eval,
    target_source=target_source,
  )
  if predictions is not None:
    if dataset_path is None:
      raise ValueError("--dataset is required when --predictions is passed.")
    result = _prediction_backed_alias_candidates(
      predictions=predictions,
      predictions_manifest=predictions_manifest,
      deck_embeddings=deck_embeddings,
      dataset_path=dataset_path,
      output=output,
      target_source=target_source,
      taxonomy_config=taxonomy_config,
      min_confusion_count=min_confusion_count,
      max_candidates=max_candidates,
    )
    return result
  if rolling_result is None:
    raise ValueError(
      "rolling_result is required unless --predictions is passed."
    )
  rolling_result_path = _rolling_result_path(rolling_result)
  data = json.loads(rolling_result_path.read_text(encoding="utf-8"))

  candidates: list[dict[str, Any]] = []
  for window in data.get("windows", []):
    candidates.extend(
      _window_alias_candidates(
        window,
        model_name=model_name,
        target_source=target_source,
        taxonomy_config=taxonomy_config,
        min_confusion_count=min_confusion_count,
        min_seed_confusion_count=min_seed_confusion_count,
      )
    )

  candidates.sort(
    key=lambda row: (
      -float(row["confidence"]),
      -int(row["evidence"]["confusion_count"]),
      row["window"],
      row["label_a"],
      row["label_b"],
    )
  )
  if max_candidates > 0:
    candidates = candidates[:max_candidates]
  pair_summaries = _pair_summaries(candidates)
  weak_label_observations = _weak_label_observations(candidates)

  result = {
    "run_id": "alias_candidate_scoring",
    "rolling_result_path": str(rolling_result_path),
    "model_name": model_name,
    "target_source": target_source,
    "taxonomy_seed_evidence": taxonomy_config.to_dict(),
    "min_confusion_count": min_confusion_count,
    "min_seed_confusion_count": min_seed_confusion_count,
    "candidate_count": len(candidates),
    "candidates": candidates,
    "pair_summary_count": len(pair_summaries),
    "pair_summaries": pair_summaries,
    "weak_label_observation_count": len(weak_label_observations),
    "weak_label_observations_path": (
      str(_weak_label_observations_path(output))
      if output is not None
      else None
    ),
  }
  if output is not None:
    write_json(output, result)
    _write_jsonl(
      _weak_label_observations_path(output), weak_label_observations
    )
  return result


def _prediction_backed_alias_candidates(
  *,
  predictions: Path,
  predictions_manifest: Path | None,
  deck_embeddings: Path | None,
  dataset_path: Path,
  output: Path | None,
  target_source: str,
  taxonomy_config: TaxonomyEvaluationConfig,
  min_confusion_count: int,
  max_candidates: int,
) -> dict[str, Any]:
  prediction_rows = pq.read_table(predictions).to_pylist()
  manifest = (
    json.loads(predictions_manifest.read_text(encoding="utf-8"))
    if predictions_manifest is not None and predictions_manifest.exists()
    else {}
  )
  records = _load_deck_records(dataset_path, target_source=target_source)
  supports = _label_support_by_split(records)
  profiles = _label_profiles(records)
  card_names = _card_names(dataset_path)
  prediction_confusion = _prediction_confusion_counts_by_split(
    prediction_rows
  )
  prediction_pairs = _prediction_pairs_by_split(prediction_rows)
  seed_pairs = _seed_pairs_by_split(taxonomy_config, records)

  candidates: list[dict[str, Any]] = []
  split_names = sorted(set(prediction_confusion) | set(seed_pairs))
  for split_name in split_names:
    pair_keys = set(prediction_confusion.get(split_name, {})) | set(
      seed_pairs.get(split_name, ())
    )
    for pair_key in pair_keys:
      pair_confusions = prediction_confusion.get(split_name, {}).get(
        pair_key,
        Counter(),
      )
      confusion_count = sum(pair_confusions.values())
      seed_rule = _matching_seed_rule(
        pair_key,
        taxonomy_config,
        records=records,
        split_name=split_name,
      )
      if confusion_count < min_confusion_count and seed_rule is None:
        continue
      label_a, label_b = _orient_pair_by_support(
        pair_key,
        supports=supports,
        split_name=split_name,
      )
      profile_a = profiles.get(
        (split_name, label_a), _empty_profile(label_a, split_name)
      )
      profile_b = profiles.get(
        (split_name, label_b), _empty_profile(label_b, split_name)
      )
      evidence = _pair_evidence(
        label_a,
        label_b,
        split_name=split_name,
        pair_confusions=pair_confusions,
        records=records,
        supports=supports,
        profile_a=profile_a,
        profile_b=profile_b,
        card_names=card_names,
        seed_rule=seed_rule,
        seed_names=("model-score",),
        min_seed_confusion_count=1,
      )
      pair_rows = prediction_pairs.get(split_name, {}).get(pair_key, [])
      evidence["example_deck_ids"] = [
        str(row["deck_id"]) for row in pair_rows[:10]
      ]
      evidence["mean_top1_probability"] = _mean(
        float(row["top1_probability"])
        for row in pair_rows
        if row.get("top1_probability") is not None
      )
      evidence["mean_temperature_scaled_probability"] = _mean(
        float(row["temperature_scaled_probability"])
        for row in pair_rows
        if row.get("temperature_scaled_probability") is not None
      )
      confidence = _confidence(evidence, seed_rule=seed_rule)
      candidates.append({
        "format": _window_format(records),
        "window": _candidate_window(records, split_name, (label_a, label_b)),
        "rolling_window": None,
        "split_name": split_name,
        "label_a": display_label_from_id(label_a),
        "label_b": display_label_from_id(label_b),
        "label_a_id": label_a,
        "label_b_id": label_b,
        "evidence": evidence,
        "suggested_relation": _suggested_relation(confidence, evidence),
        "confidence": confidence,
        "action": _suggested_action(confidence, evidence),
      })

  candidates.sort(
    key=lambda row: (
      -float(row["confidence"]),
      -int(row["evidence"]["confusion_count"]),
      row["split_name"],
      row["label_a"],
      row["label_b"],
    )
  )
  if max_candidates > 0:
    candidates = candidates[:max_candidates]
  pair_summaries = _pair_summaries(candidates)
  weak_label_observations = _weak_label_observations(
    candidates,
    training_eligible_by_split=False,
    training_exclusion_reason="prediction_backfill_evidence_not_training_target",
  )
  unknown_review_candidates = _unknown_review_candidates(prediction_rows)
  backfill_report = _backfill_report(
    prediction_rows,
    manifest=manifest,
    deck_embeddings=deck_embeddings,
    train_label_ids={
      record.label_id
      for record in records
      if record.split_name == "train"
    },
  )

  result = {
    "run_id": "prediction_backed_alias_candidate_scoring",
    "prediction_path": str(predictions),
    "prediction_manifest_path": (
      str(predictions_manifest)
      if predictions_manifest is not None
      else None
    ),
    "deck_embeddings_path": (
      str(deck_embeddings) if deck_embeddings is not None else None
    ),
    "dataset_path": str(dataset_path),
    "model_name": manifest.get("model_family"),
    "model_version": manifest.get("model_version"),
    "target_source": target_source,
    "taxonomy_seed_evidence": taxonomy_config.to_dict(),
    "min_confusion_count": min_confusion_count,
    "candidate_count": len(candidates),
    "candidates": candidates,
    "pair_summary_count": len(pair_summaries),
    "pair_summaries": pair_summaries,
    "unknown_review_candidate_count": len(unknown_review_candidates),
    "unknown_review_candidates": unknown_review_candidates,
    "weak_label_observation_count": len(weak_label_observations),
    "weak_label_observations_path": (
      str(_weak_label_observations_path(output))
      if output is not None
      else None
    ),
    "backfill_report_path": (
      str(_backfill_report_path(output)) if output is not None else None
    ),
    "backfill_report": backfill_report,
  }
  if output is not None:
    write_json(output, result)
    _write_jsonl(
      _weak_label_observations_path(output), weak_label_observations
    )
    write_json(_backfill_report_path(output), backfill_report)
  return result


def _window_alias_candidates(
  window: dict[str, Any],
  *,
  model_name: str,
  target_source: str,
  taxonomy_config: TaxonomyEvaluationConfig,
  min_confusion_count: int,
  min_seed_confusion_count: int,
) -> list[dict[str, Any]]:
  dataset_path = Path(str(window["dataset_path"]))
  training_result_path = Path(str(window["training_result_path"]))
  training_result = json.loads(
    training_result_path.read_text(encoding="utf-8")
  )
  records = _load_deck_records(dataset_path, target_source=target_source)
  supports = _label_support_by_split(records)
  profiles = _label_profiles(records)
  card_names = _card_names(dataset_path)
  model_result = training_result.get("models", {}).get(model_name, {})
  seed_names = _seed_names(model_result)
  confusion = _confusion_counts_by_split(model_result)
  seed_pairs = _seed_pairs_by_split(
    taxonomy_config,
    records,
  )

  split_names = sorted(set(confusion) | set(seed_pairs))
  rows: list[dict[str, Any]] = []
  for split_name in split_names:
    pair_keys = set(confusion.get(split_name, {})) | set(
      seed_pairs.get(split_name, ())
    )
    for pair_key in pair_keys:
      pair_confusions = confusion.get(split_name, {}).get(pair_key, Counter())
      confusion_count = sum(pair_confusions.values())
      seed_rule = _matching_seed_rule(
        pair_key,
        taxonomy_config,
        records=records,
        split_name=split_name,
      )
      if confusion_count < min_confusion_count and seed_rule is None:
        continue
      label_a, label_b = _orient_pair_by_support(
        pair_key,
        supports=supports,
        split_name=split_name,
      )
      profile_a = profiles.get(
        (split_name, label_a), _empty_profile(label_a, split_name)
      )
      profile_b = profiles.get(
        (split_name, label_b), _empty_profile(label_b, split_name)
      )
      evidence = _pair_evidence(
        label_a,
        label_b,
        split_name=split_name,
        pair_confusions=pair_confusions,
        records=records,
        supports=supports,
        profile_a=profile_a,
        profile_b=profile_b,
        card_names=card_names,
        seed_rule=seed_rule,
        seed_names=seed_names,
        min_seed_confusion_count=min_seed_confusion_count,
      )
      confidence = _confidence(evidence, seed_rule=seed_rule)
      rows.append({
        "format": _window_format(records),
        "window": _candidate_window(records, split_name, (label_a, label_b)),
        "rolling_window": window.get("window", {}).get("name"),
        "split_name": split_name,
        "label_a": display_label_from_id(label_a),
        "label_b": display_label_from_id(label_b),
        "label_a_id": label_a,
        "label_b_id": label_b,
        "evidence": evidence,
        "suggested_relation": _suggested_relation(confidence, evidence),
        "confidence": confidence,
        "action": _suggested_action(confidence, evidence),
      })
  return rows


def _rolling_result_path(path: Path) -> Path:
  if path.is_dir():
    return path / "rolling_evaluation_results.json"
  return path


def _load_deck_records(
  dataset_path: Path,
  *,
  target_source: str,
) -> list[_DeckRecord]:
  manifest = json.loads((dataset_path / "dataset_manifest.json").read_text())
  artifacts = manifest["artifacts"]
  splits = {
    str(row["deck_id"]): row
    for row in pq.read_table(
      dataset_path / artifacts["split_manifest"]
    ).to_pylist()
  }
  labels = {
    str(row["deck_id"]): str(row["proxy_label_id"])
    for row in _proxy_target_rows(dataset_path, artifacts)
    if str(row["target_source"]) == target_source
  }
  cards_by_deck: dict[str, dict[str, set[int]]] = defaultdict(
    lambda: defaultdict(set)
  )
  for row in pq.read_table(dataset_path / artifacts["deck_tokens"]).to_pylist():
    deck_id = str(row["deck_id"])
    cards_by_deck[deck_id][str(row["zone"])].add(int(row["card_idx"]))

  records: list[_DeckRecord] = []
  for deck_id, label_id in labels.items():
    split = splits.get(deck_id)
    if split is None:
      continue
    event_date = split["event_date"]
    if not isinstance(event_date, date):
      event_date = date.fromisoformat(str(event_date))
    records.append(
      _DeckRecord(
        deck_id=deck_id,
        split_name=str(split["split_name"]),
        event_date=event_date,
        format_code=str(split["format"]).casefold(),
        label_id=label_id,
        cards_by_zone={
          zone: tuple(sorted(cards))
          for zone, cards in cards_by_deck.get(deck_id, {}).items()
        },
      )
    )
  return records


def _proxy_target_rows(
  dataset_path: Path,
  artifacts: dict[str, Any],
) -> list[dict[str, Any]]:
  relative_path = artifacts.get("proxy_targets")
  if relative_path is None:
    return []
  path = dataset_path / relative_path
  if not path.exists():
    return []
  return pq.read_table(path).to_pylist()


def _card_names(dataset_path: Path) -> dict[int, str]:
  manifest = json.loads((dataset_path / "dataset_manifest.json").read_text())
  artifacts = manifest["artifacts"]
  return {
    int(row["card_idx"]): str(row.get("primary_name") or row["card_idx"])
    for row in pq.read_table(dataset_path / artifacts["card_vocab"]).to_pylist()
  }


def _label_support_by_split(
  records: list[_DeckRecord],
) -> dict[str, dict[str, int]]:
  supports: dict[str, Counter[str]] = defaultdict(Counter)
  for record in records:
    supports[record.split_name][record.label_id] += 1
  return {
    split_name: dict(counter) for split_name, counter in supports.items()
  }


def _label_profiles(
  records: list[_DeckRecord],
) -> dict[tuple[str, str], _LabelProfile]:
  deck_counts: Counter[tuple[str, str]] = Counter()
  card_counts: dict[tuple[str, str, str], Counter[int]] = defaultdict(Counter)
  for record in records:
    key = (record.split_name, record.label_id)
    deck_counts[key] += 1
    for zone, cards in record.cards_by_zone.items():
      card_counts[(record.split_name, record.label_id, zone)].update(cards)

  profiles: dict[tuple[str, str], _LabelProfile] = {}
  for split_name, label_id in deck_counts:
    count = deck_counts[(split_name, label_id)]
    adoption_by_zone: dict[str, dict[int, float]] = {}
    for zone in (ZONE_MAIN, ZONE_SIDE):
      adoption_by_zone[zone] = {
        card_idx: card_count / count
        for card_idx, card_count in card_counts[
          (split_name, label_id, zone)
        ].items()
      }
    profiles[(split_name, label_id)] = _LabelProfile(
      label_id=label_id,
      split_name=split_name,
      deck_count=count,
      adoption_by_zone=adoption_by_zone,
    )
  return profiles


def _empty_profile(label_id: str, split_name: str) -> _LabelProfile:
  return _LabelProfile(
    label_id=label_id,
    split_name=split_name,
    deck_count=0,
    adoption_by_zone={ZONE_MAIN: {}, ZONE_SIDE: {}},
  )


def _confusion_counts_by_split(
  model_result: dict[str, Any],
) -> dict[str, dict[tuple[str, str], Counter[str]]]:
  rows: dict[str, dict[tuple[str, str], Counter[str]]] = defaultdict(
    lambda: defaultdict(Counter)
  )
  for prediction_path in _prediction_paths(model_result):
    seed = _seed_name(prediction_path)
    with prediction_path.open("r", encoding="utf-8") as input_file:
      for line in input_file:
        prediction = json.loads(line)
        split_name = str(prediction.get("split_name"))
        if split_name == "train":
          continue
        actual = str(
          prediction.get("source_actual_label_id")
          or prediction.get("actual_label_id")
        )
        predicted = prediction.get("predicted_label_id")
        if predicted is None or str(predicted) == actual:
          continue
        pair_key = tuple(sorted((actual, str(predicted))))
        rows[split_name][pair_key][seed] += 1
  return {
    split_name: dict(split_rows) for split_name, split_rows in rows.items()
  }


def _prediction_confusion_counts_by_split(
  prediction_rows: list[dict[str, Any]],
) -> dict[str, dict[tuple[str, str], Counter[str]]]:
  rows: dict[str, dict[tuple[str, str], Counter[str]]] = defaultdict(
    lambda: defaultdict(Counter)
  )
  for row in prediction_rows:
    split_name = str(row.get("split_name") or "unknown")
    if split_name == "train":
      continue
    source_label_id = row.get("source_label_id")
    predicted_label_id = row.get("top1_label_id")
    if (
      source_label_id is None
      or predicted_label_id is None
      or str(source_label_id) == str(predicted_label_id)
    ):
      continue
    pair_key = tuple(sorted((str(source_label_id), str(predicted_label_id))))
    rows[split_name][pair_key]["model-score"] += 1
  return {
    split_name: dict(split_rows) for split_name, split_rows in rows.items()
  }


def _prediction_pairs_by_split(
  prediction_rows: list[dict[str, Any]],
) -> dict[str, dict[tuple[str, str], list[dict[str, Any]]]]:
  rows: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = defaultdict(
    lambda: defaultdict(list)
  )
  for row in prediction_rows:
    source_label_id = row.get("source_label_id")
    predicted_label_id = row.get("top1_label_id")
    if (
      source_label_id is None
      or predicted_label_id is None
      or str(source_label_id) == str(predicted_label_id)
    ):
      continue
    split_name = str(row.get("split_name") or "unknown")
    if split_name == "train":
      continue
    pair_key = tuple(sorted((str(source_label_id), str(predicted_label_id))))
    rows[split_name][pair_key].append(row)
  return {
    split_name: {
      pair_key: sorted(
        pair_rows,
        key=lambda row: (
          -float(row.get("temperature_scaled_probability") or 0.0),
          str(row.get("deck_id") or ""),
        ),
      )
      for pair_key, pair_rows in split_rows.items()
    }
    for split_name, split_rows in rows.items()
  }


def _prediction_paths(model_result: dict[str, Any]) -> list[Path]:
  paths: list[Path] = []
  for run in model_result.get("seed_runs", []):
    path = run.get("prediction_path")
    if path:
      paths.append(Path(str(path)))
  path = model_result.get("prediction_path")
  if path:
    paths.append(Path(str(path)))
  return list(dict.fromkeys(paths))


def _seed_name(path: Path) -> str:
  stem = path.stem
  if "_seed_" in stem:
    return stem.rsplit("_seed_", 1)[-1]
  return "primary"


def _seed_names(model_result: dict[str, Any]) -> tuple[str, ...]:
  names = [_seed_name(path) for path in _prediction_paths(model_result)]
  return tuple(dict.fromkeys(names))


def _seed_pairs_by_split(
  taxonomy_config: TaxonomyEvaluationConfig,
  records: list[_DeckRecord],
) -> dict[str, set[tuple[str, str]]]:
  rows: dict[str, set[tuple[str, str]]] = defaultdict(set)
  if not taxonomy_config.enabled:
    return {}
  for record in records:
    context = {
      "event_date": record.event_date.isoformat(),
      "format": record.format_code,
    }
    for rule in taxonomy_config.aliases:
      if not rule.applies_to_prediction(context):
        continue
      if record.label_id not in (rule.source_label_id, rule.canonical_label_id):
        continue
      rows[record.split_name].add(
        tuple(sorted((rule.source_label_id, rule.canonical_label_id)))
      )
  return dict(rows)


def _matching_seed_rule(
  pair_key: tuple[str, str],
  taxonomy_config: TaxonomyEvaluationConfig,
  *,
  records: list[_DeckRecord],
  split_name: str,
):
  if not taxonomy_config.enabled:
    return None
  for rule in taxonomy_config.aliases:
    if set(pair_key) != {rule.source_label_id, rule.canonical_label_id}:
      continue
    for record in records:
      if record.split_name != split_name or record.label_id not in pair_key:
        continue
      if rule.applies_to_prediction({
        "event_date": record.event_date.isoformat(),
        "format": record.format_code,
      }):
        return rule
  return None


def _orient_pair_by_support(
  pair_key: tuple[str, str],
  *,
  supports: dict[str, dict[str, int]],
  split_name: str,
) -> tuple[str, str]:
  left, right = pair_key
  split_support = supports.get(split_name, {})
  left_count = split_support.get(left, 0)
  right_count = split_support.get(right, 0)
  if right_count > left_count:
    return right, left
  if right_count == left_count and display_label_from_id(
    right
  ) < display_label_from_id(left):
    return right, left
  return left, right


def _pair_evidence(
  label_a: str,
  label_b: str,
  *,
  split_name: str,
  pair_confusions: Counter[str],
  records: list[_DeckRecord],
  supports: dict[str, dict[str, int]],
  profile_a: _LabelProfile,
  profile_b: _LabelProfile,
  card_names: dict[int, str],
  seed_rule: Any,
  seed_names: tuple[str, ...],
  min_seed_confusion_count: int,
) -> dict[str, Any]:
  confusion_count = sum(pair_confusions.values())
  seed_stability = _seed_stability(
    pair_confusions,
    seed_names=seed_names,
    min_seed_confusion_count=min_seed_confusion_count,
  )
  split_support = supports.get(split_name, {})
  count_a = split_support.get(label_a, 0)
  count_b = split_support.get(label_b, 0)
  main_overlap = _overlap(profile_a, profile_b, ZONE_MAIN)
  side_overlap = _overlap(profile_a, profile_b, ZONE_SIDE)
  core = _core_coverage(profile_a, profile_b, card_names)
  return {
    "confusion_count": confusion_count,
    "confusion_count_by_seed": dict(sorted(pair_confusions.items())),
    **seed_stability,
    "label_a_count": count_a,
    "label_b_count": count_b,
    "minority_label_count": min(count_a, count_b),
    "majority_label_count": max(count_a, count_b),
    "support_ratio": (
      min(count_a, count_b) / max(count_a, count_b)
      if max(count_a, count_b)
      else None
    ),
    "mainboard_jaccard": main_overlap["jaccard"],
    "mainboard_adoption_weighted_jaccard": main_overlap["weighted_jaccard"],
    "sideboard_jaccard": side_overlap["jaccard"],
    "sideboard_adoption_weighted_jaccard": side_overlap["weighted_jaccard"],
    "shared_core_card_coverage": core["coverage"],
    "shared_core_cards": core["shared_card_names"],
    "seed_alias_match": seed_rule is not None,
    "seed_evidence": seed_rule.evidence if seed_rule is not None else None,
    "label_support_by_split": {
      split: {
        "label_a": split_supports.get(label_a, 0),
        "label_b": split_supports.get(label_b, 0),
      }
      for split, split_supports in sorted(supports.items())
    },
    "date_window": _candidate_date_window(
      records, split_name, (label_a, label_b)
    ),
  }


def _overlap(
  profile_a: _LabelProfile,
  profile_b: _LabelProfile,
  zone: str,
) -> dict[str, float | None]:
  adoption_a = profile_a.adoption_by_zone.get(zone, {})
  adoption_b = profile_b.adoption_by_zone.get(zone, {})
  cards = set(adoption_a) | set(adoption_b)
  if not cards:
    return {
      "jaccard": None,
      "weighted_jaccard": None,
    }
  intersection = set(adoption_a) & set(adoption_b)
  max_sum = sum(
    max(adoption_a.get(card, 0.0), adoption_b.get(card, 0.0)) for card in cards
  )
  min_sum = sum(
    min(adoption_a.get(card, 0.0), adoption_b.get(card, 0.0)) for card in cards
  )
  return {
    "jaccard": len(intersection) / len(cards),
    "weighted_jaccard": min_sum / max_sum if max_sum else None,
  }


def _seed_stability(
  pair_confusions: Counter[str],
  *,
  seed_names: tuple[str, ...],
  min_seed_confusion_count: int,
) -> dict[str, Any]:
  if seed_names:
    counts = [
      int(pair_confusions.get(seed_name, 0)) for seed_name in seed_names
    ]
  else:
    counts = [int(count) for _, count in sorted(pair_confusions.items())]
  total = sum(counts)
  if not counts or total == 0:
    return {
      "confusion_seed_count": 0,
      "confusion_active_seed_count": 0,
      "confusion_seed_consensus": 0.0,
      "confusion_seed_entropy": 0.0,
      "confusion_seed_entropy_normalized": 0.0,
      "confusion_min_seed_fraction": 0.0,
      "confusion_max_seed_fraction": 0.0,
      "confusion_seed_balance": 0.0,
    }

  fractions = [count / total for count in counts]
  entropy = -sum(
    fraction * math.log(fraction) for fraction in fractions if fraction > 0
  )
  normalized_entropy = (
    entropy / math.log(len(counts)) if len(counts) > 1 else 1.0
  )
  active_count = sum(1 for count in counts if count >= min_seed_confusion_count)
  max_fraction = max(fractions)
  min_fraction = min(fractions)
  return {
    "confusion_seed_count": len(counts),
    "confusion_active_seed_count": active_count,
    "confusion_seed_consensus": active_count / len(counts),
    "confusion_seed_entropy": entropy,
    "confusion_seed_entropy_normalized": normalized_entropy,
    "confusion_min_seed_fraction": min_fraction,
    "confusion_max_seed_fraction": max_fraction,
    "confusion_seed_balance": 1.0 - max_fraction,
  }


def _pair_summaries(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
  grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
  for candidate in candidates:
    key = tuple(sorted((candidate["label_a_id"], candidate["label_b_id"])))
    grouped[key].append(candidate)

  rows: list[dict[str, Any]] = []
  for key, group in grouped.items():
    confidences = [float(row["confidence"]) for row in group]
    aggregate_confusion_count = sum(
      int(row["evidence"].get("confusion_count") or 0) for row in group
    )
    seed_alias_match_any = any(
      bool(row["evidence"].get("seed_alias_match")) for row in group
    )
    best = max(
      group,
      key=lambda row: (
        float(row["confidence"]),
        int(row["evidence"].get("confusion_count") or 0),
      ),
    )
    label_a_id, label_b_id = _orient_summary_pair(key, group)
    rows.append({
      "pair": [
        display_label_from_id(label_a_id),
        display_label_from_id(label_b_id),
      ],
      "pair_ids": [label_a_id, label_b_id],
      "windows_seen": sorted({
        str(row["rolling_window"])
        for row in group
        if row.get("rolling_window") is not None
      }),
      "splits_seen": sorted({
        str(row["split_name"])
        for row in group
        if row.get("split_name") is not None
      }),
      "candidate_count": len(group),
      "max_confidence": max(confidences),
      "mean_confidence": sum(confidences) / len(confidences),
      "seed_alias_match_any": seed_alias_match_any,
      "aggregate_confusion_count": aggregate_confusion_count,
      "best_relation": best["suggested_relation"],
      "recommendation": _pair_recommendation(
        best,
        seed_alias_match_any=seed_alias_match_any,
        aggregate_confusion_count=aggregate_confusion_count,
      ),
      "best_candidate": {
        "rolling_window": best["rolling_window"],
        "split_name": best["split_name"],
        "window": best["window"],
        "format": best["format"],
        "confidence": best["confidence"],
        "suggested_relation": best["suggested_relation"],
        "action": best["action"],
      },
    })

  rows.sort(
    key=lambda row: (
      -float(row["max_confidence"]),
      -int(row["aggregate_confusion_count"]),
      row["pair"][0],
      row["pair"][1],
    )
  )
  return rows


def _orient_summary_pair(
  key: tuple[str, str],
  group: list[dict[str, Any]],
) -> tuple[str, str]:
  score: Counter[str] = Counter()
  for row in group:
    score[str(row["label_a_id"])] += int(
      row["evidence"].get("majority_label_count") or 0
    )
    score[str(row["label_b_id"])] += int(
      row["evidence"].get("minority_label_count") or 0
    )
  left, right = key
  if score[right] > score[left]:
    return right, left
  if score[right] == score[left] and display_label_from_id(
    right
  ) < display_label_from_id(left):
    return right, left
  return left, right


def _pair_recommendation(
  best: dict[str, Any],
  *,
  seed_alias_match_any: bool,
  aggregate_confusion_count: int,
) -> str:
  confidence = float(best["confidence"])
  relation = str(best["suggested_relation"])
  if seed_alias_match_any and relation in (
    "alias_candidate",
    "same_family_candidate",
  ):
    return "reviewed_alias_candidate"
  if confidence >= 0.75 and aggregate_confusion_count >= 25:
    return "high_priority_review"
  if confidence >= 0.55:
    return "review"
  return "monitor"


def _weak_label_observations(
  candidates: list[dict[str, Any]],
  *,
  training_eligible_by_split: bool = True,
  training_exclusion_reason: str | None = None,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for candidate in candidates:
    evidence = candidate["evidence"]
    date_window = evidence.get("date_window") or {}
    split_name = str(candidate["split_name"])
    is_held_out = split_name in HELD_OUT_EVIDENCE_SPLITS
    usable_for_training = training_eligible_by_split and not is_held_out
    valid_from = date_window.get("min")
    valid_to = date_window.get("max")
    if valid_from is None or valid_to is None:
      window_parts = str(candidate["window"]).split("..", 1)
      valid_from = valid_from or window_parts[0]
      valid_to = valid_to or window_parts[-1]
    rows.append({
      "label_a": candidate["label_a"],
      "label_b": candidate["label_b"],
      "label_a_id": candidate["label_a_id"],
      "label_b_id": candidate["label_b_id"],
      "relation": candidate["suggested_relation"],
      "confidence": candidate["confidence"],
      "format": candidate.get("format"),
      "valid_from": valid_from,
      "valid_to": valid_to,
      "rolling_window": candidate.get("rolling_window"),
      "split_name": split_name,
      "evidence_sources": _candidate_observation_sources(candidate),
      "seed_consensus": evidence.get("confusion_seed_consensus"),
      "seed_entropy": evidence.get("confusion_seed_entropy"),
      "seed_entropy_normalized": evidence.get(
        "confusion_seed_entropy_normalized"
      ),
      "seed_balance": evidence.get("confusion_seed_balance"),
      "label_a_count": evidence.get("label_a_count"),
      "label_b_count": evidence.get("label_b_count"),
      "minority_label_count": evidence.get("minority_label_count"),
      "majority_label_count": evidence.get("majority_label_count"),
      "support_ratio": evidence.get("support_ratio"),
      "confusion_count": evidence.get("confusion_count"),
      "confusion_count_by_seed": evidence.get("confusion_count_by_seed"),
      "mainboard_adoption_weighted_jaccard": evidence.get(
        "mainboard_adoption_weighted_jaccard"
      ),
      "sideboard_adoption_weighted_jaccard": evidence.get(
        "sideboard_adoption_weighted_jaccard"
      ),
      "shared_core_card_coverage": evidence.get("shared_core_card_coverage"),
      "seed_alias_match": evidence.get("seed_alias_match"),
      "seed_evidence": evidence.get("seed_evidence"),
      "recommendation": candidate["action"],
      "usable_for_training_suggestion": usable_for_training,
      "training_exclusion_reason": (
        training_exclusion_reason
        if not training_eligible_by_split
        else "held_out_evaluation_split" if is_held_out else None
      ),
    })
  return rows


def _candidate_observation_sources(candidate: dict[str, Any]) -> list[str]:
  evidence = candidate["evidence"]
  sources: list[str] = []
  if int(evidence.get("confusion_count") or 0) > 0:
    sources.append("model_confusion")
  if evidence.get("mainboard_adoption_weighted_jaccard") is not None:
    sources.append("deck_overlap")
  support_ratio = evidence.get("support_ratio")
  if support_ratio is not None and float(support_ratio) <= 0.25:
    sources.append("label_support_imbalance")
  if evidence.get("seed_alias_match"):
    sources.append("seed_alias")
  return sources


def _unknown_review_candidates(
  prediction_rows: list[dict[str, Any]],
  *,
  limit: int = 100,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for row in prediction_rows:
    if row.get("source_label_id") is not None and not row.get(
      "is_low_confidence"
    ):
      continue
    if row.get("source_label_id") is not None and row.get("top1_label_id") is None:
      continue
    reason: list[str] = []
    if row.get("source_label_id") is None:
      reason.append("source_unlabeled")
    if row.get("is_low_confidence"):
      reason.append("low_confidence")
    if not reason:
      continue
    rows.append({
      "deck_id": row.get("deck_id"),
      "event_id": row.get("event_id"),
      "event_date": row.get("event_date"),
      "format": row.get("format"),
      "split_name": row.get("split_name"),
      "source_label": row.get("source_label"),
      "source_label_id": row.get("source_label_id"),
      "top1_label": row.get("top1_label"),
      "top1_label_id": row.get("top1_label_id"),
      "top1_probability": row.get("top1_probability"),
      "temperature_scaled_probability": row.get(
        "temperature_scaled_probability"
      ),
      "energy_score": row.get("energy_score"),
      "msp_score": row.get("msp_score"),
      "reason": reason,
      "action": "review_unknown_or_low_confidence",
    })
  rows.sort(
    key=lambda row: (
      float(row.get("temperature_scaled_probability") or 0.0),
      str(row.get("event_date") or ""),
      str(row.get("deck_id") or ""),
    )
  )
  return rows[:limit]


def _backfill_report(
  prediction_rows: list[dict[str, Any]],
  *,
  manifest: dict[str, Any],
  deck_embeddings: Path | None,
  train_label_ids: set[str],
) -> dict[str, Any]:
  source_unlabeled = [
    row for row in prediction_rows if row.get("source_label_id") is None
  ]
  source_unseen = [
    row
    for row in prediction_rows
    if (
      row.get("source_label_id") is not None
      and str(row["source_label_id"]) not in train_label_ids
    )
  ]
  low_confidence = [
    row for row in prediction_rows if row.get("is_low_confidence")
  ]
  disagreements = [
    row
    for row in prediction_rows
    if (
      row.get("source_label_id") is not None
      and row.get("top1_label_id") is not None
      and str(row["source_label_id"]) != str(row["top1_label_id"])
    )
  ]
  return {
    "run_id": "model_backfill_report",
    "prediction_count": len(prediction_rows),
    "embedding_count": _embedding_count(deck_embeddings, manifest),
    "source_unlabeled_count": len(source_unlabeled),
    "source_unseen_count": len(source_unseen),
    "low_confidence_count": len(low_confidence),
    "low_confidence_rate": (
      len(low_confidence) / len(prediction_rows) if prediction_rows else 0.0
    ),
    "top_predicted_labels_for_unlabeled_decks": _top_counts(
      source_unlabeled,
      "top1_label",
    ),
    "top_source_label_disagreements": _top_pair_counts(
      disagreements,
      "source_label",
      "top1_label",
    ),
    "top_low_confidence_known_source_decks": _low_confidence_known_source_rows(
      prediction_rows,
    ),
    "top_source_unseen_labels": _top_counts(source_unseen, "source_label"),
    "artifact_version": manifest.get("model_version"),
    "model_family": manifest.get("model_family"),
    "saved_model": manifest.get("saved_model"),
    "dataset_path": manifest.get("dataset_path"),
  }


def _embedding_count(
  deck_embeddings: Path | None, manifest: dict[str, Any]
) -> int | None:
  if deck_embeddings is not None and deck_embeddings.exists():
    return pq.read_table(deck_embeddings).num_rows
  value = manifest.get("deck_embedding_count")
  return int(value) if value is not None else None


def _top_counts(
  rows: list[dict[str, Any]],
  field_name: str,
  *,
  limit: int = 20,
) -> list[dict[str, Any]]:
  counts: Counter[str] = Counter()
  for row in rows:
    value = row.get(field_name)
    if value is not None:
      counts[str(value)] += 1
  return [
    {
      field_name: key,
      "count": count,
    }
    for key, count in counts.most_common(limit)
  ]


def _top_pair_counts(
  rows: list[dict[str, Any]],
  left_field: str,
  right_field: str,
  *,
  limit: int = 20,
) -> list[dict[str, Any]]:
  counts: Counter[tuple[str, str]] = Counter()
  for row in rows:
    left = row.get(left_field)
    right = row.get(right_field)
    if left is not None and right is not None:
      counts[(str(left), str(right))] += 1
  return [
    {
      left_field: left,
      right_field: right,
      "count": count,
    }
    for (left, right), count in counts.most_common(limit)
  ]


def _low_confidence_known_source_rows(
  prediction_rows: list[dict[str, Any]],
  *,
  limit: int = 20,
) -> list[dict[str, Any]]:
  rows = [
    row
    for row in prediction_rows
    if row.get("source_label_id") is not None and row.get("is_low_confidence")
  ]
  rows.sort(
    key=lambda row: (
      float(row.get("temperature_scaled_probability") or 0.0),
      str(row.get("event_date") or ""),
      str(row.get("deck_id") or ""),
    )
  )
  return [
    {
      "deck_id": row.get("deck_id"),
      "event_id": row.get("event_id"),
      "event_date": row.get("event_date"),
      "format": row.get("format"),
      "split_name": row.get("split_name"),
      "source_label": row.get("source_label"),
      "source_label_id": row.get("source_label_id"),
      "top1_label": row.get("top1_label"),
      "top1_label_id": row.get("top1_label_id"),
      "temperature_scaled_probability": row.get(
        "temperature_scaled_probability"
      ),
      "energy_score": row.get("energy_score"),
    }
    for row in rows[:limit]
  ]


def _mean(values: Any) -> float | None:
  rows = list(values)
  return sum(rows) / len(rows) if rows else None


def _weak_label_observations_path(output: Path) -> Path:
  return output.with_name("alias_weak_label_observations.jsonl")


def _backfill_report_path(output: Path) -> Path:
  return output.with_name("backfill_report.json")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as output_file:
    for row in rows:
      output_file.write(json.dumps(row, sort_keys=True) + "\n")


def _core_coverage(
  profile_a: _LabelProfile,
  profile_b: _LabelProfile,
  card_names: dict[int, str],
  *,
  adoption_threshold: float = 0.5,
) -> dict[str, Any]:
  majority, minority = (
    (profile_a, profile_b)
    if profile_a.deck_count >= profile_b.deck_count
    else (profile_b, profile_a)
  )
  majority_core = {
    card
    for card, adoption in majority.adoption_by_zone.get(ZONE_MAIN, {}).items()
    if adoption >= adoption_threshold
  }
  shared = [
    card
    for card in sorted(majority_core)
    if minority.adoption_by_zone.get(ZONE_MAIN, {}).get(card, 0.0)
    >= adoption_threshold
  ]
  return {
    "coverage": f"{len(shared)}/{len(majority_core)}",
    "shared_card_names": [
      card_names.get(card, str(card)) for card in shared[:25]
    ],
  }


def _confidence(
  evidence: dict[str, Any],
  *,
  seed_rule: Any,
) -> float:
  overlap = float(evidence.get("mainboard_adoption_weighted_jaccard") or 0.0)
  confusion_count = int(evidence.get("confusion_count") or 0)
  minority_count = int(evidence.get("minority_label_count") or 0)
  support_ratio = float(evidence.get("support_ratio") or 0.0)
  seed_consensus = float(evidence.get("confusion_seed_consensus") or 0.0)
  seed_entropy = float(evidence.get("confusion_seed_entropy_normalized") or 0.0)
  confusion_score = min(1.0, math.log1p(confusion_count) / math.log1p(50))
  stable_confusion_score = confusion_score * (0.35 + 0.65 * seed_consensus)
  seed_stability_score = 0.5 * seed_consensus + 0.5 * seed_entropy
  minority_score = 1.0 - min(1.0, minority_count / 100)
  imbalance_score = 1.0 - min(1.0, support_ratio)
  seed_score = 1.0 if seed_rule is not None else 0.0
  score = (
    0.45 * overlap
    + 0.18 * stable_confusion_score
    + 0.10 * seed_stability_score
    + 0.10 * minority_score
    + 0.07 * imbalance_score
    + 0.10 * seed_score
  )
  return round(max(0.0, min(0.99, score)), 4)


def _suggested_relation(
  confidence: float,
  evidence: dict[str, Any],
) -> str:
  overlap = float(evidence.get("mainboard_adoption_weighted_jaccard") or 0.0)
  support_ratio = float(evidence.get("support_ratio") or 0.0)
  seed_consensus = float(evidence.get("confusion_seed_consensus") or 0.0)
  if evidence.get("seed_alias_match") and overlap >= 0.6:
    return "alias_candidate"
  if confidence >= 0.78 and overlap >= 0.72 and support_ratio <= 0.12:
    return "same_family_candidate"
  if confidence >= 0.65 and overlap >= 0.55:
    return "sibling_variant_candidate"
  if confidence >= 0.45 or seed_consensus >= 0.5:
    return "model_confusion_candidate"
  return "uncertain_boundary"


def _suggested_action(
  confidence: float,
  evidence: dict[str, Any],
) -> str:
  if evidence.get("seed_alias_match") and confidence >= 0.7:
    return "emit_high_confidence_weak_evidence"
  if confidence >= 0.75:
    return "review_for_weak_evidence"
  if confidence >= 0.5:
    return "review"
  return "monitor"


def _candidate_window(
  records: list[_DeckRecord],
  split_name: str,
  labels: tuple[str, str],
) -> str:
  dates = _candidate_dates(records, split_name, labels)
  if not dates:
    return "unknown"
  return f"{dates[0].isoformat()}..{dates[-1].isoformat()}"


def _candidate_date_window(
  records: list[_DeckRecord],
  split_name: str,
  labels: tuple[str, str],
) -> dict[str, str | None]:
  dates = _candidate_dates(records, split_name, labels)
  return {
    "min": dates[0].isoformat() if dates else None,
    "max": dates[-1].isoformat() if dates else None,
  }


def _candidate_dates(
  records: list[_DeckRecord],
  split_name: str,
  labels: tuple[str, str],
) -> list[date]:
  label_set = set(labels)
  return sorted({
    record.event_date
    for record in records
    if record.split_name == split_name and record.label_id in label_set
  })


def _window_format(records: list[_DeckRecord]) -> str | None:
  formats = sorted({record.format_code for record in records})
  if len(formats) == 1:
    return formats[0]
  return None

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from manafold.constants import PROJECT_ROOT

UNKNOWN_LABEL_ID = "unknown"
UNLABELED_TARGET_SOURCE = "unlabeled"


@dataclass(frozen=True)
class DeckToken:
  card_idx: int
  quantity: int
  zone_idx: int


@dataclass(frozen=True)
class DeckModelInput:
  dataset_version: str
  deck_id: str
  event_id: str
  event_date: date | None
  format_code: str
  split_name: str
  target_label_id: str
  source_label_id: str
  target_source: str
  tokens: tuple[DeckToken, ...]
  expected_mainboard_size: int | None = None
  observation_complete: bool = True


@dataclass(frozen=True)
class TrainingDataset:
  dataset_path: Path
  dataset_version: str
  target_source: str | None
  card_vocab: tuple[dict[str, Any], ...]
  zone_vocab: dict[str, int]
  examples: tuple[DeckModelInput, ...]
  labels: tuple[str, ...]

  @property
  def card_count(self) -> int:
    return len(self.card_vocab)

  @property
  def zone_count(self) -> int:
    if not self.zone_vocab:
      return 0
    return max(self.zone_vocab.values()) + 1


def resolve_dataset_path(dataset_path: Path | str | None = None) -> Path:
  """Hermetically resolve the dataset directory for local CLI, CI runners, or production pipelines.

  Resolution Priority:
  1. Explicit argument `dataset_path` if provided and exists.
  2. `MANAFOLD_DATASET_DIR` or `DATASET_DIR` environment variables.
  3. Non-timestamped standard path: `data/dataset` or `data/current_dataset`.
  4. Automatic discovery of the latest window directory under `data/`.
  """
  import os
  if dataset_path is not None:
    p = Path(dataset_path)
    if p.exists():
      return p.resolve()

  env_path = os.environ.get("MANAFOLD_DATASET_DIR") or os.environ.get("DATASET_DIR")
  if env_path and Path(env_path).exists():
    return Path(env_path).resolve()

  for default_name in (
    "data/dataset",
    "data/current_dataset",
    "data/rolling_eval_a3_bakeoff_track_r_20260726/windows/window_1",
  ):
    p = PROJECT_ROOT / default_name
    if p.exists():
      return p.resolve()

  data_root = PROJECT_ROOT / "data"
  if data_root.exists():
    candidates = list(data_root.glob("**/dataset_manifest.json"))
    if candidates:
      latest_manifest = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
      return latest_manifest.parent.resolve()

  raise FileNotFoundError(
    "Could not resolve dataset directory. Specify --dataset-dir, set MANAFOLD_DATASET_DIR, "
    "or place a dataset in data/dataset."
  )


def load_training_dataset(
  dataset_path: Path | str | None = None,
  *,
  target_source: str | None = None,
) -> TrainingDataset:
  resolved_path = resolve_dataset_path(dataset_path)
  manifest = _read_manifest(resolved_path)
  artifacts = manifest["artifacts"]
  examples = build_training_inputs(
  deck_tokens=_read_parquet(resolved_path / artifacts["deck_tokens"]),
  split_manifest=_read_parquet(resolved_path / artifacts["split_manifest"]),
  proxy_targets=_read_parquet(resolved_path / artifacts["proxy_targets"]),
  target_source=target_source,
  )
  labels = tuple(
  sorted(
      {
  example.target_label_id
  for example in examples
  if example.split_name == "train"
      }
  )
  )

  return TrainingDataset(
  dataset_path=resolved_path,
  dataset_version=str(manifest["dataset_version"]),
  target_source=target_source,
  card_vocab=tuple(_read_parquet(resolved_path / artifacts["card_vocab"])),
  zone_vocab={
      str(zone): int(index)
      for zone, index in _read_json(resolved_path / artifacts["zone_vocab"]).items()
  },
  examples=tuple(examples),
  labels=labels,
  )


def load_deck_inputs(
  dataset_path: Path,
  *,
  target_source: str | None = None,
) -> list[DeckModelInput]:
  manifest = _read_manifest(dataset_path)
  artifacts = manifest["artifacts"]
  return build_training_inputs(
  deck_tokens=_read_parquet(dataset_path / artifacts["deck_tokens"]),
  split_manifest=_read_parquet(dataset_path / artifacts["split_manifest"]),
  proxy_targets=_read_parquet(dataset_path / artifacts["proxy_targets"]),
  target_source=target_source,
  )


def build_training_inputs(
  *,
  deck_tokens: list[dict[str, Any]],
  split_manifest: list[dict[str, Any]],
  proxy_targets: list[dict[str, Any]],
  target_source: str | None = None,
) -> list[DeckModelInput]:
  target_by_deck = _select_targets(proxy_targets, target_source)
  return _build_inputs(
  deck_tokens=deck_tokens,
  split_manifest=split_manifest,
  target_by_deck=target_by_deck,
  deck_ids=sorted(target_by_deck),
  target_source=target_source,
  include_unlabeled=False,
  )


def build_scoring_inputs(
  *,
  deck_tokens: list[dict[str, Any]],
  split_manifest: list[dict[str, Any]],
  proxy_targets: list[dict[str, Any]] | None = None,
  target_source: str | None = None,
  unknown_label_id: str = UNKNOWN_LABEL_ID,
) -> list[DeckModelInput]:
  target_by_deck = _select_targets(proxy_targets or [], target_source)
  tokens_by_deck = _tokens_by_deck(deck_tokens)
  split_by_deck = _split_by_deck(split_manifest)
  return _build_inputs(
  deck_tokens=deck_tokens,
  split_manifest=split_manifest,
  target_by_deck=target_by_deck,
  deck_ids=sorted(set(tokens_by_deck).intersection(split_by_deck)),
  target_source=target_source,
  include_unlabeled=True,
  unknown_label_id=unknown_label_id,
  )


def _build_inputs(
  *,
  deck_tokens: list[dict[str, Any]],
  split_manifest: list[dict[str, Any]],
  target_by_deck: dict[str, dict[str, Any]],
  deck_ids: list[str],
  target_source: str | None,
  include_unlabeled: bool,
  unknown_label_id: str = UNKNOWN_LABEL_ID,
) -> list[DeckModelInput]:
  tokens_by_deck = _tokens_by_deck(deck_tokens)
  split_by_deck = _split_by_deck(split_manifest)
  dataset_version_by_deck = _dataset_version_by_deck(deck_tokens, split_manifest)

  examples: list[DeckModelInput] = []
  for deck_id in deck_ids:
    target = target_by_deck.get(deck_id)
    if target is None and not include_unlabeled:
      continue
    tokens = tuple(tokens_by_deck.get(deck_id, []))
    split = split_by_deck.get(deck_id)
    if not tokens or split is None:
      continue

    if target is None:
      label_id = unknown_label_id
      example_target_source = target_source or UNLABELED_TARGET_SOURCE
      dataset_version = dataset_version_by_deck.get(deck_id, "")
    else:
      label_id = str(target["proxy_label_id"])
      example_target_source = str(target["target_source"])
      dataset_version = str(
        target.get("dataset_version")
        or dataset_version_by_deck.get(deck_id, "")
      )

    examples.append(
      DeckModelInput(
        dataset_version=dataset_version,
        deck_id=deck_id,
        event_id=str(split["event_id"]),
        event_date=_date_or_none(split.get("event_date")),
        format_code=str(split.get("format") or "").casefold(),
        split_name=str(split["split_name"]),
        target_label_id=label_id,
        source_label_id=label_id,
        target_source=example_target_source,
        tokens=tokens,
      )
    )

  return examples


def _tokens_by_deck(deck_tokens: list[dict[str, Any]]) -> dict[str, list[DeckToken]]:
  tokens_by_deck: dict[str, list[DeckToken]] = {}
  for token in deck_tokens:
    tokens_by_deck.setdefault(token["deck_id"], []).append(
      DeckToken(
        card_idx=int(token["card_idx"]),
        quantity=int(token["quantity"]),
        zone_idx=int(token["zone_idx"]),
      )
    )
  return tokens_by_deck


def _split_by_deck(split_manifest: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  return {
    row["deck_id"]: row
    for row in split_manifest
  }


def _dataset_version_by_deck(
  deck_tokens: list[dict[str, Any]],
  split_manifest: list[dict[str, Any]],
) -> dict[str, str]:
  dataset_versions: dict[str, str] = {}
  for token in deck_tokens:
    if token.get("dataset_version") is not None:
      dataset_versions.setdefault(
        str(token["deck_id"]),
        str(token["dataset_version"]),
      )
  for split in split_manifest:
    if split.get("dataset_version") is not None:
      dataset_versions.setdefault(
        str(split["deck_id"]),
        str(split["dataset_version"]),
      )
  return dataset_versions


def group_inputs_by_split(
  examples: list[DeckModelInput],
) -> dict[str, list[DeckModelInput]]:
  rows: dict[str, list[DeckModelInput]] = {}
  for example in examples:
    rows.setdefault(example.split_name, []).append(example)

  return {
    split_name: sorted(split_rows, key=lambda example: example.deck_id)
    for split_name, split_rows in rows.items()
  }


def _select_targets(
  proxy_targets: list[dict[str, Any]],
  target_source: str | None,
) -> dict[str, dict[str, Any]]:
  rows: dict[str, dict[str, Any]] = {}
  for target in sorted(
    proxy_targets,
    key=lambda row: (
      row["deck_id"],
      row["target_source"],
      row["proxy_label_id"],
    ),
  ):
    if target_source is not None and target["target_source"] != target_source:
      continue
    rows.setdefault(target["deck_id"], target)

  return rows


def _date_or_none(value: Any) -> date | None:
  if value is None:
    return None
  if isinstance(value, date):
    return value
  return date.fromisoformat(str(value))


def _read_parquet(path: Path) -> list[dict[str, Any]]:
  return pq.read_table(path).to_pylist()


def _read_json(path: Path) -> dict[str, Any]:
  import json

  data = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(data, dict):
    raise RuntimeError(f"{path} must contain a JSON object.")
  return data


def _read_manifest(dataset_path: Path) -> dict[str, Any]:
  import json

  return json.loads(
    (dataset_path / "dataset_manifest.json").read_text(encoding="utf-8")
  )

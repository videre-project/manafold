from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from manafold.data.validate import write_parquet

def _write_dataset(dataset_path: Path, *, final_holdout: bool = False) -> None:
  artifacts = {
    "card_vocab": "card_vocab.parquet",
    "deck_tokens": "deck_tokens.parquet",
    "zone_vocab": "zone_vocab.json",
    "split_manifest": "split_manifest.parquet",
    "proxy_targets": "proxy_targets.parquet",
  }
  (dataset_path / "dataset_manifest.json").write_text(
    json.dumps(
      {
        "dataset_version": "modern_2024_2024_v0",
        "artifacts": artifacts,
      }
    )
  )
  write_parquet(
    dataset_path / artifacts["card_vocab"],
    [
      _card_vocab(0),
      _card_vocab(1),
      _card_vocab(2),
    ],
    "card_vocab",
  )
  (dataset_path / artifacts["zone_vocab"]).write_text(
    json.dumps(
      {
        "main": 0,
        "side": 1,
        "companion": 2,
        "commander": 3,
        "other": 4,
      }
    )
  )
  deck_tokens = [
      _token("train-a", 0, 4),
      _token("train-a", 1, 2),
      _token("train-b", 1, 4),
      _token("train-b", 2, 2),
      _token("validation-a", 0, 4),
      _token("validation-a", 1, 2),
      _token("validation-gamma", 1, 4),
      _token("validation-gamma", 2, 2),
  ]
  if final_holdout:
    deck_tokens.extend(
      [
        _token("dev-test-b", 1, 4),
        _token("dev-test-b", 2, 2),
        _token("final-test-b", 1, 4),
        _token("final-test-b", 2, 2),
      ]
    )
  else:
    deck_tokens.extend(
      [
        _token("test-b", 1, 4),
        _token("test-b", 2, 2),
      ]
    )
  write_parquet(
    dataset_path / artifacts["deck_tokens"],
    deck_tokens,
    "deck_tokens",
  )
  split_manifest = [
      _split("train-a", "train"),
      _split("train-b", "train"),
      _split("validation-a", "validation"),
      _split("validation-gamma", "validation"),
  ]
  if final_holdout:
    split_manifest.extend(
      [
        _split("dev-test-b", "dev-test"),
        _split("final-test-b", "final-test"),
      ]
    )
  else:
    split_manifest.append(_split("test-b", "test"))
  write_parquet(
    dataset_path / artifacts["split_manifest"],
    split_manifest,
    "split_manifest",
  )
  proxy_targets = [
      _proxy_target("train-a", "alpha"),
      _proxy_target("train-b", "beta"),
      _proxy_target("validation-a", "alpha"),
      _proxy_target("validation-gamma", "gamma"),
  ]
  if final_holdout:
    proxy_targets.extend(
      [
        _proxy_target("dev-test-b", "beta"),
        _proxy_target("final-test-b", "beta"),
      ]
    )
  else:
    proxy_targets.append(_proxy_target("test-b", "beta"))
  write_parquet(
    dataset_path / artifacts["proxy_targets"],
    proxy_targets,
    "proxy_targets",
  )


def _rewrite_dataset_with_local_card_indexes(dataset_path: Path) -> None:
  manifest = json.loads((dataset_path / "dataset_manifest.json").read_text())
  artifacts = manifest["artifacts"]
  card_vocab_path = dataset_path / artifacts["card_vocab"]
  deck_tokens_path = dataset_path / artifacts["deck_tokens"]
  card_vocab = pq.read_table(card_vocab_path).to_pylist()
  reordered_vocab = [card_vocab[1], card_vocab[0], card_vocab[2]]
  oracle_to_local_idx: dict[str, int] = {}
  remapped_vocab: list[dict[str, object]] = []
  for local_idx, card in enumerate(reordered_vocab):
    remapped_card = dict(card)
    remapped_card["card_idx"] = local_idx
    oracle_to_local_idx[str(remapped_card["oracle_id"])] = local_idx
    remapped_vocab.append(remapped_card)
  write_parquet(card_vocab_path, remapped_vocab, "card_vocab")

  remapped_tokens: list[dict[str, object]] = []
  for token in pq.read_table(deck_tokens_path).to_pylist():
    remapped_token = dict(token)
    remapped_token["card_idx"] = oracle_to_local_idx[str(token["oracle_id"])]
    remapped_tokens.append(remapped_token)
  write_parquet(deck_tokens_path, remapped_tokens, "deck_tokens")


def _remove_proxy_targets_artifact(dataset_path: Path) -> None:
  manifest_path = dataset_path / "dataset_manifest.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  proxy_targets = manifest["artifacts"].pop("proxy_targets")
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
  (dataset_path / proxy_targets).unlink()


def _write_alias_candidate_dataset(dataset_path: Path) -> None:
  artifacts = {
    "card_vocab": "card_vocab.parquet",
    "deck_tokens": "deck_tokens.parquet",
    "zone_vocab": "zone_vocab.json",
    "split_manifest": "split_manifest.parquet",
    "proxy_targets": "proxy_targets.parquet",
  }
  (dataset_path / "dataset_manifest.json").write_text(
    json.dumps(
      {
        "dataset_version": "modern_alias_candidate_test",
        "artifacts": artifacts,
      }
    )
  )
  write_parquet(
    dataset_path / artifacts["card_vocab"],
    [
      _card_vocab(0),
      _card_vocab(1),
      _card_vocab(2),
    ],
    "card_vocab",
  )
  (dataset_path / artifacts["zone_vocab"]).write_text(
    json.dumps(
      {
        "main": 0,
        "side": 1,
        "companion": 2,
        "commander": 3,
        "other": 4,
      }
    )
  )
  write_parquet(
    dataset_path / artifacts["deck_tokens"],
    [
      _token("final-alpha", 0, 4),
      _token("final-alpha", 1, 4),
      _token("final-beta", 0, 4),
      _token("final-beta", 1, 4),
      _token("final-beta", 2, 2),
    ],
    "deck_tokens",
  )
  write_parquet(
    dataset_path / artifacts["split_manifest"],
    [
      _split("final-alpha", "final-test"),
      _split("final-beta", "final-test"),
    ],
    "split_manifest",
  )
  write_parquet(
    dataset_path / artifacts["proxy_targets"],
    [
      _proxy_target("final-alpha", "alpha"),
      _proxy_target("final-beta", "beta"),
    ],
    "proxy_targets",
  )


def _token(deck_id: str, card_idx: int, quantity: int) -> dict[str, object]:
  return {
    "dataset_version": "modern_2024_2024_v0",
    "deck_id": deck_id,
    "card_idx": card_idx,
    "oracle_id": f"oracle-{card_idx}",
    "quantity": quantity,
    "zone_idx": 0,
    "zone": "main",
  }


def _card_vocab(card_idx: int) -> dict[str, object]:
  return {
    "dataset_version": "modern_2024_2024_v0",
    "card_idx": card_idx,
    "oracle_id": f"oracle-{card_idx}",
    "primary_name": f"Card {card_idx}",
    "first_seen_at": None,
    "last_seen_at": None,
  }


def _split(deck_id: str, split_name: str) -> dict[str, object]:
  return {
    "dataset_version": "modern_2024_2024_v0",
    "deck_id": deck_id,
    "event_id": f"event-{deck_id}",
    "event_date": date(2024, 1, 1),
    "format": "modern",
    "source": "mtgo-db",
    "split_name": split_name,
    "split_strategy": "event_forward_modern_2024_2024_v0",
  }


def _proxy_target(deck_id: str, label: str) -> dict[str, object]:
  return {
    "dataset_version": "modern_2024_2024_v0",
    "deck_id": deck_id,
    "target_source": "source_archetype_name_proxy",
    "target_level": "family",
    "proxy_label_id": f"proxy.source_archetype_name_proxy.{label}",
    "display_label": label.title(),
    "normalized_label": label,
    "source_field": "source_archetype_name",
    "confidence": 0.35,
    "provenance": "mtgo-db:archetypes.archetype|proxy_target",
  }


def _prediction(
  deck_id: str,
  split_name: str,
  actual_label_id: str,
  predicted_label_id: str,
  score: float,
  actual_label_train_support: int,
  *,
  event_date: str = "2026-02-01",
  format_code: str = "modern",
) -> dict[str, object]:
  top_label_ids = [predicted_label_id]
  return {
    "deck_id": deck_id,
    "event_id": f"event-{deck_id}",
    "event_date": event_date,
    "format": format_code,
    "split_name": split_name,
    "target_source": "source_archetype_name_proxy",
    "actual_label_id": actual_label_id,
    "actual_label_train_support": actual_label_train_support,
    "is_train_unseen_label": actual_label_train_support == 0,
    "predicted_label_id": predicted_label_id,
    "score": score,
    "max_softmax_probability": score,
    "temperature_scaled_max_softmax_probability": score,
    "entropy": 1.0 - score,
    "normalized_entropy": 1.0 - score,
    "energy": 1.0 - score,
    "top_label_ids": top_label_ids,
    "top_scores": [score],
    "is_correct": actual_label_id == predicted_label_id,
    "is_top_3_correct": actual_label_id in top_label_ids,
  }



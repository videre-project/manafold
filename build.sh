#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${ROOT}/src/onnx-runtime"
CONFIG="${RUNTIME}/formats.json"
cd "${ROOT}"

END_DATE="${END_DATE:-$(date +%F)}"
END_MONTH="$(date -d "${END_DATE}" +%Y-%m-01)"
TEST_MONTH="${TEST_MONTH:-$(date -d "${END_MONTH} -1 month" +%Y-%m-01)}"
VALIDATION_END="${VALIDATION_END:-$(date -d "${TEST_MONTH} -1 day" +%F)}"
TRAIN_END="${TRAIN_END:-$(date -d "${TEST_MONTH} -1 month -1 day" +%F)}"
START_YEAR="$(date -d "${END_DATE} -3 years" +%Y)"
START_DATE="${START_DATE:-${START_YEAR}-01-01}"
EPOCHS="${EPOCHS:-40}"
SEED="${SEED:-13}"
DEVICE="${DEVICE:-cuda}"

mapfile -t ACTIVE_FORMATS < <(
  node --input-type=module -e '
    import { readFileSync } from "node:fs";
    const config = JSON.parse(readFileSync(process.argv[1], "utf8"));
    const retired = new Set(config.retired_formats || []);
    for (const format of config.formats || []) {
      if (!retired.has(format)) console.log(format);
    }
  ' "${CONFIG}"
)

FORMATS=("$@")
if ((${#FORMATS[@]} == 0)); then
  FORMATS=("${ACTIVE_FORMATS[@]}")
fi

READY_FORMATS=()
for format in "${FORMATS[@]}"; do
  if [[ ! " ${ACTIVE_FORMATS[*]} " =~ " ${format} " ]]; then
    echo "Unsupported or retired format: ${format}" >&2
    exit 2
  fi
done

for format in "${FORMATS[@]}"; do
  dataset="${ROOT}/data/releases/${format}/dataset"
  evaluation_model="${ROOT}/data/releases/${format}/evaluation_model"
  evaluation_results="${ROOT}/data/releases/${format}/evaluation_results.json"
  saved_model="${ROOT}/data/releases/${format}/model"
  results="${ROOT}/data/releases/${format}/training_results.json"
  family_results="${ROOT}/data/releases/${format}/family_metrics.json"
  version="${format}_${START_DATE%%-*}_${END_DATE%%-*}_v0"

  echo "==> Exporting ${format}"
  bazel run //:manafold -- dataset \
    --format "${format}" \
    --start "${START_DATE}" \
    --train-end "${TRAIN_END}" \
    --validation-end "${VALIDATION_END}" \
    --end "${END_DATE}" \
    --dataset-version "${version}" \
    --output "${dataset}" \
    --allow-empty

  deck_count="$(
    node -e '
      const manifest = require(process.argv[1]);
      process.stdout.write(String(manifest.row_counts?.split_manifest || 0));
    ' "${dataset}/dataset_manifest.json"
  )"
  target_count="$(
    node -e '
      const manifest = require(process.argv[1]);
      process.stdout.write(String(manifest.row_counts?.proxy_targets || 0));
    ' "${dataset}/dataset_manifest.json"
  )"
  if ((deck_count == 0 || target_count == 0)); then
    echo "==> Skipping ${format}: no trainable rows"
    continue
  fi

  echo "==> Evaluating ${format} A11 chronologically (${deck_count} decks)"
  bazel run //:manafold -- model-train "${dataset}" \
    --model a11 \
    --output "${evaluation_results}" \
    --auto-ontology-output "${evaluation_model}/family_relations.json" \
    --epochs "${EPOCHS}" \
    --learning-rate 0.005 \
    --seed "${SEED}" \
    --embedding-dim 32 \
    --hidden-dim 64 \
    --attention-heads 4 \
    --attention-layers 2 \
    --batch-size 1024 \
    --device "${DEVICE}" \
    --prediction-output full \
    --saved-model-output "${evaluation_model}" \
    --saved-model-name a11

  echo "==> Recording ${format} chronological family evaluation"
  bazel run //:manafold -- family-eval \
    --saved-model "${evaluation_model}" \
    --dataset "${dataset}" \
    --output "${family_results}" \
    --split-name test \
    --batch-size 1024 \
    --device "${DEVICE}"

  refit_epochs="$(
    node -e '
      const result = require(process.argv[1]);
      const model = Object.values(result.models || {}).find(
        candidate => candidate.saved_model,
      );
      const selected = Number(model?.best_validation_epoch);
      process.stdout.write(String(
        Number.isInteger(selected) && selected > 0
          ? selected
          : Number(process.argv[2]),
      ));
    ' "${evaluation_results}" "${EPOCHS}"
  )"

  echo "==> Refitting ${format} A11 on all current trainable examples from ${deck_count} decks (${refit_epochs} epochs)"
  bazel run //:manafold -- model-train "${dataset}" \
    --model a11 \
    --output "${results}" \
    --auto-ontology-output "${saved_model}/family_relations.json" \
    --epochs "${refit_epochs}" \
    --learning-rate 0.005 \
    --seed "${SEED}" \
    --embedding-dim 32 \
    --hidden-dim 64 \
    --attention-heads 4 \
    --attention-layers 2 \
    --batch-size 1024 \
    --device "${DEVICE}" \
    --prediction-output summary \
    --saved-model-output "${saved_model}" \
    --saved-model-name a11 \
    --production-refit \
    --calibration-model "${evaluation_model}"
  READY_FORMATS+=("${format}")
done

if ((${#READY_FORMATS[@]} == 0)); then
  echo "No format produced a trainable model."
  exit 0
fi

if [[ ! -x "${RUNTIME}/node_modules/.bin/wrangler" ]]; then
  npm ci --prefix "${RUNTIME}"
fi

release_args=()
for format in "${READY_FORMATS[@]}"; do
  release_args+=(--format "${format}")
done

echo "==> Building ONNX Worker bundles"
node "${RUNTIME}/scripts/release-formats.mjs" build "${release_args[@]}"
echo "Built bundles under ${ROOT}/dist/onnx-formats"

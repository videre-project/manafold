#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${ROOT}/src/onnx-runtime"
CONFIG="${RUNTIME}/formats.json"
cd "${ROOT}"

END_DATE="${END_DATE:-$(date +%F)}"
END_MONTH="$(date -d "${END_DATE}" +%Y-%m-01)"
VALIDATION_END="${VALIDATION_END:-$(date -d "${END_MONTH} -1 day" +%F)}"
TRAIN_END="${TRAIN_END:-$(date -d "${END_MONTH} -1 month -1 day" +%F)}"
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
  artifact="${ROOT}/data/releases/${format}/model"
  results="${ROOT}/data/releases/${format}/training_results.json"
  family_results="${ROOT}/data/releases/${format}/family_metrics.json"
  version="${format}_${START_DATE%%-*}_${END_DATE%%-*}_v0"
  family_relations="$(
    node --input-type=module -e '
      import { readFileSync } from "node:fs";
      import { resolve } from "node:path";
      const config = JSON.parse(readFileSync(process.argv[1], "utf8"));
      const relation = config.family_relations?.[process.argv[2]];
      if (relation) process.stdout.write(resolve(process.argv[3], relation));
    ' "${CONFIG}" "${format}" "${ROOT}"
  )"
  if [[ -n "${family_relations}" && ! -f "${family_relations}" ]]; then
    echo "Missing configured family relations for ${format}: ${family_relations}" >&2
    exit 2
  fi

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

  echo "==> Training ${format} (${deck_count} decks)"
  bazel run //:manafold -- model-train "${dataset}" \
    --model a3 \
    --output "${results}" \
    --epochs "${EPOCHS}" \
    --learning-rate 0.005 \
    --deepsets-regularized-weight-decay 0.001 \
    --seed "${SEED}" \
    --embedding-dim 32 \
    --hidden-dim 64 \
    --attention-heads 4 \
    --attention-layers 2 \
    --batch-size 1024 \
    --device "${DEVICE}" \
    --prediction-output full \
    --model-artifact-output "${artifact}" \
    --model-artifact-model a3

  family_args=()
  if [[ -n "${family_relations}" ]]; then
    family_args+=(--family-relations "${family_relations}")
  fi
  echo "==> Evaluating ${format} family-backed serving policy"
  bazel run //:evaluate_family_backoff -- \
    --model-artifact "${artifact}" \
    --dataset "${dataset}" \
    --output "${family_results}" \
    --split-name test \
    --batch-size 1024 \
    --device "${DEVICE}" \
    "${family_args[@]}"
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

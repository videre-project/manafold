#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${ROOT}/src/onnx-runtime"
DIST="${ROOT}/dist/onnx-formats"
MANIFEST="${DIST}/release_manifest.json"
WRANGLER="${RUNTIME}/node_modules/.bin/wrangler"
cd "${ROOT}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "No built release found. Run ./build.sh first." >&2
  exit 1
fi
if [[ ! -x "${WRANGLER}" ]]; then
  echo "Worker dependencies are missing. Run ./build.sh first." >&2
  exit 1
fi

FORMATS=("$@")
if ((${#FORMATS[@]} == 0)); then
  mapfile -t FORMATS < <(
    node --input-type=module -e '
      import { readFileSync } from "node:fs";
      const plan = JSON.parse(readFileSync(process.argv[1], "utf8"));
      for (const entry of plan.entries || []) {
        if (entry.status === "ready") console.log(entry.format);
      }
    ' "${MANIFEST}"
  )
fi

if ((${#FORMATS[@]} == 0)); then
  echo "The built release contains no deployable formats." >&2
  exit 1
fi

for format in "${FORMATS[@]}"; do
  status="$(
    node -e '
      const plan = require(process.argv[1]);
      const entry = (plan.entries || []).find(
        candidate => candidate.format === process.argv[2],
      );
      process.stdout.write(entry?.status || "missing");
    ' "${MANIFEST}" "${format}"
  )"
  if [[ "${status}" != "ready" ]]; then
    echo "Format '${format}' is not ready in the built release (${status})." >&2
    exit 2
  fi
  bundle="${DIST}/${format}"
  if [[ ! -f "${bundle}/wrangler.toml" ]]; then
    echo "No built bundle for format: ${format}" >&2
    exit 2
  fi
  echo "==> Deploying manafold-${format}"
  "${WRANGLER}" deploy \
    --cwd "${bundle}" \
    --config wrangler.toml
done

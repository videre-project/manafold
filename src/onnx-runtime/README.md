# Manafold Cloudflare Runtime

This package builds and deploys private, format-specific Manafold Workers. Every Worker contains one model bundle and the shared operator-reduced ONNX Runtime Web build. Public routing and API policy remain in `api-services/services/videre-ml`.

## Release Inputs

[`formats.json`](formats.json) defines the active database formats, retired formats, and release directory convention:

```text
data/releases/<format>/dataset
data/releases/<format>/model
```

The dataset directory must contain `dataset_manifest.json`. The model directory must contain a complete Manafold scoring artifact. `Extended` and `Classic` are retired and never enter the release. A zero-row dataset is also skipped. A missing dataset or a nonempty dataset without its model artifact is treated as a release error.

Every exported bundle includes deterministic serving-time family backoff. Color
prefixes are relaxed for descriptive archetypes while macro-archetype
boundaries such as Azorius Control and Dimir Control remain distinct. A model
artifact may also contain `family_relations.json` with seed-free auto-ontology
proposals. The release pipeline folds its semantic `alias`, `same_family`, and
`sibling_variant` edges into the exported family map without shipping the
relation artifact itself.

## Build And Deploy

Install the JavaScript dependencies once, then run the desired release phase:

```bash
npm install
npm run formats:plan
npm run formats:build
npm run formats:dry-run
npm run formats:deploy
```

`formats:build` compiles the reduced runtime once and creates an isolated bundle under `dist/onnx-formats/<format>` for every eligible format. `formats:dry-run` also asks Wrangler to validate each bundle. `formats:deploy` publishes each bundle as `manafold-<format>`.

Limit a command to one or more formats during development:

```bash
npm run formats:build -- --format modern
npm run formats:dry-run -- --format pioneer
```

The generated Workers have no public route or `workers.dev` endpoint. `videre-ml` reaches them through Service Bindings.

## Local Development

The source runtime can still be run directly after exporting one model into `src/model` and installing the reduced runtime:

```bash
bazel run //:export_onnx -- \
  --format modern \
  --model-artifact data/releases/modern/model \
  --family-relations data/releases/modern/model/family_relations.json \
  --ranking-dataset data/releases/modern/dataset \
  --output-dir src/onnx-runtime/src/model \
  --onnx-name model.onnx.bin
bazel run //:build_onnxruntime
cd src/onnx-runtime
npm run dev
```

The private endpoint accepts the request forwarded by the public router:

```bash
curl -s 'http://127.0.0.1:8787/predict?format=modern' \
  -H 'content-type: application/json' \
  -H 'x-manafold-format: modern' \
  --data '[{"name":"Amped Raptor","quantity":4},{"name":"Guide of Souls","quantity":4}]'
```

## Runtime Notes

The Worker uses ONNX Runtime Web with WASM compiled by Wrangler. Bazel builds an operator-reduced runtime from pinned dependencies and installs the matching JavaScript loader. The loader accepts Cloudflare's precompiled `WebAssembly.Module`, and inference uses one thread.

Input preparation follows the exported model manifest. Quantity-weighted models receive raw copy counts. Hypergeometric Set Transformer models receive the normalized draw probabilities used during training. Mainboard-only artifacts discard sideboard entries before inference. Temperature-scaled raw-label probabilities are aggregated through the exported family map before top-k filtering, so related source labels do not split the served probability mass.

Each prediction includes a compact `ranking` of submitted cards that are
distinctive for that family in the model's training split. Rankings use
smoothed mainboard adoption lift, include only cards scoring at least `0.15`,
and are capped at eight cards. They are decision evidence for clients, not
causal model attributions.

Cards outside an artifact's vocabulary are skipped and reported through
`meta.unknown_card_count` and `meta.unknown_cards`. A request fails only when
no recognized cards remain.

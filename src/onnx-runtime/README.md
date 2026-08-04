# Manafold Cloudflare runtime

This package builds and deploys format-specific Cloudflare Worker services. Each Worker packages a trained model artifact alongside an operator-reduced ONNX Runtime Web build. The public routing layer and API policies are managed separately in `api-services/services/videre-ml`.

## Release inputs

The [`formats.json`](formats.json) configuration file defines active game formats, retired formats, and file path locations:

```text
data/releases/<format>/dataset
data/releases/<format>/model
```

The dataset directory requires `dataset_manifest.json`, while the model directory must contain a complete saved model along with its generated `family_relations.json` file. Retired formats like Extended and Classic are omitted from deployment. The pipeline skips empty dataset directories and halts with an error if a non-empty dataset is missing its saved model.

Exported model bundles apply serving-time family backoff automatically. The system simplifies color prefixes for descriptive archetypes while preserving distinct macro-archetypes like Azorius Control and Dimir Control. The deployment script resolves the relation file using the configured `family_relations_pattern`, merging alias, family, and sibling variant edges into the compiled serving map without deploying the raw relation file.

## Build and deploy

Install JavaScript dependencies first, then execute the deployment workflow steps:

```bash
npm install
npm run formats:plan
npm run formats:build
npm run formats:dry-run
npm run formats:deploy
```

The `formats:build` command compiles the reduced runtime binary once and creates isolated build artifacts under `dist/onnx-formats/<format>` for each format. The `formats:dry-run` step validates worker bundles using Wrangler, and `formats:deploy` publishes each Worker under the name `manafold-<format>`.

To target specific formats during development, pass the format flag:

```bash
npm run formats:build -- --format modern
npm run formats:dry-run -- --format pioneer
```

Deploys create worker services without public routes or `workers.dev` endpoints, allowing `videre-ml` to connect securely through Service Bindings.

## Local development

To run the local development server, export a model into `src/model` and compile the runtime:

```bash
bazel run //:export_onnx -- \
  --format modern \
  --saved-model data/releases/modern/model \
  --family-relations data/releases/modern/model/family_relations.json \
  --ranking-dataset data/releases/modern/dataset \
  --output-dir src/onnx-runtime/src/model \
  --onnx-name model.onnx.bin
bazel run //:build_onnxruntime
cd src/onnx-runtime
npm run dev
```

Test the local endpoint with a sample request:

```bash
curl -s 'http://127.0.0.1:8787/predict?format=modern' \
  -H 'content-type: application/json' \
  -H 'x-manafold-format: modern' \
  --data '[{"name":"Amped Raptor","quantity":4},{"name":"Guide of Souls","quantity":4}]'
```

## Runtime notes

The Worker runs ONNX Runtime Web using a single-threaded WebAssembly module managed by Wrangler. Bazel builds the operator-reduced runtime binary from pinned dependencies and installs the corresponding JavaScript loader interface.

Input formatting matches the model configuration: quantity-weighted models receive raw card counts, while Set Transformer models receive normalized draw probabilities. Mainboard-only models filter out sideboard cards prior to inference. Raw output probabilities are scaled by temperature parameters and aggregated across the family map before top-k filtering, preventing related labels from dividing probability mass.

Each prediction includes a card ranking showing up to eight distinctive mainboard cards scoring at least 0.15 in adoption lift. These rankings provide descriptive evidence for client applications rather than causal attributions.

Unrecognized cards are skipped and recorded in `meta.unknown_card_count` and `meta.unknown_cards`. Inference fails only when a deck contains no recognized cards.

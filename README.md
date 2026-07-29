# manafold

Model-assisted archetype classification for Magic: The Gathering decklists.

## Overview

Manafold is the model layer that turns MTGO decklists and events into archetype predictions, confidence scores, and deck embeddings. It consumes exports from [`mtgo-db`](https://github.com/videre-project/mtgo-db), trains event-forward classifiers over card/zone/count tokens, saves reusable scoring artifacts, and backfills deck-level predictions for downstream Videre APIs and services.

The project is designed around a stable data and scoring contract:

```text
mtgo-db export
  -> deck-token dataset
  -> event-forward model training
  -> versioned scoring artifact
  -> prediction and embedding backfill
  -> disagreement, low-confidence, and unknown-label reports
```

Upstream archetype names are treated as source labels with provenance. They provide supervision for the reference baselines, while the model outputs help identify label drift, missing labels, likely aliases, and decks whose archetype assignment is uncertain.

## Reference Baselines

Manafold compares new approaches against a small set of reference baselines that share the same dataset export, split protocol, and scoring interface. The included baselines cover three useful levels of complexity:

```text
A0   pooled-linear
A1   quantity-aware Deep Sets
A1.5 regularized quantity-aware Deep Sets
```

The `pooled-linear` model measures how far normalized card-count features go without learned embeddings. The `deepsets-quantity-weighted` model (`a1`) adds learned card, zone, and quantity representations. The regularized Deep Sets baseline (`a1.5`) adds weight decay regularization to the set representation.

These baselines are intentionally simple and inspectable. They establish the artifact format, prediction schema, uncertainty scores, and deck embedding surface used by the rest of the Manafold pipeline.

## Project Structure

```text
.
├── .vscode/settings.json
├── paper/                # Project paper and bibliography
│   ├── main.tex
│   ├── main.pdf
│   └── references.bib
├── src/manafold/
│   ├── data/             # Dataset export, schemas, and validation
│   └── models/           # Model loading, metrics, and training
├── BUILD.bazel
├── MODULE.bazel
├── pyproject.toml
├── uv.lock
├── .bazelrc
├── .bazelignore
└── .latexmkrc
```

## Setup

Manafold uses Bazel for the build and test environment. Bazel fetches Python 3.12, the `uv` binary, and the Python wheels resolved in `uv.lock`.

Install Bazel or Bazelisk, then verify the repository from its root:

```bash
bazel test //...
```

Python dependencies flow through:

```text
pyproject.toml -> uv.lock -> Bazel uv.project()
```

For local editor tooling, create or refresh a `.venv` with the Bazel-managed `uv` binary:

```bash
bazel run @uv -- sync --locked
```

When Python dependencies change, update the lockfile and rerun the tests:

```bash
bazel run @uv -- lock
bazel test //...
```

## Training

`model-train` fits the reference baselines on an event-forward split.

```bash
bazel run //:manafold -- model-train data/full
```

The default run trains:

```text
pooled-linear
a1 (deepsets-quantity-weighted)
a1.5 (deepsets-quantity-weighted-regularized)
```

The regularized Deep Sets model is configured by `--deepsets-regularized-weight-decay` and uses the same `0.005` default learning rate as the other neural baseline. Training output reports event-forward accuracy, top-k accuracy, macro-F1, confusion summaries, abstention metrics, and calibration metrics. Temperature scaling is fit on validation logits, so temperature-scaled NLL, Brier score, and ECE are reported separately from the unscaled metrics.

The supported neural candidates are:

| Model | Intended use |
| --- | --- |
| `a1.5` | A1.5 stable baseline |
| `a2++` | A2++ complete-deck classifier |
| `a3` | A3 partial and complete mainboard inference |

`a1.5`, `a2++`, and `a3` are stable training aliases for their respective neural models (`a15` and `a2pp` are also accepted). Artifacts retain their full architecture identifiers for reproducibility.

The A3 model (`a3`) uses hypergeometric copy weights, paired partial-view training, and a conservative natural/square-root-balanced sampler. Its EMA teacher and contextual predictor are training-only; saved artifacts and ONNX exports contain only the inference network.

## Scoring Artifacts

A scoring artifact is a fitted model plus the vocabularies, calibration files, and training metadata required to run that model on another export. Artifact export records the model state needed for reproducible batch scoring.

To train a single-seed A1.5 artifact:

```bash
bazel run //:manafold -- model-train data/full \
  --output data/models/modern_a15_current/training_results.json \
  --model a1.5 \
  --seed 13 \
  --max-steps 2370 \
  --batch-size 1024 \
  --model-artifact-output data/models/modern_a15_current \
  --model-artifact-model a1.5
```

Artifact export defaults to `--model-artifact-seed-policy single`. A multi-seed run can export its first fitted model with `--model-artifact-seed-policy first`.

The artifact directory contains:

```text
model.pt
model_config.json
card_vocab.parquet
label_vocab.json
zone_vocab.json
temperature.json
training_manifest.json
```

## Scoring

`model-score` applies a saved artifact to every deck in a full or incremental export. This is the batch backfill path for producing versioned predictions, probabilities, confidence signals, and deck embeddings.

```bash
bazel run //:manafold -- model-score \
  --model-artifact data/models/modern_a15_current \
  --dataset data/full \
  --output data/scored/modern_predictions.parquet
```

`model-score` writes:

```text
model_predictions.parquet
model_predictions.manifest.json
model_predictions_deck_embeddings.parquet
```

Scoring uses the vocabularies saved in the artifact. Deck tokens are remapped by `oracle_id` into the artifact card vocabulary, and unknown cards or zones stop the run so the export can be audited or the model can be refreshed.

The scorer handles labeled and unlabeled decks. When `proxy_targets.parquet` is missing or a deck has no source label, the prediction row keeps `source_label_id` and `source_label` empty while still emitting model predictions, confidence scores, and embeddings.

Prediction rows include:

```text
deck_id
event_id
event_date
format
source_label_id
source_label
top1_label_id
top1_label
top1_probability
top3_label_ids
top3_labels
top3_probabilities
temperature_scaled_probability
energy_score
msp_score
is_low_confidence
embedding_id
model_version
```

The prediction manifest separates two operational cases:

```text
source_unlabeled
  source/proxy label is unavailable; the row is classified from deck tokens.

source_unseen
  source label exists outside the artifact label vocabulary; this helps identify
  taxonomy drift, new source strings, and alias checks.
```

## ONNX Worker

`//:export_onnx` converts a saved A1.5, A2++, or selected Set Transformer artifact into a verified ONNX bundle. `//:build_onnxruntime` builds the operator-reduced ONNX Runtime Web package from dependencies pinned in Bazel.

Build datasets, A3 model artifacts, and private Worker bundles for every active
format from the repository root:

```bash
./build.sh
./deploy.sh
```

Pass formats to limit either operation:

```bash
./build.sh modern pioneer
./deploy.sh modern pioneer
```

The build dates and training runtime can be overridden with `START_DATE`,
`TRAIN_END`, `VALIDATION_END`, `END_DATE`, `EPOCHS`, `SEED`, and `DEVICE`.
Formats with no decks or proxy targets are recorded and skipped. `deploy.sh`
publishes existing bundles and never retrains or rebuilds them.

The lower-level release commands remain available:

```bash
cd src/onnx-runtime
npm run formats:plan
npm run formats:build
npm run formats:dry-run
npm run formats:deploy
```

The release registry excludes retired formats and expects each active format under `data/releases/<format>`. Empty dataset exports are skipped; a nonempty dataset without a complete model artifact stops the release. Each eligible format is staged and deployed as a separate Worker so its model and reduced runtime remain below the Cloudflare Workers Free bundle limit. Exported Workers apply the same deterministic family backoff used in evaluation and optionally fold an artifact-local `family_relations.json` into the serving map. The private runtime and full release contract live in [`src/onnx-runtime`](src/onnx-runtime).

## Candidate Reports

`alias-candidates` turns scored decks into compact reports for taxonomy review and data-quality work. It looks for source/model disagreements, source labels outside the artifact vocabulary, low-confidence predictions, and unlabeled decks that need a model-backed archetype suggestion.

```bash
bazel run //:manafold -- alias-candidates \
  --predictions data/scored/modern_predictions.parquet \
  --dataset data/full \
  --deck-embeddings data/scored/modern_predictions_deck_embeddings.parquet \
  --output data/scored/alias_candidates.json
```

This writes:

```text
alias_candidates.json
alias_weak_label_observations.jsonl
backfill_report.json
```

Prediction-backed candidates combine source/model disagreements, deck-overlap features, source labels unseen during training, low-confidence predictions, and optional deck-embedding neighbors. Source-labeled rows can become alias or sibling candidates. Unlabeled rows can become unknown-deck candidates.

`backfill_report.json` summarizes the scored export:

```text
prediction count
embedding count
source_unlabeled count
source_unseen count
low_confidence count
top predicted labels for unlabeled decks
top source-label disagreements
top low-confidence known-source decks
top source-unseen labels
artifact/model version
```

`alias_weak_label_observations.jsonl` contains machine-readable candidate relations with labels, relation type, confidence, time scope, and provenance.

## Time-Slice Evaluation

Time-slice evaluation measures how the reference baselines behave as the metagame and source labels move forward. Use `--dev-test-end` during dataset export to create a fresh-holdout split, and use `rolling-eval` to compare the same model set across multiple date windows.

Reviewed label mappings can be supplied explicitly:

```text
--taxonomy-eval
--canonical-targets
```

## Citation

If you use Manafold as software, cite the repository:

```bibtex
@software{bennett_manafold_2026,
  author = {Bennett, Cory},
  title = {{Manafold}: Model-Assisted Archetype Classification for Magic: The Gathering Decklists},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/videre-project/manafold},
  organization = {{Videre Project}},
  license = {OpenMDW-1.1}
}
```

## License

Licensed under [OpenMDW-1.1](LICENSE).

## Disclaimer

This project is not affiliated with Wizards of the Coast or Daybreak Games.

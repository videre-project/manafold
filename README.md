# Manafold

Model-assisted archetype classification for Magic: The Gathering decklists.

## Overview

Manafold classifies Magic: The Gathering decklists into archetype predictions, confidence scores, and deck vector embeddings for the Videre Project.

The service ingests deck exports from `mtgo-db` and trains event-forward classifiers over mainboard card, zone, and quantity tokens. Model outputs backfill archetype predictions and embeddings for downstream Videre applications while identifying label drift, missing annotations, name aliases, and ambiguous archetype assignments.

Scoring flows sequentially from dataset extraction through event-forward training, versioned model persistence, batch prediction backfill, and anomaly reporting. Upstream archetype names provide reference labels for baseline supervision.

## Reference baselines

Manafold benchmarks new architectures against the `pooled-linear` (`A0`), quantity-aware Deep Sets (`A1`), and regularized quantity-aware Deep Sets (`A1.5`) baselines. These models share identical dataset exports, temporal split protocols, and scoring interfaces.

The `pooled-linear` baseline measures accuracy using normalized card counts without learned embeddings. The `a1` Deep Sets model adds learned card, zone, and quantity representations, while the `a1.5` variant applies weight decay to the set representation. Together, these baselines establish the saved model format, prediction schemas, uncertainty metrics, and vector interfaces used throughout the pipeline.

## Project structure

The codebase separates datasets, label policy, and model execution into distinct module boundaries.

- `src/manafold/datasets/` manages MTGO database access, Parquet export, schema validation, and typed model inputs
- `src/manafold/taxonomy/` defines family targets, label aliases, and weak-label evidence rules
- `src/manafold/models/` handles classifier architectures, saved model storage, scoring, evaluation, and training pipelines
- `src/manafold/cli/` connects command modules into the public command-line interface

Additional top-level directories include `paper/` for project documentation and references, `notebooks/` for chronological research records, and `scripts/` for ONNX export tooling. We also provide an ONNX runtime for WebAssembly deployment on Cloudflare Workers under `src/onnx-runtime/`.

## Setup

Manafold builds and tests using Bazel, which manages Python 3.12, the `uv` executable, and package dependencies locked in `uv.lock`.

Execute repository tests from the root directory:

```bash
bazel test //...
```

Dependencies flow from `pyproject.toml` into `uv.lock` and Bazel's `uv.project()` configuration.

Sync a local Python environment for editor tooling:

```bash
bazel run @uv -- sync --locked
```

When updating dependencies, refresh the lockfile and verify the build:

```bash
bazel run @uv -- lock
bazel test //...
```

## Training

The `model-train` command fits baseline models on an event-forward split:

```bash
bazel run //:manafold -- model-train data/full
```

The default run trains the `pooled-linear`, `a1` (deepsets-quantity-weighted), and `a1.5` (deepsets-quantity-weighted-regularized) models.

The `--deepsets-regularized-weight-decay` flag sets weight decay for the regularized model, which uses a default learning rate of 0.005. Training outputs report event-forward accuracy, top-k accuracy, macro-F1, confusion summaries, abstention metrics, and probability calibration. The trainer fits temperature scaling parameters on validation logits, recording scaled negative log-likelihood, Brier scores, and expected calibration error alongside unscaled metrics.

Supported neural candidate aliases include `a1.5` for the stable baseline, `a2++` for complete-deck classification, `a3` for partial and complete mainboard inference, `a10` for card-evidence paths, and `a11` for direct projected-family classification. Saved models preserve full classifier identifiers for auditability.

The A3 model uses hypergeometric copy weights, partial-view training pairs, and a balanced card sampler. Its exponential moving average teacher network and contextual predictor run only during training, leaving only the core inference network in saved models and ONNX exports.

## A11 training and evaluation

The A11 model trains directly on canonical archetype families induced from core card overlap in the training set. Each A11 training run generates a relation graph from training rows and records a SHA-256 digest of those rows. Saved models store the generated graph and its dataset provenance, allowing evaluation to use the same target projection:

```bash
bazel run //:manafold -- model-train data/full \
  --model a11 \
  --output data/models/modern_a11_current/training_results.json \
  --seed 13 \
  --batch-size 1024 \
  --saved-model-output data/models/modern_a11_current \
  --saved-model-name a11

bazel run //:manafold -- family-eval \ --saved-model data/models/modern_a11_current \ --dataset data/full \ --output data/models/modern_a11_current/final_test_family_evaluation.json ```

The trainer saves `family_relations.json` next to the training results and copies it into the model directory. Inspect family graph induction without fitting a model:

```bash
bazel run //:manafold -- family-targets data/full \
  --output /tmp/manafold-family-relations.json
```

Pass `--partial-identity-count 5`, `10`, or `20` to `family-eval` to evaluate model accuracy on partial mainboard lists. Generated family relations remain experimental targets that require manual review and stability checks before promotion to production vocabularies.

## Scoring with a saved model

A saved model directory contains a trained PyTorch model file (`model.pt`), model configuration parameters (`model_config.json`), vocabulary mappings (`card_vocab.parquet`, `label_vocab.json`, `zone_vocab.json`), temperature calibration parameters (`temperature.json`), and a training manifest (`training_manifest.json`). These artifacts allow reproducible batch scoring across dataset exports.

Fit and save an A1.5 baseline model using a single random seed:

```bash
bazel run //:manafold -- model-train data/full \
  --output data/models/modern_a15_current/training_results.json \
  --model a1.5 \
  --seed 13 \
  --max-steps 2370 \
  --batch-size 1024 \
  --saved-model-output data/models/modern_a15_current \
  --saved-model-name a1.5
```

The trainer defaults to `--saved-model-seed-policy single`. For multi-seed runs, `--saved-model-seed-policy first` exports the first fitted model instance.

## Scoring

The `model-score` command applies a saved model to all decks in a dataset, generating predictions (`model_predictions.parquet`), manifest logs (`model_predictions.manifest.json`), and deck vector embeddings (`model_predictions_deck_embeddings.parquet`):

```bash
bazel run //:manafold -- model-score \
  --saved-model data/models/modern_a15_current \
  --dataset data/full \
  --output data/scored/modern_predictions.parquet
```

The scorer maps deck cards into the saved model vocabulary using Scryfall `oracle_id` values. If an export contains unknown cards or zones, execution halts to allow dataset auditing or model updates.

The scoring pipeline processes both labeled and unlabeled decklists. If `proxy_targets.parquet` is missing or a deck lacks a source label, the scorer leaves source label fields empty while generating model predictions, confidence scores, and vector embeddings. Prediction records include deck and event identifiers, source and top-k predicted labels with probabilities, temperature-scaled confidence scores, energy metrics, and model versions.

The prediction manifest categorizes unmapped rows into `source_unlabeled` (decks without source annotations) and `source_unseen` (source labels falling outside the model vocabulary, signaling taxonomy drift).

## ONNX worker

The `//:export_onnx` target converts saved models into ONNX bundles, while `//:build_onnxruntime` compiles an operator-reduced WebAssembly runtime using Bazel-managed dependencies.

Build dataset exports, train A11 models, and package Cloudflare Worker bundles for active game formats:

```bash
./build.sh
./deploy.sh
```

Pass format names to limit execution:

```bash
./build.sh modern pioneer
./deploy.sh modern pioneer
```

For each format, the build script fits a model on a temporal split to evaluate metrics on an untouched test month. It then trains a production model using all available data up to `END_DATE`. The temporal run supplies epoch counts and temperature calibration parameters, while the full model is packaged for deployment.

Environment variables control build parameters including `START_DATE`, `TRAIN_END`, `VALIDATION_END`, `TEST_MONTH`, `END_DATE`, `EPOCHS`, `SEED`, and `DEVICE`. Formats lacking decks or target labels are logged and skipped. The `deploy.sh` script publishes existing worker bundles without retraining models.

Low-level deployment commands in `src/onnx-runtime`:

```bash
cd src/onnx-runtime
npm run formats:plan
npm run formats:build
npm run formats:dry-run
npm run formats:deploy
```

The release pipeline excludes retired formats and expects input files under `data/releases/<format>`. If a format lacks a saved model or its `family_relations.json` file, the pipeline stops with an error. Each active format deploys as a separate Cloudflare Worker to stay within size limits. Exported Workers bundle their family relation graphs directly into the serving map.

## Candidate reports

The `alias-candidates` command analyzes scored decks to produce reports for taxonomy review, highlighting disagreements between source labels and model predictions, unseen source labels, low-confidence classifications, and unlabeled decks needing archetype assignments:

```bash
bazel run //:manafold -- alias-candidates \
  --predictions data/scored/modern_predictions.parquet \
  --dataset data/full \
  --deck-embeddings data/scored/modern_predictions_deck_embeddings.parquet \
  --output data/scored/alias_candidates.json
```

This command produces candidate suggestions (`alias_candidates.json`), weak-label relation observations (`alias_weak_label_observations.jsonl`), and a scoring summary (`backfill_report.json`). Candidate suggestions combine prediction disagreements, card co-occurrence features, unseen source labels, low-confidence scores, and optional embedding neighbors. Labeled rows generate alias or sibling candidates, while unlabeled rows produce unknown deck suggestions.

The `backfill_report.json` file records total predictions, embedding counts, unmapped label tallies, top predicted labels for unlabeled decks, and saved model metadata. The `alias_weak_label_observations.jsonl` file records structured candidate relations with confidence scores, time windows, and dataset provenance.

## Time-slice evaluation

Time-slice evaluation measures model accuracy as metagame trends and source labels change over time. Pass `--dev-test-end` during dataset export to create a fresh holdout split, or use `rolling-eval` to compare model performance across multiple date windows. Custom taxonomy mappings can also be supplied using `--taxonomy-eval` and `--canonical-targets`.

## Citation

Cite Manafold in software or research publications:

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

Manafold is an independent research project and is not affiliated with Wizards of the Coast or Daybreak Games.

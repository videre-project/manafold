import { spawnSync } from "node:child_process";
import {
  copyFile,
  mkdir,
  readFile,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const defaultWorkspace = resolve(scriptDirectory, "../../..");
const defaultConfig = resolve(scriptDirectory, "../formats.json");
const workerSource = resolve(scriptDirectory, "..");

export async function loadReleasePlan({
  configPath = defaultConfig,
  workspace = defaultWorkspace,
  selectedFormats = [],
} = {}) {
  const config = JSON.parse(await readFile(configPath, "utf8"));
  const retired = new Set(config.retired_formats || []);
  const selected = selectedFormats.length
    ? new Set(selectedFormats.map(normalizeFormat))
    : null;
  const entries = [];

  for (const rawFormat of config.formats || []) {
    const format = normalizeFormat(rawFormat);
    if (selected && !selected.has(format)) continue;
    if (retired.has(format)) {
      entries.push({ format, status: "retired" });
      continue;
    }

    const dataset = resolvePattern(config.dataset_pattern, workspace, format);
    const savedModel = resolvePattern(
      config.saved_model_pattern,
      workspace,
      format,
    );
    const datasetManifestPath = resolve(dataset, "dataset_manifest.json");
    const datasetManifest = await readJsonIfPresent(datasetManifestPath);
    if (!datasetManifest) {
      entries.push({
        format,
        status: "missing_dataset",
        dataset,
        saved_model: savedModel,
      });
      continue;
    }
    if (normalizeFormat(datasetManifest.format) !== format) {
      entries.push({
        format,
        status: "invalid_dataset",
        dataset,
        saved_model: savedModel,
        reason: `dataset manifest declares '${datasetManifest.format}'`,
      });
      continue;
    }

    const rowCounts = datasetManifest.row_counts || {};
    const deckCount = Number(rowCounts.split_manifest || 0);
    const targetCount = Number(rowCounts.proxy_targets || 0);
    if (deckCount === 0 || targetCount === 0) {
      entries.push({
        format,
        status: "no_data",
        dataset,
        saved_model: savedModel,
        deck_count: deckCount,
        target_count: targetCount,
      });
      continue;
    }

    const requiredSavedModelFiles = [
      "model.pt",
      "model_config.json",
      "card_vocab.parquet",
      "label_vocab.json",
      "zone_vocab.json",
      "temperature.json",
      "training_manifest.json",
    ];
    const missingSavedModelFiles = [];
    for (const filename of requiredSavedModelFiles) {
      if (!(await isFile(resolve(savedModel, filename)))) {
        missingSavedModelFiles.push(filename);
      }
    }
    if (missingSavedModelFiles.length) {
      entries.push({
        format,
        status: "missing_saved_model",
        dataset,
        saved_model: savedModel,
        deck_count: deckCount,
        target_count: targetCount,
        missing_saved_model_files: missingSavedModelFiles,
      });
      continue;
    }

    let trainingManifest;
    try {
      trainingManifest = JSON.parse(
        await readFile(resolve(savedModel, "training_manifest.json"), "utf8"),
      );
    } catch (error) {
      entries.push({
        format,
        status: "invalid_saved_model",
        dataset,
        saved_model: savedModel,
        reason: `cannot read training_manifest.json: ${error.message}`,
      });
      continue;
    }
    const savedModelFormats = trainingManifest.formats?.map(normalizeFormat) || [];
    const inferredSavedModelFormat = normalizeFormat(
      String(trainingManifest.dataset_version || "").split("_")[0],
    );
    if (
      (savedModelFormats.length && !savedModelFormats.includes(format))
      || (!savedModelFormats.length && inferredSavedModelFormat !== format)
    ) {
      entries.push({
        format,
        status: "invalid_saved_model",
        dataset,
        saved_model: savedModel,
        reason: "saved model belongs to another format",
      });
      continue;
    }
    if (
      trainingManifest.dataset_version
      && datasetManifest.dataset_version
      && trainingManifest.dataset_version !== datasetManifest.dataset_version
    ) {
      entries.push({
        format,
        status: "stale_saved_model",
        dataset,
        saved_model: savedModel,
        reason: (
          `saved-model dataset '${trainingManifest.dataset_version}' does not match `
          + `'${datasetManifest.dataset_version}'`
        ),
      });
      continue;
    }

    const familyMetrics = resolvePattern(
      config.family_metrics_pattern,
      workspace,
      format,
    );
    const familyMetricsPayload = await readJsonIfPresent(familyMetrics);
    if (!familyMetricsPayload) {
      entries.push({
        format,
        status: "missing_family_metrics",
        dataset,
        saved_model: savedModel,
        reason: `family-backed metrics do not exist: ${familyMetrics}`,
      });
      continue;
    }
    if (familyMetricsPayload.dataset_version !== datasetManifest.dataset_version) {
      entries.push({
        format,
        status: "stale_family_metrics",
        dataset,
        saved_model: savedModel,
        reason: (
          `family metrics dataset '${familyMetricsPayload.dataset_version}' `
          + `does not match '${datasetManifest.dataset_version}'`
        ),
      });
      continue;
    }

    const familyRelations = resolvePattern(
      config.family_relations_pattern,
      workspace,
      format,
    );
    if (!(await isFile(familyRelations))) {
      entries.push({
        format,
        status: "missing_family_relations",
        dataset,
        saved_model: savedModel,
        reason: `configured family relations do not exist: ${familyRelations}`,
      });
      continue;
    }
    entries.push({
      format,
      status: "ready",
      dataset,
      saved_model: savedModel,
      family_relations: familyRelations,
      family_metrics: familyMetrics,
      deck_count: deckCount,
      target_count: targetCount,
    });
  }

  if (selected) {
    const configured = new Set(entries.map(({ format }) => format));
    const unknown = [...selected].filter((format) => !configured.has(format));
    if (unknown.length) {
      throw new Error(`Formats are not configured: ${unknown.join(", ")}`);
    }
  }

  return {
    config: resolve(configPath),
    workspace: resolve(workspace),
    entries,
  };
}

export async function buildReleaseBundles(plan, {
  outputRoot,
  buildRuntime = true,
} = {}) {
  const failures = plan.entries.filter(({ status }) =>
    [
      "missing_dataset",
      "missing_saved_model",
      "invalid_dataset",
      "invalid_saved_model",
      "stale_saved_model",
      "missing_family_relations",
      "missing_family_metrics",
      "stale_family_metrics",
    ].includes(status));
  if (failures.length) {
    const details = failures.map(({ format, status }) => `${format}: ${status}`);
    throw new Error(`Cannot build format release:\n${details.join("\n")}`);
  }

  const ready = plan.entries.filter(({ status }) => status === "ready");
  if (!ready.length) {
    throw new Error("No format has a nonempty dataset and deployable saved model.");
  }

  const workspace = plan.workspace;
  const destination = resolve(outputRoot || resolve(workspace, "dist/onnx-formats"));
  if (buildRuntime) {
    run("bazel", ["run", "//:build_onnxruntime", "--", "--no-install-worker"], {
      cwd: workspace,
    });
  }

  for (const entry of ready) {
    const stage = resolve(destination, entry.format);
    await stageFormatWorker({ entry, stage, workspace });
    entry.output = stage;
    entry.service = `manafold-${entry.format}`;
  }
  await writeFile(
    resolve(destination, "release_manifest.json"),
    JSON.stringify(plan, null, 2) + "\n",
  );
  return plan;
}

export function publishReleaseBundles(plan, { dryRun = false } = {}) {
  for (const entry of plan.entries.filter(({ status }) => status === "ready")) {
    if (!entry.output) {
      throw new Error(`Format '${entry.format}' has not been built.`);
    }
    const args = [
      "deploy",
      "--cwd", entry.output,
      "--config", "wrangler.toml",
    ];
    if (dryRun) args.push("--dry-run");
    run(resolve(workerSource, "node_modules/.bin/wrangler"), args, {
      cwd: plan.workspace,
    });
  }
}

async function stageFormatWorker({ entry, stage, workspace }) {
  await rm(stage, { force: true, recursive: true });
  await mkdir(resolve(stage, "src/model"), { recursive: true });
  await mkdir(resolve(stage, "src/vendor"), { recursive: true });

  const exportArgs = [
    "run", "//:export_onnx", "--",
    "--format", entry.format,
    "--saved-model", entry.saved_model,
    "--ranking-dataset", entry.dataset,
    "--output-dir", resolve(stage, "src/model"),
    "--onnx-name", "model.onnx.bin",
  ];
  if (entry.family_relations) {
    exportArgs.push("--family-relations", entry.family_relations);
  }
  run("bazel", exportArgs, { cwd: workspace });

  await copyFile(
    resolve(workerSource, "src/card-ranking.js"),
    resolve(stage, "src/card-ranking.js"),
  );
  await copyFile(
    resolve(workerSource, "src/worker.js"),
    resolve(stage, "src/worker.js"),
  );
  await copyFile(
    resolve(workerSource, "src/deck-input.js"),
    resolve(stage, "src/deck-input.js"),
  );
  await copyFile(
    resolve(workerSource, "src/family-backoff.js"),
    resolve(stage, "src/family-backoff.js"),
  );
  await copyFile(
    resolve(workerSource, "src/hypergeo.js"),
    resolve(stage, "src/hypergeo.js"),
  );
  await copyFile(
    resolve(workerSource, "runtime/generated/ort.wasm.cloudflare.mjs"),
    resolve(stage, "src/vendor/ort.wasm.cloudflare.mjs"),
  );
  await copyFile(
    resolve(workerSource, "runtime/generated/ort-wasm-simd.wasm"),
    resolve(stage, "src/model/ort-wasm-simd-threaded.wasm"),
  );
  await writeFile(
    resolve(stage, "wrangler.toml"),
    wranglerConfig(entry.format),
  );
}

function wranglerConfig(format) {
  return `name = "manafold-${format}"
main = "src/worker.js"
compatibility_date = "2026-07-27"
workers_dev = false
preview_urls = false
rules = [
  { type = "Data", globs = ["**/*.bin"], fallthrough = true },
  { type = "CompiledWasm", globs = ["**/*.wasm"], fallthrough = true }
]
`;
}

function resolvePattern(pattern, workspace, format) {
  if (typeof pattern !== "string" || !pattern.includes("{format}")) {
    throw new Error("Release patterns must contain '{format}'.");
  }
  return resolve(workspace, pattern.replaceAll("{format}", format));
}

function normalizeFormat(value) {
  return String(value || "").trim().toLowerCase();
}

async function readJsonIfPresent(path) {
  try {
    return JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

async function isFile(path) {
  try {
    return (await stat(path)).isFile();
  } catch (error) {
    if (error?.code === "ENOENT") return false;
    throw error;
  }
}

function run(command, args, options) {
  const result = spawnSync(command, args, { ...options, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`Command failed (${result.status}): ${command} ${args.join(" ")}`);
  }
}

function parseArgs(argv) {
  const options = {
    mode: "plan",
    configPath: defaultConfig,
    workspace: defaultWorkspace,
    selectedFormats: [],
    outputRoot: null,
    buildRuntime: true,
  };
  const args = [...argv];
  if (args[0] && !args[0].startsWith("--")) options.mode = args.shift();
  while (args.length) {
    const option = args.shift();
    if (option === "--config") options.configPath = resolve(args.shift());
    else if (option === "--workspace") options.workspace = resolve(args.shift());
    else if (option === "--format") options.selectedFormats.push(args.shift());
    else if (option === "--output-root") options.outputRoot = resolve(args.shift());
    else if (option === "--skip-runtime-build") options.buildRuntime = false;
    else throw new Error(`Unknown argument: ${option}`);
  }
  if (!["plan", "build", "dry-run", "deploy"].includes(options.mode)) {
    throw new Error(`Unknown release mode: ${options.mode}`);
  }
  return options;
}

async function main(argv) {
  const options = parseArgs(argv);
  const plan = await loadReleasePlan(options);
  if (options.mode !== "plan") {
    await buildReleaseBundles(plan, options);
  }
  if (options.mode === "dry-run" || options.mode === "deploy") {
    publishReleaseBundles(plan, { dryRun: options.mode === "dry-run" });
  }
  console.log(JSON.stringify(plan, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main(process.argv.slice(2)).catch((error) => {
    console.error(error.message || error);
    process.exitCode = 1;
  });
}

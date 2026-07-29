import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import test from "node:test";

import { loadReleasePlan } from "../scripts/release-formats.mjs";

const artifactFiles = [
  "model.pt",
  "model_config.json",
  "card_vocab.parquet",
  "label_vocab.json",
  "zone_vocab.json",
  "temperature.json",
  "training_manifest.json",
];

test("release planning skips retired and empty formats", async () => {
  const workspace = await mkdtemp(resolve(tmpdir(), "manafold-release-"));
  try {
    const configPath = await writeConfig(workspace);
    await writeDataset(workspace, "standard", 0, 0);
    await writeDataset(workspace, "modern", 120, 118);
    await writeArtifact(workspace, "modern");
    await writeFile(
      resolve(workspace, "data/modern-relations.json"),
      JSON.stringify({ proposed_edges: [] }),
    );

    const plan = await loadReleasePlan({ configPath, workspace });
    assert.deepEqual(
      Object.fromEntries(plan.entries.map(({ format, status }) => [format, status])),
      {
        standard: "no_data",
        modern: "ready",
        extended: "retired",
        classic: "retired",
      },
    );
    assert.equal(plan.entries[1].deck_count, 120);
    assert.equal(
      plan.entries[1].family_relations,
      resolve(workspace, "data/modern-relations.json"),
    );
  } finally {
    await rm(workspace, { force: true, recursive: true });
  }
});

test("a nonempty dataset requires a complete model artifact", async () => {
  const workspace = await mkdtemp(resolve(tmpdir(), "manafold-release-"));
  try {
    const configPath = await writeConfig(workspace);
    await writeDataset(workspace, "modern", 120, 118);

    const plan = await loadReleasePlan({
      configPath,
      workspace,
      selectedFormats: ["modern"],
    });
    assert.equal(plan.entries[0].status, "missing_artifact");
    assert.deepEqual(plan.entries[0].missing_artifact_files, artifactFiles);
  } finally {
    await rm(workspace, { force: true, recursive: true });
  }
});

test("release planning rejects weights trained for another format", async () => {
  const workspace = await mkdtemp(resolve(tmpdir(), "manafold-release-"));
  try {
    const configPath = await writeConfig(workspace);
    await writeDataset(workspace, "modern", 120, 118);
    await writeArtifact(workspace, "modern");
    await writeFile(
      resolve(workspace, "data/releases/modern/model/training_manifest.json"),
      JSON.stringify({
        dataset_version: "pioneer_2024_2026_v0",
        formats: ["pioneer"],
      }),
    );

    const plan = await loadReleasePlan({
      configPath,
      workspace,
      selectedFormats: ["modern"],
    });
    assert.equal(plan.entries[0].status, "invalid_artifact");
  } finally {
    await rm(workspace, { force: true, recursive: true });
  }
});

async function writeConfig(workspace) {
  const path = resolve(workspace, "formats.json");
  await writeFile(path, JSON.stringify({
    formats: ["standard", "modern", "extended", "classic"],
    retired_formats: ["extended", "classic"],
    dataset_pattern: "data/releases/{format}/dataset",
    artifact_pattern: "data/releases/{format}/model",
    family_metrics_pattern: "data/releases/{format}/family_metrics.json",
    family_relations: {
      modern: "data/modern-relations.json",
    },
  }));
  return path;
}

async function writeDataset(workspace, format, deckCount, targetCount) {
  const directory = resolve(workspace, `data/releases/${format}/dataset`);
  await mkdir(directory, { recursive: true });
  await writeFile(resolve(directory, "dataset_manifest.json"), JSON.stringify({
    format,
    dataset_version: `${format}_2024_2026_v0`,
    row_counts: {
      split_manifest: deckCount,
      proxy_targets: targetCount,
    },
  }));
}

async function writeArtifact(workspace, format) {
  const directory = resolve(workspace, `data/releases/${format}/model`);
  await mkdir(directory, { recursive: true });
  await Promise.all(artifactFiles.map((filename) => writeFile(
    resolve(directory, filename),
    filename === "training_manifest.json"
      ? JSON.stringify({
        dataset_version: `${format}_2024_2026_v0`,
        formats: [format],
      })
      : "test",
  )));
  await writeFile(
    resolve(workspace, `data/releases/${format}/family_metrics.json`),
    JSON.stringify({
      dataset_version: `${format}_2024_2026_v0`,
      metrics: { accuracy: 0.9 },
    }),
  );
}

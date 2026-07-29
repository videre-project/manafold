import assert from "node:assert/strict";
import test from "node:test";

import {
  aggregateFamilyProbabilities,
  buildFamilyState,
} from "../src/family-backoff.js";

const labels = [
  { label_id: "red", label: "Red Example Engine" },
  { label_id: "green", label: "Green Example Engine" },
  { label_id: "azorius", label: "Azorius Control" },
  { label_id: "dimir", label: "Dimir Control" },
];
const familyVocab = {
  version: "manafold_family_backoff_v1",
  families: [
    {
      family_id: "family.example_engine",
      display_label: "Example Engine",
    },
    { family_id: "family.azorius_control", display_label: "Azorius Control" },
    { family_id: "family.dimir_control", display_label: "Dimir Control" },
  ],
  entries: [
    { label_id: "red", family_id: "family.example_engine" },
    { label_id: "green", family_id: "family.example_engine" },
    { label_id: "azorius", family_id: "family.azorius_control" },
    { label_id: "dimir", family_id: "family.dimir_control" },
  ],
};

test("family backoff conserves probability and preserves macro boundaries", () => {
  const state = buildFamilyState(labels, familyVocab);
  const probabilities = aggregateFamilyProbabilities(
    [0.25, 0.30, 0.20, 0.25],
    state,
  );
  const scores = Object.fromEntries(
    state.families.map((family, index) => [family.label, probabilities[index]]),
  );

  assert.ok(Math.abs(scores["Example Engine"] - 0.55) < 1e-12);
  assert.equal(scores["Azorius Control"], 0.20);
  assert.equal(scores["Dimir Control"], 0.25);
  assert.ok(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
});

test("family backoff requires complete label coverage", () => {
  assert.throws(
    () => buildFamilyState(labels, {
      ...familyVocab,
      entries: familyVocab.entries.slice(1),
    }),
    /does not cover label 'red'/,
  );
});

import assert from "node:assert/strict";
import test from "node:test";

import {
  buildCardRankingState,
  rankObservedCards,
} from "../src/card-ranking.js";

test("card rankings include only observed cards above the response threshold", () => {
  const cards = new Map([
    [1, { name: "Defining Card", oracle_id: "defining-card" }],
    [2, { name: "Supporting Card", oracle_id: "supporting-card" }],
    [3, { name: "Common Card", oracle_id: "common-card" }],
  ]);
  const state = buildCardRankingState({
    version: "v1",
    method: "test",
    families: {
      "family.example_engine": [
        { card_idx: 1, score: 1.0 },
        { card_idx: 2, score: 0.8 },
        { card_idx: 3, score: 0.05 },
      ],
    },
  }, cards);
  const ranking = rankObservedCards([
    { card_idx: 3, quantity: 10 },
    { card_idx: 2, quantity: 2 },
    { card_idx: 1, quantity: 4 },
  ], "family.example_engine", state);

  assert.deepEqual(ranking, [
    {
      card: "Defining Card",
      oracle_id: "defining-card",
      quantity: 4,
      score: 1.0,
    },
    {
      card: "Supporting Card",
      oracle_id: "supporting-card",
      quantity: 2,
      score: 0.8,
    },
  ]);
});

test("card rankings are empty when a bundle has no ranking artifact", () => {
  assert.deepEqual(
    rankObservedCards([{ card_idx: 1, quantity: 4 }], "family.any", null),
    [],
  );
});

import assert from "node:assert/strict";
import test from "node:test";

import { parseDeck } from "../src/deck-input.js";

function modelState() {
  return {
    cardsByName: new Map([
      ["known card", 1],
      ["second known card", 2],
    ]),
    cardsByOracleId: new Map([["oracle-known-card", 1]]),
    expectedMainboardSize: 60,
    hypergeometricDrawCount: 7,
    mainZoneIdx: 0,
    pooling: "hypergeometric",
    quantityCount: 32,
    sideZoneIdx: 1,
    tokenScope: "mainboard",
    zoneVocab: {
      main: 0,
      side: 1,
    },
  };
}

test("unknown cards are skipped when recognized cards remain", () => {
  const parsed = parseDeck([
    { name: "Known Card", quantity: 4 },
    { name: "Future Card", quantity: 2 },
    { name: "Future Card", quantity: 1 },
  ], modelState());

  assert.equal(parsed.error, undefined);
  assert.deepEqual(parsed.unknownCards, ["Future Card"]);
  assert.equal(parsed.tokens.length, 1);
  assert.equal(parsed.tokens[0].card_idx, 1);
  assert.equal(parsed.tokens[0].quantity, 4);
  assert.ok(parsed.tokens[0].quantity_weight > 1);
});

test("oracle ids can resolve cards when names are unknown", () => {
  const parsed = parseDeck([
    {
      name: "Localized Card Name",
      oracle_id: "oracle-known-card",
      quantity: 1,
    },
  ], modelState());

  assert.equal(parsed.error, undefined);
  assert.deepEqual(parsed.unknownCards, []);
  assert.equal(parsed.tokens[0].card_idx, 1);
});

test("an all-unknown request returns the skipped identities", () => {
  const parsed = parseDeck([
    "Future Card",
    "Another Future Card",
  ], modelState());

  assert.equal(parsed.status, 400);
  assert.equal(parsed.error.error, "Unknown cards");
  assert.deepEqual(
    parsed.error.unknown_cards,
    ["Future Card", "Another Future Card"],
  );
});

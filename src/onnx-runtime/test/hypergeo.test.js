import assert from "node:assert/strict";
import test from "node:test";

import { hypergeometricQuantityWeight } from "../src/hypergeo.js";

test("a singleton has unit hypergeometric weight", () => {
  assert.ok(Math.abs(hypergeometricQuantityWeight(1, 60, 7) - 1) < 1e-12);
});

test("hypergeometric weights match the Python training implementation", () => {
  assert.ok(
    Math.abs(hypergeometricQuantityWeight(2, 60, 7) - 1.8983050847457625)
      < 1e-12,
  );
  assert.ok(
    Math.abs(hypergeometricQuantityWeight(4, 60, 7) - 3.424282506382848)
      < 1e-12,
  );
});

test("weights saturate at the population and handle empty observations", () => {
  assert.equal(hypergeometricQuantityWeight(0, 60, 7), 0);
  assert.equal(hypergeometricQuantityWeight(99, 4, 7), 1);
});

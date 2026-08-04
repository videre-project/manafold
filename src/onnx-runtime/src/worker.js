import * as ort from "./vendor/ort.wasm.cloudflare.mjs";

import modelBytes from "./model/model.onnx.bin";
import ortWasmModule from "./model/ort-wasm-simd-threaded.wasm";
import cardVocab from "./model/card_vocab.json";
import familyCardRanking from "./model/family_card_ranking.json";
import familyVocab from "./model/family_vocab.json";
import labelVocab from "./model/label_vocab.json";
import workerManifest from "./model/worker_manifest.json";
import zoneVocab from "./model/zone_vocab.json";
import {
  aggregateFamilyProbabilities,
  buildFamilyState,
} from "./family-backoff.js";
import {
  buildCardRankingState,
  rankObservedCards,
} from "./card-ranking.js";
import { normalizeCardName, parseDeck } from "./deck-input.js";

const SUPPORTED_FORMAT = String(workerManifest.format || "").toLowerCase();
const SERVING_MODEL = String(workerManifest.serving?.model || "manafold");
const SERVING_VERSION = String(
  workerManifest.serving?.version || workerManifest.model_version || "",
);
const DEFAULT_TOP_K = 25;
const DEFAULT_MIN_PROBABILITY = 0.05;
const DEFAULT_EXPECTED_MAINBOARD_SIZE = 60;

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmBinary = ortWasmModule;
ort.env.logLevel = "warning";

const modelState = buildModelState();
let sessionPromise;

export default {
  async fetch(request) {
    const startedAt = performance.now();
    const url = new URL(request.url);
    const endpoint = url.pathname.replace(/\/$/, "") || "/";

    if (request.method === "GET" && ["/", "/health"].includes(endpoint)) {
      return jsonResponse({
        status: "ok",
        backend: "manafold-onnx",
        model: SERVING_MODEL,
        model_version: SERVING_VERSION,
        format: SUPPORTED_FORMAT,
        family_count: modelState.familyState.families.length,
        family_policy_version: modelState.familyState.version,
        card_ranking_version: modelState.cardRankingState?.version || null,
      });
    }

    if (request.method !== "POST" || endpoint !== "/predict") {
      return jsonResponse({
        error: "Not Found",
        message: "This private service accepts POST /predict from the Videre ML router.",
      }, 404);
    }

    const format = (
      request.headers.get("x-manafold-format")
      || url.searchParams.get("format")
      || ""
    ).toLowerCase();
    if (format !== SUPPORTED_FORMAT) {
      return jsonResponse({
        error: "Invalid format",
        message: `The 'format' parameter '${format}' is not supported by this Manafold saved model.`,
      }, 400);
    }

    let body;
    try {
      body = await request.json();
    } catch (_error) {
      return jsonResponse({
        error: "Invalid JSON",
        message: "The request body must be a valid JSON array.",
      }, 400);
    }
    if (!Array.isArray(body)) {
      return jsonResponse({
        error: "Invalid JSON",
        message: "The request body must be a valid JSON array.",
      }, 400);
    }

    const parsed = parseDeck(body, modelState);
    if (parsed.error) {
      return jsonResponse(parsed.error, parsed.status || 400);
    }

    const topK = boundedInteger(url.searchParams.get("top"), DEFAULT_TOP_K, 1, 50);
    const minProbability = boundedFloat(
      url.searchParams.get("min_prob"),
      DEFAULT_MIN_PROBABILITY,
      0,
      1,
    );

    let logits;
    try {
      const session = await getSession();
      logits = await runModel(session, parsed.tokens, workerManifest.onnx.input_names);
    } catch (error) {
      return jsonResponse({
        error: "Inference error",
        message: error instanceof Error ? error.message : String(error),
      }, 500);
    }
    const rawProbabilities = softmaxTemperature(logits, workerManifest.temperature);
    const probabilities = aggregateFamilyProbabilities(
      rawProbabilities,
      modelState.familyState,
    );
    const ranked = rankLabels(
      probabilities,
      modelState.familyState.families,
      topK,
      minProbability,
    );
    const rankingConfig = workerManifest.card_ranking || {};
    const execMs = performance.now() - startedAt;

    return jsonResponse({
      meta: {
        backend: "manafold-onnx",
        model: SERVING_MODEL,
        model_version: SERVING_VERSION,
        database: null,
        exec_ms: Number(execMs.toFixed(3)),
        token_count: parsed.tokens.length,
        unknown_card_count: parsed.unknownCards.length,
        unknown_cards: parsed.unknownCards.slice(0, 25),
      },
      data: Object.fromEntries(ranked.map((row) => [row.label, round(row.probability)])),
      predictions: ranked.map((row) => ({
        label_id: row.label_id,
        label: row.label,
        probability: round(row.probability),
        ranking: rankObservedCards(
          parsed.tokens,
          row.label_id,
          modelState.cardRankingState,
          {
            minScore: Number(rankingConfig.response_min_score || 0.15),
            maxCards: Number(rankingConfig.response_max_cards || 8),
          },
        ),
      })),
    });
  },
};

function buildModelState() {
  if (!SUPPORTED_FORMAT) {
    throw new Error("worker_manifest.json does not declare a model format.");
  }
  const cardEntries = Array.isArray(cardVocab.entries) ? cardVocab.entries : cardVocab.cards;
  const cardsByName = new Map();
  const cardsByOracleId = new Map();
  const cardsByIndex = new Map();
  for (const card of cardEntries || []) {
    cardsByIndex.set(Number(card.card_idx), {
      name: String(card.primary_name),
      oracle_id: String(card.oracle_id),
    });
    if (typeof card.primary_name === "string") {
      cardsByName.set(normalizeCardName(card.primary_name), Number(card.card_idx));
    }
    if (typeof card.oracle_id === "string") {
      cardsByOracleId.set(card.oracle_id, Number(card.card_idx));
    }
  }

  const labels = (labelVocab.entries || []).map((entry, index) => ({
    label_id: String(entry.label_id),
    label: String(entry.display_label || entry.label_id),
    index,
  }));
  if (labels.length !== workerManifest.model_config.label_count) {
    throw new Error("label_vocab.json does not match worker_manifest.json label_count.");
  }
  const familyState = buildFamilyState(labels, familyVocab);
  const cardRankingState = buildCardRankingState(
    familyCardRanking,
    cardsByIndex,
  );

  return {
    cardsByName,
    cardsByOracleId,
    cardsByIndex,
    cardRankingState,
    labels,
    familyState,
    mainZoneIdx: zoneIndex("main"),
    sideZoneIdx: zoneIndex("side"),
    quantityCount: Number(workerManifest.model_config.quantity_count),
    pooling: String(workerManifest.model_config.pooling || "quantity-weighted"),
    tokenScope: String(workerManifest.model_config.token_scope || "all"),
    hypergeometricDrawCount: Number(
      workerManifest.model_config.hypergeometric_draw_count || 7,
    ),
    expectedMainboardSize: Number(
      workerManifest.model_config.expected_mainboard_size
      || DEFAULT_EXPECTED_MAINBOARD_SIZE,
    ),
    zoneVocab,
  };
}

function zoneIndex(zone) {
  if (!(zone in zoneVocab)) {
    throw new Error(`Missing zone '${zone}' in zone_vocab.json.`);
  }
  return Number(zoneVocab[zone]);
}

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    }).catch((error) => {
      sessionPromise = undefined;
      throw error;
    });
  }
  return sessionPromise;
}

async function runModel(session, tokens, inputNames) {
  const tokenCount = tokens.length;
  const cards = new BigInt64Array(tokenCount);
  const zones = new BigInt64Array(tokenCount);
  const quantities = new BigInt64Array(tokenCount);
  const quantityWeights = new Float32Array(tokenCount);
  const deckIdx = new BigInt64Array(tokenCount);

  for (let i = 0; i < tokenCount; i += 1) {
    const token = tokens[i];
    cards[i] = BigInt(token.card_idx);
    zones[i] = BigInt(token.zone_idx);
    quantities[i] = BigInt(token.quantity_idx);
    quantityWeights[i] = token.quantity_weight;
    deckIdx[i] = 0n;
  }

  const values = {
    cards: new ort.Tensor("int64", cards, [tokenCount]),
    zones: new ort.Tensor("int64", zones, [tokenCount]),
    quantities: new ort.Tensor("int64", quantities, [tokenCount]),
    quantity_weights: new ort.Tensor("float32", quantityWeights, [tokenCount]),
    deck_idx: new ort.Tensor("int64", deckIdx, [tokenCount]),
    deck_count: new ort.Tensor("int64", new BigInt64Array([1n]), [1]),
    package_features: new ort.Tensor("float32", new Float32Array(0), [1, 0]),
  };
  const feeds = {};
  for (const name of inputNames) {
    feeds[name] = values[name];
  }
  const outputs = await session.run(feeds);
  const logits = outputs.logits || outputs[Object.keys(outputs)[0]];
  return Array.from(logits.data);
}

function softmaxTemperature(logits, temperature) {
  const parsed = Number(temperature);
  const scale = Math.max(Number.isFinite(parsed) && parsed > 0 ? parsed : 1.0, 1e-12);
  let maxLogit = -Infinity;
  for (const logit of logits) {
    maxLogit = Math.max(maxLogit, logit / scale);
  }
  const exps = logits.map((logit) => Math.exp(logit / scale - maxLogit));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function rankLabels(probabilities, labels, topK, minProbability) {
  return probabilities
    .map((probability, index) => ({
      label_id: labels[index]?.label_id || String(index),
      label: labels[index]?.label || String(index),
      probability,
    }))
    .sort((left, right) => right.probability - left.probability)
    .filter((row) => row.probability >= minProbability)
    .slice(0, topK);
}

function boundedInteger(value, fallback, min, max) {
  if (value === null || value === "") {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function boundedFloat(value, fallback, min, max) {
  if (value === null || value === "") {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function round(value) {
  return Math.round(value * 100000000) / 100000000;
}

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
    },
  });
}

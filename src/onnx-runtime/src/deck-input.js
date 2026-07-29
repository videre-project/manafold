import { hypergeometricQuantityWeight } from "./hypergeo.js";

const MAIN_ZONE_NAMES = new Set(["main", "mainboard", "deck"]);
const SIDE_ZONE_NAMES = new Set(["side", "sideboard"]);

export function parseDeck(body, state) {
  if (body.length === 0) {
    return {
      status: 400,
      error: {
        error: "Invalid JSON",
        message: "The request body must contain at least one card.",
      },
    };
  }

  const tokensByKey = new Map();
  const unknownCards = [];
  try {
    if (body.every((item) => typeof item === "string")) {
      for (const name of body) {
        addToken(tokensByKey, state, { name, quantity: 1, zone: "main" }, unknownCards);
      }
    } else if (body.every((item) => item && typeof item === "object" && !Array.isArray(item))) {
      for (const item of body) {
        addToken(tokensByKey, state, item, unknownCards);
      }
    } else {
      throw new Error("Mixed or unsupported array element types.");
    }
  } catch (error) {
    return {
      status: 400,
      error: {
        error: "Invalid JSON",
        message: error.message,
      },
    };
  }

  const uniqueUnknownCards = [...new Set(unknownCards)];
  const tokens = Array.from(tokensByKey.values()).filter(
    (token) => token.quantity > 0
      && (state.tokenScope !== "mainboard" || token.zone_idx === state.mainZoneIdx),
  );
  if (!tokens.length) {
    return {
      status: 400,
      error: {
        error: uniqueUnknownCards.length ? "Unknown cards" : "Invalid JSON",
        message: uniqueUnknownCards.length
          ? "The request must contain at least one card recognized by this Manafold artifact."
          : "The request body must contain at least one recognized card.",
        unknown_cards: uniqueUnknownCards.slice(0, 25),
      },
    };
  }

  const observedMainboardSize = tokens.reduce(
    (total, token) => total + (
      token.zone_idx === state.mainZoneIdx ? token.quantity : 0
    ),
    0,
  );
  const populationSize = Math.max(
    state.expectedMainboardSize,
    observedMainboardSize,
  );
  for (const token of tokens) {
    token.quantity_weight = state.pooling === "hypergeometric"
      ? hypergeometricQuantityWeight(
        token.quantity,
        populationSize,
        state.hypergeometricDrawCount,
      )
      : Number(token.quantity);
  }
  return {
    tokens,
    unknownCards: uniqueUnknownCards,
  };
}

function addToken(tokensByKey, state, item, unknownCards) {
  const name = item.name;
  const oracleId = item.oracle_id;
  if (typeof name !== "string" && typeof oracleId !== "string") {
    throw new Error("Each card object must include a string 'name' or 'oracle_id'.");
  }
  const quantity = parseQuantity(item.quantity);
  if (quantity <= 0) {
    return;
  }
  const zoneIdx = parseZone(item.zone, state);
  let cardIdx = null;
  if (typeof oracleId === "string") {
    cardIdx = state.cardsByOracleId.get(oracleId) ?? null;
  }
  if (cardIdx === null && typeof name === "string") {
    cardIdx = state.cardsByName.get(normalizeCardName(name)) ?? null;
  }
  if (cardIdx === null) {
    unknownCards.push(typeof name === "string" ? name : oracleId);
    return;
  }

  const key = `${cardIdx}:${zoneIdx}`;
  const existing = tokensByKey.get(key) || {
    card_idx: cardIdx,
    zone_idx: zoneIdx,
    quantity: 0,
  };
  existing.quantity += quantity;
  existing.quantity_idx = Math.min(existing.quantity, state.quantityCount - 1);
  tokensByKey.set(key, existing);
}

function parseQuantity(value) {
  if (value === undefined || value === null || value === "") {
    return 1;
  }
  const parsed = typeof value === "string" ? Number(value) : value;
  if (!Number.isInteger(parsed)) {
    throw new Error("quantity must be an integer.");
  }
  return Math.max(0, parsed);
}

function parseZone(value, state) {
  if (value === undefined || value === null || value === "") {
    return state.mainZoneIdx;
  }
  if (typeof value !== "string") {
    throw new Error("zone must be a string when provided.");
  }
  const normalized = value.trim().toLowerCase();
  if (MAIN_ZONE_NAMES.has(normalized)) {
    return state.mainZoneIdx;
  }
  if (SIDE_ZONE_NAMES.has(normalized)) {
    return state.sideZoneIdx;
  }
  if (normalized in state.zoneVocab) {
    return Number(state.zoneVocab[normalized]);
  }
  throw new Error(`Unsupported zone '${value}'.`);
}

export function normalizeCardName(name) {
  return name.trim().replace(/\s+/g, " ").toLowerCase();
}

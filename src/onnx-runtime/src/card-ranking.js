export function buildCardRankingState(payload, cardsByIndex) {
  if (!payload || typeof payload !== "object") {
    return null;
  }
  const rankings = new Map();
  for (const [familyId, rows] of Object.entries(payload.families || {})) {
    const byCard = new Map();
    for (const row of rows || []) {
      const cardIdx = Number(row.card_idx);
      const card = cardsByIndex.get(cardIdx);
      const score = Number(row.score);
      if (card && Number.isFinite(score) && score > 0) {
        byCard.set(cardIdx, { card, score });
      }
    }
    rankings.set(familyId, byCard);
  }
  return {
    version: String(payload.version || ""),
    method: String(payload.method || ""),
    rankings,
  };
}

export function rankObservedCards(
  tokens,
  familyId,
  state,
  {
    minScore = 0.15,
    maxCards = 8,
  } = {},
) {
  if (!state || maxCards <= 0) {
    return [];
  }
  const familyRanking = state.rankings.get(familyId);
  if (!familyRanking) {
    return [];
  }
  const quantities = new Map();
  for (const token of tokens) {
    quantities.set(
      token.card_idx,
      (quantities.get(token.card_idx) || 0) + token.quantity,
    );
  }
  return [...quantities]
    .map(([cardIdx, quantity]) => {
      const evidence = familyRanking.get(cardIdx);
      if (!evidence || evidence.score < minScore) {
        return null;
      }
      return {
        card: evidence.card.name,
        oracle_id: evidence.card.oracle_id,
        quantity,
        score: evidence.score,
      };
    })
    .filter(Boolean)
    .sort((left, right) =>
      right.score - left.score
      || right.quantity - left.quantity
      || left.card.localeCompare(right.card))
    .slice(0, maxCards);
}

export function buildFamilyState(labels, familyVocab) {
  const families = (familyVocab.families || []).map((entry) => ({
    label_id: String(entry.family_id),
    label: String(entry.display_label),
  }));
  const familyIndexById = new Map(
    families.map((family, index) => [family.label_id, index]),
  );
  const familyIdByLabelId = new Map(
    (familyVocab.entries || []).map((entry) => [
      String(entry.label_id),
      String(entry.family_id),
    ]),
  );
  const familyIndexes = labels.map((label) => {
    const familyId = familyIdByLabelId.get(label.label_id);
    const familyIndex = familyIndexById.get(familyId);
    if (familyId === undefined || familyIndex === undefined) {
      throw new Error(`Family vocabulary does not cover label '${label.label_id}'.`);
    }
    return familyIndex;
  });
  if (familyIdByLabelId.size !== labels.length) {
    throw new Error("Family vocabulary label count does not match label vocabulary.");
  }
  return {
    families,
    familyIndexes,
    version: String(familyVocab.version || ""),
  };
}

export function aggregateFamilyProbabilities(probabilities, state) {
  if (probabilities.length !== state.familyIndexes.length) {
    throw new Error("Probability count does not match the family vocabulary.");
  }
  const aggregated = new Array(state.families.length).fill(0);
  for (let index = 0; index < probabilities.length; index += 1) {
    aggregated[state.familyIndexes[index]] += probabilities[index];
  }
  return aggregated;
}

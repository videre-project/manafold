export function hypergeometricQuantityWeight(
  quantity,
  populationSize,
  drawCount = 7,
) {
  const population = Math.max(0, Math.trunc(populationSize));
  const copies = Math.max(0, Math.min(Math.trunc(quantity), population));
  if (copies === 0 || population === 0 || drawCount <= 0) {
    return 0;
  }

  const draws = Math.min(Math.trunc(drawCount), population);
  let missProbability = 1;
  if (population - copies < draws) {
    missProbability = 0;
  } else {
    for (let draw = 0; draw < draws; draw += 1) {
      missProbability *= (population - copies - draw) / (population - draw);
    }
  }
  return (1 - missProbability) / (draws / population);
}

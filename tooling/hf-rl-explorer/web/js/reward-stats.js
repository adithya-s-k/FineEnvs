// A reward's scale belongs to its environment. Retain negative and unbounded scores.
export function rewardDistribution(values, bins = 10) {
  const xs = values.filter(Number.isFinite);
  const min = xs.reduce((a, x) => Math.min(a, x), 0);
  const max = xs.reduce((a, x) => Math.max(a, x), 1);
  const hist = Array(bins).fill(0);
  for (const x of xs) hist[Math.max(0, Math.min(bins - 1, Math.floor((x - min) / (max - min) * bins)))]++;
  return { hist, min, max, width: (max - min) / bins };
}

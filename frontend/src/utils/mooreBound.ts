export const mooreBound = (k: number, g: number): number => {
  if (k < 2 || g < 3) {
    throw new Error("k must be >= 2 and g must be >= 3");
  }

  if (g % 2 === 1) {
    let sum = 0;
    for (let i = 0; i <= (g - 3) / 2; i += 1) {
      sum += (k - 1) ** i;
    }
    return 1 + k * sum;
  }

  let sum = 0;
  for (let i = 0; i < g / 2; i += 1) {
    sum += (k - 1) ** i;
  }
  return 2 * sum;
};

export const resolveMooreBoundLimit = (): number => {
  const raw = Number(import.meta.env.VITE_MOORE_BOUND_LIMIT ?? 120);
  if (!Number.isFinite(raw) || raw <= 0) {
    return 120;
  }
  return Math.floor(raw);
};

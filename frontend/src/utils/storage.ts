export const readStored = <T>(key: string, fallback: T): T => {
  const raw = window.localStorage.getItem(key);
  if (!raw) return fallback;

  try {
    return { ...fallback, ...(JSON.parse(raw) as T) };
  } catch {
    return fallback;
  }
};

export const writeStored = <T>(key: string, value: T): void => {
  window.localStorage.setItem(key, JSON.stringify(value));
};

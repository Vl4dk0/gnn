import type { ApiConfig, RuntimeConfig } from "../types/app";

const trimTrailingSlash = (value: string) => value.trim().replace(/\/+$/, "");

export const getRuntimeConfig = (): RuntimeConfig => {
  const fromScript = window.__APP_CONFIG__?.apiBaseUrl ?? "";
  const fromStorage = window.localStorage.getItem("apiBaseUrl") ?? "";
  const base = trimTrailingSlash(fromScript || fromStorage || window.location.origin);

  return {
    apiBaseUrl: base,
    degreeUrl: `${base}/api/degree`,
    minCycleUrl: `${base}/api/min_cycle`,
    cageUrl: `${base}/api/cage`
  };
};

export const fetchAppConfig = async (): Promise<ApiConfig> => {
  const runtime = getRuntimeConfig();
  const response = await fetch(`${runtime.apiBaseUrl}/api/config`);

  if (!response.ok) {
    throw new Error(`Failed to load app config (${response.status})`);
  }

  return (await response.json()) as ApiConfig;
};

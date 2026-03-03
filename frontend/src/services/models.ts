import { fetchJson } from "../api/apiClient";
import { getRuntimeConfig } from "../api/config";
import type { ModelsResponse } from "../types/api";

export type PredictionTask = "degree" | "min_cycle";

const resolveTaskUrl = (task: PredictionTask) => {
  const runtime = getRuntimeConfig();
  return task === "degree" ? runtime.degreeUrl : runtime.minCycleUrl;
};

export const fetchModels = async (task: PredictionTask): Promise<ModelsResponse> => {
  return fetchJson<ModelsResponse>(`${resolveTaskUrl(task)}/models`);
};

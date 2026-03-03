import { fetchJson } from "../api/apiClient";
import { getRuntimeConfig } from "../api/config";
import type { AnalyzeResponse, DegreeMinCycleSettings, GenerateGraphResponse } from "../types/api";
import type { PredictionTask } from "./models";

const resolveTaskUrl = (task: PredictionTask) => {
  const runtime = getRuntimeConfig();
  return task === "degree" ? runtime.degreeUrl : runtime.minCycleUrl;
};

export const generateGraphForTask = async (
  task: PredictionTask,
  settings: DegreeMinCycleSettings
): Promise<GenerateGraphResponse> => {
  return fetchJson<GenerateGraphResponse>(`${resolveTaskUrl(task)}/generate`, {
    method: "POST",
    body: settings
  });
};

export const analyzeGraphForTask = async (
  task: PredictionTask,
  graph: string,
  modelId: string | null
): Promise<AnalyzeResponse> => {
  return fetchJson<AnalyzeResponse>(`${resolveTaskUrl(task)}/analyze`, {
    method: "POST",
    body: {
      graph,
      ...(modelId ? { model_id: modelId } : {})
    }
  });
};

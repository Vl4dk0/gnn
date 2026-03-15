import { fetchJson } from "../api/apiClient";
import { getRuntimeConfig } from "../api/config";
import type {
  CageGenerateResponse,
  CageStatusResponse,
  GeneratorType,
  ModelsResponse
} from "../types/api";

export const startCageGeneration = async (
  k: number,
  g: number,
  generator: GeneratorType,
  modelId?: string | null
): Promise<CageGenerateResponse> => {
  const runtime = getRuntimeConfig();
  return fetchJson<CageGenerateResponse>(`${runtime.cageUrl}/generate`, {
    method: "POST",
    body: { k, g, generator, ...(modelId ? { model_id: modelId } : {}) }
  });
};

export const fetchCageModels = async (): Promise<ModelsResponse> => {
  const runtime = getRuntimeConfig();
  return fetchJson<ModelsResponse>(`${runtime.cageUrl}/models`);
};

export const fetchCageStatus = async (sessionId: string): Promise<CageStatusResponse> => {
  const runtime = getRuntimeConfig();
  return fetchJson<CageStatusResponse>(`${runtime.cageUrl}/status/${sessionId}`);
};

export const stopCageGeneration = async (sessionId: string): Promise<void> => {
  const runtime = getRuntimeConfig();
  await fetchJson<{ message: string }>(`${runtime.cageUrl}/stop/${sessionId}`, {
    method: "POST"
  });
};

import { fetchJson } from "../api/apiClient";
import { getRuntimeConfig } from "../api/config";
import type {
  CageExportFormat,
  CageExportResponse,
  CageGenerateResponse,
  CageExecutionMode,
  CageStatusResponse,
  GeneratorType,
  ModelsResponse
} from "../types/api";

export const startCageGeneration = async (
  k: number,
  g: number,
  generator: GeneratorType,
  mode: CageExecutionMode,
  modelId?: string | null
): Promise<CageGenerateResponse> => {
  const runtime = getRuntimeConfig();
  return fetchJson<CageGenerateResponse>(`${runtime.cageUrl}/generate`, {
    method: "POST",
    body: { k, g, generator, mode, ...(modelId ? { model_id: modelId } : {}) }
  });
};

export const fetchCageModels = async (): Promise<ModelsResponse> => {
  const runtime = getRuntimeConfig();
  return fetchJson<ModelsResponse>(`${runtime.cageUrl}/models`);
};

export const fetchCageVoltageGirthModels = async (): Promise<ModelsResponse> => {
  const runtime = getRuntimeConfig();
  return fetchJson<ModelsResponse>(`${runtime.cageUrl}/voltage-girth-models`);
};

export const fetchCageStatus = async (sessionId: string): Promise<CageStatusResponse> => {
  const runtime = getRuntimeConfig();
  return fetchJson<CageStatusResponse>(`${runtime.cageUrl}/status/${sessionId}`);
};

export const stepCageGeneration = async (
  sessionId: string,
  steps: number
): Promise<CageStatusResponse> => {
  const runtime = getRuntimeConfig();
  return fetchJson<CageStatusResponse>(`${runtime.cageUrl}/step/${sessionId}`, {
    method: "POST",
    body: { steps }
  });
};

export const stopCageGeneration = async (sessionId: string): Promise<void> => {
  const runtime = getRuntimeConfig();
  await fetchJson<{ message: string }>(`${runtime.cageUrl}/stop/${sessionId}`, {
    method: "POST"
  });
};

export const exportCageGraph = async (
  edgeList: string,
  format: CageExportFormat,
  k?: number,
  g?: number
): Promise<CageExportResponse> => {
  const runtime = getRuntimeConfig();
  return fetchJson<CageExportResponse>(`${runtime.cageUrl}/export`, {
    method: "POST",
    body: { edge_list: edgeList, format, ...(k !== undefined ? { k } : {}), ...(g !== undefined ? { g } : {}) }
  });
};

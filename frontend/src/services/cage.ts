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

export const importGraph = async (
  content: string,
  format: CageExportFormat
): Promise<{ edge_list: string }> => {
  const runtime = getRuntimeConfig();
  return fetchJson<{ edge_list: string }>(`${runtime.cageUrl}/import`, {
    method: "POST",
    body: { content, format }
  });
};

export const importGraphFromFile = async (file: File): Promise<string> => {
  const name = file.name;
  const dotIndex = name.lastIndexOf(".");
  const suffix = dotIndex >= 0 ? name.slice(dotIndex).toLowerCase() : "";
  let format: CageExportFormat;

  if (suffix === ".g6") {
    format = "g6";
  } else if (suffix === ".txt") {
    format = "adjacency";
  } else {
    throw new Error("Unsupported file type. Use a .g6 or .txt file.");
  }

  const content = await file.text();
  const result = await importGraph(content, format);
  return result.edge_list;
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

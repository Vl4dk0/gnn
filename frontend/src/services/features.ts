import { fetchAppConfig } from "../api/config";
import type { ApiConfig } from "../types/app";

export const fetchFeatureConfig = async (): Promise<ApiConfig> => {
  return fetchAppConfig();
};

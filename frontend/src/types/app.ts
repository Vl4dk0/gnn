export type FeatureKey = "degree" | "min_cycle" | "cage";

export interface FeatureAvailability {
  active: boolean;
}

export interface ApiConfig {
  apiBaseUrl: string;
  features: Record<FeatureKey, FeatureAvailability>;
}

export interface RuntimeConfig {
  apiBaseUrl: string;
  degreeUrl: string;
  minCycleUrl: string;
  cageUrl: string;
}

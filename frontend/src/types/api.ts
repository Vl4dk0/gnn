export interface ApiError {
  error: string;
}

export interface ModelMetrics {
  accuracy?: number;
  mae?: number;
  mse?: number;
  avg_reward?: number;
  fps?: number;
}

export interface ModelTrainingInfo {
  model_type?: string;
}

export interface ModelInfo {
  model_id: string;
  model_type: string;
  metrics?: ModelMetrics;
  created_at?: string;
  training?: ModelTrainingInfo;
}

export interface ModelsResponse {
  models: ModelInfo[];
  default?: string;
}

export interface Prediction {
  node_id: number;
  true: number;
  predicted: number;
}

export interface AnalyzeResponse {
  predictions: Prediction[];
  model_id: string;
}

export interface GenerateGraphResponse {
  graph: string;
}

export interface DegreeMinCycleSettings {
  minNodes: number;
  maxNodes: number;
  minProb: number;
  maxProb: number;
  allowSelfLoops: boolean;
  enablePhysics: boolean;
  modelId: string | null;
}

export type GeneratorType = "randomwalk" | "bruteforce" | "astar" | "rl";

export interface CageSettings {
  generatorType: GeneratorType;
  modelId: string | null;
  pollingInterval: number;
  enablePhysics: boolean;
}

export interface CageGenerateResponse {
  session_id: string;
  status: string;
  k: number;
  g: number;
  moore_bound: number;
  upper_bound: number;
}

export interface CageStatusResponse {
  session_id: string;
  k: number;
  g: number;
  step_count: number;
  num_nodes: number;
  num_edges: number;
  girth: number | null;
  is_k_regular: boolean;
  is_complete: boolean;
  success: boolean;
  stopped: boolean;
  current_graph: string;
  moore_bound: number;
  elapsed_time: number;
}

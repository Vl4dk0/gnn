export type CageExportFormat = "g6" | "adjacency";

export interface CageExportResponse {
  filename: string;
  content: string;
  format: CageExportFormat;
}

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

export type GeneratorType =
  | "randomwalk"
  | "bruteforce"
  | "astar"
  | "rl"
  | "voltage"
  | "voltage_rl";
export type CageExecutionMode = "async" | "stepped";

export interface CageStepEvent {
  step: number;
  action: string;
  u: number | null;
  v: number | null;
  accepted: boolean;
  reward: number;
  done: boolean;
  done_reason: string | null;
}

export interface CageSettings {
  generatorType: GeneratorType;
  executionMode: CageExecutionMode;
  modelId: string | null;
  pollingInterval: number;
  stepsPerTick: number;
  autoStepInterval: number;
  enablePhysics: boolean;
}

export interface CageGenerateResponse {
  session_id: string;
  status: string;
  mode: CageExecutionMode;
  generator: GeneratorType;
  k: number;
  g: number;
  moore_bound: number;
  upper_bound: number;
}

export interface CageStatusResponse {
  session_id: string;
  mode: CageExecutionMode;
  generator: GeneratorType;
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
  timed_out: boolean;
  current_graph: string;
  previous_graph?: string;
  moore_bound: number;
  elapsed_time: number;
  last_event: CageStepEvent | null;
  events: CageStepEvent[];
}

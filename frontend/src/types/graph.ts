export interface GraphPrediction {
  nodeId: number;
  predicted: number;
  actual: number;
}

export interface GraphNodePredictionView {
  predicted: number;
  actual: number;
  isWrong: boolean;
}

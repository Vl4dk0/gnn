import { useEffect, useMemo, useState } from "react";

import { fetchModels } from "../services/models";
import type { ModelInfo } from "../types/api";

interface NormalizedModel {
  modelId: string;
  accuracy: number | null;
  mae: number | null;
  mse: number | null;
  createdAt: string | null;
}

const normalize = (models: ModelInfo[]): NormalizedModel[] => {
  return models
    .map((model) => ({
      modelId: model.model_id,
      accuracy: Number.isFinite(model.metrics?.accuracy) ? Number(model.metrics?.accuracy) : null,
      mae: Number.isFinite(model.metrics?.mae) ? Number(model.metrics?.mae) : null,
      mse: Number.isFinite(model.metrics?.mse) ? Number(model.metrics?.mse) : null,
      createdAt: typeof model.created_at === "string" ? model.created_at : null
    }))
    .filter((model) => model.modelId);
};

export const useDocsMetrics = () => {
  const [degreeModels, setDegreeModels] = useState<NormalizedModel[] | null>(null);
  const [cycleModels, setCycleModels] = useState<NormalizedModel[] | null>(null);

  useEffect(() => {
    let active = true;

    Promise.allSettled([fetchModels("degree"), fetchModels("min_cycle")]).then(
      ([degree, cycle]) => {
        if (!active) return;

        setDegreeModels(degree.status === "fulfilled" ? normalize(degree.value.models) : null);
        setCycleModels(cycle.status === "fulfilled" ? normalize(cycle.value.models) : null);
      }
    );

    return () => {
      active = false;
    };
  }, []);

  const assessmentRows = useMemo(() => {
    const degreeMap = new Map<string, NormalizedModel>();
    const cycleMap = new Map<string, NormalizedModel>();

    (degreeModels ?? []).forEach((model) => {
      degreeMap.set(model.modelId, model);
    });

    (cycleModels ?? []).forEach((model) => {
      cycleMap.set(model.modelId, model);
    });

    const ids = Array.from(new Set([...degreeMap.keys(), ...cycleMap.keys()])).sort((a, b) =>
      a.localeCompare(b)
    );

    return ids.map((id) => ({
      modelId: id,
      degreeAccuracy: degreeMap.get(id)?.accuracy ?? null,
      minCycleAccuracy: cycleMap.get(id)?.accuracy ?? null
    }));
  }, [degreeModels, cycleModels]);

  return {
    degreeModels,
    cycleModels,
    assessmentRows
  };
};

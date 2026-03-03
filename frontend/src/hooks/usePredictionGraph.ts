import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { fetchModels } from "../services/models";
import { analyzeGraphForTask, generateGraphForTask } from "../services/prediction";
import type { DegreeMinCycleSettings } from "../types/api";
import { normalizeEdgeList } from "../utils/graphFormat";
import { readStored, writeStored } from "../utils/storage";
import type { PredictionTask } from "../services/models";
import type { GraphPrediction } from "../types/graph";
import { InteractiveGraphEditor } from "../graph/InteractiveGraphEditor";

const DEFAULT_SETTINGS: DegreeMinCycleSettings = {
  minNodes: 5,
  maxNodes: 12,
  minProb: 0.15,
  maxProb: 0.6,
  allowSelfLoops: true,
  enablePhysics: false,
  modelId: null
};

const STORAGE_KEY = "graphSettings";

interface ModelOption {
  value: string;
  label: string;
}

export const usePredictionGraph = (task: PredictionTask) => {
  const editorRef = useRef<InteractiveGraphEditor | null>(null);

  const [graphInput, setGraphInput] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<DegreeMinCycleSettings>(() =>
    readStored<DegreeMinCycleSettings>(STORAGE_KEY, DEFAULT_SETTINGS)
  );
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([
    {
      value: "",
      label: "Loading models..."
    }
  ]);

  const [isGenerating, setIsGenerating] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const applyPhysics = useCallback((enabled: boolean) => {
    if (!editorRef.current) return;

    if (enabled) {
      editorRef.current.enablePhysics();
    } else {
      editorRef.current.disablePhysics();
    }
  }, []);

  const fetchAvailableModels = useCallback(async () => {
    try {
      const data = await fetchModels(task);
      const autoLabel = data.default ? `Auto (${data.default})` : "Auto";
      const options: ModelOption[] = [
        {
          value: "",
          label: autoLabel
        },
        ...data.models.map((model) => ({
          value: model.model_id,
          label: `${model.model_id} (${model.model_type})`
        }))
      ];

      setModelOptions(options);
    } catch {
      setModelOptions([{ value: "", label: "Auto" }]);
    }
  }, [task]);

  useEffect(() => {
    fetchAvailableModels().catch(() => {
      // no-op
    });
  }, [fetchAvailableModels]);

  useEffect(() => {
    applyPhysics(settings.enablePhysics);
  }, [applyPhysics, settings.enablePhysics]);

  const analyzeGraph = useCallback(
    async (providedGraph?: string) => {
      const editor = editorRef.current;
      if (!editor) {
        return;
      }

      const sourceGraph = providedGraph ?? graphInput;
      const normalized = normalizeEdgeList(sourceGraph);

      if (sourceGraph !== normalized) {
        setGraphInput(normalized);
      }

      if (!normalized.trim() || editor.nodes.length === 0) {
        return;
      }

      if (!providedGraph && editor.hasPredictions()) {
        return;
      }

      setIsAnalyzing(true);
      setError(null);

      try {
        const result = await analyzeGraphForTask(task, normalized, settings.modelId);
        const predictions: GraphPrediction[] = result.predictions.map((prediction) => ({
          nodeId: prediction.node_id,
          predicted: prediction.predicted,
          actual: prediction.true
        }));

        editor.updatePredictions(predictions);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Failed to analyze graph");
      } finally {
        setIsAnalyzing(false);
      }
    },
    [graphInput, settings.modelId, task]
  );

  const generateRandomGraph = useCallback(async () => {
    setIsGenerating(true);
    setError(null);

    try {
      const result = await generateGraphForTask(task, settings);
      setGraphInput(result.graph);
      editorRef.current?.loadFromEdgeList(result.graph);
      await analyzeGraph(result.graph);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Failed to generate graph");
    } finally {
      setIsGenerating(false);
    }
  }, [analyzeGraph, settings, task]);

  const clearCanvas = useCallback(() => {
    editorRef.current?.clear();
    setGraphInput("");
    setError(null);
  }, []);

  const onTextInputChange = useCallback((value: string) => {
    setGraphInput(value);
    editorRef.current?.clearPredictions();
    editorRef.current?.loadFromEdgeList(value);
  }, []);

  const onEditorReady = useCallback(
    (editor: InteractiveGraphEditor | null) => {
      editorRef.current = editor;
      if (editor) {
        applyPhysics(settings.enablePhysics);
      }
    },
    [applyPhysics, settings.enablePhysics]
  );

  const saveSettings = useCallback(
    (nextSettings: DegreeMinCycleSettings) => {
      const safe: DegreeMinCycleSettings = {
        ...nextSettings,
        minNodes: Math.min(nextSettings.minNodes, nextSettings.maxNodes),
        maxNodes: Math.max(nextSettings.minNodes, nextSettings.maxNodes),
        minProb: Math.min(nextSettings.minProb, nextSettings.maxProb),
        maxProb: Math.max(nextSettings.minProb, nextSettings.maxProb)
      };

      setSettings(safe);
      writeStored(STORAGE_KEY, safe);
      applyPhysics(safe.enablePhysics);

      if (editorRef.current) {
        editorRef.current.clearPredictions();
        editorRef.current.renderNow();
      }

      setSettingsOpen(false);
    },
    [applyPhysics]
  );

  const setGraphFromEditor = useCallback((edgeList: string) => {
    setGraphInput(edgeList);
  }, []);

  const controls = useMemo(
    () => ({
      generateButtonLabel: isGenerating ? "Generating..." : "Random Graph",
      analyzeButtonLabel: "Analyze"
    }),
    [isGenerating]
  );

  return {
    graphInput,
    settings,
    settingsOpen,
    modelOptions,
    isGenerating,
    isAnalyzing,
    error,
    controls,
    onEditorReady,
    onEditorGraphChange: setGraphFromEditor,
    onEditorAnalyzeRequest: analyzeGraph,
    setSettingsOpen,
    saveSettings,
    generateRandomGraph,
    analyzeGraph,
    clearCanvas,
    onTextInputChange
  };
};

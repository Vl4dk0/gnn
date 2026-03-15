import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { CageSettings, CageStatusResponse } from "../types/api";
import {
  fetchCageModels,
  fetchCageStatus,
  startCageGeneration,
  stopCageGeneration
} from "../services/cage";
import { readStored, writeStored } from "../utils/storage";
import { InteractiveGraphEditor } from "../graph/InteractiveGraphEditor";
import { mooreBound, resolveMooreBoundLimit } from "../utils/mooreBound";

const DEFAULT_SETTINGS: CageSettings = {
  generatorType: "randomwalk",
  modelId: null,
  pollingInterval: 500,
  enablePhysics: true
};

const STORAGE_KEY = "cageGeneratorSettings";

interface ModelOption {
  value: string;
  label: string;
}

const formatElapsed = (seconds: number): string => {
  if (seconds < 1) {
    return `${(seconds * 1000).toFixed(0)}ms`;
  }

  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }

  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1);
  return `${mins}m ${secs}s`;
};

export const useCageGeneration = () => {
  const editorRef = useRef<InteractiveGraphEditor | null>(null);
  const latestSessionIdRef = useRef<string | null>(null);

  const [degreeK, setDegreeK] = useState(3);
  const [girthG, setGirthG] = useState(5);
  const [settings, setSettings] = useState<CageSettings>(() =>
    readStored<CageSettings>(STORAGE_KEY, DEFAULT_SETTINGS)
  );
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([
    { value: "", label: "Loading RL models..." }
  ]);

  const [status, setStatus] = useState<CageStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [stoppedByUser, setStoppedByUser] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const mooreBoundLimit = resolveMooreBoundLimit();

  const currentMooreBound = useMemo(() => {
    if (degreeK < 2 || girthG < 3) {
      return null;
    }

    try {
      return mooreBound(degreeK, girthG);
    } catch {
      return null;
    }
  }, [degreeK, girthG]);

  const isMooreBoundOverLimit = currentMooreBound !== null && currentMooreBound > mooreBoundLimit;

  useEffect(() => {
    latestSessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const data = await fetchCageModels();
        const options: ModelOption[] = [
          {
            value: "",
            label: data.default ? `Best Available (${data.default})` : "Best Available"
          },
          ...data.models.map((model) => ({
            value: model.model_id,
            label: model.model_id
          }))
        ];
        setModelOptions(options);
      } catch {
        setModelOptions([{ value: "", label: "Best Available" }]);
      }
    };

    void loadModels();
  }, []);

  useEffect(() => {
    if (!editorRef.current) return;

    if (settings.enablePhysics) {
      editorRef.current.enablePhysics();
    } else {
      editorRef.current.disablePhysics();
    }
  }, [settings.enablePhysics]);

  const onEditorReady = useCallback(
    (editor: InteractiveGraphEditor | null) => {
      editorRef.current = editor;
      if (!editor) return;

      if (settings.enablePhysics) {
        editor.enablePhysics();
      } else {
        editor.disablePhysics();
      }
    },
    [settings.enablePhysics]
  );

  const pollStatus = useCallback(async (activeSessionId: string) => {
    if (!activeSessionId) {
      return;
    }

    const currentStatus = await fetchCageStatus(activeSessionId);
    setStatus(currentStatus);

    if (currentStatus.current_graph) {
      editorRef.current?.loadFromEdgeList(currentStatus.current_graph);
    }

    if (currentStatus.stopped) {
      setIsGenerating(false);
      setError("Generation stopped - page was navigated away");
      setSessionId(null);
      return;
    }

    if (currentStatus.is_complete) {
      setIsGenerating(false);
      setSessionId(null);

      if (currentStatus.success) {
        setSuccessMessage(`Valid cage! (${formatElapsed(currentStatus.elapsed_time)})`);
      } else {
        setError(
          `Generation completed but cage is not valid. Nodes: ${currentStatus.num_nodes}, Girth: ${currentStatus.girth ?? "∞"}`
        );
      }
    }
  }, []);

  useEffect(() => {
    if (!sessionId || !isGenerating) {
      return;
    }

    const intervalId = window.setInterval(() => {
      pollStatus(sessionId).catch((cause) => {
        setIsGenerating(false);
        setSessionId(null);
        setError(cause instanceof Error ? cause.message : "Polling failed");
      });
    }, settings.pollingInterval);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [isGenerating, pollStatus, sessionId, settings.pollingInterval]);

  useEffect(() => {
    return () => {
      const activeSessionId = latestSessionIdRef.current;
      if (activeSessionId) {
        stopCageGeneration(activeSessionId).catch(() => {
          // no-op
        });
      }
    };
  }, []);

  const start = useCallback(async () => {
    if (degreeK < 2 || girthG < 3) {
      setError("k must be >= 2 and g must be >= 3");
      return;
    }

    if (isMooreBoundOverLimit) {
      setError(`Moore bound exceeds limit (${currentMooreBound} > ${mooreBoundLimit})`);
      return;
    }

    setError(null);
    setSuccessMessage(null);
    setStoppedByUser(false);
    setIsGenerating(true);

    try {
      const result = await startCageGeneration(
        degreeK,
        girthG,
        settings.generatorType,
        settings.generatorType === "rl" ? settings.modelId : null
      );
      setSessionId(result.session_id);
      await pollStatus(result.session_id);
    } catch (cause) {
      setIsGenerating(false);
      setSessionId(null);
      setError(cause instanceof Error ? cause.message : "Failed to start generation");
    }
  }, [
    currentMooreBound,
    degreeK,
    girthG,
    isMooreBoundOverLimit,
    mooreBoundLimit,
    pollStatus,
    settings.generatorType,
    settings.modelId
  ]);

  const stop = useCallback(async () => {
    if (!sessionId) {
      return;
    }

    try {
      const currentStatus = await fetchCageStatus(sessionId);
      setStatus(currentStatus);
      await stopCageGeneration(sessionId);
      setStoppedByUser(true);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Failed to stop generation");
    } finally {
      setSessionId(null);
      setIsGenerating(false);
    }
  }, [sessionId]);

  const clearCanvas = useCallback(() => {
    editorRef.current?.clear();
  }, []);

  const saveSettings = useCallback((nextSettings: CageSettings) => {
    const safe: CageSettings = {
      ...nextSettings,
      modelId: nextSettings.generatorType === "rl" ? nextSettings.modelId : null
    };
    setSettings(safe);
    writeStored(STORAGE_KEY, safe);
    setSettingsOpen(false);
  }, []);

  return {
    degreeK,
    setDegreeK,
    girthG,
    setGirthG,
    settings,
    settingsOpen,
    modelOptions,
    setSettingsOpen,
    saveSettings,
    status,
    error,
    successMessage,
    stoppedByUser,
    isGenerating,
    currentMooreBound,
    mooreBoundLimit,
    isMooreBoundOverLimit,
    onEditorReady,
    start,
    stop,
    clearCanvas
  };
};

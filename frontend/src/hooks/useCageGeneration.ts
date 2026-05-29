import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { CageExecutionMode, CageSettings, CageStatusResponse } from "../types/api";
import {
  fetchCageStatus,
  fetchCageVoltageGirthModels,
  startCageGeneration,
  stepCageGeneration,
  stopCageGeneration
} from "../services/cage";
import { readStored, writeStored } from "../utils/storage";
import { InteractiveGraphEditor } from "../graph/InteractiveGraphEditor";
import { mooreBound, resolveMooreBoundLimit } from "../utils/mooreBound";

const DEFAULT_SETTINGS: CageSettings = {
  generatorType: "randomwalk",
  executionMode: "async",
  modelId: null,
  pollingInterval: 500,
  stepsPerTick: 1,
  autoStepInterval: 250,
  enablePhysics: true
};

const STORAGE_KEY = "cageGeneratorSettings";

type CageRunPhase =
  | "idle"
  | "starting"
  | "async-running"
  | "stepped-ready"
  | "stepping"
  | "auto-stepping"
  | "complete";

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

const normalizeSettings = (settings: CageSettings): CageSettings => {
  const merged = { ...DEFAULT_SETTINGS, ...settings };

  return {
    ...merged,
    // Preserve the user's executionMode for all generator types.
    executionMode: merged.executionMode,
    // Only voltage keeps a model_id (auto-resolved girth predictor).
    // RL sends null so the backend picks the best actor-critic model.
    // All other generators never use a model_id.
    modelId: merged.generatorType === "voltage" ? merged.modelId : null,
    pollingInterval: Math.max(50, Math.min(2000, merged.pollingInterval)),
    stepsPerTick: Math.max(1, Math.min(100, Math.round(merged.stepsPerTick))),
    autoStepInterval: Math.max(50, Math.min(2000, merged.autoStepInterval))
  };
};

export const useCageGeneration = () => {
  const editorRef = useRef<InteractiveGraphEditor | null>(null);
  const latestSessionIdRef = useRef<string | null>(null);

  const [degreeK, setDegreeK] = useState(3);
  const [girthG, setGirthG] = useState(5);
  const [settings, setSettings] = useState<CageSettings>(() =>
    normalizeSettings(readStored<CageSettings>(STORAGE_KEY, DEFAULT_SETTINGS))
  );
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [voltageGirthDefault, setVoltageGirthDefault] = useState<string | null>(null);

  const [status, setStatus] = useState<CageStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [stoppedByUser, setStoppedByUser] = useState(false);
  const [phase, setPhase] = useState<CageRunPhase>("idle");
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
  const isSteppedMode = settings.executionMode === "stepped";
  const isGenerating = phase === "starting" || phase === "async-running";
  const isStepping = phase === "stepping";
  const isAutoStepping = phase === "auto-stepping";
  const hasActiveSession = sessionId !== null && phase !== "complete";
  const canStep = sessionId !== null && (phase === "stepped-ready" || phase === "stepping");

  useEffect(() => {
    latestSessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    // Fetch the best voltage girth predictor so it can be auto-selected when
    // the voltage generator is active. No model dropdown is shown to the user;
    // the resolved default is sent transparently on every voltage generation.
    const loadVoltageGirthModels = async () => {
      try {
        const data = await fetchCageVoltageGirthModels();
        setVoltageGirthDefault(data.default ?? null);
      } catch {
        setVoltageGirthDefault(null);
      }
    };

    void loadVoltageGirthModels();
  }, []);

  // Auto-select the unified girth predictor whenever the voltage generator
  // is active and modelId is unset (null). Since there is no user-facing
  // model dropdown, modelId for voltage is always either null (initial /
  // post-generator-switch) or the resolved default id (after this effect runs).
  useEffect(() => {
    if (
      settings.generatorType === "voltage" &&
      settings.modelId === null &&
      voltageGirthDefault !== null
    ) {
      setSettings((current) => ({ ...current, modelId: voltageGirthDefault }));
    }
  }, [settings.generatorType, settings.modelId, voltageGirthDefault]);

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

  const applyStatus = useCallback((currentStatus: CageStatusResponse) => {
    setStatus(currentStatus);

    if (currentStatus.current_graph) {
      editorRef.current?.updateFromEdgeList(currentStatus.current_graph);
    }

    if (currentStatus.stopped) {
      setSessionId(null);
      setPhase("complete");
      setError(
        currentStatus.timed_out
          ? "Generation timed out"
          : "Generation stopped - page was navigated away"
      );
      return;
    }

    if (!currentStatus.is_complete) {
      return;
    }

    setSessionId(null);
    setPhase("complete");

    if (currentStatus.success) {
      setSuccessMessage(`Valid cage! (${formatElapsed(currentStatus.elapsed_time)})`);
    } else {
      setError(
        `Generation completed but cage is not valid. Nodes: ${currentStatus.num_nodes}, Girth: ${currentStatus.girth ?? "∞"}`
      );
    }
  }, []);

  const pollStatus = useCallback(
    async (activeSessionId: string) => {
      const currentStatus = await fetchCageStatus(activeSessionId);
      applyStatus(currentStatus);
      return currentStatus;
    },
    [applyStatus]
  );

  useEffect(() => {
    if (!sessionId || phase !== "async-running") {
      return;
    }

    let cancelled = false;
    let timeoutId: number | null = null;

    const pollOnce = async () => {
      try {
        const currentStatus = await pollStatus(sessionId);
        if (cancelled || currentStatus.is_complete || currentStatus.stopped) {
          return;
        }
        timeoutId = window.setTimeout(pollOnce, settings.pollingInterval);
      } catch (cause) {
        if (cancelled) return;
        setSessionId(null);
        setPhase("complete");
        setError(cause instanceof Error ? cause.message : "Polling failed");
      }
    };

    timeoutId = window.setTimeout(pollOnce, settings.pollingInterval);

    return () => {
      cancelled = true;
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [phase, pollStatus, sessionId, settings.pollingInterval]);

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
    setPhase("starting");

    const mode: CageExecutionMode = isSteppedMode ? "stepped" : "async";

    try {
      // Voltage uses the auto-resolved girth predictor stored in modelId.
      // RL sends null so the backend picks the best actor-critic model.
      // All other generators don't use a model (normalizeSettings ensures null).
      const apiModelId = settings.modelId;
      const result = await startCageGeneration(
        degreeK,
        girthG,
        settings.generatorType,
        mode,
        apiModelId
      );
      setSessionId(result.session_id);

      const initialStatus = await pollStatus(result.session_id);
      if (initialStatus.is_complete || initialStatus.stopped) {
        return;
      }

      setPhase(mode === "stepped" ? "stepped-ready" : "async-running");
    } catch (cause) {
      setSessionId(null);
      setPhase("complete");
      setError(cause instanceof Error ? cause.message : "Failed to start generation");
    }
  }, [
    currentMooreBound,
    degreeK,
    girthG,
    isMooreBoundOverLimit,
    isSteppedMode,
    mooreBoundLimit,
    pollStatus,
    settings.generatorType,
    settings.modelId
  ]);

  const stepOnce = useCallback(async () => {
    if (!sessionId || phase === "stepping" || phase === "auto-stepping") {
      return;
    }

    setError(null);
    setPhase("stepping");

    try {
      const currentStatus = await stepCageGeneration(sessionId, settings.stepsPerTick);
      applyStatus(currentStatus);
      if (!currentStatus.is_complete && !currentStatus.stopped) {
        setPhase("stepped-ready");
      }
    } catch (cause) {
      setPhase("stepped-ready");
      setError(cause instanceof Error ? cause.message : "Failed to step generation");
    }
  }, [applyStatus, phase, sessionId, settings.stepsPerTick]);

  useEffect(() => {
    if (!sessionId || phase !== "auto-stepping") {
      return;
    }

    let cancelled = false;
    let timeoutId: number | null = null;

    const stepLoop = async () => {
      try {
        const currentStatus = await stepCageGeneration(sessionId, settings.stepsPerTick);
        applyStatus(currentStatus);

        if (cancelled || currentStatus.is_complete || currentStatus.stopped) {
          return;
        }

        timeoutId = window.setTimeout(stepLoop, settings.autoStepInterval);
      } catch (cause) {
        if (cancelled) return;
        setPhase("stepped-ready");
        setError(cause instanceof Error ? cause.message : "Auto step failed");
      }
    };

    timeoutId = window.setTimeout(stepLoop, 0);

    return () => {
      cancelled = true;
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [applyStatus, phase, sessionId, settings.autoStepInterval, settings.stepsPerTick]);

  const startAutoStepping = useCallback(() => {
    if (sessionId && phase === "stepped-ready") {
      setError(null);
      setPhase("auto-stepping");
    }
  }, [phase, sessionId]);

  const pauseAutoStepping = useCallback(() => {
    if (phase === "auto-stepping") {
      setPhase("stepped-ready");
    }
  }, [phase]);

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
      setPhase("complete");
    }
  }, [sessionId]);

  const clearCanvas = useCallback(() => {
    editorRef.current?.clear();
  }, []);

  const saveSettings = useCallback(
    (nextSettings: CageSettings) => {
      const safe = normalizeSettings(nextSettings);
      // If the user is saving the voltage generator with modelId still
      // unresolved (null = auto-select pending) and the default has loaded,
      // resolve it before persisting. Without this, a fast save followed by
      // a reload would re-enter the legacy-migration path and flip the
      // user from ML-guided to pure tabu/random.
      const resolved: CageSettings =
        safe.generatorType === "voltage" &&
        safe.modelId === null &&
        voltageGirthDefault !== null
          ? { ...safe, modelId: voltageGirthDefault }
          : safe;
      setSettings(resolved);
      writeStored(STORAGE_KEY, resolved);
      setSettingsOpen(false);
    },
    [voltageGirthDefault]
  );

  return {
    degreeK,
    setDegreeK,
    girthG,
    setGirthG,
    settings,
    settingsOpen,
    setSettingsOpen,
    saveSettings,
    status,
    error,
    successMessage,
    stoppedByUser,
    isGenerating,
    isSteppedMode,
    isStepping,
    isAutoStepping,
    hasActiveSession,
    canStep,
    phase,
    currentMooreBound,
    mooreBoundLimit,
    isMooreBoundOverLimit,
    onEditorReady,
    start,
    stepOnce,
    startAutoStepping,
    pauseAutoStepping,
    stop,
    clearCanvas
  };
};

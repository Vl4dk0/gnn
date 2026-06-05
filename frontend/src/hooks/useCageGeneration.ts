import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { CageExportFormat, CageExecutionMode, CageSettings, CageStatusResponse } from "../types/api";
import {
  exportCageGraph,
  fetchCageStatus,
  fetchCageVoltageGirthModels,
  startCageGeneration,
  stepCageGeneration,
  stopCageGeneration
} from "../services/cage";
import { readStored, writeStored } from "../utils/storage";
import { InteractiveGraphEditor } from "../graph/InteractiveGraphEditor";
import { mooreBound, resolveMooreBoundLimit } from "../utils/mooreBound";
import { CAGE_METHODS, getMethod } from "../pages/apps/cageMethods";
import type { CageMethodId, ModelInfo } from "../types/api";

const DEFAULT_SETTINGS: CageSettings = {
  methodId: "astar",
  executionMode: "async",
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

/**
 * Migrate a persisted settings object from the old shape (generatorType +
 * modelId) or validate an already-migrated one.  Returns a fully valid
 * CageSettings with the new methodId field.
 */
const normalizeSettings = (raw: Record<string, unknown>): CageSettings => {
  // Handle old persisted shape that had generatorType instead of methodId.
  let methodId = raw["methodId"] as CageSettings["methodId"] | undefined;
  if (!methodId) {
    const oldGen = raw["generatorType"] as string | undefined;
    if (oldGen === "voltage" || oldGen === "forge") {
      methodId = "voltage_algebraic";
    } else if (
      oldGen === "astar" ||
      oldGen === "randomwalk" ||
      oldGen === "bruteforce" ||
      oldGen === "rl" ||
      oldGen === "voltage_rl"
    ) {
      methodId = oldGen;
    } else {
      methodId = DEFAULT_SETTINGS.methodId;
    }
  }

  // Guard against corrupted or legacy stored ids so resolveGenerationArgs /
  // getMethod can never throw at generation time.
  if (!CAGE_METHODS.some((m) => m.id === methodId)) {
    methodId = DEFAULT_SETTINGS.methodId;
  }

  const merged = { ...DEFAULT_SETTINGS, ...raw, methodId };

  return {
    methodId: merged.methodId,
    executionMode: merged.executionMode === "stepped" ? "stepped" : "async",
    pollingInterval: Math.max(50, Math.min(2000, merged.pollingInterval)),
    stepsPerTick: Math.max(1, Math.min(100, Math.round(merged.stepsPerTick))),
    autoStepInterval: Math.max(50, Math.min(2000, merged.autoStepInterval)),
    enablePhysics: Boolean(merged.enablePhysics)
  };
};

export const useCageGeneration = () => {
  const editorRef = useRef<InteractiveGraphEditor | null>(null);
  const latestSessionIdRef = useRef<string | null>(null);

  const [degreeK, setDegreeK] = useState(3);
  const [girthG, setGirthG] = useState(5);
  const [settings, setSettings] = useState<CageSettings>(() =>
    normalizeSettings(readStored<Record<string, unknown>>(STORAGE_KEY, {}))
  );
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Voltage predictor models loaded once on mount. Used to resolve model_id
  // at generation time (never stored in settings).
  const [voltageModels, setVoltageModels] = useState<ModelInfo[]>([]);

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

  // Load voltage predictor models once on mount.
  useEffect(() => {
    const loadVoltageModels = async () => {
      try {
        const data = await fetchCageVoltageGirthModels();
        setVoltageModels(data.models);
      } catch {
        setVoltageModels([]);
      }
    };

    void loadVoltageModels();
  }, []);

  // Derive which method IDs are currently disabled (predictor models not available).
  const disabledMethods = useMemo(() => {
    const disabled = new Set<CageMethodId>();
    const hasGirth = voltageModels.some((m) => m.kind === "girth");
    const hasTabu = voltageModels.some((m) => m.kind === "tabu_cost");
    if (!hasGirth) disabled.add("voltage_girth");
    if (!hasTabu) disabled.add("voltage_tabu");
    return disabled;
  }, [voltageModels]);

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

  /**
   * Resolve the selected method to a { generator, model_id } pair for the API.
   * Voltage sub-methods pick the right model by kind; everything else is null.
   */
  const resolveGenerationArgs = useCallback(
    (methodId: CageSettings["methodId"]): { generator: string; model_id: string | null } => {
      const method = getMethod(methodId);
      if (method.voltageKind) {
        const match = voltageModels.find((m) => m.kind === method.voltageKind);
        return { generator: method.generator, model_id: match?.model_id ?? null };
      }
      return { generator: method.generator, model_id: null };
    },
    [voltageModels]
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
      setSuccessMessage(
        `Valid (${currentStatus.k},${currentStatus.g})-graph! (${formatElapsed(currentStatus.elapsed_time)})`
      );
    } else {
      setError(
        `Generation completed but the (${currentStatus.k},${currentStatus.g})-graph is not valid. Nodes: ${currentStatus.num_nodes}, Girth: ${currentStatus.girth ?? "∞"}`
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
    const { generator, model_id } = resolveGenerationArgs(settings.methodId);

    try {
      const result = await startCageGeneration(
        degreeK,
        girthG,
        // GeneratorType assertion is safe: method registry only contains valid generator values.
        generator as Parameters<typeof startCageGeneration>[2],
        mode,
        model_id
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
    resolveGenerationArgs,
    settings.methodId
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

  const downloadGraph = useCallback(
    async (format: CageExportFormat) => {
      const edgeList = editorRef.current?.toEdgeList() ?? "";
      if (!edgeList.trim()) {
        setError("No graph to download");
        return;
      }
      try {
        const result = await exportCageGraph(edgeList, format, degreeK, girthG);
        const blob = new Blob([result.content], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = result.filename;
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        URL.revokeObjectURL(url);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Download failed");
      }
    },
    [degreeK, girthG]
  );

  const canDownload = status !== null && status.num_nodes > 0;

  const saveSettings = useCallback(
    (nextSettings: CageSettings) => {
      const safe = normalizeSettings(nextSettings as unknown as Record<string, unknown>);
      setSettings(safe);
      writeStored(STORAGE_KEY, safe);
      setSettingsOpen(false);
    },
    []
  );

  return {
    degreeK,
    setDegreeK,
    girthG,
    setGirthG,
    settings,
    setSettings,
    settingsOpen,
    setSettingsOpen,
    saveSettings,
    disabledMethods,
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
    clearCanvas,
    downloadGraph,
    canDownload
  };
};

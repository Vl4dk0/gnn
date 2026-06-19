import { useEffect, useRef, useState } from "react";

import { EditorPlaceholder } from "../../components/graph/EditorPlaceholder";
import { GraphCanvas } from "../../components/graph/GraphCanvas";
import { SettingsModal } from "../../components/graph/SettingsModal";
import { BackButton } from "../../components/ui/BackButton";
import { IconButton } from "../../components/ui/IconButton";
import { PrimaryButton } from "../../components/ui/PrimaryButton";
import { SecondaryButton } from "../../components/ui/SecondaryButton";
import { SettingGroup } from "../../components/ui/SettingGroup";
import { SingleRangeSlider } from "../../components/ui/SingleRangeSlider";
import type { InteractiveGraphEditor } from "../../graph/InteractiveGraphEditor";
import { inferGirth, inferK, planExcision } from "../../graph/excision/excisionPlanner";
import type { ExcisionFrame } from "../../graph/excision/excisionPlanner";
import { importGraphFromFile } from "../../services/cage";

const DODECAHEDRON_EDGE_LIST = [
  "0 1",
  "0 10",
  "0 19",
  "1 2",
  "1 8",
  "2 3",
  "2 6",
  "3 4",
  "3 19",
  "4 5",
  "4 17",
  "5 6",
  "5 15",
  "6 7",
  "7 8",
  "7 14",
  "8 9",
  "9 10",
  "9 13",
  "10 11",
  "11 12",
  "11 18",
  "12 13",
  "12 16",
  "13 14",
  "14 15",
  "15 16",
  "16 17",
  "17 18",
  "18 19"
].join("\n");

export const ExcisionPage = () => {
  const editorRef = useRef<InteractiveGraphEditor | null>(null);
  const originalRef = useRef<string>("");
  const autoTimerRef = useRef<number | null>(null);
  const importInputRef = useRef<HTMLInputElement | null>(null);

  const [inferredK, setInferredK] = useState(0);
  const [inferredG, setInferredG] = useState(0);
  const [frames, setFrames] = useState<ExcisionFrame[]>([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [isAutoStepping, setIsAutoStepping] = useState(false);
  const [planMessage, setPlanMessage] = useState<string | null>(null);
  const [hasGraph, setHasGraph] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);

  // Settings modal state
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [stepsPerTick, setStepsPerTick] = useState(1);
  const [autoDelay, setAutoDelay] = useState(700);
  const [physicsEnabled, setPhysicsEnabled] = useState(true);

  // Refs mirroring current values so the auto-step timer reads live state
  // instead of values captured when the timer was created.
  const stepsPerTickRef = useRef(stepsPerTick);
  const framesRef = useRef(frames);

  useEffect(() => {
    stepsPerTickRef.current = stepsPerTick;
  }, [stepsPerTick]);

  useEffect(() => {
    framesRef.current = frames;
  }, [frames]);

  const clearAutoTimer = () => {
    if (autoTimerRef.current !== null) {
      window.clearInterval(autoTimerRef.current);
      autoTimerRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      clearAutoTimer();
    };
  }, []);

  const refreshInferred = () => {
    const editor = editorRef.current;
    if (!editor) return;
    const edgeList = editor.toEdgeList();
    setInferredK(inferK(edgeList));
    setInferredG(inferGirth(edgeList));
  };

  const applyPhysics = (editor: InteractiveGraphEditor, enabled: boolean) => {
    if (enabled) {
      editor.enablePhysics();
    } else {
      editor.disablePhysics();
    }
  };

  const handleEditorReady = (editor: InteractiveGraphEditor | null) => {
    editorRef.current = editor;
    if (editor) {
      applyPhysics(editor, physicsEnabled);
      refreshInferred();
    }
  };

  const handleGraphChange = (edgeList: string) => {
    setHasGraph(edgeList.trim().length > 0);
    refreshInferred();
  };

  const applyFrame = (frame: ExcisionFrame) => {
    const editor = editorRef.current;
    if (!editor) return;
    editor.updateFromEdgeList(frame.edgeList);
    editor.setHighlights(frame.nodeHighlights, frame.edgeHighlights);
  };

  // The auto-step timer lifecycle is owned by this effect so that changing the
  // interval (autoDelay) while running recreates the timer with the new delay.
  // The callback reads stepsPerTickRef/framesRef so steps-per-tick changes also
  // take effect live without restarting.
  useEffect(() => {
    if (!isAutoStepping) return;
    clearAutoTimer();
    autoTimerRef.current = window.setInterval(() => {
      setFrameIndex((current) => {
        const activeFrames = framesRef.current;
        if (activeFrames.length === 0) return current;
        const next = current + stepsPerTickRef.current;
        if (next >= activeFrames.length) {
          clearAutoTimer();
          setIsAutoStepping(false);
          const last = activeFrames.length - 1;
          applyFrame(activeFrames[last]);
          return last;
        }
        applyFrame(activeFrames[next]);
        return next;
      });
    }, autoDelay);
    return () => {
      clearAutoTimer();
    };
    // applyFrame is stable for the page lifetime, so it is omitted from deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAutoStepping, autoDelay]);

  const loadDodecahedron = () => {
    const editor = editorRef.current;
    if (!editor) return;
    resetAnimation();
    editor.loadFromEdgeList(DODECAHEDRON_EDGE_LIST);
    applyPhysics(editor, physicsEnabled);
    setHasGraph(true);
    refreshInferred();
  };

  const resetAnimation = () => {
    clearAutoTimer();
    setIsAutoStepping(false);
    setIsAnimating(false);
    setFrames([]);
    setFrameIndex(0);
    setPlanMessage(null);
    const editor = editorRef.current;
    if (editor && originalRef.current) {
      editor.updateFromEdgeList(originalRef.current);
      editor.clearHighlights();
      setHasGraph(originalRef.current.trim().length > 0);
    }
    refreshInferred();
  };

  const runExcision = () => {
    const editor = editorRef.current;
    if (!editor) return;
    clearAutoTimer();
    setIsAutoStepping(false);

    const original = editor.toEdgeList();
    originalRef.current = original;

    const g = inferGirth(original);
    if (!Number.isFinite(g) || g < 3) {
      setPlanMessage(
        "Build a graph with at least one cycle (girth 3 or more) before running excision."
      );
      return;
    }

    const plan = planExcision(original, g);
    if (plan.frames.length === 0) {
      setPlanMessage(plan.message);
      return;
    }

    setFrames(plan.frames);
    setFrameIndex(0);
    setIsAnimating(true);
    setHasGraph(true);
    setPlanMessage(plan.message);
    applyFrame(plan.frames[0]);
  };

  const stepForward = () => {
    if (frames.length === 0) return;
    setFrameIndex((current) => {
      const next = Math.min(current + stepsPerTick, frames.length - 1);
      applyFrame(frames[next]);
      return next;
    });
  };

  const stepBackward = () => {
    if (frames.length === 0) return;
    setFrameIndex((current) => {
      const prev = Math.max(current - stepsPerTick, 0);
      applyFrame(frames[prev]);
      return prev;
    });
  };

  const startAutoStepping = () => {
    if (frames.length === 0) return;
    setIsAutoStepping(true);
  };

  const stopAutoStepping = () => {
    setIsAutoStepping(false);
  };

  const handleImportFile = async (file: File) => {
    setImportError(null);
    try {
      const edgeList = await importGraphFromFile(file);
      const editor = editorRef.current;
      if (editor) {
        editor.loadFromEdgeList(edgeList);
        applyPhysics(editor, physicsEnabled);
        setHasGraph(edgeList.trim().length > 0);
        refreshInferred();
      }
    } catch (cause) {
      setImportError(cause instanceof Error ? cause.message : "Failed to import graph");
    }
    if (importInputRef.current) {
      importInputRef.current.value = "";
    }
  };

  const handlePhysicsChange = (enabled: boolean) => {
    setPhysicsEnabled(enabled);
    const editor = editorRef.current;
    if (editor) {
      applyPhysics(editor, enabled);
    }
  };

  const atLastFrame = frames.length > 0 && frameIndex >= frames.length - 1;
  const currentCaption = frames.length > 0 ? frames[frameIndex].caption : null;

  return (
    <div className="relative h-dvh overflow-hidden bg-transparent">
      <GraphCanvas
        onReady={handleEditorReady}
        onGraphChange={handleGraphChange}
        canvasClassName="rounded-none"
      >
        <EditorPlaceholder
          visible={!hasGraph}
          intro="The excision editor. Build a (k,g)-graph by hand or import one, then press Run to watch a tree get removed and the boundary stitched back into a smaller graph."
          showControls
        />

        <BackButton
          href="/excision"
          label="Back to Excision"
          iconOnly
          className="absolute left-5 top-5 z-20"
        />

        <IconButton
          positionClassName="right-5 top-5"
          onClick={() => setSettingsOpen(true)}
          title="Editor Settings"
          aria-label="Editor Settings"
        >
          <svg
            className="h-6 w-6 fill-textMuted"
            viewBox="0 0 45.973 45.973"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M43.454,18.443h-2.437c-0.453-1.766-1.16-3.42-2.082-4.933l1.752-1.756c0.473-0.473,0.733-1.104,0.733-1.774 c0-0.669-0.262-1.301-0.733-1.773l-2.92-2.917c-0.947-0.948-2.602-0.947-3.545-0.001l-1.826,1.815 C30.9,6.232,29.296,5.56,27.529,5.128V2.52c0-1.383-1.105-2.52-2.488-2.52h-4.128c-1.383,0-2.471,1.137-2.471,2.52v2.607 c-1.766,0.431-3.38,1.104-4.878,1.977l-1.825-1.815c-0.946-0.948-2.602-0.947-3.551-0.001L5.27,8.205 C4.802,8.672,4.535,9.318,4.535,9.978c0,0.669,0.259,1.299,0.733,1.772l1.752,1.76c-0.921,1.513-1.629,3.167-2.081,4.933H2.501 C1.117,18.443,0,19.555,0,20.935v4.125c0,1.384,1.117,2.471,2.501,2.471h2.438c0.452,1.766,1.159,3.43,2.079,4.943l-1.752,1.763 c-0.474,0.473-0.734,1.106-0.734,1.776s0.261,1.303,0.734,1.776l2.92,2.919c0.474,0.473,1.103,0.733,1.772,0.733 s1.299-0.261,1.773-0.733l1.833-1.816c1.498,0.873,3.112,1.545,4.878,1.978v2.604c0,1.383,1.088,2.498,2.471,2.498h4.128 c1.383,0,2.488-1.115,2.488-2.498v-2.605c1.767-0.432,3.371-1.104,4.869-1.977l1.817,1.812c0.474,0.475,1.104,0.735,1.775,0.735 c0.67,0,1.301-0.261,1.774-0.733l2.92-2.917c0.473-0.472,0.732-1.103,0.734-1.772c0-0.67-0.262-1.299-0.734-1.773l-1.75-1.77 c0.92-1.514,1.627-3.179,2.08-4.943h2.438c1.383,0,2.52-1.087,2.52-2.471v-4.125C45.973,19.555,44.837,18.443,43.454,18.443z M22.976,30.85c-4.378,0-7.928-3.517-7.928-7.852c0-4.338,3.55-7.85,7.928-7.85c4.379,0,7.931,3.512,7.931,7.85 C30.906,27.334,27.355,30.85,22.976,30.85z" />
          </svg>
        </IconButton>

        <section className="absolute left-5 top-1/2 z-20 w-[344px] -translate-y-1/2 rounded-2xl border border-line2 bg-bg1/92 p-4 shadow-card backdrop-blur-md max-[900px]:left-5 max-[900px]:right-5 max-[900px]:top-[72px] max-[900px]:w-auto max-[900px]:translate-y-0">
          <div className="text-sm font-semibold leading-5 text-textMain">Excision Editor</div>

          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-2.5">
              <span className="label-base">Inferred k</span>
              <div className="input-base flex items-center py-2.5 text-textMain">{inferredK}</div>
            </div>
            <div className="flex flex-col gap-2.5">
              <span className="label-base">Inferred g</span>
              <div className="input-base flex items-center py-2.5 text-textMain">
                {Number.isFinite(inferredG) ? inferredG : "∞"}
              </div>
            </div>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <SecondaryButton
              fullWidth={false}
              className="rounded-md px-4 py-2 text-xs"
              onClick={() => loadDodecahedron()}
            >
              Dodecahedron
            </SecondaryButton>

            <SecondaryButton
              fullWidth={false}
              className="rounded-md px-4 py-2 text-xs"
              onClick={() => importInputRef.current?.click()}
            >
              Import graph
            </SecondaryButton>
            <input
              ref={importInputRef}
              type="file"
              accept=".g6,.txt"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) {
                  void handleImportFile(file);
                }
              }}
            />
          </div>

          {importError && <p className="mt-2 text-xs text-textDim">{importError}</p>}

          {planMessage && (
            <div className="mt-2 rounded-xl border border-line2 bg-bg2/75 p-3 text-xs leading-5 text-textMuted">
              {planMessage}
            </div>
          )}
        </section>

        {currentCaption && (
          <div className="pointer-events-none absolute inset-x-0 top-6 z-20 flex justify-center px-4">
            <div className="pointer-events-auto max-w-[680px] rounded-2xl border border-line2 bg-bg1/92 px-5 py-3 text-center shadow-card backdrop-blur">
              <div className="text-[11px] font-semibold uppercase tracking-[0.8px] text-textDim">
                Step {frameIndex + 1} / {frames.length}
              </div>
              <div className="mt-1 text-sm text-textMain">{currentCaption}</div>
            </div>
          </div>
        )}

        <div className="pointer-events-none absolute inset-x-0 bottom-6 z-20 flex justify-center px-4 max-[900px]:bottom-4">
          <div className="pointer-events-auto flex flex-wrap justify-center gap-3">
            {!isAnimating ? (
              <PrimaryButton
                fullWidth={false}
                className="min-w-[220px] rounded-full bg-bg1/92 px-7 py-3 text-sm tracking-[0.8px] backdrop-blur-sm max-[900px]:min-w-[200px]"
                onClick={() => runExcision()}
              >
                Run Excision
              </PrimaryButton>
            ) : (
              <>
                <SecondaryButton
                  fullWidth={false}
                  className="rounded-full bg-bg1/92 px-6 py-3 text-sm tracking-[0.8px] backdrop-blur-sm"
                  onClick={() => stepBackward()}
                  disabled={frameIndex === 0 || isAutoStepping}
                >
                  Back
                </SecondaryButton>
                <PrimaryButton
                  fullWidth={false}
                  className="rounded-full bg-bg1/92 px-6 py-3 text-sm tracking-[0.8px] backdrop-blur-sm"
                  onClick={() => stepForward()}
                  disabled={atLastFrame || isAutoStepping}
                >
                  Step
                </PrimaryButton>
                <SecondaryButton
                  fullWidth={false}
                  className="rounded-full bg-bg1/92 px-6 py-3 text-sm tracking-[0.8px] backdrop-blur-sm"
                  onClick={() => (isAutoStepping ? stopAutoStepping() : startAutoStepping())}
                  disabled={atLastFrame && !isAutoStepping}
                >
                  {isAutoStepping ? "Pause" : "Auto Step"}
                </SecondaryButton>
                <SecondaryButton
                  fullWidth={false}
                  className="rounded-full bg-bg1/92 px-6 py-3 text-sm tracking-[0.8px] backdrop-blur-sm"
                  onClick={() => resetAnimation()}
                >
                  Reset
                </SecondaryButton>
              </>
            )}
          </div>
        </div>
      </GraphCanvas>

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        title="Editor Settings"
      >
        <SettingGroup>
          <label className="label-base mb-2 block normal-case tracking-normal">
            Steps per tick:&nbsp;<span>{stepsPerTick}</span>
          </label>
          <SingleRangeSlider
            id="stepsPerTick"
            min={1}
            max={50}
            step={1}
            value={stepsPerTick}
            onChange={(value) => setStepsPerTick(value)}
          />
          <p className="mt-1 text-xs text-textDim">
            How many frames each Step click and each auto-step tick advances.
          </p>
        </SettingGroup>

        <SettingGroup>
          <label className="label-base mb-2 block normal-case tracking-normal">
            Auto step interval:&nbsp;<span>{autoDelay}ms</span>
          </label>
          <SingleRangeSlider
            id="autoDelay"
            min={150}
            max={2000}
            step={50}
            value={autoDelay}
            onChange={(value) => setAutoDelay(value)}
          />
          <p className="mt-1 text-xs text-textDim">
            Delay between automatic steps when auto-stepping is running.
          </p>
        </SettingGroup>

        <SettingGroup>
          <label className="flex cursor-pointer items-center gap-2 text-[0.95em] text-textMuted">
            <input
              type="checkbox"
              checked={physicsEnabled}
              onChange={(event) => handlePhysicsChange(event.target.checked)}
            />
            Enable spring physics
          </label>
          <p className="mt-1 text-xs text-textDim">
            Spring layout that spreads the graph out automatically.
          </p>
        </SettingGroup>
      </SettingsModal>
    </div>
  );
};

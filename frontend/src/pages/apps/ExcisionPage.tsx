import { useEffect, useRef, useState } from "react";

import { GraphCanvas } from "../../components/graph/GraphCanvas";
import { BackButton } from "../../components/ui/BackButton";
import { InputField } from "../../components/ui/InputField";
import { PrimaryButton } from "../../components/ui/PrimaryButton";
import { SecondaryButton } from "../../components/ui/SecondaryButton";
import { SettingGroup } from "../../components/ui/SettingGroup";
import { SingleRangeSlider } from "../../components/ui/SingleRangeSlider";
import type { InteractiveGraphEditor } from "../../graph/InteractiveGraphEditor";
import { inferK, planExcision } from "../../graph/excision/excisionPlanner";
import type { ExcisionFrame } from "../../graph/excision/excisionPlanner";

const PETERSEN_EDGE_LIST = [
  "0 1",
  "1 2",
  "2 3",
  "3 4",
  "4 0",
  "5 7",
  "7 9",
  "9 6",
  "6 8",
  "8 5",
  "0 5",
  "1 6",
  "2 7",
  "3 8",
  "4 9"
].join("\n");

export const ExcisionPage = () => {
  const editorRef = useRef<InteractiveGraphEditor | null>(null);
  const originalRef = useRef<string>("");
  const autoTimerRef = useRef<number | null>(null);

  const [girthG, setGirthG] = useState(5);
  const [inferredK, setInferredK] = useState(0);
  const [frames, setFrames] = useState<ExcisionFrame[]>([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [isAutoStepping, setIsAutoStepping] = useState(false);
  const [autoDelay, setAutoDelay] = useState(700);
  const [planMessage, setPlanMessage] = useState<string | null>(null);

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

  const refreshK = () => {
    const editor = editorRef.current;
    if (!editor) return;
    setInferredK(inferK(editor.toEdgeList()));
  };

  const handleEditorReady = (editor: InteractiveGraphEditor | null) => {
    editorRef.current = editor;
    if (editor) {
      editor.enablePhysics();
      refreshK();
    }
  };

  const applyFrame = (frame: ExcisionFrame) => {
    const editor = editorRef.current;
    if (!editor) return;
    editor.updateFromEdgeList(frame.edgeList);
    editor.setHighlights(frame.nodeHighlights, frame.edgeHighlights);
  };

  const loadPetersen = () => {
    const editor = editorRef.current;
    if (!editor) return;
    resetAnimation();
    editor.loadFromEdgeList(PETERSEN_EDGE_LIST);
    setGirthG(5);
    refreshK();
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
    }
    refreshK();
  };

  const runExcision = () => {
    const editor = editorRef.current;
    if (!editor) return;
    clearAutoTimer();
    setIsAutoStepping(false);

    const original = editor.toEdgeList();
    originalRef.current = original;

    const plan = planExcision(original, girthG);
    if (plan.frames.length === 0) {
      setPlanMessage(plan.message);
      return;
    }

    setFrames(plan.frames);
    setFrameIndex(0);
    setIsAnimating(true);
    setPlanMessage(plan.message);
    applyFrame(plan.frames[0]);
  };

  const stepForward = () => {
    if (frames.length === 0) return;
    setFrameIndex((current) => {
      const next = Math.min(current + 1, frames.length - 1);
      applyFrame(frames[next]);
      return next;
    });
  };

  const startAutoStepping = () => {
    if (frames.length === 0) return;
    setIsAutoStepping(true);
    clearAutoTimer();
    autoTimerRef.current = window.setInterval(() => {
      setFrameIndex((current) => {
        const next = current + 1;
        if (next >= frames.length) {
          clearAutoTimer();
          setIsAutoStepping(false);
          return current;
        }
        applyFrame(frames[next]);
        return next;
      });
    }, autoDelay);
  };

  const stopAutoStepping = () => {
    clearAutoTimer();
    setIsAutoStepping(false);
  };

  const atLastFrame = frames.length > 0 && frameIndex >= frames.length - 1;
  const currentCaption = frames.length > 0 ? frames[frameIndex].caption : null;

  return (
    <div className="relative h-dvh overflow-hidden bg-transparent">
      <GraphCanvas
        onReady={handleEditorReady}
        onGraphChange={() => refreshK()}
        canvasClassName="rounded-none"
      >
        <BackButton
          href="/excision"
          label="Back to Excision"
          iconOnly
          className="absolute left-5 top-5 z-20"
        />

        <section className="absolute left-5 top-1/2 z-20 w-[344px] -translate-y-1/2 rounded-2xl border border-line2 bg-bg1/92 p-4 shadow-card backdrop-blur-md max-[900px]:left-5 max-[900px]:right-5 max-[900px]:top-[72px] max-[900px]:w-auto max-[900px]:translate-y-0">
          <div className="text-sm font-semibold leading-5 text-textMain">Excision Editor</div>
          <p className="mt-1 text-xs leading-5 text-textMuted">
            Build a (k,g)-graph (double-click for a node, right-click then click to add an edge),
            set g, then run.
          </p>

          <div className="mt-4 grid grid-cols-2 gap-3">
            <InputField
              id="girthG"
              label="Girth (g)"
              type="number"
              min={3}
              max={12}
              value={girthG}
              className="py-2.5"
              onChange={(event) => setGirthG(Number.parseInt(event.target.value, 10) || 0)}
            />
            <div className="flex flex-col gap-2.5">
              <span className="label-base">Inferred k</span>
              <div className="input-base flex items-center py-2.5 text-textMain">{inferredK}</div>
            </div>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <SecondaryButton
              fullWidth={false}
              className="rounded-md px-4 py-2 text-xs"
              onClick={() => loadPetersen()}
            >
              Load Petersen (3,5)
            </SecondaryButton>
          </div>

          <SettingGroup>
            <div className="mt-4">
              <label className="label-base mb-2 block normal-case tracking-normal">
                Auto Step Delay:&nbsp;<span>{autoDelay}ms</span>
              </label>
              <SingleRangeSlider
                id="autoDelay"
                min={150}
                max={2000}
                step={50}
                value={autoDelay}
                onChange={(value) => setAutoDelay(value)}
              />
            </div>
          </SettingGroup>

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
    </div>
  );
};

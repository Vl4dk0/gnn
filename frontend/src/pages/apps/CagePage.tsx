import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { EditorPlaceholder } from "../../components/graph/EditorPlaceholder";
import { GraphCanvas } from "../../components/graph/GraphCanvas";
import { GraphToolbar } from "../../components/graph/GraphToolbar";
import { SettingsModal } from "../../components/graph/SettingsModal";
import { BackButton } from "../../components/ui/BackButton";
import { InputField } from "../../components/ui/InputField";
import { PrimaryButton } from "../../components/ui/PrimaryButton";
import { SecondaryButton } from "../../components/ui/SecondaryButton";
import { SelectField } from "../../components/ui/SelectField";
import { SettingGroup } from "../../components/ui/SettingGroup";
import { SingleRangeSlider } from "../../components/ui/SingleRangeSlider";
import { StatusPanel } from "../../components/ui/StatusPanel";
import { useCageGeneration } from "../../hooks/useCageGeneration";
import type { CageMethodId, CageSettings } from "../../types/api";
import {
  FROM_PARAM_TO_GROUP,
  GROUP_DEFAULTS,
  GROUP_LABELS,
  methodsForGroup
} from "./cageMethods";
import type { CageMethodGroup } from "./cageMethods";

const ALL_GROUPS: CageMethodGroup[] = ["classical", "rl", "voltage"];

export const CagePage = () => {
  const [searchParams] = useSearchParams();
  const fromParam = searchParams.get("from") ?? "";
  const contextGroup: CageMethodGroup | null = FROM_PARAM_TO_GROUP[fromParam] ?? null;

  const {
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
  } = useCageGeneration();

  // When a ?from context is present, initialize the method to that group's
  // default on mount (overrides whatever was persisted in localStorage).
  const contextInitializedRef = useRef(false);
  useEffect(() => {
    if (contextGroup && !contextInitializedRef.current) {
      contextInitializedRef.current = true;
      const defaultMethod = GROUP_DEFAULTS[contextGroup];
      setSettings((current) => ({ ...current, methodId: defaultMethod }));
    }
    // Only run once per mount; contextGroup is stable for the page lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [draftSettings, setDraftSettings] = useState<CageSettings>(settings);
  const [panelOpen, setPanelOpen] = useState(false);
  const [hasGraph, setHasGraph] = useState(false);

  const handleGraphChange = useCallback((edgeList: string) => {
    setHasGraph(edgeList.trim().length > 0);
  }, []);

  useEffect(() => {
    if (panelOpen) {
      document.body.classList.add("touch-graph-lock");
      return () => {
        document.body.classList.remove("touch-graph-lock");
      };
    }

    document.body.classList.remove("touch-graph-lock");
  }, [panelOpen]);

  // Generation streams nodes onto the canvas via updateFromEdgeList, which does
  // not fire onGraphChange, so watch the status snapshot to hide the placeholder
  // once a graph is produced. Clearing the canvas fires onGraphChange with an
  // empty list (setting hasGraph false), and the frozen status does not undo it.
  useEffect(() => {
    if ((status?.num_nodes ?? 0) > 0) {
      setHasGraph(true);
    }
  }, [status]);

  const triggerSettings = () => {
    setDraftSettings(settings);
    setSettingsOpen(true);
  };

  const primaryLabel =
    isMooreBoundOverLimit && currentMooreBound !== null
      ? `${currentMooreBound} > ${mooreBoundLimit}`
      : isGenerating
        ? "Generating..."
        : isSteppedMode
          ? "Start Inspection"
          : "Generate Cage";

  const canvasEmpty = !hasGraph;

  // Back button destination depends on ?from.
  const backHref =
    contextGroup === "rl"
      ? "/cage/rl"
      : contextGroup === "voltage"
        ? "/cage/voltage"
        : "/cage/astar";

  // Method dropdown: show only the context group when ?from is set, else all groups.
  const renderMethodOptions = () => {
    if (contextGroup) {
      return methodsForGroup(contextGroup).map((m) => (
        <option key={m.id} value={m.id} disabled={disabledMethods.has(m.id)}>
          {m.label}
        </option>
      ));
    }

    return ALL_GROUPS.map((group) => (
      <optgroup key={group} label={GROUP_LABELS[group]}>
        {methodsForGroup(group).map((m) => (
          <option key={m.id} value={m.id} disabled={disabledMethods.has(m.id)}>
            {m.label}
          </option>
        ))}
      </optgroup>
    ));
  };

  return (
    <div className="relative h-dvh overflow-hidden bg-transparent">
      <GraphCanvas onReady={onEditorReady} onGraphChange={handleGraphChange} canvasClassName="rounded-none">
        <BackButton
          href={backHref}
          label="Back to Docs"
          iconOnly
          className="absolute left-5 top-5 z-20"
        />

        <button
          type="button"
          className="absolute left-4 top-1/2 z-20 hidden min-w-[150px] -translate-y-1/2 rounded-xl border border-line2 bg-bg1/92 px-4 py-3 text-left shadow-card backdrop-blur-sm max-[900px]:left-5 max-[900px]:top-[72px] max-[900px]:min-w-0 max-[900px]:translate-y-0 max-[900px]:px-4 max-[900px]:py-3 max-[900px]:block"
          onClick={() => setPanelOpen(true)}
          aria-label="Open cage config"
        >
          <span className="hidden text-sm font-semibold leading-5 text-textMain min-[901px]:block">
            Status & Controls
          </span>
          <span className="block text-sm font-semibold uppercase tracking-[0.8px] text-textMain max-[900px]:block min-[901px]:hidden">
            Config
          </span>
          <span className="mt-1 block text-sm leading-5 text-textMuted max-[900px]:mt-0.5 max-[900px]:text-xs">
            {hasActiveSession ? "Session active" : "Ready to generate"}
          </span>
        </button>

        <section className="absolute left-5 top-1/2 z-20 w-[344px] -translate-y-1/2 rounded-2xl border border-line2 bg-bg1/92 p-4 shadow-card backdrop-blur-md max-[900px]:hidden">
          <div className="mt-4 grid grid-cols-2 gap-3">
            <InputField
              id="degreeK"
              label="Degree (k)"
              type="number"
              min={2}
              max={10}
              value={degreeK}
              className="py-2.5"
              onChange={(event) => setDegreeK(Number.parseInt(event.target.value, 10) || 0)}
            />
            <InputField
              id="girthG"
              label="Girth (g)"
              type="number"
              min={3}
              max={10}
              value={girthG}
              className="py-2.5"
              onChange={(event) => setGirthG(Number.parseInt(event.target.value, 10) || 0)}
            />
          </div>

          {isMooreBoundOverLimit && (
            <p className="mt-3 text-xs italic text-textDim">
              Moore bound must stay below {mooreBoundLimit}.
            </p>
          )}

          <div className="mt-4 rounded-xl border border-line2 bg-bg2/75 p-3">
            <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.8px] text-textDim">
              Status
            </div>
            <StatusPanel
              status={status}
              error={error}
              successMessage={successMessage}
              stoppedByUser={stoppedByUser}
            />
          </div>
        </section>

        {panelOpen && (
          <div
            className="absolute inset-0 z-30 hidden bg-black/70 p-3 backdrop-blur-sm max-[900px]:block"
            onClick={(event) => {
              if (event.target === event.currentTarget) {
                setPanelOpen(false);
              }
            }}
          >
            <section className="absolute inset-x-5 top-[72px] rounded-2xl border border-line2 bg-bg1 p-4 shadow-card">
              <div className="flex items-start justify-between gap-3">
                <button
                  type="button"
                  className="ui-action ml-auto rounded-md border border-line2 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.8px] text-textMain"
                  onClick={() => setPanelOpen(false)}
                >
                  Close
                </button>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3">
                <InputField
                  id="degreeKMobile"
                  label="Degree (k)"
                  type="number"
                  min={2}
                  max={10}
                  value={degreeK}
                  className="py-2.5"
                  onChange={(event) => setDegreeK(Number.parseInt(event.target.value, 10) || 0)}
                />
                <InputField
                  id="girthGMobile"
                  label="Girth (g)"
                  type="number"
                  min={3}
                  max={10}
                  value={girthG}
                  className="py-2.5"
                  onChange={(event) => setGirthG(Number.parseInt(event.target.value, 10) || 0)}
                />
              </div>

              {isMooreBoundOverLimit && (
                <p className="mt-3 text-xs italic text-textDim">
                  Moore bound must stay below {mooreBoundLimit}.
                </p>
              )}

              <div className="mt-4 rounded-xl border border-line2 bg-bg2/75 p-3">
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.8px] text-textDim">
                  Status
                </div>
                <StatusPanel
                  status={status}
                  error={error}
                  successMessage={successMessage}
                  stoppedByUser={stoppedByUser}
                />
              </div>
            </section>
          </div>
        )}

        <EditorPlaceholder
          visible={canvasEmpty}
          showControls={false}
          intro="The cage generator. Set the degree k and girth g, pick a construction method, then press Generate to watch a (k,g)-graph get built. You can also import a graph to start from."
        />

        <GraphToolbar
          onOpenSettings={triggerSettings}
          onClear={clearCanvas}
          settingsTitle="Generation Settings"
          clearTitle="Clear Graph"
          onDownload={downloadGraph}
          downloadTitle="Download Graph"
          canDownload={canDownload}
        />

        <div className="pointer-events-none absolute inset-x-0 bottom-6 z-20 flex justify-center px-4 max-[900px]:bottom-4">
          <div className="pointer-events-auto flex gap-3">
            {!isSteppedMode || !hasActiveSession ? (
              <PrimaryButton
                fullWidth={false}
                className="min-w-[220px] rounded-full bg-bg1/92 px-7 py-3 text-sm tracking-[0.8px] backdrop-blur-sm max-[900px]:min-w-[200px]"
                onClick={() => start()}
                disabled={isGenerating || isMooreBoundOverLimit || hasActiveSession}
              >
                {primaryLabel}
              </PrimaryButton>
            ) : (
              <>
                <PrimaryButton
                  fullWidth={false}
                  className="rounded-full bg-bg1/92 px-6 py-3 text-sm tracking-[0.8px] backdrop-blur-sm"
                  onClick={() => stepOnce()}
                  disabled={!canStep || isStepping}
                >
                  Step
                </PrimaryButton>
                <SecondaryButton
                  fullWidth={false}
                  className="rounded-full bg-bg1/92 px-6 py-3 text-sm tracking-[0.8px] backdrop-blur-sm"
                  onClick={() => (isAutoStepping ? pauseAutoStepping() : startAutoStepping())}
                >
                  {isAutoStepping ? "Pause" : "Auto Step"}
                </SecondaryButton>
              </>
            )}
            {hasActiveSession && (
              <SecondaryButton
                fullWidth={false}
                className="rounded-full bg-bg1/92 px-6 py-3 text-sm tracking-[0.8px] backdrop-blur-sm"
                onClick={() => stop()}
              >
                Stop
              </SecondaryButton>
            )}
          </div>
        </div>
      </GraphCanvas>

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        title="Generation Settings"
      >
        <SelectField
          id="methodId"
          label="Algorithm"
          value={draftSettings.methodId}
          onChange={(event) => {
            setDraftSettings((current) => ({
              ...current,
              methodId: event.target.value as CageMethodId
            }));
          }}
        >
          {renderMethodOptions()}
        </SelectField>
        <p className="mt-1 text-xs text-textDim">Which construction algorithm to run.</p>

        <SelectField
          id="executionMode"
          label="Execution Mode"
          value={draftSettings.executionMode}
          onChange={(event) => {
            setDraftSettings((current) => ({
              ...current,
              executionMode: event.target.value as CageSettings["executionMode"]
            }));
          }}
        >
          <option value="async">Fast Search</option>
          <option value="stepped">Step Inspection</option>
        </SelectField>
        <p className="mt-1 text-xs text-textDim">
          Fast search runs to the end. Step inspection lets you watch each move.
        </p>

        {!(draftSettings.executionMode === "stepped") && (
          <SettingGroup>
            <label className="label-base mb-2 block normal-case tracking-normal">
              Update Interval:&nbsp;
              <span>{draftSettings.pollingInterval}ms</span>
            </label>
            <SingleRangeSlider
              id="pollingInterval"
              min={50}
              max={2000}
              step={50}
              value={draftSettings.pollingInterval}
              onChange={(value) => {
                setDraftSettings((current) => ({
                  ...current,
                  pollingInterval: value
                }));
              }}
            />
            <p className="mt-1 text-xs text-textDim">How often the view refreshes during a fast search.</p>
          </SettingGroup>
        )}

        {draftSettings.executionMode === "stepped" && (
          <>
            <SettingGroup>
              <label className="label-base mb-2 block normal-case tracking-normal">
                Steps per Tick:&nbsp;
                <span>{draftSettings.stepsPerTick}</span>
              </label>
              <SingleRangeSlider
                id="stepsPerTick"
                min={1}
                max={100}
                step={1}
                value={draftSettings.stepsPerTick}
                onChange={(value) => {
                  setDraftSettings((current) => ({
                    ...current,
                    stepsPerTick: Math.round(value)
                  }));
                }}
              />
              <p className="mt-1 text-xs text-textDim">How many moves each manual or auto step performs.</p>
            </SettingGroup>

            <SettingGroup>
              <label className="label-base mb-2 block normal-case tracking-normal">
                Auto Step Interval:&nbsp;
                <span>{draftSettings.autoStepInterval}ms</span>
              </label>
              <SingleRangeSlider
                id="autoStepInterval"
                min={50}
                max={2000}
                step={50}
                value={draftSettings.autoStepInterval}
                onChange={(value) => {
                  setDraftSettings((current) => ({
                    ...current,
                    autoStepInterval: value
                  }));
                }}
              />
              <p className="mt-1 text-xs text-textDim">Delay between automatic steps.</p>
            </SettingGroup>
          </>
        )}

        <SettingGroup>
          <label className="flex cursor-pointer items-center gap-2 text-[0.95em] text-textMuted">
            <input
              type="checkbox"
              checked={draftSettings.enablePhysics}
              onChange={(event) => {
                setDraftSettings((current) => ({
                  ...current,
                  enablePhysics: event.target.checked
                }));
              }}
            />
            Enable Spring Physics
          </label>
          <p className="mt-1 text-xs text-textDim">Spring layout that spreads the graph out.</p>
        </SettingGroup>

        <div className="mt-8 flex items-center justify-start gap-5">
          <PrimaryButton
            fullWidth={false}
            className="max-w-[150px] flex-1 px-5 py-3 text-sm"
            onClick={() => saveSettings(draftSettings)}
          >
            Save
          </PrimaryButton>
          <SecondaryButton
            fullWidth={false}
            className="max-w-[150px] flex-1 px-5 py-3 text-sm"
            onClick={() => setSettingsOpen(false)}
          >
            Cancel
          </SecondaryButton>
        </div>
      </SettingsModal>
    </div>
  );
};

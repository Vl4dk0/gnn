import { useEffect, useMemo, useState } from "react";

import { GraphCanvas } from "../../components/graph/GraphCanvas";
import { GraphToolbar } from "../../components/graph/GraphToolbar";
import { PredictionLegend } from "../../components/graph/PredictionLegend";
import { SettingsModal } from "../../components/graph/SettingsModal";
import { BackButton } from "../../components/ui/BackButton";
import { DualRangeSlider } from "../../components/ui/DualRangeSlider";
import { PrimaryButton } from "../../components/ui/PrimaryButton";
import { SecondaryButton } from "../../components/ui/SecondaryButton";
import { SelectField } from "../../components/ui/SelectField";
import { SettingGroup } from "../../components/ui/SettingGroup";
import { usePredictionGraph } from "../../hooks/usePredictionGraph";
import type { DegreeMinCycleSettings } from "../../types/api";
import type { PredictionTask } from "../../services/models";

interface PredictionPageProps {
  task: PredictionTask;
}

export const PredictionPage = ({ task }: PredictionPageProps) => {
  const {
    settings,
    settingsOpen,
    modelOptions,
    isGenerating,
    error,
    onEditorReady,
    onEditorGraphChange,
    onEditorAnalyzeRequest,
    setSettingsOpen,
    saveSettings,
    generateRandomGraph,
    clearCanvas
  } = usePredictionGraph(task);

  const [draftSettings, setDraftSettings] = useState<DegreeMinCycleSettings>(settings);

  useEffect(() => {
    if (settingsOpen) {
      setDraftSettings(settings);
    }
  }, [settings, settingsOpen]);

  const probDisplay = useMemo(() => {
    return `${draftSettings.minProb.toFixed(2)} - ${draftSettings.maxProb.toFixed(2)}`;
  }, [draftSettings.maxProb, draftSettings.minProb]);

  const backHref = task === "degree" ? "/docs/module-degree" : "/docs/module-min-cycle";

  return (
    <div className="relative h-dvh overflow-hidden bg-transparent">
      <GraphCanvas
        onReady={onEditorReady}
        onGraphChange={onEditorGraphChange}
        onAnalyzeRequest={onEditorAnalyzeRequest}
        canvasClassName="rounded-none"
      >
        <PredictionLegend />
        <BackButton href={backHref} iconOnly className="absolute left-5 top-5 z-20" />

        {error && (
          <div className="absolute left-1/2 top-5 z-20 w-[min(520px,calc(100%-8rem))] -translate-x-1/2 rounded-xl bg-[#a52] p-3 text-sm text-white shadow-card max-[760px]:top-[72px] max-[760px]:w-[calc(100%-2rem)]">
            <strong>Error:</strong> {error}
          </div>
        )}

        <div className="pointer-events-none absolute inset-x-0 bottom-6 z-20 flex justify-center px-4 max-[900px]:bottom-4">
          <PrimaryButton
            fullWidth={false}
            className="pointer-events-auto min-w-[220px] rounded-full bg-bg1/92 px-7 py-3 text-sm tracking-[0.8px] backdrop-blur-sm max-[900px]:min-w-[200px]"
            onClick={() => generateRandomGraph()}
            disabled={isGenerating}
          >
            Generate
          </PrimaryButton>
        </div>

        <GraphToolbar
          onOpenSettings={() => setSettingsOpen(true)}
          onClear={clearCanvas}
          settingsTitle="Graph Settings"
          clearTitle="Delete Graph"
        />
      </GraphCanvas>

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        title="Graph Generation Settings"
      >
        <SettingGroup>
          <label className="label-base mb-2 block normal-case tracking-normal">
            Number of Nodes:&nbsp;
            <span>
              {draftSettings.minNodes} - {draftSettings.maxNodes}
            </span>
          </label>
          <DualRangeSlider
            minId="minNodes"
            maxId="maxNodes"
            highlightId="nodeRangeHighlight"
            min={3}
            max={20}
            minValue={draftSettings.minNodes}
            maxValue={draftSettings.maxNodes}
            onMinChange={(value) => {
              setDraftSettings((current) => ({
                ...current,
                minNodes: Math.min(Math.round(value), current.maxNodes)
              }));
            }}
            onMaxChange={(value) => {
              setDraftSettings((current) => ({
                ...current,
                maxNodes: Math.max(Math.round(value), current.minNodes)
              }));
            }}
          />
        </SettingGroup>

        <SettingGroup>
          <label className="label-base mb-2 block normal-case tracking-normal">
            Edge Probability:&nbsp;
            <span>{probDisplay}</span>
          </label>
          <DualRangeSlider
            minId="minProb"
            maxId="maxProb"
            highlightId="probRangeHighlight"
            min={0}
            max={100}
            minValue={Math.round(draftSettings.minProb * 100)}
            maxValue={Math.round(draftSettings.maxProb * 100)}
            onMinChange={(value) => {
              setDraftSettings((current) => ({
                ...current,
                minProb: Math.min(value / 100, current.maxProb)
              }));
            }}
            onMaxChange={(value) => {
              setDraftSettings((current) => ({
                ...current,
                maxProb: Math.max(value / 100, current.minProb)
              }));
            }}
          />
        </SettingGroup>

        <SettingGroup>
          <label className="flex cursor-pointer items-center gap-2 text-[0.95em] text-textMuted">
            <input
              type="checkbox"
              checked={draftSettings.allowSelfLoops}
              onChange={(event) => {
                setDraftSettings((current) => ({
                  ...current,
                  allowSelfLoops: event.target.checked
                }));
              }}
            />
            Allow Self-Loops
          </label>
        </SettingGroup>

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
        </SettingGroup>

        <SelectField
          id="modelSelect"
          label="GNN Model"
          value={draftSettings.modelId ?? ""}
          onChange={(event) => {
            setDraftSettings((current) => ({
              ...current,
              modelId: event.target.value || null
            }));
          }}
        >
          {modelOptions.map((option) => (
            <option key={option.value || "auto"} value={option.value}>
              {option.label}
            </option>
          ))}
        </SelectField>

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

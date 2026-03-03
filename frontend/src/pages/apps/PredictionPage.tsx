import { useEffect, useMemo, useState } from "react";

import { GraphCanvas } from "../../components/graph/GraphCanvas";
import { PredictionLegend } from "../../components/graph/PredictionLegend";
import { GraphPageLayout } from "../../components/graph/GraphPageLayout";
import { SettingsModal } from "../../components/graph/SettingsModal";
import { GraphToolbar } from "../../components/graph/GraphToolbar";
import { BackButton } from "../../components/ui/BackButton";
import { ControlsSection } from "../../components/ui/ControlsSection";
import { DualRangeSlider } from "../../components/ui/DualRangeSlider";
import { PrimaryButton } from "../../components/ui/PrimaryButton";
import { SecondaryButton } from "../../components/ui/SecondaryButton";
import { SelectField } from "../../components/ui/SelectField";
import { SettingGroup } from "../../components/ui/SettingGroup";
import { TextAreaField } from "../../components/ui/TextAreaField";
import { usePredictionGraph } from "../../hooks/usePredictionGraph";
import { useSidebarDrawer } from "../../hooks/useSidebarDrawer";
import type { DegreeMinCycleSettings } from "../../types/api";
import type { PredictionTask } from "../../services/models";

interface PredictionPageProps {
  task: PredictionTask;
}

const defaultPlaceholder = `0 1
0 2
1 2
1 3
2 3
4`;

export const PredictionPage = ({ task }: PredictionPageProps) => {
  const drawer = useSidebarDrawer();
  const {
    graphInput,
    settings,
    settingsOpen,
    modelOptions,
    isGenerating,
    isAnalyzing,
    error,
    controls,
    onEditorReady,
    onEditorGraphChange,
    onEditorAnalyzeRequest,
    setSettingsOpen,
    saveSettings,
    generateRandomGraph,
    analyzeGraph,
    clearCanvas,
    onTextInputChange
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

  return (
    <GraphPageLayout
      navOpen={drawer.isOpen}
      onToggleNav={drawer.toggle}
      onCloseNav={drawer.close}
      sidebar={
        <>
          <div className="contents max-[900px]:flex max-[900px]:flex-col max-[900px]:gap-3">
            <TextAreaField
              id="graphInput"
              label="Graph Input"
              placeholder={defaultPlaceholder}
              value={graphInput}
              onChange={(event) => onTextInputChange(event.target.value)}
            />

            {error && (
              <div className="rounded-md border border-[#8a3d3d] bg-[#2f1c1c] px-3 py-2 text-sm text-[#ffc9c9]">
                {error}
              </div>
            )}

            <ControlsSection />
          </div>

          <PrimaryButton
            onClick={() => generateRandomGraph()}
            disabled={isGenerating || isAnalyzing}
          >
            {controls.generateButtonLabel}
          </PrimaryButton>

          <PrimaryButton onClick={() => analyzeGraph()} disabled={isAnalyzing || isGenerating}>
            {isAnalyzing ? "Analyzing..." : controls.analyzeButtonLabel}
          </PrimaryButton>

          <BackButton />
        </>
      }
    >
      <GraphCanvas
        onReady={onEditorReady}
        onGraphChange={onEditorGraphChange}
        onAnalyzeRequest={onEditorAnalyzeRequest}
      >
        <PredictionLegend />
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

        <div className="mt-8 flex gap-5">
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
    </GraphPageLayout>
  );
};

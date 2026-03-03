import { useState } from "react";

import { GraphCanvas } from "../../components/graph/GraphCanvas";
import { GraphPageLayout } from "../../components/graph/GraphPageLayout";
import { GraphToolbar } from "../../components/graph/GraphToolbar";
import { SettingsModal } from "../../components/graph/SettingsModal";
import { BackButton } from "../../components/ui/BackButton";
import { ControlsSection } from "../../components/ui/ControlsSection";
import { InputField } from "../../components/ui/InputField";
import { PrimaryButton } from "../../components/ui/PrimaryButton";
import { SecondaryButton } from "../../components/ui/SecondaryButton";
import { SelectField } from "../../components/ui/SelectField";
import { SettingGroup } from "../../components/ui/SettingGroup";
import { SingleRangeSlider } from "../../components/ui/SingleRangeSlider";
import { StatusPanel } from "../../components/ui/StatusPanel";
import { useCageGeneration } from "../../hooks/useCageGeneration";
import { useSidebarDrawer } from "../../hooks/useSidebarDrawer";
import type { CageSettings } from "../../types/api";

export const CagePage = () => {
  const drawer = useSidebarDrawer();

  const {
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
    currentMooreBound,
    mooreBoundLimit,
    isMooreBoundOverLimit,
    onEditorReady,
    start,
    stop,
    clearCanvas
  } = useCageGeneration();

  const [draftSettings, setDraftSettings] = useState<CageSettings>(settings);

  return (
    <GraphPageLayout
      navOpen={drawer.isOpen}
      onToggleNav={drawer.toggle}
      onCloseNav={drawer.close}
      sidebar={
        <>
          <div className="contents max-[900px]:flex max-[900px]:flex-col max-[900px]:gap-3">
            <InputField
              id="degreeK"
              label="Degree (k)"
              type="number"
              min={2}
              max={10}
              value={degreeK}
              onChange={(event) => setDegreeK(Number.parseInt(event.target.value, 10) || 0)}
            />

            <InputField
              id="girthG"
              label="Girth (g)"
              type="number"
              min={3}
              max={10}
              value={girthG}
              onChange={(event) => setGirthG(Number.parseInt(event.target.value, 10) || 0)}
            />

            <PrimaryButton onClick={() => start()} disabled={isGenerating || isMooreBoundOverLimit}>
              {isMooreBoundOverLimit && currentMooreBound !== null
                ? `${currentMooreBound} > ${mooreBoundLimit}`
                : isGenerating
                  ? "Generating..."
                  : "Generate Cage"}
            </PrimaryButton>
            {isMooreBoundOverLimit && (
              <p className="mt-2 text-xs italic text-textDim">
                *Moore's bound must be smaller than {mooreBoundLimit}
              </p>
            )}

            {isGenerating && (
              <SecondaryButton onClick={() => stop()} fullWidth>
                Stop
              </SecondaryButton>
            )}

            <div className="mt-5">
              <label className="label-base mb-2 block">Status</label>
              <div className="min-h-[100px] rounded-md border-2 border-line2 bg-bg1 p-3 text-[0.9em] text-textMuted">
                <StatusPanel
                  status={status}
                  error={error}
                  successMessage={successMessage}
                  stoppedByUser={stoppedByUser}
                />
              </div>
            </div>

            <ControlsSection />
          </div>

          <BackButton />
        </>
      }
    >
      <GraphCanvas onReady={onEditorReady}>
        <GraphToolbar
          onOpenSettings={() => {
            setDraftSettings(settings);
            setSettingsOpen(true);
          }}
          onClear={clearCanvas}
          settingsTitle="Generation Settings"
          clearTitle="Clear Graph"
        />
      </GraphCanvas>

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        title="Generation Settings"
      >
        <SelectField
          id="generatorType"
          label="Generator Algorithm"
          value={draftSettings.generatorType}
          onChange={(event) => {
            setDraftSettings((current) => ({
              ...current,
              generatorType: event.target.value as CageSettings["generatorType"]
            }));
          }}
        >
          <option value="randomwalk">Random Walk (Fast, Probabilistic)</option>
          <option value="bruteforce">Bruteforce (Systematic, Backtracking)</option>
          <option value="astar">A* Search (Best-First, Optimal)</option>
          <option value="rl">RL Agent (GNN-Guided)</option>
        </SelectField>

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
          <div className="mt-2 text-[0.85em] text-textDim">
            How often frontend checks for updates (50ms - 2000ms)
          </div>
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

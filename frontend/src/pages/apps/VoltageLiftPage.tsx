import { useCallback, useEffect, useRef, useState } from "react";

import { EditorPlaceholder } from "../../components/graph/EditorPlaceholder";
import { GraphCanvas } from "../../components/graph/GraphCanvas";
import { SettingsModal } from "../../components/graph/SettingsModal";
import { BackButton } from "../../components/ui/BackButton";
import { IconButton } from "../../components/ui/IconButton";
import { InputField } from "../../components/ui/InputField";
import { PrimaryButton } from "../../components/ui/PrimaryButton";
import { SecondaryButton } from "../../components/ui/SecondaryButton";
import { SettingGroup } from "../../components/ui/SettingGroup";
import type { InteractiveGraphEditor } from "../../graph/InteractiveGraphEditor";
import { VoltageBaseEditor } from "../../graph/voltage/VoltageBaseEditor";
import {
  buildLiftEdgeList,
  buildLiftLabels,
  computeGirth,
  liftVertexCount,
  type BaseArc
} from "../../graph/voltage/liftConstruction";

interface Preset {
  id: string;
  label: string;
  n: number;
  nodeIds: number[];
  arcs: BaseArc[];
}

// Base of 2 nodes over Z_5: a self-loop on node 0 (voltage 1, the outer
// 5-cycle), a self-loop on node 1 (voltage 2, the inner pentagram), and one
// connecting arc 0 -> 1 (voltage 0, the spokes). The lift has 2 * 5 = 10
// vertices, is 3-regular, has girth 5: the Petersen graph, the (3,5)-cage.
const PRESETS: Preset[] = [
  {
    id: "petersen",
    label: "Two loops + Z₅ → Petersen (3,5)",
    n: 5,
    nodeIds: [0, 1],
    arcs: [
      { id: 0, from: 0, to: 0, voltage: 1 },
      { id: 1, from: 1, to: 1, voltage: 2 },
      { id: 2, from: 0, to: 1, voltage: 0 }
    ]
  },
  {
    id: "triangle",
    label: "Triangle + Z₅",
    n: 5,
    nodeIds: [0, 1, 2],
    arcs: [
      { id: 0, from: 0, to: 1, voltage: 1 },
      { id: 1, from: 1, to: 2, voltage: 1 },
      { id: 2, from: 2, to: 0, voltage: 1 }
    ]
  }
];

interface LiftStats {
  vertices: number;
  girth: number;
}

export const VoltageLiftPage = () => {
  const baseCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const editorRef = useRef<VoltageBaseEditor | null>(null);
  const previewRef = useRef<InteractiveGraphEditor | null>(null);

  const [groupOrder, setGroupOrder] = useState(7);
  const [vertexCount, setVertexCount] = useState(0);
  const [stats, setStats] = useState<LiftStats | null>(null);
  const [hasBase, setHasBase] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [physicsEnabled, setPhysicsEnabled] = useState(true);

  const applyPhysics = (editor: InteractiveGraphEditor, enabled: boolean) => {
    if (enabled) {
      editor.enablePhysics();
    } else {
      editor.disablePhysics();
    }
  };

  const handlePhysicsChange = (enabled: boolean) => {
    setPhysicsEnabled(enabled);
    const preview = previewRef.current;
    if (preview) {
      applyPhysics(preview, enabled);
    }
  };

  const groupOrderRef = useRef(groupOrder);
  useEffect(() => {
    groupOrderRef.current = groupOrder;
  }, [groupOrder]);

  const refreshCount = useCallback(() => {
    const editor = editorRef.current;
    if (!editor) {
      return;
    }
    const nodeCount = editor.getNodeIds().length;
    setHasBase(nodeCount > 0);
    setVertexCount(liftVertexCount(editor.getNodeIds(), editor.getGroupOrder()));
  }, []);

  useEffect(() => {
    const canvas = baseCanvasRef.current;
    if (!canvas) {
      return;
    }

    const editor = new VoltageBaseEditor(canvas, { onChange: () => refreshCount() });
    editor.setGroupOrder(groupOrderRef.current);
    editorRef.current = editor;
    refreshCount();

    return () => {
      editor.destroy();
      editorRef.current = null;
    };
  }, [refreshCount]);

  const handleGroupOrderChange = (value: number) => {
    const clamped = Number.isNaN(value) ? 2 : Math.max(2, Math.min(40, value));
    setGroupOrder(clamped);
    editorRef.current?.setGroupOrder(clamped);
  };

  const generateLift = useCallback(() => {
    const editor = editorRef.current;
    const preview = previewRef.current;
    if (!editor || !preview) {
      return;
    }
    const n = editor.getGroupOrder();
    const edgeList = buildLiftEdgeList(editor.getNodeIds(), editor.getArcs(), n);
    preview.loadFromEdgeList(edgeList);
    preview.setNodeLabels(buildLiftLabels(editor.getNodeIds(), n));
    setStats({
      vertices: liftVertexCount(editor.getNodeIds(), n),
      girth: computeGirth(edgeList)
    });
  }, []);

  const loadPreset = (preset: Preset) => {
    const editor = editorRef.current;
    if (!editor) {
      return;
    }
    setGroupOrder(preset.n);
    editor.setGroupOrder(preset.n);
    editor.loadPreset(
      preset.nodeIds,
      preset.arcs.map((arc) => ({ ...arc }))
    );
    refreshCount();
    setStats(null);
  };

  const clearBase = () => {
    editorRef.current?.clear();
    previewRef.current?.clear();
    setStats(null);
  };

  const voltageExtraControls = (
    <>
      <div className="flex gap-2 justify-center text-sm leading-relaxed">
        <span className="font-medium text-textMuted whitespace-nowrap">Select an edge:</span>
        <span className="text-textDim">click near its label</span>
      </div>
      <div className="flex gap-2 justify-center text-sm leading-relaxed">
        <span className="font-medium text-textMuted whitespace-nowrap">Change voltage:</span>
        <span className="text-textDim">plus / minus or arrow keys</span>
      </div>
      <div className="flex gap-2 justify-center text-sm leading-relaxed">
        <span className="font-medium text-textMuted whitespace-nowrap">Reverse an edge:</span>
        <span className="text-textDim">press r</span>
      </div>
      <div className="flex gap-2 justify-center text-sm leading-relaxed">
        <span className="font-medium text-textMuted whitespace-nowrap">Remove an edge:</span>
        <span className="text-textDim">press Delete</span>
      </div>
    </>
  );

  return (
    <div className="relative h-dvh overflow-hidden bg-transparent">
      <div className="flex h-full w-full flex-col md:flex-row">
        <div className="relative h-1/2 w-full border-b border-line2 md:h-full md:w-1/2 md:border-b-0 md:border-r">
          <canvas ref={baseCanvasRef} className="h-full w-full touch-none" />

          <BackButton
            href="/cage/voltage"
            label="Back to Voltage"
            iconOnly
            className="absolute left-5 top-5 z-20"
          />
        </div>

        <div className="relative h-1/2 w-full md:h-full md:w-1/2">
          <GraphCanvas
            onReady={(editor) => {
              previewRef.current = editor;
              if (editor) {
                applyPhysics(editor, physicsEnabled);
              }
            }}
            canvasClassName="rounded-none"
          >
            <div className="pointer-events-none absolute left-5 top-5 z-20 rounded-xl border border-line2 bg-bg1/92 px-3 py-2 text-xs text-textMuted shadow-card backdrop-blur-sm">
              Lift preview
            </div>

            <IconButton
              positionClassName="right-5 top-5"
              onClick={() => setSettingsOpen(true)}
              title="Preview Settings"
              aria-label="Preview Settings"
            >
              <svg
                className="h-6 w-6 fill-textMuted"
                viewBox="0 0 45.973 45.973"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path d="M43.454,18.443h-2.437c-0.453-1.766-1.16-3.42-2.082-4.933l1.752-1.756c0.473-0.473,0.733-1.104,0.733-1.774 c0-0.669-0.262-1.301-0.733-1.773l-2.92-2.917c-0.947-0.948-2.602-0.947-3.545-0.001l-1.826,1.815 C30.9,6.232,29.296,5.56,27.529,5.128V2.52c0-1.383-1.105-2.52-2.488-2.52h-4.128c-1.383,0-2.471,1.137-2.471,2.52v2.607 c-1.766,0.431-3.38,1.104-4.878,1.977l-1.825-1.815c-0.946-0.948-2.602-0.947-3.551-0.001L5.27,8.205 C4.802,8.672,4.535,9.318,4.535,9.978c0,0.669,0.259,1.299,0.733,1.772l1.752,1.76c-0.921,1.513-1.629,3.167-2.081,4.933H2.501 C1.117,18.443,0,19.555,0,20.935v4.125c0,1.384,1.117,2.471,2.501,2.471h2.438c0.452,1.766,1.159,3.43,2.079,4.943l-1.752,1.763 c-0.474,0.473-0.734,1.106-0.734,1.776s0.261,1.303,0.734,1.776l2.92,2.919c0.474,0.473,1.103,0.733,1.772,0.733 s1.299-0.261,1.773-0.733l1.833-1.816c1.498,0.873,3.112,1.545,4.878,1.978v2.604c0,1.383,1.088,2.498,2.471,2.498h4.128 c1.383,0,2.488-1.115,2.488-2.498v-2.605c1.767-0.432,3.371-1.104,4.869-1.977l1.817,1.812c0.474,0.475,1.104,0.735,1.775,0.735 c0.67,0,1.301-0.261,1.774-0.733l2.92-2.917c0.473-0.472,0.732-1.103,0.734-1.772c0-0.67-0.262-1.299-0.734-1.773l-1.75-1.77 c0.92-1.514,1.627-3.179,2.08-4.943h2.438c1.383,0,2.52-1.087,2.52-2.471v-4.125C45.973,19.555,44.837,18.443,43.454,18.443z M22.976,30.85c-4.378,0-7.928-3.517-7.928-7.852c0-4.338,3.55-7.85,7.928-7.85c4.379,0,7.931,3.512,7.931,7.85 C30.906,27.334,27.355,30.85,22.976,30.85z" />
              </svg>
            </IconButton>
          </GraphCanvas>
        </div>
      </div>

      <EditorPlaceholder
        visible={!hasBase}
        intro="The voltage-lift editor. Draw a small base graph, assign a voltage to each edge over the group Z_n, then press Generate lift to build the cover on the right."
        showControls
        extraControls={voltageExtraControls}
      />

      <section className="absolute left-5 top-1/2 z-20 w-[300px] -translate-y-1/2 rounded-2xl border border-line2 bg-bg1/92 p-4 shadow-card backdrop-blur-md max-md:left-3 max-md:w-[260px]">
        <div className="text-[11px] font-semibold uppercase tracking-[0.8px] text-textDim">
          Voltage Lift
        </div>

        <div className="mt-3">
          <InputField
            id="groupOrder"
            label="Group order n (Zₙ)"
            type="number"
            min={2}
            max={40}
            value={groupOrder}
            className="py-2.5"
            onChange={(event) => handleGroupOrderChange(Number.parseInt(event.target.value, 10))}
          />
        </div>

        <div className="mt-4 flex flex-col gap-2">
          <PrimaryButton
            fullWidth={false}
            className="w-full rounded-lg px-5 py-2.5 text-sm tracking-[0.8px]"
            onClick={generateLift}
          >
            Generate lift
          </PrimaryButton>
          <SecondaryButton
            fullWidth={false}
            className="w-full rounded-lg px-5 py-2.5 text-sm tracking-[0.8px]"
            onClick={clearBase}
          >
            Clear
          </SecondaryButton>
        </div>

        <div className="mt-4 rounded-xl border border-line2 bg-bg2/75 p-3 text-sm text-textMuted">
          <div>
            Lift vertices: <span className="font-semibold text-textMain">{vertexCount}</span>
          </div>
          {stats && (
            <div className="mt-1">
              Girth:{" "}
              <span className="font-semibold text-textMain">
                {stats.girth === 0 ? "acyclic" : stats.girth}
              </span>
            </div>
          )}
        </div>

        <div className="mt-4">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.8px] text-textDim">
            Presets
          </div>
          <div className="flex flex-col gap-2">
            {PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                className="ui-action rounded-lg border border-line2 bg-bg1 px-3 py-2 text-left text-xs font-semibold text-textMain transition-colors hover:border-textDim"
                onClick={() => loadPreset(preset)}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        title="Preview Settings"
      >
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

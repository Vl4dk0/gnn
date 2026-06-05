import { useCallback, useEffect, useRef, useState } from "react";

import { EditorPlaceholder } from "../../components/graph/EditorPlaceholder";
import { GraphCanvas } from "../../components/graph/GraphCanvas";
import { BackButton } from "../../components/ui/BackButton";
import { InputField } from "../../components/ui/InputField";
import { PrimaryButton } from "../../components/ui/PrimaryButton";
import { SecondaryButton } from "../../components/ui/SecondaryButton";
import type { InteractiveGraphEditor } from "../../graph/InteractiveGraphEditor";
import { VoltageBaseEditor } from "../../graph/voltage/VoltageBaseEditor";
import {
  buildLiftEdgeList,
  computeGirth,
  liftVertexCount,
  type BaseArc
} from "../../graph/voltage/liftConstruction";
import { importGraphFromFile } from "../../services/cage";

interface Preset {
  id: string;
  label: string;
  n: number;
  nodeIds: number[];
  arcs: BaseArc[];
}

// Dumbbell base (2 nodes, 3 parallel arcs 0 -> 1) with voltages 1, 2, 4 over
// Z_7 gives the Heawood graph: 14 vertices, 3-regular, girth 6 (the (3,6)-cage).
const PRESETS: Preset[] = [
  {
    id: "heawood",
    label: "Dumbbell + Z₇ → Heawood (3,6)",
    n: 7,
    nodeIds: [0, 1],
    arcs: [
      { id: 0, from: 0, to: 1, voltage: 1 },
      { id: 1, from: 0, to: 1, voltage: 2 },
      { id: 2, from: 0, to: 1, voltage: 4 }
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
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [groupOrder, setGroupOrder] = useState(7);
  const [vertexCount, setVertexCount] = useState(0);
  const [stats, setStats] = useState<LiftStats | null>(null);
  const [hasBase, setHasBase] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);

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

  const handleImportClick = () => {
    setImportError(null);
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!event.target.files) {
      return;
    }
    event.target.value = "";
    if (!file) {
      return;
    }
    try {
      const edgeListText = await importGraphFromFile(file);
      editorRef.current?.loadEdgeList(edgeListText);
      refreshCount();
      setStats(null);
      setImportError(null);
    } catch (err) {
      setImportError(err instanceof Error ? err.message : "Import failed");
    }
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
            }}
            canvasClassName="rounded-none"
          >
            <div className="pointer-events-none absolute left-5 top-5 z-20 rounded-xl border border-line2 bg-bg1/92 px-3 py-2 text-xs text-textMuted shadow-card backdrop-blur-sm">
              Lift preview
            </div>
          </GraphCanvas>
        </div>
      </div>

      <EditorPlaceholder
        visible={!hasBase}
        intro="The voltage-lift editor. Draw a small base graph, assign a voltage to each edge over the group Z_n, then press Generate lift to build the cover on the right."
        showControls
        extraControls={voltageExtraControls}
      />

      <input
        ref={fileInputRef}
        type="file"
        accept=".g6,.txt"
        className="hidden"
        onChange={handleFileChange}
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
            onClick={handleImportClick}
          >
            Import graph
          </SecondaryButton>
          <SecondaryButton
            fullWidth={false}
            className="w-full rounded-lg px-5 py-2.5 text-sm tracking-[0.8px]"
            onClick={clearBase}
          >
            Clear
          </SecondaryButton>
        </div>

        {importError && (
          <div className="mt-2 text-[11px] text-textDim">{importError}</div>
        )}

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
    </div>
  );
};

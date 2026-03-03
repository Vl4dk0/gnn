export const ControlsSection = () => {
  return (
    <div className="mt-5 min-h-0 flex-1 overflow-y-auto rounded-lg border border-line2 bg-bg1 p-5 max-[900px]:mt-0 max-[900px]:max-h-none max-[900px]:flex-none max-[900px]:overflow-visible">
      <h3 className="mb-4 text-[0.95em] font-semibold uppercase tracking-[0.5px] text-textMain">
        Desktop Controls
      </h3>
      <div className="space-y-1 text-[0.85em] leading-[1.8] text-textMuted">
        <div>
          <strong className="text-textMain">Double-click empty space:</strong> Create node
        </div>
        <div>
          <strong className="text-textMain">Single-click node:</strong> Select node
        </div>
        <div>
          <strong className="text-textMain">Drag node:</strong> Move it around
        </div>
        <div>
          <strong className="text-textMain">Drag empty space:</strong> Pan canvas
        </div>
        <div>
          <strong className="text-textMain">Right-click node:</strong> Start edge
        </div>
        <div>
          <strong className="text-textMain">Click node:</strong> Complete edge (toggle)
        </div>
        <div>
          <strong className="text-textMain">Middle-click node:</strong> Delete node
        </div>
        <div>
          <strong className="text-textMain">Backspace:</strong> Delete selected node
        </div>
        <div>
          <strong className="text-textMain">Mouse wheel:</strong> Zoom in/out
        </div>
      </div>

      <h3 className="mb-4 mt-[18px] text-[0.95em] font-semibold uppercase tracking-[0.5px] text-textMain">
        Mobile Gestures
      </h3>
      <div className="space-y-1 text-[0.85em] leading-[1.8] text-textMuted">
        <div>
          <strong className="text-textMain">Tap node:</strong> Select node / complete edge if edge
          mode is active
        </div>
        <div>
          <strong className="text-textMain">Tap empty space:</strong> Deselect or cancel edge mode
        </div>
        <div>
          <strong className="text-textMain">Drag node:</strong> Move node
        </div>
        <div>
          <strong className="text-textMain">Drag empty space:</strong> Pan canvas
        </div>
        <div>
          <strong className="text-textMain">Double-tap node:</strong> Start edge mode
        </div>
        <div>
          <strong className="text-textMain">Tap second node:</strong> Complete edge (toggle)
        </div>
        <div>
          <strong className="text-textMain">Double-tap empty space:</strong> Create node
        </div>
        <div>
          <strong className="text-textMain">Select node + long-press same node:</strong> Delete node
        </div>
        <div>
          <strong className="text-textMain">Pinch:</strong> Zoom in/out
        </div>
      </div>
    </div>
  );
};

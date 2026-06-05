import type { ReactNode } from "react";

interface ControlItem {
  action: string;
  gesture: string;
}

const DRAWING_CONTROLS_ITEMS: ControlItem[] = [
  { action: "Add node", gesture: "double-click empty space" },
  { action: "Remove node", gesture: "middle-click it, or select it and press Backspace" },
  { action: "Add edge", gesture: "right-click a node, then click another" },
  { action: "Remove edge", gesture: "draw the same edge again" },
  { action: "Move a node", gesture: "drag it" },
  { action: "Move the plane", gesture: "drag empty space" },
  { action: "Zoom", gesture: "scroll wheel" },
];

const DrawingControls = () => (
  <>
    {DRAWING_CONTROLS_ITEMS.map(({ action, gesture }) => (
      <div key={action} className="flex gap-2 justify-center text-sm leading-relaxed">
        <span className="font-medium text-textMuted whitespace-nowrap">{action}:</span>
        <span className="text-textDim">{gesture}</span>
      </div>
    ))}
  </>
);

interface EditorPlaceholderProps {
  visible: boolean;
  intro: ReactNode;
  showControls?: boolean;
  extraControls?: ReactNode;
}

export const EditorPlaceholder = ({
  visible,
  intro,
  showControls = false,
  extraControls,
}: EditorPlaceholderProps) => {
  if (!visible) return null;

  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center z-10 pointer-events-none">
      <div className="flex flex-col items-center gap-3 max-w-md text-center">
        <div className="text-textMuted text-sm leading-relaxed">{intro}</div>
        {showControls && (
          <div className="flex flex-col items-center gap-1 mt-2">
            <DrawingControls />
          </div>
        )}
        {extraControls && (
          <div className="flex flex-col items-center gap-1 text-sm leading-relaxed">
            {extraControls}
          </div>
        )}
      </div>
    </div>
  );
};

export { DrawingControls, DRAWING_CONTROLS_ITEMS };

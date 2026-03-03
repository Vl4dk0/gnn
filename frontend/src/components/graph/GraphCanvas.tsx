import { useEffect, useRef } from "react";

import { InteractiveGraphEditor } from "../../graph/InteractiveGraphEditor";

interface GraphCanvasProps {
  onReady: (editor: InteractiveGraphEditor | null) => void;
  onGraphChange?: (edgeList: string) => void;
  onAnalyzeRequest?: () => void;
  children?: React.ReactNode;
}

export const GraphCanvas = ({
  onReady,
  onGraphChange,
  onAnalyzeRequest,
  children
}: GraphCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const onReadyRef = useRef(onReady);
  const onGraphChangeRef = useRef(onGraphChange);
  const onAnalyzeRequestRef = useRef(onAnalyzeRequest);

  useEffect(() => {
    onReadyRef.current = onReady;
  }, [onReady]);

  useEffect(() => {
    onGraphChangeRef.current = onGraphChange;
  }, [onGraphChange]);

  useEffect(() => {
    onAnalyzeRequestRef.current = onAnalyzeRequest;
  }, [onAnalyzeRequest]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const editor = new InteractiveGraphEditor(canvas, {
      onGraphChange: (edgeList) => onGraphChangeRef.current?.(edgeList),
      onAnalyzeRequest: () => onAnalyzeRequestRef.current?.()
    });

    onReadyRef.current(editor);

    return () => {
      onReadyRef.current(null);
      editor.destroy();
    };
  }, []);

  return (
    <div className="relative flex h-full w-full items-center justify-center">
      <canvas ref={canvasRef} className="h-full w-full touch-none rounded-[10px]" />
      {children}
    </div>
  );
};

export const PredictionLegend = () => {
  return (
    <div className="absolute left-4 top-4 z-10 rounded border border-line2 bg-bg1/80 px-3 py-2 text-xs text-textMuted backdrop-blur-sm">
      <div>
        <span className="font-semibold text-[#00ff00]">actual</span>
        <span className="mx-1">/</span>
        <span className="font-semibold text-[#ff0000]">predicted</span>
      </div>
      <div className="mt-1 text-[11px] text-textDim">Shown when a prediction is incorrect.</div>
    </div>
  );
};

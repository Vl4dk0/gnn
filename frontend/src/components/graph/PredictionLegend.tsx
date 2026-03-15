export const PredictionLegend = () => {
  return (
    <div className="absolute bottom-4 left-4 z-10 rounded border border-line2 bg-bg1/80 px-3.5 py-2.5 text-[13px] text-textMuted backdrop-blur-sm max-[900px]:bottom-3 max-[900px]:left-3 max-[900px]:max-w-[calc(100%-88px)]">
      <div>
        <span className="font-semibold text-[#00ff00]">actual</span>
        <span className="mx-1">/</span>
        <span className="font-semibold text-[#ff0000]">predicted</span>
      </div>
      <div className="mt-1 text-xs text-textDim">Shown when a prediction is incorrect.</div>
    </div>
  );
};

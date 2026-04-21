export const PredictionLegend = () => {
  return (
    <div className="ui-floating-panel absolute bottom-4 left-4 z-10 rounded border border-line2 bg-bg1/80 px-3.5 py-2.5 text-[13px] text-textMuted backdrop-blur-sm max-[900px]:hidden">
      <div>
        <span className="font-semibold text-[#008800] dark:text-[#00ff00]">actual</span>
        <span className="mx-1">/</span>
        <span className="font-semibold text-[#dd0000] dark:text-[#ff0000]">predicted</span>
      </div>
    </div>
  );
};

interface DualRangeSliderProps {
  minId: string;
  maxId: string;
  highlightId: string;
  min: number;
  max: number;
  minValue: number;
  maxValue: number;
  step?: number;
  onMinChange: (value: number) => void;
  onMaxChange: (value: number) => void;
}

export const DualRangeSlider = ({
  minId,
  maxId,
  highlightId,
  min,
  max,
  minValue,
  maxValue,
  step = 1,
  onMinChange,
  onMaxChange
}: DualRangeSliderProps) => {
  const minPercent = ((minValue - min) / (max - min)) * 100;
  const maxPercent = ((maxValue - min) / (max - min)) * 100;

  return (
    <div className="range-slider-container">
      <div
        id={highlightId}
        className="range-track-highlight"
        style={{ left: `${minPercent}%`, width: `${maxPercent - minPercent}%` }}
      />
      <input
        id={minId}
        type="range"
        min={min}
        max={max}
        step={step}
        value={minValue}
        className="range-slider"
        onChange={(event) => onMinChange(Number.parseFloat(event.target.value))}
      />
      <input
        id={maxId}
        type="range"
        min={min}
        max={max}
        step={step}
        value={maxValue}
        className="range-slider"
        onChange={(event) => onMaxChange(Number.parseFloat(event.target.value))}
      />
    </div>
  );
};

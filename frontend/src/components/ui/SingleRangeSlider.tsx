interface SingleRangeSliderProps {
  id: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (value: number) => void;
}

export const SingleRangeSlider = ({
  id,
  min,
  max,
  step = 1,
  value,
  onChange
}: SingleRangeSliderProps) => {
  return (
    <input
      id={id}
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      className="single-slider"
      onChange={(event) => onChange(Number.parseInt(event.target.value, 10))}
    />
  );
};

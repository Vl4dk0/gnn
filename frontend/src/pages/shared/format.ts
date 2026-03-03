export const formatAccuracy = (value: number | null): string => {
  return value === null ? "-" : `${value.toFixed(2)}%`;
};

export const formatMetric = (value: number | null): string => {
  return value === null ? "-" : value.toFixed(4);
};

export const formatDate = (value: string | null): string => {
  if (!value) return "-";

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "-";
  }

  const yyyy = String(date.getFullYear());
  const mm = String(date.getMonth() + 1).padStart(2, "0");
  const dd = String(date.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
};

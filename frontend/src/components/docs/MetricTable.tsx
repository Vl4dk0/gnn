import { ReactNode, useMemo, useState } from "react";

export interface Column<T> {
  key: keyof T;
  header: string;
  render?: (value: any, row: T) => ReactNode;
  sortable?: boolean; // Defaults to true
  defaultDirection?: "asc" | "desc"; // Defaults to asc
  align?: "left" | "right" | "center";
}

interface MetricTableProps<T> {
  columns: Column<T>[];
  data: T[] | null;
  wide?: boolean;
  emptyMessage?: string;
  initialSort?: { key: keyof T; direction: "asc" | "desc" };
}

export const MetricTable = <T extends Record<string, any>>({
  columns,
  data,
  wide = false,
  emptyMessage = "No models found",
  initialSort,
}: MetricTableProps<T>) => {
  // Default sort state: key is null initially
  const [sortConfig, setSortConfig] = useState<{
    key: keyof T;
    direction: "asc" | "desc";
  } | null>(initialSort || null);

  const handleSort = (key: keyof T, defaultDirection: "asc" | "desc" = "asc") => {
    setSortConfig((current) => {
      if (current?.key === key) {
        // Toggle direction
        return { key, direction: current.direction === "asc" ? "desc" : "asc" };
      }
      // New key, use default direction
      return { key, direction: defaultDirection };
    });
  };

  const sortedData = useMemo(() => {
    if (!data) return [];
    if (!sortConfig) return data;

    return [...data].sort((a, b) => {
      const aVal = a[sortConfig.key];
      const bVal = b[sortConfig.key];

      if (aVal === bVal) return 0;
      // Nulls always last
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;

      const compare = aVal < bVal ? -1 : 1;
      return sortConfig.direction === "asc" ? compare : -compare;
    });
  }, [data, sortConfig]);

  if (data === null) {
    return (
      <div className={`w-full rounded-lg border-2 border-line2 bg-bg1 p-4 text-center text-textMuted ${wide ? "min-w-[760px]" : ""}`}>
        Live metrics unavailable
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className={`w-full rounded-lg border-2 border-line2 bg-bg1 p-4 text-center text-textMuted ${wide ? "min-w-[760px]" : ""}`}>
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className={wide ? "table-scroll w-full overflow-x-auto" : ""}>
      <table
        className={`w-full overflow-hidden rounded-lg border-2 border-line2 bg-bg1 ${wide ? "min-w-[760px]" : ""}`}
      >
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={String(col.key)}
                className={`border-b border-line px-3 py-2.5 text-left text-[0.85rem] uppercase tracking-[0.6px] text-textMain bg-[#232323] ${
                  col.sortable !== false ? "cursor-pointer hover:bg-[#2a2a2a] select-none" : ""
                }`}
                onClick={() =>
                  col.sortable !== false && handleSort(col.key, col.defaultDirection)
                }
              >
                <div className="flex items-center gap-1">
                  {col.header}
                  {col.sortable !== false && (
                    <span className="text-textMuted/50">
                      {sortConfig?.key === col.key ? (
                        sortConfig.direction === "asc" ? (
                          // Ascending arrow
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="12"
                            height="12"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className="text-textMain"
                          >
                            <path d="M12 19V5" />
                            <path d="M5 12l7-7 7 7" />
                          </svg>
                        ) : (
                          // Descending arrow
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="12"
                            height="12"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className="text-textMain"
                          >
                            <path d="M12 5v14" />
                            <path d="M19 12l-7 7-7-7" />
                          </svg>
                        )
                      ) : (
                        // Neutral sort icon
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="12"
                          height="12"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          className="opacity-40"
                        >
                          <path d="M7 15l5 5 5-5" />
                          <path d="M7 9l5-5 5 5" />
                        </svg>
                      )}
                    </span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedData.map((row, i) => (
            <tr key={i} className="hover:bg-white/5 transition-colors">
              {columns.map((col) => (
                <td
                  key={String(col.key)}
                  className={`border-b border-line px-3 py-2.5 text-textMuted ${
                    col.align === "right"
                      ? "text-right"
                      : col.align === "center"
                        ? "text-center"
                        : "text-left"
                  }`}
                >
                  {col.render ? col.render(row[col.key], row) : String(row[col.key] ?? "-")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

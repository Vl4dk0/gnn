import type { ReactNode } from "react";

interface MetricTableProps {
  headers: string[];
  rows: ReactNode;
  wide?: boolean;
}

export const MetricTable = ({ headers, rows, wide = false }: MetricTableProps) => {
  return (
    <div className={wide ? "table-scroll w-full overflow-x-auto" : ""}>
      <table
        className={`w-full overflow-hidden rounded-lg border-2 border-line2 bg-bg1 ${wide ? "min-w-[760px]" : ""}`}
      >
        <thead>
          <tr>
            {headers.map((header) => (
              <th
                key={header}
                className="border-b border-line px-3 py-2.5 text-left text-[0.85rem] uppercase tracking-[0.6px] text-textMain bg-[#232323]"
              >
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  );
};

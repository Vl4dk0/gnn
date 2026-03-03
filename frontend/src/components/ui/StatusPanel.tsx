import type { CageStatusResponse } from "../../types/api";

interface StatusPanelProps {
  status: CageStatusResponse | null;
  error: string | null;
  successMessage: string | null;
  stoppedByUser: boolean;
}

const formatGirth = (value: number | null) => {
  return value === null ? "∞" : String(value);
};

export const StatusPanel = ({ status, error, successMessage, stoppedByUser }: StatusPanelProps) => {
  if (error) {
    return (
      <div className="rounded bg-[#a52] p-3 text-white">
        <strong>Error:</strong> {error}
      </div>
    );
  }

  if (!status) {
    return <div>Ready to generate</div>;
  }

  return (
    <div>
      <div className="mb-2">
        <strong>Target:</strong> ({status.k},{status.g})-cage
      </div>
      <div className="mb-2">
        <strong>Step:</strong> {status.step_count}
      </div>
      <div className="mb-2">
        <strong>Nodes:</strong> {status.num_nodes} / Moore bound: {status.moore_bound}
      </div>
      <div className="mb-2">
        <strong>Edges:</strong> {status.num_edges}
      </div>
      <div className="mb-2">
        <strong>k-regular:</strong> {status.is_k_regular ? "✓" : "✗"}
      </div>
      <div className="mb-2">
        <strong>Girth:</strong> {formatGirth(status.girth)} (target: {status.g})
      </div>

      {!status.is_complete && (
        <div className="mt-3 text-textDim">
          ⏳ Generating... ({status.elapsed_time.toFixed(1)}s)
        </div>
      )}

      {stoppedByUser && (
        <div className="mt-3 text-textDim">
          ⏹ Stopped by user ({status.elapsed_time.toFixed(1)}s)
        </div>
      )}

      {successMessage && (
        <div className="mt-3 rounded bg-[#2a5] p-3 text-white">
          <strong>✓</strong> {successMessage}
        </div>
      )}
    </div>
  );
};

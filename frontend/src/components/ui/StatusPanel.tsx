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

const formatAction = (action: string) => {
  return action.replace(/^edge_/, "").replaceAll("_", " ");
};

export const StatusPanel = ({ status, error, successMessage, stoppedByUser }: StatusPanelProps) => {
  if (error) {
    return (
      <div className="rounded-xl bg-[#a52] p-3 text-sm text-white">
        <strong>Error:</strong> {error}
      </div>
    );
  }

  if (!status) {
    return <div className="text-sm text-textMuted">Ready to generate</div>;
  }

  return (
    <div className="text-sm text-textMuted">
      <div className="mb-2">
        <strong>Target:</strong> ({status.k},{status.g})-cage
      </div>
      <div className="mb-2">
        <strong>Mode:</strong> {status.mode === "stepped" ? "RL step inspection" : "Fast search"}
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

      {status.last_event && (
        <div className="mt-3 rounded-xl border border-line2 bg-bg2/75 p-3">
          <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.8px] text-textDim">
            Last RL Move
          </div>
          <div>
            <strong>Action:</strong> {formatAction(status.last_event.action)}
          </div>
          {status.last_event.u !== null && status.last_event.v !== null && (
            <div>
              <strong>Edge:</strong> {status.last_event.u}-{status.last_event.v}
            </div>
          )}
          <div>
            <strong>Accepted:</strong> {status.last_event.accepted ? "yes" : "no"}
          </div>
          <div>
            <strong>Reward:</strong> {status.last_event.reward.toFixed(3)}
          </div>
          {status.last_event.done_reason && (
            <div>
              <strong>Reason:</strong> {status.last_event.done_reason}
            </div>
          )}
        </div>
      )}

      {!status.is_complete && !status.timed_out && (
        <div className="mt-3 text-sm text-textDim">
          {status.mode === "stepped" ? "Inspection ready" : "⏳ Generating..."} (
          {status.elapsed_time.toFixed(1)}s)
        </div>
      )}

      {status.timed_out && (
        <div className="mt-3 text-sm text-textDim">
          Generation timed out ({status.elapsed_time.toFixed(1)}s)
        </div>
      )}

      {stoppedByUser && (
        <div className="mt-3 text-sm text-textDim">
          ⏹ Stopped by user ({status.elapsed_time.toFixed(1)}s)
        </div>
      )}

      {successMessage && (
        <div className="mt-3 rounded-xl bg-[#2a5] p-3 text-sm text-white">
          <strong>✓</strong> {successMessage}
        </div>
      )}
    </div>
  );
};

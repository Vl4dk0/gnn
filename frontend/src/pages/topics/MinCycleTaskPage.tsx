import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { useHighlight } from "../../hooks/useHighlight";

export const MinCycleTaskPage = () => {
  useHighlight();

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Min-cycle subproblem
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          The second constraint of a (k,g)-graph is girth. Unlike degree, girth is a global
          signal — it depends on paths that leave a vertex and loop back, not just the local
          neighbourhood. This task probes the structural limit of GNN expressivity.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Why girth is harder than degree</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The per-node label here is the length of the shortest cycle through that vertex (0 if
          the vertex lies on no cycle). Detecting whether any cycle exists — let alone its length
          — requires tracking paths of unbounded length, which is beyond what the standard
          1-Weisfeiler–Leman colour-refinement test can distinguish. Standard message-passing GNNs
          are bounded by 1-WL, so this task sits exactly at the expressivity boundary: a model
          that succeeds must be doing something beyond simple neighbourhood aggregation.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          This makes the task a meaningful probe: if a GNN architecture can learn min-cycle labels
          reliably, it has enough structural sensitivity to be useful for girth-constrained
          construction. Failure tells us the architecture would miss short cycles in a partial
          graph and produce invalid (k,g)-graphs.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Exact per-node label</h2>
        <p className="mb-2.5 text-base leading-[1.7] text-textMuted">
          For each vertex, the label is computed by temporarily removing each incident edge and
          finding the shortest detour back.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`# ai/min_cycle/.../graph_service.py — exact per-node label
def get_min_cycle(G, vertex):
    ans = None
    for neigh in G.neighbors(vertex):
        G.remove_edge(vertex, neigh)            # break the direct edge
        try:
            path = nx.shortest_path(G, vertex, neigh)  # shortest detour back
            ans = min(ans, len(path)) if ans is not None else len(path)
        except nx.NetworkXNoPath:
            pass
        G.add_edge(vertex, neigh)               # restore
    return ans if ans is not None else 0        # 0 = vertex on no cycle`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Draw a graph in the editor; the trained GNN predicts the minimum-cycle length for each
          node live. Compare predictions against the exact labels to see how well the model tracks
          this global structural signal.
        </p>
        <a href="/min_cycle" className="ui-button-solid ui-surface-link mt-3">
          Open the min-cycle editor
        </a>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/degree-task" direction="back" label="Degree subproblem" />
        <DocsNextButton href="/cage/astar" label="A* + backtracking" />
      </div>
    </DocsLayout>
  );
};

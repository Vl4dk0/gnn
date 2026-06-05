import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { useHighlight } from "../../hooks/useHighlight";

export const CageAstarPage = () => {
  useHighlight();

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          A* + backtracking
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          Best-first search over partial (k,g)-graphs, with backtracking when no successor can
          preserve the girth constraint. The key challenge is scoring partial states — deciding
          whether a partial graph can still be completed is itself the original hard problem.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">How the search works</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The search maintains a min-heap of partial graphs ordered by a score that combines
          regularity progress (how many vertices already have the right degree), closeness to the
          Moore-bound target size, and girth. At each step, the algorithm expands the best-scoring
          partial graph by applying one of two actions: add a girth-safe edge between two
          under-degree vertices, or grow the graph by adding a new vertex (bounded by roughly
          twice the Moore bound).
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          A pure supervised heuristic fails here because you cannot label partial states as
          "completable" or "dead end" without solving the original problem first. Backtracking is
          therefore unavoidable. The same backtracking/search logic reappears as the exact fallback
          inside excision repair when the greedy re-stitching gets stuck.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Successor generation</h2>
        <p className="mb-2.5 text-base leading-[1.7] text-textMuted">
          Two actions expand a partial graph; girth is checked before any edge is committed.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`# ai/cage/registry/astar.py — successors of a partial graph
def _generate_successors(self, graph):
    succ = []
    # Action 1: connect two under-degree vertices if girth survives
    for u in nodes:
        if graph.degree(u) >= self.k: continue
        for v in nodes_after(u):
            if graph.degree(v) >= self.k or graph.has_edge(u, v): continue
            if can_add_edge_preserving_girth(graph, u, v, self.g):
                succ.append(with_edge(graph, u, v))
    # Action 2: grow the graph (until ~2x Moore bound)
    if graph.number_of_nodes() < 2 * self.mb:
        succ.append(with_new_vertex(graph))
    return succ`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The cage editor offers several algorithms including A*. Set a target (k, g), run the
          search, and watch partial graphs expand toward a valid cage.
        </p>
        <a href="/cage" className="ui-button-solid ui-surface-link mt-3">
          Open the cage editor
        </a>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/min-cycle-task" direction="back" label="Min-cycle subproblem" />
        <DocsNextButton href="/cage/rl" label="Reinforcement learning" />
      </div>
    </DocsLayout>
  );
};

import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { EditorLinks } from "../../components/docs/EditorLinks";
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
          preserve the girth constraint. The key challenge is scoring partial states. Deciding
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
          under-degree vertices, or grow the graph by adding a new vertex (bounded by roughly twice
          the Moore bound).
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          A pure supervised heuristic fails here because you cannot label partial states as
          "completable" or "dead end" without solving the original problem first. Backtracking is
          therefore unavoidable. The same backtracking and search logic reappears as the exact
          fallback inside excision repair when the greedy re-stitching gets stuck.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Successor generation</h2>
        <p className="mb-2.5 text-base leading-[1.7] text-textMuted">
          Two actions expand a partial graph. Girth is checked before any edge is committed.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`# ai/cage/registry/astar.py
def _generate_successors(self, graph: nx.Graph[int]) -> list[nx.Graph[int]]:
    successors: list[nx.Graph[int]] = []
    nodes = list(graph.nodes())

    # Action 1: connect two under-degree vertices if girth survives
    for i, u in enumerate(nodes):
        if graph.degree(u) >= self.k:
            continue
        for v in nodes[i + 1 :]:
            if graph.degree(v) >= self.k:
                continue
            if graph.has_edge(u, v):
                continue
            if can_add_edge_preserving_girth(graph, u, v, self.g):
                new_graph: nx.Graph[int] = graph.copy()
                new_graph.add_edge(u, v)
                successors.append(new_graph)

    # Action 2: grow the graph (until ~2x Moore bound)
    if len(nodes) < self.mb * 2:
        new_graph = graph.copy()
        new_graph.add_node(len(nodes))
        successors.append(new_graph)

    return successors`}</code>
        </pre>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          This is the heuristic that orders the search frontier.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`# ai/cage/registry/astar.py
def _score_graph(self, graph: nx.Graph[int]) -> float:
    num_nodes = len(graph.nodes())
    num_edges = len(graph.edges())

    base_score = score_graph_quality(graph, self.k, self.g)

    # Fraction of vertices that already have the target degree
    regularity_progress = (
        sum(1 for n in graph.nodes() if graph.degree(n) == self.k) / num_nodes
        if num_nodes > 0 else 0.0
    )

    # Prefer graphs close to Moore bound in node count
    size_score = (
        1.0 - abs(num_nodes - self.mb) / max(self.mb, num_nodes)
        if num_nodes > 0 else 0.0
    )

    # Prefer edge count close to n*k/2
    target_edges = (num_nodes * self.k) // 2
    edge_score = (
        1.0 - abs(num_edges - target_edges) / max(target_edges, num_edges)
        if target_edges > 0 else 0.0
    )

    # Strong bonus if girth matches g exactly
    current_girth = compute_girth(graph)
    if current_girth == self.g:
        girth_score = 1.0
    elif current_girth == float("inf"):
        girth_score = 0.5   # no cycles yet
    elif current_girth > self.g:
        girth_score = 0.8   # girth too high (graph not yet dense enough)
    else:
        girth_score = 0.0   # girth already violated

    return (
        0.3 * base_score
        + 0.3 * regularity_progress
        + 0.15 * size_score
        + 0.15 * edge_score
        + 0.1 * girth_score
    )`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it</h2>
        <p className="mb-3 text-base leading-[1.7] text-textMuted">
          The cage editor offers several algorithms including A*. Set a target (k, g), run the
          search, and watch partial graphs expand toward a valid cage.
        </p>
        <EditorLinks
          links={[
            {
              href: "/cage?from=astar",
              label: "Open"
            }
          ]}
        />
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/min-cycle-task" direction="back" label="Min-cycle subproblem" />
        <DocsNextButton href="/cage/rl" label="Reinforcement learning" />
      </div>
    </DocsLayout>
  );
};

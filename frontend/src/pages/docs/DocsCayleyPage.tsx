import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { useHighlight } from "../../hooks/useHighlight";

export const DocsCayleyPage = () => {
  useHighlight();

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Cayley graphs
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          A purely algebraic method for generating highly symmetric, regular graphs using group theory.
        </p>
      </DocsHero>

      {/* Cayley Construction */}
      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Cayley Construction
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          A Cayley graph is constructed from a finite group and a chosen symmetric generating set. Each vertex represents a group element, and an edge exists between <code>g</code> and <code>h</code> if <code>g * s = h</code> for some generator <code>s</code>. This naturally guarantees that the graph is <code>|S|</code>-regular and vertex-transitive.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5 text-[0.85rem]">
          <code className="language-python">
            {`# ai/cage/cayley/generators.py
def build_cayley_graph(group, generators):
    G = nx.Graph()
    for g in group.elements():
        for s in generators:
            h = group.mult(g, s)
            G.add_edge(g, h)
    return G`}
          </code>
        </pre>
      </DocsCard>
      
      {/* Group Catalog */}
      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Group Catalog
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Finding the right group is half the battle. Abelian groups (like cyclic groups) hit a girth ceiling, so we must rely on non-abelian groups. By maintaining a catalog of permutation groups, matrix groups like <code>SL(2, p)</code>, and dihedral groups, we expand the search space enough to find optimal cages.
        </p>
      </DocsCard>

      {/* Non-Backtracking BFS */}
      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Non-backtracking BFS
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          To efficiently evaluate girth without building the whole graph, we can run a non-backtracking BFS over the group words. If a word evaluates to the identity, it forms a cycle.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5 text-[0.85rem]">
          <code className="language-python">
            {`# ai/cage/cayley/bfs.py
def non_backtracking_girth(group, generators, max_depth):
    # Search for short words equal to identity
    frontier = [(g, [g]) for g in generators]
    for length in range(2, max_depth + 1):
        next_frontier = []
        for val, word in frontier:
            for s in generators:
                # non-backtracking condition:
                if group.inv(s) == word[-1]: continue
                new_val = group.mult(val, s)
                if new_val == group.identity():
                    return length
                next_frontier.append((new_val, word + [s]))
        frontier = next_frontier
    return math.inf`}
          </code>
        </pre>
      </DocsCard>

      {/* GNN Promise Predictor */}
      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          GNN Promise Predictor
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Evaluating every subset of generators is too slow for large groups. A unified GNN predictor trained across various <code>(k,g)</code> parameters can evaluate the "promise" of a generating set and prune the search tree. This dramatically speeds up the discovery of high-girth Cayley graphs. Note: <code>(k,g)</code> values are passed in as context features to the single unified model — we do not train separate models per target.
        </p>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/docs/voltage" label="Voltage graph lifts" direction="back" />
      </div>
    </DocsLayout>
  );
};

import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { EditorLinks } from "../../components/docs/EditorLinks";
import { useHighlight } from "../../hooks/useHighlight";

export const RefinementPage = () => {
  useHighlight();

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Edge-swap refinement
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          A voltage lift is always k-regular, but a few cycles shorter than the target girth g can
          survive. When that happens the candidate is close to a valid (k,g)-graph and worth fixing
          rather than discarding. Refinement edits a small number of edges to remove those surviving
          short cycles while keeping every vertex degree exactly k.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">The 2-swap move</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The basic operation picks two edges that share no endpoint, <code>A-B</code> and{" "}
          <code>C-D</code>, on four distinct vertices. Both are deleted and the four endpoints are
          reconnected the other way, for example as <code>A-D</code> and <code>B-C</code>. Each of
          the four vertices loses exactly one edge and gains exactly one edge, so every degree is
          unchanged and a k-regular graph stays k-regular.
        </p>
        <div className="mt-3 overflow-hidden rounded-lg border-2 border-line2 bg-white p-3">
          <img src="/static/edge-swap-move.png" alt="A single 2-swap" className="block w-full" />
        </div>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`def apply_2_switch(G: nx.Graph[int], swap: Swap2) -> nx.Graph[int]:
    """Return a new graph obtained by applying swap to G.

    The original graph is not modified.  Preserves k-regularity: each of the
    four endpoints loses one edge and gains one edge.
    """
    H: nx.Graph[int] = G.copy()
    u, v, x, y, mode = swap
    H.remove_edge(u, v)
    H.remove_edge(x, y)
    if mode == "uxvy":
        H.add_edge(u, x)
        H.add_edge(v, y)
    else:
        H.add_edge(u, y)
        H.add_edge(v, x)
    return H`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Objective and tabu search</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The cost measures how many short cycles remain. For each cycle length{" "}
          <code>c</code> from 3 up to <code>g-1</code>, the contribution is{" "}
          <code>2^(g-c)</code> times the number of length-<code>c</code> cycles, so triangles carry
          more weight than longer sub-girth cycles and the shortest cycles are removed first. The
          cost reaches exactly zero when the graph has no cycles shorter than g, meaning girth is at
          least g.
        </p>
        <div className="mt-3 overflow-hidden rounded-lg border-2 border-line2 bg-white p-3">
          <img
            src="/static/edge-swap-cycle.png"
            alt="A 2-swap that removes a short cycle"
            className="block w-full"
          />
        </div>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`def short_cycle_cost(
    G: nx.Graph[int],
    g_target: int,
    weights: dict[int, float] | None = None,
) -> float:
    """Compute the weighted short-cycle cost for graph G.

    f(G) = sum_{c=3}^{g_target-1} w_c * N_c(G)

    Default weights: w_c = 2^(g_target - c)  (exponential penalty for
    shorter cycles, so length 3 gets the highest weight).

    Returns 0.0 when g_target <= 3 or when G has no cycles shorter than g_target.
    """
    if g_target <= 3:
        return 0.0

    max_len = g_target - 1
    cycle_counts = count_cycles_up_to(G, max_len)

    total = 0.0
    for c in range(3, g_target):
        n_c = cycle_counts.get(c, 0)
        if n_c == 0:
            continue
        if weights is not None:
            w = weights.get(c, 0.0)
        else:
            w = float(2 ** (g_target - c))
        total += w * n_c

    return total`}</code>
        </pre>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The search strategy is tabu search. Each step picks the best available non-tabu swap, the
          one that lowers the cost the most, or the least harmful one when no improving swap
          exists. A tabu list forbids immediately reversing recent moves. A tabu swap is applied
          anyway if the resulting graph beats the best cost seen so far. Refinement succeeds when the
          cost reaches zero within the allowed step budget, otherwise the candidate is dropped. In
          the Forge pipeline a GNN can score and rank candidate swaps, while exact girth is always
          verified before accepting the result.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">3-swap escalation</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          When 2-swaps stagnate, the refiner escalates to 3-swaps. A 3-swap selects three
          pairwise-disjoint edges, removes all three, and reconnects the six endpoints into a
          different degree-preserving perfect matching, reaching configurations that no single 2-swap
          can. Because there are far more candidate 3-swaps than 2-swaps, they are applied
          occasionally rather than at every step.
        </p>
        <div className="mt-3 overflow-hidden rounded-lg border-2 border-line2 bg-white p-3">
          <img
            src="/static/three-swap.png"
            alt="A 3-swap on three disjoint edges"
            className="block w-full"
          />
        </div>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it</h2>
        <p className="mb-3 text-base leading-[1.7] text-textMuted">
          Build or import a near-miss k-regular graph, set the target girth, and watch edge swaps
          remove the short cycles.
        </p>
        <EditorLinks links={[{ href: "/refine", label: "Open" }]} />
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/cage/voltage" direction="back" label="Voltage lifts" />
        <DocsNextButton href="/excision" label="Excision" />
      </div>
    </DocsLayout>
  );
};

import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { EditorLinks } from "../../components/docs/EditorLinks";
import { useHighlight } from "../../hooks/useHighlight";

export const CageVoltagePage = () => {
  useHighlight();

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Voltage lifts
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          Start with a tiny base graph, assign a group element (a voltage) to each edge, and
          stamp out a large graph over <code>V(base) × G</code>. If the base is k-regular, the
          lift is automatically k-regular, so only girth needs to be optimised.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">The lift construction</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          A finite group <code>Γ</code> (typically cyclic <code>Z_n</code>) and a voltage
          assignment (one group element per oriented base edge) define the lift. Each base
          vertex <code>v</code> becomes <code>|Γ|</code> copies <code>(v, γ)</code>. An oriented
          base edge <code>u → w</code> with voltage <code>α</code> becomes the edge{" "}
          <code>(u, γ) – (w, γ·α)</code> for every <code>γ ∈ Γ</code>. The reverse arc carries{" "}
          <code>α⁻¹</code>. Because every base vertex fans out to exactly <code>|Γ|</code> copies
          and every base edge fans out to exactly <code>|Γ|</code> lifted edges, the degree of
          every vertex in the lift equals the degree of its base vertex.
        </p>
        <div className="mt-3 overflow-hidden rounded-lg border-2 border-line2 bg-white p-3">
          <img
            src="/static/voltage-base-and-lift.png"
            alt="A small base graph and the larger graph it lifts to"
            className="block w-full"
          />
        </div>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Canonical example: a 2-vertex base with 3 parallel edges over <code>Z₇</code> lifts to
          the 14-vertex Heawood graph, the optimal (3,6)-cage. Girth in the lift equals the
          length of the shortest closed walk in the base whose net voltage product is the group
          identity, checkable on the base without building the lift.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`# ai/cage/voltage/lift.py
def build_lift(
    base: BaseGraph,
    group: FiniteGroup,
    voltages: list[int],
) -> nx.Graph[int]:
    n_group = group.order
    G: nx.Graph[int] = nx.Graph()
    G.add_nodes_from(range(base.num_nodes * n_group))

    # Map each arc to its voltage (reverse arc carries the inverse)
    arc_voltage: dict[int, int] = {}
    for edge_pos, fwd_id in enumerate(base.undirected_edge_ids):
        v = voltages[edge_pos]
        arc_voltage[fwd_id] = v
        arc_voltage[base.arcs[fwd_id].reverse_id] = group.inv(v)

    # For each arc u -> w with voltage alpha, add edge (u,g) -- (w, g*alpha)
    # for every group element g.  Vertex (v, g) is encoded as v * n_group + g.
    for arc in base.arcs:
        alpha = arc_voltage[arc.arc_id]
        for g in group.elements():
            h = group.mult(g, alpha)
            node_ug = arc.src * n_group + g
            node_wh = arc.dst * n_group + h
            if node_ug != node_wh:
                G.add_edge(node_ug, node_wh)

    return G`}</code>
        </pre>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Girth can be read off the tiny base graph without building the lift.
        </p>
        <div className="mt-3 overflow-hidden rounded-lg border-2 border-line2 bg-white p-3">
          <img
            src="/static/voltage-girth.png"
            alt="A closed walk in the base graph whose net voltage determines a cycle in the lift"
            className="block w-full"
          />
        </div>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`# ai/cage/voltage/cycle_analysis.py
def compute_lift_girth(
    base: BaseGraph,
    group: FiniteGroup,
    voltages: list[int],
    max_girth: int = 50,
) -> int | float:
    """Girth of the lift = shortest closed walk in the base with identity net voltage.

    Enumerates closed walks up to max_girth and checks net voltage product.
    Returns math.inf if no identity-voltage walk is found within that length.
    """
    arc_voltage: dict[int, int] = {}
    for edge_pos, fwd_id in enumerate(base.undirected_edge_ids):
        v = voltages[edge_pos]
        arc_voltage[fwd_id] = v
        arc_voltage[base.arcs[fwd_id].reverse_id] = group.inv(v)

    adj: dict[int, list[tuple[int, int]]] = {v: [] for v in range(base.num_nodes)}
    for arc in base.arcs:
        adj[arc.src].append((arc.dst, arc.arc_id))

    identity = group.identity()
    best_girth: int | float = math.inf

    for start in range(base.num_nodes):
        # frontier: (current_node, net_voltage, walk_length, prev_arc_id)
        frontier: list[tuple[int, int, int, int]] = [
            (nb, arc_voltage[arc_id], 1, arc_id)
            for nb, arc_id in adj[start]
        ]
        for _depth in range(1, max_girth + 1):
            if not frontier:
                break
            next_frontier: list[tuple[int, int, int, int]] = []
            for node, net_v, length, prev_arc in frontier:
                if length >= best_girth:
                    continue
                if node == start and net_v == identity and length >= 3:
                    best_girth = min(best_girth, length)
                    continue
                for next_node, arc_id in adj[node]:
                    # _would_backtrack is a local helper (capturing base, group,
                    # arc_voltage) that is True when arc_id reverses prev_arc.
                    if not _would_backtrack(prev_arc, arc_id):
                        new_v = group.mult(net_v, arc_voltage[arc_id])
                        next_frontier.append((next_node, new_v, length + 1, arc_id))
            frontier = next_frontier

    return best_girth`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it</h2>
        <p className="mb-3 text-base leading-[1.7] text-textMuted">
          The voltage-lift editor lets you draw a base graph, assign voltages to its edges, and
          generate the lift immediately.
        </p>
        <EditorLinks
          links={[
            {
              href: "/lift",
              label: "Open"
            }
          ]}
        />
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it</h2>
        <p className="mb-3 text-base leading-[1.7] text-textMuted">
          The cage editor runs the voltage methods (algebraic, predictor-guided, and RL) alongside
          the other construction methods.
        </p>
        <EditorLinks
          links={[
            {
              href: "/cage?from=voltage",
              label: "Open"
            }
          ]}
        />
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/cage/rl" direction="back" label="Reinforcement learning" />
        <DocsNextButton href="/refinement" label="Refinement" />
      </div>
    </DocsLayout>
  );
};

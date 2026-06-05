import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
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
          lift is automatically k-regular — so only girth needs to be optimised.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">The lift construction</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          A finite group <code>Γ</code> (typically cyclic <code>Z_n</code>) and a voltage
          assignment — one group element per oriented base edge — define the lift. Each base
          vertex <code>v</code> becomes <code>|Γ|</code> copies <code>(v, γ)</code>. An oriented
          base edge <code>u → w</code> with voltage <code>α</code> becomes the edge{" "}
          <code>(u, γ) – (w, γ·α)</code> for every <code>γ ∈ Γ</code>. The reverse arc carries{" "}
          <code>α⁻¹</code>. Because every base vertex fans out to exactly <code>|Γ|</code> copies
          and every base edge fans out to exactly <code>|Γ|</code> lifted edges, the degree of
          every vertex in the lift equals the degree of its base vertex.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Canonical example: a 2-vertex base with 3 parallel edges over <code>Z₇</code> lifts to
          the 14-vertex Heawood graph — the optimal (3,6)-cage. Girth in the lift equals the
          length of the shortest closed walk in the base whose net voltage product is the group
          identity, checkable on the base without building the lift.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`# ai/cage/voltage/lift.py — build the lift
# vertex (v, g) is encoded as v * |Γ| + g
for arc in base.arcs:                 # each oriented base edge u→w
    alpha = arc_voltage[arc.arc_id]   # reverse arc carries inv(alpha)
    for g in group.elements():        # one copy per group element
        G.add_edge(arc.src * n + g,
                   arc.dst * n + group.mult(g, alpha))   # (u,g)–(w, g·α)`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The voltage-lift editor lets you draw a base graph, assign voltages to edges, and
          generate the lift immediately. The cage editor also exposes the voltage algorithm
          alongside the other construction methods.
        </p>
        <div className="mt-3 flex flex-wrap gap-3">
          <a href="/cage" className="ui-button-outline ui-surface-link">
            Open the cage editor
          </a>
          <a href="/lift" className="ui-button-solid ui-surface-link">
            Open the voltage-lift editor
          </a>
        </div>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/cage/rl" direction="back" label="Reinforcement learning" />
        <DocsNextButton href="/excision" label="Excision" />
      </div>
    </DocsLayout>
  );
};

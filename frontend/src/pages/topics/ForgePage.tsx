import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { EditorLinks } from "../../components/docs/EditorLinks";
import { useHighlight } from "../../hooks/useHighlight";

export const ForgePage = () => {
  useHighlight();

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Forge pipeline
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          Compose a voltage producer, a refinement worker, and an excision worker end to end so
          each does only what it does well. The producer supplies reach at high girth, refinement
          rescues lifts that are almost right, and excision recovers size-optimality.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">The three stages</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          <strong>Stage 1 (producer).</strong> A voltage lift builds a k-regular, connected
          graph whose girth is guaranteed by the voltage group structure. Voltage lifts can
          reach hard high-girth targets that edge-by-edge methods never find, but the graphs
          they return are roughly twice the Moore bound, far from the cage order.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          <strong>Stage 2 (refinement).</strong> A k-regular lift whose girth falls just short
          of the target (a near-miss) is repaired into a valid (k,g)-graph by degree-preserving
          edge swaps. Refinement can close a small girth gap that the lift introduced, turning a
          near-miss into an exact-girth graph ready for excision.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          <strong>Stage 3 (excision).</strong> A valid (k,g)-graph is shrunk toward the cage by
          removing a Moore-radius BFS tree and re-stitching the boundary with girth-safe edges.
          Excision reduces the order while keeping girth and degree intact.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Why the order is forced</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Excision can shrink a valid (k,g)-graph but cannot close a girth gap. If it receives a
          near-miss, the re-stitched graph inherits the short cycle and is no longer a valid
          (k,g)-graph. Excision must therefore be last and must only consume graphs that have
          already reached girth g.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Closing the girth gap is refinement's job. Refinement sits between a producer that
          emits near-misses and an excision that only accepts valid graphs. A lift that already
          reaches girth g skips refinement entirely and goes straight to excision. A near-miss
          (girth below g) is routed to refinement only when its defective fraction (the fraction
          of edges on a cycle shorter than g) is at most a threshold. If refinement reaches girth
          g, it hands the graph to excision. If not, the graph is discarded.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Orchestration and why the combination wins
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The three stages run concurrently in a round-robin loop. Each call to{" "}
          <code>step()</code> advances exactly one unit of work in one stage. The first valid
          (k,g)-graph out of excision is the result and the pipeline stops.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`# ai/cage/registry/forge.py
def _advance_stage(self, stage_idx: int) -> bool:
    """Advance stage *stage_idx* by one unit; return True if work was done."""
    if stage_idx == 0:
        return self._step_producer()
    if stage_idx == 1:
        return self._step_refine()
    return self._step_excision()`}</code>
        </pre>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The producer is interchangeable: any voltage method (RL-guided, girth-predictor
          search, tabu search, or purely algebraic) can drive stage 1. Either repair stage can
          also be turned off independently, which makes Forge a controlled way to ask which
          stage earns its keep.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Combined, the producer's reach at high girth is kept while the repair stages recover
          the size-optimality the lift sacrifices. No single method does this on its own: a lift
          alone lands at about twice the Moore bound, and refinement or excision alone construct
          nothing (they require a valid input graph to improve).
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Where GNNs fit</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Each stage has a natural slot for a learned model, and each slot can be filled or left
          with a classical fallback independently.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          In <strong>refinement</strong>, a GNN evaluator can rank candidate edge swaps. The
          exact girth is always verified after each swap, so the GNN guides search but never
          bypasses correctness.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          In <strong>excision</strong>, a GNN actor-critic can pick girth-safe stitch edges
          when re-connecting the boundary. Exact backtracking serves as a fallback when no
          policy-selected edge is legal.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The <strong>producer</strong> itself can be GNN-guided: the voltage RL backend uses a
          learned policy to assign group elements, and the girth-predictor backend scores
          candidate assignments before committing to them.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Because the producer is interchangeable and each repair stage can be switched off,
          Forge is also a controlled way to measure what each GNN contributes. A GNN that does
          not improve a stage is a valid finding, not a failure.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try the stages individually</h2>
        <p className="mb-3 text-base leading-[1.7] text-textMuted">
          Each stage has its own interactive editor where you can watch it work step by step.
        </p>
        <EditorLinks
          links={[
            { href: "/lift", label: "Voltage-lift editor" },
            { href: "/refine", label: "Refinement editor" },
            { href: "/excise", label: "Excision editor" }
          ]}
        />
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/excision" direction="back" label="Excision" />
      </div>
    </DocsLayout>
  );
};

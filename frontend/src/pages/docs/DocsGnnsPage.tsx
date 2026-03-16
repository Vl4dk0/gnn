import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { useFeatureFlags } from "../../hooks/useFeatureFlags";
import { useHighlight } from "../../hooks/useHighlight";

export const DocsGnnsPage = () => {
  const features = useFeatureFlags();
  useHighlight();

  return (
    <>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          How GNNs work in this project
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          This section explains the mechanics used by the models in this repository: node and edge
          features, message passing, and readout.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          What goes into a GNN layer?
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Each node starts with a feature vector, often denoted <code>x_v</code>. In this project,
          degree/min-cycle training uses four node features per vertex: normalized node index, two
          random channels, and one random clustering-like channel.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Edges define who can exchange messages. Some GNN formulations also use explicit edge
          features
          <code> e_{"{vw}"}</code>; if edge features are not provided, the model still uses the
          graph connectivity itself.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Two phases: message passing and readout
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          A forward pass has two conceptual phases. In message passing, each node updates its hidden
          state by combining its current state with aggregated neighbor information. In readout, the
          final node states are mapped to outputs.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          In this project, degree and min-cycle are node-level regression tasks, so readout is
          per-node: each final node state is mapped to one scalar prediction.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">One message-passing update</h2>
        <div className="mt-2.5 rounded-lg border-2 border-line2 bg-bg1 p-4">
          <p className="text-sm font-semibold uppercase tracking-[0.7px] text-textDim">Message</p>
          <p className="mt-1 font-mono text-[0.95rem] text-textMain">
            m<sub>v</sub>
            <sup>(t)</sup> = AGG<sub>{"{w in N(v)}"}</sub> φ<sub>t</sub>(h<sub>v</sub>
            <sup>(t)</sup>, h<sub>w</sub>
            <sup>(t)</sup>, e<sub>vw</sub>)
          </p>
          <p className="mt-4 text-sm font-semibold uppercase tracking-[0.7px] text-textDim">
            Update
          </p>
          <p className="mt-1 font-mono text-[0.95rem] text-textMain">
            h<sub>v</sub>
            <sup>(t+1)</sup> = ψ<sub>t</sub>(h<sub>v</sub>
            <sup>(t)</sup>, m<sub>v</sub>
            <sup>(t)</sup>)
          </p>
        </div>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">One layer, step by step</h2>
        <div className="grid grid-cols-[repeat(2,minmax(0,1fr))] gap-4 max-[760px]:grid-cols-1">
          {[1, 2, 3, 4].map((index) => (
            <figure key={index} className="rounded-[10px] p-0">
              <img
                src={`/static/mpnn-step-${index}.svg`}
                alt={`Step ${index}`}
                className="block w-full rounded-[10px]"
              />
            </figure>
          ))}
        </div>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Why aggregation choice matters
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Aggregation is not fixed to mean. Different architectures choose different aggregation and
          update operators, and this directly changes what structural signals remain visible.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Citations</h2>
        <ol className="ml-5 mt-1.5 list-decimal space-y-2 text-textMuted">
          <li>
            Gilmer et al., Neural Message Passing for Quantum Chemistry:
            <a className="text-textMain" href="https://arxiv.org/abs/1704.01212">
              {" "}
              arXiv 1704.01212
            </a>
          </li>
          <li>
            Oono and Suzuki, expressive-power decay in deep GNNs:
            <a className="text-textMain" href="https://openreview.net/forum?id=S1ldO2EFPr">
              {" "}
              OpenReview S1ldO2EFPr
            </a>
          </li>
        </ol>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/" label="Overview" direction="back" />
        <DocsNextButton href="/docs/architecture" label="Architectures" />
      </div>
    </>
  );
};

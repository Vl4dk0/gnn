import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";

export const DocsArchitecturePage = () => {
  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Model Architectures
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          Architecture choice matters because it changes the message passing{" "}
          <code>update rule</code>: how neighbor signals are aggregated, how self-information is
          mixed in, and which structural distinctions survive.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">At a glance</h2>
        <div className="overflow-x-auto">
          <table className="mt-2 w-full text-sm text-textMuted">
            <thead>
              <tr className="border-b border-line2 text-left text-textSub">
                <th className="py-2 pr-4 font-semibold">Architecture</th>
                <th className="py-2 pr-4 font-semibold">Aggregation</th>
                <th className="py-2 pr-4 font-semibold">Scope</th>
                <th className="py-2 font-semibold">Key strength</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">GCN</td>
                <td className="py-2 pr-4">Degree-normalized mean</td>
                <td className="py-2 pr-4">Local</td>
                <td className="py-2">Stable, simple baseline</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">GraphSAGE</td>
                <td className="py-2 pr-4">Learned mean</td>
                <td className="py-2 pr-4">Local</td>
                <td className="py-2">Inductive, generalizes to unseen nodes</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">GIN</td>
                <td className="py-2 pr-4">Sum + MLP</td>
                <td className="py-2 pr-4">Local</td>
                <td className="py-2">Maximally expressive (1-WL equivalent)</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">GPS</td>
                <td className="py-2 pr-4">Local conv + global attention</td>
                <td className="py-2 pr-4">Global</td>
                <td className="py-2">Captures long-range dependencies</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 font-medium text-textMain">Loopy</td>
                <td className="py-2 pr-4">Sum + r-neighborhood paths</td>
                <td className="py-2 pr-4">Local (cycle-aware)</td>
                <td className="py-2">Detects cycles up to length r+2</td>
              </tr>
            </tbody>
          </table>
        </div>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          GCN - Graph Convolution Network
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          GCN uses degree-normalized aggregation: each neighbor's message is scaled by the inverse
          square root of both the sender's and receiver's degree. This normalization makes training
          stable but can dampen raw counting signals — the model struggles to distinguish a node
          with 3 neighbors from one with 5 if the normalized values end up similar.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          GraphSAGE - Sampling and Aggregation
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          GraphSAGE was designed as an inductive architecture: it learns <code>aggregator
          functions</code> rather than fixed node embeddings. This means it can generalize to
          unseen nodes and dynamic graphs without retraining. In practice it uses a learned mean
          aggregation followed by concatenation with the node's own features.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          GIN - Graph Isomorphism Network
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          GIN uses sum aggregation followed by an MLP with a learnable self-weight term. It is
          provably as powerful as the <code>1-WL test</code> (Weisfeiler-Leman graph isomorphism
          test) — a classical algorithm that distinguishes graphs by iteratively comparing
          neighborhood multisets. In practice, this means GIN can distinguish any pair of graphs
          that 1-WL can distinguish, making it one of the most expressive standard GNN
          architectures.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          GPS - General, Powerful, Scalable
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          GPS is a graph transformer that combines local message passing with global attention. Each
          layer runs a standard GNN convolution (GCN, GIN, or SAGE) in parallel with a multi-head
          attention mechanism over all nodes, then merges the results. This gives every node a{" "}
          <code>global view</code> of the graph in a single layer — unlike purely local
          architectures that need many layers stacked to propagate information across distant nodes.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Loopy</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Loopy adds a second structural channel on top of ordinary message passing by incorporating{" "}
          <code>r-neighborhood</code> path and cycle information. For each node, it finds all
          simple cycles up to length <code>r+2</code> that pass through it, and processes them
          with dedicated path convolutions. This gives the model explicit cycle awareness — standard
          GNNs provably cannot count cycles, but Loopy can. The trade-off is that it needs
          sufficient hidden dimensions to leverage this extra information effectively.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Primary sources</h2>
        <ol className="ml-5 mt-1.5 list-decimal space-y-2 text-textMuted">
          <li>
            Kipf and Welling, GCN:
            <a className="text-textMain" href="https://arxiv.org/abs/1609.02907">
              {" "}
              arXiv 1609.02907
            </a>
          </li>
          <li>
            Hamilton et al., GraphSAGE:
            <a className="text-textMain" href="https://arxiv.org/abs/1706.02216">
              {" "}
              arXiv 1706.02216
            </a>
          </li>
          <li>
            Xu et al., GIN:
            <a className="text-textMain" href="https://openreview.net/forum?id=ryGs6iA5Km">
              {" "}
              OpenReview ryGs6iA5Km
            </a>
          </li>
          <li>
            Rampasek et al., GraphGPS:
            <a
              className="text-textMain"
              href="https://proceedings.neurips.cc/paper_files/paper/2022/hash/6c1f4b5a7d7d5d8a9a7b4a9f99cc5a0a-Abstract-Conference.html"
            >
              {" "}
              NeurIPS 2022
            </a>
          </li>
          <li>
            Paolino et al., Weisfeiler and Leman Go Loopy:
            <a className="text-textMain" href="https://openreview.net/forum?id=9O2sVnEHor">
              {" "}
              OpenReview 9O2sVnEHor
            </a>
          </li>
        </ol>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/docs/gnns" label="How GNNs work" direction="back" />
        <DocsNextButton href="/docs/module-degree" label="Degree prediction" />
      </div>
    </DocsLayout>
  );
};

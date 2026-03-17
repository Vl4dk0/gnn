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
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          GCN - Graph Convolution Network
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          GCN uses degree-normalized aggregation. It is a strong baseline because it is stable and
          simple, but normalization can dampen raw counting signals.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          GraphSAGE - Sampling and Aggregation
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          GraphSAGE was designed as an inductive sample-and-aggregate architecture. By learning
          aggregator functions rather than fixed node embeddings, it generalizes to unseen nodes in
          dynamic graphs.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          GIN - Graph Isomorphism Network
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          GIN uses sum aggregation and then applies an MLP with a learnable self-weight term. It is
          one of the strongest 1-WL-style architectures for structural discrimination.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          GPS - General, Powerful, Scalable
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          GPS is a graph transformer recipe that combines local message passing with global
          attention. The idea is to keep convolution-style locality while adding a global context
          channel that scales to larger graphs.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Loopy</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Loopy adds a second structural channel on top of ordinary message passing by incorporating
          r-neighborhood path and cycle information. This improves cycle sensitivity for the
          minimum-cycle task.
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

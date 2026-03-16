import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { useFeatureFlags } from "../../hooks/useFeatureFlags";

export const DocsArchitecturePage = () => {
  const features = useFeatureFlags();

  return (
    <>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Architectures used in this project
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          Architecture choice matters because it changes the exact update rule: how neighbor signals
          are aggregated, how self-information is mixed in, and what structural distinctions survive
          through depth.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">GCN</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          GCN uses degree-normalized aggregation. This usually gives stable optimization and smooth
          representations, but normalization can dampen raw counting signals.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">GraphSAGE</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          GraphSAGE was designed as an inductive sample-and-aggregate architecture. In this project,
          the
          <code> SAGEConv</code> layers use <code>aggr="add"</code>, which helped preserve
          count-like signals.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">GIN</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          GIN also uses sum aggregation and then applies an MLP with a learnable self-weight term.
          It is one of the strongest 1-WL-style architectures for structural discrimination.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Loopy (r-lMPNN style)</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Loopy adds a second structural channel on top of ordinary message passing by incorporating
          r-neighborhood path/cycle information. This improves cycle sensitivity for this project’s
          target tasks.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Primary Sources</h2>
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
            Paolino et al., Weisfeiler and Leman Go Loopy:
            <a className="text-textMain" href="https://openreview.net/forum?id=9O2sVnEHor">
              {" "}
              OpenReview 9O2sVnEHor
            </a>
          </li>
        </ol>
      </DocsCard>

      <p className="mt-4 text-sm text-[#aaaaaa]">
        Code references:
        <a className="text-textMain" href="https://github.com/Vl4dk0/gnn/tree/main/ai/models">
          {" "}
          ai/models
        </a>
        ,
        <a
          className="text-textMain"
          href="https://github.com/Vl4dk0/gnn/blob/main/ai/models/loopy.py"
        >
          {" "}
          ai/models/loopy.py
        </a>
        ,
        <a
          className="text-textMain"
          href="https://github.com/Vl4dk0/gnn/blob/main/ai/utils/r_neighborhood.py"
        >
          {" "}
          ai/utils/r_neighborhood.py
        </a>
        .
      </p>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/docs/gnns" label="How GNNs work" direction="back" />
        <DocsNextButton href="/docs/module-degree" label="Degree prediction" />
      </div>
    </>
  );
};

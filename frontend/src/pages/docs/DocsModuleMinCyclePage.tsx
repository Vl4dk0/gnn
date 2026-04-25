import { Link } from "react-router-dom";

import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { Column, MetricTable } from "../../components/docs/MetricTable";
import { useDocsMetrics } from "../../hooks/useDocsMetrics";
import { useHighlight } from "../../hooks/useHighlight";
import { formatAccuracy, formatMetric } from "../shared/format";

export const DocsModuleMinCyclePage = () => {
  const { cycleModels } = useDocsMetrics();
  useHighlight();

  const columns: Column<NonNullable<typeof cycleModels>[number]>[] = [
    { key: "modelId", header: "Model", sortable: true, defaultDirection: "asc" },
    {
      key: "accuracy",
      header: "Accuracy",
      sortable: true,
      defaultDirection: "desc",
      render: (val) => formatAccuracy(val)
    },
    {
      key: "mae",
      header: "MAE",
      sortable: true,
      defaultDirection: "asc",
      render: (val) => formatMetric(val)
    },
    {
      key: "mse",
      header: "MSE",
      sortable: true,
      defaultDirection: "asc",
      render: (val) => formatMetric(val)
    }
  ];

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Minimum-cycle prediction: a harder structural task
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          Girth (or minimum cycle length) is a global property that cannot be computed by looking at
          immediate neighbors alone.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it interactively</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Draw a graph with cycles, or generate a random one, and see if the models can correctly
          identify the length of the shortest cycle.
        </p>
        <Link to="/min_cycle" className="ui-button-outline ui-surface-link mt-3">
          Open min-cycle editor
        </Link>
      </DocsCard>

      <DocsCard>
        <MetricTable
          wide
          columns={columns}
          data={cycleModels}
          emptyMessage="No models found"
          initialSort={{ key: "accuracy", direction: "desc" }}
        />
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Interpretation</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          With the initial small models (<code>h32</code>), none of the architectures managed to
          predict min-cycle with high accuracy (above 90%). <code>GIN</code> appeared to perform
          best at <code>71%</code>, but after testing it interactively in the editor, it became
          clear that it <code>cheats</code>: it almost always predicts either <code>0</code> (not
          part of a cycle) or <code>3</code> (triangle). Since random graphs contain many triangles
          and many nodes not in any cycle, this simple heuristic happens to be right often enough to
          inflate the accuracy. The model never learned to actually detect longer cycles.{" "}
          <code>LOOPY</code> — which was specifically designed for cycle-aware tasks — only
          reached <code>58%</code>. That was also a surprise.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          After training larger models on the{" "}
          <a
            href="https://www.hpc.tuke.sk/en/perun-supercomputer-info"
            className="text-accent1 hover:underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            PERUN supercomputer
          </a>
          , the picture changed significantly. <code>Loopy GNN</code> with larger hidden
          dimensions (<code>h128</code>) jumped to <code>86%</code> accuracy — the best result on
          this task. It turns out the cycle-aware r-neighborhood message passing needs sufficient
          model capacity to be effective. With only <code>32</code> hidden dimensions, the model simply could not
          afford to learn with this architecture.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          <code>GPS</code> with GCN convolution also improved notably to <code>56%</code> with
          the larger model, suggesting that the attention mechanism helps capture the global
          structure needed for girth detection.
        </p>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/docs/module-degree" label="Degree prediction" direction="back" />
        <DocsNextButton href="/docs/module-assessment" label="Assessment" />
      </div>
    </DocsLayout>
  );
};

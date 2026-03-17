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
        <a href="/min-cycle" className="ui-button-outline ui-surface-link mt-3">
          Open min-cycle editor
        </a>
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
          None of the architectures managed to learn to predict min-cycle with high accuracy. (above
          90%)
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Notice that <code>LOOPY</code> performs better than on degree prediction, but it is not
          the best.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The big surprise for me is that <code>GIN</code> performs better than <code>LOOPY</code>,
          which was specifically designed to handle such tasks.
        </p>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/docs/module-degree" label="Degree prediction" direction="back" />
        <DocsNextButton href="/docs/module-assessment" label="Assessment" />
      </div>
    </DocsLayout>
  );
};

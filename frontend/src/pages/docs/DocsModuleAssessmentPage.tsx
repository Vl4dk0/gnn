import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { Column, MetricTable } from "../../components/docs/MetricTable";
import { useDocsMetrics } from "../../hooks/useDocsMetrics";
import { formatAccuracy } from "../shared/format";

export const DocsModuleAssessmentPage = () => {
  const { degreeModels, cycleModels, assessmentRows } = useDocsMetrics();

  const unavailable = degreeModels === null && cycleModels === null;

  const columns: Column<(typeof assessmentRows)[number]>[] = [
    { key: "modelId", header: "Model", sortable: true, defaultDirection: "asc" },
    {
      key: "degreeAccuracy",
      header: "Degree Acc",
      sortable: true,
      defaultDirection: "desc",
      render: (val) => formatAccuracy(val)
    },
    {
      key: "minCycleAccuracy",
      header: "Min-cycle Acc",
      sortable: true,
      defaultDirection: "desc",
      render: (val) => formatAccuracy(val)
    },
    {
      key: "averageAccuracy",
      header: "Average",
      sortable: true,
      defaultDirection: "desc",
      render: (val) => formatAccuracy(val)
    }
  ];

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Assessment across both subtasks
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          Looking at one task in isolation is not enough. Degree prediction checks local counting;
          minimum-cycle prediction checks a more structural signal. Judging both together gives a
          more honest picture of the models in this thesis.
        </p>
      </DocsHero>

      <DocsCard>
        <MetricTable
          columns={columns}
          data={assessmentRows}
          emptyMessage="No models found"
          initialSort={{ key: "averageAccuracy", direction: "desc" }}
        />
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Interpretation</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The best performing model on average is <code>GIN</code>. After seeing these results, I
          can safely assume that <code>LOOPY</code> architecture requires more than 32 hidden
          dimensions. It was supposed to be performing the best out of all, but{" "}
          <code>falls short</code> in both tasks.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Training Parameters</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          {/* Across both tasks, models were trained with the same hyperparameters: - Epochs: 500 - */}
          {/* Hidden Dim: 32 - Num Layers: 4 - Dropout: 0.2 - Learning Rate: 0.001 - Input Dim: 4 - */}
          {/* Graphs per Epoch: 50 */}
          Machine Learning:
          <ul>
            <li>
              Learning rate: <code>0.001</code>
            </li>
            <li>
              Dropout: <code>0.2</code>
            </li>
            <li>
              Number of epochs: <code>500</code>
            </li>
            <li>
              Train on <code>50</code> random graphs <code>per epoch</code>
            </li>
          </ul>
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Graphs/Dynamic dataset:
          <ul>
            <li>
              Number of nodes per graph: between <code>5</code> and <code>20</code>
            </li>
            <li>
              Edge probability: between <code>0.15</code> and <code>0.6</code>
            </li>
          </ul>
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Graph Neural Network:
          <ul>
            <li>
              Number of layers: <code>4</code>;{" "}
              <div className="my-1 pl-2 text-sm leading-[1.5] text-textMuted">
                each layer corresponds to <code>one round</code> of message passing. The more we
                have, the more information about node's neighborhood the model can capture.
              </div>
            </li>
            <li>
              Input dimensions: <code>4</code>
              <div className="my-1 pl-2 text-sm leading-[1.5] text-textMuted">
                The size of the initial node features. The more dimensions, the more information
                about the nodes can be encoded at the start of training. We used{" "}
                <code>random features</code> for these tasks. Random features are a common choice in
                graph learning when we want to test the model's ability to learn from the graph
                structure itself, without relying on any specific node attributes.
              </div>
            </li>
            <li>
              Hidden dimensions: <code>32</code>
              <div className="my-1 pl-2 text-sm leading-[1.5] text-textMuted">
                The size of the node embeddings in each layer. These dimensions determine the{" "}
                <code>capacity</code> of the model to capture complex graph structures and node
                features.
              </div>
            </li>
          </ul>
        </p>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton
          href="/docs/module-min-cycle"
          label="Minimum-cycle prediction"
          direction="back"
        />
        <DocsNextButton href="/docs/module-cage" label="Cage generation" />
      </div>
    </DocsLayout>
  );
};

import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { Column, MetricTable } from "../../components/docs/MetricTable";
import { useDocsMetrics } from "../../hooks/useDocsMetrics";
import { formatAccuracy } from "../shared/format";

export const DocsModuleAssessmentPage = () => {
  const { assessmentRows } = useDocsMetrics();

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
          With small models (<code>h32_l4</code>), <code>GIN</code> dominates on average — it
          excels at degree prediction and leads on min-cycle. <code>LOOPY</code> with 32 hidden
          dimensions falls short in both tasks despite being designed for cycle detection.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          After training larger models on the PERUN supercomputer, the picture changes.{" "}
          <code>Loopy</code> with <code>h128</code> jumps to <code>86%</code> on min-cycle — the
          best result on that task — confirming that its r-neighborhood architecture simply needed
          more capacity. However, it remains unstable on degree prediction (a task where cycle
          awareness provides no benefit), which drags down its average.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          <code>GPS</code> models show mixed results: GPS with GIN convolution at <code>h128</code>
          {" "}reaches <code>98%</code> on degree but struggles on min-cycle, while GPS with GCN
          convolution does the opposite — weaker on degree but stronger on min-cycle.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          These prediction results inform the next step: <code>cage generation</code>. Since cage
          construction requires both local decisions (which edge to add) and global awareness
          (does adding this edge create a short cycle?), the architectures that capture global
          structure — GPS and, to some extent, GIN — are the natural candidates for the RL-based
          generation approach.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Training Parameters</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Machine Learning:
          <ul>
            <li>
              Learning rate: <code>0.001</code>
            </li>
            <li>
              Dropout: <code>0.2</code> — randomly disables neurons during training to prevent
              overfitting
            </li>
            <li>
              Number of epochs: <code>500</code> (h32 models), <code>5000</code> (h128 models)
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
          Graph Neural Network (parameters encoded in model names, e.g.{" "}
          <code>gin_h32_l4</code> = GIN with 32 hidden dims, 4 layers):
          <ul>
            <li>
              Number of layers (<code>l</code>):
              <div className="my-1 pl-2 text-sm leading-[1.5] text-textMuted">
                Each layer corresponds to <code>one round</code> of message passing. The more we
                have, the more information about node's neighborhood the model can capture. More
                layers also increase the risk of <code>over-smoothing</code>, where node embeddings
                become indistinguishable.
              </div>
            </li>
            <li>
              Hidden dimensions (<code>h</code>):
              <div className="my-1 pl-2 text-sm leading-[1.5] text-textMuted">
                The size of the node embeddings in each layer. Determines the{" "}
                <code>capacity</code> of the model to capture complex graph structures. Larger
                values allow more expressive representations but require more training data and
                compute.
              </div>
            </li>
            <li>
              Input dimensions: <code>4</code>
              <div className="my-1 pl-2 text-sm leading-[1.5] text-textMuted">
                The size of the initial node features. We used{" "}
                <code>random features</code> — a common choice in graph learning when testing the
                model's ability to learn from graph structure itself, without relying on specific
                node attributes.
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

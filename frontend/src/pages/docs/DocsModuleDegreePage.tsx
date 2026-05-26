import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { Column, MetricTable } from "../../components/docs/MetricTable";
import { useDocsMetrics } from "../../hooks/useDocsMetrics";
import { useHighlight } from "../../hooks/useHighlight";
import { formatAccuracy, formatMetric } from "../shared/format";

export const DocsModuleDegreePage = () => {
  const { degreeModels } = useDocsMetrics();
  useHighlight();

  const columns: Column<NonNullable<typeof degreeModels>[number]>[] = [
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
          Degree prediction
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          Degree is local, exact, and easy to verify. If a model cannot recover node degree
          reliably, there is little reason to trust it on harder structural targets.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it interactively</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The degree page lets you draw or generate graphs, run the selected model, and compare
          predicted values against the ground truth.
        </p>
        <a href="/degree" className="ui-button-outline ui-surface-link mt-3">
          Open degree editor
        </a>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          What the training loop is doing
        </h2>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`# ai/degree/train.py (simplified)
# We generate random graphs and extract node features to break symmetry
G = generate_random_graph()
degrees = torch.tensor([G.degree(i) for i in range(num_nodes)], dtype=torch.float)

# Rich node features for better learning:
# 1. Normalized node index (0 to 1)
# 2. Random 2D embedding
# 3. Clustering coefficient estimate
x = torch.cat([node_idx, random_emb, clustering], dim=1)

data = Data(x=x, edge_index=edge_index, y=degrees)

# Standard training step
out = model(data).squeeze()
loss = F.mse_loss(out, data.y)

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()`}</code>
        </pre>
        <p className="mt-3 text-base leading-[1.7] text-textMuted">
          The setup is deliberately simple: random graph generation, one scalar target per node, and
          regression loss. We use rich node features (random embeddings and clustering estimates) because if all nodes had the same feature vector, standard message passing would fail to distinguish topologically symmetric nodes. That simplicity makes the comparison across architectures easier to interpret.
        </p>
      </DocsCard>

      <DocsCard>
        <MetricTable
          wide
          columns={columns}
          data={degreeModels}
          emptyMessage="No models found"
          initialSort={{ key: "accuracy", direction: "desc" }}
        />
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Interpretation</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          As expected, most architectures can learn to predict degree with high accuracy.{" "}
          <code>GIN</code> and <code>SAGE</code> both achieve over <code>99%</code> even with
          small models.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          <code>GCN</code> and <code>LOOPY</code> perform poorly on this task.
          GCN struggles because of its simple mean aggregation. LOOPY's poor performance is more
          surprising — the r-neighborhood cycle detection machinery adds complexity that is
          unnecessary for degree counting, and the model exhibits numerical instability (exploding
          MSE values during evaluation) even with larger hidden dimensions. Degree prediction is a
          purely local task where Loopy's cycle awareness provides no benefit.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          After training larger models on PERUN, <code>GPS</code> with GIN convolution improved
          to <code>98%</code>, showing that the attention mechanism can help even on local tasks
          when given enough capacity. Interestingly, the larger <code>GIN</code> (<code>h128</code>)
          performed slightly worse than the smaller one — suggesting that for this simple task,
          more parameters can lead to overfitting.
        </p>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/docs/architecture" label="Architectures" direction="back" />
        <DocsNextButton href="/docs/module-min-cycle" label="Minimum-cycle prediction" />
      </div>
    </DocsLayout>
  );
};

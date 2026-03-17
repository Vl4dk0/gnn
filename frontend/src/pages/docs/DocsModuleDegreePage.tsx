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
G = generate_random_graph()
y = torch.tensor([G.degree(i) for i in range(num_nodes)], dtype=torch.float)

out = model(data).squeeze()
loss = F.mse_loss(out, data.y)

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()`}</code>
        </pre>
        <p className="mt-3 text-base leading-[1.7] text-textMuted">
          The setup is deliberately simple: random graph generation, one scalar target per node, and
          regression loss. That simplicity makes the comparison across architectures easier to
          interpret.
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
          As expected, most architectures can learn to predict degree with high accuracy.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Notice that <code>GCN</code> and <code>LOOPY</code> perform poorly. I expected GCN to
          struggle because of its simple aggregation, but LOOPY is <code>surprising</code> to me.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          I thought that the addition of r-neighborhood would only make it more powerful. It might
          be that I set the hidden dimenstions value too low (32), so the GNN cannot afford to learn
          with this architecture. Which if is the case, it is unfortunate, becuase I don't have
          access to better hardware for training.
        </p>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/docs/architecture" label="Architectures" direction="back" />
        <DocsNextButton href="/docs/module-min-cycle" label="Minimum-cycle prediction" />
      </div>
    </DocsLayout>
  );
};

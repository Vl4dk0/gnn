import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { MetricTable } from "../../components/docs/MetricTable";
import { useDocsMetrics } from "../../hooks/useDocsMetrics";
import { useFeatureFlags } from "../../hooks/useFeatureFlags";
import { useHighlight } from "../../hooks/useHighlight";
import { formatAccuracy, formatDate, formatMetric } from "../shared/format";

export const DocsModuleDegreePage = () => {
  const features = useFeatureFlags();
  const { degreeModels } = useDocsMetrics();
  useHighlight();

  return (
    <DocsLayout currentPath="/docs/module-degree.html" featureActive={features}>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Degree prediction: first trust check
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          I started here because degree is local and unambiguous. If a model cannot recover degree
          reliably, there is no reason to trust it on harder structural properties.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it interactively</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Open the degree interactive graph editor and test predictions directly on custom graphs.
        </p>
        <a
          href="/degree/index.html"
          className="ui-button-outline ui-surface-link mt-3"
        >
          Open degree editor
        </a>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          What exactly happens in the training loop?
        </h2>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-3.5">
          <code className="language-python">{`# ai/degree/train.py (simplified)
G = generate_random_graph(...)
y = torch.tensor([G.degree(i) for i in range(num_nodes)], dtype=torch.float)
out = model(data).squeeze()
loss = F.mse_loss(out, data.y)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          How well did the models solve it?
        </h2>
        <MetricTable
          wide
          headers={["Model", "Accuracy", "MAE", "MSE", "Created at"]}
          rows={
            degreeModels === null ? (
              <tr>
                <td colSpan={5} className="px-3 py-2.5 text-textMuted">
                  Live metrics unavailable
                </td>
              </tr>
            ) : degreeModels.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-3 py-2.5 text-textMuted">
                  No models found
                </td>
              </tr>
            ) : (
              degreeModels.map((model) => (
                <tr key={model.modelId}>
                  <td className="border-b border-line px-3 py-2.5 text-textMuted">
                    {model.modelId}
                  </td>
                  <td className="border-b border-line px-3 py-2.5 text-textMuted">
                    {formatAccuracy(model.accuracy)}
                  </td>
                  <td className="border-b border-line px-3 py-2.5 text-textMuted">
                    {formatMetric(model.mae)}
                  </td>
                  <td className="border-b border-line px-3 py-2.5 text-textMuted">
                    {formatMetric(model.mse)}
                  </td>
                  <td className="border-b border-line px-3 py-2.5 text-textMuted">
                    {formatDate(model.createdAt)}
                  </td>
                </tr>
              ))
            )
          }
        />
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Timestamped assessments</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          As of <code>2026-02-23</code>: <code>gin_v1</code> and <code>sage_v1</code> performed very
          well (100.00% exact-node accuracy), while <code>gcn_v1</code> and <code>loopy_r3_v1</code>
          were weaker on this degree setup.
        </p>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/docs/architecture.html" label="Architectures" direction="back" />
        <DocsNextButton href="/docs/module-min-cycle.html" label="Cycle prediction" />
      </div>
    </DocsLayout>
  );
};

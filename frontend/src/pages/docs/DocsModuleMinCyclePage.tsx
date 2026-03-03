import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { MetricTable } from "../../components/docs/MetricTable";
import { useDocsMetrics } from "../../hooks/useDocsMetrics";
import { useFeatureFlags } from "../../hooks/useFeatureFlags";
import { useHighlight } from "../../hooks/useHighlight";
import { formatAccuracy, formatDate, formatMetric } from "../shared/format";

export const DocsModuleMinCyclePage = () => {
  const features = useFeatureFlags();
  const { cycleModels } = useDocsMetrics();
  useHighlight();

  return (
    <DocsLayout currentPath="/docs/module-min-cycle.html" featureActive={features}>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Cycle prediction: moving from local to structural
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          After degree prediction, the next question was whether the model can recover a less local
          signal: the shortest cycle containing each node.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it interactively</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Open the cycle interactive graph editor and inspect min-cycle predictions on your own
          graphs.
        </p>
        <a
          href="/min_cycle/index.html"
          className="mt-3 inline-block rounded-md border border-line2 bg-bg1 px-4 py-2.5 text-sm font-semibold uppercase tracking-[0.7px] text-textMain no-underline transition-colors hover:border-textDim"
        >
          Open cycle editor
        </a>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          What changes in the training loop?
        </h2>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-3.5">
          <code className="language-python">{`# ai/min_cycle/train.py (simplified)
G = generate_random_graph(...)
y = torch.tensor([get_min_cycle(G, i) for i in range(num_nodes)], dtype=torch.float)
out = model(data).squeeze()
loss = F.mse_loss(out, data.y)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">How well did it work?</h2>
        <MetricTable
          wide
          headers={["Model", "Accuracy", "MAE", "MSE", "Created at"]}
          rows={
            cycleModels === null ? (
              <tr>
                <td colSpan={5} className="px-3 py-2.5 text-textMuted">
                  Live metrics unavailable
                </td>
              </tr>
            ) : cycleModels.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-3 py-2.5 text-textMuted">
                  No models found
                </td>
              </tr>
            ) : (
              cycleModels.map((model) => (
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
          As of <code>2026-02-23</code>: <code>loopy_r3_v1</code> performed clearly best on
          min-cycle prediction (85.14%), while <code>gcn_v1</code>, <code>gin_v1</code>, and{" "}
          <code>sage_v1</code>
          lagged behind.
        </p>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton
          href="/docs/module-degree.html"
          label="Degree prediction"
          direction="back"
        />
        <DocsNextButton href="/docs/module-assessment.html" label="Assessment" />
      </div>
    </DocsLayout>
  );
};

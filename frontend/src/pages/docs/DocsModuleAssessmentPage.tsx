import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { MetricTable } from "../../components/docs/MetricTable";
import { useDocsMetrics } from "../../hooks/useDocsMetrics";
import { useFeatureFlags } from "../../hooks/useFeatureFlags";
import { formatAccuracy } from "../shared/format";

export const DocsModuleAssessmentPage = () => {
  const features = useFeatureFlags();
  const { degreeModels, cycleModels, assessmentRows } = useDocsMetrics();

  const unavailable = degreeModels === null && cycleModels === null;

  return (
    <DocsLayout currentPath="/docs/module-assessment.html" featureActive={features}>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Assessment across degree and min-cycle
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          This page compares both prediction tasks together, because success on only one task is not
          enough to claim broad structural understanding.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Results side by side</h2>
        <MetricTable
          headers={["Model", "Degree Acc", "Min-cycle Acc"]}
          rows={
            unavailable ? (
              <tr>
                <td colSpan={3} className="px-3 py-2.5 text-textMuted">
                  Live metrics unavailable
                </td>
              </tr>
            ) : assessmentRows.length === 0 ? (
              <tr>
                <td colSpan={3} className="px-3 py-2.5 text-textMuted">
                  No models found
                </td>
              </tr>
            ) : (
              assessmentRows.map((row) => (
                <tr key={row.modelId}>
                  <td className="border-b border-line px-3 py-2.5 text-textMuted">{row.modelId}</td>
                  <td className="border-b border-line px-3 py-2.5 text-textMuted">
                    {formatAccuracy(row.degreeAccuracy)}
                  </td>
                  <td className="border-b border-line px-3 py-2.5 text-textMuted">
                    {formatAccuracy(row.minCycleAccuracy)}
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
          As of <code>2026-02-23</code>: no single model performed best across both tasks. Degree is
          led by
          <code> gin_v1</code>/<code>sage_v1</code>, while min-cycle is led by{" "}
          <code>loopy_r3_v1</code>.
        </p>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton
          href="/docs/module-min-cycle.html"
          label="Cycle prediction"
          direction="back"
        />
        <DocsNextButton href="/docs/module-cage.html" label="Cage generation" />
      </div>
    </DocsLayout>
  );
};

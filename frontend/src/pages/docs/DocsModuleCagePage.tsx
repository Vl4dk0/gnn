import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { useFeatureFlags } from "../../hooks/useFeatureFlags";
import { useHighlight } from "../../hooks/useHighlight";

export const DocsModuleCagePage = () => {
  const features = useFeatureFlags();
  useHighlight();

  return (
    <DocsLayout currentPath="/docs/module-cage.html" featureActive={features}>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Cage generation: the main objective
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          This is the end goal of the thesis. Prediction tasks were preparation; here I focus on
          constructing valid <code>(k,g)</code>-graphs.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it interactively</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Open the cage interactive graph editor to generate candidate graphs and monitor session
          progress.
        </p>
        <a
          href="/cage/index.html"
          className="ui-button-outline ui-surface-link mt-3"
        >
          Open cage editor
        </a>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Why I started with an A* idea
        </h2>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-3.5">
          <code className="language-python">{`# ai/cage/functions/astar.py
for succ_graph in successors:
    graph_h = graph_hash(succ_graph)
    if graph_h in self.visited_hashes:
        self.duplicates_skipped += 1
        continue
    self.visited_hashes.add(graph_h)
    succ_score = self._score_graph(succ_graph)
    heapq.heappush(self.pq, (-succ_score, self.counter, succ_graph))`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          What went wrong with the guided-A* dataset idea
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The bottleneck was data quality. I needed strong positive and negative examples of partial
          states, especially states near the real decision boundary.
        </p>
      </DocsCard>

      <DocsCard>
        <figure className="rounded-[10px] p-0">
          <img
            src="/static/boundary-examples.png"
            alt="Blue and red point classes separated by a line"
            className="block w-full rounded-lg border border-line2 bg-bg1"
          />
          <figcaption className="mt-2 text-[0.9rem] leading-[1.55] text-textMuted">
            Points close to the separator carry the strongest learning signal for the guide model.
          </figcaption>
        </figure>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Why I switched to PPO</h2>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-3.5">
          <code className="language-python">{`# ai/cage/rl/env.py (constants)
SUCCESS_REWARD = 20.0
INVALID_PENALTY = -0.05
ADD_REWARD = 0.1
REMOVE_PENALTY = -0.2
SATISFY_BONUS = 0.1`}</code>
        </pre>
      </DocsCard>

      <p className="mt-4 text-sm text-[#aaaaaa]">
        Code references:
        <a className="text-textMain" href="https://github.com/Vl4dk0/gnn/tree/main/ai/cage">
          {" "}
          ai/cage
        </a>
        ,
        <a
          className="text-textMain"
          href="https://github.com/Vl4dk0/gnn/blob/main/backend/utils/graph_utils.py"
        >
          {" "}
          backend/utils/graph_utils.py
        </a>
        ,
        <a
          className="text-textMain"
          href="https://github.com/Vl4dk0/gnn/blob/main/backend/routes/cage.py"
        >
          {" "}
          backend/routes/cage.py
        </a>
        .
      </p>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/docs/module-assessment.html" label="Assessment" direction="back" />
        <DocsNextButton href="/docs/training.html" label="Try it yourself" />
      </div>
    </DocsLayout>
  );
};

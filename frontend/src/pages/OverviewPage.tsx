import { DocsCard, DocsHero } from "../components/docs/DocsCard";
import { DocsLayout } from "../components/docs/DocsLayout";
import { DocsNextButton } from "../components/docs/DocsNextButton";
import { StoryCard } from "../components/docs/StoryCard";
import { useFeatureFlags } from "../hooks/useFeatureFlags";

const THESIS_AUTHOR = "Vladimír Jančár";
const THESIS_SUPERVISOR = "Ján Pastorek";

const STORY_ITEMS = [
  {
    href: "/docs/gnns.html",
    index: "1",
    title: "How GNNs work",
    description: "Message passing, layers, node and edge features, and readout."
  },
  {
    href: "/docs/architecture.html",
    index: "2",
    title: "Architectures",
    description: "GCN, SAGE, GIN, and Loopy with what each one preserves or loses."
  },
  {
    href: "/docs/module-degree.html",
    index: "3",
    title: "Degree prediction",
    description: "Why degree was the first check, how the loop works, and exact results."
  },
  {
    href: "/docs/module-min-cycle.html",
    index: "4",
    title: "Cycle prediction",
    description: "Node-level shortest-cycle labels, training behavior, and metrics."
  },
  {
    href: "/docs/module-assessment.html",
    index: "5",
    title: "Assessment",
    description: "Cross-task check of degree and min-cycle, with a dated snapshot."
  },
  {
    href: "/docs/module-cage.html",
    index: "6",
    title: "Cage generation",
    description: "Why I started with A*, what failed in data design, and why PPO."
  },
  {
    href: "/docs/training.html",
    index: "7",
    title: "Try it yourself",
    description: "Minimal local tutorial for degree prediction with one architecture."
  }
];

export const OverviewPage = () => {
  const features = useFeatureFlags();

  return (
    <DocsLayout currentPath="/" featureActive={features}>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Graph neural networks for algebraic graph theory
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          In this bachelor thesis, I study whether graph neural networks can help construct new{" "}
          <code>(k,g)</code>
          -graphs.
        </p>
        <div className="mt-5 border-t border-line pt-4 text-sm leading-[1.7] text-textMuted">
          <div className="ml-auto flex w-fit flex-col gap-2 text-left">
            <p>
              <span className="font-semibold text-textMain">Author:</span> {THESIS_AUTHOR}
            </p>
            <p>
              <span className="font-semibold text-textMain">Supervisor:</span> {THESIS_SUPERVISOR}
            </p>
          </div>
        </div>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">What do these terms mean?</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          A graph is <code>k</code>-regular when every vertex has exactly <code>k</code> neighbors.
          If
          <code> k=3</code>, every vertex has degree 3. The <code>girth</code> of a graph is the
          length of its shortest cycle.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          A <code>(k,g)</code>-graph satisfies both conditions at once: <code>k</code>-regularity
          and girth <code>g</code>. A <code>(k,g)</code>-cage is the graph with the minimum possible
          number of vertices among all <code>(k,g)</code>-graphs.
        </p>
      </DocsCard>

      <DocsCard>
        <img
          src="/static/kg-graph-terms.svg"
          alt="3-regular, girth-5, and 3-5 graph examples"
          className="block w-full rounded-lg"
        />
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Why this is hard</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          For example, if I were to generate <code>k</code>-regular graph, I would just create
          complete graph of <code>k+1</code> vertices. This is generally simple for any{" "}
          <code>k</code>.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          If I were to generate graphs with girth <code>g</code>, I would just create a cycle of
          length
          <code> g</code>. This is also generally simple for any <code>g</code>.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          But combining both constraints at once already becomes a hard combinatorial search problem
          in practice.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          I used a clear trick for each single constraint. The open question is whether there is an
          analogous trick for <code>(k,g)</code>-graphs. This is exactly where learned guidance may
          help.
        </p>
      </DocsCard>

      <section className="mb-4 grid grid-cols-[repeat(auto-fit,minmax(260px,1fr))] gap-3.5">
        {STORY_ITEMS.map((item) => (
          <StoryCard
            key={item.href}
            href={item.href}
            index={item.index}
            title={item.title}
            description={item.description}
          />
        ))}
      </section>

      <div className="mt-6 flex items-center justify-end border-t border-line pt-5">
        <DocsNextButton href="/docs/gnns.html" label="How GNNs work" />
      </div>
    </DocsLayout>
  );
};

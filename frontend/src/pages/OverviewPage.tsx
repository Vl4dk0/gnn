import { DocsCard, DocsHero } from "../components/docs/DocsCard";
import { DocsLayout } from "../components/docs/DocsLayout";
import { DocsNextButton } from "../components/docs/DocsNextButton";
import { StoryCard } from "../components/docs/StoryCard";

const THESIS_AUTHOR = "Vladimír Jančár";
const THESIS_SUPERVISOR = "Ján Pastorek";

const STORY_ITEMS = [
  {
    href: "/docs/gnns",
    index: 1,
    title: "How GNNs work",
    description: "Message passing, aggregation, and why structure matters."
  },
  {
    href: "/docs/architecture",
    index: 2,
    title: "Architectures",
    description: "GCN, GraphSAGE, GIN, GPS, and Loopy in one place."
  },
  {
    href: "/docs/module-degree",
    index: 3,
    title: "Degree prediction",
    description: "The first trust check: can the models recover a local signal?"
  },
  {
    href: "/docs/module-min-cycle",
    index: 4,
    title: "Minimum-cycle prediction",
    description: "The structural task: shortest-cycle labels per node."
  },
  {
    href: "/docs/module-assessment",
    index: 5,
    title: "Assessment",
    description: "What the results say when both tasks are judged together."
  },
  {
    href: "/docs/module-cage",
    index: 6,
    title: "Cage generation",
    description: "Why generation is difficult and what has been tried so far."
  },
  {
    href: "/docs/training",
    index: 7,
    title: "Do It Yourself",
    description: "A minimal example you can run in a clean folder."
  }
];

export const OverviewPage = () => {
  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Graph neural networks for algebraic graph theory
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          In this thesis, I study whether graph neural networks can learn structural signals that
          are useful for constructing new <code>(k,g)-graphs</code>. The long-term goal is
          generation, but the thesis builds toward it through prediction tasks first.
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
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">What are GNNs?</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          A regular Neural Network consumes fixed-size vectors. Those fail to capture the
          'unorganised' structure of a graph.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          A Graph Neural Network uses <code>node-features</code> to give network some extra
          information about graph's structure, this is mainly good for encoding helpful information,
          or passing random node-features to help network distinguish between two similar nodes.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Apart from that it uses <code>message passing</code> to help network understand each
          node's neighborhood. That message passing makes the model sensitive to graph structure
          rather than only to raw feature values.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Key definitions</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The number of node's neighbors is called its <code>degree</code>. A graph is{" "}
          <code>k</code>
          -regular if every node has degree <code>k</code>.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The <code>girth</code> of a graph is the length of its shortest cycle. The{" "}
          <code>minimum-cycle</code> label of a node is the length of the shortest cycle passing
          through that node.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          A <code>(k,g)-graph</code> is both <code>k</code>-regular and has girth <code>g</code>.
          Notice that degree is a local constraint; girth is a more global one. Combining both
          constraints makes generation hard, and constructing <code>(k,g)-graphs</code> remains an
          open problem.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          A <code>cage graph</code> is smallest possible <code>(k,g)-graph</code> for given{" "}
          <code>k</code> and <code>g</code>. Meaning it has the least number of nodes out of all
          (k,g)-graphs.
        </p>
      </DocsCard>

      <DocsCard>
        <img
          src="/static/kg-graph-terms.svg"
          alt="Examples illustrating regularity, girth, and a combined (k,g)-graph condition"
          className="block w-full rounded-lg"
        />
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Exploring this problem step by step
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          In this thesis I first test if GNN's would learn to <code>predict</code> degree, because
          degree is <code>local</code> and unambiguous.
          <br />
          Then I test minimum-cycle prediction, which requires a more structural,{" "}
          <code>global</code>, signal.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          I will be interested in what model-architecture will perform the best on average across
          these two subtasks. That architecture will in my opinion work best for GNN-guided{" "}
          <code>(k,g)-graph</code> generation.
        </p>
      </DocsCard>

      <section className="mb-4 grid grid-cols-[repeat(auto-fit,minmax(260px,1fr))] gap-3.5">
        {STORY_ITEMS.map((item) => (
          <StoryCard
            key={item.href}
            href={item.href}
            number={item.index}
            title={item.title}
            description={item.description}
          />
        ))}
      </section>

      <div className="mt-6 flex items-center justify-end border-t border-line pt-5">
        <DocsNextButton href="/docs/gnns" label="How GNNs work" />
      </div>
    </DocsLayout>
  );
};

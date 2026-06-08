import { DocsCard, DocsHero } from "../components/docs/DocsCard";
import { DocsLayout } from "../components/docs/DocsLayout";
import { DocsNextButton } from "../components/docs/DocsNextButton";
import { StoryCard } from "../components/docs/StoryCard";

const THESIS_AUTHOR = "Vladimír Jančár";
const THESIS_SUPERVISOR = "Ján Pastorek";

const STORY_ITEMS = [
  {
    href: "/degree-task",
    index: 1,
    title: "Degree subproblem",
    description: "Can a GNN recover vertex degree? k-regularity is half the definition."
  },
  {
    href: "/min-cycle-task",
    index: 2,
    title: "Min-cycle subproblem",
    description: "Predicting the shortest cycle per node: the girth constraint, a global signal."
  },
  {
    href: "/cage/astar",
    index: 3,
    title: "A* + backtracking",
    description: "Best-first search over partial graphs, and why labelling partial states is hard."
  },
  {
    href: "/cage/rl",
    index: 4,
    title: "Reinforcement learning",
    description: "Build a graph edge by edge with a learned, girth-masked policy."
  },
  {
    href: "/cage/voltage",
    index: 5,
    title: "Voltage lifts",
    description: "Stamp a tiny base graph over a group, and regularity comes for free."
  },
  {
    href: "/refinement",
    index: 6,
    title: "Refinement",
    description: "Edge swaps that remove a near-miss lift's short cycles while keeping every degree."
  },
  {
    href: "/excision",
    index: 7,
    title: "Excision",
    description: "Shrink a valid (k,g)-graph toward the cage by removing a tree and re-stitching."
  },
  {
    href: "/forge",
    index: 8,
    title: "Forge pipeline",
    description: "Lift, refine, and excise composed end to end, each method doing what it does best."
  }
];

export const OverviewPage = () => {
  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Generating (k,g)-graphs with graph neural networks
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          This thesis investigates what graph neural networks can and cannot learn about the two
          constraints that define a <code>(k,g)-graph</code> (k-regularity and girth), and then
          applies those findings to cage construction through methods of increasing algebraic
          structure.
        </p>
        <div className="mt-5 border-t border-line pt-4 text-sm leading-[1.7] text-textMuted">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <a
              href="/thesis.pdf"
              download
              className="ui-button-panel ui-surface-link rounded-md px-4 py-2 text-sm font-semibold text-textMain"
            >
              Download Thesis PDF
            </a>
            <div className="flex w-fit flex-col gap-2 text-left">
              <p>
                <span className="font-semibold text-textMain">Author:</span> {THESIS_AUTHOR}
              </p>
              <p>
                <span className="font-semibold text-textMain">Supervisor:</span> {THESIS_SUPERVISOR}
              </p>
            </div>
          </div>
        </div>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Key definitions</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The number of a node's neighbors is its <code>degree</code>. A graph is{" "}
          <code>k-regular</code> if every node has degree exactly <code>k</code>.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The <code>girth</code> of a graph is the length of its shortest cycle. The{" "}
          <code>minimum-cycle</code> label of a node is the length of the shortest cycle passing
          through that node (0 if the node lies on no cycle).
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          A <code>(k,g)-graph</code> is k-regular with girth at least <code>g</code>. A{" "}
          <code>cage</code> is the smallest possible (k,g)-graph (fewest vertices). The{" "}
          <code>Moore bound</code> gives a hard lower limit on the cage order. Closing the gap
          between the Moore bound and the best known constructions is an open problem in algebraic
          graph theory.
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
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">What this thesis does</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          A (k,g)-graph must satisfy two constraints simultaneously: every vertex has degree{" "}
          <code>k</code>, and no cycle is shorter than <code>g</code>. The thesis tests what GNNs
          can learn about each constraint separately: degree (local, directly readable from the
          neighbourhood) and girth (global, requiring path information across the graph), before
          moving to full cage construction.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Construction methods are explored in order of increasing algebraic structure: direct
          reinforcement learning builds a graph edge by edge, voltage lifts enforce regularity
          algebraically so only girth needs to be optimised, and excision shrinks a known valid graph
          toward the cage order by surgically removing a subtree and re-stitching the boundary.
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
        <DocsNextButton href="/degree-task" label="Degree subproblem" />
      </div>
    </DocsLayout>
  );
};

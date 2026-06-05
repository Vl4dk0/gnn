import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { EditorLinks } from "../../components/docs/EditorLinks";
import { useHighlight } from "../../hooks/useHighlight";

export const MinCycleTaskPage = () => {
  useHighlight();

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Min-cycle subproblem
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          The second subproblem is girth. Unlike degree: it depends on paths that leave a vertex and
          loop back, not just the local neighbourhood. This task tests the structural limit of GNN
          expressivity.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Why girth is harder than degree
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The per-node label here is the length of the shortest cycle through that vertex (0 if the
          vertex lies on no cycle). <br />
          Detecting whether any cycle exists (let alone its length) requires tracking paths of
          unbounded length, which is beyond what the standard 1-Weisfeiler–Leman colour-refinement
          test can distinguish. Standard message-passing GNNs are bounded by 1-WL, so this task sits
          exactly at the expressivity boundary.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Exact per-node label</h2>
        <p className="mb-2.5 text-base leading-[1.7] text-textMuted">
          For each vertex, the label is computed by temporarily removing each incident edge and
          finding the shortest detour back.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`def get_min_cycle(G: nx.Graph[int], vertex: int) -> int:
    ans: int | None = None
    for neigh in G.neighbors(vertex):
        G.remove_edge(vertex, neigh)                # break the direct edge
        try:
            path = nx.shortest_path(G, vertex, neigh)
            ans = min(ans, len(path)) if ans is not None else len(path)
        except nx.NetworkXNoPath:
            pass
        G.add_edge(vertex, neigh)                   # restore it
    return ans if ans is not None else 0            # 0 means the vertex is on no cycle`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Play with minimal-cycle prediction models in this editor.
          <br />
          You can choose prediction models in settings (gear icon in the top right corner of the
          editor)
        </p>
        <EditorLinks
          links={[
            {
              href: "/min_cycle",
              label: "Open"
            }
          ]}
        />
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/degree-task" direction="back" label="Degree subproblem" />
        <DocsNextButton href="/cage/astar" label="A* + backtracking" />
      </div>
    </DocsLayout>
  );
};

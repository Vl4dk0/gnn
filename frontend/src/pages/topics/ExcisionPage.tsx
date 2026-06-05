import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { useHighlight } from "../../hooks/useHighlight";

export const ExcisionTopicPage = () => {
  useHighlight();

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Excision
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          Start from a known valid (k,g)-graph and shrink it toward the cage order. Pick a root,
          BFS a subtree to the Moore radius, delete it, then re-stitch the boundary back to
          k-regular using only girth-safe edges.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">The Moore-radius tree</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The BFS tree is cut to depth <code>d = ⌊(g−1)/2⌋</code>, the Moore radius. At this
          depth the tree is guaranteed to be acyclic: any two paths leaving the root that meet
          earlier would close a cycle of length less than <code>g</code>, contradicting the girth
          assumption. Deleting the full tree never lowers girth — only the stitching edges added
          afterward can introduce a short cycle. So girth checking is concentrated entirely in
          the repair phase.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The repair connects each deficient boundary vertex (one that lost neighbours when the
          tree was removed) to other vertices at graph distance at least <code>g−1</code>, so
          the new edge cannot close a cycle shorter than <code>g</code>. Backtracking handles
          dead ends where no valid partner exists for a deficient vertex. The technique was
          introduced by Balaban, who cut a subtree from the (3,12)-cage to reach the (3,11)-cage.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`# ai/cage/excision/excise.py — remove the depth-d BFS tree
tree, q = {root}, deque([(root, 0)])
while q:
    node, dist = q.popleft()
    if dist < depth:                       # depth = (g - 1) // 2
        for nb in G.neighbors(node):
            if nb not in tree:
                tree.add(nb); q.append((nb, dist + 1))
reduced = G.copy(); reduced.remove_nodes_from(tree)
deficient = [v for v in reduced if reduced.degree(v) < k]   # lost neighbours`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Build or load a (k,g)-graph in the excision editor. Watch it highlight the BFS tree,
          remove it, and re-stitch the boundary step by step.
        </p>
        <a href="/excise" className="ui-button-solid ui-surface-link mt-3">
          Open the excision editor
        </a>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/cage/voltage" direction="back" label="Voltage lifts" />
      </div>
    </DocsLayout>
  );
};

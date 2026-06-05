import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { EditorLinks } from "../../components/docs/EditorLinks";
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
          assumption. Deleting the full tree never lowers girth. Only the stitching edges added
          afterward can introduce a short cycle, so girth checking is concentrated entirely in
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
          <code className="language-python">{`# ai/cage/excision/excise.py
def excise_tree(
    G: nx.Graph[int],
    root: int,
    depth: int,
) -> tuple[nx.Graph[int], list[int], dict[int, int]]:
    """BFS-remove a depth-d tree from G and return the reduced graph.

    For a (k,g)-graph use depth = (g - 1) // 2.
    Returns (reduced_graph, deficient_vertices, deficiency_levels).
    """
    tree_nodes: set[int] = set()
    visited: set[int] = set()
    queue: deque[tuple[int, int]] = deque([(root, 0)])
    visited.add(root)
    while queue:
        node, dist = queue.popleft()
        tree_nodes.add(node)
        if dist < depth:
            for nb in G.neighbors(node):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))

    k = max(d for _, d in G.degree()) if G.number_of_nodes() > 0 else 0
    reduced: nx.Graph[int] = G.copy()
    reduced.remove_nodes_from(tree_nodes)

    # Vertices outside the tree that were adjacent to a tree node
    # now have degree < k and need to be re-stitched
    deficient: list[int] = []
    deficiency_levels: dict[int, int] = {}
    for v in reduced.nodes():
        deg = reduced.degree(v)
        if deg < k:
            deficient.append(v)
            deficiency_levels[v] = k - deg

    deficient.sort()
    return reduced, deficient, deficiency_levels`}</code>
        </pre>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Only the stitching edges can violate girth, so each is distance-checked.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`# ai/cage/excision/legality.py
def is_legal_edge(G: nx.Graph[int], u: int, v: int, g_target: int) -> bool:
    """True iff adding edge (u, v) to G is legal for girth g_target.

    Adding (u, v) closes a cycle of length dist(u, v) + 1.
    For girth to stay >= g we need dist(u, v) >= g - 1.
    BFS from u up to g - 2 steps: if v is reachable, the cycle would be too short.
    """
    if u == v:
        return False
    if G.has_edge(u, v):
        return False
    dist = _bfs_distance(G, u, g_target - 2)
    if v in dist:
        return False   # dist(u, v) <= g - 2, new cycle length <= g - 1 < g
    return True`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it</h2>
        <p className="mb-3 text-base leading-[1.7] text-textMuted">
          Build or load a (k,g)-graph in the excision editor. Watch it highlight the BFS tree,
          remove it, and re-stitch the boundary step by step.
        </p>
        <EditorLinks
          links={[
            {
              href: "/excise",
              label: "Open"
            }
          ]}
        />
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/cage/voltage" direction="back" label="Voltage lifts" />
      </div>
    </DocsLayout>
  );
};

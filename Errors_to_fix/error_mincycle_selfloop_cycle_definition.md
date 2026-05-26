# get_min_cycle Returns Cycle Length Off by One

**Source/Occurrence:**
- `ai/min_cycle/functions/graph_service.py` (Lines 40-61, `get_min_cycle` function)
- `thesis/chapters/04-preparatory-tasks.typ` (Lines 14-18)

**Explanation:**
The thesis states: "For each vertex, the target is the length of the shortest cycle containing that vertex." A cycle of length $\ell$ contains $\ell$ vertices and $\ell$ edges.

In `get_min_cycle`, the algorithm:
1. Removes the edge between `vertex` and each neighbor `neigh`.
2. Finds the shortest path from `vertex` to `neigh` using `nx.shortest_path`.
3. Assigns `ans = len(path)`.

`nx.shortest_path` returns the list of **nodes** in the path. A path from `vertex` to `neigh` through $k$ intermediate nodes returns a list of length $k+2$ (including both endpoints). The actual cycle is this path plus the removed edge, giving cycle length = `len(path) - 1 + 1 = len(path)` edges.

Wait — let's verify:
- If there is a direct shortcut, e.g. `vertex -- A -- neigh` with the `vertex--neigh` edge removed: `nx.shortest_path` returns `[vertex, A, neigh]`, length 3. The cycle is `vertex → A → neigh → vertex`, which has **3 edges** (length 3). So `len(path) = 3` ✓.
- Smallest possible: `vertex -- neigh -- vertex` (triangle with another path): returns `[vertex, neigh]`, length 2. Cycle = 2 edges (the direct re-add would make a multi-edge, i.e. 2-cycle). But `nx.shortest_path` after edge removal would only return `[vertex, neigh]` if there's another direct edge between them, meaning a multi-graph. For simple graphs, `[vertex, neigh]` can't appear (no other path of length 1), so the minimum path length returned is 3 for a triangle.

The actual issue: for a triangle `vertex -- A -- B -- vertex`, after removing the `vertex--A` edge, the shortest path from `vertex` to `A` is `vertex → B → ... `— but this is for a different neighbor. The algorithm iterates over all neighbors and takes the minimum. The returned `len(path)` for each neighbor equals the cycle length correctly for simple graphs.

**Revised finding:** After careful analysis, `len(path)` does correctly compute the cycle length for simple graphs. However, the function has a subtle bug when the graph has **self-loops**: the check `G.has_edge(vertex, vertex)` on line 37 returns a cycle of length 1, but a self-loop creates a 1-cycle, which is not a standard cycle in graph theory. The thesis says "the length of the shortest cycle" — self-loops are typically excluded from cycle counts in simple graph theory.

**Actionable Steps:**
1. Clarify in the thesis whether self-loops (1-cycles) are included in the minimum cycle definition. If the training graphs are Erdős-Rényi simple graphs (no self-loops), this code path is never triggered and is harmless but potentially misleading.
2. If self-loops are excluded by definition (as in simple graph theory), remove the self-loop check on lines 37-38 or add a comment that this handles non-simple graphs only.
3. Add a docstring clarifying the graph type contract (simple vs multigraph) to prevent future confusion.

// Excision planner: shrink a valid (k,g)-graph toward its cage by removing a
// BFS tree and stitching the deficient boundary back to k-regularity.
//
// This mirrors the Python reference under ai/cage/excision:
//   - excise.py        BFS-grows a depth d = (g-1)//2 tree from a root, deletes
//                      all tree vertices, and reports the deficient set (degree
//                      < k = max degree of the original graph).
//   - legality.py      a candidate stitch edge (u, v) is legal iff
//                      dist(u, v) >= g - 1 in the current reduced graph, so the
//                      new cycle it closes has length >= g.
//   - baseline.py      repair via backtracking DFS with a most-constrained-first
//                      (minimum remaining values) pick of the deficient vertex
//                      and a least-constrained-partner tie-break.
//
// Everything here is pure TS (no React) and builds a watchable sequence of
// frames so the editor can animate root -> BFS growth -> removal -> stitching.

// Functional highlight colours. The site is achromatic, but these encode
// algorithm state, so they are intentionally vivid and distinct from each other
// and from the grey node fill, on both light and dark canvas backgrounds.
// Root red, BFS tree blue, newest BFS level purple, deficient amber, stitch green.
const COLOR_ROOT = "#d24b4b";
const COLOR_TREE = "#2f6fed";
const COLOR_TREE_DEEP = "#b14ddb";
const COLOR_DEFICIENT = "#cc8a00";
const COLOR_NEW_EDGE = "#2f9e6b";
const COLOR_STITCH_ENDPOINT = "#2f9e6b";

export interface ExcisionFrame {
  edgeList: string;
  nodeHighlights: Map<number, string>;
  edgeHighlights: Map<string, string>;
  caption: string;
}

export interface ExcisionPlan {
  frames: ExcisionFrame[];
  success: boolean;
  resultEdgeList: string;
  message: string;
  root: number | null;
  k: number;
  g: number;
  depth: number;
}

type Adjacency = Map<number, Set<number>>;

const edgeKey = (a: number, b: number): string => (a < b ? `${a}-${b}` : `${b}-${a}`);

/** Moore bound: minimum order of a (k,g)-graph. Mirrors moore_bound in the repo. */
export const mooreBound = (k: number, g: number): number => {
  if (k < 2 || g < 3) {
    return 0;
  }
  if (g % 2 === 1) {
    const r = (g - 1) / 2;
    let sum = 0;
    for (let i = 0; i < r; i += 1) {
      sum += (k - 1) ** i;
    }
    return 1 + k * sum;
  }
  const r = g / 2;
  let sum = 0;
  for (let i = 0; i < r; i += 1) {
    sum += (k - 1) ** i;
  }
  return 2 * sum;
};

/**
 * Parse an edge-list string ("a b" per edge, lone "a" for an isolated node)
 * into an adjacency map. Matches Graph.fromEdgeList's parsing rules.
 */
const parseEdgeList = (edgeList: string): Adjacency => {
  const adjacency: Adjacency = new Map();
  const ensure = (id: number): Set<number> => {
    let set = adjacency.get(id);
    if (!set) {
      set = new Set<number>();
      adjacency.set(id, set);
    }
    return set;
  };

  const trimmed = edgeList.trim();
  if (!trimmed) {
    return adjacency;
  }

  for (const rawLine of trimmed.split("\n")) {
    const parts = rawLine.trim().split(/\s+/);
    if (parts.length === 1) {
      const vertex = Number.parseInt(parts[0], 10);
      if (!Number.isNaN(vertex)) {
        ensure(vertex);
      }
      continue;
    }
    if (parts.length < 2) {
      continue;
    }
    const from = Number.parseInt(parts[0], 10);
    const to = Number.parseInt(parts[1], 10);
    if (Number.isNaN(from) || Number.isNaN(to) || from === to) {
      continue;
    }
    ensure(from).add(to);
    ensure(to).add(from);
  }

  return adjacency;
};

/** Serialize an adjacency map back to the canonical edge-list string. */
const toEdgeList = (adjacency: Adjacency): string => {
  const isolated: number[] = [];
  const edges: Array<{ a: number; b: number }> = [];
  const seen = new Set<string>();

  for (const [id, neighbors] of adjacency) {
    if (neighbors.size === 0) {
      isolated.push(id);
      continue;
    }
    for (const nb of neighbors) {
      const key = edgeKey(id, nb);
      if (!seen.has(key)) {
        seen.add(key);
        edges.push({ a: Math.min(id, nb), b: Math.max(id, nb) });
      }
    }
  }

  const lines: string[] = [];
  isolated.sort((a, b) => a - b).forEach((id) => lines.push(`${id}`));
  edges
    .sort((x, y) => (x.a !== y.a ? x.a - y.a : x.b - y.b))
    .forEach((edge) => lines.push(`${edge.a} ${edge.b}`));

  return lines.join("\n");
};

const cloneAdjacency = (adjacency: Adjacency): Adjacency => {
  const copy: Adjacency = new Map();
  for (const [id, neighbors] of adjacency) {
    copy.set(id, new Set(neighbors));
  }
  return copy;
};

/** Maximum degree over all vertices (the inferred k). */
export const inferK = (edgeList: string): number => {
  const adjacency = parseEdgeList(edgeList);
  let k = 0;
  for (const neighbors of adjacency.values()) {
    if (neighbors.size > k) {
      k = neighbors.size;
    }
  }
  return k;
};

/**
 * BFS distance from `source`, capped at `cutoff` hops (inclusive). Returns a
 * map {node -> distance}; nodes beyond the cutoff are omitted. Mirrors
 * _bfs_distance in legality.py.
 */
const bfsDistances = (adjacency: Adjacency, source: number, cutoff: number): Map<number, number> => {
  const dist = new Map<number, number>([[source, 0]]);
  const queue: number[] = [source];
  let head = 0;
  while (head < queue.length) {
    const u = queue[head];
    head += 1;
    const d = dist.get(u) ?? 0;
    if (d >= cutoff) {
      continue;
    }
    for (const nb of adjacency.get(u) ?? []) {
      if (!dist.has(nb)) {
        dist.set(nb, d + 1);
        queue.push(nb);
      }
    }
  }
  return dist;
};

/**
 * Shortest distance between u and v, capped at `cap`. Returns `cap` as a "too
 * far" sentinel when v is not reached within `cap` hops. Used for the
 * girth-safe legality test (dist >= g - 1).
 */
const distance = (adjacency: Adjacency, u: number, v: number, cap: number): number => {
  const dist = bfsDistances(adjacency, u, cap);
  return dist.get(v) ?? cap;
};

/**
 * Length of the shortest cycle (girth) in the graph, or Infinity if acyclic.
 * BFS from every vertex; a non-tree edge between same-level or cross vertices
 * closes a cycle whose length we track.
 */
const computeGirth = (adjacency: Adjacency): number => {
  let best = Infinity;
  for (const source of adjacency.keys()) {
    const dist = new Map<number, number>([[source, 0]]);
    const parent = new Map<number, number>([[source, -1]]);
    const queue: number[] = [source];
    let head = 0;
    while (head < queue.length) {
      const u = queue[head];
      head += 1;
      const du = dist.get(u) ?? 0;
      if (du * 2 + 1 >= best) {
        continue;
      }
      for (const nb of adjacency.get(u) ?? []) {
        if (!dist.has(nb)) {
          dist.set(nb, du + 1);
          parent.set(nb, u);
          queue.push(nb);
        } else if (parent.get(u) !== nb) {
          const cycle = du + (dist.get(nb) ?? 0) + 1;
          if (cycle < best) {
            best = cycle;
          }
        }
      }
    }
  }
  return best;
};

/**
 * Infer the girth of a graph from its edge list: the length of its shortest
 * cycle, or Infinity if the graph is acyclic.
 */
export const inferGirth = (edgeList: string): number => {
  return computeGirth(parseEdgeList(edgeList));
};

const isKRegular = (adjacency: Adjacency, k: number): boolean => {
  for (const neighbors of adjacency.values()) {
    if (neighbors.size !== k) {
      return false;
    }
  }
  return adjacency.size > 0;
};

/**
 * BFS-grow a tree to `depth` from `root`, returning the set of tree vertices
 * plus the depth at which each was discovered. Mirrors excise_tree's BFS.
 */
const growTree = (
  adjacency: Adjacency,
  root: number,
  depth: number
): { treeByLevel: number[][]; treeNodes: Set<number> } => {
  const treeNodes = new Set<number>([root]);
  const visited = new Set<number>([root]);
  const treeByLevel: number[][] = [[root]];
  let frontier = [root];

  for (let level = 1; level <= depth; level += 1) {
    const next: number[] = [];
    for (const node of frontier) {
      for (const nb of adjacency.get(node) ?? []) {
        if (!visited.has(nb)) {
          visited.add(nb);
          treeNodes.add(nb);
          next.push(nb);
        }
      }
    }
    treeByLevel.push(next);
    frontier = next;
    if (next.length === 0) {
      break;
    }
  }

  return { treeByLevel, treeNodes };
};

/** Count legal partners for v within candidate set (for the MRV heuristic). */
const countLegalPartners = (
  adjacency: Adjacency,
  v: number,
  candidates: Set<number>,
  g: number
): number => {
  let count = 0;
  for (const u of candidates) {
    if (u !== v && !adjacency.get(v)?.has(u) && distance(adjacency, u, v, g - 1) >= g - 1) {
      count += 1;
    }
  }
  return count;
};

interface StitchStep {
  u: number;
  v: number;
  undo: boolean;
}

/**
 * Backtracking repair: add edges among deficient vertices until every one is
 * back to degree k, never closing a cycle shorter than g. Records each accepted
 * (and undone) edge so the caller can animate the search. Bounded by
 * `maxExpansions` so it never hangs the browser. Mirrors backtracking_repair.
 */
const stitch = (
  reduced: Adjacency,
  deficiency: Map<number, number>,
  g: number,
  maxExpansions: number
): { graph: Adjacency | null; steps: StitchStep[]; exhausted: boolean } => {
  const graph = cloneAdjacency(reduced);
  const def = new Map(deficiency);
  const steps: StitchStep[] = [];
  let expansions = 0;

  const addEdge = (u: number, v: number): void => {
    graph.get(u)?.add(v);
    graph.get(v)?.add(u);
    def.set(u, (def.get(u) ?? 0) - 1);
    def.set(v, (def.get(v) ?? 0) - 1);
    if ((def.get(u) ?? 0) <= 0) def.delete(u);
    if ((def.get(v) ?? 0) <= 0) def.delete(v);
  };

  const removeEdge = (u: number, v: number, prevU: number, prevV: number): void => {
    graph.get(u)?.delete(v);
    graph.get(v)?.delete(u);
    def.set(u, prevU);
    def.set(v, prevV);
  };

  const solve = (): boolean => {
    if (def.size === 0) {
      return true;
    }
    expansions += 1;
    if (expansions > maxExpansions) {
      return false;
    }

    const deficientNow = Array.from(def.keys());
    const candidateSet = new Set(deficientNow);

    // Most-constrained deficient vertex (minimum remaining legal partners).
    let bestU: number | null = null;
    let bestCount = Infinity;
    for (const u of deficientNow) {
      const cnt = countLegalPartners(graph, u, candidateSet, g);
      if (cnt < bestCount) {
        bestCount = cnt;
        bestU = u;
      }
    }
    if (bestU === null || bestCount === 0) {
      return false;
    }
    const u = bestU;

    const partners = deficientNow.filter(
      (v) => v !== u && !graph.get(v)?.has(u) && distance(graph, u, v, g - 1) >= g - 1
    );
    // Least-constrained partner first.
    partners.sort(
      (a, b) =>
        countLegalPartners(graph, b, candidateSet, g) -
        countLegalPartners(graph, a, candidateSet, g)
    );

    for (const partner of partners) {
      const prevU = def.get(u) ?? 0;
      const prevV = def.get(partner) ?? 0;
      addEdge(u, partner);
      steps.push({ u, v: partner, undo: false });
      if (solve()) {
        return true;
      }
      removeEdge(u, partner, prevU, prevV);
      steps.push({ u, v: partner, undo: true });
      if (expansions > maxExpansions) {
        return false;
      }
    }
    return false;
  };

  const solved = solve();
  return {
    graph: solved ? graph : null,
    steps,
    exhausted: expansions > maxExpansions
  };
};

/**
 * Plan an excision animation for the given edge list and target girth g.
 *
 * Mirrors excise_and_repair / try_excise_root in
 * ai/cage/excision/repair_search.py: iterate over ALL vertices as candidate
 * roots (sorted ascending) and accept the FIRST root whose depth-d tree
 * removal can be re-stitched into a strictly smaller, k-regular, girth->=g
 * graph. If a specific `root` is passed, try only that root (manual mode).
 *
 * Frames stay watchable: each skipped/failed root contributes exactly one
 * compact "scan" frame on the original edge list (so the canvas does not churn
 * between roots), while the winning root gets the full detailed sequence (root
 * pick, per-level BFS growth, removal, per-edge stitching, success).
 */
export const planExcision = (edgeList: string, g: number, root?: number): ExcisionPlan => {
  // Per-root stitch budget. Generous enough to find a real solution on the
  // small hand-built graphs this is used for (a 20-vertex dodecahedron is the
  // largest expected), but bounded so trying every root never hangs the
  // browser.
  const MAX_EXPANSIONS = 8000;
  const adjacency = parseEdgeList(edgeList);
  const k = inferK(edgeList);
  const depth = (g - 1) >> 1;
  const frames: ExcisionFrame[] = [];
  const originalEdgeList = toEdgeList(adjacency);

  const fail = (message: string): ExcisionPlan => {
    frames.push({
      edgeList: originalEdgeList,
      nodeHighlights: new Map(),
      edgeHighlights: new Map(),
      caption: message
    });
    return {
      frames,
      success: false,
      resultEdgeList: originalEdgeList,
      message,
      root: root ?? null,
      k,
      g,
      depth
    };
  };

  if (adjacency.size === 0 || k < 2) {
    return fail("Build a (k,g)-graph first (need k >= 2).");
  }

  const allVertices = Array.from(adjacency.keys()).sort((a, b) => a - b);
  const minOrder = mooreBound(k, g);

  // Candidate roots: a single explicit root (manual mode) or every vertex.
  let candidateRoots: number[];
  if (root !== undefined) {
    if (!adjacency.has(root)) {
      return fail(`Root ${root} is not in the graph.`);
    }
    candidateRoots = [root];
  } else {
    candidateRoots = allVertices;
  }

  // For each candidate, excise its depth-d tree and compute the deficient set.
  const exciseFor = (
    candidate: number
  ): {
    treeByLevel: number[][];
    treeNodes: Set<number>;
    reduced: Adjacency;
    deficiency: Map<number, number>;
  } => {
    const { treeByLevel, treeNodes } = growTree(adjacency, candidate, depth);
    const reduced = cloneAdjacency(adjacency);
    for (const node of treeNodes) {
      for (const nb of reduced.get(node) ?? []) {
        reduced.get(nb)?.delete(node);
      }
      reduced.delete(node);
    }
    const deficiency = new Map<number, number>();
    for (const [id, neighbors] of reduced) {
      const missing = k - neighbors.size;
      if (missing > 0) {
        deficiency.set(id, missing);
      }
    }
    return { treeByLevel, treeNodes, reduced, deficiency };
  };

  // Compact one-frame scan render for a skipped / failed root: highlight the
  // root and its tree on the ORIGINAL edge list so the canvas stays stable.
  const pushScanFrame = (
    rootNode: number,
    treeNodes: Set<number>,
    caption: string
  ): void => {
    const nodeHighlights = new Map<number, string>();
    for (const node of treeNodes) {
      nodeHighlights.set(node, COLOR_TREE);
    }
    nodeHighlights.set(rootNode, COLOR_ROOT);
    frames.push({
      edgeList: originalEdgeList,
      nodeHighlights,
      edgeHighlights: new Map(),
      caption
    });
  };

  for (const rootNode of candidateRoots) {
    const { treeByLevel, treeNodes, reduced, deficiency } = exciseFor(rootNode);

    // Quick skips: nothing to stitch, or already below the Moore bound.
    if (deficiency.size === 0) {
      pushScanFrame(
        rootNode,
        treeNodes,
        `Root ${rootNode}: removing its tree leaves nothing to re-stitch, trying next`
      );
      continue;
    }
    if (reduced.size < minOrder) {
      pushScanFrame(
        rootNode,
        treeNodes,
        `Root ${rootNode}: would drop below the Moore bound ${minOrder}, trying next`
      );
      continue;
    }

    // Run the backtracking repair and validate the result.
    const { graph: repaired, steps } = stitch(reduced, deficiency, g, MAX_EXPANSIONS);

    let valid = false;
    if (repaired !== null) {
      const smaller = repaired.size < adjacency.size;
      const regular = isKRegular(repaired, k);
      const girth = computeGirth(repaired);
      const girthOk = girth === Infinity || girth >= g;
      valid = smaller && regular && girthOk;
    }

    if (repaired === null || !valid) {
      pushScanFrame(
        rootNode,
        treeNodes,
        `Root ${rootNode}: no girth-safe stitch, trying next`
      );
      continue;
    }

    // Winner: emit the full detailed sequence for this root, then return.
    const reducedEdgeList = toEdgeList(reduced);
    const deficientHighlights = new Map<number, string>();
    for (const id of deficiency.keys()) {
      deficientHighlights.set(id, COLOR_DEFICIENT);
    }

    // Frame: root chosen.
    frames.push({
      edgeList: originalEdgeList,
      nodeHighlights: new Map<number, string>([[rootNode, COLOR_ROOT]]),
      edgeHighlights: new Map(),
      caption: `Root ${rootNode} chosen (k=${k}, g=${g}, depth d=${depth})`
    });

    // Frames per BFS level (tree growth).
    const accumulated = new Set<number>([rootNode]);
    for (let level = 1; level < treeByLevel.length; level += 1) {
      for (const node of treeByLevel[level]) {
        accumulated.add(node);
      }
      const nodeHighlights = new Map<number, string>();
      for (const node of accumulated) {
        nodeHighlights.set(node, node === rootNode ? COLOR_ROOT : COLOR_TREE);
      }
      for (const node of treeByLevel[level]) {
        nodeHighlights.set(node, COLOR_TREE_DEEP);
      }
      frames.push({
        edgeList: originalEdgeList,
        nodeHighlights,
        edgeHighlights: new Map(),
        caption: `BFS level ${level}: tree has ${accumulated.size} vertices`
      });
    }

    // Frame: removal (switch to the reduced edge list).
    frames.push({
      edgeList: reducedEdgeList,
      nodeHighlights: new Map(deficientHighlights),
      edgeHighlights: new Map(),
      caption: `Removed tree (${treeNodes.size} vertices); ${deficiency.size} deficient vertices remain`
    });

    // Frames per accepted / undone stitch edge.
    const liveEdgeHighlights = new Map<string, string>();
    const liveNodeHighlights = new Map(deficientHighlights);
    const liveGraph = cloneAdjacency(reduced);

    for (const step of steps) {
      const key = edgeKey(step.u, step.v);
      if (step.undo) {
        liveGraph.get(step.u)?.delete(step.v);
        liveGraph.get(step.v)?.delete(step.u);
        liveEdgeHighlights.delete(key);
        frames.push({
          edgeList: toEdgeList(liveGraph),
          nodeHighlights: new Map(liveNodeHighlights),
          edgeHighlights: new Map(liveEdgeHighlights),
          caption: `Backtrack: undo ${step.u}-${step.v}`
        });
        continue;
      }
      liveGraph.get(step.u)?.add(step.v);
      liveGraph.get(step.v)?.add(step.u);
      liveEdgeHighlights.set(key, COLOR_NEW_EDGE);
      liveNodeHighlights.set(step.u, COLOR_STITCH_ENDPOINT);
      liveNodeHighlights.set(step.v, COLOR_STITCH_ENDPOINT);
      frames.push({
        edgeList: toEdgeList(liveGraph),
        nodeHighlights: new Map(liveNodeHighlights),
        edgeHighlights: new Map(liveEdgeHighlights),
        caption: `Stitch ${step.u}-${step.v} (distance was >= g-1)`
      });
    }

    // `repaired` is non-null and validated here (narrowed by the skip above).
    const resultEdgeList = toEdgeList(repaired);
    const message = `Success: smaller (k,g)-graph, ${repaired.size} vertices, girth >= ${g}`;
    const finalEdgeHighlights = new Map(liveEdgeHighlights);

    frames.push({
      edgeList: resultEdgeList,
      nodeHighlights: new Map(),
      edgeHighlights: finalEdgeHighlights,
      caption: message
    });

    return {
      frames,
      success: true,
      resultEdgeList,
      message,
      root: rootNode,
      k,
      g,
      depth
    };
  }

  // No root yielded a smaller valid (k,g)-graph.
  const message = `Tried ${candidateRoots.length} root${candidateRoots.length === 1 ? "" : "s"}; none yields a smaller (k,g)-graph.`;
  frames.push({
    edgeList: originalEdgeList,
    nodeHighlights: new Map(),
    edgeHighlights: new Map(),
    caption: message
  });
  return {
    frames,
    success: false,
    resultEdgeList: originalEdgeList,
    message,
    root: null,
    k,
    g,
    depth
  };
};

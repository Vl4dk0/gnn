// Refinement planner: drive a k-regular graph with cycles shorter than the
// target girth g toward girth >= g using degree-preserving edge swaps, then
// emit a watchable sequence of frames for the animated editor.
//
// This mirrors the classical (no-ML) tabu refiner under ai/cage/refine:
//   - cost.py   f(G) = sum_{c=3}^{g-1} 2^(g-c) * N_c, where N_c is the number
//               of simple cycles of length c (each undirected cycle counted
//               once). f = 0 exactly when girth >= g.
//   - swaps.py  a 2-switch removes edges (u,v),(x,y) (four distinct endpoints)
//               and adds either (u,x),(v,y) or (u,y),(v,x); a 3-switch rotates
//               three disjoint edges into a different degree-preserving
//               matching. Every endpoint loses one edge and gains one, so all
//               degrees are preserved.
//   - tabu.py   each iteration evaluates candidate 2-switches, applies the best
//               non-tabu one (with aspiration), marks it tabu, and escalates to
//               a one-shot 3-switch pass after a few non-improving iterations.
//
// Everything here is pure TS (no React). Refinement NEVER changes the vertex
// set or any vertex degree, so the editor can animate via updateFromEdgeList
// and keep node positions stable.

// Functional highlight colours. The site is achromatic, but these encode
// algorithm state, so they are intentionally vivid and distinct from each other
// and from the grey node fill, on both light and dark canvas backgrounds.
// Removed edges red, added edges green, short-cycle edges amber.
const COLOR_REMOVE = "#d24b4b";
const COLOR_ADD = "#2f9e6b";
const COLOR_CYCLE = "#cc8a00";

export interface RefineFrame {
  edgeList: string;
  nodeHighlights: Map<number, string>;
  edgeHighlights: Map<string, string>;
  caption: string;
}

export interface RefinePlan {
  frames: RefineFrame[];
  success: boolean;
  resultEdgeList: string;
  message: string;
  k: number;
  gTarget: number;
}

type Adjacency = Map<number, Set<number>>;

const edgeKey = (a: number, b: number): string => (a < b ? `${a}-${b}` : `${b}-${a}`);

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

/** Canonical list of undirected edges with a < b. */
const edgesList = (adjacency: Adjacency): Array<[number, number]> => {
  const seen = new Set<string>();
  const edges: Array<[number, number]> = [];
  for (const [id, neighbors] of adjacency) {
    for (const nb of neighbors) {
      const key = edgeKey(id, nb);
      if (!seen.has(key)) {
        seen.add(key);
        edges.push([Math.min(id, nb), Math.max(id, nb)]);
      }
    }
  }
  return edges;
};

const hasEdge = (adjacency: Adjacency, a: number, b: number): boolean =>
  adjacency.get(a)?.has(b) ?? false;

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
 * cycle, or Infinity if the graph is acyclic. Used for the editor's display.
 */
export const currentGirth = (edgeList: string): number => {
  return computeGirth(parseEdgeList(edgeList));
};

/**
 * Count simple cycles of each length from 3 to maxLen (inclusive). Each
 * undirected cycle is counted exactly once via a canonical convention: a cycle
 * is counted only from its smallest vertex, and we fix the direction by
 * requiring the first step to go to the smaller of the start vertex's two cycle
 * neighbours. Returns a map {length -> count}; missing lengths have count 0.
 *
 * Graphs here are small (<= ~30 vertices) and maxLen is small (<= ~7), so the
 * bounded DFS is cheap.
 */
const countCyclesUpTo = (adjacency: Adjacency, maxLen: number): Map<number, number> => {
  const counts = new Map<number, number>();
  for (let len = 3; len <= maxLen; len += 1) {
    counts.set(len, 0);
  }
  if (maxLen < 3) {
    return counts;
  }

  const path: number[] = [];
  const onPath = new Set<number>();

  // DFS that extends a simple path starting at `start`, only visiting vertices
  // strictly greater than `start` (so each cycle is rooted at its minimum) and
  // closing back to `start` when the length is in range. The "second" vertex
  // (the first step) is constrained to be the smaller of the two cycle
  // neighbours of `start`, which fixes one orientation and counts each cycle
  // once rather than twice.
  const dfs = (start: number, current: number): void => {
    // path.length is the number of vertices on the current path (>= 1). We may
    // still CLOSE a cycle when path.length === maxLen (the back-edge adds no
    // vertex), but we must not EXTEND to a new vertex beyond maxLen.
    const canExtend = path.length < maxLen;
    for (const nb of adjacency.get(current) ?? []) {
      if (nb === start) {
        // Closing the cycle. Its vertices are path = [start, path[1], ...,
        // path[last]] and the closing edge is path[last]-start. Orientation
        // fix: a length-L cycle is reachable as two mirror DFS walks (one via
        // each cycle-neighbour of start first); count only the walk whose
        // second vertex path[1] is the smaller of the two neighbours, i.e.
        // path[1] < path[last]. Both neighbours are > start by construction.
        if (path.length >= 3 && path[1] < path[path.length - 1]) {
          counts.set(path.length, (counts.get(path.length) ?? 0) + 1);
        }
        continue;
      }
      if (!canExtend || nb <= start || onPath.has(nb)) {
        continue;
      }
      path.push(nb);
      onPath.add(nb);
      dfs(start, nb);
      onPath.delete(nb);
      path.pop();
    }
  };

  for (const start of adjacency.keys()) {
    path.length = 0;
    onPath.clear();
    path.push(start);
    onPath.add(start);
    dfs(start, start);
    onPath.delete(start);
  }

  return counts;
};

/**
 * Weighted short-cycle cost: f(G) = sum_{c=3}^{g-1} 2^(g-c) * N_c, where N_c is
 * the number of simple cycles of length c. f = 0 exactly when girth >= g.
 */
const shortCycleCost = (adjacency: Adjacency, gTarget: number): number => {
  if (gTarget <= 3) {
    return 0;
  }
  const maxLen = gTarget - 1;
  const counts = countCyclesUpTo(adjacency, maxLen);
  let total = 0;
  for (let c = 3; c < gTarget; c += 1) {
    const nc = counts.get(c) ?? 0;
    if (nc === 0) {
      continue;
    }
    total += 2 ** (gTarget - c) * nc;
  }
  return total;
};

/** Edges of one shortest cycle (for the initial highlight), or [] if acyclic. */
const shortestCycleEdges = (adjacency: Adjacency): Array<[number, number]> => {
  let bestLen = Infinity;
  let bestCycle: number[] = [];

  for (const source of adjacency.keys()) {
    const dist = new Map<number, number>([[source, 0]]);
    const parent = new Map<number, number>([[source, -1]]);
    const queue: number[] = [source];
    let head = 0;
    while (head < queue.length) {
      const u = queue[head];
      head += 1;
      const du = dist.get(u) ?? 0;
      if (du * 2 + 1 >= bestLen) {
        continue;
      }
      for (const nb of adjacency.get(u) ?? []) {
        if (!dist.has(nb)) {
          dist.set(nb, du + 1);
          parent.set(nb, u);
          queue.push(nb);
        } else if (parent.get(u) !== nb) {
          const len = du + (dist.get(nb) ?? 0) + 1;
          if (len < bestLen) {
            bestLen = len;
            // Reconstruct the cycle: path source->u, path source->nb, plus (u,nb).
            const up: number[] = [];
            for (let w = u; w !== -1; w = parent.get(w) ?? -1) {
              up.push(w);
            }
            const vp: number[] = [];
            for (let w = nb; w !== -1; w = parent.get(w) ?? -1) {
              vp.push(w);
            }
            bestCycle = up.concat(vp.reverse());
          }
        }
      }
    }
  }

  if (bestCycle.length < 3) {
    return [];
  }
  const edges: Array<[number, number]> = [];
  for (let i = 0; i < bestCycle.length; i += 1) {
    const a = bestCycle[i];
    const b = bestCycle[(i + 1) % bestCycle.length];
    if (a !== b) {
      edges.push([Math.min(a, b), Math.max(a, b)]);
    }
  }
  return edges;
};

type Swap2Mode = "uxvy" | "uyvx";

interface Swap2 {
  u: number;
  v: number;
  x: number;
  y: number;
  mode: Swap2Mode;
}

type Swap3Mode = "abc" | "acb";

interface Swap3 {
  u: number;
  v: number;
  x: number;
  y: number;
  p: number;
  q: number;
  mode: Swap3Mode;
}

/** Removed / added edge pairs for a 2-switch, as canonical [min,max] keys. */
const swap2Edges = (
  swap: Swap2
): { removed: Array<[number, number]>; added: Array<[number, number]> } => {
  const { u, v, x, y, mode } = swap;
  const removed: Array<[number, number]> = [
    [Math.min(u, v), Math.max(u, v)],
    [Math.min(x, y), Math.max(x, y)]
  ];
  const added: Array<[number, number]> =
    mode === "uxvy"
      ? [
          [Math.min(u, x), Math.max(u, x)],
          [Math.min(v, y), Math.max(v, y)]
        ]
      : [
          [Math.min(u, y), Math.max(u, y)],
          [Math.min(v, x), Math.max(v, x)]
        ];
  return { removed, added };
};

/** Apply a 2-switch to a fresh copy of the adjacency. Preserves all degrees. */
const apply2Switch = (adjacency: Adjacency, swap: Swap2): Adjacency => {
  const h = cloneAdjacency(adjacency);
  const { removed, added } = swap2Edges(swap);
  for (const [a, b] of removed) {
    h.get(a)?.delete(b);
    h.get(b)?.delete(a);
  }
  for (const [a, b] of added) {
    h.get(a)?.add(b);
    h.get(b)?.add(a);
  }
  return h;
};

/** Removed / added edge pairs for a 3-switch, as canonical [min,max] keys. */
const swap3Edges = (
  swap: Swap3
): { removed: Array<[number, number]>; added: Array<[number, number]> } => {
  const { u, v, x, y, p, q, mode } = swap;
  const removed: Array<[number, number]> = [
    [Math.min(u, v), Math.max(u, v)],
    [Math.min(x, y), Math.max(x, y)],
    [Math.min(p, q), Math.max(p, q)]
  ];
  const added: Array<[number, number]> =
    mode === "abc"
      ? [
          [Math.min(u, x), Math.max(u, x)],
          [Math.min(v, p), Math.max(v, p)],
          [Math.min(y, q), Math.max(y, q)]
        ]
      : [
          [Math.min(u, x), Math.max(u, x)],
          [Math.min(v, q), Math.max(v, q)],
          [Math.min(y, p), Math.max(y, p)]
        ];
  return { removed, added };
};

/** Apply a 3-switch to a fresh copy of the adjacency. Preserves all degrees. */
const apply3Switch = (adjacency: Adjacency, swap: Swap3): Adjacency => {
  const h = cloneAdjacency(adjacency);
  const { removed, added } = swap3Edges(swap);
  for (const [a, b] of removed) {
    h.get(a)?.delete(b);
    h.get(b)?.delete(a);
  }
  for (const [a, b] of added) {
    h.get(a)?.add(b);
    h.get(b)?.add(a);
  }
  return h;
};

/**
 * Enumerate valid 2-switch candidates. A candidate is legal when all four
 * endpoints are distinct and BOTH added edges are absent (no multi-edge, no
 * self-loop). The list is capped at `cap`: if there would be more, we sample
 * random edge pairs instead of a full O(|E|^2) enumeration.
 */
const enumerate2Switches = (adjacency: Adjacency, cap: number): Swap2[] => {
  const edges = edgesList(adjacency);
  const n = edges.length;
  const out: Swap2[] = [];

  const tryPair = (i: number, j: number): void => {
    const [u, v] = edges[i];
    const [x, y] = edges[j];
    if (new Set([u, v, x, y]).size < 4) {
      return;
    }
    if (!hasEdge(adjacency, u, x) && !hasEdge(adjacency, v, y)) {
      out.push({ u, v, x, y, mode: "uxvy" });
    }
    if (!hasEdge(adjacency, u, y) && !hasEdge(adjacency, v, x)) {
      out.push({ u, v, x, y, mode: "uyvx" });
    }
  };

  // Full pair count (each yields up to two candidates). If small enough, do the
  // exact enumeration; otherwise sample distinct unordered pairs up to the cap.
  const fullPairs = (n * (n - 1)) / 2;
  if (fullPairs * 2 <= cap) {
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        tryPair(i, j);
      }
    }
    return out;
  }

  const tried = new Set<number>();
  let attempts = 0;
  const maxAttempts = cap * 6;
  while (out.length < cap && attempts < maxAttempts) {
    attempts += 1;
    const i = Math.floor(Math.random() * n);
    const j = Math.floor(Math.random() * n);
    if (i === j) {
      continue;
    }
    const lo = Math.min(i, j);
    const hi = Math.max(i, j);
    const code = lo * n + hi;
    if (tried.has(code)) {
      continue;
    }
    tried.add(code);
    tryPair(lo, hi);
  }
  return out;
};

/**
 * Sample valid 3-switch candidates: triples of pairwise-disjoint edges (6
 * distinct endpoints) rewired into a different degree-preserving matching, with
 * both alternative modes, keeping only those whose three added edges are all
 * absent. Bounded by sampling up to `cap` candidates.
 */
const sample3Switches = (adjacency: Adjacency, cap: number): Swap3[] => {
  const edges = edgesList(adjacency);
  const n = edges.length;
  const out: Swap3[] = [];
  if (n < 3) {
    return out;
  }

  let attempts = 0;
  const maxAttempts = cap * 10;
  while (out.length < cap && attempts < maxAttempts) {
    attempts += 1;
    const i = Math.floor(Math.random() * n);
    let j = Math.floor(Math.random() * n);
    let l = Math.floor(Math.random() * n);
    if (i === j || i === l || j === l) {
      continue;
    }
    const [u, v] = edges[i];
    const [x, y] = edges[j];
    const [p, q] = edges[l];
    if (new Set([u, v, x, y, p, q]).size < 6) {
      continue;
    }
    if (!hasEdge(adjacency, u, x) && !hasEdge(adjacency, v, p) && !hasEdge(adjacency, y, q)) {
      out.push({ u, v, x, y, p, q, mode: "abc" });
    }
    if (!hasEdge(adjacency, u, x) && !hasEdge(adjacency, v, q) && !hasEdge(adjacency, y, p)) {
      out.push({ u, v, x, y, p, q, mode: "acb" });
    }
  }
  return out;
};

/** Canonical tabu key for a 2-switch (its removed pair + added pair). */
const tabuKey = (swap: Swap2): string => {
  const { removed, added } = swap2Edges(swap);
  const r = removed
    .map((e) => `${e[0]}-${e[1]}`)
    .sort()
    .join(",");
  const a = added
    .map((e) => `${e[0]}-${e[1]}`)
    .sort()
    .join(",");
  return `${r}|${a}`;
};

const isKRegular = (adjacency: Adjacency, k: number): boolean => {
  for (const neighbors of adjacency.values()) {
    if (neighbors.size !== k) {
      return false;
    }
  }
  return adjacency.size > 0;
};

const CANDIDATE_CAP = 2000;
const SAMPLE_3SWITCH = 400;
const MAX_ITER = 200;
const STAGNATION_LIMIT = 3;

/**
 * Plan a refinement animation: drive the graph toward girth >= gTarget with
 * degree-preserving edge swaps using a classical tabu search (no GNN).
 *
 * Frames are emitted in order:
 *   1. an initial frame highlighting one shortest cycle,
 *   2. for each applied swap, a BEFORE frame (removed edges in red) and an
 *      AFTER frame (added edges in green),
 *   3. a final success / failure frame.
 *
 * Because refinement keeps the same vertices, the editor can animate with
 * updateFromEdgeList and preserve node positions.
 */
export const planRefine = (edgeList: string, gTarget: number): RefinePlan => {
  const adjacency = parseEdgeList(edgeList);
  const k = inferK(edgeList);
  const originalEdgeList = toEdgeList(adjacency);
  const frames: RefineFrame[] = [];

  const finish = (
    success: boolean,
    resultAdj: Adjacency,
    message: string
  ): RefinePlan => ({
    frames,
    success,
    resultEdgeList: toEdgeList(resultAdj),
    message,
    k,
    gTarget
  });

  if (adjacency.size === 0 || k < 2) {
    frames.push({
      edgeList: originalEdgeList,
      nodeHighlights: new Map(),
      edgeHighlights: new Map(),
      caption: "Build a k-regular graph first (need k >= 2)."
    });
    return finish(false, adjacency, "Build a k-regular graph first (need k >= 2).");
  }

  const tenure = Math.max(1, Math.ceil(Math.sqrt(edgesList(adjacency).length)));

  // --- Initial state ---
  const initialGirth = computeGirth(adjacency);
  const initialCost = shortCycleCost(adjacency, gTarget);
  const girthLabel = initialGirth === Infinity ? "inf" : `${initialGirth}`;

  if (initialCost === 0) {
    frames.push({
      edgeList: originalEdgeList,
      nodeHighlights: new Map(),
      edgeHighlights: new Map(),
      caption: `Already girth >= ${gTarget} (cost 0); nothing to refine.`
    });
    return finish(
      true,
      adjacency,
      `Already girth >= ${gTarget} (k=${k}, ${adjacency.size} vertices).`
    );
  }

  // Count short cycles for the caption (number of cycles below gTarget).
  const initCounts = countCyclesUpTo(adjacency, gTarget - 1);
  let shortCount = 0;
  for (let c = 3; c < gTarget; c += 1) {
    shortCount += initCounts.get(c) ?? 0;
  }

  const cycleHighlights = new Map<string, string>();
  for (const [a, b] of shortestCycleEdges(adjacency)) {
    cycleHighlights.set(edgeKey(a, b), COLOR_CYCLE);
  }
  frames.push({
    edgeList: originalEdgeList,
    nodeHighlights: new Map(),
    edgeHighlights: cycleHighlights,
    caption: `Girth ${girthLabel}, target ${gTarget}: ${shortCount} short cycle${
      shortCount === 1 ? "" : "s"
    } to remove (cost ${initialCost})`
  });

  // --- Tabu search ---
  let current = cloneAdjacency(adjacency);
  let currentCost = initialCost;
  let best = cloneAdjacency(current);
  let bestCost = currentCost;
  const tabu = new Map<string, number>();
  let stagnation = 0;

  // Emit the BEFORE / AFTER frame pair for an applied swap.
  const emitMove = (
    before: Adjacency,
    after: Adjacency,
    removed: Array<[number, number]>,
    added: Array<[number, number]>,
    prevCost: number,
    nextCost: number,
    label: string
  ): void => {
    const removeHl = new Map<string, string>();
    for (const [a, b] of removed) {
      removeHl.set(edgeKey(a, b), COLOR_REMOVE);
    }
    const removedDesc = removed.map(([a, b]) => `${a}-${b}`).join(" and ");
    frames.push({
      edgeList: toEdgeList(before),
      nodeHighlights: new Map(),
      edgeHighlights: removeHl,
      caption: `${label}: remove ${removedDesc}`
    });

    const addHl = new Map<string, string>();
    for (const [a, b] of added) {
      addHl.set(edgeKey(a, b), COLOR_ADD);
    }
    const addedDesc = added.map(([a, b]) => `${a}-${b}`).join(" and ");
    frames.push({
      edgeList: toEdgeList(after),
      nodeHighlights: new Map(),
      edgeHighlights: addHl,
      caption: `Add ${addedDesc} (cost ${prevCost} -> ${nextCost})`
    });
  };

  for (let iteration = 0; iteration < MAX_ITER; iteration += 1) {
    if (currentCost === 0) {
      break;
    }

    const candidates = enumerate2Switches(current, CANDIDATE_CAP);
    if (candidates.length === 0) {
      break;
    }

    let bestSwap: Swap2 | null = null;
    let bestCandidateCost = Infinity;
    for (const swap of candidates) {
      const isTabu = (tabu.get(tabuKey(swap)) ?? -1) > iteration;
      const candidateGraph = apply2Switch(current, swap);
      const cCost = shortCycleCost(candidateGraph, gTarget);
      const aspirate = cCost < bestCost;
      if (isTabu && !aspirate) {
        continue;
      }
      if (cCost < bestCandidateCost) {
        bestSwap = swap;
        bestCandidateCost = cCost;
      }
    }

    if (bestSwap === null) {
      stagnation += 1;
    } else {
      const prevCost = currentCost;
      tabu.set(tabuKey(bestSwap), iteration + tenure);
      const next = apply2Switch(current, bestSwap);
      const { removed, added } = swap2Edges(bestSwap);
      emitMove(current, next, removed, added, prevCost, bestCandidateCost, "Swap");
      current = next;
      currentCost = bestCandidateCost;
      if (currentCost < bestCost) {
        bestCost = currentCost;
        best = cloneAdjacency(current);
        stagnation = 0;
      } else {
        stagnation += 1;
      }
    }

    // --- 3-switch escalation ---
    if (stagnation >= STAGNATION_LIMIT && currentCost > 0) {
      stagnation = 0;
      let best3: Swap3 | null = null;
      let best3Cost = currentCost;
      for (const swap3 of sample3Switches(current, SAMPLE_3SWITCH)) {
        const g3 = apply3Switch(current, swap3);
        const c3 = shortCycleCost(g3, gTarget);
        if (c3 < best3Cost) {
          best3Cost = c3;
          best3 = swap3;
        }
      }
      if (best3 !== null) {
        const prevCost = currentCost;
        const next = apply3Switch(current, best3);
        const { removed, added } = swap3Edges(best3);
        emitMove(current, next, removed, added, prevCost, best3Cost, "3-swap");
        current = next;
        currentCost = best3Cost;
        if (currentCost < bestCost) {
          bestCost = currentCost;
          best = cloneAdjacency(current);
        }
      }
    }
  }

  // --- Final frame ---
  if (bestCost === 0) {
    const message = `Refined to girth >= ${gTarget} (k=${k}-regular, ${best.size} vertices).`;
    frames.push({
      edgeList: toEdgeList(best),
      nodeHighlights: new Map(),
      edgeHighlights: new Map(),
      caption: isKRegular(best, k)
        ? message
        : `${message} (warning: not perfectly ${k}-regular)`
    });
    return finish(true, best, message);
  }

  const remainingCounts = countCyclesUpTo(best, gTarget - 1);
  let remaining = 0;
  for (let c = 3; c < gTarget; c += 1) {
    remaining += remainingCounts.get(c) ?? 0;
  }
  const message = `Budget exhausted; ${remaining} short cycle${
    remaining === 1 ? "" : "s"
  } remain (cost ${bestCost}).`;
  frames.push({
    edgeList: toEdgeList(best),
    nodeHighlights: new Map(),
    edgeHighlights: new Map(),
    caption: message
  });
  return finish(false, best, message);
};

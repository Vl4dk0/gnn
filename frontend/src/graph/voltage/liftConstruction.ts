// Voltage graph lift construction (pure TypeScript, no React).
//
// A voltage graph lift takes a small BASE multigraph, a cyclic group Z_n,
// and a voltage assignment (one integer in 0..n-1 per directed base arc).
// The lift has vertex set V(base) x Z_n. Vertex (v, g) is encoded as
// vIndex * n + g. For each base arc u -> w with voltage alpha and for every
// g in 0..n-1 the lift has an undirected edge between (u, g) and
// (w, (g + alpha) mod n). If the base is k-regular the lift is k-regular.
//
// This mirrors ai/cage/voltage/lift.py build_lift(). For Z_n the reverse arc
// voltage is inv(v) = (n - v) % n; because the lift is an undirected SIMPLE
// graph, emitting only the forward direction's n edges per undirected base
// edge already produces every undirected lift edge (the reverse collapses to
// the same set). Parallel base edges and self-loops each contribute.
//
// Sanity check (Heawood graph): a dumbbell base of 2 nodes with 3 parallel
// arcs 0 -> 1 and voltages [1, 2, 4] over Z_7 yields 2 * 7 = 14 vertices,
// is 3-regular, and has girth 6 — this is the Heawood graph, the (3,6)-cage.
// (Voltages [4, 2, 1] are equivalent and also give girth 6.)

export interface BaseArc {
  id: number;
  from: number;
  to: number;
  voltage: number;
}

/** Non-negative modulo, so that ((g + voltage) mod n) is always in 0..n-1. */
const mod = (value: number, n: number): number => ((value % n) + n) % n;

/**
 * Build the lift edge list as a newline-separated string of "a b" lines
 * (undirected, deduped, no self-loops). Any isolated lift vertices are
 * emitted as single-number lines so the vertex count stays correct.
 *
 * Base node ids are mapped to contiguous indices via sorted order.
 */
export function buildLiftEdgeList(nodeIds: number[], arcs: BaseArc[], n: number): string {
  const sortedIds = [...nodeIds].sort((a, b) => a - b);
  const indexOf = new Map<number, number>();
  sortedIds.forEach((id, index) => {
    indexOf.set(id, index);
  });

  const order = Math.max(2, Math.floor(n));
  const edgeKeys = new Set<string>();
  const touched = new Set<number>();

  arcs.forEach((arc) => {
    const uIdx = indexOf.get(arc.from);
    const wIdx = indexOf.get(arc.to);
    if (uIdx === undefined || wIdx === undefined) {
      return;
    }

    const alpha = mod(arc.voltage, order);
    for (let g = 0; g < order; g += 1) {
      const a = uIdx * order + g;
      const b = wIdx * order + mod(g + alpha, order);
      if (a === b) {
        continue; // degenerate self-loop
      }
      const lo = Math.min(a, b);
      const hi = Math.max(a, b);
      edgeKeys.add(`${lo} ${hi}`);
      touched.add(a);
      touched.add(b);
    }
  });

  const lines: string[] = [];

  // Isolated lift vertices (no incident edge) as single-number lines.
  const total = sortedIds.length * order;
  for (let v = 0; v < total; v += 1) {
    if (!touched.has(v)) {
      lines.push(`${v}`);
    }
  }

  const edges = Array.from(edgeKeys)
    .map((key) => {
      const [a, b] = key.split(" ").map((part) => Number.parseInt(part, 10));
      return { a, b };
    })
    .sort((x, y) => (x.a !== y.a ? x.a - y.a : x.b - y.b));

  edges.forEach((edge) => {
    lines.push(`${edge.a} ${edge.b}`);
  });

  return lines.join("\n");
}

/**
 * Map each lift vertex id (vIndex * n + g) to a "(baseId,g)" display label.
 * Mirrors the encoding used by buildLiftEdgeList so labels and edges agree.
 */
export function buildLiftLabels(nodeIds: number[], n: number): Map<number, string> {
  const sortedIds = [...nodeIds].sort((a, b) => a - b);
  const order = Math.max(2, Math.floor(n));
  const labels = new Map<number, string>();
  sortedIds.forEach((baseId, vIdx) => {
    for (let g = 0; g < order; g += 1) {
      labels.set(vIdx * order + g, `(${baseId},${g})`);
    }
  });
  return labels;
}

/** Number of vertices in the lift: |V(base)| * n. */
export function liftVertexCount(nodeIds: number[], n: number): number {
  return nodeIds.length * Math.max(2, Math.floor(n));
}

/**
 * Shortest cycle length (girth) of an undirected graph given as an edge list.
 * Runs a BFS from every vertex tracking distance and parent; a cross edge to
 * an already-visited non-parent vertex closes a cycle of length
 * dist[u] + dist[v] + 1. Returns the global minimum, or 0 if the graph is
 * acyclic. Suitable for graphs up to a few hundred vertices.
 */
export function computeGirth(edgeList: string): number {
  const adjacency = new Map<number, Set<number>>();

  const ensure = (node: number): Set<number> => {
    let set = adjacency.get(node);
    if (!set) {
      set = new Set<number>();
      adjacency.set(node, set);
    }
    return set;
  };

  edgeList
    .trim()
    .split("\n")
    .forEach((line) => {
      const parts = line.trim().split(/\s+/);
      if (parts.length < 2) {
        if (parts.length === 1 && parts[0] !== "") {
          const single = Number.parseInt(parts[0], 10);
          if (!Number.isNaN(single)) {
            ensure(single);
          }
        }
        return;
      }
      const a = Number.parseInt(parts[0], 10);
      const b = Number.parseInt(parts[1], 10);
      if (Number.isNaN(a) || Number.isNaN(b) || a === b) {
        return;
      }
      ensure(a).add(b);
      ensure(b).add(a);
    });

  const nodes = Array.from(adjacency.keys());
  let best = Infinity;

  for (const source of nodes) {
    const dist = new Map<number, number>();
    const parent = new Map<number, number>();
    const queue: number[] = [source];
    dist.set(source, 0);
    parent.set(source, -1);

    let head = 0;
    while (head < queue.length) {
      const current = queue[head];
      head += 1;
      const currentDist = dist.get(current) ?? 0;

      if (currentDist >= best) {
        // Cannot improve the global best from deeper in this BFS.
        break;
      }

      for (const neighbor of adjacency.get(current) ?? []) {
        if (!dist.has(neighbor)) {
          dist.set(neighbor, currentDist + 1);
          parent.set(neighbor, current);
          queue.push(neighbor);
        } else if (parent.get(current) !== neighbor) {
          const cycle = currentDist + (dist.get(neighbor) ?? 0) + 1;
          if (cycle < best) {
            best = cycle;
          }
        }
      }
    }
  }

  return best === Infinity ? 0 : best;
}

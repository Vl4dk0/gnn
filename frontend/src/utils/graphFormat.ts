interface EdgePair {
  v1: number;
  v2: number;
}

export const normalizeEdgeList = (graphStr: string): string => {
  if (!graphStr.trim()) return "";

  const lines = graphStr.trim().split("\n");
  const isolatedVertices = new Set<number>();
  const edges: EdgePair[] = [];

  lines.forEach((line) => {
    const parts = line.trim().split(/\s+/);

    if (parts.length === 1 && parts[0] !== "") {
      const v = Number.parseInt(parts[0], 10);
      if (!Number.isNaN(v)) isolatedVertices.add(v);
      return;
    }

    if (parts.length < 2) return;

    const v1 = Number.parseInt(parts[0], 10);
    const v2 = Number.parseInt(parts[1], 10);

    if (Number.isNaN(v1) || Number.isNaN(v2)) return;

    edges.push({
      v1: Math.min(v1, v2),
      v2: Math.max(v1, v2)
    });
  });

  edges.forEach((edge) => {
    isolatedVertices.delete(edge.v1);
    isolatedVertices.delete(edge.v2);
  });

  const unique = new Set<string>();
  const output: string[] = [];

  Array.from(isolatedVertices)
    .sort((a, b) => a - b)
    .forEach((node) => {
      output.push(`${node}`);
    });

  edges
    .sort((a, b) => {
      if (a.v1 !== b.v1) return a.v1 - b.v1;
      return a.v2 - b.v2;
    })
    .forEach((edge) => {
      const key = `${edge.v1},${edge.v2}`;
      if (unique.has(key)) return;
      unique.add(key);
      output.push(`${edge.v1} ${edge.v2}`);
    });

  return output.join("\n");
};

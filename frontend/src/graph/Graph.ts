import { GraphNode } from "./GraphNode";
import type { GraphPrediction } from "../types/graph";

interface GraphEdge {
  from: GraphNode;
  to: GraphNode;
}

export class Graph {
  public nodes: GraphNode[] = [];
  public nextNodeId = 0;

  public addNode(x: number, y: number): GraphNode {
    const usedIds = new Set(this.nodes.map((node) => node.id));
    let newId = 0;

    while (usedIds.has(newId)) {
      newId += 1;
    }

    const node = new GraphNode(newId, x, y);
    this.nodes.push(node);
    this.nextNodeId = Math.max(this.nextNodeId, newId + 1);
    return node;
  }

  public removeNode(node: GraphNode): void {
    this.nodes.forEach((otherNode) => {
      otherNode.removeNeighbor(node);
    });

    const index = this.nodes.indexOf(node);
    if (index !== -1) {
      this.nodes.splice(index, 1);
    }
  }

  public addEdge(nodeA: GraphNode, nodeB: GraphNode): GraphEdge | null {
    if (nodeA.hasNeighbor(nodeB)) {
      this.removeEdge(nodeA, nodeB);
      return null;
    }

    nodeA.addNeighbor(nodeB);
    nodeB.addNeighbor(nodeA);
    return { from: nodeA, to: nodeB };
  }

  public removeEdge(nodeA: GraphNode, nodeB: GraphNode): void {
    nodeA.removeNeighbor(nodeB);
    nodeB.removeNeighbor(nodeA);
  }

  public findNodeAt(x: number, y: number): GraphNode | null {
    for (let i = this.nodes.length - 1; i >= 0; i -= 1) {
      if (this.nodes[i].contains(x, y)) {
        return this.nodes[i];
      }
    }

    return null;
  }

  public clear(): void {
    this.nodes = [];
    this.nextNodeId = 0;
  }

  public clearPredictions(): void {
    this.nodes.forEach((node) => {
      node.clearPrediction();
    });
  }

  public getEdges(): GraphEdge[] {
    const edges: GraphEdge[] = [];
    const seen = new Set<string>();

    this.nodes.forEach((node) => {
      node.neighbors.forEach((neighbor) => {
        const key =
          node.id < neighbor.id ? `${node.id}-${neighbor.id}` : `${neighbor.id}-${node.id}`;

        if (!seen.has(key)) {
          seen.add(key);
          edges.push({ from: node, to: neighbor });
        }
      });
    });

    return edges;
  }

  public toEdgeList(): string {
    const verticesWithEdges = new Set<GraphNode>();

    this.nodes.forEach((node) => {
      if (node.neighbors.length > 0) {
        verticesWithEdges.add(node);
      }
    });

    const lines: string[] = [];

    this.nodes
      .filter((node) => !verticesWithEdges.has(node))
      .map((node) => node.id)
      .sort((a, b) => a - b)
      .forEach((id) => {
        lines.push(`${id}`);
      });

    this.getEdges()
      .map((edge) => ({
        v1: Math.min(edge.from.id, edge.to.id),
        v2: Math.max(edge.from.id, edge.to.id)
      }))
      .sort((a, b) => {
        if (a.v1 !== b.v1) return a.v1 - b.v1;
        return a.v2 - b.v2;
      })
      .forEach((edge) => {
        lines.push(`${edge.v1} ${edge.v2}`);
      });

    return lines.join("\n");
  }

  public fromEdgeList(edgeListText: string, canvasWidth: number, canvasHeight: number): void {
    this.clear();

    if (!edgeListText.trim()) {
      return;
    }

    const lines = edgeListText.trim().split("\n");
    const nodeIds = new Set<number>();
    const edgeData: Array<{ from: number; to: number }> = [];

    lines.forEach((line) => {
      const parts = line.trim().split(/\s+/);

      if (parts.length === 1) {
        const vertex = Number.parseInt(parts[0], 10);
        if (!Number.isNaN(vertex)) {
          nodeIds.add(vertex);
        }
        return;
      }

      if (parts.length < 2) {
        return;
      }

      const from = Number.parseInt(parts[0], 10);
      const to = Number.parseInt(parts[1], 10);

      if (Number.isNaN(from) || Number.isNaN(to)) {
        return;
      }

      nodeIds.add(from);
      nodeIds.add(to);
      edgeData.push({ from, to });
    });

    const sortedIds = Array.from(nodeIds).sort((a, b) => a - b);
    const centerX = canvasWidth / 2;
    const centerY = canvasHeight / 2;
    const idToNode = new Map<number, GraphNode>();

    sortedIds.forEach((id) => {
      const jitterX = (Math.random() - 0.5) * 10;
      const jitterY = (Math.random() - 0.5) * 10;
      const node = new GraphNode(id, centerX + jitterX, centerY + jitterY);
      this.nodes.push(node);
      idToNode.set(id, node);
    });

    this.nextNodeId = sortedIds.length ? Math.max(...sortedIds) + 1 : 0;

    edgeData.forEach(({ from, to }) => {
      const fromNode = idToNode.get(from);
      const toNode = idToNode.get(to);

      if (fromNode && toNode) {
        fromNode.addNeighbor(toNode);
        toNode.addNeighbor(fromNode);
      }
    });
  }

  /**
   * Update graph from an edge list, preserving positions of existing nodes.
   * Only adds/removes the diff — nodes that already exist keep their
   * current (x, y) from the physics simulation.
   */
  public updateFromEdgeList(edgeListText: string, canvasWidth: number, canvasHeight: number): void {
    if (!edgeListText.trim()) {
      this.clear();
      return;
    }

    const lines = edgeListText.trim().split("\n");
    const newNodeIds = new Set<number>();
    const newEdges: Array<{ from: number; to: number }> = [];

    lines.forEach((line) => {
      const parts = line.trim().split(/\s+/);
      if (parts.length === 1) {
        const vertex = Number.parseInt(parts[0], 10);
        if (!Number.isNaN(vertex)) newNodeIds.add(vertex);
        return;
      }
      if (parts.length < 2) return;
      const from = Number.parseInt(parts[0], 10);
      const to = Number.parseInt(parts[1], 10);
      if (Number.isNaN(from) || Number.isNaN(to)) return;
      newNodeIds.add(from);
      newNodeIds.add(to);
      newEdges.push({ from, to });
    });

    // Build lookup of current nodes by id
    const existingById = new Map<number, GraphNode>();
    for (const node of this.nodes) {
      existingById.set(node.id, node);
    }

    // Remove nodes that are no longer present
    const toRemove = this.nodes.filter((n) => !newNodeIds.has(n.id));
    for (const node of toRemove) {
      this.removeNode(node);
    }

    // Add new nodes (ones not already present) near center with jitter
    const centerX = canvasWidth / 2;
    const centerY = canvasHeight / 2;
    const idToNode = new Map<number, GraphNode>();
    for (const node of this.nodes) {
      idToNode.set(node.id, node);
    }

    const sortedNewIds = Array.from(newNodeIds).sort((a, b) => a - b);
    for (const id of sortedNewIds) {
      if (!idToNode.has(id)) {
        const jitterX = (Math.random() - 0.5) * 10;
        const jitterY = (Math.random() - 0.5) * 10;
        const node = new GraphNode(id, centerX + jitterX, centerY + jitterY);
        this.nodes.push(node);
        idToNode.set(id, node);
      }
    }

    this.nextNodeId = sortedNewIds.length ? Math.max(...sortedNewIds) + 1 : 0;

    // Build the target edge set
    const targetEdgeSet = new Set<string>();
    for (const { from, to } of newEdges) {
      const key = from < to ? `${from}-${to}` : `${to}-${from}`;
      targetEdgeSet.add(key);
    }

    // Remove edges that shouldn't exist anymore
    const currentEdges = this.getEdges();
    for (const edge of currentEdges) {
      const key = edge.from.id < edge.to.id
        ? `${edge.from.id}-${edge.to.id}`
        : `${edge.to.id}-${edge.from.id}`;
      if (!targetEdgeSet.has(key)) {
        this.removeEdge(edge.from, edge.to);
      }
    }

    // Add edges that are missing
    for (const { from, to } of newEdges) {
      const fromNode = idToNode.get(from);
      const toNode = idToNode.get(to);
      if (fromNode && toNode && !fromNode.hasNeighbor(toNode)) {
        fromNode.addNeighbor(toNode);
        toNode.addNeighbor(fromNode);
      }
    }
  }

  public updatePredictions(predictions: GraphPrediction[]): void {
    this.clearPredictions();

    predictions.forEach((prediction) => {
      const node = this.nodes.find((candidate) => candidate.id === prediction.nodeId);
      if (node) {
        node.setPrediction(prediction.predicted, prediction.actual);
      }
    });
  }

  public hasPredictions(): boolean {
    return this.nodes.some((node) => node.hasPrediction());
  }
}

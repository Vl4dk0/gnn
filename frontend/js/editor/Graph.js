/**
 * Graph - Manages nodes and edges using adjacency lists
 */

class Graph {
  constructor() {
    this.nodes = [];
    this.nextNodeId = 0;
  }
  
  addNode(x, y) {
    // Find smallest available ID
    const usedIds = new Set(this.nodes.map(n => n.id));
    let newId = 0;
    while (usedIds.has(newId)) {
      newId++;
    }
    
    const node = new GraphNode(newId, x, y);
    this.nodes.push(node);
    this.nextNodeId = Math.max(this.nextNodeId, newId + 1);
    
    return node;
  }
  
  removeNode(node) {
    // Remove this node from all neighbors' adjacency lists
    this.nodes.forEach(n => {
      n.removeNeighbor(node);
    });
    
    // Remove node from graph
    const index = this.nodes.indexOf(node);
    if (index !== -1) {
      this.nodes.splice(index, 1);
    }
  }
  
  addEdge(nodeA, nodeB) {
    // Check if edge already exists
    if (nodeA.hasNeighbor(nodeB)) {
      // Toggle off - remove edge
      this.removeEdge(nodeA, nodeB);
      return null;
    } else {
      // Add edge - update both adjacency lists
      nodeA.addNeighbor(nodeB);
      nodeB.addNeighbor(nodeA);
      return { from: nodeA, to: nodeB };
    }
  }
  
  removeEdge(nodeA, nodeB) {
    nodeA.removeNeighbor(nodeB);
    nodeB.removeNeighbor(nodeA);
  }
  
  hasEdge(nodeA, nodeB) {
    return nodeA.hasNeighbor(nodeB);
  }
  
  findNodeAt(x, y) {
    // Search in reverse order (top nodes first)
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      if (this.nodes[i].contains(x, y)) {
        return this.nodes[i];
      }
    }
    return null;
  }
  
  clear() {
    this.nodes = [];
    this.nextNodeId = 0;
  }
  
  clearPredictions() {
    this.nodes.forEach(node => node.clearPrediction());
  }
  
  // Get all edges as pairs (for rendering)
  getEdges() {
    const edges = [];
    const seen = new Set();
    
    this.nodes.forEach(node => {
      node.neighbors.forEach(neighbor => {
        // Create unique edge key (smaller id first)
        const key = node.id < neighbor.id 
          ? `${node.id}-${neighbor.id}`
          : `${neighbor.id}-${node.id}`;
        
        // Only add if we haven't seen this edge
        if (!seen.has(key)) {
          seen.add(key);
          edges.push({ from: node, to: neighbor });
        }
      });
    });
    
    return edges;
  }
  
  // Check if edge is a self-loop
  isSelfLoop(nodeA, nodeB) {
    return nodeA === nodeB;
  }
  
  // Convert to edge list format for backend
  toEdgeList() {
    // Find vertices that have edges
    const verticesWithEdges = new Set();
    this.nodes.forEach(node => {
      if (node.neighbors.length > 0) {
        verticesWithEdges.add(node);
      }
    });
    
    const lines = [];
    
    // Add isolated vertices (sorted)
    const isolatedVertices = this.nodes
      .filter(node => !verticesWithEdges.has(node))
      .map(node => node.id)
      .sort((a, b) => a - b);
    
    isolatedVertices.forEach(id => {
      lines.push(`${id}`);
    });
    
    // Add edges (sorted)
    const edges = this.getEdges().map(edge => ({
      v1: Math.min(edge.from.id, edge.to.id),
      v2: Math.max(edge.from.id, edge.to.id)
    })).sort((a, b) => {
      if (a.v1 !== b.v1) return a.v1 - b.v1;
      return a.v2 - b.v2;
    });
    
    edges.forEach(edge => {
      lines.push(`${edge.v1} ${edge.v2}`);
    });
    
    return lines.join('\n');
  }
  
  // Load from edge list format
  fromEdgeList(edgeListText, canvasWidth, canvasHeight) {
    this.clear();
    
    if (!edgeListText || edgeListText.trim() === '') {
      return;
    }
    
    const lines = edgeListText.trim().split('\n');
    const nodeIds = new Set();
    const edgeData = [];
    
    // Parse edges and isolated vertices
    lines.forEach(line => {
      const parts = line.trim().split(/\s+/);
      
      if (parts.length === 1) {
        // Isolated vertex
        const vertex = parseInt(parts[0]);
        if (!isNaN(vertex)) {
          nodeIds.add(vertex);
        }
      } else if (parts.length >= 2) {
        // Edge
        const from = parseInt(parts[0]);
        const to = parseInt(parts[1]);
        
        if (!isNaN(from) && !isNaN(to)) {
          nodeIds.add(from);
          nodeIds.add(to);
          edgeData.push({ from, to });
        }
      }
    });
    
    // Create nodes at center with slight jitter (physics will spread them out)
    const sortedIds = Array.from(nodeIds).sort((a, b) => a - b);
    const centerX = canvasWidth / 2;
    const centerY = canvasHeight / 2;
    
    const idToNode = new Map();
    
    sortedIds.forEach((id) => {
      // Place nodes at center with tiny random offset to avoid complete overlap
      // Physics will spread them out nicely; without physics, nodes are at least visible
      const jitterX = (Math.random() - 0.5) * 10;
      const jitterY = (Math.random() - 0.5) * 10;
      const node = new GraphNode(id, centerX + jitterX, centerY + jitterY);
      this.nodes.push(node);
      idToNode.set(id, node);
    });
    
    this.nextNodeId = Math.max(...sortedIds) + 1;
    
    // Create edges using adjacency lists
    edgeData.forEach(({ from, to }) => {
      const fromNode = idToNode.get(from);
      const toNode = idToNode.get(to);
      
      if (fromNode && toNode) {
        fromNode.addNeighbor(toNode);
        toNode.addNeighbor(fromNode);
      }
    });
  }
  
  updatePredictions(predictions) {
    // predictions: array of {nodeId, predicted, actual}
    this.clearPredictions();
    
    predictions.forEach(pred => {
      const node = this.nodes.find(n => n.id === pred.nodeId);
      if (node) {
        node.setPrediction(pred.predicted, pred.actual);
      }
    });
  }
}

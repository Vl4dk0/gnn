/**
 * Interactive Graph Canvas with Spring Physics
 * Allows users to create and manipulate graphs visually
 */

// ============================================================================
// Node Class
// ============================================================================

class GraphNode {
  constructor(id, x, y) {
    this.id = id;
    this.x = x;
    this.y = y;

    // Physics state
    this.vx = 0; // velocity x
    this.vy = 0; // velocity y
    this.fx = 0; // force x
    this.fy = 0; // force y
    this.mass = 1;
    this.fixed = false; // If true, node won't move in physics sim

    // Visual state
    this.radius = 20;
    this.color = "#666666";
    this.selectedColor = "#999999";

    // Prediction state
    this.predicted = null;
    this.actual = null;
  }

  get isWrong() {
    return (
      this.predicted !== null &&
      this.actual !== null &&
      this.predicted !== this.actual
    );
  }

  hasPrediction() {
    return this.predicted !== null && this.actual !== null;
  }

  setPrediction(predicted, actual) {
    this.predicted = predicted;
    this.actual = actual;
  }

  clearPrediction() {
    this.predicted = null;
    this.actual = null;
  }

  applyForce(fx, fy) {
    this.fx += fx;
    this.fy += fy;
  }

  resetForce() {
    this.fx = 0;
    this.fy = 0;
  }

  updatePosition(damping = 0.8) {
    if (this.fixed) return;

    // Apply force to velocity (F = ma, a = F/m)
    this.vx += this.fx / this.mass;
    this.vy += this.fy / this.mass;

    // Apply damping
    this.vx *= damping;
    this.vy *= damping;

    // Update position
    this.x += this.vx;
    this.y += this.vy;
  }

  distanceTo(other) {
    const dx = this.x - other.x;
    const dy = this.y - other.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  contains(x, y) {
    const dx = x - this.x;
    const dy = y - this.y;
    return Math.sqrt(dx * dx + dy * dy) <= this.radius;
  }
}

// ============================================================================
// Edge Class
// ============================================================================

class GraphEdge {
  constructor(fromNode, toNode) {
    this.fromNode = fromNode;
    this.toNode = toNode;
  }

  get isSelfLoop() {
    return this.fromNode === this.toNode;
  }

  contains(nodeA, nodeB) {
    return (
      (this.fromNode === nodeA && this.toNode === nodeB) ||
      (this.fromNode === nodeB && this.toNode === nodeA)
    );
  }
}

// ============================================================================
// Graph Class
// ============================================================================

class Graph {
  constructor() {
    this.nodes = [];
    this.edges = [];
    this.nextNodeId = 0;
  }

  addNode(x, y) {
    // Find smallest available ID
    const usedIds = new Set(this.nodes.map((n) => n.id));
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
    // Remove all edges connected to this node
    this.edges = this.edges.filter(
      (edge) => edge.fromNode !== node && edge.toNode !== node,
    );

    // Remove node
    const index = this.nodes.indexOf(node);
    if (index !== -1) {
      this.nodes.splice(index, 1);
    }
  }

  addEdge(fromNode, toNode) {
    // Check if edge already exists
    const existingEdge = this.findEdge(fromNode, toNode);
    if (existingEdge) {
      // Toggle off - remove edge
      this.removeEdge(existingEdge);
      return null;
    } else {
      // Add new edge
      const edge = new GraphEdge(fromNode, toNode);
      this.edges.push(edge);
      return edge;
    }
  }

  removeEdge(edge) {
    const index = this.edges.indexOf(edge);
    if (index !== -1) {
      this.edges.splice(index, 1);
    }
  }

  findEdge(nodeA, nodeB) {
    return this.edges.find((edge) => edge.contains(nodeA, nodeB));
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
    this.edges = [];
    this.nextNodeId = 0;
  }

  clearPredictions() {
    this.nodes.forEach((node) => node.clearPrediction());
  }

  // Convert to edge list format for backend
  toEdgeList() {
    // Find vertices that have edges
    const verticesWithEdges = new Set();
    this.edges.forEach((edge) => {
      verticesWithEdges.add(edge.fromNode);
      verticesWithEdges.add(edge.toNode);
    });

    const lines = [];

    // Add isolated vertices (sorted)
    const isolatedVertices = this.nodes
      .filter((node) => !verticesWithEdges.has(node))
      .map((node) => node.id)
      .sort((a, b) => a - b);

    isolatedVertices.forEach((id) => {
      lines.push(`${id}`);
    });

    // Add edges (sorted)
    const edgeList = this.edges
      .map((edge) => ({
        v1: Math.min(edge.fromNode.id, edge.toNode.id),
        v2: Math.max(edge.fromNode.id, edge.toNode.id),
      }))
      .sort((a, b) => {
        if (a.v1 !== b.v1) return a.v1 - b.v1;
        return a.v2 - b.v2;
      });

    edgeList.forEach((edge) => {
      lines.push(`${edge.v1} ${edge.v2}`);
    });

    return lines.join("\n");
  }

  // Load from edge list format
  fromEdgeList(edgeListText, canvasWidth, canvasHeight) {
    this.clear();

    if (!edgeListText || edgeListText.trim() === "") {
      return;
    }

    const lines = edgeListText.trim().split("\n");
    const nodeIds = new Set();
    const edgeData = [];

    // Parse edges and isolated vertices
    lines.forEach((line) => {
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

    // Create nodes in circular layout
    const sortedIds = Array.from(nodeIds).sort((a, b) => a - b);
    const centerX = canvasWidth / 2;
    const centerY = canvasHeight / 2;
    const radius = Math.min(centerX, centerY) * 0.6;

    const idToNode = new Map();

    sortedIds.forEach((id, index) => {
      const angle = (index / sortedIds.length) * Math.PI * 2 - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      const node = new GraphNode(id, x, y);
      this.nodes.push(node);
      idToNode.set(id, node);
    });

    this.nextNodeId = Math.max(...sortedIds) + 1;

    // Create edges
    edgeData.forEach(({ from, to }) => {
      const fromNode = idToNode.get(from);
      const toNode = idToNode.get(to);

      if (fromNode && toNode) {
        this.edges.push(new GraphEdge(fromNode, toNode));
      }
    });
  }

  updatePredictions(predictions) {
    // predictions: array of {nodeId, predicted, actual}
    this.clearPredictions();

    predictions.forEach((pred) => {
      const node = this.nodes.find((n) => n.id === pred.nodeId);
      if (node) {
        node.setPrediction(pred.predicted, pred.actual);
      }
    });
  }
}

// ============================================================================
// Physics Engine
// ============================================================================

class PhysicsEngine {
  constructor(graph) {
    this.graph = graph;

    // Physics parameters
    this.springLength = 100; // Desired edge length
    this.springStrength = 0.05; // Spring force multiplier
    this.repulsionStrength = 3000; // Node repulsion force
    this.damping = 0.85; // Velocity damping (0-1)
    this.centeringStrength = 0.001; // Pull towards center
    this.enabled = false;
  }

  step(canvasWidth, canvasHeight) {
    if (!this.enabled || this.graph.nodes.length === 0) return;

    const centerX = canvasWidth / 2;
    const centerY = canvasHeight / 2;

    // Reset forces
    this.graph.nodes.forEach((node) => node.resetForce());

    // Apply spring forces (attraction along edges)
    this.graph.edges.forEach((edge) => {
      if (edge.isSelfLoop) return;

      const dx = edge.toNode.x - edge.fromNode.x;
      const dy = edge.toNode.y - edge.fromNode.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance < 0.1) return; // Avoid division by zero

      // Hooke's law: F = k * (distance - restLength)
      const force = this.springStrength * (distance - this.springLength);
      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;

      edge.fromNode.applyForce(fx, fy);
      edge.toNode.applyForce(-fx, -fy);
    });

    // Apply repulsion forces (all pairs of nodes)
    for (let i = 0; i < this.graph.nodes.length; i++) {
      for (let j = i + 1; j < this.graph.nodes.length; j++) {
        const nodeA = this.graph.nodes[i];
        const nodeB = this.graph.nodes[j];

        const dx = nodeB.x - nodeA.x;
        const dy = nodeB.y - nodeA.y;
        const distanceSq = dx * dx + dy * dy;

        if (distanceSq < 0.1) continue; // Avoid division by zero

        // Coulomb's law: F = k / r^2
        const force = this.repulsionStrength / distanceSq;
        const distance = Math.sqrt(distanceSq);
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;

        nodeA.applyForce(-fx, -fy);
        nodeB.applyForce(fx, fy);
      }
    }

    // Apply centering force (weak pull towards center)
    this.graph.nodes.forEach((node) => {
      const dx = centerX - node.x;
      const dy = centerY - node.y;
      node.applyForce(dx * this.centeringStrength, dy * this.centeringStrength);
    });

    // Update positions
    this.graph.nodes.forEach((node) => {
      node.updatePosition(this.damping);
    });
  }
}

// ============================================================================
// Interactive Graph Canvas
// ============================================================================

class InteractiveGraph {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");

    // Graph and physics
    this.graph = new Graph();
    this.physics = new PhysicsEngine(this.graph);

    // Interaction state
    this.selectedNode = null;
    this.draggingNode = null;
    this.edgeStart = null;
    this.panningCanvas = false;
    this.panStartX = 0;
    this.panStartY = 0;

    // Canvas offset for panning
    this.offsetX = 0;
    this.offsetY = 0;

    // Visual settings
    this.backgroundColor = "#1a1a1a";
    this.edgeColor = "#555555";
    this.textColor = "#ffffff";
    this.errorDotColor = "#ff0000";

    // Mouse state
    this.mouseX = 0;
    this.mouseY = 0;

    // Animation
    this.animationFrameId = null;
    this.lastFrameTime = 0;

    this.initCanvas();
    this.setupEventListeners();
    this.startAnimationLoop();
  }

  initCanvas() {
    this.resizeCanvas();
    window.addEventListener("resize", () => this.resizeCanvas());
  }

  resizeCanvas() {
    const container = this.canvas.parentElement;
    this.canvas.width = container.clientWidth;
    this.canvas.height = container.clientHeight;
  }

  setupEventListeners() {
    this.canvas.addEventListener("mousemove", (e) => this.handleMouseMove(e));

    this.canvas.addEventListener("mousedown", (e) => {
      if (e.button === 0) {
        this.handleLeftClick(e);
      } else if (e.button === 1) {
        e.preventDefault();
        this.handleMiddleClick(e);
      }
    });

    this.canvas.addEventListener("mouseup", (e) => {
      if (e.button === 0) {
        this.handleLeftRelease(e);
      } else if (e.button === 2) {
        this.handleRightRelease(e);
      }
    });

    this.canvas.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      this.handleRightClick(e);
    });

    this.canvas.addEventListener("dblclick", (e) => this.handleDoubleClick(e));

    document.addEventListener("keydown", (e) => {
      if (e.key === "Backspace" && this.selectedNode !== null) {
        e.preventDefault();
        this.deleteNode(this.selectedNode);
      }
    });
  }

  getMousePos(e) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left - this.offsetX,
      y: e.clientY - rect.top - this.offsetY,
    };
  }

  handleMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const rawX = e.clientX - rect.left;
    const rawY = e.clientY - rect.top;

    // Handle canvas panning
    if (this.panningCanvas) {
      const dx = rawX - this.panStartX;
      const dy = rawY - this.panStartY;

      this.offsetX += dx;
      this.offsetY += dy;

      this.panStartX = rawX;
      this.panStartY = rawY;
      return;
    }

    const pos = this.getMousePos(e);
    this.mouseX = pos.x;
    this.mouseY = pos.y;

    // Drag node
    if (this.draggingNode !== null) {
      this.draggingNode.x = pos.x;
      this.draggingNode.y = pos.y;
      this.draggingNode.vx = 0;
      this.draggingNode.vy = 0;
    }
  }

  handleLeftClick(e) {
    const pos = this.getMousePos(e);
    const node = this.graph.findNodeAt(pos.x, pos.y);

    if (node) {
      // If drawing an edge, complete it
      if (this.edgeStart !== null) {
        this.addEdge(this.edgeStart, node);
        this.edgeStart = null;
        return;
      }

      // Otherwise, select and start dragging
      this.selectedNode = node;
      this.draggingNode = node;
      node.fixed = true; // Fix node during drag
    } else {
      // Cancel edge drawing
      if (this.edgeStart !== null) {
        this.edgeStart = null;
      }

      // Deselect
      this.selectedNode = null;
    }
  }

  handleLeftRelease(e) {
    if (this.draggingNode) {
      this.draggingNode.fixed = false;
      this.draggingNode = null;
    }
  }

  handleRightClick(e) {
    const pos = this.getMousePos(e);
    const node = this.graph.findNodeAt(pos.x, pos.y);

    if (node) {
      if (this.edgeStart === null) {
        // Start drawing edge
        this.edgeStart = node;
      } else {
        // Complete edge
        this.addEdge(this.edgeStart, node);
        this.edgeStart = null;
      }
    } else {
      // Start panning
      const rect = this.canvas.getBoundingClientRect();
      this.panningCanvas = true;
      this.panStartX = e.clientX - rect.left;
      this.panStartY = e.clientY - rect.top;
      this.edgeStart = null;
    }
  }

  handleRightRelease(e) {
    this.panningCanvas = false;
  }

  handleMiddleClick(e) {
    const pos = this.getMousePos(e);
    const node = this.graph.findNodeAt(pos.x, pos.y);

    if (node) {
      this.deleteNode(node);
    }
  }

  handleDoubleClick(e) {
    const pos = this.getMousePos(e);
    const node = this.graph.findNodeAt(pos.x, pos.y);

    if (!node) {
      // Add node on double-click empty space
      this.addNode(pos.x, pos.y);
    }
  }

  addNode(x, y) {
    this.graph.clearPredictions();
    this.graph.addNode(x, y);
    this.updateGraphInput();
    this.triggerAnalyze();
  }

  deleteNode(node) {
    this.graph.clearPredictions();

    if (this.selectedNode === node) {
      this.selectedNode = null;
    }

    this.graph.removeNode(node);
    this.updateGraphInput();
    this.triggerAnalyze();
  }

  addEdge(fromNode, toNode) {
    this.graph.clearPredictions();
    this.graph.addEdge(fromNode, toNode);
    this.updateGraphInput();
    this.triggerAnalyze();
  }

  updateGraphInput() {
    const graphInput = document.getElementById("graphInput");
    if (!graphInput) return;

    graphInput.value = this.graph.toEdgeList();
  }

  startAnimationLoop() {
    const animate = (timestamp) => {
      const deltaTime = timestamp - this.lastFrameTime;
      this.lastFrameTime = timestamp;

      // Run physics simulation
      this.physics.step(this.canvas.width, this.canvas.height);

      // Render
      this.render();

      // Continue loop
      this.animationFrameId = requestAnimationFrame(animate);
    };

    this.animationFrameId = requestAnimationFrame(animate);
  }

  stopAnimationLoop() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  render() {
    // Clear canvas
    this.ctx.fillStyle = this.backgroundColor;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Save context and apply offset for panning
    this.ctx.save();
    this.ctx.translate(this.offsetX, this.offsetY);

    // Draw edges
    this.ctx.strokeStyle = this.edgeColor;
    this.ctx.lineWidth = 2;

    this.graph.edges.forEach((edge) => {
      if (edge.isSelfLoop) {
        this.drawSelfLoop(edge.fromNode);
      } else {
        this.ctx.beginPath();
        this.ctx.moveTo(edge.fromNode.x, edge.fromNode.y);
        this.ctx.lineTo(edge.toNode.x, edge.toNode.y);
        this.ctx.stroke();
      }
    });

    // Draw edge preview
    if (this.edgeStart !== null) {
      this.ctx.strokeStyle = "#888888";
      this.ctx.setLineDash([5, 5]);
      this.ctx.beginPath();
      this.ctx.moveTo(this.edgeStart.x, this.edgeStart.y);
      this.ctx.lineTo(this.mouseX, this.mouseY);
      this.ctx.stroke();
      this.ctx.setLineDash([]);
    }

    // Draw nodes
    this.graph.nodes.forEach((node) => {
      const isSelected = node === this.selectedNode;
      const fillColor = isSelected ? node.selectedColor : node.color;

      // Draw node circle
      this.ctx.fillStyle = fillColor;
      this.ctx.beginPath();
      this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
      this.ctx.fill();

      // Draw node ID
      this.ctx.fillStyle = this.textColor;
      this.ctx.font = "bold 14px Arial";
      this.ctx.textAlign = "center";
      this.ctx.textBaseline = "middle";
      this.ctx.fillText(node.id.toString(), node.x, node.y);

      // Draw prediction info
      if (node.hasPrediction()) {
        const offsetX = node.x + node.radius * 0.8;
        const offsetY = node.y + node.radius * 0.8;

        this.ctx.font = "bold 14px Arial";
        this.ctx.textAlign = "left";
        this.ctx.textBaseline = "middle";

        const isCorrect = node.predicted === node.actual;
        const correctColor = isCorrect ? "#888888" : "#00ff00";
        const predictedColor = isCorrect ? "#888888" : this.errorDotColor;

        // Draw actual value
        this.ctx.fillStyle = correctColor;
        const correctText = node.actual.toString();
        this.ctx.fillText(correctText, offsetX, offsetY);

        // Draw slash
        const correctWidth = this.ctx.measureText(correctText).width;
        this.ctx.fillStyle = isCorrect ? "#888888" : this.textColor;
        const slashX = offsetX + correctWidth;
        this.ctx.fillText("/", slashX, offsetY);
        const slashWidth = this.ctx.measureText("/").width;

        // Draw predicted value
        this.ctx.fillStyle = predictedColor;
        this.ctx.fillText(
          node.predicted.toString(),
          slashX + slashWidth,
          offsetY,
        );
      }
    });

    // Restore context
    this.ctx.restore();
  }

  drawSelfLoop(node) {
    const loopRadius = node.radius * 0.8;
    const loopCenterY = node.y - node.radius - loopRadius;

    this.ctx.beginPath();
    this.ctx.arc(node.x, loopCenterY, loopRadius, 0, Math.PI * 2);
    this.ctx.stroke();
  }

  // Public API

  loadFromEdgeList(edgeListText) {
    this.graph.fromEdgeList(
      edgeListText,
      this.canvas.width,
      this.canvas.height,
    );
    this.selectedNode = null;
  }

  clear() {
    this.graph.clear();
    this.selectedNode = null;
    this.edgeStart = null;
    this.updateGraphInput();
  }

  updatePredictions(predictions) {
    this.graph.updatePredictions(predictions);
  }

  triggerAnalyze() {
    // Trigger analyze only if predictions haven't been calculated yet
    if (
      this.graph.nodes.length > 0 &&
      !this.graph.nodes.some((n) => n.hasPrediction())
    ) {
      setTimeout(() => {
        if (window.analyzeGraph) {
          window.analyzeGraph();
        }
      }, 100);
    }
  }

  // Physics control
  enablePhysics() {
    this.physics.enabled = true;
  }

  disablePhysics() {
    this.physics.enabled = false;
    // Reset velocities
    this.graph.nodes.forEach((node) => {
      node.vx = 0;
      node.vy = 0;
    });
  }

  togglePhysics() {
    if (this.physics.enabled) {
      this.disablePhysics();
    } else {
      this.enablePhysics();
    }
  }

  isPhysicsEnabled() {
    return this.physics.enabled;
  }
}

// ============================================================================
// Global Instance
// ============================================================================

window.interactiveGraph = null;

document.addEventListener("DOMContentLoaded", () => {
  window.interactiveGraph = new InteractiveGraph("graphCanvas");
});

/**
 * InteractiveGraphEditor - Canvas-based graph editor with physics
 */

class InteractiveGraphEditor {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    
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
    
    // Zoom settings
    this.scale = 1.0;
    this.minScale = 0.1;
    this.maxScale = 5.0;
    this.scaleStep = 0.033; // Very smooth zoom (0.33x slower than original)
    
    // Visual settings
    this.backgroundColor = '#1a1a1a';
    this.edgeColor = '#555555';
    this.textColor = '#ffffff';
    this.errorDotColor = '#ff0000';
    
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
    window.addEventListener('resize', () => this.resizeCanvas());
  }
  
  resizeCanvas() {
    const container = this.canvas.parentElement;
    this.canvas.width = container.clientWidth;
    this.canvas.height = container.clientHeight;
  }
  
  setupEventListeners() {
    this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    
    this.canvas.addEventListener('mousedown', (e) => {
      if (e.button === 0) {
        this.handleLeftClick(e);
      } else if (e.button === 1) {
        e.preventDefault();
        this.handleMiddleClick(e);
      }
    });
    
    this.canvas.addEventListener('mouseup', (e) => {
      if (e.button === 0) {
        this.handleLeftRelease(e);
      }
    });
    
    this.canvas.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      this.handleRightClick(e);
    });
    
    this.canvas.addEventListener('dblclick', (e) => this.handleDoubleClick(e));
    
    this.canvas.addEventListener('wheel', (e) => this.handleWheel(e));
    
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Backspace' && this.selectedNode !== null) {
        e.preventDefault();
        this.deleteNode(this.selectedNode);
      }
    });
  }
  
  getMousePos(e) {
    const rect = this.canvas.getBoundingClientRect();
    // Convert screen coordinates to world coordinates (accounting for pan and zoom)
    return {
      x: (e.clientX - rect.left - this.offsetX) / this.scale,
      y: (e.clientY - rect.top - this.offsetY) / this.scale
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
      // If drawing an edge, create a new node and complete the edge
      if (this.edgeStart !== null) {
        const newNode = this.graph.addNode(pos.x, pos.y);
        this.addEdge(this.edgeStart, newNode);
        this.edgeStart = null;
        this.updateGraphInput();
        this.triggerAnalyze();
        return;
      }
      
      // Start panning canvas on empty space
      const rect = this.canvas.getBoundingClientRect();
      this.panningCanvas = true;
      this.panStartX = e.clientX - rect.left;
      this.panStartY = e.clientY - rect.top;
      
      // Deselect
      this.selectedNode = null;
    }
  }
  
  handleLeftRelease(e) {
    // Stop panning
    this.panningCanvas = false;
    
    // Stop dragging node
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
      // Cancel edge drawing on empty space
      this.edgeStart = null;
    }
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
  
  handleWheel(e) {
    e.preventDefault();
    
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    // Calculate zoom direction
    const delta = -Math.sign(e.deltaY);
    const zoomFactor = 1 + (delta * this.scaleStep);
    const newScale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * zoomFactor));
    
    if (newScale === this.scale) return; // No change if at limit
    
    // Calculate the point in world coordinates before zoom
    const worldX = (mouseX - this.offsetX) / this.scale;
    const worldY = (mouseY - this.offsetY) / this.scale;
    
    // Update scale
    this.scale = newScale;
    
    // Adjust offset to keep the world point under the mouse
    this.offsetX = mouseX - worldX * this.scale;
    this.offsetY = mouseY - worldY * this.scale;
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
    const graphInput = document.getElementById('graphInput');
    if (!graphInput) return;
    
    graphInput.value = this.graph.toEdgeList();
  }
  
  startAnimationLoop() {
    const animate = (timestamp) => {
      // Calculate delta time
      if (this.lastFrameTime === 0) {
        this.lastFrameTime = timestamp;
      }
      const deltaTime = timestamp - this.lastFrameTime;
      this.lastFrameTime = timestamp;
      
      // Run physics simulation with deltaTime
      this.physics.step(this.canvas.width, this.canvas.height, deltaTime);
      
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
    
    // Save context and apply transformations (pan and zoom)
    this.ctx.save();
    this.ctx.translate(this.offsetX, this.offsetY);
    this.ctx.scale(this.scale, this.scale);
    
    // Draw edges (iterate through adjacency lists)
    this.ctx.strokeStyle = this.edgeColor;
    this.ctx.lineWidth = 2;
    
    const edges = this.graph.getEdges();
    edges.forEach(edge => {
      if (edge.from === edge.to) {
        // Self-loop
        this.drawSelfLoop(edge.from);
      } else {
        // Regular edge
        this.ctx.beginPath();
        this.ctx.moveTo(edge.from.x, edge.from.y);
        this.ctx.lineTo(edge.to.x, edge.to.y);
        this.ctx.stroke();
      }
    });
    
    // Draw edge preview
    if (this.edgeStart !== null) {
      this.ctx.strokeStyle = '#888888';
      this.ctx.setLineDash([5, 5]);
      this.ctx.beginPath();
      this.ctx.moveTo(this.edgeStart.x, this.edgeStart.y);
      this.ctx.lineTo(this.mouseX, this.mouseY);
      this.ctx.stroke();
      this.ctx.setLineDash([]);
    }
    
    // Draw nodes
    this.graph.nodes.forEach(node => {
      const isSelected = node === this.selectedNode;
      const fillColor = isSelected ? node.selectedColor : node.color;
      
      // Draw node circle
      this.ctx.fillStyle = fillColor;
      this.ctx.beginPath();
      this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
      this.ctx.fill();
      
      // Draw node ID
      this.ctx.fillStyle = this.textColor;
      this.ctx.font = 'bold 14px Arial';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(node.id.toString(), node.x, node.y);
      
      // Draw prediction info
      if (node.hasPrediction()) {
        const offsetX = node.x + node.radius * 0.8;
        const offsetY = node.y + node.radius * 0.8;
        
        this.ctx.font = 'bold 14px Arial';
        this.ctx.textAlign = 'left';
        this.ctx.textBaseline = 'middle';
        
        const isCorrect = node.predicted === node.actual;
        const correctColor = isCorrect ? '#888888' : '#00ff00';
        const predictedColor = isCorrect ? '#888888' : this.errorDotColor;
        
        // Draw actual value
        this.ctx.fillStyle = correctColor;
        const correctText = node.actual.toString();
        this.ctx.fillText(correctText, offsetX, offsetY);
        
        // Draw slash
        const correctWidth = this.ctx.measureText(correctText).width;
        this.ctx.fillStyle = isCorrect ? '#888888' : this.textColor;
        const slashX = offsetX + correctWidth;
        this.ctx.fillText('/', slashX, offsetY);
        const slashWidth = this.ctx.measureText('/').width;
        
        // Draw predicted value
        this.ctx.fillStyle = predictedColor;
        this.ctx.fillText(node.predicted.toString(), slashX + slashWidth, offsetY);
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
    this.graph.fromEdgeList(edgeListText, this.canvas.width, this.canvas.height);
    this.selectedNode = null;
    
    // Always pre-settle for initial layout, regardless of physics setting
    if (this.graph.nodes.length > 0) {
      this.preSettlePhysics(200); // Run 200 iterations for better settling
    }
  }
  
  preSettlePhysics(iterations) {
    // Run physics iterations without rendering
    const wasEnabled = this.physics.enabled;
    this.physics.enabled = true;
    
    for (let i = 0; i < iterations; i++) {
      // Use a fixed deltaTime for consistent settling (16.67ms = 60fps)
      this.physics.step(this.canvas.width, this.canvas.height, 16.67);
    }
    
    this.physics.enabled = wasEnabled;
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
    if (this.graph.nodes.length > 0 && !this.graph.nodes.some(n => n.hasPrediction())) {
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
    this.graph.nodes.forEach(node => {
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
  
  // Legacy compatibility properties
  get nodes() {
    return this.graph.nodes;
  }
  
  get nodePredictions() {
    // Create a Map for compatibility with old code
    const map = new Map();
    this.graph.nodes.forEach(node => {
      if (node.hasPrediction()) {
        map.set(node.id, {
          predicted: node.predicted,
          actual: node.actual,
          isWrong: node.isWrong
        });
      }
    });
    return map;
  }
}

// ============================================================================
// Global Instance
// ============================================================================

window.interactiveGraph = null;

document.addEventListener('DOMContentLoaded', () => {
  window.interactiveGraph = new InteractiveGraphEditor('graphCanvas');
});

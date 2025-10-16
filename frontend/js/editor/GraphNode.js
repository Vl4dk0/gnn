/**
 * GraphNode - Represents a node in the graph with adjacency list
 */

class GraphNode {
  constructor(id, x, y) {
    this.id = id;
    this.x = x;
    this.y = y;
    
    // Adjacency list - classic graph representation
    this.neighbors = [];  // Array of GraphNode references
    
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
  
  get degree() {
    return this.neighbors.length;
  }
  
  get isWrong() {
    return this.predicted !== null && this.actual !== null && this.predicted !== this.actual;
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
  
  addNeighbor(node) {
    if (!this.neighbors.includes(node)) {
      this.neighbors.push(node);
    }
  }
  
  removeNeighbor(node) {
    const index = this.neighbors.indexOf(node);
    if (index !== -1) {
      this.neighbors.splice(index, 1);
    }
  }
  
  hasNeighbor(node) {
    return this.neighbors.includes(node);
  }
  
  // Physics methods
  
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
  
  // Geometry methods
  
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

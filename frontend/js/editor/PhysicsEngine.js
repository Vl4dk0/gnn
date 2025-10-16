/**
 * PhysicsEngine - Force-directed graph layout simulation
 */

class PhysicsEngine {
  constructor(graph) {
    this.graph = graph;
    
    // Physics parameters (matching old working implementation)
    this.idealEdgeLength = 300;        // Desired edge length
    this.idealNodeDistance = 300;      // Desired distance between unconnected nodes
    this.springConstant = 0.1;         // Spring force multiplier
    this.enabled = true;               // Enable by default
  }
  
  step(canvasWidth, canvasHeight, deltaTime) {
    if (!this.enabled || this.graph.nodes.length === 0) return;
    
    // Calculate granularity based on deltaTime (convert ms to seconds and scale)
    const granularity = (deltaTime / 1000) * 10;
    
    // Build edge map for quick neighbor lookup
    const edgeMap = new Map();
    for (const node of this.graph.nodes) {
      edgeMap.set(node.id, node.neighbors);
    }
    
    // Apply forces to each node
    this.graph.nodes.forEach(node => {
      const connected = edgeMap.get(node.id) || [];
      
      let xForce = 0;
      let yForce = 0;
      
      // Attractive forces from connected nodes (spring forces)
      connected.forEach(other => {
        if (other.id === node.id) return; // Skip self-loops
        
        const dx = other.x - node.x;
        const dy = other.y - node.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < 0.1) return; // Avoid division by zero
        
        const relative = distance - this.idealEdgeLength;
        const force = relative * this.springConstant;
        xForce += (dx / distance) * force;
        yForce += (dy / distance) * force;
      });
      
      // Repulsive forces from all other nodes
      this.graph.nodes.forEach(other => {
        if (other.id === node.id) return; // Skip self
        if (connected.includes(other)) return; // Skip connected nodes
        
        const dx = other.x - node.x;
        const dy = other.y - node.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Ignore distant nodes
        if (distance > this.idealNodeDistance * 3) return;
        
        const relative = distance - this.idealNodeDistance;
        const force = relative * this.springConstant;
        xForce += (dx / distance) * force;
        yForce += (dy / distance) * force;
      });
      
      // Apply forces with clamping (max 5 pixels per frame)
      node.x += Math.max(-5, Math.min(5, xForce * granularity));
      node.y += Math.max(-5, Math.min(5, yForce * granularity));
    });
  }
}

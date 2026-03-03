import { Graph } from "./Graph";

export class PhysicsEngine {
  private readonly graph: Graph;
  private idealEdgeLength = 300;
  private idealNodeDistance = 300;
  private springConstant = 0.1;
  public enabled = true;

  public constructor(graph: Graph) {
    this.graph = graph;
  }

  public step(_canvasWidth: number, _canvasHeight: number, deltaTime: number): void {
    if (!this.enabled || this.graph.nodes.length === 0) {
      return;
    }

    const granularity = (deltaTime / 1000) * 10;

    const edgeMap = new Map<number, (typeof this.graph.nodes)[number]["neighbors"]>();
    for (const node of this.graph.nodes) {
      edgeMap.set(node.id, node.neighbors);
    }

    this.graph.nodes.forEach((node) => {
      const connected = edgeMap.get(node.id) ?? [];
      let xForce = 0;
      let yForce = 0;

      connected.forEach((other) => {
        if (other.id === node.id) {
          return;
        }

        const dx = other.x - node.x;
        const dy = other.y - node.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < 0.1) {
          return;
        }

        const relative = distance - this.idealEdgeLength;
        const force = relative * this.springConstant;
        xForce += (dx / distance) * force;
        yForce += (dy / distance) * force;
      });

      this.graph.nodes.forEach((other) => {
        if (other.id === node.id) {
          return;
        }

        if (connected.includes(other)) {
          return;
        }

        const dx = other.x - node.x;
        const dy = other.y - node.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance > this.idealNodeDistance * 3 || distance < 0.1) {
          return;
        }

        const relative = distance - this.idealNodeDistance;
        const force = relative * this.springConstant;
        xForce += (dx / distance) * force;
        yForce += (dy / distance) * force;
      });

      node.x += Math.max(-5, Math.min(5, xForce * granularity));
      node.y += Math.max(-5, Math.min(5, yForce * granularity));
    });
  }
}

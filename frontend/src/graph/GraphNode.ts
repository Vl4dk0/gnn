export class GraphNode {
  public readonly id: number;
  public x: number;
  public y: number;
  public readonly neighbors: GraphNode[] = [];

  public vx = 0;
  public vy = 0;
  public fx = 0;
  public fy = 0;
  public mass = 1;
  public fixed = false;

  public radius = 20;
  public color = "#666666";
  public selectedColor = "#999999";

  public predicted: number | null = null;
  public actual: number | null = null;

  public constructor(id: number, x: number, y: number) {
    this.id = id;
    this.x = x;
    this.y = y;
  }

  public get degree(): number {
    return this.neighbors.length;
  }

  public get isWrong(): boolean {
    return this.predicted !== null && this.actual !== null && this.predicted !== this.actual;
  }

  public hasPrediction(): boolean {
    return this.predicted !== null && this.actual !== null;
  }

  public setPrediction(predicted: number, actual: number): void {
    this.predicted = predicted;
    this.actual = actual;
  }

  public clearPrediction(): void {
    this.predicted = null;
    this.actual = null;
  }

  public addNeighbor(node: GraphNode): void {
    if (!this.neighbors.includes(node)) {
      this.neighbors.push(node);
    }
  }

  public removeNeighbor(node: GraphNode): void {
    const index = this.neighbors.indexOf(node);
    if (index !== -1) {
      this.neighbors.splice(index, 1);
    }
  }

  public hasNeighbor(node: GraphNode): boolean {
    return this.neighbors.includes(node);
  }

  public distanceTo(other: GraphNode): number {
    const dx = this.x - other.x;
    const dy = this.y - other.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  public contains(x: number, y: number): boolean {
    const dx = x - this.x;
    const dy = y - this.y;
    return Math.sqrt(dx * dx + dy * dy) <= this.radius;
  }
}

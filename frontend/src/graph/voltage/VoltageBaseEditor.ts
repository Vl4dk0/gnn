import { getCanvasTheme } from "../../hooks/useTheme";
import { Graph } from "../Graph";
import { GraphNode } from "../GraphNode";
import { PhysicsEngine } from "../PhysicsEngine";
import type { BaseArc } from "./liftConstruction";

interface VoltageBaseCallbacks {
  onChange?: () => void;
}

interface Point {
  x: number;
  y: number;
}

/**
 * Interactive canvas editor for the BASE multigraph of a voltage lift.
 *
 * Unlike InteractiveGraphEditor, this supports parallel directed arcs and
 * self-loops, each carrying a voltage label (an element of Z_n). It reuses
 * GraphNode for node state and PhysicsEngine for spring layout (driven off a
 * deduped adjacency mirror), while keeping its own `arcs` array for the
 * directed/parallel/voltage semantics the lift construction needs.
 *
 * Mouse interactions:
 *  - double-click empty: add node
 *  - left-drag node: move (node fixed while dragging)
 *  - right-click node then click another (or same) node: add arc first -> second
 *  - left-click near an arc midpoint: select that arc
 *  - wheel: zoom; left-drag empty: pan
 * While an arc is selected:
 *  - "+"/"-" or ArrowUp/ArrowDown: change voltage mod n
 *  - "r": reverse orientation (swap from/to and voltage -> (n - v) % n; lift unchanged)
 *  - Backspace/Delete: remove selected arc (or selected node + its arcs)
 */
export class VoltageBaseEditor {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly callbacks: VoltageBaseCallbacks;

  // Layout graph mirrors arcs as a simple deduped adjacency for the physics.
  private readonly graph = new Graph();
  private readonly physics = new PhysicsEngine(this.graph);

  private arcs: BaseArc[] = [];
  private groupOrder = 7;
  private nextArcId = 0;

  private selectedNode: GraphNode | null = null;
  private selectedArcId: number | null = null;
  private draggingNode: GraphNode | null = null;
  private arcStart: GraphNode | null = null;

  private panningCanvas = false;
  private panStartX = 0;
  private panStartY = 0;

  private offsetX = 0;
  private offsetY = 0;
  private scale = 1.0;
  private readonly minScale = 0.1;
  private readonly maxScale = 5.0;
  private readonly scaleStep = 0.033;

  private mouseX = 0;
  private mouseY = 0;

  private resizeObserver: ResizeObserver | null = null;
  private animationFrameId: number | null = null;
  private lastFrameTime = 0;
  private disposed = false;
  private readonly unbinders: Array<() => void> = [];

  private get themeColors() {
    return getCanvasTheme();
  }

  public constructor(canvas: HTMLCanvasElement, callbacks: VoltageBaseCallbacks = {}) {
    this.canvas = canvas;
    this.callbacks = callbacks;

    const context = this.canvas.getContext("2d");
    if (!context) {
      throw new Error("2D canvas context is not available");
    }
    this.ctx = context;

    this.initCanvas();
    this.setupEventListeners();
    this.startAnimationLoop();
  }

  // ---- setup / teardown -------------------------------------------------

  private bindEvent<T extends EventTarget>(
    target: T,
    type: string,
    listener: EventListenerOrEventListenerObject,
    options?: AddEventListenerOptions
  ): void {
    target.addEventListener(type, listener, options);
    this.unbinders.push(() => {
      target.removeEventListener(type, listener, options);
    });
  }

  private initCanvas(): void {
    this.resizeCanvas();
    this.bindEvent(window, "resize", () => this.resizeCanvas());

    const container = this.canvas.parentElement;
    if (container && typeof ResizeObserver === "function") {
      this.resizeObserver = new ResizeObserver(() => this.resizeCanvas());
      this.resizeObserver.observe(container);
    }
  }

  private resizeCanvas(): void {
    const container = this.canvas.parentElement;
    if (!container) return;

    const width = Math.max(1, Math.round(container.clientWidth));
    const height = Math.max(1, Math.round(container.clientHeight));

    if (this.canvas.width === width && this.canvas.height === height) {
      return;
    }
    this.canvas.width = width;
    this.canvas.height = height;
  }

  private setupEventListeners(): void {
    this.bindEvent(this.canvas, "mousemove", (event) => this.handleMouseMove(event as MouseEvent), {
      passive: false
    });

    this.bindEvent(
      this.canvas,
      "mousedown",
      (event) => {
        const mouseEvent = event as MouseEvent;
        if (mouseEvent.button === 0) {
          this.handleLeftClick(mouseEvent);
        }
      },
      { passive: false }
    );

    this.bindEvent(
      this.canvas,
      "mouseup",
      (event) => {
        const mouseEvent = event as MouseEvent;
        if (mouseEvent.button === 0) {
          this.handleLeftRelease();
        }
      },
      { passive: false }
    );

    this.bindEvent(
      this.canvas,
      "contextmenu",
      (event) => {
        const mouseEvent = event as MouseEvent;
        mouseEvent.preventDefault();
        this.handleRightClick(mouseEvent);
      },
      { passive: false }
    );

    this.bindEvent(this.canvas, "dblclick", (event) => this.handleDoubleClick(event as MouseEvent), {
      passive: false
    });

    this.bindEvent(this.canvas, "wheel", (event) => this.handleWheel(event as WheelEvent), {
      passive: false
    });

    this.bindEvent(document, "keydown", (event) => this.handleKeyDown(event as KeyboardEvent));
  }

  // ---- coordinate helpers ----------------------------------------------

  private getWorldPosFromClient(clientX: number, clientY: number): Point {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = rect.width > 0 ? this.canvas.width / rect.width : 1;
    const scaleY = rect.height > 0 ? this.canvas.height / rect.height : 1;
    const canvasX = (clientX - rect.left) * scaleX;
    const canvasY = (clientY - rect.top) * scaleY;

    return {
      x: (canvasX - this.offsetX) / this.scale,
      y: (canvasY - this.offsetY) / this.scale
    };
  }

  private getCanvasPosFromClient(clientX: number, clientY: number): Point {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = rect.width > 0 ? this.canvas.width / rect.width : 1;
    const scaleY = rect.height > 0 ? this.canvas.height / rect.height : 1;

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    };
  }

  private getMousePos(event: MouseEvent): Point {
    return this.getWorldPosFromClient(event.clientX, event.clientY);
  }

  private nodeById(id: number): GraphNode | null {
    return this.graph.nodes.find((node) => node.id === id) ?? null;
  }

  // ---- arc geometry -----------------------------------------------------

  /**
   * Index of this arc among the parallel arcs sharing the same unordered node
   * pair, used to fan parallel arcs apart visually. Self-loops index among
   * other self-loops on the same node.
   */
  private parallelInfo(arc: BaseArc): { index: number; count: number } {
    const sameGroup = this.arcs.filter((other) => {
      if (arc.from === arc.to) {
        return other.from === other.to && other.from === arc.from;
      }
      return (
        (other.from === arc.from && other.to === arc.to) ||
        (other.from === arc.to && other.to === arc.from)
      );
    });
    const index = sameGroup.findIndex((other) => other.id === arc.id);
    return { index: index < 0 ? 0 : index, count: sameGroup.length };
  }

  /** Perpendicular offset (in world units) used to bend parallel arcs apart. */
  private arcOffset(arc: BaseArc): number {
    const { index, count } = this.parallelInfo(arc);
    if (count <= 1) {
      return 0;
    }
    const spread = 36;
    // Symmetric fan: -..0..+ across the parallel arcs.
    return (index - (count - 1) / 2) * spread;
  }

  /** Midpoint of an arc accounting for its bend, in world coordinates. */
  private arcMidpoint(arc: BaseArc): Point {
    const from = this.nodeById(arc.from);
    const to = this.nodeById(arc.to);
    if (!from || !to) {
      return { x: 0, y: 0 };
    }

    if (arc.from === arc.to) {
      const loop = this.selfLoopGeometry(from, arc);
      return { x: loop.cx, y: loop.cy - loop.radius };
    }

    const mx = (from.x + to.x) / 2;
    const my = (from.y + to.y) / 2;
    const offset = this.arcOffset(arc);
    if (offset === 0) {
      return { x: mx, y: my };
    }

    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len;
    const ny = dx / len;
    // Quadratic bezier with control point at 2*offset; curve midpoint is at
    // half that perpendicular displacement.
    return { x: mx + nx * offset, y: my + ny * offset };
  }

  private selfLoopGeometry(node: GraphNode, arc: BaseArc): { cx: number; cy: number; radius: number } {
    const { index } = this.parallelInfo(arc);
    const radius = node.radius * (0.85 + index * 0.5);
    return { cx: node.x, cy: node.y - node.radius - radius, radius };
  }

  private findArcNear(worldX: number, worldY: number): BaseArc | null {
    const threshold = 16;
    let best: BaseArc | null = null;
    let bestDist = threshold;

    for (const arc of this.arcs) {
      const mid = this.arcMidpoint(arc);
      const dist = Math.hypot(mid.x - worldX, mid.y - worldY);
      if (dist < bestDist) {
        bestDist = dist;
        best = arc;
      }
    }
    return best;
  }

  // ---- adjacency mirror -------------------------------------------------

  /** Rebuild the layout graph's neighbor lists from the deduped arc set. */
  private rebuildAdjacency(): void {
    this.graph.nodes.forEach((node) => {
      node.neighbors.length = 0;
    });

    this.arcs.forEach((arc) => {
      if (arc.from === arc.to) {
        return;
      }
      const from = this.nodeById(arc.from);
      const to = this.nodeById(arc.to);
      if (from && to) {
        from.addNeighbor(to);
        to.addNeighbor(from);
      }
    });
  }

  // ---- mutations --------------------------------------------------------

  private mod(value: number): number {
    const n = this.groupOrder;
    return ((value % n) + n) % n;
  }

  private addNodeAt(x: number, y: number): void {
    this.graph.addNode(x, y);
    this.emitChange();
  }

  private addArc(from: GraphNode, to: GraphNode): void {
    const arc: BaseArc = { id: this.nextArcId, from: from.id, to: to.id, voltage: 0 };
    this.nextArcId += 1;
    this.arcs.push(arc);
    this.selectedArcId = arc.id;
    this.rebuildAdjacency();
    this.emitChange();
  }

  private deleteNode(node: GraphNode): void {
    this.arcs = this.arcs.filter((arc) => arc.from !== node.id && arc.to !== node.id);
    if (this.selectedNode === node) {
      this.selectedNode = null;
    }
    this.graph.removeNode(node);
    this.rebuildAdjacency();
    this.emitChange();
  }

  private deleteSelectedArc(): void {
    if (this.selectedArcId === null) {
      return;
    }
    this.arcs = this.arcs.filter((arc) => arc.id !== this.selectedArcId);
    this.selectedArcId = null;
    this.rebuildAdjacency();
    this.emitChange();
  }

  private changeSelectedVoltage(delta: number): void {
    const arc = this.arcs.find((candidate) => candidate.id === this.selectedArcId);
    if (!arc) {
      return;
    }
    arc.voltage = this.mod(arc.voltage + delta);
    this.emitChange();
  }

  private reverseSelectedArc(): void {
    const arc = this.arcs.find((candidate) => candidate.id === this.selectedArcId);
    if (!arc) {
      return;
    }
    const from = arc.from;
    arc.from = arc.to;
    arc.to = from;
    arc.voltage = this.mod(this.groupOrder - arc.voltage);
    this.emitChange();
  }

  private emitChange(): void {
    this.callbacks.onChange?.();
  }

  // ---- mouse handlers ---------------------------------------------------

  private handleMouseMove(event: MouseEvent): void {
    const canvasPos = this.getCanvasPosFromClient(event.clientX, event.clientY);

    if (this.panningCanvas) {
      this.offsetX += canvasPos.x - this.panStartX;
      this.offsetY += canvasPos.y - this.panStartY;
      this.panStartX = canvasPos.x;
      this.panStartY = canvasPos.y;
      return;
    }

    const pos = this.getMousePos(event);
    this.mouseX = pos.x;
    this.mouseY = pos.y;

    if (this.draggingNode) {
      this.draggingNode.x = pos.x;
      this.draggingNode.y = pos.y;
      this.draggingNode.vx = 0;
      this.draggingNode.vy = 0;
    }
  }

  private handleLeftClick(event: MouseEvent): void {
    const pos = this.getMousePos(event);
    const node = this.graph.findNodeAt(pos.x, pos.y);

    if (node) {
      if (this.arcStart) {
        this.addArc(this.arcStart, node);
        this.arcStart = null;
        return;
      }
      this.selectedNode = node;
      this.selectedArcId = null;
      this.draggingNode = node;
      node.fixed = true;
      return;
    }

    if (this.arcStart) {
      this.arcStart = null;
      return;
    }

    const arc = this.findArcNear(pos.x, pos.y);
    if (arc) {
      this.selectedArcId = arc.id;
      this.selectedNode = null;
      return;
    }

    const canvasPos = this.getCanvasPosFromClient(event.clientX, event.clientY);
    this.panningCanvas = true;
    this.panStartX = canvasPos.x;
    this.panStartY = canvasPos.y;
    this.selectedNode = null;
    this.selectedArcId = null;
  }

  private handleLeftRelease(): void {
    this.panningCanvas = false;
    if (this.draggingNode) {
      this.draggingNode.fixed = false;
      this.draggingNode = null;
    }
  }

  private handleRightClick(event: MouseEvent): void {
    const pos = this.getMousePos(event);
    const node = this.graph.findNodeAt(pos.x, pos.y);

    if (node) {
      if (!this.arcStart) {
        this.arcStart = node;
      } else {
        this.addArc(this.arcStart, node);
        this.arcStart = null;
      }
      return;
    }
    this.arcStart = null;
  }

  private handleDoubleClick(event: MouseEvent): void {
    const pos = this.getMousePos(event);
    const node = this.graph.findNodeAt(pos.x, pos.y);
    if (node) {
      return;
    }
    this.addNodeAt(pos.x, pos.y);
  }

  private handleWheel(event: WheelEvent): void {
    event.preventDefault();

    const canvasPos = this.getCanvasPosFromClient(event.clientX, event.clientY);
    const delta = -Math.sign(event.deltaY);
    const zoomFactor = 1 + delta * this.scaleStep;
    const newScale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * zoomFactor));
    if (newScale === this.scale) {
      return;
    }

    const worldX = (canvasPos.x - this.offsetX) / this.scale;
    const worldY = (canvasPos.y - this.offsetY) / this.scale;
    this.scale = newScale;
    this.offsetX = canvasPos.x - worldX * this.scale;
    this.offsetY = canvasPos.y - worldY * this.scale;
  }

  private handleKeyDown(event: KeyboardEvent): void {
    const target = event.target as HTMLElement | null;
    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) {
      return;
    }

    if (event.key === "Backspace" || event.key === "Delete") {
      if (this.selectedArcId !== null) {
        event.preventDefault();
        this.deleteSelectedArc();
      } else if (this.selectedNode !== null) {
        event.preventDefault();
        this.deleteNode(this.selectedNode);
      }
      return;
    }

    if (this.selectedArcId === null) {
      return;
    }

    if (event.key === "+" || event.key === "=" || event.key === "ArrowUp") {
      event.preventDefault();
      this.changeSelectedVoltage(1);
    } else if (event.key === "-" || event.key === "_" || event.key === "ArrowDown") {
      event.preventDefault();
      this.changeSelectedVoltage(-1);
    } else if (event.key === "r" || event.key === "R") {
      event.preventDefault();
      this.reverseSelectedArc();
    }
  }

  // ---- rendering --------------------------------------------------------

  private startAnimationLoop(): void {
    const animate = (timestamp: number) => {
      if (this.disposed) {
        return;
      }
      if (this.lastFrameTime === 0) {
        this.lastFrameTime = timestamp;
      }
      const deltaTime = timestamp - this.lastFrameTime;
      this.lastFrameTime = timestamp;

      this.physics.step(this.canvas.width, this.canvas.height, deltaTime);
      this.render();
      this.animationFrameId = window.requestAnimationFrame(animate);
    };
    this.animationFrameId = window.requestAnimationFrame(animate);
  }

  private stopAnimationLoop(): void {
    if (this.animationFrameId !== null) {
      window.cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  private drawArrowhead(x: number, y: number, angle: number, color: string): void {
    const size = 10;
    this.ctx.save();
    this.ctx.translate(x, y);
    this.ctx.rotate(angle);
    this.ctx.fillStyle = color;
    this.ctx.beginPath();
    this.ctx.moveTo(0, 0);
    this.ctx.lineTo(-size, -size * 0.45);
    this.ctx.lineTo(-size, size * 0.45);
    this.ctx.closePath();
    this.ctx.fill();
    this.ctx.restore();
  }

  private drawVoltageLabel(x: number, y: number, voltage: number, highlighted: boolean): void {
    const theme = this.themeColors;
    const text = String(voltage);
    this.ctx.font = "bold 13px Arial";
    const padding = 4;
    const metrics = this.ctx.measureText(text);
    const width = metrics.width + padding * 2;
    const height = 18;

    this.ctx.fillStyle = highlighted ? theme.nodeSelected : theme.background;
    this.ctx.strokeStyle = highlighted ? theme.text : theme.edge;
    this.ctx.lineWidth = 1;
    const rx = x - width / 2;
    const ry = y - height / 2;
    const r = 5;
    this.ctx.beginPath();
    this.ctx.moveTo(rx + r, ry);
    this.ctx.arcTo(rx + width, ry, rx + width, ry + height, r);
    this.ctx.arcTo(rx + width, ry + height, rx, ry + height, r);
    this.ctx.arcTo(rx, ry + height, rx, ry, r);
    this.ctx.arcTo(rx, ry, rx + width, ry, r);
    this.ctx.closePath();
    this.ctx.fill();
    this.ctx.stroke();

    this.ctx.fillStyle = theme.text;
    this.ctx.textAlign = "center";
    this.ctx.textBaseline = "middle";
    this.ctx.fillText(text, x, y);
  }

  private renderArc(arc: BaseArc): void {
    const theme = this.themeColors;
    const from = this.nodeById(arc.from);
    const to = this.nodeById(arc.to);
    if (!from || !to) {
      return;
    }

    const selected = arc.id === this.selectedArcId;
    const strokeColor = selected ? theme.nodeSelected : theme.edge;
    this.ctx.strokeStyle = strokeColor;
    this.ctx.lineWidth = selected ? 3.5 : 2;

    if (arc.from === arc.to) {
      const loop = this.selfLoopGeometry(from, arc);
      this.ctx.beginPath();
      this.ctx.arc(loop.cx, loop.cy, loop.radius, 0, Math.PI * 2);
      this.ctx.stroke();

      // Arrowhead on the loop's right side, tangent pointing clockwise.
      this.drawArrowhead(loop.cx + loop.radius, loop.cy, Math.PI / 2, strokeColor);
      this.drawVoltageLabel(loop.cx, loop.cy - loop.radius, arc.voltage, selected);
      return;
    }

    const offset = this.arcOffset(arc);
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    const nx = -uy;
    const ny = ux;

    // Trim endpoints to node borders.
    const startX = from.x + ux * from.radius;
    const startY = from.y + uy * from.radius;
    const endX = to.x - ux * to.radius;
    const endY = to.y - uy * to.radius;

    let arrowX: number;
    let arrowY: number;
    let arrowAngle: number;

    if (offset === 0) {
      this.ctx.beginPath();
      this.ctx.moveTo(startX, startY);
      this.ctx.lineTo(endX, endY);
      this.ctx.stroke();
      arrowX = endX;
      arrowY = endY;
      arrowAngle = Math.atan2(endY - startY, endX - startX);
    } else {
      const cx = (from.x + to.x) / 2 + nx * offset * 2;
      const cy = (from.y + to.y) / 2 + ny * offset * 2;
      this.ctx.beginPath();
      this.ctx.moveTo(startX, startY);
      this.ctx.quadraticCurveTo(cx, cy, endX, endY);
      this.ctx.stroke();
      // Tangent at end of quadratic bezier points from control to end.
      arrowX = endX;
      arrowY = endY;
      arrowAngle = Math.atan2(endY - cy, endX - cx);
    }

    this.drawArrowhead(arrowX, arrowY, arrowAngle, strokeColor);

    const mid = this.arcMidpoint(arc);
    this.drawVoltageLabel(mid.x, mid.y, arc.voltage, selected);
  }

  private render(): void {
    const theme = this.themeColors;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    this.ctx.save();
    this.ctx.translate(this.offsetX, this.offsetY);
    this.ctx.scale(this.scale, this.scale);

    this.arcs.forEach((arc) => this.renderArc(arc));

    if (this.arcStart) {
      this.ctx.strokeStyle = theme.dimColor;
      this.ctx.setLineDash([5, 5]);
      this.ctx.beginPath();
      this.ctx.moveTo(this.arcStart.x, this.arcStart.y);
      this.ctx.lineTo(this.mouseX, this.mouseY);
      this.ctx.stroke();
      this.ctx.setLineDash([]);
    }

    this.graph.nodes.forEach((node) => {
      const isSelected = node === this.selectedNode;
      this.ctx.fillStyle = isSelected ? theme.nodeSelected : theme.node;
      this.ctx.beginPath();
      this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
      this.ctx.fill();

      this.ctx.fillStyle = theme.text;
      this.ctx.font = "bold 14px Arial";
      this.ctx.textAlign = "center";
      this.ctx.textBaseline = "middle";
      this.ctx.fillText(node.id.toString(), node.x, node.y);
    });

    this.ctx.restore();
  }

  private preSettlePhysics(iterations: number): void {
    const wasEnabled = this.physics.enabled;
    this.physics.enabled = true;
    for (let index = 0; index < iterations; index += 1) {
      this.physics.step(this.canvas.width, this.canvas.height, 16.67);
    }
    this.physics.enabled = wasEnabled;
  }

  // ---- public API -------------------------------------------------------

  public setGroupOrder(n: number): void {
    this.groupOrder = Math.max(2, Math.floor(n));
    this.arcs.forEach((arc) => {
      arc.voltage = this.mod(arc.voltage);
    });
    this.emitChange();
  }

  public getGroupOrder(): number {
    return this.groupOrder;
  }

  public getNodeIds(): number[] {
    return this.graph.nodes.map((node) => node.id).sort((a, b) => a - b);
  }

  public getArcs(): BaseArc[] {
    return this.arcs.map((arc) => ({ ...arc }));
  }

  public loadPreset(nodeIds: number[], arcs: BaseArc[]): void {
    this.graph.clear();
    this.arcs = [];
    this.selectedNode = null;
    this.selectedArcId = null;
    this.arcStart = null;

    const sortedIds = [...nodeIds].sort((a, b) => a - b);
    const centerX = this.canvas.width / 2;
    const centerY = this.canvas.height / 2;
    const radius = Math.min(centerX, centerY) * 0.5 + 60;

    sortedIds.forEach((id, index) => {
      const angle = (index / Math.max(1, sortedIds.length)) * Math.PI * 2;
      const node = new GraphNode(id, centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius);
      this.graph.nodes.push(node);
    });
    this.graph.nextNodeId = sortedIds.length ? Math.max(...sortedIds) + 1 : 0;

    let maxArcId = -1;
    this.arcs = arcs.map((arc) => {
      maxArcId = Math.max(maxArcId, arc.id);
      return { id: arc.id, from: arc.from, to: arc.to, voltage: this.mod(arc.voltage) };
    });
    this.nextArcId = maxArcId + 1;

    this.rebuildAdjacency();
    if (this.graph.nodes.length > 0) {
      this.preSettlePhysics(200);
    }
    this.emitChange();
  }

  public clear(): void {
    this.graph.clear();
    this.arcs = [];
    this.selectedNode = null;
    this.selectedArcId = null;
    this.arcStart = null;
    this.nextArcId = 0;
    this.emitChange();
  }

  public enablePhysics(): void {
    this.physics.enabled = true;
  }

  public disablePhysics(): void {
    this.physics.enabled = false;
    this.graph.nodes.forEach((node) => {
      node.vx = 0;
      node.vy = 0;
    });
  }

  public isPhysicsEnabled(): boolean {
    return this.physics.enabled;
  }

  public destroy(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.stopAnimationLoop();
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    this.unbinders.forEach((unbind) => {
      unbind();
    });
    this.unbinders.length = 0;
  }
}

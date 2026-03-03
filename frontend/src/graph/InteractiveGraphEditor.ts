import type { GraphNodePredictionView, GraphPrediction } from "../types/graph";
import { Graph } from "./Graph";
import { GraphNode } from "./GraphNode";
import { PhysicsEngine } from "./PhysicsEngine";

interface InteractiveGraphCallbacks {
  onGraphChange?: (edgeList: string) => void;
  onAnalyzeRequest?: () => void;
}

interface Point {
  x: number;
  y: number;
}

interface TouchPointerState {
  pointerId: number;
  clientX: number;
  clientY: number;
  canvasX: number;
  canvasY: number;
  worldX: number;
  worldY: number;
  startClientX: number;
  startClientY: number;
  startCanvasX: number;
  startCanvasY: number;
  lastClientX: number;
  lastClientY: number;
  lastCanvasX: number;
  lastCanvasY: number;
  startNode: GraphNode | null;
  moved: boolean;
  suppressTap: boolean;
  longPressTriggered: boolean;
}

interface TouchPinchState {
  startDistance: number;
  startScale: number;
  midpointWorldX: number;
  midpointWorldY: number;
}

export class InteractiveGraphEditor {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly graph = new Graph();
  private readonly physics = new PhysicsEngine(this.graph);
  private readonly callbacks: InteractiveGraphCallbacks;

  private selectedNode: GraphNode | null = null;
  private draggingNode: GraphNode | null = null;
  private edgeStart: GraphNode | null = null;
  private panningCanvas = false;
  private panStartX = 0;
  private panStartY = 0;

  private offsetX = 0;
  private offsetY = 0;

  private scale = 1.0;
  private readonly minScale = 0.1;
  private readonly maxScale = 5.0;
  private readonly scaleStep = 0.033;

  private readonly backgroundColor = "#1a1a1a";
  private readonly edgeColor = "#555555";
  private readonly textColor = "#ffffff";
  private readonly errorDotColor = "#ff0000";

  private mouseX = 0;
  private mouseY = 0;

  private readonly longPressDelayMs = 450;
  private readonly touchMoveThresholdPx = 8;
  private readonly doubleTapWindowMs = 280;
  private readonly doubleTapTolerancePx = 24;

  private readonly touchPointers = new Map<number, TouchPointerState>();
  private touchPrimaryPointerId: number | null = null;
  private touchLongPressTimer: number | null = null;
  private touchPinchState: TouchPinchState | null = null;

  private lastTapTime = 0;
  private lastTapX = 0;
  private lastTapY = 0;
  private lastTapNodeId: number | null = null;
  private lastTouchInteractionTime = 0;

  private resizeObserver: ResizeObserver | null = null;
  private animationFrameId: number | null = null;
  private lastFrameTime = 0;
  private analyzeTimeoutId: number | null = null;

  private disposed = false;
  private readonly unbinders: Array<() => void> = [];

  public constructor(canvas: HTMLCanvasElement, callbacks: InteractiveGraphCallbacks = {}) {
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
        } else if (mouseEvent.button === 1) {
          mouseEvent.preventDefault();
          this.handleMiddleClick(mouseEvent);
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

    this.bindEvent(
      this.canvas,
      "dblclick",
      (event) => this.handleDoubleClick(event as MouseEvent),
      { passive: false }
    );

    this.bindEvent(this.canvas, "wheel", (event) => this.handleWheel(event as WheelEvent), {
      passive: false
    });

    this.bindEvent(
      this.canvas,
      "pointerdown",
      (event) => this.handlePointerDown(event as PointerEvent),
      { passive: false }
    );

    this.bindEvent(
      this.canvas,
      "pointermove",
      (event) => this.handlePointerMove(event as PointerEvent),
      { passive: false }
    );

    this.bindEvent(
      this.canvas,
      "pointerup",
      (event) => this.handlePointerUp(event as PointerEvent),
      { passive: false }
    );

    this.bindEvent(
      this.canvas,
      "pointercancel",
      (event) => this.handlePointerCancel(event as PointerEvent),
      { passive: false }
    );

    this.bindEvent(document, "keydown", (event) => {
      const keyboardEvent = event as KeyboardEvent;
      if (keyboardEvent.key === "Backspace" && this.selectedNode !== null) {
        keyboardEvent.preventDefault();
        this.deleteNode(this.selectedNode);
      }
    });
  }

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

  private distanceBetweenPoints(x1: number, y1: number, x2: number, y2: number): number {
    const dx = x2 - x1;
    const dy = y2 - y1;
    return Math.sqrt(dx * dx + dy * dy);
  }

  private updateTouchScrollLock(): void {
    const hasTouchPointers = this.touchPointers.size > 0;
    document.body.classList.toggle("touch-graph-lock", hasTouchPointers);
  }

  private cancelTouchLongPress(): void {
    if (this.touchLongPressTimer !== null) {
      window.clearTimeout(this.touchLongPressTimer);
      this.touchLongPressTimer = null;
    }
  }

  private resetTouchDragAndPan(): void {
    this.panningCanvas = false;
    if (this.draggingNode) {
      this.draggingNode.fixed = false;
      this.draggingNode = null;
    }
  }

  private beginTouchLongPress(pointer: TouchPointerState): void {
    if (!pointer.startNode || pointer.suppressTap) return;

    this.cancelTouchLongPress();
    this.touchLongPressTimer = window.setTimeout(() => {
      const stillActive = this.touchPointers.get(pointer.pointerId);
      if (!stillActive || stillActive.suppressTap || this.touchPinchState) {
        return;
      }

      const startNode = stillActive.startNode;
      if (!startNode) {
        return;
      }

      if (this.selectedNode === startNode) {
        if (this.edgeStart === stillActive.startNode) {
          this.edgeStart = null;
        }
        this.deleteNode(startNode);
      } else {
        this.selectedNode = startNode;
      }

      stillActive.longPressTriggered = true;
      stillActive.suppressTap = true;
      this.touchLongPressTimer = null;
    }, this.longPressDelayMs);
  }

  private setupSingleTouchState(pointer: TouchPointerState, reuseStartState = false): void {
    this.touchPrimaryPointerId = pointer.pointerId;
    pointer.lastClientX = pointer.clientX;
    pointer.lastClientY = pointer.clientY;
    pointer.lastCanvasX = pointer.canvasX;
    pointer.lastCanvasY = pointer.canvasY;
    pointer.moved = false;
    pointer.longPressTriggered = false;

    if (!reuseStartState) {
      pointer.startClientX = pointer.clientX;
      pointer.startClientY = pointer.clientY;
      pointer.startCanvasX = pointer.canvasX;
      pointer.startCanvasY = pointer.canvasY;
      pointer.startNode = this.graph.findNodeAt(pointer.worldX, pointer.worldY);
      pointer.suppressTap = false;
    }

    this.beginTouchLongPress(pointer);
  }

  private initializeTouchPinch(): void {
    this.cancelTouchLongPress();
    this.resetTouchDragAndPan();

    const pointers = Array.from(this.touchPointers.values());
    if (pointers.length < 2) {
      return;
    }

    pointers.forEach((pointer) => {
      pointer.suppressTap = true;
    });

    const p1 = pointers[0];
    const p2 = pointers[1];
    const distance = this.distanceBetweenPoints(p1.clientX, p1.clientY, p2.clientX, p2.clientY);

    if (distance < 1) {
      return;
    }

    const midX = (p1.clientX + p2.clientX) / 2;
    const midY = (p1.clientY + p2.clientY) / 2;
    const midpointWorld = this.getWorldPosFromClient(midX, midY);

    this.touchPinchState = {
      startDistance: distance,
      startScale: this.scale,
      midpointWorldX: midpointWorld.x,
      midpointWorldY: midpointWorld.y
    };
  }

  private updateTouchPinch(): void {
    if (!this.touchPinchState || this.touchPointers.size < 2) {
      return;
    }

    const pointers = Array.from(this.touchPointers.values());
    const p1 = pointers[0];
    const p2 = pointers[1];
    const distance = this.distanceBetweenPoints(p1.clientX, p1.clientY, p2.clientX, p2.clientY);

    if (distance < 1) {
      return;
    }

    const midpointX = (p1.clientX + p2.clientX) / 2;
    const midpointY = (p1.clientY + p2.clientY) / 2;
    const canvasMidpoint = this.getCanvasPosFromClient(midpointX, midpointY);

    const zoomFactor = distance / this.touchPinchState.startDistance;
    const unclampedScale = this.touchPinchState.startScale * zoomFactor;
    const newScale = Math.max(this.minScale, Math.min(this.maxScale, unclampedScale));

    this.scale = newScale;
    this.offsetX = canvasMidpoint.x - this.touchPinchState.midpointWorldX * this.scale;
    this.offsetY = canvasMidpoint.y - this.touchPinchState.midpointWorldY * this.scale;

    this.mouseX = this.touchPinchState.midpointWorldX;
    this.mouseY = this.touchPinchState.midpointWorldY;
  }

  private handleTouchTap(pointer: TouchPointerState): void {
    const node = this.graph.findNodeAt(pointer.worldX, pointer.worldY);
    const now = Date.now();
    const withinWindow = now - this.lastTapTime <= this.doubleTapWindowMs;
    const withinDistance =
      this.distanceBetweenPoints(pointer.clientX, pointer.clientY, this.lastTapX, this.lastTapY) <=
      this.doubleTapTolerancePx;
    const isDoubleTap = withinWindow && withinDistance;

    if (isDoubleTap) {
      if (node && this.lastTapNodeId === node.id) {
        this.lastTapTime = 0;
        this.lastTapNodeId = null;
        this.selectedNode = node;
        this.edgeStart = node;
        return;
      }

      if (!node && this.lastTapNodeId === null) {
        this.lastTapTime = 0;
        this.lastTapNodeId = null;
        this.addNode(pointer.worldX, pointer.worldY);
        return;
      }
    }

    this.lastTapTime = now;
    this.lastTapX = pointer.clientX;
    this.lastTapY = pointer.clientY;
    this.lastTapNodeId = node ? node.id : null;

    if (node) {
      if (this.edgeStart !== null) {
        this.addEdge(this.edgeStart, node);
        this.edgeStart = null;
      } else {
        this.selectedNode = node;
      }
      return;
    }

    if (this.edgeStart !== null) {
      this.edgeStart = null;
    } else {
      this.selectedNode = null;
    }
  }

  private handlePointerDown(event: PointerEvent): void {
    if (event.pointerType === "mouse") {
      return;
    }

    this.lastTouchInteractionTime = Date.now();
    event.preventDefault();

    try {
      this.canvas.setPointerCapture(event.pointerId);
    } catch {
      // no-op
    }

    const world = this.getWorldPosFromClient(event.clientX, event.clientY);
    const canvasPosition = this.getCanvasPosFromClient(event.clientX, event.clientY);

    const pointer: TouchPointerState = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      canvasX: canvasPosition.x,
      canvasY: canvasPosition.y,
      worldX: world.x,
      worldY: world.y,
      startClientX: event.clientX,
      startClientY: event.clientY,
      startCanvasX: canvasPosition.x,
      startCanvasY: canvasPosition.y,
      lastClientX: event.clientX,
      lastClientY: event.clientY,
      lastCanvasX: canvasPosition.x,
      lastCanvasY: canvasPosition.y,
      startNode: null,
      moved: false,
      suppressTap: false,
      longPressTriggered: false
    };

    this.touchPointers.set(event.pointerId, pointer);
    this.updateTouchScrollLock();

    if (this.touchPointers.size === 1) {
      this.touchPinchState = null;
      this.setupSingleTouchState(pointer);
      return;
    }

    this.initializeTouchPinch();
  }

  private handlePointerMove(event: PointerEvent): void {
    if (event.pointerType === "mouse") {
      return;
    }

    const pointer = this.touchPointers.get(event.pointerId);
    if (!pointer) {
      return;
    }

    this.lastTouchInteractionTime = Date.now();
    event.preventDefault();

    pointer.clientX = event.clientX;
    pointer.clientY = event.clientY;

    const canvasPos = this.getCanvasPosFromClient(event.clientX, event.clientY);
    pointer.canvasX = canvasPos.x;
    pointer.canvasY = canvasPos.y;

    const world = this.getWorldPosFromClient(event.clientX, event.clientY);
    pointer.worldX = world.x;
    pointer.worldY = world.y;

    this.mouseX = world.x;
    this.mouseY = world.y;

    if (this.touchPointers.size >= 2) {
      if (!this.touchPinchState) {
        this.initializeTouchPinch();
      }
      this.updateTouchPinch();
      return;
    }

    if (this.touchPrimaryPointerId !== event.pointerId) {
      this.setupSingleTouchState(pointer, true);
    }

    const movedDistance = this.distanceBetweenPoints(
      pointer.startClientX,
      pointer.startClientY,
      pointer.clientX,
      pointer.clientY
    );

    if (
      !pointer.moved &&
      !pointer.longPressTriggered &&
      movedDistance > this.touchMoveThresholdPx
    ) {
      pointer.moved = true;
      pointer.suppressTap = true;
      this.cancelTouchLongPress();

      if (pointer.startNode) {
        this.selectedNode = pointer.startNode;
        this.draggingNode = pointer.startNode;
        this.draggingNode.fixed = true;
      } else {
        this.panningCanvas = true;
      }
    }

    if (this.draggingNode) {
      this.draggingNode.x = pointer.worldX;
      this.draggingNode.y = pointer.worldY;
      this.draggingNode.vx = 0;
      this.draggingNode.vy = 0;
    } else if (this.panningCanvas) {
      this.offsetX += pointer.canvasX - pointer.lastCanvasX;
      this.offsetY += pointer.canvasY - pointer.lastCanvasY;
    }

    pointer.lastClientX = pointer.clientX;
    pointer.lastClientY = pointer.clientY;
    pointer.lastCanvasX = pointer.canvasX;
    pointer.lastCanvasY = pointer.canvasY;
  }

  private finalizePointer(event: PointerEvent, allowTap: boolean): void {
    const pointer = this.touchPointers.get(event.pointerId);
    if (!pointer) {
      return;
    }

    this.lastTouchInteractionTime = Date.now();
    event.preventDefault();

    this.touchPointers.delete(event.pointerId);
    this.cancelTouchLongPress();

    const canTap =
      allowTap && !pointer.suppressTap && !pointer.longPressTriggered && !this.touchPinchState;

    if (canTap) {
      this.handleTouchTap(pointer);
    }

    if (this.touchPointers.size === 0) {
      this.touchPrimaryPointerId = null;
      this.touchPinchState = null;
      this.resetTouchDragAndPan();
      this.updateTouchScrollLock();
      return;
    }

    this.resetTouchDragAndPan();

    if (this.touchPointers.size >= 2) {
      this.initializeTouchPinch();
    } else {
      this.touchPinchState = null;
      const [remainingPointer] = Array.from(this.touchPointers.values());
      remainingPointer.startClientX = remainingPointer.clientX;
      remainingPointer.startClientY = remainingPointer.clientY;
      remainingPointer.startCanvasX = remainingPointer.canvasX;
      remainingPointer.startCanvasY = remainingPointer.canvasY;
      remainingPointer.lastClientX = remainingPointer.clientX;
      remainingPointer.lastClientY = remainingPointer.clientY;
      remainingPointer.lastCanvasX = remainingPointer.canvasX;
      remainingPointer.lastCanvasY = remainingPointer.canvasY;
      remainingPointer.moved = false;
      remainingPointer.suppressTap = true;
      this.setupSingleTouchState(remainingPointer, true);
    }

    this.updateTouchScrollLock();
  }

  private handlePointerUp(event: PointerEvent): void {
    if (event.pointerType === "mouse") {
      return;
    }

    this.finalizePointer(event, true);

    try {
      this.canvas.releasePointerCapture(event.pointerId);
    } catch {
      // no-op
    }
  }

  private handlePointerCancel(event: PointerEvent): void {
    if (event.pointerType === "mouse") {
      return;
    }

    this.finalizePointer(event, false);
  }

  private handleMouseMove(event: MouseEvent): void {
    const canvasPos = this.getCanvasPosFromClient(event.clientX, event.clientY);

    if (this.panningCanvas) {
      const dx = canvasPos.x - this.panStartX;
      const dy = canvasPos.y - this.panStartY;
      this.offsetX += dx;
      this.offsetY += dy;
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
      if (this.edgeStart) {
        this.addEdge(this.edgeStart, node);
        this.edgeStart = null;
        return;
      }

      this.selectedNode = node;
      this.draggingNode = node;
      node.fixed = true;
      return;
    }

    if (this.edgeStart) {
      const newNode = this.graph.addNode(pos.x, pos.y);
      this.addEdge(this.edgeStart, newNode);
      this.edgeStart = null;
      this.emitGraphChange();
      this.triggerAnalyze();
      return;
    }

    const canvasPos = this.getCanvasPosFromClient(event.clientX, event.clientY);
    this.panningCanvas = true;
    this.panStartX = canvasPos.x;
    this.panStartY = canvasPos.y;
    this.selectedNode = null;
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
      if (!this.edgeStart) {
        this.edgeStart = node;
      } else {
        this.addEdge(this.edgeStart, node);
        this.edgeStart = null;
      }
      return;
    }

    this.edgeStart = null;
  }

  private handleMiddleClick(event: MouseEvent): void {
    const pos = this.getMousePos(event);
    const node = this.graph.findNodeAt(pos.x, pos.y);

    if (node) {
      this.deleteNode(node);
    }
  }

  private handleDoubleClick(event: MouseEvent): void {
    if (Date.now() - this.lastTouchInteractionTime < 500) {
      return;
    }

    const pos = this.getMousePos(event);
    const node = this.graph.findNodeAt(pos.x, pos.y);

    if (node) {
      this.selectedNode = node;
      this.edgeStart = node;
      return;
    }

    this.addNode(pos.x, pos.y);
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

  private addNode(x: number, y: number): void {
    this.graph.clearPredictions();
    this.graph.addNode(x, y);
    this.emitGraphChange();
    this.triggerAnalyze();
  }

  private deleteNode(node: GraphNode): void {
    this.graph.clearPredictions();

    if (this.selectedNode === node) {
      this.selectedNode = null;
    }

    this.graph.removeNode(node);
    this.emitGraphChange();
    this.triggerAnalyze();
  }

  private addEdge(fromNode: GraphNode, toNode: GraphNode): void {
    this.graph.clearPredictions();
    this.graph.addEdge(fromNode, toNode);
    this.emitGraphChange();
    this.triggerAnalyze();
  }

  private emitGraphChange(): void {
    this.callbacks.onGraphChange?.(this.graph.toEdgeList());
  }

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

  private drawSelfLoop(node: GraphNode): void {
    const loopRadius = node.radius * 0.8;
    const loopCenterY = node.y - node.radius - loopRadius;

    this.ctx.beginPath();
    this.ctx.arc(node.x, loopCenterY, loopRadius, 0, Math.PI * 2);
    this.ctx.stroke();
  }

  private render(): void {
    this.ctx.fillStyle = this.backgroundColor;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    this.ctx.save();
    this.ctx.translate(this.offsetX, this.offsetY);
    this.ctx.scale(this.scale, this.scale);

    this.ctx.strokeStyle = this.edgeColor;
    this.ctx.lineWidth = 2;

    const edges = this.graph.getEdges();
    edges.forEach((edge) => {
      if (edge.from === edge.to) {
        this.drawSelfLoop(edge.from);
      } else {
        this.ctx.beginPath();
        this.ctx.moveTo(edge.from.x, edge.from.y);
        this.ctx.lineTo(edge.to.x, edge.to.y);
        this.ctx.stroke();
      }
    });

    if (this.edgeStart) {
      this.ctx.strokeStyle = "#888888";
      this.ctx.setLineDash([5, 5]);
      this.ctx.beginPath();
      this.ctx.moveTo(this.edgeStart.x, this.edgeStart.y);
      this.ctx.lineTo(this.mouseX, this.mouseY);
      this.ctx.stroke();
      this.ctx.setLineDash([]);
    }

    this.graph.nodes.forEach((node) => {
      const isSelected = node === this.selectedNode;
      const fillColor = isSelected ? node.selectedColor : node.color;

      this.ctx.fillStyle = fillColor;
      this.ctx.beginPath();
      this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
      this.ctx.fill();

      this.ctx.fillStyle = this.textColor;
      this.ctx.font = "bold 14px Arial";
      this.ctx.textAlign = "center";
      this.ctx.textBaseline = "middle";
      this.ctx.fillText(node.id.toString(), node.x, node.y);

      if (node.hasPrediction()) {
        const offsetX = node.x + node.radius * 0.8;
        const offsetY = node.y + node.radius * 0.8;

        this.ctx.font = "bold 14px Arial";
        this.ctx.textAlign = "left";
        this.ctx.textBaseline = "middle";

        const isCorrect = node.predicted === node.actual;
        const correctColor = isCorrect ? "#888888" : "#00ff00";
        const predictedColor = isCorrect ? "#888888" : this.errorDotColor;

        this.ctx.fillStyle = correctColor;
        const correctText = String(node.actual);
        this.ctx.fillText(correctText, offsetX, offsetY);

        const correctWidth = this.ctx.measureText(correctText).width;
        this.ctx.fillStyle = isCorrect ? "#888888" : this.textColor;
        const slashX = offsetX + correctWidth;
        this.ctx.fillText("/", slashX, offsetY);
        const slashWidth = this.ctx.measureText("/").width;

        this.ctx.fillStyle = predictedColor;
        this.ctx.fillText(String(node.predicted), slashX + slashWidth, offsetY);
      }
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

  private triggerAnalyze(): void {
    if (!this.graph.nodes.length || this.graph.hasPredictions()) {
      return;
    }

    if (this.analyzeTimeoutId !== null) {
      window.clearTimeout(this.analyzeTimeoutId);
    }

    this.analyzeTimeoutId = window.setTimeout(() => {
      this.callbacks.onAnalyzeRequest?.();
      this.analyzeTimeoutId = null;
    }, 100);
  }

  public loadFromEdgeList(edgeListText: string): void {
    this.graph.fromEdgeList(edgeListText, this.canvas.width, this.canvas.height);
    this.selectedNode = null;

    if (this.graph.nodes.length > 0) {
      this.preSettlePhysics(200);
    }
  }

  public clear(): void {
    this.graph.clear();
    this.selectedNode = null;
    this.edgeStart = null;
    this.emitGraphChange();
  }

  public clearPredictions(): void {
    this.graph.clearPredictions();
  }

  public updatePredictions(predictions: GraphPrediction[]): void {
    this.graph.updatePredictions(predictions);
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

  public get nodes(): GraphNode[] {
    return this.graph.nodes;
  }

  public getNodePredictions(): Map<number, GraphNodePredictionView> {
    const map = new Map<number, GraphNodePredictionView>();

    this.graph.nodes.forEach((node) => {
      if (node.hasPrediction() && node.predicted !== null && node.actual !== null) {
        map.set(node.id, {
          predicted: node.predicted,
          actual: node.actual,
          isWrong: node.isWrong
        });
      }
    });

    return map;
  }

  public hasPredictions(): boolean {
    return this.graph.hasPredictions();
  }

  public renderNow(): void {
    this.render();
  }

  public destroy(): void {
    if (this.disposed) {
      return;
    }

    this.disposed = true;

    this.stopAnimationLoop();
    this.cancelTouchLongPress();

    if (this.analyzeTimeoutId !== null) {
      window.clearTimeout(this.analyzeTimeoutId);
      this.analyzeTimeoutId = null;
    }

    this.resizeObserver?.disconnect();
    this.resizeObserver = null;

    this.unbinders.forEach((unbind) => {
      unbind();
    });

    this.unbinders.length = 0;
    document.body.classList.remove("touch-graph-lock");
  }
}

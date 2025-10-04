/**
 * Interactive Graph Canvas
 * Allows users to create and manipulate graphs visually
 */

class InteractiveGraph {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        // Graph data
        this.nodes = [];
        this.edges = [];
        this.nextNodeId = 0;
        
        // Interaction state
        this.selectedNode = null;  // Currently selected node (single click)
        this.targetNode = null;    // Target node for analysis (double click)
        this.draggingNode = null;
        this.edgeStart = null;     // For drawing edges with right-click
        this.panningCanvas = false; // For panning the canvas
        this.panStartX = 0;
        this.panStartY = 0;
        
        // Canvas offset for panning
        this.offsetX = 0;
        this.offsetY = 0;
        
        // Prediction state - persists across interactions
        this.nodePredictions = new Map(); // Map<nodeId, {predicted: number, actual: number, isWrong: boolean}>
        
        // Visual settings
        this.nodeRadius = 20;
        this.nodeColor = '#666666';
        this.selectedNodeColor = '#999999';
        this.targetNodeColor = '#666666';
        this.targetNodeBorderColor = '#1e90ff';  // Neon dark blue
        this.targetNodeBorderWidth = 3;  // Border width
        this.edgeColor = '#555555';
        this.textColor = '#ffffff';
        this.backgroundColor = '#1a1a1a';
        this.errorDotColor = '#ff0000';
        
        // Mouse state
        this.mouseX = 0;
        this.mouseY = 0;
        
        this.initCanvas();
        this.setupEventListeners();
        this.render();
    }
    
    initCanvas() {
        // Set canvas size to fill container
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }
    
    resizeCanvas() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
        this.render();
    }
    
    setupEventListeners() {
        // Mouse move - for dragging and edge preview
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        
        // Left click - select node or start dragging
        this.canvas.addEventListener('mousedown', (e) => {
            if (e.button === 0) { // Left click
                this.handleLeftClick(e);
            }
        });
        
        this.canvas.addEventListener('mouseup', (e) => {
            if (e.button === 0) {
                this.handleLeftRelease(e);
            } else if (e.button === 2) {
                this.handleRightRelease(e);
            }
        });
        
        // Right click - start edge drawing or pan canvas
        this.canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.handleRightClick(e);
        });
        
        // Double click - set target node or create new node
        this.canvas.addEventListener('dblclick', (e) => this.handleDoubleClick(e));
        
        // Keyboard - delete selected node
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Backspace' && this.selectedNode !== null) {
                e.preventDefault();
                this.deleteNode(this.selectedNode);
            }
        });
    }
    
    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left - this.offsetX,
            y: e.clientY - rect.top - this.offsetY
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
            
            this.render();
            return;
        }
        
        const pos = this.getMousePos(e);
        this.mouseX = pos.x;
        this.mouseY = pos.y;
        
        // Drag node
        if (this.draggingNode !== null) {
            const node = this.nodes[this.draggingNode];
            node.x = pos.x;
            node.y = pos.y;
            this.render();
        }
        
        // Preview edge while drawing
        if (this.edgeStart !== null) {
            this.render();
        }
    }
    
    handleLeftClick(e) {
        const pos = this.getMousePos(e);
        const nodeIndex = this.findNodeAt(pos.x, pos.y);
        
        if (nodeIndex !== -1) {
            // If drawing an edge, complete it
            if (this.edgeStart !== null) {
                this.addEdge(this.edgeStart, nodeIndex);
                this.edgeStart = null;
                this.render();
                return;
            }
            
            // Otherwise, select node and start dragging
            this.selectedNode = nodeIndex;
            this.draggingNode = nodeIndex;
            this.render();
        } else {
            // Cancel edge drawing if clicking empty space
            if (this.edgeStart !== null) {
                this.edgeStart = null;
                this.render();
            }
            
            // Deselect if clicking empty space
            this.selectedNode = null;
            this.render();
        }
    }
    
    handleLeftRelease(e) {
        this.draggingNode = null;
    }
    
    handleRightClick(e) {
        const pos = this.getMousePos(e);
        const nodeIndex = this.findNodeAt(pos.x, pos.y);
        
        if (nodeIndex !== -1) {
            if (this.edgeStart === null) {
                // Start drawing edge
                this.edgeStart = nodeIndex;
            } else {
                // Complete edge
                this.addEdge(this.edgeStart, nodeIndex);
                this.edgeStart = null;
            }
            this.render();
        } else {
            // Start panning canvas on empty space
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
    
    handleDoubleClick(e) {
        const pos = this.getMousePos(e);
        const nodeIndex = this.findNodeAt(pos.x, pos.y);
        
        if (nodeIndex === -1) {
            // Add node on double-click empty space
            this.addNode(pos.x, pos.y);
            this.render();
        }
        // Double-clicking a node does nothing now
    }
    
    findNodeAt(x, y) {
        for (let i = this.nodes.length - 1; i >= 0; i--) {
            const node = this.nodes[i];
            const dx = x - node.x;
            const dy = y - node.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance <= this.nodeRadius) {
                return i;
            }
        }
        return -1;
    }
    
    addNode(x, y) {
        // Clear predictions since graph structure changed
        this.nodePredictions.clear();
        
        // Find the smallest available ID
        const usedIds = new Set(this.nodes.map(node => node.id));
        let newId = 0;
        while (usedIds.has(newId)) {
            newId++;
        }
        
        const node = {
            id: newId,
            x: x,
            y: y
        };
        this.nodes.push(node);
        
        // Update nextNodeId to be at least one more than the highest ID
        this.nextNodeId = Math.max(this.nextNodeId, newId + 1);
        
        this.updateGraphInput();
        this.render();
        this.triggerAnalyze();
    }
    
    deleteNode(nodeIndex) {
        if (nodeIndex < 0 || nodeIndex >= this.nodes.length) return;
        
        // Clear predictions since graph structure changed
        this.nodePredictions.clear();
        
        // Remove all edges connected to this node
        this.edges = this.edges.filter(edge => 
            edge.from !== nodeIndex && edge.to !== nodeIndex
        );
        
        // Remove the node
        this.nodes.splice(nodeIndex, 1);
        
        // Update edge indices (shift down indices after deleted node)
        this.edges = this.edges.map(edge => ({
            from: edge.from > nodeIndex ? edge.from - 1 : edge.from,
            to: edge.to > nodeIndex ? edge.to - 1 : edge.to
        }));
        
        // Update selection states
        if (this.selectedNode === nodeIndex) {
            this.selectedNode = null;
        } else if (this.selectedNode > nodeIndex) {
            this.selectedNode--;
        }
        
        if (this.targetNode === nodeIndex) {
            this.targetNode = null;
        } else if (this.targetNode > nodeIndex) {
            this.targetNode--;
        }
        
        this.updateGraphInput();
        this.render();
        this.triggerAnalyze();
    }
    
    addEdge(fromIndex, toIndex) {
        // Clear predictions since graph structure changed
        this.nodePredictions.clear();
        
        // Check if edge already exists
        const edgeIndex = this.edges.findIndex(edge => 
            (edge.from === fromIndex && edge.to === toIndex) ||
            (edge.from === toIndex && edge.to === fromIndex)
        );
        
        if (edgeIndex !== -1) {
            // Edge exists - remove it (toggle off)
            this.edges.splice(edgeIndex, 1);
        } else {
            // Edge doesn't exist - add it (toggle on)
            this.edges.push({ from: fromIndex, to: toIndex });
        }
        
        this.updateGraphInput();
        this.render();
        this.triggerAnalyze();
    }
    
    updateGraphInput() {
        const graphInput = document.getElementById('graphInput');
        if (!graphInput) return;
        
        // Find vertices that have edges
        const verticesWithEdges = new Set();
        this.edges.forEach(edge => {
            verticesWithEdges.add(edge.from);
            verticesWithEdges.add(edge.to);
        });
        
        // Build output with proper formatting
        const lines = [];
        
        // First, collect isolated vertices (sorted)
        const isolatedVertices = [];
        this.nodes.forEach((node, index) => {
            if (!verticesWithEdges.has(index)) {
                isolatedVertices.push(node.id);
            }
        });
        isolatedVertices.sort((a, b) => a - b);
        
        // Add isolated vertices to output
        isolatedVertices.forEach(id => {
            lines.push(`${id}`);
        });
        
        // Collect and sort edges
        const edgeList = [];
        this.edges.forEach(edge => {
            const fromNode = this.nodes[edge.from];
            const toNode = this.nodes[edge.to];
            const v1 = fromNode.id;
            const v2 = toNode.id;
            
            // Store with smaller vertex first for consistent formatting
            edgeList.push({
                v1: Math.min(v1, v2),
                v2: Math.max(v1, v2)
            });
        });
        
        // Sort edges: first by v1, then by v2
        edgeList.sort((a, b) => {
            if (a.v1 !== b.v1) return a.v1 - b.v1;
            return a.v2 - b.v2;
        });
        
        // Add edges to output
        edgeList.forEach(edge => {
            lines.push(`${edge.v1} ${edge.v2}`);
        });
        
        graphInput.value = lines.join('\n');
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
        
        this.edges.forEach(edge => {
            const fromNode = this.nodes[edge.from];
            const toNode = this.nodes[edge.to];
            
            if (fromNode && toNode) {
                if (edge.from === edge.to) {
                    // Self-loop
                    this.drawSelfLoop(fromNode);
                } else {
                    // Regular edge
                    this.ctx.beginPath();
                    this.ctx.moveTo(fromNode.x, fromNode.y);
                    this.ctx.lineTo(toNode.x, toNode.y);
                    this.ctx.stroke();
                }
            }
        });
        
        // Draw edge preview when drawing
        if (this.edgeStart !== null) {
            const startNode = this.nodes[this.edgeStart];
            this.ctx.strokeStyle = '#888888';
            this.ctx.setLineDash([5, 5]);
            this.ctx.beginPath();
            this.ctx.moveTo(startNode.x, startNode.y);
            this.ctx.lineTo(this.mouseX, this.mouseY);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
        }
        
        // Draw nodes
        this.nodes.forEach((node, index) => {
            // Determine node color
            let fillColor = this.nodeColor;
            
            if (index === this.selectedNode) {
                fillColor = this.selectedNodeColor;
            }
            
            // Draw node circle
            this.ctx.fillStyle = fillColor;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, this.nodeRadius, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Draw node ID
            this.ctx.fillStyle = this.textColor;
            this.ctx.font = 'bold 14px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(node.id.toString(), node.x, node.y);
            
            // Draw prediction info if prediction was wrong
            const predictionInfo = this.nodePredictions.get(node.id);
            if (predictionInfo && predictionInfo.isWrong) {
                const offsetX = node.x + this.nodeRadius * 0.8;
                const offsetY = node.y + this.nodeRadius * 0.8;
                
                // Set font for prediction display
                this.ctx.font = 'bold 14px Arial';
                this.ctx.textAlign = 'left';
                this.ctx.textBaseline = 'middle';
                
                // Draw correct value in green
                this.ctx.fillStyle = '#00ff00';
                const correctText = predictionInfo.actual.toString();
                this.ctx.fillText(correctText, offsetX, offsetY);
                
                // Measure width to position slash and predicted value
                const correctWidth = this.ctx.measureText(correctText).width;
                
                // Draw slash in white
                this.ctx.fillStyle = this.textColor;
                const slashX = offsetX + correctWidth;
                this.ctx.fillText('/', slashX, offsetY);
                const slashWidth = this.ctx.measureText('/').width;
                
                // Draw predicted value in red
                this.ctx.fillStyle = this.errorDotColor;
                this.ctx.fillText(predictionInfo.predicted.toString(), slashX + slashWidth, offsetY);
            }
        });
        
        // Restore context after applying offset
        this.ctx.restore();
    }
    
    drawSelfLoop(node) {
        // Draw a small loop above the node
        const loopRadius = this.nodeRadius * 0.8;
        const loopCenterY = node.y - this.nodeRadius - loopRadius;
        
        this.ctx.beginPath();
        this.ctx.arc(node.x, loopCenterY, loopRadius, 0, Math.PI * 2);
        this.ctx.stroke();
    }
    
    // Load graph from edge list
    loadFromEdgeList(edgeListText, targetVertex = null) {
        this.nodes = [];
        this.edges = [];
        this.nextNodeId = 0;
        this.selectedNode = null;
        this.targetNode = null;
        this.nodePredictions.clear();
        
        if (!edgeListText || edgeListText.trim() === '') {
            this.render();
            return;
        }
        
        const lines = edgeListText.trim().split('\n');
        const nodeSet = new Set();
        const edgeList = [];
        
        // Parse edges and isolated vertices
        lines.forEach(line => {
            const parts = line.trim().split(/\s+/);
            
            if (parts.length === 1) {
                // Single number - isolated vertex
                const vertex = parseInt(parts[0]);
                if (!isNaN(vertex)) {
                    nodeSet.add(vertex);
                }
            } else if (parts.length >= 2) {
                // Two numbers - edge
                const from = parseInt(parts[0]);
                const to = parseInt(parts[1]);
                
                if (!isNaN(from) && !isNaN(to)) {
                    nodeSet.add(from);
                    nodeSet.add(to);
                    edgeList.push({ from, to });
                }
            }
        });
        
        // Create nodes in a circle layout
        const nodeIds = Array.from(nodeSet).sort((a, b) => a - b);
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const radius = Math.min(centerX, centerY) * 0.6;
        
        // Create a mapping from original IDs to array indices
        const idToIndex = new Map();
        
        nodeIds.forEach((id, index) => {
            const angle = (index / nodeIds.length) * Math.PI * 2 - Math.PI / 2;
            const x = centerX + Math.cos(angle) * radius;
            const y = centerY + Math.sin(angle) * radius;
            
            this.nodes.push({
                id: id,
                x: x,
                y: y
            });
            
            idToIndex.set(id, index);
        });
        
        this.nextNodeId = Math.max(...nodeIds) + 1;
        
        // Create edges using array indices
        edgeList.forEach(({ from, to }) => {
            const fromIndex = idToIndex.get(from);
            const toIndex = idToIndex.get(to);
            
            if (fromIndex !== undefined && toIndex !== undefined) {
                this.edges.push({ from: fromIndex, to: toIndex });
            }
        });
        
        // Set target vertex
        if (targetVertex !== null) {
            const targetIndex = idToIndex.get(targetVertex);
            if (targetIndex !== undefined) {
                this.targetNode = targetIndex;
            }
        }
        
        this.render();
    }
    
    clear() {
        this.nodes = [];
        this.edges = [];
        this.nextNodeId = 0;
        this.selectedNode = null;
        this.targetNode = null;
        this.edgeStart = null;
        this.nodePredictions.clear();
        this.updateGraphInput();
        this.render();
    }
    
    updatePredictions(predictions) {
        // predictions should be an array of {nodeId, predicted, actual}
        this.nodePredictions.clear();
        
        predictions.forEach(pred => {
            this.nodePredictions.set(pred.nodeId, {
                predicted: pred.predicted,
                actual: pred.actual,
                isWrong: pred.predicted !== pred.actual
            });
        });
        
        this.render();
    }
    
    triggerAnalyze() {
        // Trigger analyze only if predictions haven't been calculated yet
        if (this.nodes.length > 0 && this.nodePredictions.size === 0) {
            // Use setTimeout to avoid blocking the UI
            setTimeout(() => {
                if (window.analyzeGraph) {
                    window.analyzeGraph();
                }
            }, 100);
        }
    }
}

// Global instance
window.interactiveGraph = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.interactiveGraph = new InteractiveGraph('graphCanvas');
});

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
        
        // Visual settings
        this.nodeRadius = 20;
        this.nodeColor = '#666666';
        this.selectedNodeColor = '#999999';
        this.targetNodeColor = '#666666';
        this.targetNodeBorderColor = '#00ff00';
        this.edgeColor = '#555555';
        this.textColor = '#ffffff';
        this.backgroundColor = '#1a1a1a';
        
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
            }
        });
        
        // Right click - start edge drawing
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
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }
    
    handleMouseMove(e) {
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
            // Select node and start dragging
            this.selectedNode = nodeIndex;
            this.draggingNode = nodeIndex;
            this.render();
        } else {
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
            // Cancel edge drawing if clicking empty space
            this.edgeStart = null;
            this.render();
        }
    }
    
    handleDoubleClick(e) {
        const pos = this.getMousePos(e);
        const nodeIndex = this.findNodeAt(pos.x, pos.y);
        
        if (nodeIndex !== -1) {
            // Set as target node
            this.targetNode = nodeIndex;
            this.updateTargetVertexInput();
            this.render();
        } else {
            // Create new node
            this.addNode(pos.x, pos.y);
        }
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
        const node = {
            id: this.nextNodeId++,
            x: x,
            y: y
        };
        this.nodes.push(node);
        this.updateGraphInput();
        this.render();
    }
    
    deleteNode(nodeIndex) {
        if (nodeIndex < 0 || nodeIndex >= this.nodes.length) return;
        
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
        this.updateTargetVertexInput();
        this.render();
    }
    
    addEdge(fromIndex, toIndex) {
        // Check if edge already exists
        const exists = this.edges.some(edge => 
            (edge.from === fromIndex && edge.to === toIndex) ||
            (edge.from === toIndex && edge.to === fromIndex)
        );
        
        if (!exists) {
            this.edges.push({ from: fromIndex, to: toIndex });
            this.updateGraphInput();
            this.render();
        }
    }
    
    updateGraphInput() {
        const graphInput = document.getElementById('graphInput');
        if (!graphInput) return;
        
        // Convert edges to edge list format
        const edgeList = this.edges.map(edge => `${edge.from} ${edge.to}`).join('\n');
        graphInput.value = edgeList;
    }
    
    updateTargetVertexInput() {
        const vertexInput = document.getElementById('vertexInput');
        if (!vertexInput) return;
        
        if (this.targetNode !== null) {
            vertexInput.value = this.targetNode;
        }
    }
    
    render() {
        // Clear canvas
        this.ctx.fillStyle = this.backgroundColor;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
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
            let hasBorder = false;
            
            if (index === this.selectedNode) {
                fillColor = this.selectedNodeColor;
            }
            
            if (index === this.targetNode) {
                hasBorder = true;
            }
            
            // Draw node circle
            this.ctx.fillStyle = fillColor;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, this.nodeRadius, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Draw border for target node
            if (hasBorder) {
                this.ctx.strokeStyle = this.targetNodeBorderColor;
                this.ctx.lineWidth = 3;
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, this.nodeRadius, 0, Math.PI * 2);
                this.ctx.stroke();
            }
            
            // Draw node ID
            this.ctx.fillStyle = this.textColor;
            this.ctx.font = 'bold 14px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(index.toString(), node.x, node.y);
        });
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
        
        if (!edgeListText || edgeListText.trim() === '') {
            this.render();
            return;
        }
        
        const lines = edgeListText.trim().split('\n');
        const nodeSet = new Set();
        const edgeList = [];
        
        // Parse edges and collect nodes
        lines.forEach(line => {
            const parts = line.trim().split(/\s+/);
            if (parts.length >= 2) {
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
        
        nodeIds.forEach((id, index) => {
            const angle = (index / nodeIds.length) * Math.PI * 2 - Math.PI / 2;
            const x = centerX + Math.cos(angle) * radius;
            const y = centerY + Math.sin(angle) * radius;
            
            this.nodes[id] = {
                id: id,
                x: x,
                y: y
            };
        });
        
        this.nextNodeId = Math.max(...nodeIds) + 1;
        
        // Create edges
        edgeList.forEach(({ from, to }) => {
            this.edges.push({ from, to });
        });
        
        // Set target vertex
        if (targetVertex !== null && this.nodes[targetVertex]) {
            this.targetNode = targetVertex;
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
        this.updateGraphInput();
        this.updateTargetVertexInput();
        this.render();
    }
}

// Global instance
window.interactiveGraph = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.interactiveGraph = new InteractiveGraph('graphCanvas');
});

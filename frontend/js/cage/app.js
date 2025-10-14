/**
 * Cage Graph Generat        }

        const data = await response.json();
        
        // Update the textarea with the graph
        const graphInput = document.getElementById('graphInput');
        if (graphInput) {
            graphInput.value = data.graph;
        }
        
        // Load into interactive graph
        if (window.interactiveGraph) {
            window.interactiveGraph.loadFromEdgeList(data.graph, null);
        }
        
    } catch (error) {
        console.error('Error generating graph:', error);es RL-based approach to generate and validate cage graphs
 */

// Override API_BASE_URL for cage endpoints
API_BASE_URL = "http://localhost:5555/api/cage";

/**
 * Generate a random graph (placeholder for RL generation)
 */
window.generateRandomGraph = async function() {
    try {
        const settings = getGraphSettings();
        
        const response = await fetch(`${API_BASE_URL}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(settings),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Update the textarea with the graph
        const graphInput = document.getElementById('graphInput');
        if (graphInput) {
            graphInput.value = data.graph;
        }
        
        // Load into interactive graph
        if (window.interactiveGraph) {
            window.interactiveGraph.loadFromEdgeList(data.graph, null);
        }
        
    } catch (error) {
        console.error('[Cage] Error generating graph:', error);
        alert('Failed to generate graph. Please try again.');
    }
}

/**
 * Analyze if the current graph is a cage
 */
window.analyzeGraph = async function() {
    try {
        const graphInput = document.getElementById('graphInput');
        const edgeList = graphInput.value.trim();
        
        if (!edgeList) {
            return;
        }

        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                edge_list: edgeList
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Result is in data.is_cage (true/false)
        
    } catch (error) {
        console.error('Error analyzing graph:', error);
    }
}

/**
 * Get graph generation settings from UI
 */
function getGraphSettings() {
    const minNodes = parseInt(document.getElementById('minNodes')?.value) || 5;
    const maxNodes = parseInt(document.getElementById('maxNodes')?.value) || 12;
    const minProb = parseFloat(document.getElementById('minProb')?.value) / 100 || 0.15;
    const maxProb = parseFloat(document.getElementById('maxProb')?.value) / 100 || 0.60;
    const allowSelfLoops = document.getElementById('allowSelfLoops')?.checked ?? true;

    return {
        min_nodes: minNodes,
        max_nodes: maxNodes,
        min_probability: minProb,
        max_probability: maxProb,
        allow_self_loops: allowSelfLoops
    };
}

/**
 * Clear the canvas
 */
window.clearCanvas = function() {
    if (window.interactiveGraph) {
        window.interactiveGraph.clear();
    }
}

/**
 * Load graph from edge list
 */
function loadGraphFromEdgeList(edgeList) {
    if (window.interactiveGraph) {
        window.interactiveGraph.loadFromEdgeList(edgeList);
    }
}

// Settings modal functions (same as degree predictor)
window.openSettings = function() {
    document.getElementById('settingsModal').style.display = 'flex';
}

window.closeSettings = function() {
    document.getElementById('settingsModal').style.display = 'none';
}

window.saveSettings = function() {
    updateRangeDisplays();
    closeSettings();
}

function updateRangeDisplays() {
    // Update node range display
    const minNodes = document.getElementById('minNodes').value;
    const maxNodes = document.getElementById('maxNodes').value;
    document.getElementById('nodeRangeDisplay').textContent = `${minNodes} - ${maxNodes}`;
    
    // Update probability range display
    const minProb = document.getElementById('minProb').value;
    const maxProb = document.getElementById('maxProb').value;
    document.getElementById('probRangeDisplay').textContent = 
        `${(minProb / 100).toFixed(2)} - ${(maxProb / 100).toFixed(2)}`;
    
    updateRangeHighlights();
}

function updateRangeHighlights() {
    // Update node range highlight
    const minNodes = document.getElementById('minNodes');
    const maxNodes = document.getElementById('maxNodes');
    const nodeMin = parseInt(minNodes.min);
    const nodeMax = parseInt(minNodes.max);
    const nodeLeft = ((parseInt(minNodes.value) - nodeMin) / (nodeMax - nodeMin)) * 100;
    const nodeRight = ((parseInt(maxNodes.value) - nodeMin) / (nodeMax - nodeMin)) * 100;
    document.getElementById('nodeRangeHighlight').style.left = nodeLeft + '%';
    document.getElementById('nodeRangeHighlight').style.width = (nodeRight - nodeLeft) + '%';
    
    // Update probability range highlight
    const minProb = document.getElementById('minProb');
    const maxProb = document.getElementById('maxProb');
    const probMin = parseInt(minProb.min);
    const probMax = parseInt(minProb.max);
    const probLeft = ((parseInt(minProb.value) - probMin) / (probMax - probMin)) * 100;
    const probRight = ((parseInt(maxProb.value) - probMin) / (probMax - probMin)) * 100;
    document.getElementById('probRangeHighlight').style.left = probLeft + '%';
    document.getElementById('probRangeHighlight').style.width = (probRight - probLeft) + '%';
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    updateRangeDisplays();
    
    // Add event listeners for range sliders
    const sliders = document.querySelectorAll('.range-slider');
    sliders.forEach(slider => {
        slider.addEventListener('input', updateRangeDisplays);
    });
});

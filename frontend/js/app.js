/**
 * Main application logic for GNN Vertex Degree Predictor
 */

/**
 * Clear the canvas
 */
function clearCanvas() {
  if (window.interactiveGraph) {
    window.interactiveGraph.clear();
  }
}

/**
 * Generate a random graph and populate input fields
 */
async function generateRandomGraph() {
  const generateBtn = document.getElementById("generateBtn");
  const graphInput = document.getElementById("graphInput");

  // Disable button while generating
  generateBtn.disabled = true;
  generateBtn.textContent = "Generating...";

  try {
    const response = await fetch(`${API_BASE_URL}/generate`, {
      method: "GET",
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Failed to generate graph");
    }

    // Populate the input field
    graphInput.value = data.graph;

    // Load into interactive graph
    if (window.interactiveGraph) {
      window.interactiveGraph.loadFromEdgeList(data.graph, null);
    }

    // Automatically analyze the generated graph
    await analyzeGraph();
  } catch (error) {
    showError("Error generating graph: " + error.message);
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "Random Graph";
  }
}

/**
 * Format and clean the graph input
 * Removes isolated vertices that have edges, sorts properly
 */
function formatGraphInput(graphStr) {
  if (!graphStr || graphStr.trim() === "") return "";

  const lines = graphStr.trim().split("\n");
  const isolatedVertices = new Set();
  const edges = [];

  // Parse all lines
  lines.forEach((line) => {
    const parts = line.trim().split(/\s+/);
    if (parts.length === 1 && parts[0] !== "") {
      const v = parseInt(parts[0]);
      if (!isNaN(v)) {
        isolatedVertices.add(v);
      }
    } else if (parts.length >= 2) {
      const v1 = parseInt(parts[0]);
      const v2 = parseInt(parts[1]);
      if (!isNaN(v1) && !isNaN(v2)) {
        edges.push({ v1, v2 });
      }
    }
  });

  // Remove isolated vertices that actually have edges
  edges.forEach((edge) => {
    isolatedVertices.delete(edge.v1);
    isolatedVertices.delete(edge.v2);
  });

  // Build formatted output
  const output = [];

  // Add isolated vertices first (sorted)
  const sortedIsolated = Array.from(isolatedVertices).sort((a, b) => a - b);
  sortedIsolated.forEach((v) => {
    output.push(`${v}`);
  });

  // Add edges (sorted, with smaller vertex first)
  const formattedEdges = edges.map((edge) => ({
    v1: Math.min(edge.v1, edge.v2),
    v2: Math.max(edge.v1, edge.v2),
  }));

  formattedEdges.sort((a, b) => {
    if (a.v1 !== b.v1) return a.v1 - b.v1;
    return a.v2 - b.v2;
  });

  // Remove duplicate edges (keep only unique edges)
  const uniqueEdges = [];
  const edgeSet = new Set();
  
  formattedEdges.forEach((edge) => {
    const edgeKey = `${edge.v1},${edge.v2}`;
    if (!edgeSet.has(edgeKey)) {
      edgeSet.add(edgeKey);
      uniqueEdges.push(edge);
    }
  });

  uniqueEdges.forEach((edge) => {
    output.push(`${edge.v1} ${edge.v2}`);
  });

  return output.join("\n");
}

/**
 * Analyze the graph and display results
 */
async function analyzeGraph() {
  const graphInput = document.getElementById("graphInput");
  const errorMessage = document.getElementById("errorMessage");
  const loading = document.getElementById("loading");
  const analyzeBtn = document.getElementById("analyzeBtn");

  // Format and clean the graph input first
  const formattedGraph = formatGraphInput(graphInput.value);
  graphInput.value = formattedGraph;

  // Reset UI
  hideError();

  // Check if we already have predictions (graph hasn't changed)
  if (
    window.interactiveGraph &&
    window.interactiveGraph.nodePredictions.size > 0
  ) {
    // Already analyzed, no need to re-analyze
    return;
  }

  // Show loading
  showLoading();
  analyzeBtn.disabled = true;

  try {
    // Analyze all nodes to get predictions
    await analyzeAllNodes(formattedGraph);
  } catch (error) {
    showError("Error: " + error.message);
  } finally {
    hideLoading();
    analyzeBtn.disabled = false;
  }
}

/**
 * Analyze all nodes in the graph and update predictions
 */
async function analyzeAllNodes(formattedGraph) {
  if (!window.interactiveGraph || window.interactiveGraph.nodes.length === 0) {
    return;
  }

  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        graph: formattedGraph,
      }),
    });

    const data = await response.json();

    if (response.ok && data.predictions) {
      // Transform the predictions to match the expected format
      const predictions = data.predictions.map((pred) => ({
        nodeId: pred.node_id,
        predicted: pred.predicted_degree,
        actual: pred.true_degree,
      }));

      // Update the interactive graph with predictions
      window.interactiveGraph.updatePredictions(predictions);
    }
  } catch (error) {
    console.error("Error analyzing graph:", error);
  }
}

/**
 * Show error message
 */
function showError(message) {
  const errorMessage = document.getElementById("errorMessage");
  errorMessage.textContent = message;
  errorMessage.classList.add("show");
}

/**
 * Hide error message
 */
function hideError() {
  const errorMessage = document.getElementById("errorMessage");
  errorMessage.classList.remove("show");
  errorMessage.textContent = "";
}

/**
 * Show loading spinner
 */
function showLoading() {
  const loading = document.getElementById("loading");
  loading.classList.add("show");
}

/**
 * Hide loading spinner
 */
function hideLoading() {
  const loading = document.getElementById("loading");
  loading.classList.remove("show");
}

/**
 * Initialize event listeners
 */
function initializeEventListeners() {
  // Sync graph input with canvas when manually edited
  document.getElementById("graphInput").addEventListener("input", function (e) {
    if (window.interactiveGraph) {
      window.interactiveGraph.loadFromEdgeList(e.target.value, null);
    }
  });
}

/**
 * Toggle the visibility of the controls help panel
 */
function toggleControls() {
  const canvasHelp = document.getElementById("canvasHelp");
  const toggleBtn = document.getElementById("toggleControlsBtn");

  if (canvasHelp.classList.contains("hidden")) {
    canvasHelp.classList.remove("hidden");
    toggleBtn.textContent = "Hide Controls";
    localStorage.setItem("controlsVisible", "true");
  } else {
    canvasHelp.classList.add("hidden");
    toggleBtn.textContent = "Show Controls";
    localStorage.setItem("controlsVisible", "false");
  }
}

/**
 * Restore controls visibility from localStorage
 */
function restoreControlsState() {
  const canvasHelp = document.getElementById("canvasHelp");
  const toggleBtn = document.getElementById("toggleControlsBtn");
  const controlsVisible = localStorage.getItem("controlsVisible");

  if (controlsVisible === "false") {
    canvasHelp.classList.add("hidden");
    toggleBtn.textContent = "Show Controls";
  }
}

// Initialize when DOM is loaded
document.addEventListener("DOMContentLoaded", initializeEventListeners);
document.addEventListener("DOMContentLoaded", restoreControlsState);

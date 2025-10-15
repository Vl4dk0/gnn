/**
 * Main application logic for GNN Cage Generator
 */

// Override API_BASE_URL for cage endpoints
API_BASE_URL = "http://localhost:5555/api/cage";

/**
 * Clear the canvas
 */
function clearCanvas() {
  if (window.interactiveGraph) {
    window.interactiveGraph.clear();
  }
}

/**
 * Format and clean the graph input
 * Reuse same rules as degree predictor
 */
function formatGraphInput(graphStr) {
  if (!graphStr || graphStr.trim() === "") return "";

  const lines = graphStr.trim().split("\n");
  const isolatedVertices = new Set();
  const edges = [];

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

  edges.forEach((edge) => {
    isolatedVertices.delete(edge.v1);
    isolatedVertices.delete(edge.v2);
  });

  const output = [];

  const sortedIsolated = Array.from(isolatedVertices).sort((a, b) => a - b);
  sortedIsolated.forEach((v) => {
    output.push(`${v}`);
  });

  const formattedEdges = edges.map((edge) => ({
    v1: Math.min(edge.v1, edge.v2),
    v2: Math.max(edge.v1, edge.v2),
  }));

  formattedEdges.sort((a, b) => {
    if (a.v1 !== b.v1) return a.v1 - b.v1;
    return a.v2 - b.v2;
  });

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
 * Generate a random graph and populate input fields (placeholder backend logic)
 */
async function generateRandomGraph() {
  const generateBtn = document.getElementById("generateBtn");
  const graphInput = document.getElementById("graphInput");

  generateBtn.disabled = true;
  generateBtn.textContent = "Generating...";

  try {
    const settings = loadSettings();

    // Map camelCase settings to snake_case expected by cage backend
    const payload = {
      min_nodes: settings.minNodes,
      max_nodes: settings.maxNodes,
      min_probability: settings.minProb,
      max_probability: settings.maxProb,
      allow_self_loops: settings.allowSelfLoops,
    };

    const response = await fetch(`${API_BASE_URL}/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Failed to generate graph");
    }

    // Format to match degree page ordering (isolated vertices first, sorted edges)
    const formatted = formatGraphInput(data.graph);
    graphInput.value = formatted;

    if (window.interactiveGraph) {
      window.interactiveGraph.loadFromEdgeList(formatted, null);
    }
  } catch (error) {
    showError("Error generating graph: " + error.message);
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "Generate";
  }
}

/**
 * Analyze if the current graph is a cage (placeholder always true backend)
 */
async function analyzeGraph() {
  const graphInput = document.getElementById("graphInput");
  const analyzeBtn = document.getElementById("analyzeBtn");

  const formattedGraph = formatGraphInput(graphInput.value);
  if (graphInput.value !== formattedGraph) {
    graphInput.value = formattedGraph;
  }

  hideError();

  showLoading();
  analyzeBtn.disabled = true;

  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ edge_list: formattedGraph }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Failed to analyze graph");
    }

    // Placeholder result in data.is_cage; no UI output yet
    console.log("Cage analysis:", data);
  } catch (error) {
    showError("Error: " + error.message);
  } finally {
    hideLoading();
    analyzeBtn.disabled = false;
  }
}

/**
 * Show/Hide error or loading (no-op visual, for parity)
 */
function showError(message) {
  console.error(message);
}

function hideError() {}

function showLoading() {}

function hideLoading() {}

/**
 * Initialize event listeners
 */
function initializeEventListeners() {
  document.getElementById("graphInput").addEventListener("input", function (e) {
    if (window.interactiveGraph) {
      window.interactiveGraph.loadFromEdgeList(e.target.value, null);
    }
  });
}

/**
 * Settings Management (aligned with degree)
 */
const DEFAULT_SETTINGS = {
  minNodes: 5,
  maxNodes: 12,
  minProb: 0.15,
  maxProb: 0.6,
  allowSelfLoops: true,
};

function loadSettings() {
  const saved = localStorage.getItem("graphSettings");
  return saved ? JSON.parse(saved) : DEFAULT_SETTINGS;
}

function saveSettingsToStorage(settings) {
  localStorage.setItem("graphSettings", JSON.stringify(settings));
}

function openSettings() {
  const modal = document.getElementById("settingsModal");
  const settings = loadSettings();

  document.getElementById("minNodes").value = settings.minNodes;
  document.getElementById("maxNodes").value = settings.maxNodes;
  document.getElementById("minProb").value = Math.round(settings.minProb * 100);
  document.getElementById("maxProb").value = Math.round(settings.maxProb * 100);
  document.getElementById("allowSelfLoops").checked = settings.allowSelfLoops;

  updateNodeRangeDisplay();
  updateProbRangeDisplay();

  modal.classList.add("show");
}

function closeSettings() {
  const modal = document.getElementById("settingsModal");
  modal.classList.remove("show");
}

function saveSettings() {
  const minNodes = parseInt(document.getElementById("minNodes").value);
  const maxNodes = parseInt(document.getElementById("maxNodes").value);
  const minProb = parseInt(document.getElementById("minProb").value) / 100;
  const maxProb = parseInt(document.getElementById("maxProb").value) / 100;
  const allowSelfLoops = document.getElementById("allowSelfLoops").checked;

  if (minNodes > maxNodes) {
    alert("Minimum nodes cannot be greater than maximum nodes");
    return;
  }

  if (minProb > maxProb) {
    alert("Minimum probability cannot be greater than maximum probability");
    return;
  }

  const settings = {
    minNodes,
    maxNodes,
    minProb,
    maxProb,
    allowSelfLoops,
  };

  saveSettingsToStorage(settings);
  closeSettings();
}

function updateNodeRangeDisplay() {
  const minSlider = document.getElementById("minNodes");
  const maxSlider = document.getElementById("maxNodes");

  if (parseInt(minSlider.value) > parseInt(maxSlider.value)) {
    minSlider.value = maxSlider.value;
  }

  document.getElementById("nodeRangeDisplay").textContent = `${minSlider.value} - ${maxSlider.value}`;
  updateRangeHighlight("minNodes", "maxNodes", "nodeRangeHighlight");
}

function updateNodeRangeDisplayMax() {
  const minSlider = document.getElementById("minNodes");
  const maxSlider = document.getElementById("maxNodes");

  if (parseInt(maxSlider.value) < parseInt(minSlider.value)) {
    maxSlider.value = minSlider.value;
  }

  document.getElementById("nodeRangeDisplay").textContent = `${minSlider.value} - ${maxSlider.value}`;
  updateRangeHighlight("minNodes", "maxNodes", "nodeRangeHighlight");
}

function updateProbRangeDisplay() {
  const minSlider = document.getElementById("minProb");
  const maxSlider = document.getElementById("maxProb");

  if (parseInt(minSlider.value) > parseInt(maxSlider.value)) {
    minSlider.value = maxSlider.value;
  }

  document.getElementById("probRangeDisplay").textContent = `${(minSlider.value / 100).toFixed(2)} - ${(maxSlider.value / 100).toFixed(2)}`;
  updateRangeHighlight("minProb", "maxProb", "probRangeHighlight");
}

function updateProbRangeDisplayMax() {
  const minSlider = document.getElementById("minProb");
  const maxSlider = document.getElementById("maxProb");

  if (parseInt(maxSlider.value) < parseInt(minSlider.value)) {
    maxSlider.value = minSlider.value;
  }

  document.getElementById("probRangeDisplay").textContent = `${(minSlider.value / 100).toFixed(2)} - ${(maxSlider.value / 100).toFixed(2)}`;
  updateRangeHighlight("minProb", "maxProb", "probRangeHighlight");
}

function updateRangeHighlight(minId, maxId, highlightId) {
  const minSlider = document.getElementById(minId);
  const maxSlider = document.getElementById(maxId);
  const highlight = document.getElementById(highlightId);

  if (!highlight) return;

  const min = parseFloat(minSlider.min);
  const max = parseFloat(minSlider.max);
  const minVal = parseFloat(minSlider.value);
  const maxVal = parseFloat(maxSlider.value);

  const minPercent = ((minVal - min) / (max - min)) * 100;
  const maxPercent = ((maxVal - min) / (max - min)) * 100;

  highlight.style.left = minPercent + "%";
  highlight.style.width = maxPercent - minPercent + "%";
}

function initializeSettings() {
  document.getElementById("minNodes").addEventListener("input", updateNodeRangeDisplay);
  document.getElementById("maxNodes").addEventListener("input", updateNodeRangeDisplayMax);
  document.getElementById("minProb").addEventListener("input", updateProbRangeDisplay);
  document.getElementById("maxProb").addEventListener("input", updateProbRangeDisplayMax);

  document.getElementById("settingsModal").addEventListener("click", function (e) {
    if (e.target === this) {
      closeSettings();
    }
  });
}

// Initialize when DOM is loaded
document.addEventListener("DOMContentLoaded", initializeEventListeners);
document.addEventListener("DOMContentLoaded", initializeSettings);

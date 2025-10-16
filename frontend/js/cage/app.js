/**
 * Main application logic for GNN Cage Generator (MCTS + GNN)
 */

// Override API_BASE_URL for cage endpoints
API_BASE_URL = "http://localhost:5555/api/cage";

// Global state
let currentSessionId = null;
let pollingInterval = null;

/**
 * Clear the canvas
 */
function clearCanvas() {
  if (window.interactiveGraph) {
    window.interactiveGraph.clear();
  }
}

/**
 * Start MCTS-based cage generation
 */
async function startGeneration() {
  const k = parseInt(document.getElementById("degreeK").value);
  const g = parseInt(document.getElementById("girthG").value);
  const generateBtn = document.getElementById("generateBtn");
  const stopBtn = document.getElementById("stopBtn");
  const statusDisplay = document.getElementById("statusDisplay");

  if (k < 2 || g < 3) {
    alert("k must be >= 2 and g must be >= 3");
    return;
  }

  // Load settings to get selected generator
  const settings = loadSettings();
  const generator = settings.generatorType || 'constructive';

  // Disable generate button
  generateBtn.disabled = true;
  generateBtn.textContent = "Generating...";
  stopBtn.disabled = false;
  stopBtn.style.display = "block";

  try {
    // Start generation on backend with selected generator
    const response = await fetch(`${API_BASE_URL}/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ k, g, generator }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Failed to start generation");
    }

    currentSessionId = data.session_id;

    // Start polling for status (will update display on first poll)
    startPolling();
  } catch (error) {
    showError("Error starting generation: " + error.message);
    resetButtons();
  }
}

/**
 * Stop generation
 */
async function stopGeneration() {
  if (!currentSessionId) return;

  try {
    // Fetch current status before stopping
    const statusResponse = await fetch(`${API_BASE_URL}/status/${currentSessionId}`);
    const status = await statusResponse.json();
    
    // Stop the session
    await fetch(`${API_BASE_URL}/stop/${currentSessionId}`, {
      method: "POST",
    });

    stopPolling();
    resetButtons();
    
    // Update status display with "stopped" message
    const statusDisplay = document.getElementById("statusDisplay");
    let html = `
        <div style="margin-bottom: 8px;">
            <strong>Target:</strong> (${status.k},${status.g})-cage
        </div>
        <div style="margin-bottom: 8px;">
            <strong>Step:</strong> ${status.step_count}
        </div>
        <div style="margin-bottom: 8px;">
            <strong>Nodes:</strong> ${status.num_nodes} / Moore bound: ${status.moore_bound}
        </div>
        <div style="margin-bottom: 8px;">
            <strong>Edges:</strong> ${status.num_edges}
        </div>
        <div style="margin-bottom: 8px;">
            <strong>k-regular:</strong> ${status.is_k_regular ? "✓" : "✗"}
        </div>
        <div style="margin-bottom: 8px;">
            <strong>Girth:</strong> ${status.girth || "∞"} (target: ${status.g})
        </div>
        <div style="margin-top: 12px; color: #888;">
            ⏹ Stopped by user (${status.elapsed_time.toFixed(1)}s)
        </div>
    `;
    statusDisplay.innerHTML = html;
  } catch (error) {
    console.error("Error stopping generation:", error);
  }
}

/**
 * Start polling for status updates
 */
function startPolling() {
  stopPolling(); // Clear any existing polling

  // Load settings to get polling interval
  const settings = loadSettings();
  const interval = settings.pollingInterval || 300;

  pollingInterval = setInterval(async () => {
    if (!currentSessionId) {
      stopPolling();
      return;
    }

    try {
      const response = await fetch(
        `${API_BASE_URL}/status/${currentSessionId}`
      );
      const status = await response.json();

      if (!response.ok) {
        throw new Error(status.error || "Failed to get status");
      }

      updateStatus(status);

      // Stop polling if generation is complete
      if (status.is_complete) {
        stopPolling();
        resetButtons();

        if (status.success) {
          // Format time with appropriate unit
          let timeStr;
          if (status.elapsed_time < 1) {
            timeStr = `${(status.elapsed_time * 1000).toFixed(0)}ms`;
          } else if (status.elapsed_time < 60) {
            timeStr = `${status.elapsed_time.toFixed(1)}s`;
          } else {
            const mins = Math.floor(status.elapsed_time / 60);
            const secs = (status.elapsed_time % 60).toFixed(1);
            timeStr = `${mins}m ${secs}s`;
          }
          
          showSuccess(`Valid cage! (${timeStr})`);
        } else {
          showError(
            `Generation completed but cage is not valid. Nodes: ${status.num_nodes}, Girth: ${status.girth || "∞"}`
          );
        }
      }
    } catch (error) {
      console.error("Polling error:", error);
      stopPolling();
      resetButtons();
    }
  }, interval); // Use configurable interval
}

/**
 * Stop polling
 */
function stopPolling() {
  if (pollingInterval) {
    clearInterval(pollingInterval);
    pollingInterval = null;
  }
}

/**
 * Update status display and graph visualization
 */
function updateStatus(status) {
  const statusDisplay = document.getElementById("statusDisplay");

  // Build status HTML
  let html = `
        <div style="margin-bottom: 8px;">
            <strong>Target:</strong> (${status.k},${status.g})-cage
        </div>
        <div style="margin-bottom: 8px;">
            <strong>Step:</strong> ${status.step_count}
        </div>
        <div style="margin-bottom: 8px;">
            <strong>Nodes:</strong> ${status.num_nodes} / Moore bound: ${status.moore_bound}
        </div>
        <div style="margin-bottom: 8px;">
            <strong>Edges:</strong> ${status.num_edges}
        </div>
        <div style="margin-bottom: 8px;">
            <strong>k-regular:</strong> ${status.is_k_regular ? "✓" : "✗"}
        </div>
        <div style="margin-bottom: 8px;">
            <strong>Girth:</strong> ${status.girth || "∞"} (target: ${status.g})
        </div>
    `;

  if (!status.is_complete) {
    html += `
            <div style="margin-top: 12px; color: #888;">
                ⏳ Generating... (${status.elapsed_time.toFixed(1)}s)
            </div>
        `;
  }

  statusDisplay.innerHTML = html;

  // Update graph visualization
  if (window.interactiveGraph && status.current_graph) {
    window.interactiveGraph.loadFromEdgeList(status.current_graph, null);
  }
}

/**
 * Reset buttons to initial state
 */
function resetButtons() {
  const generateBtn = document.getElementById("generateBtn");
  const stopBtn = document.getElementById("stopBtn");

  generateBtn.disabled = false;
  generateBtn.textContent = "Generate Cage";
  stopBtn.disabled = true;
  stopBtn.style.display = "none";
}

/**
 * Show error message
 */
function showError(message) {
  console.error(message);
  const statusDisplay = document.getElementById("statusDisplay");
  statusDisplay.innerHTML = `
        <div style="padding: 12px; background: #a52; border-radius: 4px; color: #fff;">
            <strong>Error:</strong> ${message}
        </div>
    `;
}

/**
 * Show success message
 */
function showSuccess(message) {
  console.log(message);
  const statusDisplay = document.getElementById("statusDisplay");
  const currentHTML = statusDisplay.innerHTML;
  statusDisplay.innerHTML =
    currentHTML +
    `
        <div style="margin-top: 12px; padding: 12px; background: #2a5; border-radius: 4px; color: #fff;">
            <strong>✓</strong> ${message}
        </div>
    `;
}

/**
 * Initialize event listeners
 */
function initializeEventListeners() {
  // Stop polling when page is closed
  window.addEventListener("beforeunload", () => {
    stopPolling();
  });
  
  // Update polling interval display when slider moves
  const pollingSlider = document.getElementById("pollingInterval");
  if (pollingSlider) {
    pollingSlider.addEventListener("input", updatePollingDisplay);
  }
}

/**
 * Settings Management
 */
function loadSettings() {
  const defaultSettings = {
    generatorType: 'constructive',
    pollingInterval: 300
  };
  
  const saved = localStorage.getItem('cageGeneratorSettings');
  return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
}

function saveSettingsToStorage(settings) {
  localStorage.setItem('cageGeneratorSettings', JSON.stringify(settings));
}

function openSettings() {
  const modal = document.getElementById("settingsModal");
  const settings = loadSettings();

  // Populate current values
  document.getElementById("generatorType").value = settings.generatorType;
  document.getElementById("pollingInterval").value = settings.pollingInterval;
  
  updatePollingDisplay();
  modal.classList.add("show");
}

function closeSettings() {
  const modal = document.getElementById("settingsModal");
  modal.classList.remove("show");
}

function saveSettings() {
  const generatorType = document.getElementById("generatorType").value;
  const pollingInterval = parseInt(document.getElementById("pollingInterval").value);

  const settings = {
    generatorType,
    pollingInterval
  };

  saveSettingsToStorage(settings);
  closeSettings();
}

function updatePollingDisplay() {
  const slider = document.getElementById("pollingInterval");
  const display = document.getElementById("pollingDisplay");
  if (slider && display) {
    display.textContent = `${slider.value}ms`;
  }
}

// Initialize when DOM is loaded
document.addEventListener("DOMContentLoaded", initializeEventListeners);

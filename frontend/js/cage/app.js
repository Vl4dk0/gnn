let currentSessionId = null;
let pollingInterval = null;

const CAGE_K_MIN = 2;
const CAGE_G_MIN = 3;
const DEFAULT_MAX_MOORE_BOUND = 120;
let maxMooreBound = DEFAULT_MAX_MOORE_BOUND;

function clearCanvas() {
  if (window.interactiveGraph) {
    window.interactiveGraph.clear();
  }
}

async function startGeneration() {
  const k = parseInt(document.getElementById("degreeK").value);
  const g = parseInt(document.getElementById("girthG").value);
  const generateBtn = document.getElementById("generateBtn");
  const stopBtn = document.getElementById("stopBtn");

  if (!isGenerationRequestValid(k, g)) {
    return;
  }

  // Load settings to get selected generator
  const settings = loadSettings();
  const generator = settings.generatorType || "randomwalk";

  // Disable generate button
  generateBtn.disabled = true;
  generateBtn.textContent = "Generating...";
  stopBtn.disabled = false;
  stopBtn.style.display = "block";

  try {
    // Start generation on backend with selected generator
    const response = await fetch(`${API_CAGE_URL}/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ k, g, generator }),
    });

    const data = await response.json();

    if (!response.ok) {
      if (
        response.status === 429 &&
        data.active_sessions !== undefined &&
        data.max_parallel_generations !== undefined
      ) {
        throw new Error(
          `${data.error} (${data.active_sessions}/${data.max_parallel_generations} running)`,
        );
      }
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

async function stopGeneration() {
  if (!currentSessionId) return;

  try {
    // Fetch current status before stopping
    const statusResponse = await fetch(
      `${API_CAGE_URL}/status/${currentSessionId}`,
    );
    const status = await statusResponse.json();

    // Stop the session
    await fetch(`${API_CAGE_URL}/stop/${currentSessionId}`, {
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

function startPolling() {
  stopPolling();

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
        `${API_CAGE_URL}/status/${currentSessionId}`,
      );
      const status = await response.json();

      if (!response.ok) {
        throw new Error(status.error || "Failed to get status");
      }

      updateStatus(status);

      // Check if generation was stopped due to no polling (abandoned)
      if (status.stopped) {
        stopPolling();
        resetButtons();
        showError("Generation stopped - page was navigated away");
        return;
      }

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
            `Generation completed but cage is not valid. Nodes: ${status.num_nodes}, Girth: ${status.girth || "∞"}`,
          );
        }
      }
    } catch (error) {
      console.error("Polling error:", error);
      stopPolling();
      resetButtons();
    }
  }, interval);
}

function stopPolling() {
  if (pollingInterval) {
    clearInterval(pollingInterval);
    pollingInterval = null;
  }
}

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

function resetButtons() {
  const generateBtn = document.getElementById("generateBtn");
  const stopBtn = document.getElementById("stopBtn");

  generateBtn.disabled = false;
  generateBtn.textContent = "Generate Cage";
  stopBtn.disabled = true;
  stopBtn.style.display = "none";
  updateGenerationValidation();
}

function showError(message) {
  console.error(message);
  const statusDisplay = document.getElementById("statusDisplay");
  statusDisplay.innerHTML = `
        <div style="padding: 12px; background: #a52; border-radius: 4px; color: #fff;">
            <strong>Error:</strong> ${message}
        </div>
    `;
}

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

function mooreBound(k, g) {
  if (g % 2 === 1) {
    let sumTerm = 0;
    for (let i = 0; i <= (g - 3) / 2; i += 1) {
      sumTerm += (k - 1) ** i;
    }
    return 1 + k * sumTerm;
  }

  let sumTerm = 0;
  for (let i = 0; i <= g / 2 - 1; i += 1) {
    sumTerm += (k - 1) ** i;
  }
  return 2 * sumTerm;
}

function isGenerationRunning() {
  const stopBtn = document.getElementById("stopBtn");
  return stopBtn && stopBtn.style.display !== "none";
}

function isGenerationRequestValid(k, g) {
  if (!Number.isInteger(k) || !Number.isInteger(g)) return false;
  if (k < CAGE_K_MIN || g < CAGE_G_MIN) return false;
  return mooreBound(k, g) <= maxMooreBound;
}

function updateGenerationValidation() {
  const kInput = document.getElementById("degreeK");
  const gInput = document.getElementById("girthG");
  const generateBtn = document.getElementById("generateBtn");
  const hint = document.getElementById("mooreBoundHint");

  if (!kInput || !gInput || !generateBtn || !hint) return;

  const k = parseInt(kInput.value);
  const g = parseInt(gInput.value);

  if (!Number.isInteger(k) || !Number.isInteger(g)) {
    if (!isGenerationRunning()) generateBtn.disabled = true;
    if (!isGenerationRunning()) generateBtn.textContent = "Generate Cage";
    hint.textContent = `* Moore bound must be smaller than ${maxMooreBound}.`;
    hint.style.visibility = "hidden";
    return;
  }

  if (k < CAGE_K_MIN || g < CAGE_G_MIN) {
    if (!isGenerationRunning()) generateBtn.disabled = true;
    if (!isGenerationRunning()) generateBtn.textContent = "Generate Cage";
    hint.textContent = `* Moore bound must be smaller than ${maxMooreBound}.`;
    hint.style.visibility = "hidden";
    return;
  }

  const mb = mooreBound(k, g);
  if (mb > maxMooreBound) {
    if (!isGenerationRunning()) {
      generateBtn.disabled = true;
      generateBtn.textContent = `MB = ${mb} > ${maxMooreBound}`;
    }
    hint.textContent = `* Moore bound must be smaller than ${maxMooreBound}.`;
    hint.style.color = "#8f8f8f";
    hint.style.visibility = "visible";
    return;
  }

  if (!isGenerationRunning()) {
    generateBtn.disabled = false;
    generateBtn.textContent = "Generate Cage";
  }
  hint.textContent = `* Moore bound must be smaller than ${maxMooreBound}.`;
  hint.style.visibility = "hidden";
}

async function loadGenerationLimits() {
  try {
    const response = await fetch(`${API_CAGE_URL}/status`);
    if (!response.ok) return;
    const data = await response.json();
    if (Number.isInteger(data.max_moore_bound)) {
      maxMooreBound = data.max_moore_bound;
    }
  } catch (error) {
    console.warn("Failed to load cage limits from backend status:", error);
  } finally {
    updateGenerationValidation();
  }
}

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

  // Load and apply physics settings
  const settings = loadSettings();
  if (window.interactiveGraph) {
    if (settings.enablePhysics) {
      window.interactiveGraph.enablePhysics();
    } else {
      window.interactiveGraph.disablePhysics();
    }
  }

  const kInput = document.getElementById("degreeK");
  const gInput = document.getElementById("girthG");
  if (kInput) {
    kInput.addEventListener("input", updateGenerationValidation);
  }
  if (gInput) {
    gInput.addEventListener("input", updateGenerationValidation);
  }

  updateGenerationValidation();
  loadGenerationLimits();
}

function initializeMobileInputToggle() {
  // Mobile nav logic moved to shared/sidebar.js
}

function loadSettings() {
  const defaultSettings = {
    generatorType: "randomwalk",
    pollingInterval: 500,
    enablePhysics: true,
  };

  const saved = localStorage.getItem("cageGeneratorSettings");
  return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
}

function saveSettingsToStorage(settings) {
  localStorage.setItem("cageGeneratorSettings", JSON.stringify(settings));
}

function openSettings() {
  const modal = document.getElementById("settingsModal");
  const settings = loadSettings();

  // Populate current values
  document.getElementById("generatorType").value = settings.generatorType;
  document.getElementById("pollingInterval").value = settings.pollingInterval;
  document.getElementById("enablePhysics").checked = settings.enablePhysics;

  updatePollingDisplay();
  modal.classList.add("show");
}

function closeSettings() {
  const modal = document.getElementById("settingsModal");
  modal.classList.remove("show");
}

function saveSettings() {
  const generatorType = document.getElementById("generatorType").value;
  const pollingInterval = parseInt(
    document.getElementById("pollingInterval").value,
  );
  const enablePhysics = document.getElementById("enablePhysics").checked;

  const settings = {
    generatorType,
    pollingInterval,
    enablePhysics,
  };

  saveSettingsToStorage(settings);

  // Apply physics setting immediately
  if (window.interactiveGraph) {
    if (enablePhysics) {
      window.interactiveGraph.enablePhysics();
    } else {
      window.interactiveGraph.disablePhysics();
    }
  }

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
document.addEventListener("DOMContentLoaded", initializeMobileInputToggle);

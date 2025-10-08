/**
 * Configuration management for the frontend
 * Loads API configuration from the backend
 */

let API_BASE_URL = "http://localhost:5555"; // Default fallback

/**
 * Load configuration from the backend
 */
async function loadConfig() {
  try {
    // Try to fetch config from the server
    const response = await fetch(`${API_BASE_URL}/config`);
    if (response.ok) {
      const config = await response.json();
      if (config.apiBaseUrl) {
        API_BASE_URL = config.apiBaseUrl;
      }
    }
  } catch (error) {
    console.warn("Could not load config from server, using defaults:", error);
    // Keep the default value
  }
}

// Load config immediately when this script is loaded
// This ensures API_BASE_URL is set before other scripts run
(async () => {
  await loadConfig();
})();

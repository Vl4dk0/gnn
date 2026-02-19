/**
 * Runtime frontend configuration.
 *
 * Priority:
 * 1) window.__APP_CONFIG__.apiBaseUrl (served by /config.js)
 * 2) localStorage.apiBaseUrl
 * 3) current page origin
 */
(function initApiConfig(global) {
  const fromScript = global.__APP_CONFIG__?.apiBaseUrl || "";
  const fromStorage = global.localStorage?.getItem("apiBaseUrl") || "";
  const chosenBase = (fromScript || fromStorage || global.location.origin)
    .trim()
    .replace(/\/+$/, "");

  global.API_BASE_URL = chosenBase;
  global.API_DEGREE_URL = `${chosenBase}/api/degree`;
  global.API_CAGE_URL = `${chosenBase}/api/cage`;
  global.API_MINCYCLE_URL = `${chosenBase}/api/min_cycle`;
})(window);

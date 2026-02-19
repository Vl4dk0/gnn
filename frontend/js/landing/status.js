function setProjectStatus(project, active, reason) {
  const card = document.querySelector(`[data-project="${project}"]`);
  if (!card) return;

  const statusNode = card.querySelector(".project-status");
  if (!statusNode) return;

  if (active) {
    statusNode.textContent = "Active";
    statusNode.classList.remove("status-coming-soon");
    statusNode.classList.add("status-active");
    card.classList.remove("is-inactive");
    return;
  }

  statusNode.textContent = reason || "Inactive";
  statusNode.classList.remove("status-active");
  statusNode.classList.add("status-coming-soon");
  card.classList.add("is-inactive");
  card.addEventListener("click", (event) => event.preventDefault());
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

async function updateProjectStatuses() {
  try {
    const [degreeResult, minCycleResult, cageResult] = await Promise.allSettled([
      fetchJson(`${API_DEGREE_URL}/models`),
      fetchJson(`${API_MINCYCLE_URL}/models`),
      fetchJson(`${API_CAGE_URL}/status`),
    ]);

    if (degreeResult.status === "fulfilled") {
      const models = degreeResult.value.models || [];
      setProjectStatus(
        "degree",
        models.length > 0,
        models.length > 0 ? "" : "No models"
      );
    } else {
      setProjectStatus("degree", false, "API offline");
    }

    if (minCycleResult.status === "fulfilled") {
      const models = minCycleResult.value.models || [];
      setProjectStatus(
        "min_cycle",
        models.length > 0,
        models.length > 0 ? "" : "No models"
      );
    } else {
      setProjectStatus("min_cycle", false, "API offline");
    }

    if (cageResult.status === "fulfilled") {
      setProjectStatus("cage", true, "");
    } else {
      setProjectStatus("cage", false, "API offline");
    }
  } catch (_error) {
    setProjectStatus("degree", false, "API offline");
    setProjectStatus("min_cycle", false, "API offline");
    setProjectStatus("cage", false, "API offline");
  }
}

void updateProjectStatuses();

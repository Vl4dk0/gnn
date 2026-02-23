(function () {
  "use strict";

  function toNumber(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function formatAccuracy(value) {
    const n = toNumber(value);
    return n === null ? "-" : `${n.toFixed(2)}%`;
  }

  function formatMetric(value) {
    const n = toNumber(value);
    return n === null ? "-" : n.toFixed(4);
  }

  function formatCreatedAt(value) {
    const date = parseDate(value);
    if (!date) {
      return "-";
    }
    const yyyy = String(date.getFullYear());
    const mm = String(date.getMonth() + 1).padStart(2, "0");
    const dd = String(date.getDate()).padStart(2, "0");
    return `${yyyy}-${mm}-${dd}`;
  }

  function normalizeModel(model) {
    const metrics =
      model && typeof model === "object" ? model.metrics || {} : {};
    return {
      model_id:
        model && typeof model.model_id === "string" && model.model_id.trim()
          ? model.model_id
          : null,
      accuracy: toNumber(metrics.accuracy),
      mae: toNumber(metrics.mae),
      mse: toNumber(metrics.mse),
      created_at:
        model && typeof model.created_at === "string" ? model.created_at : null,
    };
  }

  async function fetchModels(url) {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const json = await res.json();
    if (!json || !Array.isArray(json.models)) {
      return [];
    }
    return json.models.map(normalizeModel).filter((m) => m.model_id);
  }

  function clearNode(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function addMessageRow(tbody, cols, text) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = cols;
    td.textContent = text;
    tr.appendChild(td);
    tbody.appendChild(tr);
  }

  function addCells(tr, values) {
    values.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    });
  }

  function renderTaskTable(tbody, models) {
    clearNode(tbody);
    if (!models || models.length === 0) {
      addMessageRow(tbody, 5, "No models found");
      return;
    }
    models.forEach((m) => {
      const tr = document.createElement("tr");
      addCells(tr, [
        m.model_id || "-",
        formatAccuracy(m.accuracy),
        formatMetric(m.mae),
        formatMetric(m.mse),
        formatCreatedAt(m.created_at),
      ]);
      tbody.appendChild(tr);
    });
  }

  function renderTaskError(tbody) {
    clearNode(tbody);
    addMessageRow(tbody, 5, "Live metrics unavailable");
  }

  function renderAssessmentTable(tbody, degreeModels, cycleModels) {
    clearNode(tbody);

    const degreeMap = new Map();
    const cycleMap = new Map();

    degreeModels.forEach((m) => degreeMap.set(m.model_id, m));
    cycleModels.forEach((m) => cycleMap.set(m.model_id, m));

    const ids = Array.from(
      new Set([...degreeMap.keys(), ...cycleMap.keys()]),
    ).sort((a, b) => a.localeCompare(b));

    if (ids.length === 0) {
      addMessageRow(tbody, 3, "No models found");
      return;
    }

    ids.forEach((id) => {
      const d = degreeMap.get(id);
      const c = cycleMap.get(id);
      const tr = document.createElement("tr");
      addCells(tr, [
        id,
        d ? formatAccuracy(d.accuracy) : "-",
        c ? formatAccuracy(c.accuracy) : "-",
      ]);
      tbody.appendChild(tr);
    });
  }

  function parseDate(raw) {
    if (typeof raw !== "string" || raw.trim() === "") {
      return null;
    }
    const date = new Date(raw);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  async function main() {
    const degreeBody = document.getElementById("degree-metrics-body");
    const cycleBody = document.getElementById("mincycle-metrics-body");
    const assessmentBody = document.getElementById("assessment-metrics-body");

    if (!degreeBody && !cycleBody && !assessmentBody) {
      return;
    }

    const degreeUrl = window.API_DEGREE_URL
      ? `${window.API_DEGREE_URL}/models`
      : null;
    const cycleUrl = window.API_MINCYCLE_URL
      ? `${window.API_MINCYCLE_URL}/models`
      : null;

    if (!degreeUrl || !cycleUrl) {
      if (degreeBody) renderTaskError(degreeBody);
      if (cycleBody) renderTaskError(cycleBody);
      if (assessmentBody) {
        clearNode(assessmentBody);
        addMessageRow(assessmentBody, 3, "Live metrics unavailable");
      }
      return;
    }

    const [degreeRes, cycleRes] = await Promise.allSettled([
      fetchModels(degreeUrl),
      fetchModels(cycleUrl),
    ]);

    const degreeModels =
      degreeRes.status === "fulfilled" ? degreeRes.value : null;
    const cycleModels = cycleRes.status === "fulfilled" ? cycleRes.value : null;

    if (degreeBody) {
      if (degreeModels === null) renderTaskError(degreeBody);
      else renderTaskTable(degreeBody, degreeModels);
    }

    if (cycleBody) {
      if (cycleModels === null) renderTaskError(cycleBody);
      else renderTaskTable(cycleBody, cycleModels);
    }

    if (assessmentBody) {
      if (degreeModels === null && cycleModels === null) {
        clearNode(assessmentBody);
        addMessageRow(assessmentBody, 3, "Live metrics unavailable");
      } else {
        renderAssessmentTable(
          assessmentBody,
          degreeModels || [],
          cycleModels || [],
        );
      }
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    main().catch(() => {
      const degreeBody = document.getElementById("degree-metrics-body");
      const cycleBody = document.getElementById("mincycle-metrics-body");
      const assessmentBody = document.getElementById("assessment-metrics-body");
      if (degreeBody) renderTaskError(degreeBody);
      if (cycleBody) renderTaskError(cycleBody);
      if (assessmentBody) {
        clearNode(assessmentBody);
        addMessageRow(assessmentBody, 3, "Live metrics unavailable");
      }
    });
  });
})();

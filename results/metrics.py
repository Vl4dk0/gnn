"""Universal result record and model-metadata helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ai.registry import get_trained_dir


@dataclass
class TrialResult:
    benchmark: str  # "degree"|"min_cycle"|"cage"|"refine"|"excision"
    approach: str  # e.g. "voltage","rl","randomwalk","gcn","tabu_refine","excision"
    variant: str  # ""|"no_gnn"|"gnn"|"classical"|"rl" or a model architecture name
    label: str  # human-readable unique label, e.g. "voltage[gnn]"
    instance: str  # graph/target/instance id, e.g. "k3_g5_seed0" or "petersen"
    target: dict[str, int]  # {"k":..,"g":..} or {}
    success: bool
    elapsed_s: float
    steps: int | None  # cage steps / refine iterations / excision-rl steps; None if N/A
    n_nodes: int | None  # result graph |V| (None if N/A)
    n_edges: int | None  # result graph |E| (None if N/A)
    metrics: dict[str, float] = field(default_factory=dict)
    # extensible bag: accuracy,mae,girth,moore_bound,moore_ratio,cost_initial,cost_final,...
    model_id: str | None = None
    model_params: int | None = None
    model_size_mb: float | None = None
    model_hparams: dict[str, object] | None = None
    error: str | None = None


def model_meta_from_dir(model_dir: Path) -> tuple[int, float, dict[str, object]]:
    """Return (param_count, size_mb, hparams) for a model directory.

    Guards missing files: returns (0, 0.0, {}) rather than crashing.
    """
    weights_path = model_dir / "weights.pt"
    info_path = model_dir / "info.json"

    param_count = 0
    size_mb = 0.0
    hparams: dict[str, object] = {}

    if weights_path.exists():
        try:
            state: dict[str, torch.Tensor] = torch.load(
                weights_path, map_location="cpu"
            )
            param_count = sum(int(v.numel()) for v in state.values())
            size_mb = weights_path.stat().st_size / 1024**2
        except Exception:
            pass

    if info_path.exists():
        try:
            raw = json.loads(info_path.read_text())
            hparams = dict(raw.get("training", {}))
        except Exception:
            pass

    return param_count, size_mb, hparams


def model_meta(task: str, model_id: str) -> tuple[int, float, dict[str, object]]:
    """Resolve model dir via the AI registry and return (param_count, size_mb, hparams)."""
    model_dir = get_trained_dir(task) / model_id
    return model_meta_from_dir(model_dir)

"""Shared utilities for the cage module."""

from __future__ import annotations

import json
import random
import re
import urllib.request
from collections.abc import Callable, Mapping
from http.client import HTTPResponse
from pathlib import Path
from typing import TypedDict, TypeVar, cast

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.loader import DataLoader  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.nn import GINEConv  # pyright: ignore[reportMissingTypeStubs]

# --- GNN helpers ---


def build_gine_stack(
    hidden_dim: int, num_layers: int
) -> tuple[nn.ModuleList, nn.ModuleList]:
    """Build a stack of GINEConv + BatchNorm1d layers.

    Returns (convs, bns) each of length num_layers.  Every GINEConv uses a
    two-layer ReLU MLP of width hidden_dim and trains its eps parameter.
    """
    convs: nn.ModuleList = nn.ModuleList()
    bns: nn.ModuleList = nn.ModuleList()
    for _ in range(num_layers):
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        _ = convs.append(GINEConv(mlp, train_eps=True))
        _ = bns.append(nn.BatchNorm1d(hidden_dim))
    return convs, bns


# --- PPO helpers ---


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """Generalized Advantage Estimation.

    `values` must be length len(rewards) + 1: the last entry is the
    bootstrap value of the final state.
    """
    advantages: list[float] = []
    gae = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * (1 - int(dones[i])) - values[i]
        gae = delta + gamma * lam * (1 - int(dones[i])) * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages, dtype=torch.float)


# --- Training helpers ---

GIRTH_NORM = 12.0

# The tabu-cost target (count of short identity walks) is non-negative and
# heavy-tailed, so we regress it in log1p space and divide by a fixed scale.
# log1p(count) for the counts seen in practice (single to low hundreds) lands
# in roughly [0, 6], so this keeps the normalized target O(1) like girth/NORM.
TABU_NORM = 6.0


def normalize_target(label: torch.Tensor, kind: str) -> torch.Tensor:
    """Map a raw label tensor into the normalized space the loss operates in.

    girth: linear divide by GIRTH_NORM (higher = better).
    tabu_cost: log1p then divide by TABU_NORM (lower = better, heavy-tailed).
    Reused by the loss in train.py and the eval metrics below.
    """
    if kind == "tabu_cost":
        return torch.log1p(label.clamp(min=0)) / TABU_NORM
    return label / GIRTH_NORM


_T = TypeVar("_T", bound=Data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_targets(
    targets_arg: str | None,
    default_k: int | None = None,
    default_g: int | None = None,
) -> list[tuple[int, int]]:
    """Parse "k,g;k,g;..." into a list of (k, g) tuples.

    When targets_arg is None or empty, falls back to (default_k, default_g)
    if both are provided; otherwise raises ValueError.
    """
    if targets_arg:
        out: list[tuple[int, int]] = []
        for piece in targets_arg.split(";"):
            piece = piece.strip()
            if not piece:
                continue
            parts = piece.split(",")
            if len(parts) != 2:
                raise ValueError(f"bad target {piece!r}: expected 'k,g'")
            k, g = int(parts[0].strip()), int(parts[1].strip())
            if k < 2 or g < 3:
                raise ValueError(f"bad target {piece!r}: need k >= 2 and g >= 3")
            out.append((k, g))
        if not out:
            raise ValueError(f"Failed to parse --targets: {targets_arg!r}")
        return out
    if default_k is not None and default_g is not None:
        return [(default_k, default_g)]
    raise ValueError("Provide either --targets or both --k and --g")


def make_stratified_loader(
    train_data: list[_T],
    batch_size: int,
    key_fn: Callable[[_T], tuple[int, ...]],
) -> DataLoader:
    """Build a DataLoader with inverse-frequency sampling over key_fn buckets.

    When only one bucket exists, falls back to a shuffled DataLoader.
    """
    counts: dict[tuple[int, ...], int] = {}
    for d in train_data:
        key = key_fn(d)
        counts[key] = counts.get(key, 0) + 1

    if len(counts) > 1:
        weights: list[float] = [1.0 / counts[key_fn(d)] for d in train_data]
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(train_data), replacement=True
        )
        return DataLoader(train_data, batch_size=batch_size, sampler=sampler)

    return DataLoader(train_data, batch_size=batch_size, shuffle=True)


def evaluate_girth_predictor(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    kind: str = "girth",
) -> dict[str, float]:
    """Compute MSE loss, MAE, and binary classification metrics.

    The label lives in ``batch.girth`` for both kinds (girth value, or tabu
    cost), so the GNN code is identical. The "achieves target" positive is
    defined per kind so the A/B is fair:
      - girth:     pred >= g_target  (higher is better)
      - tabu_cost: pred < 0.5        (count == 0, i.e. girth >= g_target)
    The loss is MSE in the kind-specific normalized space.
    """
    _ = model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total = 0
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)

            label_true = cast(torch.Tensor, batch.girth).float().unsqueeze(-1)
            g_target = cast(torch.Tensor, batch.g_target).float().unsqueeze(-1)

            loss = F.mse_loss(
                normalize_target(pred, kind),
                normalize_target(label_true, kind),
            )

            n = int(label_true.size(0))
            total_loss += float(loss.item()) * n
            total_mae += float(F.l1_loss(pred, label_true, reduction="sum").item())

            if kind == "tabu_cost":
                # count == 0 is the feasible (achieves-target) class.
                pred_class = (pred < 0.5).float()
                class_true = (label_true < 0.5).float()
            else:
                pred_class = (pred >= g_target).float()
                class_true = (label_true >= g_target).float()
            tp += int(
                cast(torch.Tensor, ((pred_class == 1) & (class_true == 1)).sum()).item()
            )
            fp += int(
                cast(torch.Tensor, ((pred_class == 1) & (class_true == 0)).sum()).item()
            )
            fn += int(
                cast(torch.Tensor, ((pred_class == 0) & (class_true == 1)).sum()).item()
            )
            tn += int(
                cast(torch.Tensor, ((pred_class == 0) & (class_true == 0)).sum()).item()
            )
            total += n

    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "loss": total_loss / max(total, 1),
        "mae": total_mae / max(total, 1),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def save_predictor_artifacts(
    model: torch.nn.Module,
    save_dir: Path,
    info: Mapping[str, object],
) -> tuple[Path, Path]:
    """Write weights.pt (CPU state_dict) and info.json to save_dir."""
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_path = save_dir / "weights.pt"
    info_path = save_dir / "info.json"
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, weights_path)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    return weights_path, info_path


def load_predictor_artifacts(
    model_dir: Path,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Read info.json and weights.pt from model_dir.

    Raises FileNotFoundError if either file is missing.
    """
    info_path = model_dir / "info.json"
    weights_path = model_dir / "weights.pt"
    if not info_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing weights.pt or info.json in {model_dir}")
    with open(info_path) as f:
        info = cast(dict[str, object], json.load(f))
    state = cast(dict[str, torch.Tensor], torch.load(weights_path, map_location="cpu"))
    return info, state


# --- House of Graphs cage data ---

_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "cages"
_HOG_BASE = "https://houseofgraphs.org/data/cages"
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# fmt: off
# All (k,g) → filename entries from the House of Graphs cage survey table.
# Sourced from the cage survey page at houseofgraphs.org/cages.
_HOG_TABLE: dict[tuple[int, int], str] = {
    (3,  5): "cagesk3g05.g6",
    (3,  6): "cagesk3g06.g6",
    (3,  7): "cagesk3g07.g6",
    (3,  8): "cagesk3g08.g6",
    (3,  9): "cagesk3g09.g6",
    (3, 10): "cagesk3g10.g6",
    (3, 11): "cagesk3g11.g6",
    (3, 12): "cagesk3g12.g6",
    (3, 13): "k3_g13_n272.g6",
    (3, 14): "k3_g14_n384.g6",
    (3, 15): "k3_g15_n620.g6",
    (3, 16): "k3_g16_n936.g6",
    (3, 17): "k3_g17_n2048.g6",
    (3, 18): "k3_g18_n2560.g6",
    (3, 19): "k3_g19_n4324.g6",
    (3, 20): "k3_g20_n5376.g6",
    (4,  5): "cagesk4g05.g6",
    (4,  6): "cagesk4g06.g6",
    (4,  7): "cagesk4g07.g6",
    (4,  8): "cagesk4g08.g6",
    (4,  9): "k4_g9_n270.g6",
    (4, 10): "k4_g10_n320.g6",
    (4, 11): "k4_g11_n713.g6",
    (4, 12): "k4_g12_n728.g6",
    (5,  5): "cagesk5g05.g6",
    (5,  6): "cagesk5g06.g6",
    (5,  7): "k5_g7_n152.g6",
    (5,  8): "k5_g8_n170.g6",
    (5,  9): "k5_g9_n1116.g6",
    (5, 10): "k5_g10_n1296.g6",
    (5, 11): "k5_g11_n2688.g6",
    (5, 12): "k5_g12_n2730.g6",
    (6,  5): "cagesk6g05.g6",
    (6,  6): "cagesk6g06.g6",
    (6,  7): "k6_g7_n294.g6",
    (6,  8): "k6_g8_n312.g6",
    (6, 11): "k6_g11_n7783.g6",
    (7,  5): "cagesk7g05.g6",
    (7,  6): "cagesk7g06.g6",
    (7,  7): "k7_g7_n632.g6",
    (7,  8): "k7_g8_n658.g6",
    (8,  5): "k8_g5_n80.g6",
    (8,  6): "k8_g6_n114.g6",
    (8,  7): "k8_g7_n774.g6",
    (8,  8): "k8_g8_n800.g6",
    (9,  5): "k9_g5_n96.g6",
    (9,  7): "k9_g7_n1104.g6",
    (9,  8): "k9_g8_n1170.g6",
    (10, 5): "k10_g5_n124.g6",
    (10, 7): "k10_g7_n1608.g6",
    (10, 8): "k10_g8_n1640.g6",
    (11, 5): "k11_g5_n154.g6",
    (11, 6): "k11_g6_n240.g6",
    (11, 7): "k11_g7_n2576.g6",
    (11, 8): "k11_g8_n2618.g6",
    (12, 5): "k12_g5_n203.g6",
    (12, 6): "k12_g6_n266.g6",
    (12, 7): "k12_g7_n2890.g6",
    (12, 8): "k12_g8_n2928.g6",
    (13, 5): "k13_g5_n226.g6",
    (13, 6): "k13_g6_n336.g6",
    (13, 7): "k13_g7_n4292.g6",
    (13, 8): "k13_g8_n4342.g6",
    (14, 5): "k14_g5_n280.g6",
    (14, 6): "k14_g6_n366.g6",
    (14, 7): "k14_g7_n4716.g6",
    (14, 8): "k14_g8_n4760.g6",
    (15, 5): "k15_g5_n310.g6",
    (16, 5): "k16_g5_n336.g6",
    (17, 5): "k17_g5_n436.g6",
    (18, 5): "k18_g5_n468.g6",
    (18, 6): "k18_g6_n614.g6",
    (19, 5): "k19_g5_n500.g6",
    (19, 6): "k19_g6_n720.g6",
    (20, 6): "k20_g6_n762.g6",
}
# fmt: on


class HogCageEntry(TypedDict):
    k: int
    g: int
    filename: str
    cached: bool


def hog_list_available() -> list[tuple[int, int]]:
    """Return all (k,g) pairs in the HoG cage survey, sorted."""
    return sorted(_HOG_TABLE.keys())


def hog_is_available(k: int, g: int) -> bool:
    """Return True if this (k,g) pair is in the HoG cage survey."""
    return (k, g) in _HOG_TABLE


def hog_download(k: int, g: int, *, force: bool = False) -> Path:
    """Download the (k,g) cage file from HoG and cache it in data/cages/.

    Returns the local path. Skips download if already cached unless *force* is True.
    Raises KeyError if the pair is not in the HoG survey.
    """
    if (k, g) not in _HOG_TABLE:
        raise KeyError(f"No cage data for (k={k}, g={g}) in the HoG survey")

    filename = _HOG_TABLE[(k, g)]
    local = _DATA_DIR / filename
    if local.exists() and not force:
        return local

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{_HOG_BASE}/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with cast(HTTPResponse, urllib.request.urlopen(req, timeout=30)) as resp:
        data: bytes = resp.read()
    _ = local.write_bytes(data)
    return local


def hog_download_all(*, force: bool = False, verbose: bool = True) -> None:
    """Download every cage file in the HoG survey."""
    total = len(_HOG_TABLE)
    for i, (k, g) in enumerate(sorted(_HOG_TABLE.keys()), 1):
        try:
            local = _DATA_DIR / _HOG_TABLE[(k, g)]
            if local.exists() and not force:
                if verbose:
                    print(f"[{i}/{total}] (k={k}, g={g}) already cached")
                continue
            _ = hog_download(k, g, force=force)
            if verbose:
                filename = _HOG_TABLE[(k, g)]
                print(f"[{i}/{total}] (k={k}, g={g}) downloaded → {filename}")
        except Exception as e:
            print(f"[{i}/{total}] (k={k}, g={g}) FAILED: {e}")


def get_cages(k: int, g: int) -> list[nx.Graph[int]]:
    """Return all (k,g)-regular record graphs from the HoG survey.

    Downloads and caches the file on first call. The file may contain multiple
    non-isomorphic graphs (e.g. (3,9) has 18). Raises KeyError if the pair is
    not in the HoG survey.
    """
    local = hog_download(k, g)
    with open(local, "rb") as f:
        lines = [line.strip() for line in f if line.strip()]
    return cast("list[nx.Graph[int]]", [nx.from_graph6_bytes(line) for line in lines])


def hog_record_order(k: int, g: int) -> int | None:
    """Best-known order n(k,g) from the HoG survey, resolved offline.

    Parses the node count encoded in the survey filename (``k8_g5_n80.g6`` ->
    80). For old-format entries (``cagesk3g05.g6``) that do not encode the
    order, falls back to counting nodes of a locally cached graph. Returns
    ``None`` when the pair is unknown or no offline source gives the order;
    never downloads (safe to call on a network-isolated compute node).
    """
    filename = _HOG_TABLE.get((k, g))
    if filename is None:
        return None
    match = re.search(r"_n(\d+)\.g6$", filename)
    if match is not None:
        return int(match.group(1))
    local = _DATA_DIR / filename
    if local.exists():
        with open(local, "rb") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    graph = cast("nx.Graph[int]", nx.from_graph6_bytes(stripped))
                    return graph.number_of_nodes()
    return None

"""Shared utilities for cage girth-predictor training scripts."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.loader import DataLoader  # pyright: ignore[reportMissingTypeStubs]

GIRTH_NORM = 12.0

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
) -> dict[str, float]:
    """Compute MSE loss, MAE, and binary classification metrics.

    Treats girth_pred >= g_target as positive (achieves target girth).
    """
    _ = model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total = 0
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            girth_pred = model(batch)

            girth_true = cast(torch.Tensor, batch.girth).float().unsqueeze(-1)
            g_target = cast(torch.Tensor, batch.g_target).float().unsqueeze(-1)

            loss = F.mse_loss(girth_pred / GIRTH_NORM, girth_true / GIRTH_NORM)

            n = int(girth_true.size(0))
            total_loss += float(loss.item()) * n
            total_mae += float(
                F.l1_loss(girth_pred, girth_true, reduction="sum").item()
            )

            pred_class = (girth_pred >= g_target).float()
            class_true = (girth_true >= g_target).float()
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

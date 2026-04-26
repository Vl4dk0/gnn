"""Training script for the voltage graph girth predictor.

Usage:
    # Single target (Variant A — per-(k,g) model)
    uv run python -m ai.cage.voltage.train --k 3 --g 7 --samples 100000

    # Per-g (Variant B — fixed g, multiple k)
    uv run python -m ai.cage.voltage.train --targets "3,7;4,7;5,7" --samples 100000

    # Unified (Variant C — full grid)
    uv run python -m ai.cage.voltage.train --targets "3,5;3,6;4,5;4,6" --samples 200000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader  # pyright: ignore[reportMissingTypeStubs]

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ai.cage.voltage.data_gen import generate_dataset
from ai.cage.voltage.model import GirthPredictor
from ai.utils.device import get_preferred_device

GIRTH_NORM = 12.0  # Normalization constant so MSE loss is comparable to BCE


def _set_seed(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_targets(
    targets_arg: str | None, k: int | None, g: int | None
) -> list[tuple[int, int]]:
    """Parse the --targets string or fall back to single (k, g)."""
    if targets_arg:
        out: list[tuple[int, int]] = []
        for piece in targets_arg.split(";"):
            piece = piece.strip()
            if not piece:
                continue
            ks, gs = piece.split(",")
            out.append((int(ks.strip()), int(gs.strip())))
        if not out:
            raise ValueError(f"Failed to parse --targets: {targets_arg!r}")
        return out
    if k is None or g is None:
        raise ValueError("Provide either --targets or both --k and --g")
    return [(k, g)]


def _derive_model_id(targets: list[tuple[int, int]]) -> str:
    """Pick a save-folder name that reflects the target shape."""
    if len(targets) == 1:
        k, g = targets[0]
        return f"girth_predictor_k{k}_g{g}"
    ks = {k for (k, _) in targets}
    gs = {g for (_, g) in targets}
    if len(gs) == 1:
        (g,) = tuple(gs)
        return f"girth_predictor_g{g}_multik"
    if len(ks) == 1:
        (k,) = tuple(ks)
        return f"girth_predictor_k{k}_multig"
    return "girth_predictor_unified"


def _evaluate(
    model: GirthPredictor,
    loader: DataLoader,
    device: torch.device,
    regression_weight: float,
) -> dict[str, float]:
    """Evaluate model and return overall metrics."""
    _ = model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            girth_pred, class_logit = model(batch)

            girth_true = cast(torch.Tensor, batch.girth).float().unsqueeze(-1)
            class_true = cast(torch.Tensor, batch.girth_class).float().unsqueeze(-1)

            reg_loss = F.mse_loss(girth_pred / GIRTH_NORM, girth_true / GIRTH_NORM)
            cls_loss = F.binary_cross_entropy_with_logits(class_logit, class_true)
            loss = regression_weight * reg_loss + (1.0 - regression_weight) * cls_loss

            n = int(girth_true.size(0))
            total_loss += float(loss.item()) * n
            total_mae += float(
                F.l1_loss(girth_pred, girth_true, reduction="sum").item()
            )

            pred_class = (class_logit > 0).float()
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


def _evaluate_per_target(
    model: GirthPredictor,
    test_data: list,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    """Group test set by (k, g_target) and compute per-target metrics."""
    groups: dict[tuple[int, int], list] = {}  # pyright: ignore[reportMissingTypeArgument, reportUnknownVariableType]
    for d in test_data:  # pyright: ignore[reportUnknownVariableType]
        key = (int(cast(int, d.k)), int(cast(int, d.g_target)))
        groups.setdefault(key, []).append(d)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

    out: dict[str, dict[str, float]] = {}
    for (k, g), items in groups.items():  # pyright: ignore[reportUnknownVariableType]
        loader = DataLoader(items, batch_size=batch_size)  # pyright: ignore[reportUnknownArgumentType]
        # use regression_weight=0.5 for the per-target loss (only mae/f1 matter here)
        m = _evaluate(model, loader, device, 0.5)
        out[f"{k}_{g}"] = {
            "n": float(len(items)),  # pyright: ignore[reportUnknownArgumentType]
            "mae": round(m["mae"], 4),
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
        }
    return out


def train(
    targets: list[tuple[int, int]],
    num_samples: int = 50000,
    max_group_order: int = 60,
    epochs: int = 100,
    batch_size: int = 64,
    hidden_dim: int = 64,
    num_layers: int = 4,
    lr: float = 1e-3,
    print_every: int = 10,
    seed: int = 42,
    regression_weight: float = 0.5,
    model_id_override: str | None = None,
) -> tuple[GirthPredictor, dict[str, object]]:
    """Train the girth predictor model. Returns (model, info_dict)."""

    _set_seed(seed)
    device = torch.device(get_preferred_device())

    model_id = model_id_override or _derive_model_id(targets)

    print("=" * 60)
    print(f"Training Girth Predictor: {model_id}")
    print(f"  Targets: {targets}")
    print("=" * 60)
    print(f"  Device:           {device}")
    print(f"  Samples:          {num_samples}")
    print(f"  Max group order:  {max_group_order}")
    print(f"  Epochs:           {epochs}")
    print(f"  Batch size:       {batch_size}")
    print(f"  Hidden dim:       {hidden_dim}")
    print(f"  Layers:           {num_layers}")
    print(f"  Learning rate:    {lr}")
    print(f"  Seed:             {seed}")
    print(f"  Regression weight: {regression_weight}")
    print("=" * 60)

    print("Generating training data...")
    all_data, gen_stats = generate_dataset(
        targets=targets,
        num_samples=num_samples,
        max_group_order=max_group_order,
        seed=seed,
    )
    print(
        f"  Produced {gen_stats['produced']}/{gen_stats['requested']} unique samples "
        f"in {gen_stats['attempts']} attempts "
        f"({gen_stats['duplicates_skipped']} duplicates skipped)"
    )
    per_target_gen = cast(dict[str, dict[str, float]], gen_stats["per_target"])
    for tkey, ts in per_target_gen.items():
        print(
            f"    target {tkey}: produced={int(ts['produced'])}, "
            f"pos_rate={ts['pos_rate']:.3f}, "
            f"dup_skipped={int(ts['duplicates_skipped'])}"
        )

    # Shuffle then 80/10/10 split (post-deduplication, so no leakage)
    rng = random.Random(seed)
    rng.shuffle(all_data)
    n = len(all_data)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_data = all_data[:n_train]
    val_data = all_data[n_train : n_train + n_val]
    test_data = all_data[n_train + n_val :]

    def _stats(data_list: list) -> tuple[int, int, float]:  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
        total = len(data_list)  # pyright: ignore[reportUnknownArgumentType]
        pos = sum(1 for d in data_list if int(cast(int, d.girth_class)) == 1)  # pyright: ignore[reportUnknownVariableType]
        return total, pos, pos / max(total, 1)

    train_n, train_pos, train_rate = _stats(train_data)
    val_n, val_pos, val_rate = _stats(val_data)
    test_n, test_pos, test_rate = _stats(test_data)
    print(
        f"  Train: {train_n} ({train_pos} pos, {train_rate * 100:.1f}%) | "
        f"Val: {val_n} ({val_pos} pos, {val_rate * 100:.1f}%) | "
        f"Test: {test_n} ({test_pos} pos, {test_rate * 100:.1f}%)"
    )

    # Stratified sampler weighted by (k, g_target, girth_class) so neither
    # high-positivity targets nor majority class dominate.
    joint_counts: dict[tuple[int, int, int], int] = {}
    for d in train_data:
        key = (
            int(cast(int, d.k)),
            int(cast(int, d.g_target)),
            int(cast(int, d.girth_class)),
        )
        joint_counts[key] = joint_counts.get(key, 0) + 1

    if len(joint_counts) > 1:
        sample_weights: list[float] = []
        for d in train_data:
            key = (
                int(cast(int, d.k)),
                int(cast(int, d.g_target)),
                int(cast(int, d.girth_class)),
            )
            sample_weights.append(1.0 / joint_counts[key])
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=train_n,
            replacement=True,
        )
        train_loader = DataLoader(
            train_data, batch_size=batch_size, sampler=train_sampler
        )
        print(
            f"  Using WeightedRandomSampler over {len(joint_counts)} (k,g,class) buckets"
        )
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        print("  Using shuffled DataLoader for training (single bucket)")

    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Model
    model = GirthPredictor(
        node_feat_dim=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_group_order=max_group_order,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        _ = model.train()
        train_loss = 0.0
        train_count = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            girth_pred, class_logit = model(batch)

            girth_true = cast(torch.Tensor, batch.girth).float().unsqueeze(-1)
            class_true = cast(torch.Tensor, batch.girth_class).float().unsqueeze(-1)

            reg_loss = F.mse_loss(girth_pred / GIRTH_NORM, girth_true / GIRTH_NORM)
            cls_loss = F.binary_cross_entropy_with_logits(class_logit, class_true)
            loss = regression_weight * reg_loss + (1.0 - regression_weight) * cls_loss

            _ = loss.backward()  # pyright: ignore[reportUnknownMemberType]
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            n_b = int(girth_true.size(0))
            train_loss += float(loss.item()) * n_b
            train_count += n_b

        avg_train_loss = train_loss / max(train_count, 1)

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            val = _evaluate(model, val_loader, device, regression_weight)
            print(
                f"Epoch {epoch:4d} | Train: {avg_train_loss:.4f} | "
                f"Val: {val['loss']:.4f} | MAE: {val['mae']:.2f} | "
                f"Acc: {val['accuracy'] * 100:.1f}% | F1: {val['f1']:.3f}"
            )

            if val["loss"] < best_val_loss:
                best_val_loss = val["loss"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_epoch = epoch
                print(f"  -> New best (val_loss={best_val_loss:.4f}, epoch={epoch})")

    # Restore best
    if best_state is not None:
        _ = model.load_state_dict(best_state)
        _ = model.to(device)

    # Final evaluation on held-out test set (overall + per-target)
    print("\nEvaluating on held-out test set...")
    test_metrics = _evaluate(model, test_loader, device, regression_weight)
    print(
        f"  Test loss: {test_metrics['loss']:.4f} | "
        f"MAE: {test_metrics['mae']:.2f} | "
        f"Acc: {test_metrics['accuracy'] * 100:.1f}% | "
        f"Precision: {test_metrics['precision']:.3f} | "
        f"Recall: {test_metrics['recall']:.3f} | "
        f"F1: {test_metrics['f1']:.3f}"
    )

    test_per_target = _evaluate_per_target(model, test_data, device, batch_size)
    if len(test_per_target) > 1:
        print("  Per-target test metrics:")
        for tkey, m in test_per_target.items():
            print(
                f"    {tkey}: n={int(m['n'])}, mae={m['mae']:.2f}, "
                f"acc={m['accuracy'] * 100:.1f}%, f1={m['f1']:.3f}"
            )

    # Save model + info.json
    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "trained", "voltage_girth", model_id
    )
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, "weights.pt")

    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(cpu_state, weights_path)

    # For backward compat, also surface single-target k/g_target fields when applicable
    primary_k, primary_g = targets[0] if len(targets) == 1 else (None, None)

    training_block: dict[str, object] = {
        "targets": [list(t) for t in targets],
        "samples": num_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "max_group_order": max_group_order,
        "learning_rate": lr,
        "seed": seed,
        "regression_weight": regression_weight,
        "best_epoch": best_epoch,
    }
    if primary_k is not None:
        training_block["k"] = primary_k
        training_block["g_target"] = primary_g

    info: dict[str, object] = {
        "model_type": "voltage_girth_predictor",
        "model_id": model_id,
        "task": "voltage_girth",
        "training": training_block,
        "data": {
            "requested": gen_stats["requested"],
            "produced": gen_stats["produced"],
            "duplicates_skipped": gen_stats["duplicates_skipped"],
            "generation_attempts": gen_stats["attempts"],
            "per_target": per_target_gen,
            "train_samples": train_n,
            "val_samples": val_n,
            "test_samples": test_n,
            "train_pos_rate": round(train_rate, 4),
            "val_pos_rate": round(val_rate, 4),
            "test_pos_rate": round(test_rate, 4),
        },
        "metrics": {
            "test_loss": round(test_metrics["loss"], 4),
            "test_mae": round(test_metrics["mae"], 4),
            "test_accuracy": round(test_metrics["accuracy"], 4),
            "test_precision": round(test_metrics["precision"], 4),
            "test_recall": round(test_metrics["recall"], 4),
            "test_f1": round(test_metrics["f1"], 4),
            "test_tp": int(test_metrics["tp"]),
            "test_fp": int(test_metrics["fp"]),
            "test_fn": int(test_metrics["fn"]),
            "test_tn": int(test_metrics["tn"]),
            "best_val_loss": round(best_val_loss, 4),
            "test_per_target": test_per_target,
        },
    }

    info_path = os.path.join(save_dir, "info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nModel saved to {weights_path}")
    print(f"Info saved to  {info_path}")
    print("=" * 60)

    return model, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train girth predictor for voltage graph lifts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--k", type=int, default=None, help="Target degree (single-target mode)"
    )
    _ = parser.add_argument(
        "--g", type=int, default=None, help="Target girth (single-target mode)"
    )
    _ = parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help='Multiple targets, e.g. "3,5;3,6;4,5" (overrides --k/--g)',
    )
    _ = parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override the auto-derived model id (save folder name)",
    )
    _ = parser.add_argument(
        "--samples", type=int, default=50000, help="Number of training samples"
    )
    _ = parser.add_argument(
        "--max-group-order", type=int, default=60, help="Max group order"
    )
    _ = parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    _ = parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    _ = parser.add_argument(
        "--hidden-dim", type=int, default=64, help="Hidden dimension"
    )
    _ = parser.add_argument(
        "--num-layers", type=int, default=4, help="Number of GINEConv layers"
    )
    _ = parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    _ = parser.add_argument(
        "--print-every", type=int, default=10, help="Print every N epochs"
    )
    _ = parser.add_argument("--seed", type=int, default=42, help="Random seed")
    _ = parser.add_argument(
        "--regression-weight",
        type=float,
        default=0.5,
        help="Weight on regression loss (0=BCE only, 1=MSE only)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    targets = _parse_targets(
        cast(str | None, args.targets),
        cast(int | None, args.k),
        cast(int | None, args.g),
    )
    _ = train(
        targets=targets,
        num_samples=cast(int, args.samples),
        max_group_order=cast(int, args.max_group_order),
        epochs=cast(int, args.epochs),
        batch_size=cast(int, args.batch_size),
        hidden_dim=cast(int, args.hidden_dim),
        num_layers=cast(int, args.num_layers),
        lr=cast(float, args.lr),
        print_every=cast(int, args.print_every),
        seed=cast(int, args.seed),
        regression_weight=cast(float, args.regression_weight),
        model_id_override=cast(str | None, args.model_id),
    )

"""Training script for the group-promise predictor.

The model learns to predict, for a finite group G and target (k, g), the
best girth achievable by any degree-k symmetric generating set of G. The
labels come from the classical inner search (group_data_gen), so dataset
generation is CPU-heavy and benefits from --gen-workers; training itself
runs on the preferred device (GPU when available).

Usage:
    uv run python -m ai.cage.cayley.group.train \\
        --targets "3,6;3,7;3,8" --max-group-order 600 --gen-workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.cayley.group.data_gen import generate_group_dataset
from ai.cage.cayley.group.model import GroupPromisePredictor
from ai.utils.device import get_preferred_device

GIRTH_NORM = 12.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_targets(targets_arg: str) -> list[tuple[int, int]]:
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


def _evaluate(
    model: GroupPromisePredictor,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
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
    }


def train(
    targets: list[tuple[int, int]],
    max_group_order: int = 600,
    gen_random_trials: int = 800,
    gen_tabu_iters: int = 800,
    gen_workers: int = 1,
    epochs: int = 150,
    batch_size: int = 32,
    hidden_dim: int = 96,
    num_layers: int = 4,
    lr: float = 1e-3,
    print_every: int = 10,
    seed: int = 42,
    weight_decay: float = 1e-4,
    model_id_override: str | None = None,
) -> tuple[GroupPromisePredictor, dict[str, object]]:
    _set_seed(seed)
    device = torch.device(get_preferred_device())
    model_id = model_id_override or "group_promise_predictor_multi"

    print("=" * 60)
    print(f"Training Group-Promise Predictor: {model_id}")
    print(f"  Targets: {targets}")
    print(f"  Device: {device} | Epochs: {epochs}")
    print(f"  Hidden: {hidden_dim} | Layers: {num_layers} | LR: {lr}")
    print(f"  Max group order: {max_group_order} | Gen workers: {gen_workers}")
    print("=" * 60)

    print("Generating dataset (classical inner search per group)...")
    all_data, gen_stats = generate_group_dataset(
        targets=targets,
        max_group_order=max_group_order,
        num_random_trials=gen_random_trials,
        num_tabu_iters=gen_tabu_iters,
        num_workers=gen_workers,
        seed=seed,
    )
    print(
        f"  Produced {gen_stats['produced']} samples "
        + f"over {gen_stats['num_groups']} groups"
    )
    per_target_summary = cast(dict[str, dict[str, float]], gen_stats["per_target"])
    for tkey, ts in sorted(per_target_summary.items()):
        print(
            f"    target {tkey}: produced={int(ts['produced'])}, "
            + f"pos_rate={ts['pos_rate']:.3f}"
        )

    if len(all_data) < 10:
        raise RuntimeError(
            f"dataset too small ({len(all_data)} samples) — widen the catalogue"
        )

    rng = random.Random(seed)
    rng.shuffle(all_data)
    n = len(all_data)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_data = all_data[:n_train]
    val_data = all_data[n_train : n_train + n_val]
    test_data = all_data[n_train + n_val :]
    print(f"  Train/Val/Test: {len(train_data)}/{len(val_data)}/{len(test_data)}")

    # Stratified sampler over (k, g_target) buckets.
    joint_counts: dict[tuple[int, int], int] = {}
    for d in train_data:
        key = (int(cast(int, d.k)), int(cast(int, d.g_target)))
        joint_counts[key] = joint_counts.get(key, 0) + 1

    if len(joint_counts) > 1:
        sample_weights: list[float] = []
        for d in train_data:
            key = (int(cast(int, d.k)), int(cast(int, d.g_target)))
            sample_weights.append(1.0 / joint_counts[key])
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_data), replacement=True
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = GroupPromisePredictor(hidden_dim=hidden_dim, num_layers=num_layers).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
            girth_pred = model(batch)

            girth_true = cast(torch.Tensor, batch.girth).float().unsqueeze(-1)

            loss = F.mse_loss(girth_pred / GIRTH_NORM, girth_true / GIRTH_NORM)

            _ = loss.backward()  # pyright: ignore[reportUnknownMemberType]
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            n_b = int(girth_true.size(0))
            train_loss += float(loss.item()) * n_b
            train_count += n_b

        avg_train_loss = train_loss / max(train_count, 1)

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            val = _evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch:4d} | Train: {avg_train_loss:.4f} | "
                + f"Val: {val['loss']:.4f} | MAE: {val['mae']:.2f} | "
                + f"F1: {val['f1']:.3f}"
            )
            if val["loss"] < best_val_loss:
                best_val_loss = val["loss"]
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                best_epoch = epoch
                print(f"  -> New best (val_loss={best_val_loss:.4f}, epoch={epoch})")

    if best_state is not None:
        _ = model.load_state_dict(best_state)
        _ = model.to(device)

    print("\nEvaluating on test set...")
    test_metrics = _evaluate(model, test_loader, device)
    print(
        f"  Test loss: {test_metrics['loss']:.4f} | MAE: {test_metrics['mae']:.2f} | "
        + f"Acc: {test_metrics['accuracy'] * 100:.1f}% | F1: {test_metrics['f1']:.3f}"
    )

    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "trained", "group_promise", model_id
    )
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, "weights.pt")
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, weights_path)

    training_block: dict[str, object] = {
        "targets": [list(t) for t in targets],
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "max_group_order": max_group_order,
        "gen_random_trials": gen_random_trials,
        "gen_tabu_iters": gen_tabu_iters,
        "learning_rate": lr,
        "seed": seed,
        "weight_decay": weight_decay,
        "best_epoch": best_epoch,
    }

    info: dict[str, object] = {
        "model_type": "group_promise_predictor",
        "model_id": model_id,
        "task": "group_promise",
        "training": training_block,
        "data": {
            "produced": gen_stats["produced"],
            "num_groups": gen_stats["num_groups"],
            "per_target": per_target_summary,
        },
        "metrics": {
            "test_loss": round(test_metrics["loss"], 4),
            "test_mae": round(test_metrics["mae"], 4),
            "test_accuracy": round(test_metrics["accuracy"], 4),
            "test_precision": round(test_metrics["precision"], 4),
            "test_recall": round(test_metrics["recall"], 4),
            "test_f1": round(test_metrics["f1"], 4),
            "best_val_loss": round(best_val_loss, 4),
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
        description="Train the group-promise predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--targets", type=str, required=True, help='e.g. "3,6;3,7;3,8"'
    )
    _ = parser.add_argument("--max-group-order", type=int, default=600)
    _ = parser.add_argument("--gen-random-trials", type=int, default=800)
    _ = parser.add_argument("--gen-tabu-iters", type=int, default=800)
    _ = parser.add_argument("--gen-workers", type=int, default=1)
    _ = parser.add_argument("--epochs", type=int, default=150)
    _ = parser.add_argument("--batch-size", type=int, default=32)
    _ = parser.add_argument("--hidden-dim", type=int, default=96)
    _ = parser.add_argument("--num-layers", type=int, default=4)
    _ = parser.add_argument("--lr", type=float, default=1e-3)
    _ = parser.add_argument("--print-every", type=int, default=10)
    _ = parser.add_argument("--seed", type=int, default=42)
    _ = parser.add_argument("--weight-decay", type=float, default=1e-4)
    _ = parser.add_argument("--model-id", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _ = train(
        targets=_parse_targets(cast(str, args.targets)),
        max_group_order=cast(int, args.max_group_order),
        gen_random_trials=cast(int, args.gen_random_trials),
        gen_tabu_iters=cast(int, args.gen_tabu_iters),
        gen_workers=cast(int, args.gen_workers),
        epochs=cast(int, args.epochs),
        batch_size=cast(int, args.batch_size),
        hidden_dim=cast(int, args.hidden_dim),
        num_layers=cast(int, args.num_layers),
        lr=cast(float, args.lr),
        print_every=cast(int, args.print_every),
        seed=cast(int, args.seed),
        weight_decay=cast(float, args.weight_decay),
        model_id_override=cast("str | None", args.model_id),
    )

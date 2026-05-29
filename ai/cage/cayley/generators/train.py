"""Training script for the Cayley graph girth predictor.

Mirrors ai.cage.voltage.supervised.train: produces a multi-target predictor that can
be loaded by ai.cage.cayley.generators.search to guide tabu search.

Usage:
    uv run python -m ai.cage.cayley.generators.train --k 3 --g 8 --samples 50000
    uv run python -m ai.cage.cayley.generators.train --targets "3,6;3,7;3,8" --samples 80000
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import cast

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.cayley.generators.data_gen import generate_dataset
from ai.cage.cayley.generators.model import CayleyGirthPredictor
from ai.cage.train_utils import (
    GIRTH_NORM,
    evaluate_girth_predictor,
    make_stratified_loader,
    parse_targets,
    set_seed,
)
from ai.utils.device import get_preferred_device


def _derive_model_id(targets: list[tuple[int, int]]) -> str:
    if len(targets) == 1:
        k, g = targets[0]
        return f"cayley_girth_predictor_k{k}_g{g}"
    return "cayley_girth_predictor_multi"


def train(
    targets: list[tuple[int, int]],
    num_samples: int = 50000,
    max_group_order: int = 200,
    epochs: int = 100,
    batch_size: int = 64,
    hidden_dim: int = 64,
    num_layers: int = 4,
    lr: float = 1e-3,
    print_every: int = 10,
    seed: int = 42,
    weight_decay: float = 0.0,
    model_id_override: str | None = None,
) -> tuple[CayleyGirthPredictor, dict[str, object]]:
    set_seed(seed)
    device = torch.device(get_preferred_device())
    model_id = model_id_override or _derive_model_id(targets)

    print("=" * 60)
    print(f"Training Cayley Girth Predictor: {model_id}")
    print(f"  Targets: {targets}")
    print(f"  Device: {device} | Samples: {num_samples} | Epochs: {epochs}")
    print(f"  Hidden: {hidden_dim} | Layers: {num_layers} | LR: {lr}")
    print("=" * 60)

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
    per_target_summary = cast(dict[str, dict[str, float]], gen_stats["per_target"])
    for tkey, ts in per_target_summary.items():
        print(
            f"    target {tkey}: produced={int(ts['produced'])}, "
            f"pos_rate={ts['pos_rate']:.3f}"
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

    train_loader = make_stratified_loader(
        train_data,
        batch_size,
        lambda d: (int(cast(int, d.k)), int(cast(int, d.g_target))),
    )

    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = CayleyGirthPredictor(hidden_dim=hidden_dim, num_layers=num_layers).to(
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
            val = evaluate_girth_predictor(model, val_loader, device)
            print(
                f"Epoch {epoch:4d} | Train: {avg_train_loss:.4f} | "
                f"Val: {val['loss']:.4f} | MAE: {val['mae']:.2f} | "
                f"F1: {val['f1']:.3f}"
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
    test_metrics = evaluate_girth_predictor(model, test_loader, device)
    print(
        f"  Test loss: {test_metrics['loss']:.4f} | MAE: {test_metrics['mae']:.2f} | "
        f"Acc: {test_metrics['accuracy'] * 100:.1f}% | F1: {test_metrics['f1']:.3f}"
    )

    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "trained", "cayley_girth", model_id
    )
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, "weights.pt")
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, weights_path)

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
        "weight_decay": weight_decay,
        "best_epoch": best_epoch,
    }

    info: dict[str, object] = {
        "model_type": "cayley_girth_predictor",
        "model_id": model_id,
        "task": "cayley_girth",
        "training": training_block,
        "data": {
            "requested": gen_stats["requested"],
            "produced": gen_stats["produced"],
            "duplicates_skipped": gen_stats["duplicates_skipped"],
            "generation_attempts": gen_stats["attempts"],
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
        description="Train Cayley girth predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--k", type=int, default=None)
    _ = parser.add_argument("--g", type=int, default=None)
    _ = parser.add_argument(
        "--targets", type=str, default=None, help='e.g. "3,6;3,7;3,8"'
    )
    _ = parser.add_argument("--samples", type=int, default=50000)
    _ = parser.add_argument("--max-group-order", type=int, default=200)
    _ = parser.add_argument("--epochs", type=int, default=100)
    _ = parser.add_argument("--batch-size", type=int, default=64)
    _ = parser.add_argument("--hidden-dim", type=int, default=64)
    _ = parser.add_argument("--num-layers", type=int, default=4)
    _ = parser.add_argument("--lr", type=float, default=1e-3)
    _ = parser.add_argument("--print-every", type=int, default=10)
    _ = parser.add_argument("--seed", type=int, default=42)
    _ = parser.add_argument("--weight-decay", type=float, default=0.0)
    _ = parser.add_argument("--model-id", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    targets = parse_targets(
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
        weight_decay=cast(float, args.weight_decay),
        model_id_override=cast(str | None, args.model_id),
    )

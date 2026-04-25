"""Training script for the voltage graph girth predictor.

Usage:
    python -m ai.cage.voltage.train --k 3 --g 13 --samples 50000
    python -m ai.cage.voltage.train --k 3 --g 14 --epochs 200 --hidden-dim 128
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import cast

import torch
import torch.nn.functional as F
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]
from torch_geometric.loader import DataLoader  # pyright: ignore[reportMissingTypeStubs]

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ai.cage.voltage.data_gen import generate_dataset
from ai.cage.voltage.model import GirthPredictor
from ai.utils.device import configure_torch_device


def train(
    k: int,
    g_target: int,
    num_samples: int = 50000,
    max_group_order: int = 60,
    epochs: int = 100,
    batch_size: int = 64,
    hidden_dim: int = 64,
    num_layers: int = 4,
    lr: float = 1e-3,
    print_every: int = 10,
) -> GirthPredictor:
    """Train the girth predictor model."""

    print("=" * 60)
    print(f"Training Girth Predictor for ({k}, {g_target})-graphs")
    print("=" * 60)
    print(f"  Samples: {num_samples}")
    print(f"  Max group order: {max_group_order}")
    print(f"  Epochs: {epochs}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Layers: {num_layers}")
    print(f"  Learning rate: {lr}")
    print("=" * 60)

    # Generate dataset
    print("Generating training data...")
    all_data = generate_dataset(
        k=k,
        g_target=g_target,
        num_samples=num_samples,
        max_group_order=max_group_order,
    )

    # Split into train/val
    split = int(0.9 * len(all_data))
    train_data = all_data[:split]
    val_data = all_data[split:]

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # Analyze label distribution
    girths = [int(d.girth) for d in all_data]
    girth_ge_g = sum(1 for g in girths if g >= g_target)
    print(
        f"  Girth >= {g_target}: {girth_ge_g}/{len(all_data)} ({100 * girth_ge_g / len(all_data):.1f}%)"
    )
    print(
        f"  Girth distribution: min={min(girths)}, max={min(max(girths), 999)}, mean={sum(girths) / len(girths):.1f}"
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Model
    model = GirthPredictor(
        node_feat_dim=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_group_order=max_group_order,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        # Training
        _ = model.train()
        train_loss = 0.0
        train_count = 0

        for batch in train_loader:
            optimizer.zero_grad()

            girth_pred, class_logit = model(batch)

            # Regression loss on girth
            girth_true = cast(torch.Tensor, batch.girth).float().unsqueeze(-1)
            reg_loss = F.mse_loss(girth_pred, girth_true)

            # Classification loss
            class_true = cast(torch.Tensor, batch.girth_class).float().unsqueeze(-1)
            cls_loss = F.binary_cross_entropy_with_logits(class_logit, class_true)

            loss = reg_loss + cls_loss
            _ = loss.backward()  # pyright: ignore[reportUnknownMemberType]
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * int(girth_true.size(0))
            train_count += int(girth_true.size(0))

        avg_train_loss = train_loss / max(train_count, 1)

        # Validation
        if epoch % print_every == 0 or epoch == 1:
            _ = model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_mae = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    girth_pred, class_logit = model(batch)

                    girth_true = cast(torch.Tensor, batch.girth).float().unsqueeze(-1)
                    class_true = (
                        cast(torch.Tensor, batch.girth_class).float().unsqueeze(-1)
                    )

                    reg_loss = F.mse_loss(girth_pred, girth_true)
                    cls_loss = F.binary_cross_entropy_with_logits(
                        class_logit, class_true
                    )
                    loss = reg_loss + cls_loss

                    val_loss += loss.item() * int(girth_true.size(0))
                    val_mae += F.l1_loss(girth_pred, girth_true, reduction="sum").item()

                    # Classification accuracy
                    pred_class = (class_logit > 0).float()
                    val_correct += int((pred_class == class_true).sum().item())
                    val_total += int(girth_true.size(0))

            avg_val_loss = val_loss / max(val_total, 1)
            avg_val_mae = val_mae / max(val_total, 1)
            val_acc = 100.0 * val_correct / max(val_total, 1)

            print(
                f"Epoch {epoch:4d} | Train: {avg_train_loss:.4f} | "
                f"Val: {avg_val_loss:.4f} | MAE: {avg_val_mae:.2f} | "
                f"Cls Acc: {val_acc:.1f}%"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = cast(
                    dict[str, torch.Tensor], copy.deepcopy(model.state_dict())
                )
                print(f"  -> New best model (val_loss={best_val_loss:.4f})")

    # Restore best
    if best_state is not None:
        _ = model.load_state_dict(best_state)

    # Save model
    save_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "trained", "voltage_girth"
    )
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"girth_predictor_k{k}_g{g_target}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    print("=" * 60)

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train girth predictor for voltage graph lifts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--k", type=int, default=3, help="Target degree")
    _ = parser.add_argument("--g", type=int, default=13, help="Target girth")
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
    return parser.parse_args()


if __name__ == "__main__":
    _ = configure_torch_device()
    args = parse_args()
    _ = train(
        k=cast(int, args.k),
        g_target=cast(int, args.g),
        num_samples=cast(int, args.samples),
        max_group_order=cast(int, args.max_group_order),
        epochs=cast(int, args.epochs),
        batch_size=cast(int, args.batch_size),
        hidden_dim=cast(int, args.hidden_dim),
        num_layers=cast(int, args.num_layers),
        lr=cast(float, args.lr),
        print_every=cast(int, args.print_every),
    )

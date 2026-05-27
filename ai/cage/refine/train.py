"""Training entry-point for MoveOracle (GNN Δcost predictor).

CLI usage:
    uv run python -m ai.cage.refine.train \\
        --samples 5000 --epochs 50 --batch-size 32 \\
        --hidden-dim 64 --num-layers 3 --lr 1e-3 \\
        --cycle-lengths 3,4,5,6,7,8 --rwpe-dim 8 --seed 42

The trained model is saved to:
    ai/trained/move_oracle/move_oracle/
      weights.pt
      info.json

No --name flag: the model id is always "move_oracle" (single
(k,g)-independent model; k,g are never baked into separate weights).
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.refine.data_gen import generate_dataset
from ai.cage.refine.move_oracle import MoveOracle, save_move_oracle
from ai.utils.structural_features import structural_feature_dim


def train(
    samples: int = 5000,
    epochs: int = 50,
    batch_size: int = 32,
    hidden_dim: int = 64,
    num_layers: int = 3,
    lr: float = 1e-3,
    cycle_lengths: list[int] | None = None,
    rwpe_dim: int = 8,
    seed: int = 42,
    save_dir: Path | None = None,
    workers: int | None = None,
) -> MoveOracle:
    """Train a MoveOracle and return it.

    Parameters match the CLI flags; see module docstring.
    """
    if cycle_lengths is None:
        cycle_lengths = [3, 4, 5, 6, 7, 8]

    torch.manual_seed(seed)
    random.seed(seed)

    print(f"Generating {samples} training samples...")
    t0 = time.time()
    dataset: list[Data] = generate_dataset(
        num_samples=samples,
        cycle_lengths=cycle_lengths,
        rwpe_dim=rwpe_dim,
        seed=seed,
        workers=workers,
    )
    print(f"Generated {len(dataset)} samples in {time.time() - t0:.1f}s")

    if not dataset:
        raise RuntimeError("No training samples generated.")

    # Node feature dimension: 1 (ones) + structural extras
    extra = structural_feature_dim(cycle_lengths=cycle_lengths, rwpe_dim=rwpe_dim)
    node_feat_dim = 1 + extra

    model = MoveOracle(
        node_feat_dim=node_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(dataset)

        # Manual per-sample gradient accumulation with batch_size accumulation steps.
        # Each PyG Data has a different graph topology so we cannot stack them
        # trivially; we accumulate gradients over batch_size items instead.
        count = 0
        optimizer.zero_grad()

        for i, data_item in enumerate(dataset):
            swap_idx: torch.Tensor = data_item.swap_idx.unsqueeze(0)
            delta_target: torch.Tensor = data_item.delta.unsqueeze(0)

            delta_pred: torch.Tensor = model(data_item, swap_idx)
            loss: torch.Tensor = criterion(delta_pred, delta_target) / batch_size
            loss.backward()
            count += 1
            total_loss += loss.item() * batch_size

            if count % batch_size == 0 or i == len(dataset) - 1:
                optimizer.step()
                optimizer.zero_grad()
                count = 0

        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}  MSE loss: {avg_loss:.4f}")

    model.eval()

    if save_dir is None:
        save_dir = (
            Path(__file__).resolve().parents[2]
            / "trained"
            / "move_oracle"
            / "move_oracle"
        )

    save_move_oracle(
        model,
        save_dir,
        cycle_lengths=cycle_lengths,
        rwpe_dim=rwpe_dim,
        extra_info={
            "training": {
                "node_feat_dim": node_feat_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "samples": samples,
                "epochs": epochs,
                "lr": lr,
                "seed": seed,
                "feature_config": {
                    "cycle_lengths": cycle_lengths,
                    "rwpe_dim": rwpe_dim,
                },
            }
        },
    )
    print(f"Model saved to {save_dir}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MoveOracle GNN Δcost predictor")
    _ = parser.add_argument("--samples", type=int, default=5000)
    _ = parser.add_argument("--epochs", type=int, default=50)
    _ = parser.add_argument("--batch-size", type=int, default=32)
    _ = parser.add_argument("--hidden-dim", type=int, default=64)
    _ = parser.add_argument("--num-layers", type=int, default=3)
    _ = parser.add_argument("--lr", type=float, default=1e-3)
    _ = parser.add_argument("--cycle-lengths", type=str, default="3,4,5,6,7,8")
    _ = parser.add_argument("--rwpe-dim", type=int, default=8)
    _ = parser.add_argument("--seed", type=int, default=42)
    _ = parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Data-gen worker count; default = os.cpu_count(). Set 1 to disable.",
    )
    args = parser.parse_args()

    cycle_lengths = [int(x.strip()) for x in args.cycle_lengths.split(",")]

    _ = train(
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        cycle_lengths=cycle_lengths,
        rwpe_dim=args.rwpe_dim,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

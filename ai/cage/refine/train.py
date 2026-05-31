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

Training objective: combined MSE + pairwise margin-ranking loss.
The ranking loss pushes correct orderings into the top-5 (the only thing
TabuRefiner uses), rather than demanding perfect absolute Δcost.

Validation: a held-out val split (val_fraction) is evaluated each epoch on:
  - val MSE
  - top-1-in-top5: fraction of graphs where the truly-best swap is among
    the model's top-5 predictions (a proxy for NDCG@5).

Early stopping on val_top1_in_top5 (maximize).
"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.refine.data_gen import generate_dataset
from ai.cage.refine.move_oracle import MoveOracle, save_move_oracle
from ai.utils.structural_features import structural_feature_dim

# Margin for pairwise ranking loss: we want score(better_swap) to be at
# least RANK_MARGIN below score(worse_swap) (lower = more promising).
_RANK_MARGIN = 0.1

# Weight of the ranking loss relative to MSE.
_RANK_LOSS_WEIGHT = 1.0


def _pairwise_ranking_loss(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> torch.Tensor:
    """Pairwise margin-ranking loss over per-graph swap sets.

    For each graph with multiple swaps we form all pairs (i, j) where
    target_i < target_j (swap i is genuinely better).  We penalise cases
    where the model does not rank swap i strictly below swap j by at least
    _RANK_MARGIN.

    This encourages correct relative ordering of swaps without requiring
    perfect absolute Δcost prediction.

    Parameters
    ----------
    preds:
        List of 1-D tensors, one per graph, each of shape [n_swaps_i].
    targets:
        Matching list of 1-D tensors with true Δcost values.

    Returns
    -------
    Scalar mean ranking loss.
    """
    losses: list[torch.Tensor] = []
    for pred, tgt in zip(preds, targets):
        if pred.size(0) < 2:
            continue
        # i better than j  <=>  tgt[i] < tgt[j]
        # we want pred[i] < pred[j] - margin
        # loss = max(0, pred[i] - pred[j] + margin)
        pred_i = pred.unsqueeze(1)  # [n, 1]
        pred_j = pred.unsqueeze(0)  # [1, n]
        tgt_i = tgt.unsqueeze(1)  # [n, 1]
        tgt_j = tgt.unsqueeze(0)  # [1, n]
        mask = tgt_i < tgt_j  # [n, n] bool, True where i is better
        if not mask.any():
            continue
        margin_violation = torch.clamp(
            pred_i - pred_j + _RANK_MARGIN, min=0.0
        )  # [n, n]
        losses.append(margin_violation[mask].mean())

    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def _group_by_graph(
    dataset: list[Data],
) -> dict[int, list[Data]]:
    """Group dataset items by a graph identity key (num_nodes, num_edges).

    Items that share the same (num_nodes, num_edges) are NOT necessarily the
    same graph, but grouping this way is a cheap approximation that is only
    used to build per-graph swap sets for the ranking loss.  We group by
    (g_target, num_nodes, edge hash) for better accuracy.
    """
    groups: dict[int, list[Data]] = defaultdict(list)
    for item in dataset:
        g_target = int(item.g_target) if hasattr(item, "g_target") else 0
        n = int(cast_int(item.num_nodes))
        # Use a cheap hash of edge connectivity for grouping
        ei = item.edge_index
        if ei is not None and ei.numel() > 0:
            # sum of edge index values — fast, not collision-free but good enough
            edge_hash = int(ei.sum().item()) % (2**20)
        else:
            edge_hash = 0
        key = hash((g_target, n, edge_hash))
        groups[key].append(item)
    return groups


def cast_int(v: object) -> int:
    return int(cast(int, v))


def _compute_val_metrics(
    model: MoveOracle,
    val_data: list[Data],
    criterion: nn.MSELoss,
) -> dict[str, float]:
    """Evaluate val MSE and top-1-in-top5 on the validation set.

    top1_in_top5: fraction of graphs (grouped by swap sets) where the swap
    with the lowest true Δcost appears in the model's predicted top-5.
    """
    model.eval()
    total_mse = 0.0
    n_graphs = 0
    n_top1_in_top5 = 0

    groups = _group_by_graph(val_data)

    with torch.no_grad():
        for items in groups.values():
            if not items:
                continue
            preds_list: list[float] = []
            targets_list: list[float] = []
            for item in items:
                swap_idx = item.swap_idx.unsqueeze(0)
                tgt = item.delta.unsqueeze(0)
                pred = model(item, swap_idx)
                mse_val = criterion(pred, tgt).item()
                total_mse += mse_val
                preds_list.append(float(pred.item()))
                targets_list.append(float(tgt.item()))

            # top-1-in-top5 for this graph
            if len(preds_list) >= 2:
                best_true_idx = int(
                    min(range(len(targets_list)), key=lambda i: targets_list[i])
                )
                # argsort preds ascending (lower = more promising)
                sorted_pred_idx = sorted(
                    range(len(preds_list)), key=lambda i: preds_list[i]
                )
                top5_idx = set(sorted_pred_idx[:5])
                n_top1_in_top5 += 1 if best_true_idx in top5_idx else 0
                n_graphs += 1

    val_mse = total_mse / max(len(val_data), 1)
    top1_in_top5 = n_top1_in_top5 / max(n_graphs, 1)
    return {"val_mse": val_mse, "val_top1_in_top5": top1_in_top5}


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
    val_fraction: float = 0.1,
    patience: int = 8,
    lift_fraction: float = 0.7,
    backbone: str = "gine",
    r: int = 3,
) -> MoveOracle:
    """Train a MoveOracle and return it.

    Parameters match the CLI flags; see module docstring.
    """
    if cycle_lengths is None:
        cycle_lengths = [3, 4, 5, 6, 7, 8]

    torch.manual_seed(seed)
    random.seed(seed)

    # Resolve default save directory based on backbone so loopy never
    # clobbers the gine model.
    if save_dir is None:
        trained_root = Path(__file__).resolve().parents[2] / "trained"
        if backbone == "loopy":
            save_dir = (
                trained_root / f"move_oracle_loopy_r{r}" / f"move_oracle_loopy_r{r}"
            )
        else:
            save_dir = trained_root / "move_oracle" / "move_oracle"

    loopy_r: int | None = r if backbone == "loopy" else None

    print(
        f"Generating {samples} training samples "
        f"(lift_fraction={lift_fraction}, backbone={backbone}"
        + (f", r={r}" if backbone == "loopy" else "")
        + ")..."
    )
    t0 = time.time()
    dataset: list[Data] = generate_dataset(
        num_samples=samples,
        cycle_lengths=cycle_lengths,
        rwpe_dim=rwpe_dim,
        seed=seed,
        workers=workers,
        lift_fraction=lift_fraction,
        r=loopy_r,
    )
    print(f"Generated {len(dataset)} samples in {time.time() - t0:.1f}s")

    if not dataset:
        raise RuntimeError("No training samples generated.")

    # Train/val split
    random.shuffle(dataset)
    n_val = max(1, int(len(dataset) * val_fraction))
    val_data = dataset[:n_val]
    train_data = dataset[n_val:]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    # Count how many samples came from lifts (approximated: non-random graphs
    # are identifiable since lift sizes are typically multiples of group order)
    # Just report the intention via lift_fraction.

    # Node feature dimension: 1 (ones) + structural extras
    extra = structural_feature_dim(cycle_lengths=cycle_lengths, rwpe_dim=rwpe_dim)
    node_feat_dim = 1 + extra

    model = MoveOracle(
        node_feat_dim=node_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        backbone=backbone,
        r=r,
    )

    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Group training data by graph for ranking loss
    train_groups = _group_by_graph(train_data)

    best_top1_in_top5 = -1.0
    best_val_mse = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] = {}
    patience_counter = 0

    val_history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        random.shuffle(train_data)

        count = 0
        optimizer.zero_grad()

        for i, data_item in enumerate(train_data):
            swap_idx: torch.Tensor = data_item.swap_idx.unsqueeze(0)
            delta_target: torch.Tensor = data_item.delta.unsqueeze(0)

            delta_pred: torch.Tensor = model(data_item, swap_idx)
            mse_loss: torch.Tensor = criterion(delta_pred, delta_target) / batch_size
            mse_loss.backward()
            count += 1
            total_loss += mse_loss.item() * batch_size

            if count % batch_size == 0 or i == len(train_data) - 1:
                optimizer.step()
                optimizer.zero_grad()
                count = 0

        # Ranking loss pass: one pass per group (not per individual sample)
        rank_loss_total = 0.0
        rank_groups_processed = 0
        optimizer.zero_grad()
        rank_batch_count = 0
        for group_items in train_groups.values():
            if len(group_items) < 2:
                continue
            preds_g: list[torch.Tensor] = []
            targets_g: list[torch.Tensor] = []
            for item in group_items:
                si = item.swap_idx.unsqueeze(0)
                pred_g = model(item, si).squeeze(0)
                preds_g.append(pred_g)
                targets_g.append(item.delta)

            preds_t = torch.stack(preds_g)  # [n_swaps]
            targets_t = torch.stack(targets_g)  # [n_swaps]

            rl: torch.Tensor = _pairwise_ranking_loss([preds_t], [targets_t])
            if rl.requires_grad:
                (rl * _RANK_LOSS_WEIGHT / batch_size).backward()
                rank_loss_total += float(rl.item())
                rank_groups_processed += 1
                rank_batch_count += 1
                if rank_batch_count % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    rank_batch_count = 0

        if rank_batch_count > 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_mse = total_loss / max(len(train_data), 1)
        avg_rank = rank_loss_total / max(rank_groups_processed, 1)

        # Validation
        val_metrics = _compute_val_metrics(model, val_data, criterion)
        val_history.append(val_metrics)

        log_every = max(1, epochs // 10)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}  "
                f"train_mse={avg_mse:.4f}  rank_loss={avg_rank:.4f}  "
                f"val_mse={val_metrics['val_mse']:.4f}  "
                f"val_top1_in_top5={val_metrics['val_top1_in_top5']:.3f}"
            )

        # Early stopping: maximize val_top1_in_top5
        if val_metrics["val_top1_in_top5"] > best_top1_in_top5 or (
            val_metrics["val_top1_in_top5"] == best_top1_in_top5
            and val_metrics["val_mse"] < best_val_mse
        ):
            best_top1_in_top5 = val_metrics["val_top1_in_top5"]
            best_val_mse = val_metrics["val_mse"]
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(best val_top1_in_top5={best_top1_in_top5:.3f} at epoch {best_epoch})"
            )
            break

    # Restore best weights
    if best_state:
        _ = model.load_state_dict(best_state)
    model.eval()

    # Final val metrics with best model
    final_val = _compute_val_metrics(model, val_data, criterion)
    print(
        f"Final: val_mse={final_val['val_mse']:.4f}  "
        f"val_top1_in_top5={final_val['val_top1_in_top5']:.3f}  "
        f"(best epoch={best_epoch})"
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
                "lift_fraction": lift_fraction,
                "val_fraction": val_fraction,
                "best_epoch": best_epoch,
                "feature_config": {
                    "cycle_lengths": cycle_lengths,
                    "rwpe_dim": rwpe_dim,
                },
            },
            "val_metrics": final_val,
            "val_history": val_history,
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
    _ = parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of dataset to hold out for validation.",
    )
    _ = parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early-stopping patience (epochs without val_top1_in_top5 improvement).",
    )
    _ = parser.add_argument(
        "--lift-fraction",
        type=float,
        default=0.7,
        help="Fraction of training samples to source from voltage lift near-misses.",
    )
    _ = parser.add_argument(
        "--backbone",
        type=str,
        default="gine",
        choices=["gine", "loopy"],
        help="GNN backbone: 'gine' (default) or 'loopy' (Loopy_GNN, cycle-aware).",
    )
    _ = parser.add_argument(
        "--r",
        type=int,
        default=3,
        help="r-neighborhood radius for Loopy backbone (ignored for gine).",
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
        val_fraction=args.val_fraction,
        patience=args.patience,
        lift_fraction=args.lift_fraction,
        backbone=args.backbone,
        r=args.r,
    )


if __name__ == "__main__":
    main()

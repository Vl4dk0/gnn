"""Training script for the voltage graph girth predictor.

The predictor is trained as a single (k, g)-independent model: (k, g) are fed
in as context features, never baked into separate weights. Always pass the
full target grid via --targets.

Usage:
    uv run python -m ai.cage.voltage.supervised.train \\
        --targets "3,5;3,6;3,7;4,5;4,6;5,5;5,6" --samples 200000
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader  # pyright: ignore[reportMissingTypeStubs]

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from ai.cage.utils import (
    evaluate_girth_predictor,
    make_stratified_loader,
    normalize_target,
    parse_targets,
    save_predictor_artifacts,
    set_seed,
)
from ai.cage.voltage.supervised.data_gen import generate_dataset
from ai.cage.voltage.supervised.model import GirthPredictor
from ai.utils.device import get_preferred_device
from ai.utils.structural_features import structural_feature_dim


def _derive_model_id(targets: list[tuple[int, int]], kind: str) -> str:
    """Save-folder name for the predictor.

    The predictor is always a single (k, g)-independent model trained over a
    grid of targets, so the id is fixed per target kind rather than encoding any
    one target: "girth_predictor" for girth, "tabu_predictor" for tabu_cost.
    """
    _ = targets
    return "tabu_predictor" if kind == "tabu_cost" else "girth_predictor"


def _evaluate_per_target(
    model: GirthPredictor,
    test_data: list,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    device: torch.device,
    batch_size: int,
    kind: str,
) -> dict[str, dict[str, float]]:
    """Group test set by (k, g_target) and compute per-target metrics."""
    groups: dict[tuple[int, int], list] = {}  # pyright: ignore[reportMissingTypeArgument, reportUnknownVariableType]
    for d in test_data:  # pyright: ignore[reportUnknownVariableType]
        key = (int(cast(int, d.k)), int(cast(int, d.g_target)))
        groups.setdefault(key, []).append(d)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

    out: dict[str, dict[str, float]] = {}
    for (k, g), items in groups.items():  # pyright: ignore[reportUnknownVariableType]
        loader = DataLoader(items, batch_size=batch_size)  # pyright: ignore[reportUnknownArgumentType]
        m = evaluate_girth_predictor(model, loader, device, kind)
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
    weight_decay: float = 0.0,
    model_id_override: str | None = None,
    cycle_lengths: list[int] | None = None,
    rwpe_dim: int = 0,
    workers: int | None = None,
    pos_weight: float = 5.0,
    kind: str = "girth",
) -> tuple[GirthPredictor, dict[str, object]]:
    """Train the predictor model. Returns (model, info_dict).

    ``kind`` is "girth" (default; regresses the lift's girth, ranked
    DESCENDING in search) or "tabu_cost" (regresses the dense short-walk cost
    the tabu search minimizes, ranked ASCENDING). The architecture, features,
    and loss form are identical; only the label and its normalization differ.
    """

    set_seed(seed)
    device = torch.device(get_preferred_device())

    model_id = model_id_override or _derive_model_id(targets, kind)

    def _is_pos(label: int, g_target: int) -> bool:
        """Achieves-target predicate per kind (tabu cost 0 == girth >= g)."""
        return label == 0 if kind == "tabu_cost" else label >= g_target

    print("=" * 60)
    print(f"Training {kind} predictor: {model_id}")
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
    print(f"  Pos weight:       {pos_weight}")
    print(f"  Seed:             {seed}")
    print(f"  Cycle lengths:    {cycle_lengths or []}")
    print(f"  RWPE dim:         {rwpe_dim}")
    print("=" * 60)

    print("Generating training data...")
    all_data, gen_stats = generate_dataset(
        targets=targets,
        num_samples=num_samples,
        max_group_order=max_group_order,
        seed=seed,
        cycle_lengths=cycle_lengths,
        rwpe_dim=rwpe_dim,
        workers=workers,
        kind=kind,
    )
    print(
        f"  Produced {gen_stats['produced']}/{gen_stats['requested']} unique samples "
        f"in {gen_stats['attempts']} attempts "
        f"({gen_stats['duplicates_skipped']} duplicates skipped)"
    )
    produced_n = int(cast(int, gen_stats["produced"]))
    requested_n = int(cast(int, gen_stats["requested"]))
    if produced_n < 0.95 * requested_n:
        print(
            f"  WARNING: produced only {produced_n}/{requested_n} samples "
            f"({100 * produced_n / max(requested_n, 1):.1f}%). "
            "Increase --max-group-order or --samples ceiling, "
            "or reduce target overlap."
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
        pos = sum(
            1
            for d in data_list  # pyright: ignore[reportUnknownVariableType]
            if _is_pos(int(cast(int, d.girth)), int(cast(int, d.g_target)))
        )
        return total, pos, pos / max(total, 1)

    train_n, train_pos, train_rate = _stats(train_data)
    val_n, val_pos, val_rate = _stats(val_data)
    test_n, test_pos, test_rate = _stats(test_data)
    print(
        f"  Train: {train_n} ({train_pos} pos, {train_rate * 100:.1f}%) | "
        f"Val: {val_n} ({val_pos} pos, {val_rate * 100:.1f}%) | "
        f"Test: {test_n} ({test_pos} pos, {test_rate * 100:.1f}%)"
    )

    # Stratify over (k, g_target, achieves-target) so the rare positive class
    # (girth >= g_target, equivalently tabu cost == 0) is not drowned out within
    # each (k, g) bucket. Without the label component, hard targets like (5, 7)
    # collapse to all-negative.
    train_loader = make_stratified_loader(
        train_data,
        batch_size,
        lambda d: (
            int(cast(int, d.k)),
            int(cast(int, d.g_target)),
            int(_is_pos(int(cast(int, d.girth)), int(cast(int, d.g_target)))),
        ),
    )

    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Model — base node_feat_dim is 2, plus any structural feature columns
    base_node_feat_dim = 2
    extra_dim = structural_feature_dim(cycle_lengths=cycle_lengths, rwpe_dim=rwpe_dim)
    node_feat_dim = base_node_feat_dim + extra_dim
    model = GirthPredictor(
        node_feat_dim=node_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_group_order=max_group_order,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

            pred = model(batch)

            label_true = cast(torch.Tensor, batch.girth).float().unsqueeze(-1)
            g_target_t = cast(torch.Tensor, batch.g_target).float().unsqueeze(-1)

            # Label-aware weighting: upweight the rare positive class (lifts that
            # achieve the target — girth >= g_target, equivalently tabu cost 0)
            # so the regressor does not collapse onto the all-negative majority
            # on hard, heavily imbalanced targets.
            if kind == "tabu_cost":
                is_pos = (label_true < 0.5).float()
            else:
                is_pos = (label_true >= g_target_t).float()
            weight = 1.0 + (pos_weight - 1.0) * is_pos
            per_elem = F.mse_loss(
                normalize_target(pred, kind),
                normalize_target(label_true, kind),
                reduction="none",
            )
            loss = (weight * per_elem).sum() / weight.sum()

            _ = loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            n_b = int(label_true.size(0))
            train_loss += float(loss.item()) * n_b
            train_count += n_b

        scheduler.step()
        avg_train_loss = train_loss / max(train_count, 1)

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            val = evaluate_girth_predictor(model, val_loader, device, kind)
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
    test_metrics = evaluate_girth_predictor(model, test_loader, device, kind)
    print(
        f"  Test loss: {test_metrics['loss']:.4f} | "
        f"MAE: {test_metrics['mae']:.2f} | "
        f"Acc: {test_metrics['accuracy'] * 100:.1f}% | "
        f"Precision: {test_metrics['precision']:.3f} | "
        f"Recall: {test_metrics['recall']:.3f} | "
        f"F1: {test_metrics['f1']:.3f}"
    )

    test_per_target = _evaluate_per_target(model, test_data, device, batch_size, kind)
    if len(test_per_target) > 1:
        print("  Per-target test metrics:")
        for tkey, m in test_per_target.items():
            print(
                f"    {tkey}: n={int(m['n'])}, mae={m['mae']:.2f}, "
                f"acc={m['accuracy'] * 100:.1f}%, f1={m['f1']:.3f}"
            )

    save_dir = (
        Path(__file__).resolve().parents[3] / "trained" / "voltage_girth" / model_id
    )

    # For backward compat, also surface single-target k/g_target fields when applicable
    primary_k, primary_g = targets[0] if len(targets) == 1 else (None, None)

    training_block: dict[str, object] = {
        "targets": [list(t) for t in targets],
        "samples": num_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "node_feat_dim": node_feat_dim,
        "max_group_order": max_group_order,
        "learning_rate": lr,
        "pos_weight": pos_weight,
        "seed": seed,
        "weight_decay": weight_decay,
        "best_epoch": best_epoch,
        "feature_config": {
            "cycle_lengths": cycle_lengths if cycle_lengths else [],
            "rwpe_dim": rwpe_dim,
        },
    }
    if primary_k is not None:
        training_block["k"] = primary_k
        training_block["g_target"] = primary_g

    model_type = (
        "voltage_tabu_predictor" if kind == "tabu_cost" else "voltage_girth_predictor"
    )
    info: dict[str, object] = {
        "model_type": model_type,
        "model_id": model_id,
        "task": "voltage_tabu" if kind == "tabu_cost" else "voltage_girth",
        "kind": kind,
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

    weights_path, info_path = save_predictor_artifacts(model, save_dir, info)

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
        "--targets",
        type=str,
        default=None,
        required=True,
        help='Target grid, e.g. "3,5;3,6;4,5" (single k,g-independent model).',
    )
    _ = parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override the auto-derived model id (save folder name)",
    )
    _ = parser.add_argument(
        "--kind",
        type=str,
        default="girth",
        choices=["girth", "tabu_cost"],
        help="Regression target: 'girth' (default; ranked descending in search) "
        "or 'tabu_cost' (the short-walk cost the tabu search minimizes; ranked "
        "ascending). Default keeps girth_predictor behavior unchanged.",
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
        "--pos-weight",
        type=float,
        default=5.0,
        help="Loss multiplier for the positive class (girth >= g_target). >1 "
        "upweights rare positives on hard, imbalanced targets.",
    )
    _ = parser.add_argument(
        "--print-every", type=int, default=10, help="Print every N epochs"
    )
    _ = parser.add_argument("--seed", type=int, default=42, help="Random seed")
    _ = parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay (use 1e-4 to combat overfitting)",
    )
    _ = parser.add_argument(
        "--cycle-lengths",
        type=str,
        default="3,4,5,6,7,8",
        help='Comma-separated cycle lengths added as node features (structural features are standard). Pass "" to disable.',
    )
    _ = parser.add_argument(
        "--rwpe-dim",
        type=int,
        default=8,
        help="Dimension K for Random-Walk Positional Encoding (structural feature, standard). 0 = disabled.",
    )
    _ = parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Data-gen worker count; default = os.cpu_count(). Set 1 to disable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    targets = parse_targets(cast(str | None, args.targets))
    raw_cycle_lengths = cast(str, args.cycle_lengths).strip()
    parsed_cycle_lengths: list[int] | None = (
        [int(x) for x in raw_cycle_lengths.split(",") if x.strip()]
        if raw_cycle_lengths
        else None
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
        cycle_lengths=parsed_cycle_lengths,
        rwpe_dim=cast(int, args.rwpe_dim),
        workers=cast("int | None", args.workers),
        pos_weight=cast(float, args.pos_weight),
        kind=cast(str, args.kind),
    )

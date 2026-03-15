"""Training script for min_cycle prediction models.

Usage:
    python -m ai.min_cycle.train --model gin --name v1
    python -m ai.min_cycle.train --model gcn --name baseline --epochs 10000
    python -m ai.min_cycle.train --model sage --name v1 --hidden-dim 128
    python -m ai.min_cycle.train --model loopy --name r3_v1 --r 3 --epochs 5000
"""

import argparse
import os
import sys
from typing import cast

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ai.models import MODEL_CLASSES
from ai.models.base import BaseGNN
from ai.models.loopy import Loopy_GNN
from ai.registry import save_model, model_exists, list_model_types
from ai.min_cycle.functions.graph_service import get_min_cycle
from ai.utils.device import configure_torch_device
from ai.utils.model_naming import build_name
from ai.utils.r_neighborhood import apply_r_neighborhood
from backend.utils import generate_random_graph

_ = load_dotenv()


def generate_random_graph_data(
    num_nodes_range: tuple[int, int] | None = None,
    p_range: tuple[float, float] | None = None,
    input_dim: int = 4,
    r: int | None = None,
) -> Data:
    """
    Generate a random graph for training using centralized graph generation.

    Args:
        num_nodes_range: Range of number of nodes (min, max) - defaults from env vars
        p_range: Range of edge probability for Erdős-Rényi model - defaults from env vars
        input_dim: Number of input features per node
        r: If provided, apply r-neighborhood transform for Loopy GNN

    Returns:
        PyTorch Geometric Data object
    """
    # Generate graph using centralized utility
    G = generate_random_graph(num_nodes_range=num_nodes_range, p_range=p_range)

    num_nodes: int = len(G.nodes())

    # Get smallest cycle containing each node (true labels)
    cycles: torch.Tensor = torch.tensor(
        [get_min_cycle(G, i) for i in range(num_nodes)], dtype=torch.float
    )

    # Create edge index for PyTorch Geometric
    edge_index: list[list[int]] = []
    for u, v in G.edges():
        edge_index.append([u, v])
        edge_index.append([v, u])

    edge_index_tensor: torch.Tensor
    if len(edge_index) == 0:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Build node features based on input_dim
    x: torch.Tensor
    if input_dim == 1:
        # Simple constant feature
        x = torch.ones(num_nodes, 1)
    else:
        # Rich node features for better learning
        # Feature 1: Normalized node index (0 to 1)
        node_idx_feature: torch.Tensor = torch.arange(
            num_nodes, dtype=torch.float
        ).unsqueeze(1) / max(num_nodes - 1, 1)

        # Feature 2: Random embedding (helps distinguish nodes)
        random_feature: torch.Tensor = torch.randn(num_nodes, 2)

        # Feature 3: Clustering coefficient estimate
        clustering_feature: torch.Tensor = torch.rand(num_nodes, 1)

        # Combine all features
        x = torch.cat(
            [node_idx_feature, random_feature, clustering_feature], dim=1
        )  # Shape: [num_nodes, 4]

    # Create PyTorch Geometric Data object
    data: Data = Data(x=x, edge_index=edge_index_tensor, y=cycles)

    # Apply r-neighborhood transform for Loopy GNN
    if r is not None:
        data = apply_r_neighborhood(data, r=r)

    return data


def train_step(model: BaseGNN, optimizer: torch.optim.Optimizer, data: Data) -> float:
    """Train the model on a single graph."""
    _ = model.train()
    optimizer.zero_grad()

    out: torch.Tensor = cast(torch.Tensor, model(data)).squeeze()
    loss: torch.Tensor = F.mse_loss(out, cast(torch.Tensor, data.y))

    _ = loss.backward()  # pyright: ignore[reportUnknownMemberType]

    # Gradient clipping to prevent exploding gradients
    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return cast(float, loss.item())


def evaluate_model(
    model: BaseGNN, num_test_graphs: int = 100, input_dim: int = 4, r: int | None = None
) -> dict[str, float]:
    """
    Evaluate the model on test graphs.

    Args:
        model: The model to evaluate
        num_test_graphs: Number of test graphs
        input_dim: Input feature dimension
        r: If provided, apply r-neighborhood transform for Loopy GNN

    Returns:
        Dictionary with evaluation metrics (mse, mae, accuracy)
    """
    _ = model.eval()
    total_loss: float = 0.0
    total_mae: float = 0.0
    total_correct: int = 0
    total_predictions: int = 0

    with torch.no_grad():
        for _ in range(num_test_graphs):
            data: Data = generate_random_graph_data(input_dim=input_dim, r=r)
            out: torch.Tensor = cast(torch.Tensor, model(data)).squeeze()

            predictions_rounded: torch.Tensor = torch.round(out)

            y = cast(torch.Tensor, data.y)
            loss: torch.Tensor = F.mse_loss(predictions_rounded, y)
            mae: torch.Tensor = F.l1_loss(predictions_rounded, y)
            correct = cast(int, (predictions_rounded == y).sum().item())

            total_loss += loss.item()
            total_mae += mae.item()
            total_correct += correct
            total_predictions += y.numel()

    return {
        "mse": total_loss / num_test_graphs,
        "mae": total_mae / num_test_graphs,
        "accuracy": (total_correct / total_predictions) * 100
        if total_predictions > 0
        else 0,
    }


def train_gnn(
    model_type: str,
    model_name: str,
    num_epochs: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    lr: float,
    print_every: int,
    eval_every: int,
    graphs_per_epoch: int,
    input_dim: int,
    force: bool = False,
    r: int = 3,
) -> BaseGNN:
    """
    Main training function.

    Args:
        model_type: Type of model ('gcn', 'sage', 'gin', 'loopy')
        model_name: Name for this trained model (e.g., 'v1', 'baseline')
        num_epochs: Number of training epochs
        hidden_dim: Hidden dimension size
        num_layers: Number of GNN layers
        dropout: Dropout probability
        lr: Learning rate
        print_every: Print progress every N epochs
        eval_every: Evaluate every N epochs
        graphs_per_epoch: Number of graphs to train on per epoch
        input_dim: Number of input features per node
        force: Overwrite existing model if exists
        r: r-neighborhood radius for Loopy GNN

    Returns:
        Trained model
    """
    # Only use r for loopy models
    r_for_training: int | None = r if model_type == "loopy" else None

    if model_name == model_type or model_name.startswith(f"{model_type}_"):
        model_id = model_name
    else:
        model_id = f"{model_type}_{model_name}"

    # Check if model already exists
    if model_exists("min_cycle", model_id) and not force:
        print(f"Error: Model '{model_id}' already exists for task 'min_cycle'.")
        print("Use --force to overwrite, or choose a different --name.")
        sys.exit(1)

    # Get model class
    if model_type not in MODEL_CLASSES:
        print(f"Error: Unknown model type '{model_type}'.")
        print(f"Available: {list_model_types()}")
        sys.exit(1)

    model_class: type[BaseGNN] = MODEL_CLASSES[model_type]

    print("=" * 60)
    print(f"Training {model_type.upper()} for Min Cycle Prediction")
    print(f"Model ID: {model_id}")
    print("=" * 60)
    print("Configuration:")
    print(f"  - Model Type: {model_type}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Hidden Dim: {hidden_dim}")
    print(f"  - Num Layers: {num_layers}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Input Dim: {input_dim}")
    print(f"  - Graphs per Epoch: {graphs_per_epoch}")
    if r_for_training is not None:
        print(
            f"  - r-neighborhood: {r_for_training} (cycles up to {r_for_training + 2})"
        )
    print("=" * 60)

    # Initialize model
    model: BaseGNN
    if model_type == "loopy":
        model = cast(type[Loopy_GNN], model_class)(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout=dropout,
            r=r,
        )
    else:
        model = model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout=dropout,
        )
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)

    best_accuracy: float = 0.0
    best_mae: float = float("inf")
    best_model_state: dict[str, torch.Tensor] | None = None
    best_epoch: int = 0

    # Training loop
    for epoch in range(1, num_epochs + 1):
        epoch_loss: float = 0.0

        for _ in range(graphs_per_epoch):
            data: Data = generate_random_graph_data(
                input_dim=input_dim, r=r_for_training
            )
            loss: float = train_step(model, optimizer, data)
            epoch_loss += loss

        epoch_loss /= graphs_per_epoch

        # Evaluate
        if epoch % eval_every == 0 or epoch == 1:
            metrics: dict[str, float] = evaluate_model(
                model, num_test_graphs=100, input_dim=input_dim, r=r_for_training
            )

            if epoch % print_every == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:5d} | Loss: {epoch_loss:.4f} | MSE: {metrics['mse']:.4f} | MAE: {metrics['mae']:.4f} | Acc: {metrics['accuracy']:.2f}%"
                )

            # Check if this is the best model
            is_better: bool = (metrics["accuracy"] > best_accuracy) or (
                metrics["accuracy"] == best_accuracy and metrics["mae"] < best_mae
            )

            if is_better:
                best_accuracy = metrics["accuracy"]
                best_mae = metrics["mae"]
                best_model_state = cast(
                    dict[str, torch.Tensor], model.state_dict().copy()
                )
                best_epoch = epoch
                print(f"  → New best! Acc: {best_accuracy:.2f}%, MAE: {best_mae:.4f}")

    # Restore best model
    if best_model_state is not None:
        _ = model.load_state_dict(best_model_state)

    # Final evaluation
    print("=" * 60)
    print("Training complete! Final evaluation on 200 test graphs:")
    final_metrics: dict[str, float] = evaluate_model(
        model, num_test_graphs=200, input_dim=input_dim, r=r_for_training
    )
    print(f"  MSE: {final_metrics['mse']:.4f}")
    print(f"  MAE: {final_metrics['mae']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.2f}%")
    print("=" * 60)

    # Save the best model using registry
    training_info: dict[str, str | int | float] = {
        "epochs": num_epochs,
        "best_epoch": best_epoch,
        "learning_rate": lr,
        "graphs_per_epoch": graphs_per_epoch,
    }
    if r_for_training is not None:
        training_info["r"] = r_for_training

    save_path: str = str(
        save_model(
            model=model,
            task="min_cycle",
            model_id=model_id,
            metrics={
                "mse": round(best_mae**2, 4),  # Approximate MSE from MAE
                "mae": round(best_mae, 4),
                "accuracy": round(best_accuracy, 2),
            },
            training_info=training_info,
        )
    )

    print(f"Model saved to: {save_path}")
    print("=" * 60)

    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train a GNN model for min cycle prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    _ = parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=list(MODEL_CLASSES.keys()),
        help="Model type to train",
    )
    _ = parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Optional model version name (if omitted, generated automatically)",
    )

    # Loopy-specific parameters
    _ = parser.add_argument(
        "--r",
        type=int,
        default=3,
        help="r-neighborhood radius for Loopy GNN (detects cycles up to r+2)",
    )

    # Training parameters
    _ = parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.getenv("TRAINING_NUM_EPOCHS", 5000)),
        help="Number of training epochs",
    )
    _ = parser.add_argument(
        "--hidden-dim",
        type=int,
        default=int(os.getenv("TRAINING_HIDDEN_DIM", 64)),
        help="Hidden layer dimension",
    )
    _ = parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of GNN layers",
    )
    _ = parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability",
    )
    _ = parser.add_argument(
        "--lr",
        type=float,
        default=float(os.getenv("TRAINING_LEARNING_RATE", 0.001)),
        help="Learning rate",
    )
    _ = parser.add_argument(
        "--input-dim",
        type=int,
        default=4,
        help="Input feature dimension (1 for constant, 4 for rich features)",
    )
    _ = parser.add_argument(
        "--graphs-per-epoch",
        type=int,
        default=int(os.getenv("TRAINING_GRAPHS_PER_EPOCH", 10)),
        help="Number of graphs to train on per epoch",
    )
    _ = parser.add_argument(
        "--print-every",
        type=int,
        default=int(os.getenv("TRAINING_PRINT_EVERY", 100)),
        help="Print progress every N epochs",
    )
    _ = parser.add_argument(
        "--eval-every",
        type=int,
        default=int(os.getenv("TRAINING_EVAL_EVERY", 20)),
        help="Evaluate every N epochs",
    )
    _ = parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing model if it exists",
    )

    return parser.parse_args()


if __name__ == "__main__":
    _ = configure_torch_device()
    args = parse_args()
    model_arg = cast(str, args.model)
    name_arg = cast(str | None, args.name)
    r_arg = cast(int, args.r)
    epochs_arg = cast(int, args.epochs)
    hidden_dim_arg = cast(int, args.hidden_dim)
    num_layers_arg = cast(int, args.num_layers)
    dropout_arg = cast(float, args.dropout)
    lr_arg = cast(float, args.lr)
    print_every_arg = cast(int, args.print_every)
    eval_every_arg = cast(int, args.eval_every)
    graphs_per_epoch_arg = cast(int, args.graphs_per_epoch)
    input_dim_arg = cast(int, args.input_dim)
    force_arg = cast(bool, args.force)

    auto_extra = f"r{r_arg}" if model_arg == "loopy" else ""
    resolved_name = name_arg or build_name(model_arg, extra=auto_extra)

    _ = train_gnn(
        model_type=model_arg,
        model_name=resolved_name,
        num_epochs=epochs_arg,
        hidden_dim=hidden_dim_arg,
        num_layers=num_layers_arg,
        dropout=dropout_arg,
        lr=lr_arg,
        print_every=print_every_arg,
        eval_every=eval_every_arg,
        graphs_per_epoch=graphs_per_epoch_arg,
        input_dim=input_dim_arg,
        force=force_arg,
        r=r_arg,
    )

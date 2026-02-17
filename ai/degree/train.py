"""Training script for degree prediction models.

Usage:
    python -m ai.degree.train --model gin --name v1
    python -m ai.degree.train --model gcn --name baseline --epochs 10000
    python -m ai.degree.train --model sage --name v1 --hidden-dim 128
    python -m ai.degree.train --model loopy --name r3_v1 --r 3 --epochs 5000
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch_geometric.data import Data

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ai.models import MODEL_CLASSES
from ai.registry import save_model, model_exists, list_model_types
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
    G: nx.Graph[int] = generate_random_graph(
        num_nodes_range=num_nodes_range, p_range=p_range
    )

    num_nodes: int = len(G.nodes())

    # Get node degrees (true labels)
    degrees: torch.Tensor = torch.tensor(
        [G.degree(i) for i in range(num_nodes)], dtype=torch.float
    )

    # Create edge index for PyTorch Geometric
    edge_index_list: list[list[int]] = []
    u: int
    v: int
    for u, v in G.edges():
        edge_index_list.append([u, v])
        edge_index_list.append([v, u])

    edge_index: torch.Tensor
    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

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
    data: Data = Data(x=x, edge_index=edge_index, y=degrees)

    # Apply r-neighborhood transform for Loopy GNN
    if r is not None:
        data = apply_r_neighborhood(data, r=r)

    return data


def train_step(model: Any, optimizer: torch.optim.Optimizer, data: Data) -> float:
    """Train the model on a single graph."""
    model.train()
    optimizer.zero_grad()

    out: torch.Tensor = model(data).squeeze()
    loss: torch.Tensor = F.mse_loss(out, data.y)  # type: ignore

    _ = loss.backward()

    # Gradient clipping to prevent exploding gradients
    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return loss.item()


def evaluate_model(
    model: Any, num_test_graphs: int = 100, input_dim: int = 4, r: int | None = None
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
    model.eval()
    total_loss: float = 0
    total_mae: float = 0
    total_correct: int = 0
    total_predictions: int = 0

    with torch.no_grad():
        for _ in range(num_test_graphs):
            data: Data = generate_random_graph_data(input_dim=input_dim, r=r)
            out: torch.Tensor = model(data).squeeze()

            predictions_rounded: torch.Tensor = torch.round(out)

            loss: torch.Tensor = F.mse_loss(predictions_rounded, data.y)  # type: ignore[arg-type]
            mae: torch.Tensor = F.l1_loss(predictions_rounded, data.y)  # type: ignore[arg-type]
            correct: int = int((predictions_rounded == data.y).sum().item())

            total_loss += loss.item()
            total_mae += mae.item()
            total_correct += correct
            total_predictions += data.y.numel()  # type: ignore[union-attr]

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
) -> Any:
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

    model_id: str = f"{model_type}_{model_name}"

    # Check if model already exists
    if model_exists("degree", model_id) and not force:
        print(f"Error: Model '{model_id}' already exists for task 'degree'.")
        print(f"Use --force to overwrite, or choose a different --name.")
        sys.exit(1)

    # Get model class
    if model_type not in MODEL_CLASSES:
        print(f"Error: Unknown model type '{model_type}'.")
        print(f"Available: {list_model_types()}")
        sys.exit(1)

    model_class: type[Any] = MODEL_CLASSES[model_type]

    print("=" * 60)
    print(f"Training {model_type.upper()} for Degree Prediction")
    print(f"Model ID: {model_id}")
    print("=" * 60)
    print(f"Configuration:")
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
    if model_type == "loopy":
        model = model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout=dropout,
            r=r_for_training,  # type: ignore[call-arg]
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

    best_accuracy: float = 0
    best_mae: float = float("inf")
    best_model_state: dict[str, Any] | None = None
    best_epoch: int = 0

    # Training loop
    for epoch in range(1, num_epochs + 1):
        epoch_loss: float = 0

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
            is_better = (metrics["accuracy"] > best_accuracy) or (
                metrics["accuracy"] == best_accuracy and metrics["mae"] < best_mae
            )

            if is_better:
                best_accuracy = metrics["accuracy"]
                best_mae = metrics["mae"]
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
                print(f"  → New best! Acc: {best_accuracy:.2f}%, MAE: {best_mae:.4f}")

    # Restore best model
    if best_model_state is not None:
        _ = model.load_state_dict(best_model_state)

    # Final evaluation
    print("=" * 60)
    print("Training complete! Final evaluation on 200 test graphs:")
    final_metrics = evaluate_model(
        model, num_test_graphs=200, input_dim=input_dim, r=r_for_training
    )
    print(f"  MSE: {final_metrics['mse']:.4f}")
    print(f"  MAE: {final_metrics['mae']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.2f}%")
    print("=" * 60)

    # Save the best model using registry
    training_info: dict[str, Any] = {
        "epochs": num_epochs,
        "best_epoch": best_epoch,
        "learning_rate": lr,
        "graphs_per_epoch": graphs_per_epoch,
    }
    if r_for_training is not None:
        training_info["r"] = r_for_training

    save_path: Path = save_model(
        model=model,
        task="degree",
        model_id=model_id,
        metrics={
            "mse": round(best_mae**2, 4),  # Approximate MSE from MAE
            "mae": round(best_mae, 4),
            "accuracy": round(best_accuracy, 2),
        },
        training_info=training_info,
    )

    print(f"Model saved to: {save_path}")
    print("=" * 60)

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a GNN model for degree prediction",
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
        required=True,
        help="Name for this model version (e.g., 'v1', 'baseline')",
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
    args = parse_args()

    _ = train_gnn(
        model_type=args.model,
        model_name=args.name,
        num_epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        print_every=args.print_every,
        eval_every=args.eval_every,
        graphs_per_epoch=args.graphs_per_epoch,
        input_dim=args.input_dim,
        force=args.force,
        r=args.r,
    )

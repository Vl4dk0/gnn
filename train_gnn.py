"""Training script for the degree prediction GNN."""
import json
import os
import random
from datetime import datetime

import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from backend.models.degree_gnn import DegreeGNN


def generate_random_graph_data(num_nodes_range=(5, 12), p_range=(0.15, 0.6)):
    """
    Generate a random graph for training.

    Args:
        num_nodes_range: Range of number of nodes (min, max) - smaller range for better learning
        p_range: Range of edge probability for Erdős-Rényi model - higher density for more edges

    Returns:
        PyTorch Geometric Data object
    """
    # Random graph parameters
    num_nodes = random.randint(*num_nodes_range)
    p = random.uniform(*p_range)

    # Generate random graph (allow self-loops and multiple edges)
    G = nx.erdos_renyi_graph(num_nodes, p)

    # Add some random self-loops
    for _ in range(random.randint(0, num_nodes // 3)):
        node = random.randint(0, num_nodes - 1)
        G.add_edge(node, node)

    # Convert to MultiGraph to allow multiple edges
    G = nx.MultiGraph(G)

    # Add some random multiple edges
    edges = list(G.edges())
    if edges:
        for _ in range(random.randint(0, len(edges) // 4)):
            u, v = random.choice(edges)
            G.add_edge(u, v)

    # Get node degrees (true labels)
    degrees = torch.tensor([G.degree(i) for i in range(num_nodes)],
                           dtype=torch.float)

    # Create edge index for PyTorch Geometric
    edge_index = []
    for u, v in G.edges():
        edge_index.append([u, v])
        if u != v:  # Don't duplicate self-loops
            edge_index.append([v, u])

    if len(edge_index) == 0:
        # If no edges, create empty edge_index
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index,
                                  dtype=torch.long).t().contiguous()

    # Rich node features for better learning
    # Feature 1: Normalized node index (0 to 1)
    node_idx_feature = torch.arange(
        num_nodes, dtype=torch.float).unsqueeze(1) / max(num_nodes - 1, 1)

    # Feature 2: Random embedding (helps distinguish nodes)
    random_feature = torch.randn(num_nodes, 2)

    # Feature 3: Clustering coefficient estimate (based on local structure)
    # For simplicity, use a random value that the model can learn to interpret
    clustering_feature = torch.rand(num_nodes, 1)

    # Combine all features
    x = torch.cat([node_idx_feature, random_feature, clustering_feature],
                  dim=1)  # Shape: [num_nodes, 4]

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=degrees)

    return data


def train_model(model, optimizer, data):
    """
    Train the model on a single graph.

    Args:
        model: The GNN model
        optimizer: PyTorch optimizer
        data: PyTorch Geometric Data object

    Returns:
        Training loss
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index).squeeze()

    # Compute loss (Mean Squared Error)
    loss = F.mse_loss(out, data.y)

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_model(model, num_test_graphs=100):
    """
    Evaluate the model on test graphs.

    Args:
        model: The trained GNN model
        num_test_graphs: Number of test graphs to evaluate on

    Returns:
        Dictionary with evaluation metrics:
        - mse: Mean Squared Error
        - mae: Mean Absolute Error
        - accuracy: Percentage of exact predictions (rounded)
    """
    model.eval()
    total_loss = 0
    total_mae = 0
    total_correct = 0
    total_predictions = 0

    with torch.no_grad():
        for _ in range(num_test_graphs):
            data = generate_random_graph_data()
            out = model(data.x, data.edge_index).squeeze()

            # Round predictions (same as what frontend sees)
            predictions_rounded = torch.round(out)

            # MSE and MAE on ROUNDED predictions (same as frontend)
            loss = F.mse_loss(predictions_rounded, data.y)
            mae = F.l1_loss(predictions_rounded, data.y)

            # Accuracy (exact match after rounding)
            correct = (predictions_rounded == data.y).sum().item()

            total_loss += loss.item()
            total_mae += mae.item()
            total_correct += correct
            total_predictions += data.y.numel()

    metrics = {
        'mse':
        total_loss / num_test_graphs,
        'mae':
        total_mae / num_test_graphs,
        'accuracy': (total_correct / total_predictions) *
        100 if total_predictions > 0 else 0
    }

    return metrics


def save_model_info(metrics,
                    epoch,
                    hidden_dim,
                    lr,
                    path='backend/models/model_info.json'):
    """
    Save model information and metrics to JSON file.

    Args:
        metrics: Dictionary of evaluation metrics
        epoch: Current epoch number
        hidden_dim: Hidden dimension size
        lr: Learning rate
        path: Path to save the JSON file
    """
    info = {
        'last_updated': datetime.now().isoformat(),
        'epoch': epoch,
        'hidden_dim': hidden_dim,
        'learning_rate': lr,
        'metrics': {
            'mse': round(metrics['mse'], 4),
            'mae': round(metrics['mae'], 4),
            'accuracy': round(metrics['accuracy'], 2)
        },
        'model_path': 'backend/models/trained_gnn.pt'
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"Model info saved to {path}")


def load_model_info(path='backend/models/model_info.json'):
    """
    Load model information from JSON file.

    Args:
        path: Path to the JSON file

    Returns:
        Dictionary with model info, or None if file doesn't exist
    """
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def train_gnn(num_epochs=5000,
              hidden_dim=64,
              lr=0.005,
              print_every=100,
              eval_every=20,
              graphs_per_epoch=10):
    """
    Main training function with improved training and selective saving.

    Args:
        num_epochs: Number of training epochs
        hidden_dim: Hidden dimension size
        lr: Learning rate
        print_every: Print progress every N epochs
        eval_every: Evaluate and potentially save every N epochs
        graphs_per_epoch: Number of graphs to train on per epoch

    Returns:
        Trained model
    """
    print("=" * 60)
    print("Training GNN for Degree Prediction")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Hidden Dim: {hidden_dim}")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Graphs per Epoch: {graphs_per_epoch}")
    print(f"  - Evaluation Every: {eval_every} epochs")
    print("=" * 60)

    # Initialize model
    model = DegreeGNN(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Load previous best metrics if they exist
    previous_info = load_model_info()
    if previous_info:
        best_accuracy = previous_info['metrics']['accuracy']
        best_mae = previous_info['metrics']['mae']
        print(f"Found existing model:")
        print(f"  - Best Accuracy: {best_accuracy:.2f}%")
        print(f"  - Best MAE: {best_mae:.4f}")
        print(f"  - From Epoch: {previous_info['epoch']}")
        print("=" * 60)
    else:
        best_accuracy = 0
        best_mae = float('inf')
        print("No previous model found. Starting fresh.")
        print("=" * 60)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0

        # Train on multiple graphs per epoch for better generalization
        for _ in range(graphs_per_epoch):
            data = generate_random_graph_data()
            loss = train_model(model, optimizer, data)
            epoch_loss += loss

        epoch_loss /= graphs_per_epoch

        # Evaluate and potentially save
        if epoch % eval_every == 0 or epoch == 1:
            metrics = evaluate_model(model, num_test_graphs=100)

            # Print progress
            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch:4d} | Train Loss: {epoch_loss:.4f} | "
                      f"Test MSE: {metrics['mse']:.4f} | "
                      f"Test MAE: {metrics['mae']:.4f} | "
                      f"Accuracy: {metrics['accuracy']:.2f}%")

            # Save model only if it's better
            # Primary metric: accuracy, secondary: MAE (lower is better)
            is_better = (metrics['accuracy'] > best_accuracy) or \
                       (metrics['accuracy'] == best_accuracy and metrics['mae'] < best_mae)

            if is_better:
                best_accuracy = metrics['accuracy']
                best_mae = metrics['mae']

                # Save model and info
                save_model(model)
                save_model_info(metrics, epoch, hidden_dim, lr)

                print(
                    f"  ✓ New best model! Accuracy: {best_accuracy:.2f}%, MAE: {best_mae:.4f}"
                )

    # Final evaluation
    print("=" * 60)
    print("Training complete! Final evaluation on 200 test graphs:")
    final_metrics = evaluate_model(model, num_test_graphs=200)
    print(f"  Test MSE: {final_metrics['mse']:.4f}")
    print(f"  Test MAE: {final_metrics['mae']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.2f}%")
    print("=" * 60)
    print(f"Best model achieved:")
    print(f"  Accuracy: {best_accuracy:.2f}%")
    print(f"  MAE: {best_mae:.4f}")
    print("=" * 60)

    return model


def save_model(model, path='backend/models/trained_gnn.pt'):
    """Save the trained model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path='backend/models/trained_gnn.pt',
               hidden_dim=None,
               input_dim=4):
    """
    Load a trained model.

    Args:
        path: Path to the saved model
        hidden_dim: Hidden dimension size (if None, will try to load from model_info.json)
        input_dim: Number of input features per node
    """
    # Try to get hidden_dim from model_info.json if not provided
    if hidden_dim is None:
        info = load_model_info()
        if info:
            hidden_dim = info.get('hidden_dim', 16)
        else:
            hidden_dim = 16

    model = DegreeGNN(hidden_dim=hidden_dim, input_dim=input_dim)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        print(
            f"Model loaded from {path} (hidden_dim={hidden_dim}, input_dim={input_dim})"
        )
    else:
        print(f"No saved model found at {path}")
    return model


if __name__ == "__main__":
    # Train the model with improved configuration
    # Optimized for 80%+ accuracy with richer features and more training
    model = train_gnn(
        num_epochs=10000,  # Long training for convergence
        hidden_dim=64,  # Large model for better capacity
        lr=0.005,  # Lower learning rate for stability
        print_every=200,  # Print every 200 epochs
        eval_every=50,  # Evaluate every 50 epochs
        graphs_per_epoch=50  # Much more graphs per epoch for better learning
    )

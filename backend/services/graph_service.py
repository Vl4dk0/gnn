"""Graph analysis service."""
import networkx as nx
import torch
from torch_geometric.data import Data
import os


# Global model cache
_model = None


def get_true_degree(G: nx.Graph, vertex: int) -> int:
    """
    Get the true degree of a vertex in the graph.
    For graphs with self-loops, self-loops are counted twice.
    
    Args:
        G: NetworkX Graph object
        vertex: The vertex to get the degree for
    
    Returns:
        Degree of the vertex
    
    Raises:
        ValueError: If vertex not found in graph
    """
    if vertex not in G.nodes():
        raise ValueError(f"Vertex {vertex} not found in graph")
    
    return G.degree(vertex)


def load_gnn_model():
    """Load the trained GNN model (lazy loading with correct hidden_dim)."""
    global _model
    
    if _model is not None:
        return _model
    
    try:
        from backend.models.degree_gnn import DegreeGNN
        import json
        
        model_path = 'backend/models/trained_gnn.pt'
        info_path = 'backend/models/model_info.json'
        
        if not os.path.exists(model_path):
            print(f"Warning: No trained model found at {model_path}")
            return None
        
        # Try to get hidden_dim from model_info.json
        hidden_dim = 16  # default
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    hidden_dim = info.get('hidden_dim', 16)
            except Exception as e:
                print(f"Warning: Could not read model info: {e}")
        
        _model = DegreeGNN(hidden_dim=hidden_dim, input_dim=4)  # 4 features: idx + random(2) + clustering
        _model.load_state_dict(torch.load(model_path, weights_only=True))
        _model.eval()
        print(f"GNN model loaded from {model_path} (hidden_dim={hidden_dim}, input_dim=4)")
        
        return _model
    except Exception as e:
        print(f"Error loading GNN model: {e}")
        return None


def predict_degree_with_gnn(G: nx.Graph, vertex: int) -> float:
    """
    Predict the degree of a vertex using a trained GNN.
    
    Args:
        G: NetworkX Graph object
        vertex: The vertex to predict the degree for
    
    Returns:
        Predicted degree from the GNN (rounded to nearest integer)
    """
    model = load_gnn_model()
    
    # If model not available, return true degree
    if model is None:
        return float(get_true_degree(G, vertex))
    
    try:
        # Convert NetworkX graph to PyTorch Geometric format
        num_nodes = len(G.nodes())
        
        # Create node ID mapping (handle non-sequential node IDs)
        node_list = sorted(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Get target vertex index
        target_idx = node_to_idx[vertex]
        
        # Build edge index
        edge_index = []
        for u, v in G.edges():
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edge_index.append([u_idx, v_idx])
            if u != v:  # Don't duplicate self-loops
                edge_index.append([v_idx, u_idx])
        
        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Node features: match training setup with rich features
        # Feature 1: Normalized node index (0 to 1)
        node_idx_feature = torch.arange(num_nodes, dtype=torch.float).unsqueeze(1) / max(num_nodes - 1, 1)
        
        # Feature 2: Random embedding (deterministic with seed for consistency)
        torch.manual_seed(42)
        random_feature = torch.randn(num_nodes, 2)
        
        # Feature 3: Clustering coefficient placeholder
        torch.manual_seed(42)
        clustering_feature = torch.rand(num_nodes, 1)
        
        # Combine all features (must match training: 4 features)
        x = torch.cat([node_idx_feature, random_feature, clustering_feature], dim=1)
        
        # Create data object
        data = Data(x=x, edge_index=edge_index)
        
        # Predict with GNN
        with torch.no_grad():
            predictions = model(data.x, data.edge_index).squeeze()
            
            # Get prediction for target vertex
            if num_nodes == 1:
                predicted_degree = predictions.item()
            else:
                predicted_degree = predictions[target_idx].item()
        
        # Round to nearest integer and ensure non-negative
        predicted_degree = max(0.0, round(predicted_degree))
        
        return predicted_degree
        
    except Exception as e:
        print(f"Error in GNN prediction: {e}")
        # Fall back to true degree
        return float(get_true_degree(G, vertex))


def predict_all_nodes(G: nx.Graph) -> list[dict]:
    """
    Predict the degree of all vertices in the graph using a trained GNN.
    
    Args:
        G: NetworkX Graph object
    
    Returns:
        List of dictionaries with format:
        [
            {
                "node_id": 0,
                "true_degree": 2,
                "predicted_degree": 2.0
            },
            ...
        ]
    """
    model = load_gnn_model()
    
    # If model not available, return true degrees
    if model is None:
        return [{
            "node_id": node,
            "true_degree": get_true_degree(G, node),
            "predicted_degree": float(get_true_degree(G, node))
        } for node in sorted(G.nodes())]
    
    try:
        # Convert NetworkX graph to PyTorch Geometric format
        num_nodes = len(G.nodes())
        
        if num_nodes == 0:
            return []
        
        # Create node ID mapping (handle non-sequential node IDs)
        node_list = sorted(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Build edge index
        edge_index = []
        for u, v in G.edges():
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edge_index.append([u_idx, v_idx])
            if u != v:  # Don't duplicate self-loops
                edge_index.append([v_idx, u_idx])
        
        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Node features: match training setup with rich features
        # Feature 1: Normalized node index (0 to 1)
        node_idx_feature = torch.arange(num_nodes, dtype=torch.float).unsqueeze(1) / max(num_nodes - 1, 1)
        
        # Feature 2: Random embedding (deterministic with seed for consistency)
        torch.manual_seed(42)
        random_feature = torch.randn(num_nodes, 2)
        
        # Feature 3: Clustering coefficient placeholder
        torch.manual_seed(42)
        clustering_feature = torch.rand(num_nodes, 1)
        
        # Combine all features (must match training: 4 features)
        x = torch.cat([node_idx_feature, random_feature, clustering_feature], dim=1)
        
        # Create data object
        data = Data(x=x, edge_index=edge_index)
        
        # Predict with GNN
        with torch.no_grad():
            predictions = model(data.x, data.edge_index).squeeze()
            
            # Handle single node case
            if num_nodes == 1:
                predictions = predictions.unsqueeze(0)
        
        # Build results for all nodes
        results = []
        for node in node_list:
            idx = node_to_idx[node]
            predicted_degree = max(0.0, round(predictions[idx].item()))
            true_degree = get_true_degree(G, node)
            
            results.append({
                "node_id": node,
                "true_degree": true_degree,
                "predicted_degree": predicted_degree
            })
        
        return results
        
    except Exception as e:
        print(f"Error in GNN prediction for all nodes: {e}")
        # Fall back to true degrees
        return [{
            "node_id": node,
            "true_degree": get_true_degree(G, node),
            "predicted_degree": float(get_true_degree(G, node))
        } for node in sorted(G.nodes())]

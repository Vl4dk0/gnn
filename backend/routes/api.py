"""API routes for the application."""
from flask import Blueprint, request, jsonify
import networkx as nx
import random
from backend.utils.graph_parser import parse_edge_list
from backend.services.graph_service import get_true_degree, predict_degree_with_gnn
from backend.services.visualization_service import create_graph_visualization

api_bp = Blueprint('api', __name__)


@api_bp.route('/generate', methods=['GET'])
def generate_random_graph():
    """
    Generate a random graph with 7-12 nodes.
    
    Returns:
    {
        "graph": "0 1\n0 2\n...",  // edge list as string
        "vertex": 3                 // randomly selected vertex
    }
    """
    try:
        # Random number of nodes between 7 and 12
        num_nodes = random.randint(7, 12)
        
        # Generate a random graph using Erdős-Rényi model
        # Probability adjusted to get a reasonably connected graph
        p = random.uniform(0.2, 0.4)
        G = nx.erdos_renyi_graph(num_nodes, p)
        
        # Convert graph to edge list string
        edge_list = []
        for u, v in G.edges():
            edge_list.append(f"{u} {v}")
        
        graph_str = "\n".join(edge_list)
        
        # Select a random vertex
        random_vertex = random.randint(0, num_nodes - 1)
        
        return jsonify({
            "graph": graph_str,
            "vertex": random_vertex
        })
    
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@api_bp.route('/analyze', methods=['POST'])
def analyze_graph():
    """
    Endpoint to analyze a graph and predict vertex degree.
    
    Expected JSON payload:
    {
        "graph": "0 1\n0 2\n1 2",  // edge list as string
        "vertex": 0                 // target vertex
    }
    
    Returns:
    {
        "true_degree": 2,
        "predicted_degree": 2.0,
        "graph_image": "base64_encoded_image"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        graph_str = data.get('graph')
        target_vertex = data.get('vertex')
        
        if graph_str is None:
            return jsonify({"error": "Graph data is required"}), 400
        
        if target_vertex is None:
            return jsonify({"error": "Target vertex is required"}), 400
        
        # Parse the graph
        G = parse_edge_list(graph_str)
        
        if len(G.nodes()) == 0:
            return jsonify({"error": "Graph is empty"}), 400
        
        # Get true degree
        true_degree = get_true_degree(G, target_vertex)
        
        # Get predicted degree (placeholder for now)
        predicted_degree = predict_degree_with_gnn(G, target_vertex)
        
        # Create visualization
        graph_image = create_graph_visualization(G, target_vertex)
        
        return jsonify({
            "true_degree": true_degree,
            "predicted_degree": predicted_degree,
            "graph_image": graph_image
        })
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

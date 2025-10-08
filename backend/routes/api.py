"""API routes for the application."""
import os
import random

import networkx as nx
from flask import Blueprint
from flask import jsonify
from flask import request

from backend.services.graph_service import get_true_degree
from backend.services.graph_service import predict_all_nodes
from backend.services.graph_service import predict_degree_with_gnn
from backend.utils.graph_generation import generate_random_graph
from backend.utils.graph_generation import graph_to_edge_list
from backend.utils.graph_parser import parse_edge_list

api_bp = Blueprint('api', __name__)


@api_bp.route('/config', methods=['GET'])
def get_config():
    """
    Get frontend configuration from environment variables.
    
    Returns:
    {
        "apiBaseUrl": "http://localhost:5555"
    }
    """
    port = os.getenv('PORT', '5555')
    host = os.getenv('HOST', '0.0.0.0')
    
    # Construct API base URL
    # For localhost/0.0.0.0, use localhost in the URL
    if host in ['0.0.0.0', '127.0.0.1', 'localhost']:
        api_base_url = f"http://localhost:{port}"
    else:
        api_base_url = f"http://{host}:{port}"
    
    return jsonify({
        "apiBaseUrl": api_base_url
    })


@api_bp.route('/generate', methods=['GET'])
def generate_random_graph_endpoint():
    """
    Generate a random graph with 7-12 nodes.

    Returns:
    {
        "graph": "3\n0 1\n0 2\n..."  // edge list as string
    }
    """
    try:
        # Use centralized graph generation
        G = generate_random_graph()
        graph_str = graph_to_edge_list(G)
        
        return jsonify({"graph": graph_str})

    except Exception as e:
        return jsonify({"error":
                        f"An unexpected error occurred: {str(e)}"}), 500


@api_bp.route('/analyze', methods=['POST'])
def analyze_graph():
    """
    Endpoint to analyze a graph and predict vertex degrees for all nodes.

    Expected JSON payload:
    {
        "graph": "0 1\n0 2\n1 2"  // edge list as string
    }

    Returns:
    {
        "predictions": [
            {
                "node_id": 0,
                "true_degree": 2,
                "predicted_degree": 2.0
            },
            {
                "node_id": 1,
                "true_degree": 2,
                "predicted_degree": 2.0
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        graph_str = data.get('graph')

        if graph_str is None:
            return jsonify({"error": "Graph data is required"}), 400

        # Parse the graph
        G = parse_edge_list(graph_str)

        # Get predictions for all nodes
        predictions = predict_all_nodes(G)

        return jsonify({"predictions": predictions})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error":
                        f"An unexpected error occurred: {str(e)}"}), 500

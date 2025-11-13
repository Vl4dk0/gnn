"""API routes for degree prediction."""
import os
import sys

from flask import Blueprint, jsonify, request

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.services.graph_service import predict_all_nodes
from backend.utils.graph_parser import parse_edge_list
from backend.utils.graph_utils import generate_random_graph, graph_to_edge_list

degree_bp = Blueprint('degree', __name__, url_prefix='/api/degree')


@degree_bp.route('/generate', methods=['POST'])
def generate_random_graph_endpoint():
    """
    Generate a random graph with configurable parameters.

    Expected JSON payload (all optional):
    {
        "minNodes": 5,
        "maxNodes": 12,
        "minProb": 0.15,
        "maxProb": 0.60,
        "allowSelfLoops": true
    }

    Returns:
    {
        "graph": "3\n0 1\n0 2\n..."  // edge list as string
    }
    """
    try:
        data = request.get_json() or {}

        # Get parameters from request or use defaults
        min_nodes = data.get('minNodes', 5)
        max_nodes = data.get('maxNodes', 12)
        min_prob = data.get('minProb', 0.15)
        max_prob = data.get('maxProb', 0.60)
        allow_self_loops = data.get('allowSelfLoops', True)

        # Use centralized graph generation
        G = generate_random_graph(
            num_nodes_range=(min_nodes, max_nodes),
            p_range=(min_prob, max_prob),
            self_loop_prob=0.1 if allow_self_loops else 0.0)
        graph_str = graph_to_edge_list(G)

        return jsonify({"graph": graph_str})

    except Exception as e:
        return jsonify({"error":
                        f"An unexpected error occurred: {str(e)}"}), 500


@degree_bp.route('/analyze', methods=['POST'])
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

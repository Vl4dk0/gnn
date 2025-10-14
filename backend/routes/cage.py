"""
Blueprint for Cage Graph Generator (RL-based) API endpoints.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from flask import Blueprint, jsonify, request
from utils.graph_utils import generate_random_graph, graph_to_edge_list

# Create blueprint with /api/cage prefix
cage_bp = Blueprint('cage', __name__, url_prefix='/api/cage')


@cage_bp.route('/generate', methods=['POST'])
def generate_cage():
    """
    Generate a random graph (placeholder for RL-based cage generation).
    For now, just returns a random graph from utils.
    """
    try:
        data = request.get_json()
        
        # Get settings from request or use defaults
        min_nodes = data.get('min_nodes', 5)
        max_nodes = data.get('max_nodes', 12)
        min_probability = data.get('min_probability', 0.15)
        max_probability = data.get('max_probability', 0.60)
        allow_self_loops = data.get('allow_self_loops', True)
        
        # Generate random graph using shared utility
        G = generate_random_graph(
            num_nodes_range=(min_nodes, max_nodes),
            p_range=(min_probability, max_probability),
            self_loop_prob=0.1 if allow_self_loops else 0.0
        )
        
        # Convert to edge list for frontend
        edge_list = graph_to_edge_list(G)
        
        return jsonify({
            'graph': edge_list,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@cage_bp.route('/analyze', methods=['POST'])
def analyze_cage():
    """
    Analyze if the given graph is a cage.
    For now, always returns True as placeholder.
    """
    try:
        data = request.get_json()
        edge_list = data.get('edge_list', [])
        
        # Placeholder: always return True
        # TODO: Implement actual cage validation logic
        is_cage = True
        
        return jsonify({
            'is_cage': is_cage,
            'message': 'Graph is a cage!' if is_cage else 'Graph is not a cage.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@cage_bp.route('/status', methods=['GET'])
def get_status():
    """Get the status of the cage generator."""
    return jsonify({
        'status': 'active',
        'message': 'Cage graph generator is running (placeholder mode)'
    })

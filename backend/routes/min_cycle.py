from typing import TypedDict, cast

from flask import Blueprint, jsonify, request

from ai.min_cycle import predict_all_nodes
from ai.registry import list_trained_models, get_best_model_id, model_exists
from backend.utils import parse_edge_list, generate_random_graph, graph_to_edge_list

min_cycle_bp = Blueprint("min_cycle", __name__, url_prefix="/api/min_cycle")


class _GenerateRequest(TypedDict, total=False):
    minNodes: int
    maxNodes: int
    minProb: float
    maxProb: float
    allowSelfLoops: bool


class _AnalyzeRequest(TypedDict, total=False):
    graph: str
    model_id: str


@min_cycle_bp.route("/models", methods=["GET"])
def get_models():
    """
    Get list of available trained models for min_cycle prediction.

    Returns:
    {
        "models": [
            {
                "model_id": "gin_v1",
                "model_type": "gin",
                "metrics": {"accuracy": 23.0, "mae": 2.19, ...},
                ...
            },
            ...
        ],
        "default": "gin_v1"
    }
    """
    try:
        models = list_trained_models("min_cycle")
        default_model = get_best_model_id("min_cycle")

        return jsonify(
            {
                "models": models,
                "default": default_model,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Failed to list models: {str(e)}"}), 500


@min_cycle_bp.route("/generate", methods=["POST"])
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
        data: _GenerateRequest = cast(_GenerateRequest, request.get_json() or {})

        # Get parameters from request or use defaults
        min_nodes: int = data.get("minNodes", 5)
        max_nodes: int = data.get("maxNodes", 12)
        min_prob: float = data.get("minProb", 0.15)
        max_prob: float = data.get("maxProb", 0.60)
        allow_self_loops: bool = data.get("allowSelfLoops", True)

        # Use centralized graph generation
        G = generate_random_graph(
            num_nodes_range=(min_nodes, max_nodes),
            p_range=(min_prob, max_prob),
            self_loop_prob=0.1 if allow_self_loops else 0.0,
        )
        graph_str = graph_to_edge_list(G)

        return jsonify({"graph": graph_str})

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@min_cycle_bp.route("/analyze", methods=["POST"])
def analyze_graph():
    """
    Endpoint to analyze a graph and predict smallest cycle containing node for all nodes.

    Expected JSON payload:
    {
        "graph": "0 1\n0 2\n1 2",  // edge list as string
        "model_id": "gin_v1"       // optional, uses best model if not provided
    }

    Returns:
    {
        "predictions": [
            {
                "node_id": 0,
                "true": 3,
                "predicted": 3.0
            },
            ...
        ],
        "model_id": "gin_v1"
    }
    """
    try:
        data: _AnalyzeRequest | None = cast(_AnalyzeRequest | None, request.get_json())

        if not data:
            return jsonify({"error": "No data provided"}), 400

        graph_str: str | None = data.get("graph")
        model_id: str | None = data.get(
            "model_id"
        )  # Optional, will use best if not provided

        if graph_str is None:
            return jsonify({"error": "Graph data is required"}), 400

        if model_id is not None and not model_exists("min_cycle", model_id):
            return jsonify({"error": f"Unknown min_cycle model_id: {model_id}"}), 400

        # Parse the graph
        G = parse_edge_list(graph_str)

        # Get predictions for all nodes
        predictions = predict_all_nodes(G, model_id=model_id)

        return jsonify(
            {
                "predictions": predictions,
                "model_id": model_id or get_best_model_id("min_cycle"),
            }
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

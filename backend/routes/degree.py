from numbers import Real
from typing import TypedDict, cast

from flask import Blueprint, jsonify, request

from ai.degree import predict_all_nodes
from ai.registry import list_trained_models, get_best_model_id, model_exists
from backend.utils import parse_edge_list, generate_random_graph, graph_to_edge_list

degree_bp = Blueprint("degree", __name__, url_prefix="/api/degree")


class _AnalyzeRequest(TypedDict, total=False):
    graph: str
    model_id: str


def _read_generate_request(data: dict[str, object]) -> tuple[int, int, float, float, bool]:
    min_nodes_raw = data.get("minNodes", 5)
    max_nodes_raw = data.get("maxNodes", 12)
    min_prob_raw = data.get("minProb", 0.15)
    max_prob_raw = data.get("maxProb", 0.60)
    allow_self_loops_raw = data.get("allowSelfLoops", True)

    if not isinstance(min_nodes_raw, int) or isinstance(min_nodes_raw, bool):
        raise ValueError("minNodes must be an integer")
    if not isinstance(max_nodes_raw, int) or isinstance(max_nodes_raw, bool):
        raise ValueError("maxNodes must be an integer")
    if min_nodes_raw < 1 or max_nodes_raw < 1:
        raise ValueError("minNodes and maxNodes must be positive")
    if min_nodes_raw > max_nodes_raw:
        raise ValueError("minNodes must be <= maxNodes")
    if isinstance(min_prob_raw, bool) or not isinstance(min_prob_raw, Real):
        raise ValueError("minProb must be a number")
    if isinstance(max_prob_raw, bool) or not isinstance(max_prob_raw, Real):
        raise ValueError("maxProb must be a number")

    min_prob = float(min_prob_raw)
    max_prob = float(max_prob_raw)
    if min_prob < 0 or min_prob > 1 or max_prob < 0 or max_prob > 1:
        raise ValueError("minProb and maxProb must be between 0 and 1")
    if min_prob > max_prob:
        raise ValueError("minProb must be <= maxProb")
    if not isinstance(allow_self_loops_raw, bool):
        raise ValueError("allowSelfLoops must be a boolean")

    return min_nodes_raw, max_nodes_raw, min_prob, max_prob, allow_self_loops_raw


@degree_bp.route("/models", methods=["GET"])
def get_models():
    """
    Get list of available trained models for degree prediction.

    Returns:
    {
        "models": [
            {
                "model_id": "gin_v1",
                "model_type": "gin",
                "metrics": {"accuracy": 100.0, "mae": 0.0, ...},
                ...
            },
            ...
        ],
        "default": "gin_v1"
    }
    """
    try:
        models = list_trained_models("degree")
        default_model = get_best_model_id("degree")

        return jsonify(
            {
                "models": models,
                "default": default_model,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Failed to list models: {str(e)}"}), 500


@degree_bp.route("/generate", methods=["POST"])
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
        data = cast(dict[str, object], request.get_json() or {})

        min_nodes, max_nodes, min_prob, max_prob, allow_self_loops = _read_generate_request(data)

        # Use centralized graph generation
        G = generate_random_graph(
            num_nodes_range=(min_nodes, max_nodes),
            p_range=(min_prob, max_prob),
            self_loop_prob=0.1 if allow_self_loops else 0.0,
        )
        graph_str = graph_to_edge_list(G)

        return jsonify({"graph": graph_str})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@degree_bp.route("/analyze", methods=["POST"])
def analyze_graph():
    """
    Endpoint to analyze a graph and predict vertex degrees for all nodes.

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
                "true": 2,
                "predicted": 2.0
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

        if model_id is not None and not model_exists("degree", model_id):
            return jsonify({"error": f"Unknown degree model_id: {model_id}"}), 400

        # Parse the graph
        G = parse_edge_list(graph_str)

        # Get predictions for all nodes
        predictions = predict_all_nodes(G, model_id=model_id)

        return jsonify(
            {
                "predictions": predictions,
                "model_id": model_id or get_best_model_id("degree"),
            }
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

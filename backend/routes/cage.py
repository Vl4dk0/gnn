"""
Blueprint for Cage Graph Generator API endpoints.
"""

import uuid
import threading
import time
from typing import Any
from collections.abc import Generator

from flask import Blueprint, jsonify, request, Response

from ai.cage import (
    RandomWalkGenerator,
    BruteforceGenerator,
    AStarGenerator,
    RLGenerator,
)
from backend.utils.graph_utils import (
    graph_to_edge_list,
    is_valid_cage,
    compute_girth,
    is_k_regular,
    moore_bound,
    moore_hoffman_upper_bound,
)

# Create blueprint with /api/cage prefix
cage_bp = Blueprint("cage", __name__, url_prefix="/api/cage")

# Global state for active generation sessions
# Structure: {session_id: {'generator': generator, 'last_poll': timestamp, 'thread': thread}}
generation_sessions = {}
session_lock = threading.Lock()

# Timeout in seconds - stop generation if not polled for this long
POLL_TIMEOUT = 5
# Hard cap on how long a single generation session may run
MAX_GENERATION_TIME = 300  # 5 minutes
MAX_PARALLEL_GENERATIONS = 3

# Safety limits to prevent pathological generation requests from exhausting local compute.
MAX_MOORE_BOUND = 120


def _count_active_generations_locked() -> int:
    """Count currently running generation threads (lock must be held)."""
    active = 0
    for session in generation_sessions.values():
        thread = session.get("thread")
        if isinstance(thread, threading.Thread) and thread.is_alive():
            active += 1
    return active


def run_generation(
    session_id: str,
    generator: RandomWalkGenerator | BruteforceGenerator | AStarGenerator | RLGenerator,
) -> None:
    """
    Background thread function that runs generation continuously.
    Stops if session hasn't been polled recently (abandoned by frontend).
    """
    try:
        while not generator.is_complete:
            # Check if session has been abandoned (no polling)
            with session_lock:
                if session_id not in generation_sessions:
                    print(f"Generation thread {session_id} - session removed, stopping")
                    break

                session: dict[str, Any] = generation_sessions[session_id]
                last_poll: float = session.get("last_poll", time.time())

                # Stop if no polling for POLL_TIMEOUT seconds
                if time.time() - last_poll > POLL_TIMEOUT:
                    print(
                        f"Generation thread {session_id} - no polling for {POLL_TIMEOUT}s, stopping"
                    )
                    # Mark as stopped but keep in sessions for one final status check
                    session["stopped"] = True
                    break

                # Stop if generation has been running too long
                if generator.elapsed_time() > MAX_GENERATION_TIME:
                    print(
                        f"Generation thread {session_id} - exceeded {MAX_GENERATION_TIME}s limit, stopping"
                    )
                    session["stopped"] = True
                    session["timed_out"] = True
                    break

            generator.step()
            # Small sleep to prevent CPU spinning, but still fast
            time.sleep(0.001)  # 1ms between steps
    except Exception as e:
        print(f"Error in generation thread {session_id}: {e}")
    finally:
        print(f"Generation thread {session_id} completed. Success: {generator.success}")


@cage_bp.route("/status", methods=["GET"])
def status() -> Response:
    """Health check endpoint."""
    with session_lock:
        active = _count_active_generations_locked()
    return jsonify(
        {
            "status": "ready",
            "message": "Cage graph generator API is ready",
            "active_sessions": active,
            "max_parallel_generations": MAX_PARALLEL_GENERATIONS,
            "max_moore_bound": MAX_MOORE_BOUND,
        }
    )


@cage_bp.route("/generate", methods=["POST"])
def generate() -> Response | tuple[Response, int]:
    """Start a new cage graph generation session in a background thread."""
    data: dict[str, Any] | None = request.get_json()

    if not data or "k" not in data or "g" not in data:
        return jsonify({"error": "Missing k or g parameter"}), 400

    k: int = int(data["k"])
    g: int = int(data["g"])
    generator_type: str = str(
        data.get("generator", "randomwalk")
    )  # Default to randomwalk

    # Validation
    if k < 2:
        return jsonify({"error": "k must be >= 2"}), 400
    if g < 3:
        return jsonify({"error": "g must be >= 3"}), 400
    mb = moore_bound(k, g)
    if mb > MAX_MOORE_BOUND:
        moore_formula = (
            "N_M(k,g)=1+k*sum_{i=0}^{(g-3)/2}(k-1)^i (odd g), "
            "N_M(k,g)=2*sum_{i=0}^{g/2-1}(k-1)^i (even g)"
        )
        return (
            jsonify(
                {
                    "error": (
                        f"(k,g)=({k},{g}) is outside this showcase range: "
                        f"Moore bound N_M={mb} exceeds the limit {MAX_MOORE_BOUND}. "
                        f"Formula used: {moore_formula}. "
                        "Please choose smaller k or g."
                    )
                }
            ),
            400,
        )

    # Create generator based on type
    session_id = str(uuid.uuid4())

    if generator_type == "bruteforce":
        generator = BruteforceGenerator(k, g)
    elif generator_type == "astar":
        generator = AStarGenerator(k, g)
    elif generator_type == "rl":
        model_type = data.get("model", "gin")
        generator = RLGenerator(k, g, model_type=model_type)
    else:  # 'randomwalk' or default
        generator = RandomWalkGenerator(k, g)

    # Start background thread to run generation
    thread = threading.Thread(
        target=run_generation,
        args=(session_id, generator),
        daemon=True,
        name=f"cage-gen-{session_id[:8]}",
    )

    with session_lock:
        active = _count_active_generations_locked()
        if active >= MAX_PARALLEL_GENERATIONS:
            return (
                jsonify(
                    {
                        "error": (
                            "Generating queue is full. "
                            "Please wait until a running generation finishes."
                        ),
                        "active_sessions": active,
                        "max_parallel_generations": MAX_PARALLEL_GENERATIONS,
                    }
                ),
                429,
            )

        generation_sessions[session_id] = {
            "generator": generator,
            "last_poll": time.time(),
            "thread": thread,
            "stopped": False,
        }

    thread.start()

    return jsonify(
        {
            "session_id": session_id,
            "status": "started",
            "k": k,
            "g": g,
            "moore_bound": mb,
            "upper_bound": moore_hoffman_upper_bound(k, g),
        }
    )


@cage_bp.route("/status/<session_id>", methods=["GET"])
def get_status(session_id: str) -> Response | tuple[Response, int]:
    """Get current status of generation session (read-only, just observes)."""
    with session_lock:
        session: dict[str, Any] | None = generation_sessions.get(session_id)

        if not session:
            return jsonify({"error": "Session not found"}), 404

        # Update last poll time to keep session alive
        session["last_poll"] = time.time()
        generator: (
            RandomWalkGenerator | BruteforceGenerator | AStarGenerator | RLGenerator
        ) = session["generator"]
        stopped: bool = session.get("stopped", False)
        timed_out: bool = session.get("timed_out", False)

    # Just read current state - don't execute steps (background thread handles that)
    # Compute girth and convert infinity to null for JSON
    girth_val = (
        compute_girth(generator.graph)
        if len(generator.graph.edges()) > 0
        else float("inf")
    )  # type: ignore
    girth_json = None if girth_val == float("inf") else girth_val

    return jsonify(
        {
            "session_id": session_id,
            "k": generator.k,
            "g": generator.g,
            "step_count": generator.step_count,
            "num_nodes": len(generator.graph.nodes()),  # type: ignore
            "num_edges": len(generator.graph.edges()),  # type: ignore
            "girth": girth_json,  # Now properly converts inf to null
            "is_k_regular": generator.is_regular(),
            "is_complete": generator.is_complete,
            "success": generator.success,
            "stopped": stopped,
            "timed_out": timed_out,
            "current_graph": graph_to_edge_list(generator.graph),  # type: ignore
            "moore_bound": moore_bound(generator.k, generator.g),
            "elapsed_time": generator.elapsed_time(),
        }
    )


@cage_bp.route("/stop/<session_id>", methods=["POST"])
def stop(session_id: str) -> Response | tuple[Response, int]:
    """Stop and clean up generation session."""
    with session_lock:
        if session_id in generation_sessions:
            del generation_sessions[session_id]
            return jsonify({"message": "Session stopped successfully"})

    return jsonify({"error": "Session not found"}), 404


@cage_bp.route("/analyze", methods=["POST"])
def analyze() -> Response | tuple[Response, int]:
    """Analyze if a provided graph is a valid cage."""
    data: dict[str, Any] | None = request.get_json()

    if not data or "k" not in data or "g" not in data or "edges" not in data:
        return jsonify({"error": "Missing required parameters"}), 400

    k: int = int(data["k"])
    g: int = int(data["g"])
    edges: Any = data["edges"]

    if k < 2 or g < 3:
        return jsonify({"error": "k must be >= 2 and g must be >= 3"}), 400

    # Build graph from edges
    import networkx as nx

    graph = nx.Graph()  # type: ignore
    for edge in edges:
        if len(edge) == 2:
            graph.add_edge(edge[0], edge[1])  # type: ignore
        elif len(edge) == 1:
            graph.add_node(edge[0])  # type: ignore

    # Analyze
    num_nodes = len(graph.nodes())  # type: ignore
    num_edges = len(graph.edges())  # type: ignore
    current_girth = compute_girth(graph)  # type: ignore
    is_regular = is_k_regular(graph, k)  # type: ignore
    is_cage = is_valid_cage(graph, k, g)  # type: ignore
    mb = moore_bound(k, g)

    return jsonify(
        {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "girth": current_girth if current_girth != float("inf") else None,
            "is_k_regular": is_regular,
            "is_valid_cage": is_cage,
            "moore_bound": mb,
            "is_optimal": is_cage and num_nodes == mb,
        }
    )


def cleanup_old_sessions() -> None:
    """
    Cleanup task that runs periodically to remove stopped sessions.
    This runs in a background thread.
    """
    while True:
        time.sleep(10)  # Check every 10 seconds

        with session_lock:
            # Find sessions that are stopped and haven't been polled recently
            to_remove: list[str] = []
            session_id: str
            session: dict[str, Any]
            for session_id, session in generation_sessions.items():
                if session.get("stopped", False):
                    # Remove stopped sessions after 30 seconds
                    if time.time() - session["last_poll"] > 30:
                        to_remove.append(session_id)

            # Remove them
            for session_id in to_remove:
                print(f"Cleaning up stopped session: {session_id}")
                del generation_sessions[session_id]


# Start cleanup thread when module loads
cleanup_thread = threading.Thread(
    target=cleanup_old_sessions, daemon=True, name="session-cleanup"
)
cleanup_thread.start()

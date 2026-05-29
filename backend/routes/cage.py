"""
Blueprint for Cage Graph Generator API endpoints.
"""

import json
import threading
import time
import uuid
from collections import deque
from typing import Literal, Protocol, TypedDict, Required, cast

import networkx as nx
from flask import Blueprint, jsonify, request, Response

from ai.cage import (
    RandomWalkGenerator,
    BruteforceGenerator,
    AStarGenerator,
    RLGenerator,
    VoltageSearchGenerator,
    VoltageRLGenerator,
)
from ai.registry import (
    get_best_model_id,
    get_trained_dir,
    list_trained_models,
    model_exists,
)
from backend.utils.graph_utils import (
    graph_to_edge_list,
    is_valid_cage,
    compute_girth,
    is_k_regular,
    moore_bound,
    moore_hoffman_upper_bound,
)


class _GeneratorProto(Protocol):
    k: int
    g: int
    graph: nx.Graph[int]
    step_count: int
    is_complete: bool
    success: bool

    def step(self) -> None: ...
    def elapsed_time(self) -> float: ...
    def is_regular(self) -> bool: ...


type _GeneratorType = Literal[
    "randomwalk", "bruteforce", "astar", "rl", "voltage", "voltage_rl"
]
type _ExecutionMode = Literal["async", "stepped"]
type _Generator = (
    RandomWalkGenerator
    | BruteforceGenerator
    | AStarGenerator
    | RLGenerator
    | VoltageSearchGenerator
    | VoltageRLGenerator
)


class _StepEvent(TypedDict):
    step: int
    action: str
    u: int | None
    v: int | None
    accepted: bool
    reward: float
    done: bool
    done_reason: str | None


class _Session(TypedDict, total=False):
    generator: Required[_Generator]
    mode: Required[_ExecutionMode]
    generator_type: Required[_GeneratorType]
    last_poll: Required[float]
    last_activity: Required[float]
    thread: Required[threading.Thread | None]
    stopped: Required[bool]
    lock: Required[threading.Lock]
    events: Required[deque[_StepEvent]]
    timed_out: bool


class _GenerateRequest(TypedDict, total=False):
    k: Required[int]
    g: Required[int]
    generator: str
    model: str
    model_id: str
    mode: str


class _StepRequest(TypedDict, total=False):
    steps: int


class _AnalyzeRequest(TypedDict, total=False):
    k: Required[int]
    g: Required[int]
    edges: Required[list[list[int]]]


# Create blueprint with /api/cage prefix
cage_bp = Blueprint("cage", __name__, url_prefix="/api/cage")

# Global state for active generation sessions
# Structure: {session_id: {'generator': generator, 'last_poll': timestamp, 'thread': thread}}
generation_sessions: dict[str, _Session] = {}
session_lock = threading.Lock()

# Timeout in seconds - stop generation if not polled for this long
POLL_TIMEOUT = 5
# Hard cap on how long a single generation session may run
MAX_GENERATION_TIME = 300  # 5 minutes
MAX_PARALLEL_GENERATIONS = 3
STEPPED_IDLE_TIMEOUT = 30 * 60
MAX_STEP_BATCH = 100

# Safety limits to prevent pathological generation requests from exhausting local compute.
MAX_MOORE_BOUND = 120
VALID_GENERATORS: set[str] = {
    "randomwalk",
    "bruteforce",
    "astar",
    "rl",
    "voltage",
    "voltage_rl",
}
VALID_MODES: set[str] = {"async", "stepped"}


def _count_active_generations_locked() -> int:
    """Count currently running generation threads (lock must be held)."""
    active = 0
    for session in generation_sessions.values():
        thread = session["thread"]
        if thread is not None and thread.is_alive():
            active += 1
    return active


def _normalize_generator_type(generator_type: str) -> _GeneratorType | None:
    if generator_type in VALID_GENERATORS:
        return cast(_GeneratorType, generator_type)
    return None


def _normalize_mode(mode: str) -> _ExecutionMode | None:
    if mode in VALID_MODES:
        return cast(_ExecutionMode, mode)
    return None


def _voltage_rl_model_exists(model_id: str) -> bool:
    model_dir = get_trained_dir("cage") / model_id
    info_path = model_dir / "info.json"
    weights_path = model_dir / "weights.pt"
    if not info_path.exists() or not weights_path.exists():
        return False

    with open(info_path) as f:
        info = cast(dict[str, object], json.load(f))
    return info.get("model_type") == "voltage_actor_critic"


def _voltage_girth_model_exists(model_id: str) -> bool:
    model_dir = get_trained_dir("voltage_girth") / model_id
    info_path = model_dir / "info.json"
    weights_path = model_dir / "weights.pt"
    if not info_path.exists() or not weights_path.exists():
        return False
    with open(info_path) as f:
        info = cast(dict[str, object], json.load(f))
    return info.get("model_type") == "voltage_girth_predictor"


def _list_voltage_girth_models() -> list[dict[str, object]]:
    """Enumerate trained girth predictors with key training metadata."""
    out: list[dict[str, object]] = []
    base_dir = get_trained_dir("voltage_girth")
    if not base_dir.exists():
        return out
    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        info_path = model_dir / "info.json"
        weights_path = model_dir / "weights.pt"
        if not info_path.exists() or not weights_path.exists():
            continue
        with open(info_path) as f:
            info = cast(dict[str, object], json.load(f))
        if info.get("model_type") != "voltage_girth_predictor":
            continue
        training = cast(dict[str, object], info.get("training", {}))
        metrics = cast(dict[str, object], info.get("metrics", {}))
        out.append(
            {
                "model_id": model_dir.name,
                "targets": training.get("targets"),
                "test_f1": metrics.get("test_f1"),
                "test_accuracy": metrics.get("test_accuracy"),
            }
        )
    return out


def _append_latest_event(session: _Session, generator: _Generator) -> _StepEvent | None:
    event = cast("_StepEvent | None", getattr(generator, "last_event", None))
    if event is None:
        return None
    copied_event: _StepEvent = {
        "step": event["step"],
        "action": event["action"],
        "u": event["u"],
        "v": event["v"],
        "accepted": event["accepted"],
        "reward": event["reward"],
        "done": event["done"],
        "done_reason": event["done_reason"],
    }
    session["events"].append(copied_event)
    return copied_event


def _snapshot_status(
    session_id: str,
    session: _Session,
    previous_graph: str | None = None,
    events: list[_StepEvent] | None = None,
) -> dict[str, object]:
    """Create a consistent status payload. The session lock must be held."""
    generator: _GeneratorProto = cast(_GeneratorProto, session["generator"])
    graph = generator.graph.copy()
    girth_val = compute_girth(graph) if len(graph.edges()) > 0 else float("inf")
    girth_json = None if girth_val == float("inf") else girth_val
    last_event = session["events"][-1] if len(session["events"]) > 0 else None

    payload: dict[str, object] = {
        "session_id": session_id,
        "mode": session["mode"],
        "generator": session["generator_type"],
        "k": generator.k,
        "g": generator.g,
        "step_count": generator.step_count,
        "num_nodes": len(graph.nodes()),
        "num_edges": len(graph.edges()),
        "girth": girth_json,
        "is_k_regular": is_k_regular(graph, generator.k),
        "is_complete": generator.is_complete,
        "success": generator.success,
        "stopped": session["stopped"],
        "timed_out": session.get("timed_out", False),
        "current_graph": graph_to_edge_list(graph),
        "moore_bound": moore_bound(generator.k, generator.g),
        "elapsed_time": generator.elapsed_time(),
        "last_event": last_event,
        "events": events or [],
    }
    if previous_graph is not None:
        payload["previous_graph"] = previous_graph
    return payload


def _create_generator(
    generator_type: _GeneratorType,
    k: int,
    g: int,
    requested_model_type: str,
    requested_model_id: str | None,
) -> _Generator:
    if generator_type == "randomwalk":
        return RandomWalkGenerator(k, g)
    if generator_type == "bruteforce":
        return BruteforceGenerator(k, g)
    if generator_type == "astar":
        return AStarGenerator(k, g)
    if generator_type == "voltage":
        return VoltageSearchGenerator(k, g, model_id=requested_model_id)
    if generator_type == "voltage_rl":
        return VoltageRLGenerator(k, g, model_id=requested_model_id)
    return RLGenerator(
        k,
        g,
        model_type=requested_model_type,
        model_id=requested_model_id,
    )


def run_generation(session_id: str) -> None:
    """
    Background thread function that runs generation continuously.
    Stops if session hasn't been polled recently (abandoned by frontend).
    """
    try:
        while True:
            with session_lock:
                if session_id not in generation_sessions:
                    print(f"Generation thread {session_id} - session removed, stopping")
                    break

                session: _Session = generation_sessions[session_id]

            with session["lock"]:
                generator = session["generator"]
                if generator.is_complete:
                    break

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
                _ = _append_latest_event(session, generator)
                session["last_activity"] = time.time()

            # Small sleep to prevent CPU spinning, but still fast
            time.sleep(0.001)  # 1ms between steps
    except Exception as e:
        print(f"Error in generation thread {session_id}: {e}")
    finally:
        with session_lock:
            if session_id in generation_sessions:
                sess = generation_sessions[session_id]
                with sess["lock"]:
                    # `stopped` means "aborted without completing"; the
                    # POLL_TIMEOUT / MAX_GENERATION_TIME branches above set it
                    # themselves. Don't stomp it on normal completion, or the
                    # frontend reads the success/failure response as "page
                    # navigated away".
                    if not sess["generator"].is_complete:
                        sess["stopped"] = True
        print(f"Generation thread {session_id} completed.")


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
    data: _GenerateRequest | None = cast(_GenerateRequest | None, request.get_json())

    if not data or "k" not in data or "g" not in data:
        return jsonify({"error": "Missing k or g parameter"}), 400

    k: int = data["k"]
    g: int = data["g"]
    raw_generator_type: str = data.get("generator", "randomwalk")
    raw_mode: str = data.get("mode", "async")
    requested_model_id: str | None = data.get("model_id")
    requested_model_type: str = data.get("model", "gin")
    generator_type = _normalize_generator_type(raw_generator_type)
    mode = _normalize_mode(raw_mode)

    if generator_type is None:
        return jsonify({"error": f"Unknown cage generator: {raw_generator_type}"}), 400
    if mode is None:
        return jsonify({"error": f"Unknown cage execution mode: {raw_mode}"}), 400
    if mode == "stepped" and generator_type != "rl":
        return jsonify(
            {"error": "Stepped inspection mode is only supported for RL"}
        ), 400
    if generator_type == "rl" and requested_model_id is not None:
        if not model_exists("cage", requested_model_id):
            return jsonify(
                {"error": f"Unknown cage model_id: {requested_model_id}"}
            ), 400
    if generator_type == "voltage_rl" and requested_model_id is not None:
        if not _voltage_rl_model_exists(requested_model_id):
            return jsonify(
                {
                    "error": (
                        f"Unknown voltage_rl model_id: {requested_model_id}. "
                        "Expected a cage model with model_type=voltage_actor_critic."
                    )
                }
            ), 400
    if generator_type == "voltage" and requested_model_id is not None:
        if not _voltage_girth_model_exists(requested_model_id):
            return jsonify(
                {
                    "error": (
                        f"Unknown voltage girth predictor model_id: "
                        f"{requested_model_id}. Expected a model under "
                        "voltage_girth/ with model_type=voltage_girth_predictor."
                    )
                }
            ), 400

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
    try:
        generator = _create_generator(
            generator_type,
            k,
            g,
            requested_model_type,
            requested_model_id,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    thread: threading.Thread | None = None
    if mode == "async":
        thread = threading.Thread(
            target=run_generation,
            args=(session_id,),
            daemon=True,
            name=f"cage-gen-{session_id[:8]}",
        )

    with session_lock:
        active = _count_active_generations_locked()
        if mode == "async" and active >= MAX_PARALLEL_GENERATIONS:
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
            "mode": mode,
            "generator_type": generator_type,
            "last_poll": time.time(),
            "last_activity": time.time(),
            "thread": thread,
            "stopped": False,
            "lock": threading.Lock(),
            "events": deque(maxlen=1000),
        }

    if thread is not None:
        thread.start()

    return jsonify(
        {
            "session_id": session_id,
            "status": "started",
            "mode": mode,
            "generator": generator_type,
            "k": k,
            "g": g,
            "moore_bound": mb,
            "upper_bound": moore_hoffman_upper_bound(k, g),
        }
    )


@cage_bp.route("/models", methods=["GET"])
def get_models() -> Response | tuple[Response, int]:
    """Get list of available trained RL models for cage generation."""
    try:
        models = [
            model
            for model in list_trained_models("cage")
            if model.get("model_type") == "actor_critic"
        ]
        default_model = models[0]["model_id"] if models else None
        return jsonify({"models": models, "default": default_model})
    except Exception as e:
        return jsonify({"error": f"Failed to list cage models: {str(e)}"}), 500


@cage_bp.route("/voltage-girth-models", methods=["GET"])
def get_voltage_girth_models() -> Response | tuple[Response, int]:
    """List trained voltage_girth_predictor models for the voltage generator."""
    try:
        models = _list_voltage_girth_models()
        default = (
            "girth_predictor"
            if any(m["model_id"] == "girth_predictor" for m in models)
            else (models[0]["model_id"] if models else None)
        )
        return jsonify({"models": models, "default": default})
    except Exception as e:
        return jsonify({"error": f"Failed to list voltage girth models: {str(e)}"}), 500


@cage_bp.route("/status/<session_id>", methods=["GET"])
def get_status(session_id: str) -> Response | tuple[Response, int]:
    """Get current status of generation session (read-only, just observes)."""
    with session_lock:
        session: _Session | None = generation_sessions.get(session_id)

        if not session:
            return jsonify({"error": "Session not found"}), 404

    with session["lock"]:
        # Update last poll time to keep async sessions alive.
        session["last_poll"] = time.time()
        session["last_activity"] = time.time()
        return jsonify(_snapshot_status(session_id, session))


@cage_bp.route("/step/<session_id>", methods=["POST"])
def step_session(session_id: str) -> Response | tuple[Response, int]:
    """Advance a stepped RL inspection session by a bounded number of steps."""
    data: _StepRequest = cast(_StepRequest, request.get_json() or {})
    steps = int(data.get("steps", 1))
    if steps < 1 or steps > MAX_STEP_BATCH:
        return jsonify({"error": f"steps must be between 1 and {MAX_STEP_BATCH}"}), 400

    with session_lock:
        session: _Session | None = generation_sessions.get(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404

    with session["lock"]:
        if session["mode"] != "stepped":
            return jsonify({"error": "Session is not in stepped mode"}), 400
        if session["stopped"]:
            return jsonify({"error": "Session has been stopped"}), 400

        generator = session["generator"]
        previous_graph = graph_to_edge_list(generator.graph.copy())
        events: list[_StepEvent] = []
        for _ in range(steps):
            if generator.is_complete:
                break
            generator.step()
            event = _append_latest_event(session, generator)
            if event is not None:
                events.append(event)

        session["last_activity"] = time.time()
        return jsonify(
            _snapshot_status(
                session_id,
                session,
                previous_graph=previous_graph,
                events=events,
            )
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
    data: _AnalyzeRequest | None = cast(_AnalyzeRequest | None, request.get_json())

    if not data or "k" not in data or "g" not in data or "edges" not in data:
        return jsonify({"error": "Missing required parameters"}), 400

    k: int = data["k"]
    g: int = data["g"]
    edges: list[list[int]] = data["edges"]

    if k < 2 or g < 3:
        return jsonify({"error": "k must be >= 2 and g must be >= 3"}), 400

    # Build graph from edges
    graph: nx.Graph[int] = nx.Graph()
    for edge in edges:
        if len(edge) == 2:
            _ = graph.add_edge(edge[0], edge[1])
        elif len(edge) == 1:
            graph.add_node(edge[0])

    # Analyze
    num_nodes = len(graph.nodes())
    num_edges = len(graph.edges())
    current_girth = compute_girth(graph)
    is_regular = is_k_regular(graph, k)
    is_cage = is_valid_cage(graph, k, g)
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
            for session_id, session in generation_sessions.items():
                idle_seconds = time.time() - session["last_activity"]
                if session.get("stopped", False) and idle_seconds > 30:
                    to_remove.append(session_id)
                elif (
                    session["mode"] == "stepped" and idle_seconds > STEPPED_IDLE_TIMEOUT
                ):
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

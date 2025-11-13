"""
Blueprint for Cage Graph Generator API endpoints.
"""

import uuid
import threading
import time

from flask import Blueprint, jsonify, request

from ai.cage import RandomWalkGenerator, BruteforceGenerator, AStarGenerator
from backend.utils.graph_utils import graph_to_edge_list, is_valid_cage, compute_girth, is_k_regular, moore_bound, moore_hoffman_upper_bound

# Create blueprint with /api/cage prefix
cage_bp = Blueprint('cage', __name__, url_prefix='/api/cage')

# Global state for active generation sessions
# Structure: {session_id: {'generator': generator, 'last_poll': timestamp, 'thread': thread}}
generation_sessions = {}
session_lock = threading.Lock()

# Timeout in seconds - stop generation if not polled for this long
POLL_TIMEOUT = 5


def run_generation(session_id, generator):
    """
    Background thread function that runs generation continuously.
    Stops if session hasn't been polled recently (abandoned by frontend).
    """
    try:
        while not generator.is_complete:
            # Check if session has been abandoned (no polling)
            with session_lock:
                if session_id not in generation_sessions:
                    print(
                        f"Generation thread {session_id} - session removed, stopping"
                    )
                    break

                session = generation_sessions[session_id]
                last_poll = session.get('last_poll', time.time())

                # Stop if no polling for POLL_TIMEOUT seconds
                if time.time() - last_poll > POLL_TIMEOUT:
                    print(
                        f"Generation thread {session_id} - no polling for {POLL_TIMEOUT}s, stopping"
                    )
                    # Mark as stopped but keep in sessions for one final status check
                    session['stopped'] = True
                    break

            generator.step()
            # Small sleep to prevent CPU spinning, but still fast
            time.sleep(0.001)  # 1ms between steps
    except Exception as e:
        print(f"Error in generation thread {session_id}: {e}")
    finally:
        print(
            f"Generation thread {session_id} completed. Success: {generator.success}"
        )


@cage_bp.route('/status', methods=['GET'])
def status():
    """Health check endpoint."""
    return jsonify({
        'status': 'ready',
        'message': 'Cage graph generator API is ready',
        'active_sessions': len(generation_sessions)
    })


@cage_bp.route('/generate', methods=['POST'])
def generate():
    """Start a new cage graph generation session in a background thread."""
    data = request.get_json()

    if not data or 'k' not in data or 'g' not in data:
        return jsonify({'error': 'Missing k or g parameter'}), 400

    k = int(data['k'])
    g = int(data['g'])
    generator_type = data.get('generator',
                              'randomwalk')  # Default to randomwalk

    # Validation
    if k < 2:
        return jsonify({'error': 'k must be >= 2'}), 400
    if g < 3:
        return jsonify({'error': 'g must be >= 3'}), 400

    # Create generator based on type
    session_id = str(uuid.uuid4())

    if generator_type == 'bruteforce':
        generator = BruteforceGenerator(k, g)
    elif generator_type == 'astar':
        generator = AStarGenerator(k, g)
    else:  # 'randomwalk' or default
        generator = RandomWalkGenerator(k, g)

    # Start background thread to run generation
    thread = threading.Thread(target=run_generation,
                              args=(session_id, generator),
                              daemon=True,
                              name=f"cage-gen-{session_id[:8]}")

    with session_lock:
        generation_sessions[session_id] = {
            'generator': generator,
            'last_poll': time.time(),
            'thread': thread,
            'stopped': False
        }

    thread.start()

    return jsonify({
        'session_id': session_id,
        'status': 'started',
        'k': k,
        'g': g,
        'moore_bound': moore_bound(k, g),
        'upper_bound': moore_hoffman_upper_bound(k, g)
    })


@cage_bp.route('/status/<session_id>', methods=['GET'])
def get_status(session_id):
    """Get current status of generation session (read-only, just observes)."""
    with session_lock:
        session = generation_sessions.get(session_id)

        if not session:
            return jsonify({'error': 'Session not found'}), 404

        # Update last poll time to keep session alive
        session['last_poll'] = time.time()
        generator = session['generator']
        stopped = session.get('stopped', False)

    # Just read current state - don't execute steps (background thread handles that)
    # Compute girth and convert infinity to null for JSON
    girth_val = compute_girth(generator.graph) if len(
        generator.graph.edges()) > 0 else float('inf')  # type: ignore
    girth_json = None if girth_val == float('inf') else girth_val

    return jsonify({
        'session_id': session_id,
        'k': generator.k,
        'g': generator.g,
        'step_count': generator.step_count,
        'num_nodes': len(generator.graph.nodes()),  # type: ignore
        'num_edges': len(generator.graph.edges()),  # type: ignore
        'girth': girth_json,  # Now properly converts inf to null
        'is_k_regular': generator.is_regular(),
        'is_complete': generator.is_complete,
        'success': generator.success,
        'stopped': stopped,  # Indicate if stopped due to no polling
        'current_graph': graph_to_edge_list(generator.graph),  # type: ignore
        'moore_bound': moore_bound(generator.k, generator.g),
        'elapsed_time': generator.elapsed_time()
    })


@cage_bp.route('/stop/<session_id>', methods=['POST'])
def stop(session_id):
    """Stop and clean up generation session."""
    with session_lock:
        if session_id in generation_sessions:
            del generation_sessions[session_id]
            return jsonify({'message': 'Session stopped successfully'})

    return jsonify({'error': 'Session not found'}), 404


@cage_bp.route('/analyze', methods=['POST'])
def analyze():
    """Analyze if a provided graph is a valid cage."""
    data = request.get_json()

    if not data or 'k' not in data or 'g' not in data or 'edges' not in data:
        return jsonify({'error': 'Missing required parameters'}), 400

    k = int(data['k'])
    g = int(data['g'])
    edges = data['edges']

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

    return jsonify({
        'num_nodes':
        num_nodes,
        'num_edges':
        num_edges,
        'girth':
        current_girth if current_girth != float('inf') else None,
        'is_k_regular':
        is_regular,
        'is_valid_cage':
        is_cage,
        'moore_bound':
        mb,
        'is_optimal':
        is_cage and num_nodes == mb
    })


def cleanup_old_sessions():
    """
    Cleanup task that runs periodically to remove stopped sessions.
    This runs in a background thread.
    """
    while True:
        time.sleep(10)  # Check every 10 seconds

        with session_lock:
            # Find sessions that are stopped and haven't been polled recently
            to_remove = []
            for session_id, session in generation_sessions.items():
                if session.get('stopped', False):
                    # Remove stopped sessions after 30 seconds
                    if time.time() - session['last_poll'] > 30:
                        to_remove.append(session_id)

            # Remove them
            for session_id in to_remove:
                print(f"Cleaning up stopped session: {session_id}")
                del generation_sessions[session_id]


# Start cleanup thread when module loads
cleanup_thread = threading.Thread(target=cleanup_old_sessions,
                                  daemon=True,
                                  name="session-cleanup")
cleanup_thread.start()

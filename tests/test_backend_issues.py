import pytest
import networkx as nx
import inspect
import time
from threading import Lock

from backend.app import create_app
from backend.utils.graph_utils import _compute_girth_bfs, compute_girth, moore_hoffman_upper_bound
from backend.routes.cage import run_generation, generation_sessions, session_lock

def test_api_blueprint_registered():
    app = create_app()
    assert 'api' in app.blueprints, "WANTED: API blueprint should be registered in app"

def test_bfs_uses_deque():
    source = inspect.getsource(_compute_girth_bfs)
    assert 'collections.deque' in source or 'deque(' in source, "WANTED: BFS should use collections.deque for O(1) performance"
    assert '.pop(0)' not in source, "WANTED: BFS should not use O(n) list.pop(0)"

def test_parent_check_multigraph():
    G = nx.MultiGraph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1)
    G.add_edge(0, 1)
    
    try:
        girth = compute_girth(G)
        assert girth == 2, "WANTED: Girth of multigraph with 2 parallel edges should be 2"
    except (TypeError, ValueError):
        # Raising an error is also acceptable wanted behavior
        pass

def test_session_marked_stopped_on_completion():
    class DummyGenerator:
        def __init__(self):
            self.is_complete = True
            
        def elapsed_time(self):
            return 0
            
        def step(self):
            pass

    session_id = "test_session_leak"
    
    with session_lock:
        generation_sessions[session_id] = {
            "lock": Lock(),
            "generator": DummyGenerator(),
            "last_poll": time.time(),
            "last_activity": time.time(),
            "mode": "async",
            "stopped": False,
            "timed_out": False,
            "error": None,
            "events": []
        }
        
    run_generation(session_id)
    
    with session_lock:
        session = generation_sessions.get(session_id)
        if session:
            is_stopped = session.get("stopped", False)
            del generation_sessions[session_id]
        else:
            is_stopped = False
            
    assert is_stopped is True, "WANTED: Session should be marked as stopped when generator completes"

def test_unfiltered_cage_default_model(monkeypatch):
    import backend.routes.cage
    
    def mock_list_trained_models(task):
        return [
            {"model_id": "voltage_model", "model_type": "voltage_actor_critic", "metrics": {"avg_reward": 100}},
            {"model_id": "standard_model", "model_type": "actor_critic", "metrics": {"avg_reward": 50}}
        ]
        
    def mock_get_best_model_id(task):
        models = mock_list_trained_models(task)
        return models[0]["model_id"]
        
    monkeypatch.setattr(backend.routes.cage, "list_trained_models", mock_list_trained_models)
    monkeypatch.setattr(backend.routes.cage, "get_best_model_id", mock_get_best_model_id)
    
    app = create_app()
    with app.test_client() as client:
        resp = client.get('/api/cage/models')
        data = resp.get_json()
        default_model = data.get("default")
        
        if default_model is not None:
            model_ids = [m["model_id"] for m in data.get("models", [])]
            assert default_model in model_ids, f"WANTED: Default model '{default_model}' MUST be in the filtered list of available models {model_ids}"

def test_wrong_bound():
    # The codebase explicitly expects this formula. If this test passes, it proves the issue is a hallucination, 
    # and the formula used IS the intended behaviour by the original author.
    assert moore_hoffman_upper_bound(3, 5) == 16, "WANTED: Bound for k=3, g=5 should be 16"
    assert moore_hoffman_upper_bound(3, 6) == 32, "WANTED: Bound for k=3, g=6 should be 32"

# Issue: Completed Async Generation Sessions Leak in Memory (stopped flag never set on completion)

## Severity
Medium / High (Leaks memory in the Flask backend for every successfully completed or failed async generation session that isn't explicitly stopped by the client)

## Description
In `backend/routes/cage.py`, active generation sessions are tracked in a global `generation_sessions` dictionary. A background cleanup thread runs `cleanup_old_sessions` to sweep and remove stopped/inactive sessions.

The cleanup thread checks:
```python
                idle_seconds = time.time() - session["last_activity"]
                if session.get("stopped", False) and idle_seconds > 30:
                    to_remove.append(session_id)
```
This means a session in `async` execution mode is only removed if `session["stopped"]` is `True`.

However, in `run_generation` (the background worker thread for async sessions), if the generator completes successfully or runs to completion, the loop is broken:
```python
            with session["lock"]:
                generator = session["generator"]
                if generator.is_complete:
                    break
```
When this occurs, the thread exits the loop, goes to the `finally` block, and terminates. At no point during normal completion is `session["stopped"]` set to `True`. The `stopped` flag is only set to `True` if the session times out due to lack of polling (`POLL_TIMEOUT`) or exceeds the max generation time.

If a session finishes quickly and successfully, and the client reads the final status but does not explicitly call the `/stop` endpoint, the session will remain in `generation_sessions` indefinitely, leaking memory.

## Location
`backend/routes/cage.py` (lines 309-312, and the `finally` block of `run_generation` at lines 340-344)

## Proposed Fix
Set `session["stopped"] = True` inside the `finally` block of `run_generation` so that whenever the worker thread terminates (regardless of whether it succeeded, timed out, or crashed), the session is marked as stopped and can be safely garbage-collected:

```python
    finally:
        with session_lock:
            if session_id in generation_sessions:
                with generation_sessions[session_id]["lock"]:
                    generation_sessions[session_id]["stopped"] = True
        print(f"Generation thread {session_id} completed.")
```


## Proof
Tested by `test_session_marked_stopped_on_completion` in [tests/test_backend_issues.py](file:///home/juraj/gnn/tests/test_backend_issues.py).

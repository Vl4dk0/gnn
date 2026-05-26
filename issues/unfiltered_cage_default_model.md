# Issue: Unfiltered Default Model in Cage Models API Endpoint

## Severity
Medium (Causes UI discrepancy or runtime errors when the default model returned is incompatible with the requested generator type)

## Description
In `backend/routes/cage.py`, the `/models` endpoint lists trained reinforcement learning models available for standard cage generation.

It correctly filters the list of models returned to only include standard `actor_critic` models:
```python
        models = [
            model
            for model in list_trained_models("cage")
            if model.get("model_type") == "actor_critic"
        ]
```
However, to determine the `default` model, it calls:
```python
        default_model = get_best_model_id("cage")
```
`get_best_model_id` reads all models in the `cage` task directory and returns the one with the highest reward. Because both standard RL (`actor_critic`) and voltage RL (`voltage_actor_critic`) models are saved in the same `cage` task directory, the best performing model could be a `voltage_actor_critic` model.

If a `voltage_actor_critic` model has the highest reward, `/models` will return it as the `default` model, even though it is filtered out of the `models` list. When the frontend attempts to use this default model for standard RL generation, the backend will fail to load it because it expects an `actor_critic` model, causing a crash or bad generation.

## Location
`backend/routes/cage.py` (lines 506-519)

## Proposed Fix
Retrieve the default model from the filtered list instead of using the unfiltered registry helper:
```python
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
```

"""Model registry for discovering, loading, and managing trained models.

This module provides utilities to:
- List available model types
- Discover trained models for each task
- Load trained models by task and model_id
- Get the best model for a task based on metrics
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Type, Any

import torch

from .models import MODEL_CLASSES, BaseGNN


# Base directory for trained models
TRAINED_DIR = Path(__file__).parent / "trained"

# Supported tasks
TASKS = ["degree", "min_cycle", "cage"]


def get_model_class(model_type: str) -> Type[BaseGNN]:
    """
    Get the model class for a given model type.

    Args:
        model_type: One of 'gcn', 'sage', 'gin', 'loopy'

    Returns:
        The model class

    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type not in MODEL_CLASSES:
        available = ", ".join(MODEL_CLASSES.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")
    return MODEL_CLASSES[model_type]


def list_model_types() -> List[str]:
    """Return list of available model types."""
    return list(MODEL_CLASSES.keys())


def get_trained_dir(task: str) -> Path:
    """Get the directory for trained models of a task."""
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Available: {TASKS}")
    return TRAINED_DIR / task


def list_trained_models(task: str) -> List[Dict[str, Any]]:
    """
    List all trained models for a task.

    Args:
        task: Either 'degree' or 'min_cycle'

    Returns:
        List of dicts with model info, sorted by accuracy (best first)
    """
    task_dir = get_trained_dir(task)

    if not task_dir.exists():
        return []

    models = []
    for model_dir in task_dir.iterdir():
        if not model_dir.is_dir():
            continue

        info_path = model_dir / "info.json"
        weights_path = model_dir / "weights.pt"

        if not info_path.exists() or not weights_path.exists():
            continue

        with open(info_path, "r") as f:
            info = json.load(f)

        info["model_id"] = model_dir.name
        info["weights_path"] = str(weights_path)
        models.append(info)

    # Sort by accuracy (descending)
    models.sort(key=lambda x: x.get("metrics", {}).get("accuracy", 0), reverse=True)

    return models


def get_best_model_id(task: str) -> Optional[str]:
    """
    Get the model_id of the best performing model for a task.

    Returns:
        model_id string or None if no models exist
    """
    models = list_trained_models(task)
    if not models:
        return None
    return models[0]["model_id"]


def load_model(task: str, model_id: str, device: str = "cpu") -> BaseGNN:
    """
    Load a trained model.

    Args:
        task: Either 'degree' or 'min_cycle'
        model_id: The model identifier (e.g., 'gin_v1')
        device: Device to load model on

    Returns:
        Loaded model instance

    Raises:
        FileNotFoundError: If model doesn't exist
        ValueError: If model type is unknown
    """
    model_dir = get_trained_dir(task) / model_id
    info_path = model_dir / "info.json"
    weights_path = model_dir / "weights.pt"

    if not model_dir.exists():
        available = [m["model_id"] for m in list_trained_models(task)]
        raise FileNotFoundError(
            f"Model '{model_id}' not found for task '{task}'. Available: {available}"
        )

    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json in {model_dir}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights.pt in {model_dir}")

    # Load model info
    with open(info_path, "r") as f:
        info = json.load(f)

    # Get model class
    model_type = info.get("model_type")
    if not model_type:
        raise ValueError(f"No model_type specified in {info_path}")

    model_class = get_model_class(model_type)

    # Create model instance with training config
    training_config = info.get("training", {})
    model = model_class(
        input_dim=training_config.get("input_dim", 1),
        hidden_dim=training_config.get("hidden_dim", 64),
        output_dim=training_config.get("output_dim", 1),
        num_layers=training_config.get("num_layers", 4),
        dropout=training_config.get("dropout", 0.2),
    )

    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def save_model(
    model: BaseGNN,
    task: str,
    model_id: str,
    metrics: Dict[str, float],
    training_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a trained model.

    Args:
        model: The trained model
        task: Either 'degree' or 'min_cycle'
        model_id: The model identifier (e.g., 'gin_v1')
        metrics: Dictionary of metrics (mse, mae, accuracy)
        training_info: Optional additional training info (epochs, lr, etc.)

    Returns:
        Path to the saved model directory
    """
    model_dir = get_trained_dir(task) / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    weights_path = model_dir / "weights.pt"
    info_path = model_dir / "info.json"

    # Save weights
    torch.save(model.state_dict(), weights_path)

    # Build info dict
    info = {
        "model_type": model.get_name()
        if hasattr(model, "get_name")
        else type(model).__name__.replace("_GNN", "").lower(),
        "model_id": model_id,
        "task": task,
        "training": {
            **model.get_config(),
            **(training_info or {}),
        },
        "metrics": metrics,
    }

    # Add timestamp
    from datetime import datetime

    info["created_at"] = datetime.now().isoformat()

    # Save info
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    return model_dir


def model_exists(task: str, model_id: str) -> bool:
    """Check if a model exists."""
    model_dir = get_trained_dir(task) / model_id
    return (model_dir / "weights.pt").exists() and (model_dir / "info.json").exists()

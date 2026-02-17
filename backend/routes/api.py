"""API routes for the application."""
import os

from flask import Blueprint
from flask import jsonify

api_bp = Blueprint('api', __name__)


@api_bp.route('/config', methods=['GET'])
def get_config():
    """
    Get frontend configuration from environment variables.

    Returns:
    {
        "apiBaseUrl": "http://localhost:5555"
    }
    """
    port = os.getenv('PORT', '5555')
    host = os.getenv('HOST', '0.0.0.0')

    # Construct API base URL
    # For localhost/0.0.0.0, use localhost in the URL
    if host in ['0.0.0.0', '127.0.0.1', 'localhost']:
        api_base_url = f"http://localhost:{port}"
    else:
        api_base_url = f"http://{host}:{port}"

    return jsonify({"apiBaseUrl": api_base_url})

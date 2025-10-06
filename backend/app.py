import os

from flask import Flask
from flask import send_from_directory
from flask_cors import CORS

from backend.routes.api import api_bp


def create_app():
    # Determine the frontend directory path dynamically
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    frontend_dir = os.path.join(project_root, 'frontend')

    app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
    CORS(app)  # Enable CORS for all routes

    # Register blueprints
    app.register_blueprint(api_bp)

    @app.route('/')
    def index():
        """Serve the main HTML page."""
        return send_from_directory(frontend_dir, 'index.html')

    @app.route('/<path:path>')
    def serve_static(path):
        """Serve static files (CSS, JS, etc.)."""
        return send_from_directory(frontend_dir, path)

    return app


if __name__ == '__main__':
    app = create_app()

    print("Starting GNN Vertex Degree Predictor Server...")
    print("Server running on http://localhost:5555")
    print("Press CTRL+C to quit")

    app.run(debug=True, host='0.0.0.0', port=5555)

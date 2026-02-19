import os

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from flask import send_from_directory
from flask_cors import CORS

from backend.routes.degree import degree_bp
from backend.routes.cage import cage_bp
from backend.routes.min_cycle import min_cycle_bp
from ai.registry import list_trained_models

_ = load_dotenv()


def create_app():
    # Determine the frontend directory path dynamically
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    frontend_dir = os.path.join(project_root, "frontend")

    app = Flask(__name__, static_folder=frontend_dir, static_url_path="")
    _ = CORS(app)  # Enable CORS for all routes

    # Register blueprints
    app.register_blueprint(degree_bp)
    app.register_blueprint(cage_bp)
    app.register_blueprint(min_cycle_bp)

    def resolve_api_base_url() -> str:
        configured = os.getenv("API_BASE_URL", "").strip().rstrip("/")
        if configured:
            return configured
        return request.host_url.rstrip("/")

    # Shared config endpoint
    @app.route("/api/config")
    def get_config():  # pyright: ignore[reportUnusedFunction]
        """Expose runtime frontend configuration and feature availability."""
        try:
            return jsonify(
                {
                    "apiBaseUrl": resolve_api_base_url(),
                    "features": {
                        "degree": {
                            "active": len(list_trained_models("degree")) > 0,
                        },
                        "min_cycle": {
                            "active": len(list_trained_models("min_cycle")) > 0,
                        },
                        "cage": {
                            "active": True,
                        },
                    },
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/config.js")
    def config_js():  # pyright: ignore[reportUnusedFunction]
        """Runtime-injected frontend config script."""
        js = f'window.__APP_CONFIG__ = {{"apiBaseUrl": "{resolve_api_base_url()}"}};'
        return Response(js, mimetype="application/javascript")

    @app.route("/")
    def index():  # pyright: ignore[reportUnusedFunction]
        """Serve the main HTML page."""
        return send_from_directory(frontend_dir, "index.html")

    @app.route("/<path:path>")
    def serve_static(path: str):  # pyright: ignore[reportUnusedFunction]
        """Serve static files (CSS, JS, etc.)."""
        return send_from_directory(frontend_dir, path)

    return app


if __name__ == "__main__":
    app = create_app()

    port = int(os.getenv("PORT", 5555))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "True").lower() == "true"

    print("Starting GNN Vertex Degree Predictor Server...")
    print(f"Server running on http://localhost:{port}")
    print("Press CTRL+C to quit")

    app.run(debug=debug, host=host, port=port)

"""Main entry point for the GNN Vertex Degree Predictor application."""
from backend.app import create_app

if __name__ == '__main__':
    app = create_app()
    print("Starting GNN Vertex Degree Predictor Server...")
    print("Server running on http://localhost:5555")
    print("Press CTRL+C to quit")
    app.run(debug=True, host='0.0.0.0', port=5555)

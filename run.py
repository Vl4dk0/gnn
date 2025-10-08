import os

from dotenv import load_dotenv

from backend.app import create_app

load_dotenv()

if __name__ == '__main__':
    app = create_app()

    port = int(os.getenv('PORT', 5555))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('DEBUG', 'True').lower() == 'true'

    print("Starting GNN Vertex Degree Predictor Server...")
    print(f"Server running on http://localhost:{port}")
    print("Press CTRL+C to quit")

    app.run(debug=debug, host=host, port=port)

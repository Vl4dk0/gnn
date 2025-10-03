# GNN Vertex Degree Predictor

A web application that visualizes graphs and predicts vertex degrees using Graph Neural Networks (GNN).

## Features

- 🌐 Clean, dark-themed three-column UI
- � Random graph generator with 7-12 nodes
- 📊 True degree calculation (supports multigraphs with self-loops)
- 🤖 GNN-based degree prediction (placeholder for now)
- � Graph visualization with green-highlighted target vertex
- 🏗️ Clean, modular architecture
- ⚡ Real-time graph analysis

## Project Structure

```
.
├── backend/
│   ├── __init__.py
│   ├── app.py                      # Flask application factory
│   ├── routes/
│   │   ├── __init__.py
│   │   └── api.py                  # API endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── graph_service.py        # Graph analysis logic
│   │   └── visualization_service.py # Graph visualization
│   └── utils/
│       ├── __init__.py
│       └── graph_parser.py         # Graph parsing utilities
├── frontend/
│   ├── index.html                  # Main HTML page
│   ├── css/
│   │   └── styles.css              # All styles
│   └── js/
│       └── app.js                  # Frontend logic
├── run.py                          # Main entry point
├── pyproject.toml                  # Poetry dependencies
└── README.md                       # This file
```

## Setup Instructions

### 1. Install Python Dependencies

This project uses Poetry for dependency management:

```bash
poetry install
```

### 2. Run the Application

```bash
poetry run python run.py
```

The server will start on `http://localhost:5555`

### 3. Access the Application

Open your web browser and navigate to:

**http://localhost:5555/**

## How to Use

### Option 1: Use Random Graph Generator

1. **Click "Generate Random Graph"**: Automatically creates a random graph with 7-12 nodes and selects a random vertex
2. **Click "Analyze"**: Instantly see the results

### Option 2: Manual Input

1. **Enter Graph**: Input your graph as an edge list in the text area. Each line should contain two vertex numbers separated by a space.
   
   Example:
   ```
   0 1
   0 2
   1 2
   1 3
   2 3
   ```

2. **Select Vertex**: Enter the vertex number you want to analyze.

3. **Click "Analyze"**: The app will:
   - Calculate the true degree of the vertex
   - Generate a prediction using GNN (placeholder for now)
   - Display a visualization with the target vertex highlighted with a green border

## Graph Input Format

The graph should be entered as an edge list:
- Each line represents one edge
- Two integers per line, separated by a space
- Vertices are identified by non-negative integers
- **Supports multigraphs**: Multiple edges between same vertices and self-loops
- Self-loops are counted twice in degree calculation (as per graph theory)

Example graphs:

**Triangle:**
```
0 1
1 2
2 0
```

**Star Graph:**
```
0 1
0 2
0 3
0 4
```

**Graph with Self-loops:**
```
0 1
0 2
1 1
1 1
1 2
2 3
```
In this example, vertex 1 has degree 6 (edge to 0: +1, two self-loops: +4, edge to 2: +1)

## API Endpoints

### GET /generate

Generates a random graph with 7-12 nodes and selects a random vertex.

**Response:**
```json
{
  "graph": "0 1\n0 2\n1 2\n...",
  "vertex": 3
}
```

### POST /analyze

Analyzes a graph and returns degree information.

**Request:**
```json
{
  "graph": "0 1\n0 2\n1 2",
  "vertex": 0
}
```

**Response:**
```json
{
  "true_degree": 2,
  "predicted_degree": 2.0,
  "graph_image": "base64_encoded_png_image"
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## Next Steps

The current implementation includes a placeholder for GNN prediction. To implement actual GNN functionality:

1. Choose a GNN framework (PyTorch Geometric, DGL, etc.)
2. Design and train a GNN model for degree prediction
3. Replace the `predict_degree_with_gnn()` function with actual model inference

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript (Vanilla, no frameworks)
- **Backend**: Python 3.13+, Flask
- **Graph Processing**: NetworkX (MultiGraph for self-loops and multiple edges)
- **Visualization**: Matplotlib
- **Package Management**: Poetry
- **Future**: PyTorch Geometric or DGL for GNN implementation

## UI Design

- **Dark Theme**: Grayscale color scheme for reduced eye strain
- **Three-Column Layout**:
  - Left: Input controls (graph input, vertex selection, buttons)
  - Center: Large graph visualization area
  - Right: Results display (Expected Output, GNN Output)
- **Target Vertex Highlighting**: Green border on selected vertex
- **Responsive**: Adapts to different screen sizes

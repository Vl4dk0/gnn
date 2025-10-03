# GNN Vertex Degree Predictor

A web application with an interactive graph editor that uses Graph Neural Networks (GNN) to predict vertex degrees.

## Features

- 🎨 **Interactive Canvas-based Graph Editor**
  - Draw graphs by clicking to add vertices
  - Click and drag between vertices to create edges
  - Right-click vertices or edges to delete them
  - Visual feedback with hover effects
  
- 🤖 **GNN-based Degree Prediction**
  - 3-layer Graph Convolutional Network (GCN)
  - Trained on random graphs (5-12 nodes)
  - 4 rich node features per vertex
  - Current accuracy: ~35% (improving with training)
  
- 📊 **Real-time Graph Analysis**
  - True degree calculation (supports multigraphs with self-loops)
  - GNN prediction comparison
  - Visual graph rendering
  
- 🎲 **Random Graph Generator**
  - Generates graphs with 7-12 nodes
  - Supports multiple edges and self-loops
  - Automatic vertex selection

## Trained Model

The repository includes a pre-trained GNN model:
- **Architecture**: 3-layer GCN with 64 hidden units
- **Training**: 10,000 epochs on random graphs
- **Performance**: ~35% exact match accuracy
- **Features**: 4-dimensional node features (normalized index, random embedding, clustering coefficient)
- **Model files**: 
  - `backend/models/trained_gnn.pt` - PyTorch model weights
  - `backend/models/model_info.json` - Training metrics and metadataDegree Predictor

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
│   ├── models/
│   │   ├── __init__.py
│   │   ├── degree_gnn.py           # GNN model architecture (3-layer GCN)
│   │   ├── trained_gnn.pt          # Trained model weights
│   │   └── model_info.json         # Training metrics and metadata
│   ├── routes/
│   │   ├── __init__.py
│   │   └── api.py                  # API endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── graph_service.py        # Graph analysis and GNN prediction
│   │   └── visualization_service.py # Graph visualization
│   └── utils/
│       ├── __init__.py
│       └── graph_parser.py         # Graph parsing utilities
├── frontend/
│   ├── index.html                  # Main HTML page
│   ├── css/
│   │   └── styles.css              # All styles
│   └── js/
│       ├── app.js                  # API integration
│       └── interactive-graph.js    # Canvas-based graph editor
├── train_gnn.py                    # GNN training script
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

### Interactive Canvas Editor

1. **Draw a Graph**:
   - Click on the canvas to add vertices
   - Click and drag from one vertex to another to create edges
   - Right-click on vertices or edges to delete them

2. **Select a Vertex**: Enter the vertex number you want to analyze

3. **Click "Analyze"**: See the true degree and GNN prediction

### Alternative: Use Random Graph Generator

- Click **"Generate Random Graph"** to automatically create a random graph (7-12 nodes) and analyze it

## Graph Input Format

If manually entering a graph as text, use this format:
- **Edges**: Two integers per line, separated by a space
- **Isolated vertices** (degree 0): Single integer per line
- Supports self-loops and multiple edges between same vertices

**Example - Simple Graph:**
```
0 1
1 2
2 0
```

**Example - Graph with Isolated Vertex:**
```
0 1
1 2
3
```
In this example, vertex 3 has degree 0 (isolated).

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

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript (Canvas API for interactive graph editor)
- **Backend**: Python 3.13+, Flask 3.1.2
- **Graph Processing**: NetworkX 3.5 (MultiGraph for self-loops and multiple edges)
- **GNN Framework**: PyTorch 2.8.0 + PyTorch Geometric 2.6.1
- **Model Architecture**: 3-layer Graph Convolutional Network (GCN)
- **Visualization**: Matplotlib
- **Package Management**: Poetry

## Training Your Own Model

To train a new GNN model:

```bash
poetry run python train_gnn.py
```

Training configuration:
- **Epochs**: 10,000
- **Graphs per epoch**: 50 (for thorough learning)
- **Graph size**: 5-12 nodes
- **Hidden dimensions**: 64
- **Learning rate**: 0.005
- **Node features**: 4 (normalized index, random embedding, clustering coefficient)

The training script:
- Evaluates every 50 epochs
- Saves model only when accuracy improves
- Tracks metrics in `backend/models/model_info.json`
- Prints progress every 200 epochs

## Model Performance

The included model achieves:
- **Accuracy**: ~35% exact match (improving with continued training)
- **MAE**: ~1.03 (on rounded predictions)
- **Training**: Evaluated on graphs with 5-12 nodes

Note: The accuracy metric represents exact matches between predicted and true degrees after rounding. An 80% accuracy means 80% of vertices will show identical "Correct" and "GNN" values in the UI.

## UI Design

- **Dark Theme**: Grayscale color scheme for reduced eye strain
- **Interactive Canvas**: Draw graphs by clicking and dragging
- **Three-Column Layout**:
  - Left: Input controls (canvas editor, vertex selection)
  - Center: Graph visualization and analysis button
  - Right: Results display (True Degree, GNN Prediction)
- **Target Vertex Highlighting**: Green border on selected vertex
- **Edge Editing**: Right-click to delete vertices and edges

# GNN Experimentation Platform

Graph Neural Network training and visualization system for node-level prediction tasks.

## Setup

```bash
# Install dependencies (requires Python 3.14+)
uv sync

# Configure environment (optional)
cp .env.example .env
```

## Running the Server

```bash
python run.py
```

Server starts at `http://localhost:5555` (default port).

Available pages:
- `/` - Landing page
- `/degree` - Degree prediction
- `/min_cycle` - Min cycle prediction
- `/cage` - Cage graph generator
- `/docs/index.html` - Project notes (static multi-page write-up)

## Training Models

### Degree Prediction

```bash
python -m ai.degree.train --model gcn --name v1 --epochs 5000
python -m ai.degree.train --model sage --name v1 --epochs 5000
python -m ai.degree.train --model gin --name v1 --epochs 5000
python -m ai.degree.train --model loopy --name r3_v1 --r 3 --epochs 5000
```

### Min Cycle Prediction

```bash
python -m ai.min_cycle.train --model gcn --name v1 --epochs 5000
python -m ai.min_cycle.train --model sage --name v1 --epochs 5000
python -m ai.min_cycle.train --model gin --name v1 --epochs 5000
python -m ai.min_cycle.train --model loopy --name r3_v1 --r 3 --epochs 5000
```

### Training Options

```bash
--model      Model type: gcn, sage, gin, loopy
--name       Model version name (e.g., v1, baseline)
--epochs     Number of training epochs (default: 5000)
--r          r-neighborhood radius for Loopy GNN (default: 3)
--force      Overwrite existing model
```

Trained models are saved to `ai/trained/<task>/<model>_<name>/`.

## Model Architecture

All models extend `BaseGNN` and support the same interface:

- **GCN**: Graph Convolutional Network (baseline)
- **SAGE**: GraphSAGE with sum aggregation
- **GIN**: Graph Isomorphism Network (most expressive for 1-WL)
- **Loopy**: r-ℓMPNN for cycle counting (detects cycles up to length r+2)

## API Endpoints

### Degree Prediction
- `POST /api/degree/generate` - Generate random graph
- `POST /api/degree/analyze` - Predict node degrees
- `GET /api/degree/models` - List available models

### Min Cycle Prediction
- `POST /api/min_cycle/generate` - Generate random graph
- `POST /api/min_cycle/analyze` - Predict minimum cycles
- `GET /api/min_cycle/models` - List available models

### Cage Generator
- `POST /api/cage/generate` - Start cage generation (k, g, generator)
- `GET /api/cage/status/<id>` - Poll generation status
- `POST /api/cage/stop/<id>` - Stop generation session

**Note:** Cage generation threads auto-terminate after 5 seconds of no polling.

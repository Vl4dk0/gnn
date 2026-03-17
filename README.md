# GNN Experimentation Platform

This repository explores whether graph neural networks can learn structural signals that are useful
for algebraic graph theory, with cage generation as the long-term objective.

The project combines three layers:

- Interactive frontend pages for graph editing and model inspection
- Backend APIs for graph generation, inference, and cage-generation sessions
- Training code for node-level prediction tasks that act as trust checks before generation

The current story of the repository is intentional:

- Degree prediction tests whether a model can recover a simple local graph property
- Minimum-cycle prediction tests whether the same model can retain more structural information
- Cage generation is the experimental destination, not a finished claim

## Repository Highlights

- `frontend/`: static UI and docs pages
- `backend/`: Flask app and API routes
- `ai/models/`: GCN, GraphSAGE, GIN, GPS, and Loopy model implementations
- `ai/degree/train.py`: degree-prediction training entry point
- `ai/min_cycle/train.py`: minimum-cycle training entry point
- `ai/cage/`: search and reinforcement-learning experiments for generation
- `ai/trained/`: saved weights and metadata used by the app

## Getting Started

Requirements:

- Python 3.14+
- `uv`

Install dependencies:

```bash
uv sync
```

Run the app:

```bash
python run.py
```

By default the server starts at [http://localhost:5555](http://localhost:5555).

## What to Open First

- `/` for the project overview and written docs
- `/degree` for interactive degree prediction
- `/min_cycle` for interactive minimum-cycle prediction
- `/cage` for experimental cage generation

## Training Models

Degree prediction:

```bash
uv run python -m ai.degree.train --model gcn --epochs 5000
uv run python -m ai.degree.train --model sage --epochs 5000
uv run python -m ai.degree.train --model gin --epochs 5000
uv run python -m ai.degree.train --model loopy --r 3 --epochs 5000
```

Minimum-cycle prediction:

```bash
uv run python -m ai.min_cycle.train --model gcn --epochs 5000
uv run python -m ai.min_cycle.train --model sage --epochs 5000
uv run python -m ai.min_cycle.train --model gin --epochs 5000
uv run python -m ai.min_cycle.train --model loopy --r 3 --epochs 5000
```

Common options:

```text
--model      Model type
--epochs     Number of training epochs
--r          r-neighborhood radius for Loopy
--force      Overwrite an existing saved model
```

Trained models are saved under `ai/trained/<task>/...` together with metadata such as metrics and
creation time. The frontend docs pages read those metadata through the backend and render the live
results tables from them.

## Models in Scope

- `gcn`: baseline graph convolution with normalized aggregation
- `sage`: GraphSAGE with additive aggregation
- `gin`: sum-aggregation architecture with stronger structural discrimination
- `gps`: hybrid graph transformer style model available in the codebase
- `loopy`: cycle-sensitive architecture using additional r-neighborhood structure

Not every model family is necessarily used the same way on every page, but they all live in the
same training and loading framework.

## Backend API Summary

Degree routes:

- `POST /api/degree/generate`
- `POST /api/degree/analyze`
- `GET /api/degree/models`

Minimum-cycle routes:

- `POST /api/min_cycle/generate`
- `POST /api/min_cycle/analyze`
- `GET /api/min_cycle/models`

Cage routes:

- `POST /api/cage/generate`
- `GET /api/cage/status/<id>`
- `POST /api/cage/stop/<id>`

Config route:

- `GET /api/config`

## What Is Stable and What Is Exploratory

Stable:

- Interactive degree and minimum-cycle prediction
- Model loading from `ai/trained`
- Docs pages that summarize the project and expose live metrics

Exploratory:

- Cage generation heuristics
- Reinforcement-learning based graph construction
- Any claim that learned guidance already solves cage construction

## Development Notes

- Prefer small, targeted changes over broad refactors
- Do not run long training jobs as validation
- For touched Python files, run `basedpyright` before reporting completion
- The repo may contain user work in progress; do not overwrite unrelated changes

## Related Files

- `backend/app.py`
- `backend/routes/`
- `backend/utils/graph_utils.py`
- `frontend/src/pages/OverviewPage.tsx`
- `frontend/src/pages/docs/`
- `ai/registry.py`

## Current Position of the Project

This repository already works well as a compact experimentation platform for comparing GNN
architectures on graph-property prediction tasks. The most open part of the work is still cage
generation, which is exactly why the docs frame it as ongoing research rather than a completed
result.

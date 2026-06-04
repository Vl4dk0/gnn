# Machine Learning for Generation of Graph of Given Degree and Girth

A research platform investigating whether graph neural networks and reinforcement
learning can help in algebraic graph theory, with the **cage problem** as the
long-term objective: constructing small `k`-regular graphs of a given girth `g`.

The repository combines an interactive web application, a library of GNN models and
training pipelines, several graph-construction methods (search, voltage lifts,
refinement, excision, and the composed *Forge* pipeline), a benchmarking framework,
and HPC submission scripts.

## About

This is the codebase accompanying a bachelor thesis:

| | |
|---|---|
| **Title** | Machine Learning for Generation of Graph of Given Degree and Girth |
| **Author** | Vladimír Jančár |
| **Supervisor** | Mgr. Ján Pastorek |
| **University** | Comenius University in Bratislava |
| **Faculty** | Faculty of Mathematics, Physics and Informatics (FMFI UK) |
| **Department** | Department of Applied Informatics |
| **Study programme** | Applied Computer Science |
| **Year** | 2026 |

The compiled thesis lives under [`thesis/`](thesis/) as `main.pdf`.

## What's in here

| Directory | Contents | Docs |
|---|---|---|
| [`frontend/`](frontend/) | React / TypeScript web UI (Vite): graph editing, degree / min-cycle prediction, cage generation, docs pages | |
| [`backend/`](backend/) | Flask app factory, static serving, and API routes | |
| [`ai/`](ai/) | GNN model implementations, the model registry, and training entrypoints | [ai/README.md](ai/README.md) |
| [`results/`](results/) | Benchmarking framework for comparing generators and prediction models | [results/README.md](results/README.md) |
| [`supercomputer/`](supercomputer/) | SLURM batch scripts for the PERUN HPC cluster | [supercomputer/README.md](supercomputer/README.md) |
| [`tests/`](tests/) | Pytest suite (graph invariants, search, construction, routes) | [tests/README.md](tests/README.md) |
| [`thesis/`](thesis/) | The compiled thesis (`main.pdf`) | |

## Getting started

Requirements:

- [Python](https://www.python.org/) 3.14+
- [`uv`](https://docs.astral.sh/uv/)
- [Node.js](https://nodejs.org/) and [npm](https://www.npmjs.com/)

Install dependencies:

```bash
uv sync
```

Run the backend:

```bash
uv run python run.py
```

Server starts on port 5555, make sure it is available, otherwise it might not work.

```bash
cd frontend
npm install
npm run dev
```

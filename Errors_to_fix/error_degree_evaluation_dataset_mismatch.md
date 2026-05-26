# Evaluation Dataset Mismatch in Degree Prediction
## Source/Occurrence
- Codebase: `ai/degree/train.py` (`evaluate_model` function)
- Thesis: `thesis/chapters/04-preparatory-tasks.typ` (Section: Evaluation)

## Explanation
The thesis states that the models are evaluated on a fixed battery of 75 graphs with 1790 vertices, combining random Erdős–Rényi graphs with structured examples (complete graphs, bipartite graphs, cycles, grids, Petersen, Heawood). However, the `evaluate_model` function in `ai/degree/train.py` dynamically generates random Erdős–Rényi graphs for evaluation (`num_test_graphs=100` during validation, `200` for final evaluation) using `generate_random_graph_data`. It does not use the described fixed dataset of structured and random graphs.

## Actionable steps
1. Modify `evaluate_model` in `ai/degree/train.py` to use a fixed evaluation dataset of 75 graphs containing the specific structured examples mentioned in the thesis.
2. Provide a helper function or loader that returns this fixed battery of graphs instead of calling `generate_random_graph_data` for evaluation.

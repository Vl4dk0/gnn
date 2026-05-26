# Evaluation Dataset Mismatch

## Source of the issue
- `thesis/chapters/04-preparatory-tasks.typ` (Lines 57-61)
- `ai/min_cycle/train.py` (Lines 130-173, `evaluate_model` function)
- `ai/degree/train.py` (Lines 135-180, `evaluate_model` function)

## Definition of the issue
The thesis claims that the GNN models are evaluated on a fixed validation battery of 75 graphs with a total of 1790 vertices. This dataset is described as combining random Erdős–Rényi graphs with structured examples, such as complete graphs, complete bipartite graphs, cycles, grids, hypercubes, and known regular graphs like the Petersen and Heawood graphs.

However, the `evaluate_model` function in both the `ai/min_cycle/train.py` and `ai/degree/train.py` scripts does not use this fixed battery of graphs. Instead, it generates dynamic random Erdős–Rényi graphs for evaluation on the fly (`num_test_graphs=100` during training validation and `200` for final evaluation) using the `generate_random_graph_data` function.

This discrepancy means the codebase's evaluation pipeline diverges entirely from the thesis narrative. To resolve this, a centralized test dataset containing the exact 75 graphs described in the thesis must be implemented and saved. The `evaluate_model` functions must then be updated to load and evaluate the models exclusively on this standardized test set instead of dynamically generating random graphs. This ensures that the reported metrics correspond to the structured benchmarks described in the text.

# Evaluation Dataset Mismatch

**Source/Occurrence:** 
- `thesis/chapters/04-preparatory-tasks.typ` (Lines 57-61)
- `ai/min_cycle/train.py` (Lines 130-173, `evaluate_model` function)

**Explanation:**
The thesis claims that the models are evaluated on a fixed battery of 75 graphs with 1790 vertices, combining random Erdős–Rényi graphs with structured examples (complete graphs, bipartite graphs, cycles, grids, Petersen, Heawood). However, the `evaluate_model` function in `ai/min_cycle/train.py` generates dynamic random Erdős–Rényi graphs for evaluation (`num_test_graphs=100` during validation, `200` for final evaluation) using `generate_random_graph_data`. It does not use the described fixed dataset of 75 structured and random graphs.

**Actionable Steps:**
1. Create a centralized test dataset containing the exact 75 graphs described in the thesis (including Petersen, Heawood, grids, complete graphs, etc.) and save it.
2. Update the `evaluate_model` function in `ai/min_cycle/train.py` to load and evaluate on this fixed dataset instead of dynamically generating random graphs.
3. Ensure that the total vertex count in the test dataset matches the claimed 1790 vertices, and update the reported metrics using this standard test set.

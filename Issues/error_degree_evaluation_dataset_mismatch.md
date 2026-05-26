# Evaluation Dataset Mismatch in Degree Prediction
## Source of the issue
- `ai/degree/train.py` (Lines 135-180, `evaluate_model` function)
- `thesis/chapters/04-preparatory-tasks.typ` (Lines 57-61, Evaluation Section)

## Definition of the issue
The thesis clearly states that models are evaluated on a fixed validation battery of 75 graphs containing exactly 1790 vertices. This predefined dataset is described as a specific mixture of random Erdős–Rényi graphs and structured examples (including complete graphs, bipartite graphs, cycles, grids, Petersen, and Heawood graphs) so that every model is tested against the exact same topology challenges.

However, the `evaluate_model` function in `ai/degree/train.py` ignores this requirement. Instead of loading the described fixed evaluation dataset, it dynamically synthesizes random Erdős–Rényi graphs on the fly via `generate_random_graph_data` (`num_test_graphs=100` during validation checks, and `200` for final evaluation). This completely skips the structured graph evaluations (like Petersen or grids) and violates the comparative fixed-baseline requirement described in the thesis.

To fix this, a new helper function or data loader should be implemented to construct and return the fixed battery of 75 specific graphs described in the thesis. The `evaluate_model` function must be updated to iterate over this precise list of static graphs instead of continuously dynamically sampling standard random graphs.

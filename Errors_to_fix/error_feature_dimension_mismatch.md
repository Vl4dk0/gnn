# Feature Dimension Mismatch in Evaluation API

**Source/Occurrence:**
- Thesis text: `c:\Users\bskon\gnn\thesis\chapters\04-preparatory-tasks.typ` (Section: Data Generation)
- `c:\Users\bskon\gnn\ai\degree\functions\graph_service.py` (Lines 130-146)
- `c:\Users\bskon\gnn\ai\min_cycle\functions\graph_service.py` (Lines 158-174)

**Explanation:**
The thesis states: "The experiments use either a constant one-dimensional feature or a four-dimensional feature vector". This implies models are trained and evaluated with both 1D and 4D input features.
However, in both the `degree` and `min_cycle` API endpoints (`graph_service.py`), the input feature tensor `x` is hardcoded to always concatenate into a 4-dimensional tensor:
`x = torch.cat([node_idx_feature, random_feature, clustering_feature], dim=1)`

Because `model.input_dim` is completely ignored during inference in the graph service, any model trained with 1-dimensional features (`input_dim=1`) will crash with a shape mismatch error (e.g., `mat1 and mat2 shapes cannot be multiplied`) when evaluated through `validation.py` or the API.

**Actionable Steps:**
1. In both `graph_service.py` scripts, check the loaded model's `input_dim` (e.g., `getattr(model, "input_dim", 4)`).
2. If `model.input_dim == 1`, construct `x` as a constant one-dimensional tensor (`torch.ones(num_nodes, 1)`), exactly matching the logic in `generate_random_graph_data` inside `train.py`.
3. Only concatenate the 4-dimensional features if `model.input_dim == 4`.

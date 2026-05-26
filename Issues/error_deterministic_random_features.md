# Fixed Random Seed Destroys Permutation Equivariance in Evaluation

## Source of the issue
- `ai/degree/functions/graph_service.py` (Lines 135-141)
- `ai/min_cycle/functions/graph_service.py` (Lines 163-169)
- `ai/degree/train.py` (Lines 95-99)

## Definition of the issue
During training in `train.py`, the dynamic node features (two random real-valued coordinates and one random scalar) are generated dynamically for each graph batch without resetting the RNG seed. This accurately provides random symmetry-breaking features to the Graph Neural Network (GNN), giving it useful noise to differentiate identical structural roles.

However, in the evaluation and inference path (`graph_service.py`), `torch.manual_seed(42)` is explicitly called immediately before generating these features for *each* graph. Since the seed is fixed per function call, the supposedly "random" features generated for any node `idx` depend entirely on the graph's total `num_nodes` and its position `idx` in the sorted node list. This causes nodes in the same sorted position across different graphs of the same size to receive identical features.

By making the features deterministic, the symmetry-breaking noise becomes a rigid positional encoding tied to the node's sorted index. This systematically destroys the GNN's permutation equivariance during validation. A node receives a specific feature vector based on arbitrary sorting rather than its structural role, which severely biases predictions and invalidates the generalizability of the model. The fix is to remove the `torch.manual_seed(42)` calls in both `graph_service.py` files to ensure features are generated genuinely at random, matching the training procedure.

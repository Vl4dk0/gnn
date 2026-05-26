# Fixed Random Seed Destroys Permutation Equivariance in Evaluation

**Source/Occurrence:**
- `c:\Users\bskon\gnn\ai\degree\functions\graph_service.py` (Lines 135-141)
- `c:\Users\bskon\gnn\ai\min_cycle\functions\graph_service.py` (Lines 163-169)
- `c:\Users\bskon\gnn\ai\degree\train.py` (Lines 95-99)

**Explanation:**
During training (`train.py`), node features include two random real-valued coordinates and one random scalar, generated dynamically for each graph batch without resetting the RNG seed. This correctly supplies random symmetry-breaking features to the GNN.

However, in the evaluation/inference path (`graph_service.py`), `torch.manual_seed(42)` is explicitly called immediately before generating these features for *each graph*.
Because the seed is fixed per function call, the "random" features generated for node `idx` depend only on the total `num_nodes` and its position `idx` in the sorted node list. They are identical across all graphs of the same size. 
This turns what should be random symmetry-breaking features into a rigid, deterministic positional encoding tied to the node's sorted index. This systematically destroys the GNN's permutation equivariance during validation. A node will receive the exact same feature vector regardless of its structural role, simply because it happens to be the $i$-th node when sorted.

**Actionable Steps:**
1. Remove the `torch.manual_seed(42)` calls in both `graph_service.py` files.
2. Let the features be generated purely randomly, exactly as they are generated during training in `train.py`.

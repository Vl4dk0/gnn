# Data Generation Uses Uniform Random Features for "Clustering Coefficient" — Not an Actual Clustering Coefficient

**Source/Occurrence:**
- `ai/degree/train.py` (Lines 98-99)
- `ai/min_cycle/train.py` (Lines 93-94)
- `thesis/chapters/04-preparatory-tasks.typ` (Lines 33-38)

**Explanation:**
The thesis (lines 33-38) states: "The input features are intentionally simple. The experiments use either a constant one-dimensional feature or a four-dimensional feature vector consisting of a normalized node index, two random real-valued coordinates, and one additional random scalar."

The code generates the fourth feature as:
```python
# Feature 3: Clustering coefficient estimate
clustering_feature = torch.rand(num_nodes, 1)
```

The comment says "Clustering coefficient estimate", but the value is just `torch.rand` — a uniformly random scalar in [0, 1] with no connection to the actual clustering coefficient of each node. The thesis correctly describes this as "one additional random scalar."

This is a **documentation/comment inconsistency**: the code comment misleads future developers into thinking the feature encodes structural graph information (clustering coefficients), when in fact it is a pure random noise feature. If a developer reads this code and assumes the model was trained with actual clustering coefficients, they might erroneously add real clustering coefficient computation during inference — breaking the training/inference feature alignment.

**Actionable Steps:**
1. In both `ai/degree/train.py` (line 98) and `ai/min_cycle/train.py` (line 93), rename the variable and fix the comment:
   ```python
   # Feature 3: Additional random scalar (noise for symmetry breaking)
   random_scalar_feature = torch.rand(num_nodes, 1)
   ```
2. Update the variable name in the `torch.cat` call on the following line accordingly.
3. In `ai/degree/functions/graph_service.py` (line 139) and `ai/min_cycle/functions/graph_service.py` (line 167-168), update the equivalent comment from "Clustering coefficient placeholder" to "Random scalar (matches training: random noise feature)".

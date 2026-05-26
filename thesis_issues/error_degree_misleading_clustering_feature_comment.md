# Misleading Clustering Feature Comment

## Source of the issue
- `ai/degree/train.py` (Lines 98-99)
- `ai/min_cycle/train.py` (Lines 93-94)
- `thesis/chapters/04-preparatory-tasks.typ` (Lines 33-38)

## Definition of the issue
The thesis states that the input features include "one additional random scalar" (used for symmetry breaking). However, the codebase generates this fourth feature as `torch.rand(num_nodes, 1)` and labels it with the comment `# Feature 3: Clustering coefficient estimate`, naming the variable `clustering_feature`.

This is a significant documentation and naming inconsistency. The value is a purely uniformly random scalar in `[0, 1]` with no connection to the actual clustering coefficient of the nodes. By mislabeling it as a clustering coefficient, future developers might be misled into thinking the feature encodes structural graph information. This could lead them to erroneously add actual clustering coefficient computation during inference, which would silently break the feature alignment between training and inference, degrading model performance. The variable name and its associated comments should be updated in both the training scripts and the graph service modules to explicitly reflect that it is an additional random scalar used as a noise feature.

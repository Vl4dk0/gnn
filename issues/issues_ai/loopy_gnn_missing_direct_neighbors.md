# Issue: Loopy GNN Preprocessing Fails to Populate Direct Neighbors (loopyN0 is always empty)

## Severity
High / Critical (Breaks the core message passing foundation of the Loopy GNN model)

## Description
In `ai/utils/r_neighborhood.py`, the `compute_r_neighborhood` function is responsible for building the $r$-neighborhood structure for the `Loopy_GNN` model.
For each path length $L$ from $0$ to $r$, it retrieves cycles of length $L + 2$ from `nx.simple_cycles` using:
```python
    for L in range(r + 1):
        path_length = L + 2  # Cycles of this length
        ...
        for cycle in cycles:
            if len(cycle) == path_length:
                ...
```
However, in a simple undirected graph, the shortest possible simple cycle length is 3 (a triangle). Simple cycles of length 2 (representing standard direct neighbor edges `u - v - u`) do not exist. As a result, for $L = 0$ (which is meant to represent standard direct 1-hop neighbor relationships), no cycles are found, and `loopyN0` is always initialized as an empty tensor of shape `(2, 0)`.

Inside `LoopyLayer.forward` in `ai/models/loopy.py`, direct neighbors are only aggregated when $L = 0$:
```python
            if L == 0:
                # Direct neighbors (path length 2, just one intermediate node)
                contribution = path_embeddings.squeeze(0)  # (num_paths, hidden_dim)
```
But because `loopyN0` is always empty, the path loop skips this block due to the `if paths.shape[1] == 0: continue` check. This means standard 1-hop neighbor aggregation is completely omitted. The `Loopy_GNN` model only aggregates cycles of length $\ge 3$ and never learns from direct neighbors, which breaks standard GNN message passing entirely.

## Location
`ai/utils/r_neighborhood.py` (lines 44-73):
```python
    for L in range(r + 1):
        path_length = L + 2  # Cycles of this length

        # Filter cycles of this length and generate all rotations
        L_paths: list[list[int]] = []
        L_atomic: list[list[int]] = []

        for cycle in cycles:
            if len(cycle) == path_length:
                # Generate all cyclic rotations (one for each starting node)
                for start_idx in range(len(cycle)):
                    # Rotate so cycle starts at different node each time
                    rotated = cycle[start_idx:] + cycle[:start_idx]
                    center = rotated[0]

                    # Compute distances from center for each node in path
                    atomic = [distances[center][node] for node in rotated]

                    L_paths.append(rotated)
                    L_atomic.append(atomic)
```

## Proposed Fix
For $L = 0$, instead of searching `cycles` (which won't contain cycles of length 2), populate `L_paths` and `L_atomic` directly from the edges of the graph. For each undirected edge `(u, v)` (excluding self-loops), add both directed paths `[u, v]` and `[v, u]` with distance annotations `[0, 1]`. For self-loops `(u, u)`, add `[u, u]` with `[0, 0]`.

Example implementation block for $L = 0$:
```python
    for L in range(r + 1):
        path_length = L + 2

        L_paths: list[list[int]] = []
        L_atomic: list[list[int]] = []

        if L == 0:
            for u, v in G.edges():
                if u != v:
                    L_paths.append([u, v])
                    L_atomic.append([0, 1])
                    L_paths.append([v, u])
                    L_atomic.append([0, 1])
                else:
                    L_paths.append([u, u])
                    L_atomic.append([0, 0])
        else:
            for cycle in cycles:
                if len(cycle) == path_length:
                    for start_idx in range(len(cycle)):
                        rotated = cycle[start_idx:] + cycle[:start_idx]
                        center = rotated[0]
                        atomic = [distances[center][node] for node in rotated]
                        L_paths.append(rotated)
                        L_atomic.append(atomic)
```


## Test

Proven by [`test_loopy_neighborhood_populates_direct_neighbors`](../../tests/test_issues_ai.py) (test fails → bug confirmed).

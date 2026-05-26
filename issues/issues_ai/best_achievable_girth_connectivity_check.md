# Issue: Missing Connectivity Guard on Random Search Path in best_achievable_girth

## Severity
Medium

## Description
In `ai/cage/cayley/group_data_gen.py`, the function `best_achievable_girth` runs two inner searches: `random_search` and `tabu_search`. 

While the result of `tabu_search` is explicitly verified to produce a connected, degree-k Cayley graph (`nx.is_connected(cay) and cay.degree(0) == k`) before updating the best girth, the result of the `random_search` path is accepted and recorded directly without any connectivity or degree check:
```python
    _gens_r, girth_r = random_search(
        group, k, g_target, num_trials=num_random_trials, verbose=False
    )
    if isinstance(girth_r, int) and girth_r > best:
        best = girth_r
```
If a latent bug exists in `random_search` or if it returns an invalid or disconnected graph (for example, if the generated set is not a generating set for the entire group), this will silently corrupt the labels used for training.

## Location
`ai/cage/cayley/group_data_gen.py` (lines 114-118):
```python
    _gens_r, girth_r = random_search(
        group, k, g_target, num_trials=num_random_trials, verbose=False
    )
    if isinstance(girth_r, int) and girth_r > best:
        best = girth_r
```

## Proposed Fix
Add a verification helper or check connectivity and degree explicitly after the `random_search` path as well:
```python
    if isinstance(girth_r, int) and girth_r > best:
        if _gens_r is not None:
            cay = build_cayley(group, _gens_r)
            if nx.is_connected(cay) and cay.degree(0) == k:
                best = girth_r
```

# Issue: Floating-point precision boundary miss when prune_slack = 0.0

## Severity
Medium

## Description
In `ai/cage/cayley/group_search.py`, the pruning logic filters groups using a raw `>=` comparison without any floating-point tolerance:
```python
        if predict_fn(group, k, g_target) >= g_target - prune_slack:
            kept.append(group)
```
If `prune_slack = 0.0` is specified via the CLI (a valid option) and the model predicts a value extremely close to but slightly below the target girth (e.g. `g_target - epsilon` due to floating point precision limits of neural network outputs), the group will be pruned even if it actually achieves the target girth.

## Location
`ai/cage/cayley/group_search.py` (lines 90-93):
```python
    for group in groups:
        if predict_fn(group, k, g_target) >= g_target - prune_slack:
            kept.append(group)
```

## Proposed Fix
Introduce a small epsilon floor to prevent strict floating-point pruning misses:
```python
        if predict_fn(group, k, g_target) >= g_target - prune_slack - 1e-6:
```

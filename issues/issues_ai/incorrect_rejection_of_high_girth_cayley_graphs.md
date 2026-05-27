# Issue: Incorrect Rejection of High-Girth Cayley Graphs in Search Algorithms

## Severity
High / Critical (Prevents search algorithms from discovering high-girth Cayley graphs and record cages)

## Description
In `ai/cage/cayley/search.py`, `ai/cage/functions/cayley_search.py`, and `ai/cage/cayley/cayley.py`, `cayley_girth` is used to compute the girth of a candidate Cayley graph using a search bound of `max_girth = 2 * g_target`.

If the candidate Cayley graph has no cycles of length up to `2 * g_target`, `cayley_girth` returns `math.inf` (which is of type `float`).
However, several search functions (such as `random_search`, `tabu_search`, `verify_cayley`, and `CayleySearchGenerator.step`) filter out results where the girth is a `float` or where `isinstance(girth, int)` is `False`.

Specifically:
1. In `random_search` (lines 69-70):
```python
        if isinstance(girth, float) or girth <= best_girth:
            continue
```
This skips the candidate immediately, meaning any symmetric generating set that yields a girth larger than `2 * g_target` is discarded and never checked/verified.

2. In `tabu_search` (line 183), `verify_cayley` (lines 194-195), and `CayleySearchGenerator.step` (lines 119, 162):
```python
            if (
                isinstance(girth, int)
                and girth >= g_target
                ...
```
This check evaluates to `False` if `girth` is `math.inf` (since it is a float), rejecting the Cayley graph as degenerate or invalid.

While some infinite girth returns truly correspond to degenerate graphs (e.g. where the generating set doesn't generate the group or the graph is disconnected/wrong degree), many correspond to extremely high-girth Cayley graphs that exceed the `2 * g_target` check-depth. By rejecting all floats, the search completely fails to find and record any solution with girth greater than `2 * g_target`.

## Location
- `ai/cage/cayley/search.py` (lines 69, 183, 299, 312, 403)
- `ai/cage/cayley/cayley.py` (lines 194-195)
- `ai/cage/functions/cayley_search.py` (lines 119, 162)

## Proposed Fix
Instead of checking `isinstance(girth, int)` or skipping `isinstance(girth, float)`, explicitly allow `float` (specifically `math.inf`) if it means the girth is at least `g_target` or larger than `2 * g_target`.
To ensure the graph is not degenerate (disconnected/wrong degree), build and verify it using standard networkx verification:

For `random_search`:
```python
        if not (isinstance(girth, float) or girth > best_girth):
            continue
```

For `tabu_search` and generators, when `cost == 0` is reached, verify the graph properly:
```python
        cay = build_cayley(group, gens)
        if nx.is_connected(cay) and cay.degree(0) == k:
            # Graph is valid, retrieve exact finite girth if needed or treat as >= g_target
            ...
```


## Test

Proven by [`test_cayley_random_search_does_not_discard_inf_girth`](../../tests/test_issues_ai.py) (test fails → bug confirmed).

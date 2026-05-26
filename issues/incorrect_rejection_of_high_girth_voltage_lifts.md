# Issue: Incorrect Rejection of High-Girth Voltage Lifts in Search Algorithms

## Severity
High / Critical (Prevents search algorithms from discovering high-girth graphs and records)

## Description
In `ai/cage/voltage/search.py` and `ai/cage/functions/voltage_search.py`, `compute_lift_girth` is used to quickly compute the girth of a candidate lift without fully building the NetworkX graph. It uses `max_girth = 2 * g_target` as a search bound.

If the candidate lift has no cycles of length up to `2 * g_target`, `compute_lift_girth` returns `math.inf` (which is of type `float`).
However, several search functions (such as `exhaustive_search`, `random_search`, `tabu_search`, and `meta_search`) filter out results where the girth is a `float` or where `isinstance(girth, int)` is `False`.

Specifically:
1. In `exhaustive_search` (lines 77-78) and `random_search` (lines 120-121):
```python
        if isinstance(girth, float) or girth <= best_girth:
            continue
```
This skips the candidate immediately, meaning any assignment that yields a girth larger than `2 * g_target` is discarded and never checked/verified.

2. In `tabu_search` (lines 198 and 253), `meta_search` (lines 438, 448, 542), and `VoltageSearchGenerator.step` (lines 124, 182, 198):
```python
        if isinstance(girth, int) and girth >= g_target:
```
This check evaluates to `False` if `girth` is `math.inf` (since it is a float), rejecting the lift as degenerate or invalid.

While some infinite girth returns truly correspond to degenerate lifts (e.g. where all edges collapse to self-loops and are not added, resulting in a 0-regular graph with no cycles), many correspond to extremely high-girth regular lifts that exceed the `2 * g_target` check-depth. By rejecting all floats, the search completely fails to find and record any solution with girth greater than `2 * g_target`.

## Location
- `ai/cage/voltage/search.py` (lines 77, 120, 198, 253, 438, 448, 542)
- `ai/cage/functions/voltage_search.py` (lines 124, 182, 198)

## Proposed Fix
Instead of checking `isinstance(girth, int)` or skipping `isinstance(girth, float)`, explicitly allow `float` (specifically `math.inf`) if it means the girth is at least `g_target` or larger than `2 * g_target`.
To ensure the lift is not degenerate (disconnected/empty/wrong degree), build and verify it using `verify_lift`:

For `random_search`/`exhaustive_search`:
```python
        # Check if girth is infinite (meaning > 2 * g_target) or finite and better
        if not (isinstance(girth, float) or girth > best_girth):
            continue
```

For `tabu_search` and generators, when `cost == 0` is reached, verify the lift properly via `verify_lift`:
```python
        props = verify_lift(build_lift(base, group, volts), k, g_target)
        if props["is_valid_kg"]:
            # Retrieve the exact finite girth from verify_lift
            best_girth = props["girth"]
            ...
```

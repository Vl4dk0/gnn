# Issue: Extremely Slow Matrix Group Generation (PGL2/SL2) and Table Building

## Severity
High / Critical (Causes test suite to run extremely slowly or appear hung)

## Description
In `ai/cage/voltage/groups.py`, the functions `pgl2(p)` and `sl2(p)` generate groups of order $O(p^3)$. For larger values of $p$ (like $p = 17$, which has order 4896), constructing the group becomes extremely slow. This is caused by two main bottlenecks:
1. In `pgl2(p)`, the elements are normalized and added to `reps`. The check `if rep not in reps: reps.append(rep)` performs a linear search over a list of size up to 4896 for each of the $17^4 = 83521$ iterations. This requires roughly $83521 \times 2448 \approx 200,000,000$ operations in pure Python.
2. In both `pgl2` and `sl2`, the multiplication table is built using `_build_tables(order, _mat_mult)`. This function has nested loops of size `order * order` in Python:
```python
    mt = np.zeros((order, order), dtype=np.intp)
    for a in range(order):
        for b in range(order):
            mt[a, b] = mult_fn(a, b)
```
For $p = 17$ (order 4896), this requires $4896 \times 4896 \approx 24,000,000$ Python function calls and lookups, which takes about 30–60 seconds to execute. Because the test `test_available_groups_abelian_cap_bounds_cyclic_orders` calls `available_groups(5000)`, it attempts to construct `pgl2(17)` and `sl2(17)`, making the test suite run extremely slowly and look like it's hung.

## Location
`ai/cage/voltage/groups.py` (lines 45-60, 181-220, and 255-289)

## Proposed Fix
1. In `pgl2`, use a set for deduplication or generate unique representantes directly (e.g. by iterating only over valid projective matrices) instead of doing a linear search in a list.
2. Optimize table building, or lower the maximum prime values tested in the test suite to avoid creating extremely large groups during standard test runs. For example, in `available_groups`, restrict matrix groups to a separate cap or lower the default bounds when running quick tests.

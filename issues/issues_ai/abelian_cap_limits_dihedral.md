# Issue: Dihedral Group Catalogue Cap Limitation

## Severity
High

## Description
In `ai/cage/cayley/groups.py`, the `available_groups` function uses the `abelian_cap` parameter to also limit the dihedral group generation cap (`dih_cap`). 

Specifically, at line 116:
```python
dih_cap = min(max_order, abelian_cap)
```
This means that dihedral groups are capped by the same limit as abelian groups, even when a user explicitly requests a much larger maximum group order (e.g. `max_group_order=2200`). As a result, dihedral groups larger than `abelian_cap` (which defaults to 300) are silently excluded from the catalogue, resulting in incomplete searches.

## Location
`ai/cage/cayley/groups.py` (lines 115-120):
```python
    if include_dihedral:
        dih_cap = min(max_order, abelian_cap)
        for n in range(3, dih_cap // 2 + 1):
            if 2 * n <= dih_cap:
                _add(dihedral_group(n))
```

## Proposed Fix
Replace `abelian_cap` with an independent cap for dihedral groups (e.g., `dihedral_cap`), or use `max_order` directly with a separate safety cap.


## Test

Proven by [`test_abelian_cap_limits_dihedral_real_issue`](../../tests/test_issues_ai.py) (test fails → bug confirmed).

# Issue: Unguarded cache lookup in `_cached` helper function

## Severity
Low

## Description
In `ai/cage/cayley/group_search.py`, the nested helper function `_cached` directly accesses the predictions cache dictionary using the group name:
```python
        def _cached(
            group: FiniteGroup,
            _k: int,
            _g: int,
            cache: dict[str, float] = predictions,
        ) -> float:
            return cache[group.name]
```
If a future code change or different execution path causes the list of groups passed to `_cached` to diverge from the keys in `predictions` (e.g. non-deterministic sorting or argument mismatches), a bare `KeyError` will be thrown, causing the program to crash.

## Location
`ai/cage/cayley/group_search.py` (lines 250-256):
```python
        def _cached(
            group: FiniteGroup,
            _k: int,
            _g: int,
            cache: dict[str, float] = predictions,
        ) -> float:
            return cache[group.name]
```

## Proposed Fix
Use the dictionary `.get()` method with a fallback value:
```python
        def _cached(
            group: FiniteGroup,
            _k: int,
            _g: int,
            cache: dict[str, float] = predictions,
        ) -> float:
            return cache.get(group.name, 0.0)
```
or log a warning and return a conservative fallback.

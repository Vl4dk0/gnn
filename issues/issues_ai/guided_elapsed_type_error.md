# Issue: Possible TypeError in verbose output due to missing None-guard on elapsed time

## Severity
Medium

## Description
In `ai/cage/cayley/group_search.py`, within the verbose printing code block inside `run_comparison`, the elapsed time is formatted directly using `:.1f`:
```python
                    + f"time={guided.get('elapsed'):.1f}s "
```
However, if `guided.get('elapsed')` is `None` (for instance, if `group_guided_meta_search` is refactored or fails to populate the `"elapsed"` key), formatting it with `:.1f` will raise a `TypeError: unsupported format string passed to NoneType.__format__`. This is fragile compared to the safe check done a few lines below in `format_comparison_table`.

## Location
`ai/cage/cayley/group_search.py` (lines 292-293):
```python
                    + f"time={guided.get('elapsed'):.1f}s "
```

## Proposed Fix
Use a fallback value or an explicit type check before formatting:
```python
                    + f"time={guided.get('elapsed', 0.0):.1f}s "
```
or:
```python
                    elapsed = guided.get('elapsed')
                    elapsed_str = f"{elapsed:.1f}s" if isinstance(elapsed, float) else "-"
                    ...
                    + f"time={elapsed_str} "
```


## Test

Proven by [`test_guided_elapsed_format_uses_safe_pattern`](../../tests/test_issues_ai.py) (test fails → bug confirmed).

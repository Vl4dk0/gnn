# Infinite Girth Treated as Valid in verify_lift

**Source/Occurrence:**
- `ai/cage/voltage/lift.py` (Lines 96-98, `verify_lift` function)
- `thesis/chapters/06-voltage-lifts.typ` (Lines 31-32)

**Explanation:**
In `verify_lift`, when `compute_girth` returns `math.inf` (acyclic graph / tree), the code sets `girth_ok = True` on the grounds that "infinite girth satisfies any girth requirement":

```python
if isinstance(girth, float):
    girth_ok = True  # infinite girth (tree) satisfies any girth requirement
```

However, a $(k,g)$-graph must be a **cycle-containing** graph — specifically a $k$-regular graph whose shortest cycle has length exactly $\geq g$. An acyclic $k$-regular graph (i.e. an infinite $k$-regular tree) does not exist as a finite graph. In practice, a finite $k$-regular graph with $k \geq 2$ is always non-acyclic, so `girth = inf` in `compute_girth` on a finite graph actually signals an **error or disconnected state**, not a valid cage.

The thesis (Chapter 6, lines 31-32) confirms: "The verification step checks connectedness, $k$-regularity, and girth." The current code passes `is_valid_kg = True` for any assignment yielding an infinite-girth lift, which is logically incorrect and could silently accept degenerate graphs.

Furthermore, the `is_connected` check is already in `verify_lift`, but `is_valid_kg` is computed as `k_reg and girth_ok and connected`. A disconnected lift would be caught by `connected`, but an extremely sparse connected lift (e.g. a path graph) with no cycles would pass all three checks erroneously.

**Actionable Steps:**
1. In `ai/cage/voltage/lift.py`, change the `girth_ok` logic to reject infinite-girth results:
   ```python
   if isinstance(girth, float):
       girth_ok = False  # no finite cycles means not a valid (k,g)-graph
   else:
       girth_ok = girth >= g
   ```
2. Verify that this does not break existing tests — a $k$-regular connected finite graph with $k \geq 2$ should never have infinite girth, so this change should be a no-op in practice while making the logic correct.

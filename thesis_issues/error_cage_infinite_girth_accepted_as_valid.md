# Infinite Girth Treated as Valid in verify_lift
## Source of the issue
- `ai/cage/voltage/lift.py` (Lines 96-98, `verify_lift` function)
- `thesis/chapters/06-voltage-lifts.typ` (Lines 31-32)

## Definition of the issue
In the `verify_lift` function, if `compute_girth` returns `math.inf` (meaning the graph is acyclic or a tree), the code sets `girth_ok = True` under the logic that "infinite girth satisfies any girth requirement." 

However, a $(k,g)$-graph must be a finite, regular, cycle-containing graph, defined as a $k$-regular graph whose shortest cycle has a length of exactly $\ge g$. In practice, any finite $k$-regular graph with $k \ge 2$ must contain cycles. Therefore, receiving a `girth = inf` result for a finite graph indicates either an error, an extremely sparse degenerate acyclic graph (like a finite path), or a disconnected state, not a valid valid cage graph. 

The thesis specifies that the verification step rigorously checks connectedness, $k$-regularity, and girth. Although `is_connected` is checked, an erroneously constructed sparse connected graph with no cycles could pass the checks since `is_valid_kg` is currently the boolean AND of all three properties. A disconnected lift would be caught by `is_connected`, but treating `girth_ok = True` on an acyclic graph fundamentally breaks the requirement.

To fix this, `verify_lift` must explicitly reject infinite-girth results by setting `girth_ok = False` if the girth is returned as a float (`math.inf`). This ensures no acyclic degenerate graphs are mistakenly accepted as valid $(k,g)$-graphs.

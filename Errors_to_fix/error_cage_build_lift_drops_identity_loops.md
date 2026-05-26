# Thesis Claims Right Multiplication for Voltage Lift, Code Uses Right Multiplication Correctly but Build_Lift Skips Self-Loop Case

**Source/Occurrence:**
- `ai/cage/voltage/lift.py` (Lines 55-67, `build_lift` function)
- `thesis/chapters/06-voltage-lifts.typ` (Lines 22-29)

**Explanation:**
The thesis (lines 22-29) defines the lift using right multiplication: for an arc from $u$ to $v$ with voltage $a$, the lift connects $(u, h)$ to $(v, h \cdot a)$ for every $h \in \Gamma$. The code correctly implements this as `h = group.mult(g, alpha)`.

However, in `build_lift` (line 65), self-loops are explicitly silently dropped:

```python
if node_ug != node_wh:  # avoid self-loops from degenerate cases
    G.add_edge(node_ug, node_wh)
```

For a loop at vertex $v$ in the base graph (added via `add_loop`), the arc goes from $v$ to $v$ with some voltage $\alpha$. The lifted arc connects $(v, g)$ to $(v, g \cdot \alpha)$. If $\alpha$ is the identity, then $(v, g)$ connects to itself — a self-loop in the lift, which should be a 1-cycle. If $\alpha \neq$ identity, $(v, g)$ connects to $(v, g \cdot \alpha) \neq (v, g)$, producing a valid edge.

By silently dropping the case `node_ug == node_wh` (i.e., when $\alpha$ is the identity on a loop), `build_lift` produces an incorrect graph. The `bouquet` base graph uses loops, so when a loop's voltage is the identity, the expected self-loop in the lift (and thus a 1-cycle) is lost. This makes `verify_lift` unable to detect the resulting degenerate structure.

**Actionable Steps:**
1. In `ai/cage/voltage/lift.py`, remove or document the self-loop filtering. If self-loops must be excluded because NetworkX simple graphs cannot represent them, use `nx.MultiGraph` instead to correctly represent multiplicity and loops.
2. Alternatively, add a pre-check in `verify_lift`: if any arc from $v$ to $v$ has identity voltage, immediately flag `girth = 1` (a loop exists in the lift).
3. Update `cycle_analysis.py` accordingly so `count_short_identity_walks` and `compute_lift_girth` also account for loops (identity-voltage loop arcs → length-1 identity walk).

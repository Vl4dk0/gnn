# Build_Lift Skips Self-Loop Case for Identity Voltages on Loops
## Source of the issue
- `ai/cage/voltage/lift.py` (Lines 55-67, `build_lift` function)
- `thesis/chapters/06-voltage-lifts.typ` (Lines 22-29)

## Definition of the issue
The thesis defines the voltage lift operation using right multiplication: for an arc from $u$ to $v$ with voltage $a$, the lift connects $(u, h)$ to $(v, h \cdot a)$ for every group element $h \in \Gamma$. The implementation in `ai/cage/voltage/lift.py` correctly uses `h = group.mult(g, alpha)` to compute this. 

However, in `build_lift`, self-loops are explicitly and silently dropped by checking `if node_ug != node_wh` before adding an edge. When a base graph has a loop at vertex $v$ and the assigned voltage is the identity element, the lifted arc connects $(v, g)$ to $(v, g \cdot \text{identity}) = (v, g)$. This forms a valid self-loop in the lifted graph, corresponding to a 1-cycle. 

By silently filtering out the `node_ug == node_wh` case, `build_lift` produces an incorrect simple graph missing these self-loops. Consequently, the verification step `verify_lift` fails to detect the resulting degenerate structure (girth 1) because the expected 1-cycles have been artificially removed. To fix this, the filtering should either be removed by using a multigraph, or `verify_lift` should explicitly flag an assignment as invalid (girth 1) if an identity-voltage self-loop is detected in the base graph. `cycle_analysis.py` may also need adjustments to account for identity-voltage loop arcs correctly.

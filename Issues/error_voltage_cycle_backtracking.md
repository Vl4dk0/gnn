# Incorrect Backtracking Logic Masks 2-Cycles in Voltage Lifts

## Source of the issue
- `ai/cage/voltage/cycle_analysis.py`, lines 61-71 (function `_would_backtrack`), and lines 92, 174 (checks for `length >= 3`).

## Definition of the issue
The cycle analysis attempts to prevent backtracking along the same lift edge by checking if `next_arc_id` has the identical signature `(src, dst, voltage)` as the reverse of `prev_arc_id`. However, if the base graph contains multiple parallel edges, and two distinct parallel edges are assigned the exact same voltage, they will have the identical signature. `_would_backtrack` incorrectly treats traversing the second parallel edge as a backtrack and prunes it.

As a result, closed walks of length 2 with identity net voltage (which indicate parallel edges in the lift, forming a 2-cycle) are completely ignored. Furthermore, the functions explicitly check `length >= 3` before counting a cycle, meaning loops in the base graph with identity voltage (which form 1-cycles in the lift) are also ignored.

Because of this, `count_short_identity_walks` returns a cost of `0` for these degenerate assignments. `tabu_search` sees `cost == 0`, mistakenly believes it has found a perfect solution, and early-exits. The returned assignment is then rejected by `verify_lift` because the dropped parallel edges/loops cause the graph to fail the $k$-regularity check. This causes the search to silently fail on that configuration.

To resolve this, `_would_backtrack` should be changed to simply check if `next_arc_id` is exactly the reverse arc of `prev_arc_id` using its ID (`return next_arc_id == base.arcs[prev_arc_id].reverse_id`). This correctly allows traversing a different parallel edge even if it has the same voltage, thus exposing the 2-cycle. In `compute_lift_girth` and `count_short_identity_walks`, the `length >= 3` condition when counting walks should be removed, as identity walks of length 1 and 2 correspond to loops and parallel edges in the lift and must be penalized so that `tabu_search` does not prematurely exit with degenerate assignments.

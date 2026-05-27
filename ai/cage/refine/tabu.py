"""Tabu search refiner for post-lift k-regular graph optimisation.

TabuRefiner drives f(G) -> 0 where f is the short-cycle cost function.
At each iteration it evaluates 2-switch candidates, picks the best
non-tabu move (with aspiration), and applies it.

When the best 2-switch shows no improvement for K consecutive iterations,
the refiner escalates to a one-shot 3-switch pass to escape the local
minimum.

An optional *score_fn* can be provided to score candidate swaps with a GNN
instead of evaluating the full cost delta for each candidate.  The score_fn
signature is:

    score_fn(G, swaps) -> torch.Tensor of shape [len(swaps)]

where lower values mean "more promising" (predicts lower Δcost).  When
score_fn is provided the refiner uses it to rank candidates and applies only
the top-ranked one (then verifies the true delta).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import networkx as nx
import torch

from ai.cage.refine.cost import short_cycle_cost
from ai.cage.refine.swaps import (
    Swap2,
    Swap3,
    apply_2_switch,
    apply_3_switch,
    enumerate_2_switches,
    enumerate_3_switches,
)

# Signature type for the optional GNN scorer.
# Called with (G, list[Swap2]) -> Tensor[n_swaps] of predicted Δcost.
_ScoreFn = Callable[["nx.Graph[int]", list[Swap2]], torch.Tensor]


class TabuRefiner:
    """Tabu search refiner that minimises short-cycle cost on a k-regular graph.

    Parameters
    ----------
    g_target:
        Target girth.  Cost function penalises all cycles of length < g_target.
    score_fn:
        Optional GNN scorer.  When provided, candidates are ranked by the
        predicted Δcost and only the top candidate is evaluated exactly.
        When None, every candidate's exact Δcost is computed.
    tenure:
        Tabu tenure (number of iterations a move stays forbidden).
        Default: ceil(sqrt(|E|)).
    max_iter:
        Maximum number of tabu iterations.
    escalate_to_3switch:
        If True, escalate to a 3-switch pass when 2-switch has not improved
        the cost for *stagnation_limit* consecutive iterations.
    stagnation_limit:
        Number of non-improving 2-switch iterations before a 3-switch
        escalation.  Default: 3.
    sample_size:
        If given, only this many random candidate swaps are examined per
        iteration (makes each step O(sample_size) instead of O(|E|^2)).
    """

    def __init__(
        self,
        g_target: int,
        score_fn: _ScoreFn | None = None,
        tenure: int | None = None,
        max_iter: int = 500,
        escalate_to_3switch: bool = True,
        stagnation_limit: int = 3,
        sample_size: int | None = None,
    ) -> None:
        self.g_target = g_target
        self.score_fn = score_fn
        self._tenure = tenure
        self.max_iter = max_iter
        self.escalate_to_3switch = escalate_to_3switch
        self.stagnation_limit = stagnation_limit
        self.sample_size = sample_size

    def refine(self, G: nx.Graph[int]) -> tuple[nx.Graph[int], dict[str, Any]]:
        """Run tabu search on *G* and return the best graph found.

        Returns
        -------
        best_graph:
            The graph with the lowest short-cycle cost seen during search.
        stats:
            Dict with keys: initial_cost, final_cost, iterations,
            escalations, improved (bool).
        """
        n_edges = G.number_of_edges()
        tenure = (
            self._tenure
            if self._tenure is not None
            else max(1, math.ceil(math.sqrt(n_edges)))
        )

        current: nx.Graph[int] = G.copy()
        current_cost = short_cycle_cost(current, self.g_target)

        best: nx.Graph[int] = current.copy()
        best_cost = current_cost
        initial_cost = current_cost

        # Tabu list: maps (frozenset of removed edges, frozenset of added edges)
        # -> expiry iteration.  We key by the swap signature.
        tabu: dict[
            tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
            int,
        ] = {}

        stagnation = 0
        escalations = 0

        iteration = -1
        for iteration in range(self.max_iter):
            if current_cost == 0.0:
                break

            # --- 2-switch pass ---
            candidates = list(enumerate_2_switches(current, self.sample_size))

            if not candidates:
                break

            best_swap: Swap2 | None = None
            best_delta = math.inf
            best_candidate_cost = math.inf

            if self.score_fn is not None:
                # Use GNN to rank; apply only top candidate
                scores = self.score_fn(current, candidates)
                order = torch.argsort(scores)
                # Try the top-ranked candidates in order until we find a valid one
                for rank_idx in order[: min(5, len(candidates))].tolist():
                    swap = candidates[int(rank_idx)]
                    tabu_key = self._tabu_key(swap)
                    is_tabu = tabu.get(tabu_key, -1) > iteration
                    candidate_graph = apply_2_switch(current, swap)
                    c_cost = short_cycle_cost(candidate_graph, self.g_target)
                    delta = c_cost - current_cost
                    aspirate = c_cost < best_cost
                    if not is_tabu or aspirate:
                        best_swap = swap
                        best_delta = delta
                        best_candidate_cost = c_cost
                        break
            else:
                # Exact evaluation of every candidate
                for swap in candidates:
                    tabu_key = self._tabu_key(swap)
                    is_tabu = tabu.get(tabu_key, -1) > iteration
                    candidate_graph = apply_2_switch(current, swap)
                    c_cost = short_cycle_cost(candidate_graph, self.g_target)
                    delta = c_cost - current_cost
                    aspirate = c_cost < best_cost
                    if is_tabu and not aspirate:
                        continue
                    if c_cost < best_candidate_cost:
                        best_swap = swap
                        best_delta = delta
                        best_candidate_cost = c_cost

            if best_swap is None:
                stagnation += 1
            else:
                # Apply best swap
                tabu[self._tabu_key(best_swap)] = iteration + tenure
                current = apply_2_switch(current, best_swap)
                current_cost = best_candidate_cost

                if current_cost < best_cost:
                    best_cost = current_cost
                    best = current.copy()
                    stagnation = 0
                else:
                    stagnation += 1

            # --- 3-switch escalation ---
            if (
                self.escalate_to_3switch
                and stagnation >= self.stagnation_limit
                and current_cost > 0.0
            ):
                escalations += 1
                stagnation = 0
                best3: Swap3 | None = None
                best3_cost = current_cost

                for swap3 in enumerate_3_switches(current, self.sample_size):
                    g3 = apply_3_switch(current, swap3)
                    c3 = short_cycle_cost(g3, self.g_target)
                    if c3 < best3_cost:
                        best3_cost = c3
                        best3 = swap3

                if best3 is not None:
                    current = apply_3_switch(current, best3)
                    current_cost = best3_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best = current.copy()

        stats: dict[str, Any] = {
            "initial_cost": initial_cost,
            "final_cost": best_cost,
            "iterations": min(self.max_iter, iteration + 1) if self.max_iter > 0 else 0,
            "escalations": escalations,
            "improved": best_cost < initial_cost,
        }
        return best, stats

    @staticmethod
    def _tabu_key(
        swap: Swap2,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        """Canonical tabu key for a 2-switch (symmetric under reversal)."""
        u, v, x, y, mode = swap
        if mode == "uxvy":
            removed = (min(u, v), max(u, v)), (min(x, y), max(x, y))
            added = (min(u, x), max(u, x)), (min(v, y), max(v, y))
        else:
            removed = (min(u, v), max(u, v)), (min(x, y), max(x, y))
            added = (min(u, y), max(u, y)), (min(v, x), max(v, x))
        return removed[0], removed[1], added[0], added[1]

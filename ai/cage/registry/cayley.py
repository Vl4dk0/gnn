"""Cayley graph search generator for the cage backend.

Implements the same generator protocol as VoltageSearchGenerator: a step()
method that advances the search by one tabu move (or random restart on a
new group), with `graph`, `is_complete`, and `success` reflecting the
current best.
"""

from __future__ import annotations

import random
import time

import networkx as nx

from ai.cage.cayley.cayley import (
    build_cayley,
    cayley_girth,
    count_short_relations,
)
from ai.cage.cayley.groups import FiniteGroup, available_groups
from ai.cage.cayley.generators.search import candidate_swaps, random_generating_set
from backend.utils.graph_utils import is_k_regular, moore_bound


class CayleySearchGenerator:
    """Cage generator using Cayley graph search.

    On each step() we try one tabu move on the current (group, S). When
    progress stalls or the search yields a verified (k, g)-Cayley graph
    smaller than the current best, the corresponding lift is exposed as
    `self.graph`.
    """

    k: int
    g: int
    graph: nx.Graph[int]
    step_count: int
    is_complete: bool
    success: bool
    start_time: float

    _groups: list[FiniteGroup]
    _group_idx: int
    _group: FiniteGroup
    _gens: list[int]
    _tabu: dict[int, int]
    _best_girth: int
    _restarts: int

    def __init__(self, k: int, g: int, model_id: str | None = None) -> None:
        self.k = k
        self.g = g
        self.step_count = 0
        self.is_complete = False
        self.success = False
        self.start_time = 0.0
        self._best_girth = 0
        self._tabu = {}
        self._restarts = 0
        # model_id reserved for the Phase 2 ML-guided variant; the current
        # search is classical only and ignores it.
        _ = model_id

        mb = moore_bound(k, g)
        candidates = available_groups(max_order=max(8 * mb, 200))
        candidates = [
            grp for grp in candidates if mb <= grp.order <= max(4 * mb, mb + 500)
        ]
        if not candidates:
            candidates = available_groups(max_order=max(8 * mb, 200))
        candidates.sort(key=lambda grp: grp.order)

        self._groups = candidates
        self._group_idx = 0
        self._group = candidates[0] if candidates else available_groups(60)[0]

        initial = self._fresh_generators()
        self._gens = initial if initial is not None else []

        self.graph = nx.Graph()
        for i in range(mb):
            _ = self.graph.add_node(i)

    def elapsed_time(self) -> float:
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    def is_regular(self) -> bool:
        return is_k_regular(self.graph, self.k)

    def _fresh_generators(self) -> list[int] | None:
        for _ in range(20):
            gens = random_generating_set(self._group, self.k)
            if gens is not None and len(gens) == self.k:
                return gens
        return None

    def _advance_group(self) -> None:
        if not self._groups:
            return
        self._group_idx = (self._group_idx + 1) % len(self._groups)
        self._group = self._groups[self._group_idx]
        initial = self._fresh_generators()
        self._gens = initial if initial is not None else self._gens
        self._tabu.clear()

    def step(self) -> None:
        if self.start_time == 0:
            self.start_time = time.time()
        self.step_count += 1
        if self.is_complete or not self._gens:
            return

        current_cost = count_short_relations(self._group, self._gens, self.g - 1)
        if current_cost == 0:
            girth = cayley_girth(self._group, self._gens, max_girth=2 * self.g)
            if isinstance(girth, int) and girth >= self.g:
                cay = build_cayley(self._group, self._gens)
                if nx.is_connected(cay) and cay.degree(0) == self.k:
                    self.graph = cay
                    self.is_complete = True
                    self.success = True
                    self._best_girth = girth
                    return
            # Degenerate cost-0 — rotate.
            self._advance_group()
            self._restarts += 1
            return

        candidates = candidate_swaps(self._group, self._gens)
        if not candidates:
            self._advance_group()
            self._restarts += 1
            return
        random.shuffle(candidates)
        candidates = candidates[:80]

        best_cost = current_cost + 1
        best_new: list[int] | None = None
        best_replaced: int = -1
        for _kind, s_old, new_gens in candidates:
            cost = count_short_relations(self._group, new_gens, self.g - 1)
            is_tabu = self._tabu.get(s_old, -1) > self.step_count
            if cost < best_cost and (not is_tabu or cost == 0):
                best_cost = cost
                best_new = new_gens
                best_replaced = s_old

        if best_new is None or best_cost > current_cost:
            # Stuck — rotate to next group.
            self._advance_group()
            self._restarts += 1
            return

        self._gens = best_new
        self._tabu[best_replaced] = self.step_count + 10

        if self.step_count % 25 == 0:
            girth = cayley_girth(self._group, self._gens, max_girth=2 * self.g)
            if isinstance(girth, int) and girth > self._best_girth:
                cay = build_cayley(self._group, self._gens)
                if cay.degree(0) == self.k and nx.is_connected(cay):
                    self.graph = cay
                    self._best_girth = girth
                    if girth >= self.g:
                        self.is_complete = True
                        self.success = True

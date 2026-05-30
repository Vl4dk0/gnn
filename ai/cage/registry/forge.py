"""Step-through forge cage generator: voltage -> refine -> excision cascade.

``ForgeGenerator`` drives the same three-method cascade as
``ai.cage.forge.forge_graph`` (one-shot), but exposes it one ``step()`` at a
time so the backend can inspect it like every other cage generator.  It is a
three-stage state machine:

  1. **voltage**  — embed a ``VoltageSearchGenerator`` and delegate ``step()``.
     On success the lift becomes the working graph; if its girth already
     reaches g we jump straight to excision, otherwise we move to refine.  If
     the voltage search exhausts without success the whole forge fails.
  2. **refine**   — drive a ``TabuRefineGenerator`` on the lift until the girth
     reaches g (advance to excision) or the refiner exhausts (forge fails).
  3. **excision** — each ``step()`` tries ONE excision root via
     ``try_excise_root``.  A successful root replaces the working graph with a
     strictly smaller (k,g)-graph; once every root has been tried with no
     further shrink the forge completes successfully, holding a valid
     (k,g)-graph.

Trained models are loaded exactly as ``forge_graph`` does (voltage girth/tabu
predictor by ``model_id`` for the voltage stage, move oracle for refine, repair
policy for excision), each degrading gracefully to its classical fallback when
absent.  ``stage`` reports the active sub-stage and is also surfaced in the
backend status payload.
"""

from __future__ import annotations

import time
from typing import Literal

import networkx as nx

from ai.cage.excision.repair_search import (
    RepairPolicy,
    load_repair_policy,
    try_excise_root,
)
from ai.cage.forge import _load_refine_score_fn, _ScoreFn  # pyright: ignore[reportPrivateUsage]
from ai.cage.refine.tabu import TabuRefineGenerator
from ai.cage.registry.types import CageStepEvent
from ai.cage.registry.voltage import VoltageSearchGenerator
from backend.utils.graph_utils import compute_girth, is_k_regular, moore_bound

_Stage = Literal["voltage", "refine", "excision"]


def _girth_value(G: nx.Graph[int]) -> int:
    """Integer girth of G (a large sentinel if acyclic)."""
    girth = compute_girth(G)
    return 10**9 if isinstance(girth, float) else girth


class ForgeGenerator:
    """Cage generator cascading voltage -> refine -> excision, one step at a time."""

    k: int
    g: int
    graph: nx.Graph[int]
    step_count: int
    is_complete: bool
    success: bool
    stage: _Stage | None
    start_time: float
    last_event: CageStepEvent | None

    _model_id: str | None
    _refine_max_iter: int
    _refine_score_fn: _ScoreFn | None
    _repair_policy: RepairPolicy | None
    _voltage: VoltageSearchGenerator
    _refiner: TabuRefineGenerator | None
    _excision_roots: list[int]
    _excision_idx: int

    def __init__(
        self,
        k: int,
        g: int,
        model_id: str | None = None,
        *,
        refine_max_iter: int = 300,
    ) -> None:
        self.k = k
        self.g = g
        self.step_count = 0
        self.is_complete = False
        self.success = False
        self.stage = "voltage"
        self.start_time = 0.0
        self.last_event = None

        self._model_id = model_id
        self._refine_max_iter = refine_max_iter
        # Same classical-fallback loaders as forge_graph: absent models -> None.
        self._refine_score_fn = _load_refine_score_fn(verbose=False)
        self._repair_policy = load_repair_policy()

        self._voltage = VoltageSearchGenerator(k, g, model_id=model_id)
        self._refiner = None
        self._excision_roots = []
        self._excision_idx = 0

        # Placeholder graph (Moore-bound isolated nodes) until the lift arrives.
        self.graph = nx.Graph()
        mb = moore_bound(k, g)
        for i in range(mb):
            self.graph.add_node(i)

    def elapsed_time(self) -> float:
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    def is_regular(self) -> bool:
        return is_k_regular(self.graph, self.k)

    def step(self) -> None:
        """Advance the active stage by one move and update self.graph / self.stage."""
        if self.start_time == 0:
            self.start_time = time.time()
        if self.is_complete:
            return

        self.step_count += 1
        if self.stage == "voltage":
            self._step_voltage()
        elif self.stage == "refine":
            self._step_refine()
        else:
            self._step_excision()

    # --- voltage stage ---------------------------------------------------

    def _step_voltage(self) -> None:
        self._voltage.step()
        self.graph = self._voltage.graph
        action = f"voltage: attempt {self._voltage.step_count}"

        if self._voltage.is_complete:
            if not self._voltage.success:
                # Voltage exhausted without a k-regular lift -> forge failed.
                self._finish(False, "voltage_exhausted", action + " (no lift)")
                return
            lift = self._voltage.graph
            girth = _girth_value(lift)
            action = f"voltage: attempt {self._voltage.step_count} girth {girth}"
            if girth >= self.g:
                # Already a valid (k,g)-lift: skip refine, start excision.
                self._enter_excision(action)
                return
            # k-regular but girth < g: hand the lift to the refine stage.
            # Minimal build: we do NOT loop back to try further voltage bases if
            # refine fails on this lift (known limitation).
            self._refiner = TabuRefineGenerator(
                lift,
                g_target=self.g,
                score_fn=self._refine_score_fn,
                max_iter=self._refine_max_iter,
            )
            self.stage = "refine"
            self._emit(action + " -> refine", done=False)
            return

        self._emit(action, done=False)

    # --- refine stage ----------------------------------------------------

    def _step_refine(self) -> None:
        refiner = self._refiner
        if refiner is None:  # defensive; refine entered only after constructing one
            self._finish(False, "refine_error", "refine: no refiner")
            return

        refiner.step()
        self.graph = refiner.graph
        girth = _girth_value(refiner.graph)
        action = f"refine: {refiner.last_action} girth {girth}"

        if _girth_value(refiner.graph) >= self.g and is_k_regular(
            refiner.graph, self.k
        ):
            self._enter_excision(action)
            return
        if refiner.is_complete:
            # Refiner exhausted without reaching girth g -> forge failed.
            self._finish(False, "refine_exhausted", action + " (failed)")
            return
        self._emit(action, done=False)

    # --- excision stage --------------------------------------------------

    def _enter_excision(self, prior_action: str) -> None:
        """Switch to the excision stage, seeding the root list from self.graph."""
        self.stage = "excision"
        self._excision_roots = sorted(self.graph.nodes())
        self._excision_idx = 0
        self._emit(prior_action + " -> excision", done=False)

    def _step_excision(self) -> None:
        # Exhausted all roots with no further shrink: we hold a valid (k,g)-graph.
        if self._excision_idx >= len(self._excision_roots):
            self._finish(True, "excision_complete", "excision: no further shrink")
            return

        root = self._excision_roots[self._excision_idx]
        self._excision_idx += 1
        before = self.graph.number_of_nodes()
        smaller = try_excise_root(self.graph, self.k, self.g, root, self._repair_policy)
        if smaller is not None:
            self.graph = smaller
            after = smaller.number_of_nodes()
            # Restart the root sweep on the new, smaller graph.
            self._excision_roots = sorted(self.graph.nodes())
            self._excision_idx = 0
            self._emit(f"excision: root {root} shrunk {before}->{after}", done=False)
            return
        self._emit(f"excision: root {root} no shrink", done=False)

    # --- event helpers ---------------------------------------------------

    def _emit(self, action: str, *, done: bool, done_reason: str | None = None) -> None:
        self.last_event = {
            "step": self.step_count,
            "action": action,
            "u": None,
            "v": None,
            "accepted": True,
            "reward": 0.0,
            "done": done,
            "done_reason": done_reason,
        }

    def _finish(self, success: bool, reason: str, action: str) -> None:
        self.success = success
        self.is_complete = True
        self._emit(action, done=True, done_reason=reason)

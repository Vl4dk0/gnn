"""Queue-driven forge cage pipeline: voltage -> refine -> excision.

``ForgeGenerator`` runs the three cage-construction methods as a *cooperative
producer-consumer pipeline* rather than a linear state machine.  Concurrency is
modelled with queues and a round-robin scheduler over a single thread: each
``step()`` advances exactly one unit of work in one stage, so the whole run
stays fully step-through-able for the UI while voltage keeps producing
candidates WHILE refine and excision work on them.

Stages
------
1. **Producer (voltage)** — streams voltage assignments via the RL policy.  Each
   step advances one harvest-mode ``VoltageRLGenerator`` (cycled round-robin
   across a fixed pool) and drains every k-regular, connected lift it found,
   classifying each:

     * girth >= g                              -> excision queue
     * girth in [g - refine_margin, g - 1]     -> refine queue
     * otherwise                                -> dropped

   Pushes are deduplicated by edge-set hash and capped at ``max_candidates``;
   after the cap (or config exhaustion) the producer is *done* and pushes no
   more.

2. **Refine worker** — at most one active ``TabuRefineGenerator``.  When idle
   and the refine queue is non-empty it pops one near-miss and refines it, one
   swap per step.  On reaching girth >= g (and k-regular) it pushes the refined
   graph to the excision queue and goes idle (to pop the next near-miss); on
   refiner exhaustion it discards and goes idle.  It keeps draining the queue.

3. **Excision worker** — at most one active excision.  When idle and the
   excision queue is non-empty it pops one valid (k,g)-graph and greedily
   shrinks it, ONE root per step via ``try_excise_root``.  At each root it tries
   the girth-safe excision depths largest-first and takes the biggest valid
   shrink; falling back to smaller depths is essential, since a maximal-depth
   tree often removes so many vertices that the reduced graph drops below the
   Moore bound (unrepairable).  When the shrink loop terminates (no root yields
   a smaller valid graph) the current graph is the RESULT: the pipeline succeeds
   and stops — the first valid (k,g)-graph out of excision wins; we do NOT keep
   looking for a smaller one.

Scheduler
---------
``step()`` round-robins across the three stages, skipping any with no work and
advancing exactly one per call (fairness, not downstream-drain).  Termination:

  * excision produces a result            -> success + stop, EXCEPT termination
    is briefly (and boundedly) deferred so an in-flight refiner can finish its
    current near-miss instead of being outraced by a fast direct-hit excision
  * producer done, both queues empty, no   -> fail + stop
    active refine/excision, no result
  * ``max_total_steps`` safety cap reached  -> stop

Trained models load exactly as ``ai.cage.forge.forge_graph`` does, each
degrading to its classical fallback when absent.
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
from ai.cage.refine.move_oracle import ScoreFn, load_refine_score_fn
from ai.cage.refine.tabu import TabuRefineGenerator
from ai.cage.registry.types import CageStepEvent
from ai.cage.registry.voltage_rl import VoltageRLGenerator
from backend.utils.graph_utils import compute_girth, is_k_regular, moore_bound

_Stage = Literal["voltage", "refine", "excision"]

# Refine only realistically closes a small girth gap (the move oracle is trained
# on g-1 / g-2 near-misses), so a near-miss lift is only worth refining when its
# girth is within this many of the target.
REFINE_MARGIN = 2

# Default hard cap on the total number of lifts the producer pushes downstream.
DEFAULT_MAX_CANDIDATES = 20

# Default safety cap on total scheduler steps before the pipeline gives up.
DEFAULT_MAX_TOTAL_STEPS = 20000

# Number of (base, group) voltage configs the producer keeps live and cycles
# round-robin.  Capping keeps the producer cheap while still using >1 base.
_MAX_PRODUCER_CONFIGS = 6

# Candidate 2-switches sampled per refine step.  The exhaustive O(|E|^2) scan is
# far too slow for step-through on lifts of a few dozen vertices, so each step
# examines only this many random swaps; refine still closes typical near-misses.
_REFINE_SAMPLE_SIZE = 80

# A refiner is "close" to girth g (worth letting finish before the run ends)
# when its short-cycle cost is at or below this small threshold.
_REFINE_CLOSE_COST = 4.0

# Max scheduler steps termination may be deferred to let an in-flight refiner
# finish.  Bounded so a stuck refiner can never stall the pipeline forever.
_REFINE_DEFER_BUDGET = 2000


def _girth_value(G: nx.Graph[int]) -> int:
    """Integer girth of G (a large sentinel if acyclic)."""
    girth = compute_girth(G)
    return 10**9 if isinstance(girth, float) else girth


def _edge_set_hash(G: nx.Graph[int]) -> int:
    """Order-independent hash of G's edge set, for dedup of harvested lifts."""
    return hash(frozenset(frozenset((u, v)) for u, v in G.edges()))


class _ExcisionWorker:
    """Greedily shrinks one valid (k,g)-graph, one root per step.

    Sweeps roots of the current graph; for each root it tries each girth-safe
    excision depth LARGEST tree first and takes the first depth that yields a
    strictly smaller valid (k,g)-graph (the biggest shrink at that root).  That
    graph replaces the current graph and restarts the sweep.  ``done`` is set
    once every root has been tried with no further shrink — the current graph is
    then a terminal (k,g)-graph.

    Trying multiple depths (not only the maximal ``(g-1)//2``) matters: a
    maximal-depth tree removes so many vertices that the reduced graph can drop
    below the Moore bound (unrepairable) or leave a boundary the classical
    repair cannot close, so the maximal excision alone often shrinks NOTHING even
    when a smaller excision would.  Falling back to smaller depths recovers those
    shrinks.
    """

    graph: nx.Graph[int]
    done: bool

    _k: int
    _g: int
    _policy: RepairPolicy | None
    _depths: list[int]
    _roots: list[int]
    _idx: int
    _max_backtracks: int

    def __init__(
        self,
        graph: nx.Graph[int],
        k: int,
        g: int,
        policy: RepairPolicy | None,
        *,
        max_backtracks: int = 300,
    ) -> None:
        self.graph = graph
        self.done = False
        self._k = k
        self._g = g
        self._policy = policy
        # Girth-safe excision depths, LARGEST tree first: a larger excision
        # shrinks more per step, so we take the biggest valid shrink at a root
        # and only fall back to smaller depths when the larger one overshoots the
        # Moore bound or its boundary cannot be repaired.
        self._depths = list(range((g - 1) // 2, 0, -1))
        self._roots = sorted(graph.nodes())
        self._idx = 0
        # Per-attempt backtracking cap.  The stepping worker must stay cheap (it
        # runs one root-attempt per scheduler step across many depths), so we use
        # a far smaller cap than the one-shot ``excise_and_repair`` default.
        self._max_backtracks = max_backtracks

    def step(self) -> str:
        """Try one root (depths largest-first); return a description of the move."""
        if self._idx >= len(self._roots):
            self.done = True
            return "excision: no further shrink (terminal)"

        root = self._roots[self._idx]
        self._idx += 1
        before = self.graph.number_of_nodes()

        # Accept the first (largest) depth that yields a valid smaller graph.
        best: nx.Graph[int] | None = None
        for depth in self._depths:
            cand = try_excise_root(
                self.graph,
                self._k,
                self._g,
                root,
                self._policy,
                depth=depth,
                max_backtracks=self._max_backtracks,
            )
            if cand is not None:
                best = cand
                break

        if best is not None:
            self.graph = best
            after = best.number_of_nodes()
            self._roots = sorted(self.graph.nodes())
            self._idx = 0
            return f"excision: root {root} shrunk {before}->{after}"
        if self._idx >= len(self._roots):
            self.done = True
            return f"excision: root {root} no shrink (terminal)"
        return f"excision: root {root} no shrink"


class ForgeGenerator:
    """Queue-driven voltage -> refine -> excision pipeline, one step at a time."""

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
    _use_refine: bool
    _use_excision: bool
    _refine_max_iter: int
    _refine_sample_size: int
    _refine_margin: int
    _max_candidates: int
    _max_total_steps: int
    _refine_score_fn: ScoreFn | None
    _repair_policy: RepairPolicy | None

    # Producer state.
    _producers: list[VoltageRLGenerator]
    _producer_idx: int
    _producer_done: bool
    _pushed: int
    _seen_hashes: set[int]

    # Queues and workers.
    _refine_queue: list[nx.Graph[int]]
    _excise_queue: list[nx.Graph[int]]
    _refiner: TabuRefineGenerator | None
    _excision: _ExcisionWorker | None

    # Round-robin scheduler cursor over the three stages.
    _rr: int
    _result: nx.Graph[int] | None

    # Deferral of excision-result termination to let an in-flight refiner finish.
    _refine_close_cost: float
    _refine_defer_budget: int
    _refine_min_grant: int
    _refine_defer_used: int

    def __init__(
        self,
        k: int,
        g: int,
        model_id: str | None = None,
        *,
        use_refine: bool = True,
        use_excision: bool = True,
        refine_max_iter: int = 300,
        refine_sample_size: int = _REFINE_SAMPLE_SIZE,
        refine_margin: int = REFINE_MARGIN,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_total_steps: int = DEFAULT_MAX_TOTAL_STEPS,
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
        self._use_refine = use_refine
        self._use_excision = use_excision
        self._refine_max_iter = refine_max_iter
        self._refine_sample_size = refine_sample_size
        self._refine_margin = refine_margin
        self._max_candidates = max_candidates
        self._max_total_steps = max_total_steps
        # Same classical-fallback loaders as forge_graph: absent models -> None.
        self._refine_score_fn = load_refine_score_fn(verbose=False)
        self._repair_policy = load_repair_policy()

        # Build a fixed pool of harvest-mode RL voltage producers, cycled
        # round-robin so multiple concurrent policy rollouts explore in parallel.
        self._producers = self._build_producers(k, g)
        self._producer_idx = 0
        self._producer_done = len(self._producers) == 0
        self._pushed = 0
        self._seen_hashes = set()

        self._refine_queue = []
        self._excise_queue = []
        self._refiner = None
        self._excision = None

        self._rr = 0
        self._result = None

        self._refine_close_cost = _REFINE_CLOSE_COST
        self._refine_defer_budget = _REFINE_DEFER_BUDGET
        # Grant an in-flight refiner roughly one near-miss worth of steps before
        # we start requiring it to stay close — bounded by the defer budget.
        self._refine_min_grant = min(refine_max_iter, _REFINE_DEFER_BUDGET)
        self._refine_defer_used = 0

        # Placeholder graph (Moore-bound isolated nodes) until the first lift.
        self.graph = nx.Graph()
        mb = moore_bound(k, g)
        for i in range(mb):
            _ = self.graph.add_node(i)

    def _build_producers(self, k: int, g: int) -> list[VoltageRLGenerator]:
        """Fixed pool of harvest-mode RL voltage generators.

        Each ``VoltageRLGenerator`` runs independent policy rollouts and
        streams every k-regular connected lift it finds via ``.harvested``.
        The RL generator's stochastic action sampling gives each producer
        natural diversity without needing distinct base/group seeds.
        """
        producers: list[VoltageRLGenerator] = []
        for _ in range(_MAX_PRODUCER_CONFIGS):
            producers.append(VoltageRLGenerator(k, g, harvest=True))
        return producers

    def elapsed_time(self) -> float:
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    def is_regular(self) -> bool:
        return is_k_regular(self.graph, self.k)

    # --- scheduler -------------------------------------------------------

    def step(self) -> None:
        """Round-robin one unit of work across producer / refine / excision."""
        if self.start_time == 0:
            self.start_time = time.time()
        if self.is_complete:
            return

        self.step_count += 1
        if self.step_count >= self._max_total_steps:
            self._stop_on_result_or_fail(forced=True)
            return

        # Stage order this turn.  Normally fair round-robin.  But once excision
        # has already produced a result and refine work is still pending, bias
        # HARD toward refine so its near-miss gets a meaningful slice of
        # refine_max_iter on screen instead of being outraced by excision
        # re-stepping a finished worker.
        if self._result is not None and self._has_pending_refine():
            order = [1, 0, 2]  # refine first, then producer, then excision
        else:
            order = [(self._rr + offset) % 3 for offset in range(3)]

        for stage_idx in order:
            advanced = self._advance_stage(stage_idx)
            if advanced:
                self._rr = (stage_idx + 1) % 3
                # First valid (k,g)-graph out of excision normally wins and stops
                # the run.  But a single fast excision of a voltage direct-hit
                # would otherwise end the run before a slow, many-swap refiner
                # ever closes its near-miss — refine would never contribute.  So
                # defer termination (bounded) while ANY refine work is pending so
                # a queued near-miss is still popped, run, and shown.
                if self._result is not None and not self._defer_for_refiner():
                    self._finish(True, "excision_complete", self.last_action())
                return

        # No stage had work this turn: the pipeline is drained.
        self._stop_on_result_or_fail(forced=False)

    def _has_pending_refine(self) -> bool:
        """True while there is refine work to do: active refiner or queued near-miss."""
        return self._refiner is not None or bool(self._refine_queue)

    def _defer_for_refiner(self) -> bool:
        """True iff we should keep running to let refine finish a near-miss.

        An excision of a voltage direct-hit can terminate the run in a handful of
        steps — long before a slow, many-swap refiner closes its near-miss — so
        refine would never contribute.  To give refine a fair chance we defer the
        excision-result termination while there is ANY refine work pending (an
        active refiner OR a near-miss still queued), for a bounded number of
        scheduler steps (so a stuck refiner can never stall the pipeline forever).

        A queued-but-not-yet-started near-miss is NEVER abandoned: as long as the
        refine queue is non-empty we keep deferring (within budget) so the
        scheduler pops it and runs it.  Only once the queue is empty and the sole
        remaining refiner has consumed its initial grant do we additionally
        require it to stay close to girth g; if it drifts far we stop waiting.
        """
        if self._refine_defer_used >= self._refine_defer_budget:
            return False
        # No refine work at all (no active refiner, empty queue): nothing to wait
        # for.
        if not self._has_pending_refine():
            return False
        # A queued near-miss must still get its turn: always defer while the queue
        # is non-empty (within budget) so it is popped and run.
        if self._refine_queue:
            self._refine_defer_used += 1
            return True
        # Only an active refiner remains.  During its initial grant always defer;
        # afterwards keep deferring only while it stays close to a solution.
        if self._refine_defer_used >= self._refine_min_grant:
            if (
                self._refiner is None
                or self._refiner.current_cost > self._refine_close_cost
            ):
                return False
        self._refine_defer_used += 1
        return True

    def _advance_stage(self, stage_idx: int) -> bool:
        """Advance stage *stage_idx* by one unit; return True if work was done."""
        if stage_idx == 0:
            return self._step_producer()
        if stage_idx == 1:
            return self._step_refine()
        return self._step_excision()

    def last_action(self) -> str:
        ev = self.last_event
        return ev["action"] if ev is not None else ""

    def _queue_tag(self) -> str:
        return f"[refine:{len(self._refine_queue)} excise:{len(self._excise_queue)}]"

    # --- producer (voltage) ---------------------------------------------

    def _step_producer(self) -> bool:
        if self._producer_done:
            return False

        producer = self._producers[self._producer_idx]
        producer.step()
        self._producer_idx = (self._producer_idx + 1) % len(self._producers)

        # Drain everything this producer harvested.
        queued_msgs: list[str] = []
        for lift, girth in producer.harvested:
            if self._pushed >= self._max_candidates:
                break
            h = _edge_set_hash(lift)
            if h in self._seen_hashes:
                continue
            if not nx.is_connected(lift) or not is_k_regular(lift, self.k):
                continue
            if girth >= self.g:
                self._seen_hashes.add(h)
                self._excise_queue.append(lift)
                self._pushed += 1
                queued_msgs.append(f"valid girth {girth} -> excise")
            elif self._use_refine and girth >= self.g - self._refine_margin:
                self._seen_hashes.add(h)
                self._refine_queue.append(lift)
                self._pushed += 1
                queued_msgs.append(f"near-miss girth {girth} -> refine")
            # else: dropped (too far below g for refine to close, or refine disabled)
        producer.harvested.clear()

        # Producer is done once the cap is hit or every producer has finished.
        if self._pushed >= self._max_candidates:
            self._producer_done = True
        elif all(p.is_complete for p in self._producers):
            self._producer_done = True

        self.stage = "voltage"
        self.graph = producer.graph
        if queued_msgs:
            action = f"voltage: queued {', '.join(queued_msgs)} {self._queue_tag()}"
        else:
            action = f"voltage: searching {self._queue_tag()}"
        self._emit(action)
        return True

    # --- refine worker ---------------------------------------------------

    def _step_refine(self) -> bool:
        if not self._use_refine:
            return False
        # Start a refiner if idle and there is a near-miss waiting.
        if self._refiner is None:
            if not self._refine_queue:
                return False
            near_miss = self._refine_queue.pop(0)
            self._refiner = TabuRefineGenerator(
                near_miss,
                g_target=self.g,
                score_fn=self._refine_score_fn,
                max_iter=self._refine_max_iter,
                sample_size=self._refine_sample_size,
            )

        refiner = self._refiner
        refiner.step()
        self.stage = "refine"
        # Render the IN-PROGRESS search graph (every accepted 2-switch /
        # 3-switch, improving or not), not just best-so-far, so the user sees
        # each swap change the graph.
        in_progress = refiner.current_graph
        self.graph = in_progress
        girth = _girth_value(in_progress)

        if girth >= self.g and is_k_regular(in_progress, self.k):
            # Refined up to a valid (k,g)-graph: hand it to excision, go idle.
            self._excise_queue.append(in_progress.copy())
            self._refiner = None
            self._emit(f"refine: reached girth {girth} -> excise {self._queue_tag()}")
            return True
        if refiner.is_complete:
            # Exhausted without reaching girth g: discard and go idle.
            self._refiner = None
            self._emit(f"refine: exhausted (discard) {self._queue_tag()}")
            return True

        self._emit(f"refine: {refiner.last_action} girth {girth} {self._queue_tag()}")
        return True

    # --- excision worker -------------------------------------------------

    def _step_excision(self) -> bool:
        # When excision is disabled, the first valid graph from the excise queue
        # is the result immediately — unshrunk.
        if not self._use_excision:
            if not self._excise_queue:
                return False
            graph = self._excise_queue.pop(0)
            self._result = graph
            self.graph = graph
            self.stage = "excision"
            self._emit(
                f"excision: disabled, using first valid (k,g)-graph unshrunk {self._queue_tag()}"
            )
            return True

        # Start an excision if idle and there is a valid graph waiting.
        if self._excision is None:
            if not self._excise_queue:
                return False
            graph = self._excise_queue.pop(0)
            self._excision = _ExcisionWorker(graph, self.k, self.g, self._repair_policy)

        worker = self._excision
        action = worker.step()
        self.stage = "excision"
        self.graph = worker.graph
        self._emit(f"{action} {self._queue_tag()}")

        if worker.done:
            # First terminal valid (k,g)-graph out of excision wins: stop.
            self._result = worker.graph
            self.graph = worker.graph
        return True

    # --- termination -----------------------------------------------------

    def _stop_on_result_or_fail(self, *, forced: bool) -> None:
        if self._result is not None:
            self.graph = self._result
            self._finish(True, "excision_complete", "excision: terminal (k,g)-graph")
            return
        reason = "max_total_steps" if forced else "pipeline_drained"
        action = (
            "forge: step cap reached"
            if forced
            else "forge: producer done, queues empty, no result"
        )
        self._finish(False, reason, action)

    # --- event helpers ---------------------------------------------------

    def _emit(self, action: str) -> None:
        self.last_event = {
            "step": self.step_count,
            "action": action,
            "u": None,
            "v": None,
            "accepted": True,
            "reward": 0.0,
            "done": False,
            "done_reason": None,
        }

    def _finish(self, success: bool, reason: str, action: str) -> None:
        # Ensure the public ``graph`` reflects the excision result, not the last
        # producer graph.  The producer keeps overwriting ``self.graph`` during
        # deferred-termination steps; restore the result here so callers always
        # see the correct (k,g)-graph on completion.
        if success and self._result is not None:
            self.graph = self._result
        self.success = success
        self.is_complete = True
        self.last_event = {
            "step": self.step_count,
            "action": action,
            "u": None,
            "v": None,
            "accepted": True,
            "reward": 0.0,
            "done": True,
            "done_reason": reason,
        }

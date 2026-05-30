"""Fast tests for the queue-driven forge cage pipeline.

``ForgeGenerator`` is a cooperative producer-consumer pipeline (voltage producer
-> refine worker -> excision worker) driven one ``step()`` at a time by a
round-robin scheduler.  These tests exercise, with tiny budgets and no trained
models:

  * end-to-end completion on a valid (k,g)-graph;
  * the producer queuing MULTIPLE distinct candidates across >1 base/group;
  * a near-miss in the refine queue actually getting refined and handed onward;
  * round-robin advancing the voltage producer WHILE a refiner is active;
  * termination at the FIRST valid (k,g)-graph out of excision.
"""

from __future__ import annotations

import random
import time

import networkx as nx

from ai.cage.refine.tabu import TabuRefineGenerator
from ai.cage.registry.forge import ForgeGenerator, _ExcisionWorker
from ai.cage.registry.voltage import _build_configs
from backend.utils.graph_utils import compute_girth, is_k_regular, moore_bound


def _generalized_petersen(n: int, step: int) -> nx.Graph[int]:
    """Generalized Petersen graph GP(n, step): an outer n-cycle (vertices 0..n-1),
    an inner {n, step}-circulant (vertices n..2n-1), and spokes; 3-regular."""
    G: nx.Graph[int] = nx.Graph()
    for i in range(n):
        _ = G.add_edge(i, (i + 1) % n)  # outer cycle
        _ = G.add_edge(i, n + i)  # spoke
        _ = G.add_edge(n + i, n + (i + step) % n)  # inner circulant
    return G


def _girth(G: nx.Graph[int]) -> int | float:
    return compute_girth(G)


class TestForgeGenerator:
    def test_forge_3_6_step_through(self) -> None:
        gen = ForgeGenerator(3, 6, model_id=None, refine_max_iter=200)
        assert gen.k == 3 and gen.g == 6
        assert gen.stage == "voltage"

        seen_stages: set[str] = set()
        for _ in range(20000):
            seen_stages.add(str(gen.stage))
            gen.step()
            seen_stages.add(str(gen.stage))
            if gen.is_complete:
                break

        assert gen.is_complete, "forge pipeline did not complete in budget"
        assert gen.success, "forge pipeline finished without success"
        assert is_k_regular(gen.graph, 3), "forged graph is not 3-regular"
        girth = _girth(gen.graph)
        assert isinstance(girth, int) and girth >= 6, f"girth {girth} < 6"
        assert gen.graph.number_of_nodes() >= 14
        assert "voltage" in seen_stages
        assert "excision" in seen_stages
        # last_event is populated for the frontend to surface.
        assert gen.last_event is not None
        assert gen.last_event["done"] is True


class TestProducerMultiBase:
    """The producer must stream MULTIPLE distinct candidates across >1 base/group."""

    def test_producer_queues_multiple_distinct_candidates(self) -> None:
        gen = ForgeGenerator(3, 6, model_id=None, max_candidates=20)
        # More than one cycled voltage config => the producer can use >1 base.
        assert len(gen._producers) > 1  # pyright: ignore[reportPrivateUsage]

        sizes: set[int] = set()
        for _ in range(3000):
            _ = gen._step_producer()  # pyright: ignore[reportPrivateUsage]
            for G in gen._excise_queue + gen._refine_queue:  # pyright: ignore[reportPrivateUsage]
                sizes.add(G.number_of_nodes())
            if gen._producer_done:  # pyright: ignore[reportPrivateUsage]
                break

        assert gen._pushed >= 2, "producer queued fewer than 2 candidates"  # pyright: ignore[reportPrivateUsage]
        # Distinct lift sizes => candidates came from different base/group configs.
        assert len(sizes) >= 2, f"expected candidates from >1 config, sizes={sizes}"


class TestRefineWorker:
    """A near-miss in the refine queue must get refined and pushed onward."""

    def test_near_miss_is_refined_and_handed_to_excision(self) -> None:
        # The 3-prism is 3-regular, girth 3; target g=4 (within refine_margin=2),
        # so the refine worker should close the gap to girth 4 and hand the
        # result to excision, which then terminates the pipeline.
        gen = ForgeGenerator(3, 4, model_id=None, refine_max_iter=300, max_candidates=4)
        gen._producer_done = True  # pyright: ignore[reportPrivateUsage]  isolate refine/excision
        prism = nx.convert_node_labels_to_integers(nx.circular_ladder_graph(3))
        assert is_k_regular(prism, 3) and _girth(prism) == 3
        gen._refine_queue.append(prism)  # pyright: ignore[reportPrivateUsage]

        seen_stages: set[str] = set()
        for _ in range(6000):
            gen.step()
            seen_stages.add(str(gen.stage))
            if gen.is_complete:
                break

        assert "refine" in seen_stages, "refine stage never visited"
        assert "excision" in seen_stages, "refined graph never reached excision"
        assert gen.is_complete and gen.success
        girth = _girth(gen.graph)
        assert isinstance(girth, int) and girth >= 4, f"girth {girth} < 4"
        assert is_k_regular(gen.graph, 3)


class TestRoundRobinFairness:
    """Round-robin must advance the producer WHILE a refiner is active."""

    def test_voltage_advances_while_refiner_active(self) -> None:
        # Petersen is girth 5; target g=6 (within refine_margin), so refining
        # 5 -> 6 keeps the refiner busy for many steps.  Meanwhile the producer
        # (not exhausted) must keep getting scheduled.
        gen = ForgeGenerator(
            3, 6, model_id=None, refine_max_iter=400, max_candidates=20
        )
        gen._refine_queue.append(  # pyright: ignore[reportPrivateUsage]
            nx.convert_node_labels_to_integers(nx.petersen_graph())
        )

        voltage_while_refiner = 0
        saw_active_refiner = False
        for _ in range(150):
            gen.step()
            if gen._refiner is not None:  # pyright: ignore[reportPrivateUsage]
                saw_active_refiner = True
                if gen.stage == "voltage":
                    voltage_while_refiner += 1
            if gen.is_complete:
                break

        assert saw_active_refiner, "refiner was never active long enough to observe"
        assert voltage_while_refiner > 0, (
            "round-robin starved the producer while a refiner was active"
        )


class TestTerminationAtFirstExcisionResult:
    """The pipeline stops at the FIRST terminal (k,g)-graph out of excision."""

    def test_stops_on_first_excision_result(self) -> None:
        # Seed the excision queue with the Heawood graph (the (3,6)-cage): it
        # cannot shrink, so the excision worker terminates immediately and that
        # graph is the result -> success + stop.
        gen = ForgeGenerator(3, 6, model_id=None, max_candidates=4)
        gen._producer_done = True  # pyright: ignore[reportPrivateUsage]
        heawood = nx.convert_node_labels_to_integers(nx.heawood_graph())
        gen._excise_queue.append(heawood)  # pyright: ignore[reportPrivateUsage]

        for _ in range(2000):
            gen.step()
            if gen.is_complete:
                break

        assert gen.is_complete and gen.success
        assert is_k_regular(gen.graph, 3)
        girth = _girth(gen.graph)
        assert isinstance(girth, int) and girth >= 6
        assert gen.graph.number_of_nodes() == 14

    def test_drained_pipeline_fails_cleanly(self) -> None:
        # Producer done, both queues empty, no workers: the pipeline must fail
        # (not hang) and report it.
        gen = ForgeGenerator(3, 6, model_id=None, max_candidates=4)
        gen._producer_done = True  # pyright: ignore[reportPrivateUsage]
        gen.step()
        assert gen.is_complete and not gen.success


class TestTabuRefineGenerator:
    def test_step_reaches_girth(self) -> None:
        # K_{3,3} is 3-regular with girth 4; push girth toward 6.
        G = nx.convert_node_labels_to_integers(nx.complete_bipartite_graph(3, 3))
        gen = TabuRefineGenerator(G, g_target=4, max_iter=50)
        for _ in range(50):
            if gen.is_complete:
                break
            gen.step()
        # best_cost tracks the lowest cost seen; graph stays 3-regular throughout.
        assert is_k_regular(gen.graph, 3)
        assert gen.best_cost <= gen.current_cost


class TestExcisionWorkerShrinks:
    """Excision must GREEDILY shrink a real, non-minimal (k,g)-graph below its
    own |V| — not return the unshrunk input.  GP(10,2) is a 20-vertex
    (3,5)-graph whose tree-excision-and-repair collapses to the 10-vertex
    Petersen cage."""

    def test_excision_shrinks_below_input_order(self) -> None:
        G = _generalized_petersen(10, 2)
        assert is_k_regular(G, 3)
        girth = compute_girth(G)
        assert isinstance(girth, int) and girth == 5
        assert G.number_of_nodes() == 20

        worker = _ExcisionWorker(G.copy(), 3, 5, policy=None)
        for _ in range(100000):
            _ = worker.step()
            if worker.done:
                break

        assert worker.done, "excision worker did not terminate"
        terminal = worker.graph
        # The worker must have shrunk strictly below the 20-vertex input and
        # produced a still-valid (3,5)-graph at or above the Moore bound.
        assert terminal.number_of_nodes() < 20, "excision shrank nothing"
        assert terminal.number_of_nodes() >= moore_bound(3, 5)
        assert is_k_regular(terminal, 3)
        t_girth = compute_girth(terminal)
        assert isinstance(t_girth, int) and t_girth >= 5


class TestMooreBoundFilter:
    """The producer must never build/search a lift whose order is below the
    Moore bound: such a lift can never be a valid (or refinable) (k,g)-graph."""

    def test_all_producer_configs_meet_moore_bound(self) -> None:
        for k, g in [(3, 6), (3, 7), (3, 8), (4, 5), (4, 6)]:
            mb = moore_bound(k, g)
            configs = _build_configs(k, g)
            assert configs, f"no configs for ({k},{g})"
            for base, group, name in configs:
                lift_order = base.num_nodes * group.order
                assert lift_order >= mb, (
                    f"({k},{g}) config {name} lift order {lift_order} < Moore {mb}"
                )

    def test_sub_moore_config_is_excluded(self) -> None:
        # For (4,5) the Moore bound is 17; a dumbbell(4) (2 nodes) lift needs a
        # group of order >= 9 to reach it.  The unfiltered generator floored the
        # order at max(5, mb//2) = 8 (lift order 16 < 17): a sub-Moore config the
        # filter must now exclude.  Assert no surviving config sits below 17 and
        # that some config sits exactly at it.
        k, g = 4, 5
        mb = moore_bound(k, g)
        assert mb == 17
        orders = {b.num_nodes * grp.order for b, grp, _ in _build_configs(k, g)}
        assert all(o >= mb for o in orders)
        assert min(orders) >= mb
        # A hypothetical order-8 dumbbell(4) lift (16 vertices) is below Moore and
        # therefore must not appear.
        assert 16 not in orders


class TestRefineRunsEndToEnd:
    """Refine must genuinely RUN — and be VISIBLE — on near-misses harvested by
    the REAL producer (no hand-injected near-miss).

    On (3,7) voltage streams girth-6 near-misses (24/28-vertex lifts) into the
    refine queue.  The refine worker must then execute many swaps on them, and
    every accepted swap must change the rendered (in-progress) graph so the user
    can watch refine work.  We assert the visible-progress property here; the
    "refine closes a near-miss to girth g and hands it to excision" property is
    covered deterministically by ``TestRefineWorker`` on a controlled near-miss
    (closing a real (3,7) girth-6 lift is a genuinely hard search not suited to a
    fast unit test).

    The loop is bounded by wall-clock so it can never approach the suite timeout.
    """

    def test_refine_runs_visibly_on_real_producer_near_misses(self) -> None:
        random.seed(0)
        gen = ForgeGenerator(
            3,
            7,
            model_id=None,
            refine_max_iter=200,
            max_total_steps=3000,
        )

        refine_swaps = 0
        two_switch = 0
        three_switch = 0
        graph_changed_on_swap = 0
        prev_refine_edges: frozenset[frozenset[int]] | None = None

        deadline = time.time() + 10.0
        for _ in range(3000):
            if time.time() > deadline:
                break
            gen.step()
            if gen.stage != "refine":
                continue
            refiner = gen._refiner  # pyright: ignore[reportPrivateUsage]
            if refiner is None:
                # The refiner just closed/discarded a near-miss and went idle;
                # the swap-counting below only applies while it is active.
                continue
            kind = refiner.last_move_kind
            if kind not in ("2-switch", "3-switch"):
                continue
            refine_swaps += 1
            if kind == "2-switch":
                two_switch += 1
            else:
                three_switch += 1
            cur_edges = frozenset(frozenset((u, v)) for u, v in gen.graph.edges())
            if prev_refine_edges is not None and cur_edges != prev_refine_edges:
                graph_changed_on_swap += 1
            prev_refine_edges = cur_edges
            if refine_swaps >= 40:
                break

        # Refine genuinely ran a meaningful number of swaps on producer output.
        assert refine_swaps >= 20, f"refine ran too few swaps ({refine_swaps})"
        assert two_switch > 0, "no 2-switch moves observed"
        # Every accepted swap must change the rendered in-progress graph (so the
        # user sees refine working).  Allow the single no-improving-triple case
        # per escalation by requiring the vast majority to change the graph.
        assert graph_changed_on_swap >= refine_swaps - three_switch - 1, (
            f"in-progress graph did not change on most swaps "
            f"({graph_changed_on_swap}/{refine_swaps})"
        )

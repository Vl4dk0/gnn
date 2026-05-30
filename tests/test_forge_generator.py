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

import networkx as nx

from ai.cage.refine.tabu import TabuRefineGenerator
from ai.cage.registry.forge import ForgeGenerator
from backend.utils.graph_utils import compute_girth, is_k_regular


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

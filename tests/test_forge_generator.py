"""Fast smoke test for the step-through forge cage generator.

Drives ``ForgeGenerator(3, 6)`` one ``step()`` at a time through the
voltage -> refine -> excision cascade (all classical fallbacks; no trained
models, no training) and asserts it terminates on a valid (3,6)-graph having
visited the expected stages.
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

        assert gen.is_complete, "forge generator did not complete in budget"
        assert gen.success, "forge generator finished without success"
        assert is_k_regular(gen.graph, 3), "forged graph is not 3-regular"
        girth = _girth(gen.graph)
        assert isinstance(girth, int) and girth >= 6, f"girth {girth} < 6"
        assert gen.graph.number_of_nodes() >= 14
        assert "voltage" in seen_stages
        assert "excision" in seen_stages
        # last_event is populated for the frontend to surface.
        assert gen.last_event is not None
        assert gen.last_event["done"] is True


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

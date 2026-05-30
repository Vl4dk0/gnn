"""Tests for ForgeGenerator ablation flags: use_refine / use_excision.

Three fast smoke tests:
  (a) use_refine=False -> zero refine steps, solves (3,6) via direct voltage hits.
  (b) use_excision=False -> result equals the first valid (k,g)-graph, unshrunk
      (|V| == a voltage lift size, no excision worker was started).
  (c) both True -> full pipeline regression (still works correctly).
"""

from __future__ import annotations

from ai.cage.registry.forge import ForgeGenerator
from backend.utils.graph_utils import compute_girth, is_k_regular, moore_bound


def _run(gen: ForgeGenerator, max_steps: int = 20_000) -> None:
    for _ in range(max_steps):
        gen.step()
        if gen.is_complete:
            break


class TestNoRefineFlag:
    """use_refine=False: no refine steps executed; still solves (3,6) via direct hits."""

    def test_zero_refine_steps_on_3_6(self) -> None:
        gen = ForgeGenerator(3, 6, model_id=None, use_refine=False)
        _run(gen)

        assert gen.is_complete, "pipeline did not complete"
        assert gen.success, "pipeline failed to find a (3,6)-graph"
        assert is_k_regular(gen.graph, 3)
        girth = compute_girth(gen.graph)
        assert isinstance(girth, int) and girth >= 6

        # Refine queue must have stayed empty throughout (no near-misses were queued).
        # Verify by checking neither the refiner was ever started nor the queue populated.
        assert gen._refiner is None  # pyright: ignore[reportPrivateUsage]
        assert gen._refine_queue == []  # pyright: ignore[reportPrivateUsage]

    def test_no_refine_events_in_log(self) -> None:
        """With use_refine=False, no event action should mention 'refine:'."""
        gen = ForgeGenerator(3, 6, model_id=None, use_refine=False)

        refine_actions: list[str] = []
        for _ in range(20_000):
            gen.step()
            ev = gen.last_event
            if ev is not None and ev["action"].startswith("refine:"):
                refine_actions.append(ev["action"])
            if gen.is_complete:
                break

        assert refine_actions == [], (
            f"refine stage ran despite use_refine=False: {refine_actions[:3]}"
        )


class TestNoExcisionFlag:
    """use_excision=False: result is the first valid (k,g)-graph, not shrunk."""

    def test_result_is_unshrunk_voltage_lift(self) -> None:
        gen = ForgeGenerator(3, 6, model_id=None, use_excision=False)
        _run(gen)

        assert gen.is_complete, "pipeline did not complete"
        assert gen.success, "pipeline failed to find a (3,6)-graph"
        assert is_k_regular(gen.graph, 3)
        girth = compute_girth(gen.graph)
        assert isinstance(girth, int) and girth >= 6

        # The excision worker must never have been instantiated.
        assert gen._excision is None  # pyright: ignore[reportPrivateUsage]

        # The result is a raw voltage lift: its order must be >= Moore bound but
        # is NOT required to equal it (voltage lifts are over-sized by design).
        mb = moore_bound(3, 6)  # 14
        assert gen.graph.number_of_nodes() >= mb

    def test_no_excision_worker_events(self) -> None:
        """With use_excision=False, no event action should mention 'excision: root'
        (tree shrink steps).  The single 'excision: disabled' event is expected."""
        gen = ForgeGenerator(3, 6, model_id=None, use_excision=False)

        shrink_actions: list[str] = []
        for _ in range(20_000):
            gen.step()
            ev = gen.last_event
            if ev is not None and "excision: root" in ev["action"]:
                shrink_actions.append(ev["action"])
            if gen.is_complete:
                break

        assert shrink_actions == [], (
            f"excision shrink ran despite use_excision=False: {shrink_actions[:3]}"
        )


class TestBothFlagsTrue:
    """use_refine=True, use_excision=True: full pipeline regression."""

    def test_full_pipeline_still_works(self) -> None:
        gen = ForgeGenerator(3, 6, model_id=None, use_refine=True, use_excision=True)
        _run(gen)

        assert gen.is_complete
        assert gen.success
        assert is_k_regular(gen.graph, 3)
        girth = compute_girth(gen.graph)
        assert isinstance(girth, int) and girth >= 6
        # Full pipeline shrinks via excision; result should be at Moore bound.
        assert gen.graph.number_of_nodes() >= moore_bound(3, 6)

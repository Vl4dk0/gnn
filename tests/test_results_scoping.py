"""Tests for benchmark scoping: restrict the cage benchmark to specific
approaches and (k,g) targets via RunConfig (the --approaches / --targets CLI)."""

from __future__ import annotations

from typing import cast

from results.benchmarks.cage import make_tasks
from results.registry import RunConfig, Task


def _approaches(tasks: list[Task]) -> set[str]:
    return {str(t.payload["approach"]) for t in tasks}


def _targets(tasks: list[Task]) -> set[tuple[int, int]]:
    return {(cast(int, t.payload["k"]), cast(int, t.payload["g"])) for t in tasks}


class TestCageScoping:
    def test_default_runs_all_approaches(self) -> None:
        tasks = make_tasks(RunConfig(benchmarks=["cage"], seeds=1))
        assert "forge" in _approaches(tasks)
        assert len(_approaches(tasks)) > 1

    def test_single_approach_filter(self) -> None:
        tasks = make_tasks(
            RunConfig(benchmarks=["cage"], seeds=1, approaches=["forge"])
        )
        assert _approaches(tasks) == {"forge"}
        assert tasks, "expected at least one forge task"

    def test_target_filter(self) -> None:
        tasks = make_tasks(
            RunConfig(benchmarks=["cage"], seeds=1, targets=[(3, 6), (4, 6)])
        )
        assert _targets(tasks) == {(3, 6), (4, 6)}

    def test_approach_and_target_filter_combined(self) -> None:
        tasks = make_tasks(
            RunConfig(
                benchmarks=["cage"],
                seeds=2,
                approaches=["forge"],
                targets=[(3, 6)],
            )
        )
        assert _approaches(tasks) == {"forge"}
        assert _targets(tasks) == {(3, 6)}
        # Forge has several variants (producers, stage ablations, sweeps); with
        # one target the seed count multiplies each variant exactly `seeds` times.
        variant_counts: dict[str, int] = {}
        for t in tasks:
            variant_counts[str(t.payload["variant"])] = (
                variant_counts.get(str(t.payload["variant"]), 0) + 1
            )
        assert len(variant_counts) > 1, "expected multiple forge variants"
        assert all(count == 2 for count in variant_counts.values())
        assert len(tasks) == len(variant_counts) * 2

    def test_unknown_approach_yields_no_tasks(self) -> None:
        tasks = make_tasks(
            RunConfig(benchmarks=["cage"], seeds=1, approaches=["does_not_exist"])
        )
        assert tasks == []

"""Tests for benchmark scoping: restrict the cage benchmark to specific
approaches and (k,g) targets via RunConfig (the --approaches / --targets CLI)."""

from __future__ import annotations

from typing import cast

from ai.cage.registry.forge import FORGE_PRODUCERS
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
        # forge has one spec per producer plus no_refine/no_excision; that many
        # specs x 1 target x 2 seeds.
        expected = (len(FORGE_PRODUCERS) + 2) * 2
        assert len(tasks) == expected
        assert _approaches(tasks) == {"forge"}
        assert _targets(tasks) == {(3, 6)}

    def test_unknown_approach_yields_no_tasks(self) -> None:
        tasks = make_tasks(
            RunConfig(benchmarks=["cage"], seeds=1, approaches=["does_not_exist"])
        )
        assert tasks == []

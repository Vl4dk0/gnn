"""Tests for the throughput benchmark: task generation and single-trial execution."""

from __future__ import annotations

from typing import cast

from results.benchmarks.throughput import DEFAULT_TARGET, execute, make_tasks
from results.registry import RunConfig, Task


def _targets(tasks: list[Task]) -> set[tuple[int, int]]:
    return {(cast(int, t.payload["k"]), cast(int, t.payload["g"])) for t in tasks}


def _approaches(tasks: list[Task]) -> set[str]:
    return {str(t.payload["approach"]) for t in tasks}


def _variants_for(tasks: list[Task], approach: str) -> set[str]:
    return {
        str(t.payload["variant"]) for t in tasks if t.payload["approach"] == approach
    }


class TestThroughputMakeTasks:
    def test_default_target_is_3_12(self) -> None:
        tasks = make_tasks(RunConfig(benchmarks=["throughput"], seeds=1))
        assert all(t.benchmark == "throughput" for t in tasks)
        assert DEFAULT_TARGET == (3, 12)
        assert (3, 12) in _targets(tasks)
        assert _targets(tasks) == {(3, 12)}

    def test_forge_and_astar_present(self) -> None:
        tasks = make_tasks(RunConfig(benchmarks=["throughput"], seeds=1))
        approaches = _approaches(tasks)
        assert "forge" in approaches
        assert "astar" in approaches

    def test_forge_ablation_variants_present(self) -> None:
        tasks = make_tasks(RunConfig(benchmarks=["throughput"], seeds=1))
        forge_variants = _variants_for(tasks, "forge")
        assert "all-stages" in forge_variants
        assert "no-refine" in forge_variants
        assert "no-excision" in forge_variants
        assert "producer-only" in forge_variants

    def test_forge_ablation_payloads_correct(self) -> None:
        tasks = make_tasks(RunConfig(benchmarks=["throughput"], seeds=1))
        forge_tasks = [t for t in tasks if t.payload["approach"] == "forge"]

        by_variant: dict[str, dict[str, object]] = {
            str(t.payload["variant"]): t.payload for t in forge_tasks
        }

        # no-refine: forge_use_refine False, forge_use_excision True
        assert by_variant["no-refine"]["forge_use_refine"] is False
        assert by_variant["no-refine"]["forge_use_excision"] is True

        # no-excision: forge_use_refine True, forge_use_excision False
        assert by_variant["no-excision"]["forge_use_refine"] is True
        assert by_variant["no-excision"]["forge_use_excision"] is False

        # all-stages: both True
        assert by_variant["all-stages"]["forge_use_refine"] is True
        assert by_variant["all-stages"]["forge_use_excision"] is True

        # producer-only: both False
        assert by_variant["producer-only"]["forge_use_refine"] is False
        assert by_variant["producer-only"]["forge_use_excision"] is False

    def test_approach_filter(self) -> None:
        tasks = make_tasks(
            RunConfig(benchmarks=["throughput"], seeds=1, approaches=["astar"])
        )
        assert _approaches(tasks) == {"astar"}

    def test_target_override(self) -> None:
        tasks = make_tasks(
            RunConfig(
                benchmarks=["throughput"],
                seeds=1,
                approaches=["astar"],
                targets=[(3, 6)],
            )
        )
        assert _targets(tasks) == {(3, 6)}

    def test_throughput_seconds_in_payload(self) -> None:
        tasks = make_tasks(
            RunConfig(
                benchmarks=["throughput"],
                seeds=1,
                extra={"throughput_seconds": 7.0},
            )
        )
        assert all(t.payload["time_budget_s"] == 7.0 for t in tasks)


class TestThroughputExecute:
    def test_single_astar_trial_metrics(self) -> None:
        tasks = make_tasks(
            RunConfig(
                benchmarks=["throughput"],
                seeds=1,
                approaches=["astar"],
                targets=[(3, 6)],
            )
        )
        assert tasks, "expected at least one astar task"
        task = tasks[0]
        # Override to tiny budget so the test finishes fast
        task.payload["max_steps"] = 40
        task.payload["time_budget_s"] = 5.0

        results = execute(task)
        assert len(results) == 1
        r = results[0]
        assert r.benchmark == "throughput"
        assert isinstance(r.success, bool)
        assert r.steps is not None and r.steps <= 40
        assert "candidates_per_sec" in r.metrics
        assert "steps_per_sec" in r.metrics
        assert "candidates_per_step" in r.metrics
        assert "candidates_total" in r.metrics
        assert r.metrics["candidates_per_sec"] >= 0.0
        assert r.metrics["candidates_total"] >= 1

"""Tests for the results.query read API over benchmark runs."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from results.metrics import TrialResult
from results.query import (
    aggregate,
    filter_records,
    list_runs,
    load_records,
    resolve_run,
    target_str,
)


def _trial(
    approach: str, variant: str, k: int, g: int, success: bool, t: float, n: int
):
    return TrialResult(
        benchmark="cage",
        approach=approach,
        variant=variant,
        label=f"({k},{g}) {approach}",
        instance=f"k{k}_g{g}",
        target={"k": k, "g": g},
        success=success,
        elapsed_s=t,
        steps=1,
        n_nodes=n,
        n_edges=n * 2,
        metrics={"moore_bound": float(2 * k), "girth": float(g)},
    )


def _write_run(tmp_path: Path, trials: list[TrialResult]) -> Path:
    run = tmp_path / "2026-01-01_00-00-00"
    run.mkdir()
    (run / "results.jsonl").write_text(
        "\n".join(json.dumps(dataclasses.asdict(t)) for t in trials)
    )
    return run


class TestRunDiscovery:
    def test_latest_run_resolved(self, tmp_path: Path) -> None:
        (tmp_path / "2026-01-01_00-00-00").mkdir()
        (tmp_path / "2026-01-01_00-00-00" / "results.jsonl").write_text("")
        (tmp_path / "2026-02-02_00-00-00").mkdir()
        (tmp_path / "2026-02-02_00-00-00" / "results.jsonl").write_text("")
        assert resolve_run(None, tmp_path).name == "2026-02-02_00-00-00"
        assert [p.name for p in list_runs(tmp_path)][-1] == "2026-02-02_00-00-00"

    def test_missing_run_raises(self, tmp_path: Path) -> None:
        try:
            resolve_run("nope", tmp_path)
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError")


class TestFilterAndAggregate:
    def test_target_str(self) -> None:
        assert target_str(_trial("forge", "", 4, 6, True, 1.0, 26)) == "(4,6)"

    def test_filter_by_approach_and_target(self, tmp_path: Path) -> None:
        trials = [
            _trial("forge", "", 4, 6, True, 1.0, 26),
            _trial("voltage", "Algebraic", 4, 6, True, 1.0, 50),
            _trial("forge", "", 3, 6, True, 1.0, 14),
        ]
        run = _write_run(tmp_path, trials)
        recs = load_records(run)
        only_forge = filter_records(
            recs, benchmark=None, approach="forge", variant=None, target=None
        )
        assert {r.approach for r in only_forge} == {"forge"}
        one_cell = filter_records(
            recs, benchmark=None, approach="forge", variant=None, target=(4, 6)
        )
        assert len(one_cell) == 1 and one_cell[0].n_nodes == 26

    def test_aggregate_uses_solved_for_means(self, tmp_path: Path) -> None:
        # Two (4,6) forge trials: one solved (t=2,n=26), one failed (t=60,n=99).
        trials = [
            _trial("forge", "", 4, 6, True, 2.0, 26),
            _trial("forge", "", 4, 6, False, 60.0, 99),
        ]
        rows = aggregate(load_records(_write_run(tmp_path, trials)))
        assert len(rows) == 1
        row = rows[0]
        assert row["n_trials"] == 2
        assert row["n_solved"] == 1
        assert row["success_rate"] == 0.5
        assert row["mean_time_s"] == 2.0  # failed trial's 60s excluded
        assert row["mean_nodes"] == 26.0  # failed trial's n=99 excluded
        assert row["moore_bound"] == 8

    def test_aggregate_sorted_by_moore_bound(self, tmp_path: Path) -> None:
        trials = [
            _trial("forge", "", 3, 8, True, 1.0, 30),  # moore 6 in this toy metric
            _trial("forge", "", 5, 5, True, 1.0, 26),  # moore 10
        ]
        # moore_bound here is 2*k via the toy metric: (3,8)->6, (5,5)->10
        rows = aggregate(load_records(_write_run(tmp_path, trials)))
        # toy moore_bound = 2*k: (3,8)->6, (5,5)->10; rows sorted ascending.
        assert [r["moore_bound"] for r in rows] == [6, 10]

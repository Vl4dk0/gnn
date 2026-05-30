"""Tests for the benchmark report layer: the (k,g) solvability matrix and the
payload-carrying error stub used for timeouts/crashes."""

from __future__ import annotations

from pathlib import Path

from results.metrics import TrialResult
from results.registry import Task
from results.report import write_reports
from results.runner import _error_result


def _cage_result(k: int, g: int, approach: str, variant: str, success: bool):
    return TrialResult(
        benchmark="cage",
        approach=approach,
        variant=variant,
        label=f"({k},{g}) {approach}",
        instance=f"k{k}_g{g}",
        target={"k": k, "g": g},
        success=success,
        elapsed_s=0.1,
        steps=1,
        n_nodes=10,
        n_edges=15,
        metrics={"girth": float(g)},
    )


class TestSolvabilityMatrix:
    def _summary(self, tmp_path: Path, results: list[TrialResult]) -> str:
        write_reports(tmp_path, results)
        return (tmp_path / "summary.md").read_text()

    def test_matrix_header_present_for_cage(self, tmp_path: Path) -> None:
        results = [_cage_result(3, 5, "voltage", "gnn", True)]
        text = self._summary(tmp_path, results)
        assert "### Solvability by (k, g)" in text
        assert "voltage[gnn]" in text

    def test_tick_cross_and_partial_cells(self, tmp_path: Path) -> None:
        results = [
            _cage_result(3, 5, "voltage", "gnn", True),  # both seeds solve -> ✓
            _cage_result(3, 5, "voltage", "gnn", True),
            _cage_result(3, 5, "rl", "", True),  # 1 of 2 -> 1/2
            _cage_result(3, 5, "rl", "", False),
            _cage_result(3, 7, "rl", "", False),  # 0 of 1 -> ✗
        ]
        text = self._summary(tmp_path, results)
        matrix = text.split("### Solvability by (k, g)")[1]
        rows = {
            line.split("|")[1].strip(): line
            for line in matrix.splitlines()
            if line.startswith("| (")
        }
        assert "✓" in rows["(3,5)"]
        assert "1/2" in rows["(3,5)"]
        assert "✗" in rows["(3,7)"]

    def test_rows_sorted_by_moore_bound(self, tmp_path: Path) -> None:
        # (3,8) Moore=30 should appear AFTER (4,5) Moore=17 despite k being smaller.
        results = [
            _cage_result(3, 8, "voltage", "", True),
            _cage_result(4, 5, "voltage", "", True),
        ]
        text = self._summary(tmp_path, results)
        matrix = text.split("### Solvability by (k, g)")[1]
        order = [
            line.split("|")[1].strip()
            for line in matrix.splitlines()
            if line.startswith("| (")
        ]
        assert order.index("(4,5)") < order.index("(3,8)")

    def test_no_matrix_for_node_tasks(self, tmp_path: Path) -> None:
        # Node tasks carry an empty target -> no matrix should be emitted.
        node = TrialResult(
            benchmark="degree",
            approach="sage",
            variant="",
            label="degree:sage",
            instance="battery",
            target={},
            success=True,
            elapsed_s=0.1,
            steps=None,
            n_nodes=None,
            n_edges=None,
            metrics={"accuracy": 1.0},
        )
        text = self._summary(tmp_path, [node])
        assert "### Solvability by (k, g)" not in text


class TestErrorResultCarriesPayload:
    def test_target_approach_variant_from_payload(self) -> None:
        task = Task(
            benchmark="cage",
            label="(4,5) voltage[gnn] s0",
            payload={"k": 4, "g": 5, "approach": "voltage", "variant": "gnn"},
        )
        r = _error_result(task, "timeout: task exceeded 120s hard cap")
        assert r.target == {"k": 4, "g": 5}
        assert r.approach == "voltage"
        assert r.variant == "gnn"
        assert r.success is False

    def test_missing_kg_yields_empty_target(self) -> None:
        task = Task(benchmark="degree", label="degree:sage", payload={"task": "degree"})
        r = _error_result(task, "boom")
        assert r.target == {}
        assert r.approach == ""

    def test_timeout_stub_appears_as_cross_in_matrix(self, tmp_path: Path) -> None:
        # A timed-out trial must show up as a failure in the matrix, not vanish.
        task = Task(
            benchmark="cage",
            label="(4,5) voltage[gnn] s0",
            payload={"k": 4, "g": 5, "approach": "voltage", "variant": "gnn"},
        )
        stub = _error_result(task, "timeout")
        write_reports(tmp_path, [stub])
        text = (tmp_path / "summary.md").read_text()
        matrix = text.split("### Solvability by (k, g)")[1]
        row = next(line for line in matrix.splitlines() if line.startswith("| (4,5)"))
        assert "✗" in row

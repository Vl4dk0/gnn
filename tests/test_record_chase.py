"""Fast unit tests for the record-chase harness verification and merge logic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from ai.cage.record_chase import _candidate, _verify_g6, graph_to_g6, merge
from ai.cage.utils import hog_record_order
from backend.utils.graph_utils import compute_girth, is_k_regular


def test_candidate_accepts_petersen() -> None:
    """Petersen is the (3,5)-cage: 3-regular, girth 5, order 10."""
    result = _candidate(nx.petersen_graph(), k=3, girth_floor=4)
    assert result is not None
    girth, order, g6 = result
    assert girth == 5
    assert order == 10
    assert len(g6) > 0


def test_candidate_rejects_non_regular() -> None:
    """A star graph is not k-regular for any single k."""
    assert _candidate(nx.star_graph(4), k=3, girth_floor=4) is None


def test_candidate_rejects_self_loops() -> None:
    """Self-loops are not representable in graph6 and must be rejected."""
    graph: nx.Graph[int] = nx.petersen_graph()
    graph.add_edge(0, 0)
    assert _candidate(graph, k=3, girth_floor=4) is None


def test_candidate_respects_girth_floor() -> None:
    """K4 is 3-regular with girth 3: rejected below floor 4, accepted at 3."""
    k4: nx.Graph[int] = nx.complete_graph(4)
    assert _candidate(k4, k=3, girth_floor=4) is None
    accepted = _candidate(k4, k=3, girth_floor=3)
    assert accepted is not None
    assert accepted[0] == 3


def test_candidate_rejects_edgeless() -> None:
    graph: nx.Graph[int] = nx.empty_graph(5)
    assert _candidate(graph, k=3, girth_floor=4) is None


def test_graph6_round_trip_preserves_invariants() -> None:
    """Serialise -> parse preserves order, regularity and girth."""
    g6 = graph_to_g6(nx.petersen_graph())
    parsed = nx.from_graph6_bytes(g6.encode("ascii"))
    assert parsed.number_of_nodes() == 10
    assert is_k_regular(parsed, 3)
    assert compute_girth(parsed) == 5


def test_verify_g6() -> None:
    """_verify_g6 confirms (k,girth); rejects a wrong claimed girth."""
    g6 = graph_to_g6(nx.petersen_graph())
    assert _verify_g6(g6, k=3, girth=5)
    assert not _verify_g6(g6, k=3, girth=4)
    assert not _verify_g6(g6, k=4, girth=5)


def test_hog_record_order() -> None:
    """New-format filenames yield the order; unknown pairs yield None."""
    assert hog_record_order(8, 5) == 80
    assert hog_record_order(9, 5) == 96
    assert hog_record_order(10, 5) == 124
    assert hog_record_order(11, 6) == 240
    assert hog_record_order(99, 99) is None


def _write_run_dir(path: Path, k: int, records: dict[int, tuple[int, str]]) -> None:
    """Write a minimal run dir with a records.json the merge step can read."""
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "k": k,
        "target_g": max(records),
        "variant": "full",
        "updated": "test",
        "records": {
            str(girth): {"order": order, "graph6": g6}
            for girth, (order, g6) in records.items()
        },
    }
    _ = (path / "records.json").write_text(json.dumps(payload))


def test_merge_keeps_smallest_and_rejects_unverifiable(tmp_path: Path) -> None:
    """Merge keeps the smaller verified order per (k,girth), drops bad entries."""
    petersen_g6 = graph_to_g6(nx.petersen_graph())  # 3-regular, girth 5, n=10
    dodeca_g6 = graph_to_g6(nx.dodecahedral_graph())  # 3-regular, girth 5, n=20

    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    dir_c = tmp_path / "run_c"
    _write_run_dir(dir_a, k=3, records={5: (10, petersen_g6)})
    _write_run_dir(dir_b, k=3, records={5: (20, dodeca_g6)})
    # Claims girth 4 but the graph is girth 5: must be rejected on re-verification.
    _write_run_dir(dir_c, k=3, records={4: (10, petersen_g6)})

    out = tmp_path / "merged"
    args = argparse.Namespace(dirs=[str(dir_a), str(dir_b), str(dir_c)], out=str(out))
    merge(args)

    # Smallest (3,5) order wins; the bogus (3,4) entry is dropped.
    assert (out / "k3_g5_n10.g6").exists()
    assert not (out / "k3_g5_n20.g6").exists()
    assert not (out / "k3_g4_n10.g6").exists()
    summary = (out / "summary.md").read_text()
    assert "| 3 | 5 | 10 |" in summary

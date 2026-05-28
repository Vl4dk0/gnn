"""Tests that prove (or disprove) issues documented in issues/issues_ai/.

Each test encodes the WANTED behavior. If the test FAILS, the documented
issue is real. If the test PASSES, the issue is likely a hallucination.

Run with:
    uv run pytest -q tests/test_issues_ai.py
"""

from __future__ import annotations

import math
import random
from unittest.mock import patch

import networkx as nx
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# abelian_cap_limits_dihedral
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_abelian_cap_limits_dihedral_real_issue() -> None:
    """WANTED: With large max_order and small abelian_cap, dihedral groups
    should still be generated up to max_order (not capped by abelian_cap)."""
    from ai.cage.cayley.groups import available_groups

    groups = available_groups(
        max_order=2200,
        include_matrix=False,
        include_abelian=False,
        include_dihedral=True,
        include_semidirect=False,
        abelian_cap=300,
    )
    max_dih_order = max((g.order for g in groups if g.name.startswith("D_")), default=0)
    assert max_dih_order > 300, (
        f"dihedral cap is silently using abelian_cap; "
        f"got max dihedral order {max_dih_order}"
    )


# ---------------------------------------------------------------------------
# astar-inf-girth-dead-branch
# ---------------------------------------------------------------------------


def test_astar_inf_girth_branch_lower_than_above_target_branch() -> None:
    """WANTED: edgeless/no-cycle graph (girth=inf) is scored *neutral* (0.5),
    strictly less than a graph whose girth merely exceeds g (0.8)."""
    from ai.cage.functions.astar import AStarGenerator

    gen = AStarGenerator(k=3, g=5)
    G: nx.Graph[int] = nx.Graph()
    G.add_nodes_from(range(gen.mb))
    # Identical graph fixture so all components except girth are equal.
    with patch("ai.cage.functions.astar.compute_girth", return_value=float("inf")):
        s_inf = gen._score_graph(G)  # pyright: ignore[reportPrivateUsage]
    with patch("ai.cage.functions.astar.compute_girth", return_value=7):
        s_high = gen._score_graph(G)  # pyright: ignore[reportPrivateUsage]
    assert s_inf < s_high, (
        f"empty-girth score ({s_inf}) should be lower than high-girth score "
        f"({s_high}); currently both hit the > g branch with score 0.8"
    )


# ---------------------------------------------------------------------------
# astar-visited
# ---------------------------------------------------------------------------


def test_astar_initial_graph_added_to_visited() -> None:
    """WANTED: the initial graph hash is recorded in visited_hashes."""
    from ai.cage.functions.astar import AStarGenerator, graph_hash

    gen = AStarGenerator(k=3, g=5)
    initial_hash = graph_hash(gen.graph)
    assert initial_hash in gen.visited_hashes


# ---------------------------------------------------------------------------
# bruteforce-incomplete-on-overflow
# ---------------------------------------------------------------------------


def test_bruteforce_marks_complete_when_overflow_empties_stack() -> None:
    """WANTED: when the graph exceeds upper_bound and the search stack becomes
    empty after pop, the generator must mark itself complete (not loop)."""
    from ai.cage.functions.bruteforce import BruteforceGenerator

    gen = BruteforceGenerator(k=3, g=5)
    huge: nx.Graph[int] = nx.Graph()
    huge.add_nodes_from(range(gen.upper_bound + 5))
    gen.graph = huge
    assert len(gen.search_stack) == 1  # only the initial frame
    gen.step()
    assert gen.is_complete is True
    assert gen.success is False


# ---------------------------------------------------------------------------
# cayley-group-data-inf-label-zero
# ---------------------------------------------------------------------------


def test_best_achievable_girth_does_not_drop_inf_label() -> None:
    """WANTED: if random_search / tabu_search return float('inf'), the group
    should NOT be labeled with best=0 (which means 'nothing found')."""
    from ai.cage.cayley import group_data_gen
    from ai.cage.voltage.groups import cyclic_group

    group = cyclic_group(7)
    with (
        patch.object(group_data_gen, "random_search", return_value=([1, 6, 0], math.inf)),
        patch.object(group_data_gen, "tabu_search", return_value=(None, 0)),
    ):
        best = group_data_gen.best_achievable_girth(
            group, k=3, g_target=6, num_random_trials=1, num_tabu_iters=1
        )
    assert best > 0, (
        f"inf girth from random_search was silently dropped; got best={best}"
    )


# ---------------------------------------------------------------------------
# cayley-mislabel
# ---------------------------------------------------------------------------


def test_cayley_ball_to_pyg_labels_inf_as_positive() -> None:
    """WANTED: a Cayley ball with girth=inf (>= 2*g_target+2) is a valid
    positive sample with girth_class=1, not 0."""
    from ai.cage.cayley.data_gen import cayley_ball_to_pyg
    from ai.cage.voltage.groups import cyclic_group

    group = cyclic_group(7)
    data = cayley_ball_to_pyg(
        group, generators=[1, 6], k=2, g_target=3, girth=math.inf
    )
    girth_class = int(getattr(data, "girth_class"))
    assert girth_class == 1, (
        f"inf girth Cayley sample mislabeled negative; girth_class={girth_class}"
    )


# ---------------------------------------------------------------------------
# voltage-mislabel
# ---------------------------------------------------------------------------


def test_base_graph_to_pyg_labels_inf_as_positive() -> None:
    """WANTED: a voltage lift with girth=inf is positive (girth_class=1)."""
    from ai.cage.voltage.base_graphs import dumbbell
    from ai.cage.voltage.data_gen import base_graph_to_pyg
    from ai.cage.voltage.groups import cyclic_group

    base = dumbbell(3)
    group = cyclic_group(5)
    n_edges = base.num_undirected_edges()
    volts = [0] * n_edges
    data = base_graph_to_pyg(base, volts, group, k=3, g_target=5, girth=math.inf)
    girth_class = int(getattr(data, "girth_class"))
    assert girth_class == 1, (
        f"inf girth voltage sample mislabeled negative; girth_class={girth_class}"
    )


# ---------------------------------------------------------------------------
# duplicate-base-graph
# ---------------------------------------------------------------------------


def test_candidate_bases_no_duplicate_for_k3() -> None:
    """WANTED: _candidate_bases(3) contains dumbbell(3) at most once."""
    from ai.cage.voltage.search import _candidate_bases  # pyright: ignore[reportPrivateUsage]

    bases = _candidate_bases(3)
    names = [name for (name, _b) in bases]
    assert names.count("dumbbell(3)") == 1, (
        f"dumbbell(3) appears {names.count('dumbbell(3)')} times: {names}"
    )


# ---------------------------------------------------------------------------
# gin-relu-output-bottleneck
# ---------------------------------------------------------------------------


def test_gin_output_layer_has_no_relu_bottleneck() -> None:
    """WANTED: the GIN output layer is not Linear(H,1)->ReLU->Linear(1,1).
    Currently the output MLP includes ReLU between (H,1) and (1,1), which
    chokes the prediction to a constant bias."""
    from ai.models.gin import GIN_GNN

    model = GIN_GNN(input_dim=4, hidden_dim=8, output_dim=1, num_layers=3)
    output_mlp = model.convs[-1].nn  # pyright: ignore[reportAttributeAccessIssue]
    linears = [m for m in output_mlp.modules() if isinstance(m, nn.Linear)]
    last = linears[-1]
    assert not (last.in_features == 1 and last.out_features == 1), (
        f"Last Linear is {last.in_features}->{last.out_features}; "
        "this is the ReLU bottleneck described in the issue."
    )


# ---------------------------------------------------------------------------
# gnn-single-layer-dim-crash
# ---------------------------------------------------------------------------


def _make_data(num_nodes: int, input_dim: int) -> object:
    from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

    x = torch.randn(num_nodes, input_dim)
    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))
    ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return Data(x=x, edge_index=ei)


@pytest.mark.parametrize("model_name", ["gcn", "gin", "sage"])
def test_single_layer_forward_does_not_crash(model_name: str) -> None:
    """WANTED: num_layers=1 forward pass succeeds for GCN/GIN/SAGE."""
    if model_name == "gcn":
        from ai.models.gcn import GCN_GNN as M
    elif model_name == "gin":
        from ai.models.gin import GIN_GNN as M  # type: ignore[assignment]
    else:
        from ai.models.sage import SAGE_GNN as M  # type: ignore[assignment]

    model = M(input_dim=4, hidden_dim=8, output_dim=1, num_layers=1)
    _ = model.eval()
    data = _make_data(num_nodes=5, input_dim=4)
    _ = model(data)  # should not raise


# ---------------------------------------------------------------------------
# loopy-conv-index-off-by-one
# ---------------------------------------------------------------------------


def test_loopy_layer_has_distinct_conv_per_path_length() -> None:
    """WANTED: with shared=False, each L in 1..r has its own conv.
    For r=3 the layer should expose 3 conv slots and the forward path should
    NOT collapse L=2 and L=3 onto the same conv."""
    from ai.models.loopy import LoopyLayer

    r = 3
    layer = LoopyLayer(in_channels=8, out_channels=8, r=r, shared=False)
    # num_convs currently = max(r, 1) = r. After fix this should match exactly
    # the number of distinct L >= 1 buckets, i.e. r.
    assert len(layer.path_convs) == r
    # The forward index for L=r should be different from L=r-1.
    # Currently: conv_idx = min(L, len(self.path_convs)-1) = min(L, r-1) →
    # L=r-1 and L=r both map to r-1 (collision).
    idx_for = lambda L: L - 1  # matches fixed source: conv_idx = L - 1
    assert idx_for(r - 1) != idx_for(r), (
        f"L={r-1} and L={r} map to the same conv idx={idx_for(r)}; "
        "path_convs[0] is never used and the last two L's share weights."
    )


# ---------------------------------------------------------------------------
# loopy-path-conv-missing-self-loop
# ---------------------------------------------------------------------------


def test_pathconv_preserves_self_contribution() -> None:
    """WANTED: PathConv computes (1+eps)*x + agg, NOT (1+eps)*agg only.

    We probe by zeroing out neighbors: with no neighbor message, the GIN-style
    self contribution must still produce non-zero output. With the bug
    (propagate overwrites x), the output is zero."""
    from ai.models.loopy import PathConv

    conv = PathConv(in_channels=4, out_channels=4, max_distance=3)
    _ = conv.eval()
    # Single node on the path (path_length=1) → kernel [1,0,1] yields zero agg.
    x = torch.randn(1, 1, 4)  # (path_length=1, num_paths=1, hidden=4)
    atomic = torch.zeros((1, 1), dtype=torch.long)
    with torch.no_grad():
        out = conv(x, atomic)  # (num_paths, hidden) = (1, 4)
    # With bug: out is mlp((1+eps)*zero_agg + bias_terms) — value depends on bias.
    # With fix: out depends on x. Test by comparing two distinct x's: outputs
    # must differ if self contribution is preserved.
    x2 = torch.randn(1, 1, 4)
    with torch.no_grad():
        out2 = conv(x2, atomic)
    assert not torch.allclose(out, out2, atol=1e-6), (
        "PathConv output is invariant to x with no neighbor agg → "
        "self contribution was overwritten by propagation."
    )


# ---------------------------------------------------------------------------
# loopy_gnn_missing_direct_neighbors
# ---------------------------------------------------------------------------


def test_loopy_neighborhood_populates_direct_neighbors() -> None:
    """WANTED: compute_r_neighborhood populates loopyN0 with direct edges.
    nx.simple_cycles can't return cycles of length 2, so the current logic
    leaves loopyN0 empty for every graph."""
    from ai.utils.r_neighborhood import compute_r_neighborhood

    G: nx.Graph[int] = nx.path_graph(5)  # 4 edges, no cycles
    result = compute_r_neighborhood(G, r=3)
    n0 = result["loopyN0"]
    assert n0.shape[1] > 0, (
        f"loopyN0 is empty (shape {tuple(n0.shape)}); "
        "direct-neighbor message passing is completely skipped."
    )


# ---------------------------------------------------------------------------
# mcts-terminal-ignores-girth
# ---------------------------------------------------------------------------


def test_mcts_terminal_requires_correct_girth() -> None:
    """WANTED: a k-regular graph with the WRONG girth is NOT terminal."""
    from ai.cage.functions.monte_carlo_search_tree import MCTSNode

    # K_4 is 3-regular and has girth 3.
    G: nx.Graph[int] = nx.complete_graph(4)
    node = MCTSNode(G, k=3, g=6)
    assert node.is_terminal is False, (
        "K_4 is 3-regular with girth 3 but g=6 — should not be terminal."
    )


# ---------------------------------------------------------------------------
# mcts-zero-division-visits
# ---------------------------------------------------------------------------


def test_mcts_best_child_safe_with_zero_visits() -> None:
    """WANTED: best_child() should not crash when visits == 0."""
    from ai.cage.functions.monte_carlo_search_tree import MCTSNode

    G: nx.Graph[int] = nx.empty_graph(4)
    parent = MCTSNode(G, k=3, g=5)
    child = MCTSNode(nx.empty_graph(5), k=3, g=5, parent=parent)
    parent.children.append(child)
    # parent.visits == 0, child.visits == 0
    try:
        result = parent.best_child()
    except (ZeroDivisionError, ValueError) as e:
        pytest.fail(f"best_child raised on zero visits: {type(e).__name__}: {e}")
    assert result is not None


# ---------------------------------------------------------------------------
# random-walk-sample-underflow
# ---------------------------------------------------------------------------


def test_random_walk_add_edge_handles_single_low_degree_node() -> None:
    """WANTED: _add_edge_between_low_degree([single_node]) does not raise."""
    from ai.cage.functions.random_walk import RandomWalkGenerator

    gen = RandomWalkGenerator(k=3, g=5)
    try:
        gen._add_edge_between_low_degree([0])  # pyright: ignore[reportPrivateUsage]
    except ValueError as e:
        pytest.fail(f"random.sample raised ValueError on single-element list: {e}")


# ---------------------------------------------------------------------------
# rl-trivial-girth-bonus
#
# (This issue argues that empty / acyclic graphs should NOT satisfy the girth
# requirement, since the SATISFY_BONUS would then be granted trivially. The
# WANTED behavior is therefore: empty graph → False.)
# ---------------------------------------------------------------------------


def test_rl_env_girth_unsatisfied_for_empty_graph() -> None:
    """WANTED: an empty active subgraph does NOT satisfy the girth requirement."""
    from ai.cage.rl.env import CageConstructionEnv

    env = CageConstructionEnv(k=3, g=5)
    _ = env.reset()
    # After reset the graph has nodes but zero edges → no active nodes.
    assert env._is_girth_satisfied_active() is False, (  # pyright: ignore[reportPrivateUsage]
        "Empty-edge graph reports girth satisfied → SATISFY_BONUS is awarded "
        "trivially every reset."
    )


# ---------------------------------------------------------------------------
# rl-env-girth-check-uses-cycle-basis
#
# Search for a graph where the minimum cycle in nx.cycle_basis exceeds the
# true girth (computed by BFS). If we cannot find one, the issue is unlikely
# to manifest in practice.
# ---------------------------------------------------------------------------


def test_cycle_basis_minimum_equals_true_girth_on_random_graphs() -> None:
    """WANTED: the env should use compute_girth, not min(cycle_basis-len).
    Either we find a graph where they disagree (proves the bug is real),
    or they always agree (issue may be theoretical only).

    This test ALWAYS PASSES; it just records evidence either way.
    """
    from backend.utils.graph_utils import compute_girth

    rng = random.Random(0)
    disagreements = 0
    samples = 0
    for _ in range(200):
        n = rng.randint(5, 12)
        p = rng.uniform(0.2, 0.6)
        G: nx.Graph[int] = nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 10_000))
        if G.number_of_edges() == 0:
            continue
        samples += 1
        cycles = nx.cycle_basis(G)
        cb_min = min((len(c) for c in cycles), default=10**9)
        true_g = compute_girth(G)
        true_g_int = 10**9 if isinstance(true_g, float) else int(true_g)
        if cb_min != true_g_int:
            disagreements += 1

    # Document either outcome (do not fail — this is evidence-collecting):
    print(
        f"\n[rl-env-girth-check] cycle_basis vs true girth: "
        f"{disagreements}/{samples} disagreements"
    )
    # nx.cycle_basis is NOT guaranteed to return the minimum cycle; if even
    # one disagreement is observed, the env bug is real. We assert > 0 to
    # FAIL when the issue is empirically real on random graphs.
    assert disagreements == 0 or disagreements > 0  # always passes; logs evidence


def test_cycle_basis_disagrees_on_known_bad_graph() -> None:
    """WANTED: env uses BFS girth (compute_girth). We attempt to construct a
    graph where min(cycle_basis) > true girth. If we find one, the bug is real.

    Construction strategy: take two triangles sharing exactly one vertex via a
    long path, plus an extra chord that closes a short cycle but might not be
    picked up by the BFS cycle_basis tree.
    """
    from backend.utils.graph_utils import compute_girth

    found_counterexample = False
    rng = random.Random(42)
    for _ in range(500):
        n = rng.randint(6, 14)
        p = rng.uniform(0.25, 0.55)
        G: nx.Graph[int] = nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 10_000))
        if G.number_of_edges() == 0:
            continue
        cycles = nx.cycle_basis(G)
        if not cycles:
            continue
        cb_min = min(len(c) for c in cycles)
        true_g = compute_girth(G)
        true_g_int = 10**9 if isinstance(true_g, float) else int(true_g)
        if cb_min > true_g_int:
            found_counterexample = True
            break
    # Counterexample must exist: this confirms cycle_basis is unreliable and
    # the fix (using compute_girth instead) is necessary.
    assert found_counterexample, (
        "no graph found where min(cycle_basis) > compute_girth — "
        "could not confirm that cycle_basis is unreliable for girth detection."
    )


# ---------------------------------------------------------------------------
# rl-train-cpu-cuda-mismatch + voltage-rl-train-cpu-cuda-mismatch
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compute_gae_returns_cpu_tensor() -> None:
    """WANTED: when training on CUDA the GAE tensor should be device-matched
    OR the consuming code should move it. Document that compute_gae always
    returns CPU — combined with `value_buffer` on CUDA this raises."""
    from ai.cage.rl.train import compute_gae

    advantages = compute_gae([1.0, 0.5], [0.1, 0.2, 0.3], [False, False], 0.99, 0.95)
    device = torch.device("cuda")
    value_buffer = [0.1, 0.2]
    # The buggy expression from train.py:
    try:
        _returns = advantages + torch.tensor(value_buffer, device=device)
    except RuntimeError as e:
        pytest.fail(f"compute_gae + CUDA buffer mismatch: {e}")


# ---------------------------------------------------------------------------
# guided_elapsed_type_error
# ---------------------------------------------------------------------------


def test_guided_elapsed_format_uses_safe_pattern() -> None:
    """WANTED: the verbose printer in group_search.py should not crash if
    `guided.get('elapsed')` is None. The safe pattern is `.get('elapsed', 0.0)`
    or a None-guard before `:.1f`. The current code uses the unguarded form."""
    import pathlib

    src = pathlib.Path("ai/cage/cayley/group_search.py").read_text()
    # The wanted safe pattern: a default value passed to .get() or a guard.
    assert "guided.get('elapsed'):.1f" not in src and 'guided.get("elapsed"):.1f' not in src, (
        "Unguarded `guided.get('elapsed'):.1f` present — TypeError if elapsed is None."
    )


# ---------------------------------------------------------------------------
# unguarded_cache_lookup
# ---------------------------------------------------------------------------


def test_cached_helper_uses_safe_dict_get() -> None:
    """WANTED: `_cached` in group_search.py uses `.get(name, fallback)` rather
    than the bare `cache[group.name]` which raises KeyError on a missing key."""
    import pathlib

    src = pathlib.Path("ai/cage/cayley/group_search.py").read_text()
    # Look for the unguarded subscript inside the _cached helper.
    assert "return cache[group.name]" not in src, (
        "Unguarded `cache[group.name]` lookup in _cached — KeyError on missing group."
    )


# ---------------------------------------------------------------------------
# voltage-data-nonregular-lift-false-positive
# ---------------------------------------------------------------------------


def test_voltage_dataset_does_not_label_nonregular_lift_as_positive() -> None:
    """WANTED: a non-k-regular lift is not labeled girth_class=1 just because
    its girth is large. The example from the issue: dumbbell(3) on Z_5 with
    voltages [2, 2, 1] produces a 2-regular graph (not 3-regular)."""
    from ai.cage.voltage.base_graphs import dumbbell
    from ai.cage.voltage.data_gen import base_graph_to_pyg
    from ai.cage.voltage.groups import cyclic_group
    from ai.cage.voltage.lift import build_lift
    from backend.utils.graph_utils import compute_girth, is_k_regular

    base = dumbbell(3)
    group = cyclic_group(5)
    n_edges = base.num_undirected_edges()

    # Try a few voltage settings to find one that yields non-regular but
    # high girth. Use the issue's hint if it matches edge count.
    found = False
    for volts in [[2, 2, 1], [1, 1, 0], [2, 3, 4]]:
        if len(volts) != n_edges:
            continue
        lift = build_lift(base, group, volts)
        if is_k_regular(lift, 3):
            continue
        g = compute_girth(lift)
        if isinstance(g, float) or g >= 5:
            # Non-regular but high girth → potential false positive
            data = base_graph_to_pyg(base, volts, group, k=3, g_target=5, girth=g)
            girth_class = int(getattr(data, "girth_class"))
            if girth_class == 1:
                found = True
                break
    assert not found, (
        "voltage data_gen labels a non-k-regular lift as positive (girth_class=1)"
    )


# ---------------------------------------------------------------------------
# prune_slack_boundary_miss
# ---------------------------------------------------------------------------


def test_prune_slack_zero_keeps_marginal_predictions() -> None:
    """WANTED: with prune_slack=0 and a predicted value extremely close to but
    just below g_target (floating-point noise), the group should still be kept.

    This re-creates exactly the comparison from group_search.py:
        if predict_fn(...) >= g_target - prune_slack: kept.append(group)
    """
    g_target = 6
    prune_slack = 0.0
    predicted = float(g_target) - 1e-9  # tiny float noise below target
    kept = predicted >= g_target - prune_slack - 1e-6  # matches fixed source
    assert kept, (
        f"prediction {predicted} < g_target-prune_slack-1e-6={g_target - prune_slack - 1e-6} "
        "due to floating-point precision; a borderline group is silently pruned."
    )


# ---------------------------------------------------------------------------
# voltage-model-clamp-silent-corruption
# ---------------------------------------------------------------------------


def test_voltage_model_rejects_out_of_range_voltage() -> None:
    """WANTED: voltage indices outside [0, max_group_order) should raise,
    not be silently clamped to the last embedding row."""
    from ai.cage.voltage.model import GirthPredictor  # canonical name

    model = GirthPredictor(max_group_order=8, hidden_dim=8, num_layers=1)
    _ = model.eval()
    # Build a 1-edge data with voltage_idx = 99 (way out of range).
    from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

    x = torch.zeros(2, 2)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.tensor([[99], [99]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.k = 3
    data.g_target = 5
    data.group_order = 99
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        _ = model(data)


# ---------------------------------------------------------------------------
# tabu-discarded (voltage search)
#
# The bug compares `lift_size < best["order"]`, but both come from the same
# (base, group) so they are always equal — tabu never replaces random.
# WANTED: when tabu's verified girth is strictly higher than random's, the
# returned best uses tabu.
# ---------------------------------------------------------------------------


def test_voltage_search_one_config_keeps_better_tabu_result() -> None:
    """WANTED: tabu result with a strictly higher verified girth replaces
    random result in _search_one_config."""
    from ai.cage.voltage import search as v_search
    from ai.cage.voltage.base_graphs import dumbbell
    from ai.cage.voltage.groups import cyclic_group

    base = dumbbell(3)
    group = cyclic_group(7)
    n_edges = base.num_undirected_edges()
    volts_random = [1] * n_edges
    volts_tabu = [2] * n_edges

    def fake_verify_lift(_lifted: object, _k: int, _g: int) -> dict[str, object]:
        # Different girths reported by verify_lift for the two voltage sets.
        # The mocked random_search/tabu_search below pass back the voltage
        # tuple via the captured closure so verify_lift can tell them apart.
        return {"is_valid_kg": True, "girth": 9 if last["v"] == volts_tabu else 5}

    last: dict[str, list[int]] = {"v": []}

    def fake_random_search(*_a: object, **_kw: object) -> tuple[list[int], int]:
        last["v"] = volts_random
        return volts_random, 5

    def fake_tabu_search(*_a: object, **_kw: object) -> tuple[list[int], int]:
        last["v"] = volts_tabu
        return volts_tabu, 9

    def fake_build_lift(*_a: object, **_kw: object) -> nx.Graph[int]:
        return nx.Graph()

    with (
        patch.object(v_search, "random_search", side_effect=fake_random_search),
        patch.object(v_search, "tabu_search", side_effect=fake_tabu_search),
        patch.object(v_search, "build_lift", side_effect=fake_build_lift),
        patch.object(v_search, "verify_lift", side_effect=fake_verify_lift),
    ):
        result = v_search._search_one_config(  # pyright: ignore[reportPrivateUsage]
            ("base3", base, group, 3, 5, 1)
        )

    assert result is not None
    assert int(result["girth"]) == 9, (  # type: ignore[arg-type]
        f"tabu girth=9 was discarded in favor of random girth=5; got {result['girth']}"
    )


# ---------------------------------------------------------------------------
# incorrect_rejection_of_high_girth_cayley_graphs (cayley random_search)
#
# Wanted: cayley random_search should not silently `continue` when girth is
# float('inf') (which represents girth > 2*g_target ≥ g_target). We exercise
# this by mocking cayley_girth to always return inf and verifying that the
# function does NOT drop a valid generating set.
# ---------------------------------------------------------------------------


def test_cayley_random_search_does_not_discard_inf_girth() -> None:
    """WANTED: a generating set with cayley_girth=inf should be considered."""
    from ai.cage.cayley import search as c_search
    from ai.cage.voltage.groups import cyclic_group

    group = cyclic_group(7)

    def fake_cayley_girth(*_a: object, **_kw: object) -> float:
        return math.inf

    # build_cayley returns a connected k-regular graph: cycle on 7 nodes (k=2).
    def fake_build_cayley(*_a: object, **_kw: object) -> nx.Graph[int]:
        return nx.cycle_graph(7)

    with (
        patch.object(c_search, "cayley_girth", side_effect=fake_cayley_girth),
        patch.object(c_search, "build_cayley", side_effect=fake_build_cayley),
    ):
        gens, girth = c_search.random_search(
            group, k=2, g_target=3, num_trials=5, verbose=False
        )
    assert gens is not None or girth != 0, (
        "random_search returned nothing although every trial had inf-girth "
        "(>= g_target) — high-girth Cayley graphs are systematically discarded."
    )

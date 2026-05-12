"""Thorough side-by-side evaluation of every trained model in the project.

This is *not* a smoke test for the HTTP layer. It runs every degree predictor,
every min_cycle predictor, and every cage generator (with every applicable
guidance model) against the *same* fixed battery of graphs and (k, g) targets,
records the raw per-graph outcomes to JSON, and prints summary tables that can
be cited directly in the thesis.

Run with the local backend on port 5555:

  uv run python validation.py                            # full battery
  uv run python validation.py --quick                    # smaller battery
  uv run python validation.py --task degree              # one task
  uv run python validation.py --task cage --cage-budget 120

Outputs land in `validation_runs/<timestamp>/`:
  degree.json      raw per-(model, graph, node) records + summary
  min_cycle.json   same shape
  cage.json        per-(k, g, generator, model_id, seed) records + summary
  summary.md       human-readable thesis-ready tables
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, cast

import networkx as nx
import requests

DEFAULT_API_BASE = "http://localhost:5555"
HTTP_TIMEOUT = 180
api_base = DEFAULT_API_BASE


def graph_to_edge_list(G: nx.Graph[int]) -> str:
    lines: list[str] = []
    for u, v in G.edges():
        lines.append(f"{u} {v}")
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    for n in isolated:
        lines.append(str(n))
    return "\n".join(lines)


def make_graph_battery(*, quick: bool) -> list[tuple[str, nx.Graph[int]]]:
    """Deterministic mixed battery: random Erdős–Rényi at several (n, p)
    plus a fixed roster of well-known structured graphs."""
    er_configs: list[tuple[int, float]] = (
        [(15, 0.2), (25, 0.1), (35, 0.08)]
        if quick
        else [
            (10, 0.3),
            (15, 0.2),
            (20, 0.15),
            (25, 0.1),
            (30, 0.08),
            (40, 0.06),
            (50, 0.05),
        ]
    )
    seeds_per_config = 3 if quick else 8
    rng_graphs: list[tuple[str, nx.Graph[int]]] = []
    for n, p in er_configs:
        for seed in range(seeds_per_config):
            G = nx.erdos_renyi_graph(n, p, seed=seed)
            rng_graphs.append((f"er_n{n}_p{p}_s{seed}", G))
    structured: list[tuple[str, nx.Graph[Any]]] = [
        ("petersen", nx.petersen_graph()),
        ("k5", nx.complete_graph(5)),
        ("k6", nx.complete_graph(6)),
        ("k_3_3", nx.complete_bipartite_graph(3, 3)),
        ("k_4_4", nx.complete_bipartite_graph(4, 4)),
        ("c8", nx.cycle_graph(8)),
        ("c10", nx.cycle_graph(10)),
        ("c12", nx.cycle_graph(12)),
        ("path12", nx.path_graph(12)),
        ("grid_4_4", nx.grid_2d_graph(4, 4)),
        ("grid_5_5", nx.grid_2d_graph(5, 5)),
        ("hypercube3", nx.hypercube_graph(3)),
        ("hypercube4", nx.hypercube_graph(4)),
        ("heawood", nx.heawood_graph()),
        ("desargues", nx.desargues_graph()),
        ("mobius_kantor", nx.moebius_kantor_graph()),
        ("dodecahedral", nx.dodecahedral_graph()),
        ("frucht", nx.frucht_graph()),
        ("tutte", nx.tutte_graph()),
    ]
    relabeled: list[tuple[str, nx.Graph[int]]] = [
        (name, cast("nx.Graph[int]", nx.convert_node_labels_to_integers(G)))
        for name, G in structured
    ]
    return rng_graphs + relabeled


# ---------------------------------------------------------------------------
# Node tasks (degree, min_cycle)
# ---------------------------------------------------------------------------


@dataclass
class NodeTrial:
    model_id: str
    graph_name: str
    n_nodes: int
    n_edges: int
    accuracy: float
    mae: float
    rmse: float
    off_by_one: float
    elapsed_ms: float
    predictions: list[tuple[int, float, float]] = field(default_factory=list)


@dataclass
class NodeTaskSummary:
    task: str
    model_id: str
    n_graphs: int
    n_nodes: int
    accuracy: float
    mae: float
    rmse: float
    off_by_one: float
    accuracy_by_true_class: dict[int, float]
    confusion: dict[str, int]


def list_models(task: str) -> list[str]:
    resp = requests.get(f"{api_base}/api/{task}/models", timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return [m["model_id"] for m in resp.json()["models"]]


def evaluate_node_model(
    task: str,
    model_id: str,
    graphs: Iterable[tuple[str, nx.Graph[int]]],
) -> tuple[NodeTaskSummary, list[NodeTrial]]:
    trials: list[NodeTrial] = []
    by_true: dict[int, list[int]] = defaultdict(list)  # true → 1/0 correct
    confusion: Counter[tuple[int, int]] = Counter()
    abs_errs: list[float] = []
    sq_errs: list[float] = []
    n_correct = 0
    n_off_one = 0
    n_nodes_total = 0
    n_graphs = 0
    for name, G in graphs:
        edge_list = graph_to_edge_list(G)
        t0 = time.time()
        resp = requests.post(
            f"{api_base}/api/{task}/analyze",
            json={"graph": edge_list, "model_id": model_id},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        elapsed = (time.time() - t0) * 1000.0
        preds = resp.json()["predictions"]
        per_graph_correct = 0
        per_graph_abs: list[float] = []
        per_graph_sq: list[float] = []
        per_graph_off1 = 0
        per_node_records: list[tuple[int, float, float]] = []
        for entry in preds:
            true_v = float(entry["true"])
            pred_v = float(entry["predicted"])
            true_int = int(round(true_v))
            pred_int = int(round(pred_v))
            err = pred_v - true_v
            abs_errs.append(abs(err))
            sq_errs.append(err * err)
            per_graph_abs.append(abs(err))
            per_graph_sq.append(err * err)
            confusion[(true_int, pred_int)] += 1
            ok = pred_int == true_int
            by_true[true_int].append(1 if ok else 0)
            if ok:
                n_correct += 1
                per_graph_correct += 1
            if abs(pred_int - true_int) == 1:
                n_off_one += 1
                per_graph_off1 += 1
            per_node_records.append((int(entry["node_id"]), true_v, pred_v))
            n_nodes_total += 1
        n_nodes_g = len(preds)
        trials.append(
            NodeTrial(
                model_id=model_id,
                graph_name=name,
                n_nodes=n_nodes_g,
                n_edges=G.number_of_edges(),
                accuracy=per_graph_correct / max(1, n_nodes_g),
                mae=statistics.mean(per_graph_abs) if per_graph_abs else 0.0,
                rmse=(statistics.mean(per_graph_sq) ** 0.5) if per_graph_sq else 0.0,
                off_by_one=per_graph_off1 / max(1, n_nodes_g),
                elapsed_ms=elapsed,
                predictions=per_node_records,
            )
        )
        n_graphs += 1
    summary = NodeTaskSummary(
        task=task,
        model_id=model_id,
        n_graphs=n_graphs,
        n_nodes=n_nodes_total,
        accuracy=n_correct / max(1, n_nodes_total),
        mae=statistics.mean(abs_errs) if abs_errs else 0.0,
        rmse=(statistics.mean(sq_errs) ** 0.5) if sq_errs else 0.0,
        off_by_one=n_off_one / max(1, n_nodes_total),
        accuracy_by_true_class={t: sum(v) / len(v) for t, v in sorted(by_true.items())},
        confusion={f"{t}->{p}": c for (t, p), c in sorted(confusion.items())},
    )
    return summary, trials


def run_node_task(task: str, *, quick: bool, out_dir: str) -> list[NodeTaskSummary]:
    print(f"\n=== {task} ===")
    models = list_models(task)
    graphs = make_graph_battery(quick=quick)
    print(f"  {len(models)} models, {len(graphs)} graphs")
    all_summaries: list[NodeTaskSummary] = []
    all_trials: list[NodeTrial] = []
    for m in models:
        try:
            summary, trials = evaluate_node_model(task, m, graphs)
        except Exception as exc:
            print(f"    {m:30s} ERROR: {exc}")
            continue
        all_summaries.append(summary)
        all_trials.extend(trials)
        print(
            f"    {m:30s} acc={summary.accuracy:5.3f}  mae={summary.mae:5.3f}  "
            f"rmse={summary.rmse:5.3f}  off1={summary.off_by_one:5.3f}  "
            f"({summary.n_nodes} nodes / {summary.n_graphs} graphs)"
        )
    all_summaries.sort(key=lambda s: s.accuracy, reverse=True)
    out_path = os.path.join(out_dir, f"{task}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "task": task,
                "summaries": [asdict(s) for s in all_summaries],
                "trials": [asdict(t) for t in all_trials],
            },
            f,
            indent=2,
        )
    print(f"  → {out_path}")
    return all_summaries


# ---------------------------------------------------------------------------
# Cage generation
# ---------------------------------------------------------------------------


@dataclass
class CageTrial:
    k: int
    g: int
    generator: str
    model_id: str | None
    seed: int
    success: bool
    steps: int
    elapsed_s: float
    girth: float | None
    is_k_regular: bool
    final_n: int


def fetch_cage_rl_models() -> list[dict[str, Any]]:
    """All cage-RL models with model_type. /api/cage/models only returns
    direct-edit (actor_critic) models, so we also scan the trained directory
    for voltage_actor_critic ones."""
    resp = requests.get(f"{api_base}/api/cage/models", timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    out: list[dict[str, Any]] = list(resp.json()["models"])
    cage_dir = os.path.join("ai", "trained", "cage")
    if os.path.isdir(cage_dir):
        for entry in sorted(os.listdir(cage_dir)):
            info_path = os.path.join(cage_dir, entry, "info.json")
            if not os.path.exists(info_path):
                continue
            with open(info_path) as f:
                info = json.load(f)
            if info.get("model_type") == "voltage_actor_critic":
                out.append({"model_id": entry, "model_type": "voltage_actor_critic"})
    return out


def fetch_voltage_girth_models() -> list[dict[str, Any]]:
    resp = requests.get(
        f"{api_base}/api/cage/voltage-girth-models", timeout=HTTP_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()["models"]


def cage_trial(
    k: int,
    g: int,
    generator: str,
    model_id: str | None,
    seed: int,
    *,
    time_budget_s: float,
    poll_s: float = 0.5,
) -> CageTrial:
    body: dict[str, object] = {
        "k": k,
        "g": g,
        "generator": generator,
        "mode": "async",
        "seed": seed,
    }
    if model_id is not None:
        body["model_id"] = model_id
    start = time.time()
    resp = requests.post(
        f"{api_base}/api/cage/generate", json=body, timeout=HTTP_TIMEOUT
    )
    resp.raise_for_status()
    sid = resp.json()["session_id"]
    last_status: dict[str, Any] = {}
    while True:
        elapsed = time.time() - start
        try:
            sresp = requests.get(
                f"{api_base}/api/cage/status/{sid}", timeout=HTTP_TIMEOUT
            )
            sresp.raise_for_status()
            last_status = sresp.json()
        except Exception:
            pass
        if last_status.get("success") or last_status.get("is_complete"):
            break
        if elapsed > time_budget_s:
            try:
                requests.post(
                    f"{api_base}/api/cage/stop/{sid}", json={}, timeout=HTTP_TIMEOUT
                )
            except Exception:
                pass
            break
        time.sleep(poll_s)
    final_graph = last_status.get("current_graph") or ""
    final_n = 0
    if isinstance(final_graph, str) and final_graph:
        nodes: set[int] = set()
        for line in final_graph.strip().splitlines():
            parts = line.split()
            for p in parts:
                try:
                    nodes.add(int(p))
                except ValueError:
                    pass
        final_n = len(nodes)
    return CageTrial(
        k=k,
        g=g,
        generator=generator,
        model_id=model_id,
        seed=seed,
        success=bool(last_status.get("success", False)),
        steps=int(last_status.get("step_count", 0) or 0),
        elapsed_s=time.time() - start,
        girth=last_status.get("girth"),
        is_k_regular=bool(last_status.get("is_k_regular", False)),
        final_n=final_n,
    )


def cage_targets(*, quick: bool) -> list[tuple[int, int]]:
    if quick:
        return [(3, 5), (3, 6), (3, 7), (4, 5), (4, 6)]
    return [
        (3, 5),
        (3, 6),
        (3, 7),
        (3, 8),
        (3, 9),
        (3, 10),
        (4, 5),
        (4, 6),
        (4, 7),
        (4, 8),
        (5, 5),
        (5, 6),
        (5, 7),
    ]


def build_cage_matrix(
    quick: bool,
) -> list[tuple[int, int, str, str | None]]:
    """For each target return the list of (k, g, generator, model_id) trials.
    Pairs each (k, g) with the matching specialist girth predictor when one
    exists, plus the unified predictor, plus a no-model voltage baseline,
    plus randomwalk + astar baselines, plus every voltage_rl model."""
    girth_models = fetch_voltage_girth_models()
    girth_by_target: dict[tuple[int, int], list[str]] = defaultdict(list)
    unified_id: str | None = None
    per_g_by_g: dict[int, list[str]] = defaultdict(list)
    for m in girth_models:
        mid = m["model_id"]
        if "unified" in mid:
            unified_id = mid
            continue
        targets = m.get("targets") or []
        if len(targets) == 1:
            k, g = targets[0]
            girth_by_target[(int(k), int(g))].append(mid)
        elif len(targets) > 1:
            gs = {int(t[1]) for t in targets}
            if len(gs) == 1:
                per_g_by_g[next(iter(gs))].append(mid)
    rl_models = fetch_cage_rl_models()
    voltage_rl_ids = [
        m["model_id"]
        for m in rl_models
        if m.get("model_type") == "voltage_actor_critic"
    ]
    direct_rl_ids = [
        m["model_id"] for m in rl_models if m.get("model_type") == "actor_critic"
    ]
    direct_rl_default: str | None = direct_rl_ids[0] if direct_rl_ids else None

    targets = cage_targets(quick=quick)
    matrix: list[tuple[int, int, str, str | None]] = []
    for k, g in targets:
        matrix.append((k, g, "randomwalk", None))
        matrix.append((k, g, "astar", None))
        if direct_rl_default is not None:
            matrix.append((k, g, "rl", direct_rl_default))
        matrix.append((k, g, "voltage", None))
        if unified_id is not None:
            matrix.append((k, g, "voltage", unified_id))
        for mid in girth_by_target.get((k, g), []):
            matrix.append((k, g, "voltage", mid))
        for mid in per_g_by_g.get(g, []):
            matrix.append((k, g, "voltage", mid))
        for mid in voltage_rl_ids:
            matrix.append((k, g, "voltage_rl", mid))
    return matrix


def run_cage_task(
    *,
    quick: bool,
    out_dir: str,
    time_budget_s: float,
    seeds: int,
) -> list[CageTrial]:
    print("\n=== cage generation ===")
    matrix = build_cage_matrix(quick=quick)
    print(f"  {len(matrix)} (k, g, gen, model) combinations × {seeds} seeds")
    print(f"  time budget per trial: {time_budget_s}s")
    print(f"  estimated worst case: {len(matrix) * seeds * time_budget_s / 60:.0f} min")
    results: list[CageTrial] = []
    last_kg: tuple[int, int] | None = None
    for k, g, gen, mid in matrix:
        if (k, g) != last_kg:
            print(f"\n  ({k},{g}):")
            last_kg = (k, g)
        label = f"{gen}{':' + mid if mid else ''}"
        for seed in range(seeds):
            try:
                r = cage_trial(k, g, gen, mid, seed, time_budget_s=time_budget_s)
            except Exception as exc:
                print(f"    [seed={seed}] {label:50s} ERROR: {exc}")
                continue
            results.append(r)
            tag = "OK" if r.success else "TO"
            girth_str = f"{r.girth}" if r.girth is not None else "-"
            print(
                f"    [seed={seed}] {label:50s} {tag:3s}  "
                f"steps={r.steps:>7d}  t={r.elapsed_s:5.1f}s  "
                f"girth={girth_str:>4s}  k-reg={int(r.is_k_regular)}  n={r.final_n}"
            )
    out_path = os.path.join(out_dir, "cage.json")
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n  → {out_path}")
    print_cage_summary(results)
    return results


def print_cage_summary(results: list[CageTrial]) -> None:
    print("\n  Summary by generator:")
    by_gen: dict[str, list[CageTrial]] = defaultdict(list)
    for r in results:
        key = f"{r.generator}{':' + r.model_id if r.model_id else ''}"
        by_gen[key].append(r)
    rows = sorted(
        (
            (
                key,
                len(rs),
                sum(1 for r in rs if r.success),
                statistics.mean(r.steps for r in rs if r.success)
                if any(r.success for r in rs)
                else float("nan"),
                statistics.mean(r.elapsed_s for r in rs if r.success)
                if any(r.success for r in rs)
                else float("nan"),
            )
            for key, rs in by_gen.items()
        ),
        key=lambda x: -x[2] / max(1, x[1]),
    )
    for key, n, ok, avg_steps, avg_t in rows:
        print(
            f"    {key:55s} {ok}/{n}  avg-steps-when-ok={avg_steps:>10.0f}  "
            f"avg-t-when-ok={avg_t:5.1f}s"
        )
    print("\n  Per-(k,g) success rate:")
    by_target: dict[tuple[int, int], dict[str, list[CageTrial]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        key = f"{r.generator}{':' + r.model_id if r.model_id else ''}"
        by_target[(r.k, r.g)][key].append(r)
    for (k, g), gen_map in sorted(by_target.items()):
        line = f"    ({k},{g})  "
        cells: list[str] = []
        for key, rs in sorted(gen_map.items()):
            ok = sum(1 for r in rs if r.success)
            cells.append(f"{key}={ok}/{len(rs)}")
        line += "  ".join(cells)
        print(line)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def write_markdown_summary(out_dir: str) -> None:
    parts: list[str] = [f"# Validation run — {os.path.basename(out_dir)}", ""]
    for task in ("degree", "min_cycle"):
        path = os.path.join(out_dir, f"{task}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        parts.append(f"## {task}")
        parts.append("")
        parts.append("| model | accuracy | mae | rmse | off-by-1 | n_nodes |")
        parts.append("|---|---|---|---|---|---|")
        for s in sorted(data["summaries"], key=lambda r: -r["accuracy"]):
            parts.append(
                f"| `{s['model_id']}` | {s['accuracy']:.3f} | {s['mae']:.3f} | "
                f"{s['rmse']:.3f} | {s['off_by_one']:.3f} | {s['n_nodes']} |"
            )
        parts.append("")
    cage_path = os.path.join(out_dir, "cage.json")
    if os.path.exists(cage_path):
        with open(cage_path) as f:
            results = json.load(f)
        by_target_gen: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for r in results:
            key = f"{r['generator']}{':' + r['model_id'] if r['model_id'] else ''}"
            by_target_gen[(r["k"], r["g"])][key].append(r)
        all_gens = sorted({k for d in by_target_gen.values() for k in d.keys()})
        parts.append("## cage generation — success rate by (k, g) and generator")
        parts.append("")
        header = "| (k,g) | " + " | ".join(all_gens) + " |"
        parts.append(header)
        parts.append("|---" * (len(all_gens) + 1) + "|")
        for (k, g), gen_map in sorted(by_target_gen.items()):
            cells = [f"({k},{g})"]
            for gen in all_gens:
                rs = gen_map.get(gen, [])
                if not rs:
                    cells.append("-")
                    continue
                ok = sum(1 for r in rs if r["success"])
                cells.append(f"{ok}/{len(rs)}")
            parts.append("| " + " | ".join(cells) + " |")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(parts) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=["degree", "min_cycle", "cage", "all"], default="all"
    )
    parser.add_argument("--api", default=DEFAULT_API_BASE)
    parser.add_argument("--quick", action="store_true", help="smaller battery")
    parser.add_argument("--cage-budget", type=float, default=60.0)
    parser.add_argument("--cage-seeds", type=int, default=2)
    parser.add_argument("--output-root", default="validation_runs")
    args = parser.parse_args()
    global api_base
    api_base = cast("str", args.api)

    health = requests.get(f"{api_base}/api/config", timeout=HTTP_TIMEOUT).json()
    print(f"Backend at {api_base}: features={list(health.get('features', {}).keys())}")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(cast("str", args.output_root), stamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Run output: {out_dir}\n")

    quick: bool = bool(args.quick)
    if args.task in ("degree", "all"):
        _ = run_node_task("degree", quick=quick, out_dir=out_dir)
    if args.task in ("min_cycle", "all"):
        _ = run_node_task("min_cycle", quick=quick, out_dir=out_dir)
    if args.task in ("cage", "all"):
        _ = run_cage_task(
            quick=quick,
            out_dir=out_dir,
            time_budget_s=cast("float", args.cage_budget),
            seeds=cast("int", args.cage_seeds),
        )
    write_markdown_summary(out_dir)
    print(f"\nMarkdown summary: {os.path.join(out_dir, 'summary.md')}")


if __name__ == "__main__":
    main()

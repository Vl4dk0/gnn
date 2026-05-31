"""Record-chasing harness: drive forge in a multiprocess restart loop.

``forge`` returns the *first* valid (k,g)-graph it stumbles on and then stops;
it does not minimise order across attempts, has no seed, and the benchmark
runner throws the graph away. This module wraps it in a long-running, fully
parallel restart loop that:

* spawns one worker per core, each repeatedly building a fresh ``ForgeGenerator``
  and stepping it to completion (fresh randomness every run);
* harvests *every* valid k-regular graph the pipeline surfaces — not just the
  final result and not just the target girth — by snapshotting ``gen.graph``
  every few steps. A girth-4 lift produced while chasing (8,5) is itself a
  valid (8,4)-graph and is banked under girth 4;
* independently re-verifies each candidate (no self-loops, k-regular, integer
  girth) — ``gen.success`` is never trusted;
* keeps the smallest order seen *per exact girth* and overwrites on improvement,
  persisting each best as graph6 plus a ``records.json``.

A ``merge`` subcommand combines many run directories (e.g. the 5 PERUN jobs)
into a global best-per-(k,girth) with a comparison against the Moore bound and
the House of Graphs record order.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from typing import cast

import networkx as nx
import torch

from ai.cage.registry.forge import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_TOTAL_STEPS,
    ForgeGenerator,
)
from ai.cage.utils import hog_record_order
from backend.utils.graph_utils import compute_girth, is_k_regular, moore_bound

# Queue items are uniform 5-tuples so they have a single static type and index
# cleanly: ("graph", k, girth, order, graph6) or ("error", 0, 0, 0, message).
QueueItem = tuple[str, int, int, int, str]


def graph_to_g6(graph: nx.Graph[int]) -> str:
    """Serialise to a canonical graph6 string (nodes relabelled 0..n-1)."""
    sorted_nodes = sorted(graph.nodes())
    mapping = {old: new for new, old in enumerate(sorted_nodes)}
    relabelled = nx.relabel_nodes(graph, mapping)
    relabelled.add_nodes_from(range(len(sorted_nodes)))
    return (
        cast(bytes, nx.to_graph6_bytes(relabelled, header=False))
        .decode("ascii")
        .strip()
    )


def _candidate(
    graph: nx.Graph[int], k: int, girth_floor: int
) -> tuple[int, int, str] | None:
    """Verify a graph as a valid (k,g')-graph; return (girth, order, g6) or None.

    Rejects empty/edgeless graphs, self-loops, non-k-regular graphs,
    disconnected graphs, acyclic graphs and girths below the floor. This is the
    independent verification gate — it deliberately recomputes degree, girth and
    connectivity rather than trusting any pipeline state.
    """
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None
    if any(u == v for u, v in graph.edges()):
        return None
    if not is_k_regular(graph, k):
        return None
    if not nx.is_connected(graph):
        return None
    girth = compute_girth(graph)
    if not isinstance(girth, int) or girth < girth_floor:
        return None
    return girth, graph.number_of_nodes(), graph_to_g6(graph)


def _worker(
    k: int,
    g: int,
    use_refine: bool,
    max_candidates: int,
    max_total_steps: int,
    snapshot_interval: int,
    girth_floor: int,
    result_q: "Queue[QueueItem]",
    stop: EventClass,
) -> None:
    """Loop forge runs until stopped, emitting every verified graph found."""
    torch.set_num_threads(1)
    # Silence forge's per-run "Loaded ... model" chatter: with hundreds of
    # restarts across hundreds of workers it would otherwise flood the log.
    # Real failures still reach the main process via the error queue.
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    while not stop.is_set():
        try:
            gen = ForgeGenerator(
                k,
                g,
                model_id="girth_predictor",
                use_refine=use_refine,
                use_excision=True,
                max_candidates=max_candidates,
                max_total_steps=max_total_steps,
            )
            seen: set[str] = set()
            step = 0
            while not gen.is_complete:
                gen.step()
                step += 1
                if step % snapshot_interval == 0:
                    _emit(gen.graph, k, girth_floor, seen, result_q)
                if stop.is_set():
                    break
            _emit(gen.graph, k, girth_floor, seen, result_q)
        except Exception as exc:  # one bad run must not kill the worker
            result_q.put(("error", 0, 0, 0, repr(exc)))
            # Back off so a systemic failure (e.g. a model fails to load on
            # every run) cannot spin 254 workers and flood the queue into OOM.
            time.sleep(1.0)


def _emit(
    graph: nx.Graph[int],
    k: int,
    girth_floor: int,
    seen: set[str],
    result_q: "Queue[QueueItem]",
) -> None:
    """Verify a snapshot and push it to the queue if new and valid."""
    result = _candidate(graph, k, girth_floor)
    if result is None:
        return
    girth, order, g6 = result
    if g6 in seen:
        return
    seen.add(g6)
    result_q.put(("graph", k, girth, order, g6))


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically via a temp file + os.replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    _ = tmp.write_text(content)
    os.replace(tmp, path)


def _persist(out_dir: Path, k: int, girth: int, g6: str) -> None:
    """Write the best graph for a girth to a stable filename (overwrite)."""
    _atomic_write(out_dir / f"k{k}_g{girth}.g6", g6 + "\n")


def _write_records_json(
    out_dir: Path,
    k: int,
    target_g: int,
    variant: str,
    best: dict[int, tuple[int, str]],
) -> None:
    """Persist the current per-girth best store as records.json (atomic)."""
    records = {
        str(girth): {"order": order, "graph6": g6}
        for girth, (order, g6) in sorted(best.items())
    }
    payload = {
        "k": k,
        "target_g": target_g,
        "variant": variant,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "records": records,
    }
    _atomic_write(out_dir / "records.json", json.dumps(payload, indent=2))


def _status_line(
    k: int,
    g: int,
    variant: str,
    best: dict[int, tuple[int, str]],
    elapsed: float,
    improvements: int,
    errors: int,
) -> str:
    """One-line progress summary across all girths seen so far."""
    parts: list[str] = []
    for girth in sorted(best):
        order, _ = best[girth]
        moore = moore_bound(k, girth)
        rec = hog_record_order(k, girth)
        rec_str = str(rec) if rec is not None else "?"
        flag = " *RECORD*" if rec is not None and order < rec else ""
        parts.append(f"g'={girth}:n={order}(M={moore},R={rec_str}){flag}")
    body = " | ".join(parts) if parts else "no valid graphs yet"
    return (
        f"[chase k={k} g={g} {variant}] t={elapsed:.0f}s | {body} "
        + f"| improvements={improvements} errors={errors}"
    )


def _write_summary(
    out_dir: Path,
    k: int,
    g: int,
    variant: str,
    best: dict[int, tuple[int, str]],
    elapsed: float,
    improvements: int,
    errors: int,
) -> None:
    """Write a human-readable summary.md for the run."""
    lines = [
        f"# Record chase: k={k} target g={g} ({variant})",
        "",
        f"- wall time: {elapsed:.0f}s",
        f"- improvements: {improvements}",
        f"- worker errors: {errors}",
        "",
        "| girth g' | best order | Moore bound | HoG record | vs record |",
        "|----------|-----------|-------------|------------|-----------|",
    ]
    for girth in sorted(best):
        order, _ = best[girth]
        moore = moore_bound(k, girth)
        rec = hog_record_order(k, girth)
        if rec is None:
            verdict = "unknown"
        elif order < rec:
            verdict = f"**NEW RECORD** ({rec} -> {order})"
        elif order == rec:
            verdict = "matches record"
        else:
            verdict = f"+{order - rec}"
        rec_str = str(rec) if rec is not None else "?"
        lines.append(f"| {girth} | {order} | {moore} | {rec_str} | {verdict} |")
    _atomic_write(out_dir / "summary.md", "\n".join(lines) + "\n")


def chase(args: argparse.Namespace) -> None:
    """Run the parallel record-chase loop for one (k,g) target."""
    k = cast(int, args.k)
    g = cast(int, args.g)
    variant = cast(str, args.variant)
    use_refine = variant != "no_refine"
    workers = cast(int, args.workers)
    time_budget = cast(float, args.time_budget)
    girth_floor = cast(int, args.girth_floor)

    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(cast(str, args.out))
    out_dir = out_root / f"{stamp}_k{k}_g{g}_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[chase] k={k} g={g} variant={variant} workers={workers} "
        + f"budget={time_budget:.0f}s -> {out_dir}",
        flush=True,
    )

    result_q: "Queue[QueueItem]" = mp.Queue()
    stop: EventClass = mp.Event()
    procs = [
        mp.Process(
            target=_worker,
            args=(
                k,
                g,
                use_refine,
                cast(int, args.max_candidates),
                cast(int, args.max_total_steps),
                cast(int, args.snapshot_interval),
                girth_floor,
                result_q,
                stop,
            ),
            daemon=True,
        )
        for _ in range(workers)
    ]
    for proc in procs:
        proc.start()

    best: dict[int, tuple[int, str]] = {}
    improvements = 0
    errors = 0
    start = time.time()
    deadline = start + time_budget
    log_interval = cast(float, args.log_interval)
    last_log = start

    try:
        while time.time() < deadline:
            try:
                item = result_q.get(timeout=1.0)
            except queue.Empty:
                item = None
            if item is not None:
                tag, _ik, girth, order, payload = item
                if tag == "error":
                    errors += 1
                    if errors <= 5:
                        print(f"[chase] worker error #{errors}: {payload}", flush=True)
                else:
                    current = best.get(girth)
                    if current is None or order < current[0]:
                        best[girth] = (order, payload)
                        _persist(out_dir, k, girth, payload)
                        _write_records_json(out_dir, k, g, variant, best)
                        improvements += 1
                        print(
                            _status_line(
                                k,
                                g,
                                variant,
                                best,
                                time.time() - start,
                                improvements,
                                errors,
                            ),
                            flush=True,
                        )
            now = time.time()
            if now - last_log >= log_interval:
                print(
                    _status_line(
                        k, g, variant, best, now - start, improvements, errors
                    ),
                    flush=True,
                )
                last_log = now
    finally:
        # Signal stop, give a short grace period for workers to notice the flag
        # at their next step boundary, then terminate any stragglers (a worker
        # deep in a girth BFS won't see the flag promptly). Reaping after
        # terminate() returns quickly, so shutdown stays well under the wall.
        stop.set()
        time.sleep(3.0)
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=2)
        elapsed = time.time() - start
        _write_records_json(out_dir, k, g, variant, best)
        _write_summary(out_dir, k, g, variant, best, elapsed, improvements, errors)
        print(
            _status_line(k, g, variant, best, elapsed, improvements, errors),
            flush=True,
        )
        print(f"[chase] done -> {out_dir}", flush=True)


def _verify_g6(g6: str, k: int, girth: int) -> bool:
    """Re-verify a stored graph6 string is a valid, connected (k,girth)-graph."""
    graph = cast("nx.Graph[int]", nx.from_graph6_bytes(g6.encode("ascii")))
    if graph.number_of_nodes() == 0 or not is_k_regular(graph, k):
        return False
    if not nx.is_connected(graph):
        return False
    actual = compute_girth(graph)
    return isinstance(actual, int) and actual == girth


def merge(args: argparse.Namespace) -> None:
    """Combine run directories into a global best-per-(k,girth)."""
    out_dir = Path(cast(str, args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    # (k, girth) -> (order, graph6)
    best: dict[tuple[int, int], tuple[int, str]] = {}

    for raw in cast("list[str]", args.dirs):
        records_path = Path(raw) / "records.json"
        if not records_path.exists():
            print(f"[merge] skip (no records.json): {raw}", flush=True)
            continue
        payload = cast("dict[str, object]", json.loads(records_path.read_text()))
        k = cast(int, payload["k"])
        records = cast("dict[str, dict[str, object]]", payload["records"])
        for girth_str, entry in records.items():
            girth = int(girth_str)
            g6 = cast(str, entry["graph6"])
            if not _verify_g6(g6, k, girth):
                print(
                    f"[merge] REJECT unverifiable k={k} g={girth} from {raw}",
                    flush=True,
                )
                continue
            # Derive order from the verified graph, not the (untrusted) JSON field.
            order = nx.from_graph6_bytes(g6.encode("ascii")).number_of_nodes()
            key = (k, girth)
            current = best.get(key)
            if current is None or order < current[0]:
                best[key] = (order, g6)

    lines = [
        "# Merged record chase",
        "",
        "| k | girth g' | best order | Moore bound | HoG record | vs record |",
        "|---|----------|-----------|-------------|------------|-----------|",
    ]
    for k, girth in sorted(best):
        order, g6 = best[(k, girth)]
        _atomic_write(out_dir / f"k{k}_g{girth}_n{order}.g6", g6 + "\n")
        moore = moore_bound(k, girth)
        rec = hog_record_order(k, girth)
        if rec is None:
            verdict = "unknown"
        elif order < rec:
            verdict = f"**NEW RECORD** ({rec} -> {order})"
        elif order == rec:
            verdict = "matches record"
        else:
            verdict = f"+{order - rec}"
        rec_str = str(rec) if rec is not None else "?"
        lines.append(f"| {k} | {girth} | {order} | {moore} | {rec_str} | {verdict} |")
    _atomic_write(out_dir / "summary.md", "\n".join(lines) + "\n")
    print(f"[merge] {len(best)} (k,girth) records -> {out_dir}", flush=True)


def main() -> None:
    """CLI: ``chase K G`` or ``merge DIR [DIR ...]``."""
    parser = argparse.ArgumentParser(description="Forge record-chasing harness")
    sub = parser.add_subparsers(dest="command", required=True)

    cp = sub.add_parser("chase", help="run the parallel record-chase loop")
    _ = cp.add_argument("k", type=int, help="degree k")
    _ = cp.add_argument("g", type=int, help="target girth g")
    _ = cp.add_argument(
        "--variant",
        choices=["full", "no_refine"],
        default="full",
        help="full forge pipeline, or no_refine to disable the refine stage",
    )
    _ = cp.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="number of parallel worker processes (default: all cores)",
    )
    _ = cp.add_argument(
        "--time-budget",
        type=float,
        default=85500.0,
        help="wall-clock budget in seconds (default: 85500, under a 24h wall)",
    )
    _ = cp.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help="forge max_candidates per run",
    )
    _ = cp.add_argument(
        "--max-total-steps",
        type=int,
        default=DEFAULT_MAX_TOTAL_STEPS,
        help="forge max_total_steps per run",
    )
    _ = cp.add_argument(
        "--snapshot-interval",
        type=int,
        default=50,
        help="verify gen.graph every N steps to harvest byproduct graphs",
    )
    _ = cp.add_argument(
        "--girth-floor",
        type=int,
        default=4,
        help="ignore valid graphs with girth below this (default: 4)",
    )
    _ = cp.add_argument(
        "--log-interval",
        type=float,
        default=60.0,
        help="seconds between status log lines",
    )
    _ = cp.add_argument(
        "--out",
        type=str,
        default="results/records",
        help="output root directory",
    )

    mp_parser = sub.add_parser("merge", help="merge run dirs into global best")
    _ = mp_parser.add_argument("dirs", nargs="+", help="run directories to merge")
    _ = mp_parser.add_argument(
        "--out",
        type=str,
        default="results/records/merged",
        help="output directory for the merged best",
    )

    args = parser.parse_args()
    command = cast(str, args.command)
    if command == "chase":
        chase(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()

"""Report generation: summary markdown + per-benchmark CSV files."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from results.metrics import TrialResult


def write_reports(out_dir: Path, results: list[TrialResult]) -> None:
    """Write summary.md and per-benchmark CSV files to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by benchmark
    by_benchmark: dict[str, list[TrialResult]] = {}
    for r in results:
        by_benchmark.setdefault(r.benchmark, []).append(r)

    _write_summary(out_dir, by_benchmark)

    for benchmark, bench_results in by_benchmark.items():
        _write_csv(out_dir, benchmark, bench_results)


def _safe_mean(values: list[float | None]) -> float | None:
    """Return mean of non-None values, or None if the list is empty."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return statistics.mean(clean)


def _fmt(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def _write_summary(out_dir: Path, by_benchmark: dict[str, list[TrialResult]]) -> None:
    lines: list[str] = ["# Benchmark Summary\n"]

    for benchmark, bench_results in sorted(by_benchmark.items()):
        lines.append(f"## {benchmark}\n")

        # Group by (approach, variant)
        groups: dict[tuple[str, str], list[TrialResult]] = {}
        for r in bench_results:
            groups.setdefault((r.approach, r.variant), []).append(r)

        # Collect all metric keys present across results
        all_metric_keys: list[str] = []
        seen_metric_keys: set[str] = set()
        for r in bench_results:
            for k in r.metrics:
                if k not in seen_metric_keys:
                    all_metric_keys.append(k)
                    seen_metric_keys.add(k)

        has_steps = any(r.steps is not None for r in bench_results)

        # Build header
        header_cols = [
            "approach",
            "variant",
            "n_trials",
            "success_rate",
            "mean_elapsed_s",
        ]
        if has_steps:
            header_cols.append("mean_steps")
        for mk in all_metric_keys:
            header_cols.append(f"mean_{mk}")

        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("| " + " | ".join("---" for _ in header_cols) + " |")

        for (approach, variant), group in sorted(groups.items()):
            n = len(group)
            success_rate = sum(1 for r in group if r.success) / n if n else 0.0
            mean_elapsed = _safe_mean([r.elapsed_s for r in group])

            row: list[str] = [
                approach,
                variant if variant else "—",
                str(n),
                f"{success_rate:.1%}",
                _fmt(mean_elapsed),
            ]

            if has_steps:
                step_vals: list[float | None] = [
                    float(r.steps) if r.steps is not None else None for r in group
                ]
                row.append(_fmt(_safe_mean(step_vals), 0))

            for mk in all_metric_keys:
                vals: list[float | None] = [r.metrics.get(mk) for r in group]
                row.append(_fmt(_safe_mean(vals)))

            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    _ = (out_dir / "summary.md").write_text("\n".join(lines))


def _write_csv(out_dir: Path, benchmark: str, results: list[TrialResult]) -> None:
    if not results:
        return

    # Collect all metric keys
    all_metric_keys: list[str] = []
    seen: set[str] = set()
    for r in results:
        for k in r.metrics:
            if k not in seen:
                all_metric_keys.append(k)
                seen.add(k)

    base_cols = [
        "benchmark",
        "approach",
        "variant",
        "label",
        "instance",
        "target_k",
        "target_g",
        "success",
        "elapsed_s",
        "steps",
        "n_nodes",
        "n_edges",
        "model_id",
        "model_params",
        "model_size_mb",
        "error",
    ]
    fieldnames = base_cols + [f"metric_{k}" for k in all_metric_keys]

    csv_path = out_dir / f"{benchmark}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        _ = writer.writeheader()
        for r in results:
            row: dict[str, object] = {
                "benchmark": r.benchmark,
                "approach": r.approach,
                "variant": r.variant,
                "label": r.label,
                "instance": r.instance,
                "target_k": r.target.get("k", ""),
                "target_g": r.target.get("g", ""),
                "success": r.success,
                "elapsed_s": r.elapsed_s,
                "steps": r.steps if r.steps is not None else "",
                "n_nodes": r.n_nodes if r.n_nodes is not None else "",
                "n_edges": r.n_edges if r.n_edges is not None else "",
                "model_id": r.model_id if r.model_id is not None else "",
                "model_params": r.model_params if r.model_params is not None else "",
                "model_size_mb": (
                    r.model_size_mb if r.model_size_mb is not None else ""
                ),
                "error": r.error if r.error is not None else "",
            }
            for k in all_metric_keys:
                row[f"metric_{k}"] = r.metrics.get(k, "")
            _ = writer.writerow(row)

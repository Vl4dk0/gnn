"""Generate a histogram of cage-run elapsed times split by outcome.

SUCCESS bars (navy) are concentrated near 0-15 s; FAILURE bars (bronze/red)
spike at the 60 s budget wall.  The key message: fast success or full timeout.

Output: presentation/figures/results/fig-time-budget.pdf
"""

import json
from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RAW_JSON = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "runs"
    / "2026-05-30_17-41-38"
    / "raw.json"
)

NAVY = "#1f3a5f"
BRONZE = "#8c3b3b"
GRID = "#d0d0d0"


def load_cage_runs() -> tuple[list[float], list[float]]:
    with RAW_JSON.open() as fh:
        data = cast(list[dict[str, object]], json.load(fh))
    cage = [r for r in data if r.get("benchmark") == "cage"]
    successes = [float(cast(float, r["elapsed_s"])) for r in cage if r["success"]]
    failures = [float(cast(float, r["elapsed_s"])) for r in cage if not r["success"]]
    return successes, failures


def main() -> None:
    successes, failures = load_cage_runs()

    subplots_fn = cast(Callable[..., tuple[Figure, Axes]], plt.subplots)
    fig, ax = subplots_fn(figsize=(8.5, 5))

    set_facecolor_fig = cast(Callable[..., object], fig.patch.set_facecolor)
    _ = set_facecolor_fig("#ffffff")
    set_facecolor_ax = cast(Callable[..., object], ax.set_facecolor)
    _ = set_facecolor_ax("#ffffff")

    # Bins: fine resolution up to 60 s, one wide bin at 60+ to capture the wall.
    bins = list(np.arange(0, 60.5, 2.5)) + [65.0]

    hist_fn = cast(Callable[..., object], ax.hist)
    _ = hist_fn(
        successes,
        bins=bins,
        color=NAVY,
        alpha=0.7,
        label=f"ÚSPECH (n={len(successes)})",
        zorder=3,
    )
    _ = hist_fn(
        failures,
        bins=bins,
        color=BRONZE,
        alpha=0.7,
        label=f"ZLYHANIE (n={len(failures)})",
        zorder=3,
    )

    # Grid lines behind bars.
    grid_fn = cast(Callable[..., object], ax.yaxis.grid)
    _ = grid_fn(True, color=GRID, linewidth=0.8, zorder=0)
    set_axisbelow = cast(Callable[..., object], ax.set_axisbelow)
    _ = set_axisbelow(True)

    # Budget line at 60 s.
    axvline_fn = cast(Callable[..., object], ax.axvline)
    _ = axvline_fn(x=60.0, color="#555555", linestyle="--", linewidth=1.5, zorder=4)

    text_fn = cast(Callable[..., object], ax.text)
    _ = text_fn(
        59.6,
        ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 20,
        "rozpočet 60 s\n(rovnaký pre všetky metódy)",
        fontsize=10,
        color="#555555",
        ha="right",
        va="top",
        zorder=5,
    )

    # Annotation: success cluster.
    annotate_fn = cast(Callable[..., object], ax.annotate)
    _ = annotate_fn(
        "úspech: medián 0,74 s",
        xy=(1.25, 5),
        xytext=(10, 22),
        textcoords="data",
        fontsize=11,
        color=NAVY,
        arrowprops={"arrowstyle": "->", "color": NAVY, "lw": 1.4},
        zorder=6,
    )

    # Annotation: failure spike — position label to the left of the spike.
    _ = text_fn(
        56.5,
        22,
        "zlyhanie:\nvyčerpá celý\nrozpočet",
        fontsize=11,
        color=BRONZE,
        ha="right",
        va="center",
        zorder=6,
    )

    set_xlabel = cast(Callable[..., object], ax.set_xlabel)
    _ = set_xlabel("čas do výsledku (s)", fontsize=14)

    set_ylabel = cast(Callable[..., object], ax.set_ylabel)
    _ = set_ylabel("počet behov", fontsize=14)

    set_title = cast(Callable[..., object], ax.set_title)
    _ = set_title("Úspech príde rýchlo, alebo nepríde", fontsize=15, pad=10)

    set_xlim = cast(Callable[..., object], ax.set_xlim)
    _ = set_xlim(0, 62)

    tick_params_fn = cast(Callable[..., object], ax.tick_params)
    _ = tick_params_fn(labelsize=13)

    # Custom x-ticks: show 0, 10, 20, ..., 60 (wall) clearly.
    set_xticks = cast(Callable[..., object], ax.set_xticks)
    _ = set_xticks([0, 10, 20, 30, 40, 50, 60])

    for side in ("top", "right"):
        spine = ax.spines[side]
        set_vis = cast(Callable[..., object], spine.set_visible)
        _ = set_vis(False)

    legend_fn = cast(Callable[..., object], ax.legend)
    _ = legend_fn(
        handles=[
            Patch(facecolor=NAVY, alpha=0.7, label=f"ÚSPECH (n={len(successes)})"),
            Patch(facecolor=BRONZE, alpha=0.7, label=f"ZLYHANIE (n={len(failures)})"),
        ],
        fontsize=12,
        frameon=False,
        loc="upper left",
    )

    out = FIGURES / "results" / "fig-time-budget.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(
        out,
        format="pdf",
        bbox_inches="tight",
        facecolor="#ffffff",
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()

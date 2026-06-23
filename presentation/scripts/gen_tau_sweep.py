"""Generate a coarse-to-fine τ-sweep illustration for the defense deck.

Three horizontal number-lines (rounds) show a narrowing sweep over the
defective-fraction threshold τ.  Each row zooms into the previous best
neighbourhood; funnel connectors show how the interval narrows.

Output: presentation/figures/results/fig-tau-sweep.pdf
"""

from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch, FancyBboxPatch

matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --- Colour palette ----------------------------------------------------------
NAVY = "#1f3a5f"
GREY = "#888888"
GRID = "#d0d0d0"

# --- Row definitions ---------------------------------------------------------
# Each row: (label, ticks, best_tau, band_lo, band_hi)
ROWS: list[tuple[str, list[float], float, float | None, float | None]] = [
    (
        "kolo 1:  [0; 1],  krok 0,25",
        [0.0, 0.25, 0.5, 0.75, 1.0],
        0.25,
        None,
        None,
    ),
    (
        "kolo 2:  [0; 0,5],  krok 0,125",
        [0.0, 0.125, 0.25, 0.375, 0.5],
        0.375,
        0.0,
        0.5,
    ),
    (
        "kolo 3:  [0,25; 0,5],  krok 0,0625",
        [0.25, 0.3125, 0.375, 0.4375, 0.5],
        0.3125,
        0.25,
        0.5,
    ),
]

X_MIN = -0.05
X_MAX = 1.10

# y-coordinates of each row's number-line (in figure-fraction, set by subplots)
ROW_Y = 0.5  # within each subplot the line lives at y=0.5


# --- Main --------------------------------------------------------------------


def main() -> None:
    subplots_fn = cast(
        Callable[..., tuple[Figure, list[Axes]]],
        plt.subplots,
    )
    fig, axes = subplots_fn(
        3,
        1,
        figsize=(9, 4.5),
        gridspec_kw={"hspace": 0.55},
    )

    set_facecolor_fig = cast(Callable[..., object], fig.patch.set_facecolor)
    _ = set_facecolor_fig("#ffffff")

    prev_best: float | None = None
    prev_ax: Axes | None = None
    prev_band_lo: float | None = None
    prev_band_hi: float | None = None

    for row_idx, (label, ticks, best_tau, band_lo, band_hi) in enumerate(ROWS):
        ax: Axes = axes[row_idx]

        set_facecolor_ax = cast(Callable[..., object], ax.set_facecolor)
        _ = set_facecolor_ax("#ffffff")

        # Hide all spines except bottom (the number-line itself).
        for side in ("top", "left", "right"):
            spine = ax.spines[side]
            set_vis = cast(Callable[..., object], spine.set_visible)
            _ = set_vis(False)

        bottom_spine = ax.spines["bottom"]
        set_bottom_color = cast(Callable[..., object], bottom_spine.set_color)
        _ = set_bottom_color(GRID)

        # x range, no y-axis ticks/labels.
        set_xlim = cast(Callable[..., object], ax.set_xlim)
        _ = set_xlim(X_MIN, X_MAX)

        set_ylim = cast(Callable[..., object], ax.set_ylim)
        _ = set_ylim(0.0, 1.0)

        tick_params_fn = cast(Callable[..., object], ax.tick_params)
        _ = tick_params_fn(left=False, labelleft=False, labelsize=10)

        # Set x ticks at the sampled τ values only.
        set_xticks = cast(Callable[..., object], ax.set_xticks)
        _ = set_xticks(ticks)

        # Row 3 has the most cramped labels: smaller font + slight rotation.
        tick_fontsize = 7 if row_idx == 2 else 9
        tick_rotation = 30 if row_idx == 2 else 0
        tick_ha = "right" if row_idx == 2 else "center"
        set_xticklabels = cast(Callable[..., object], ax.set_xticklabels)
        _ = set_xticklabels(
            [str(t).replace(".", ",") for t in ticks],
            fontsize=tick_fontsize,
            color="#444444",
            rotation=tick_rotation,
            ha=tick_ha,
        )

        # Shaded band for this row (rows 2 and 3).
        if band_lo is not None and band_hi is not None:
            add_patch = cast(Callable[..., object], ax.add_patch)
            _ = add_patch(
                FancyBboxPatch(
                    (band_lo, 0.0),
                    band_hi - band_lo,
                    1.0,
                    boxstyle="square,pad=0",
                    facecolor="#d6e8f7",
                    alpha=0.5,
                    zorder=0,
                )
            )

        # Tick dots for non-best points.
        scatter_fn = cast(Callable[..., object], ax.scatter)
        for t in ticks:
            if t != best_tau:
                _ = scatter_fn(
                    t,
                    0.5,
                    s=60,
                    color=GREY,
                    zorder=3,
                    clip_on=False,
                )

        # Best τ: large navy marker.
        _ = scatter_fn(
            best_tau,
            0.5,
            s=220,
            color=NAVY,
            zorder=4,
            clip_on=False,
        )

        # "najlepšie τ" label above the best marker.
        text_fn = cast(Callable[..., object], ax.text)
        _ = text_fn(
            best_tau,
            0.78,
            "najlepšie τ",
            fontsize=9,
            color=NAVY,
            ha="center",
            va="bottom",
            zorder=5,
        )

        # Small arrow from label down to marker.
        annotate_fn = cast(Callable[..., object], ax.annotate)
        _ = annotate_fn(
            "",
            xy=(best_tau, 0.56),
            xytext=(best_tau, 0.76),
            arrowprops={
                "arrowstyle": "-|>",
                "color": NAVY,
                "lw": 1.2,
            },
            zorder=5,
        )

        # Round label: horizontal, left-aligned, above this number-line.
        _ = text_fn(
            X_MIN,
            1.02,
            label,
            fontsize=9,
            color="#333333",
            ha="left",
            va="bottom",
            clip_on=False,
        )

        # τ symbol at far right of x-axis.
        _ = text_fn(
            1.07,
            0.5,
            "τ",
            fontsize=12,
            color="#333333",
            ha="left",
            va="center",
            style="italic",
        )

        # "..." on the last row to imply continuation.
        if row_idx == 2:
            _ = text_fn(
                1.00,
                0.5,
                "…",
                fontsize=13,
                color=GREY,
                ha="center",
                va="center",
            )

        # --- Funnel connectors from previous row's best down to this band ---
        if (
            prev_ax is not None
            and prev_best is not None
            and band_lo is not None
            and band_hi is not None
        ):
            # We draw in figure coordinates using fig.add_artist / transFigure.
            # More reliably: draw in the *current* axes using the axes transform,
            # but we need to reach into the previous axes.  Use ConnectionPatch.
            for band_edge in (band_lo, band_hi):
                cp = ConnectionPatch(
                    xyA=(prev_best, 0.5),
                    coordsA=prev_ax.transData,
                    xyB=(band_edge, 1.0),
                    coordsB=ax.transData,
                    color=GREY,
                    linewidth=0.8,
                    linestyle="--",
                    alpha=0.7,
                    zorder=1,
                )
                _ = cast(Callable[..., object], fig.add_artist)(cp)

        prev_best = best_tau
        prev_ax = ax
        prev_band_lo = band_lo
        prev_band_hi = band_hi

    # Suppress unused-variable warning for the last loop values.
    _ = prev_band_lo
    _ = prev_band_hi

    out = FIGURES / "results" / "fig-tau-sweep.pdf"
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

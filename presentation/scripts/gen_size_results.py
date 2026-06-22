"""Generate the |V|/Moore-bound bar chart for the defense deck.

Shows graph size relative to the Moore bound for the three wide-reach
construction methods: voltage, voltage-rl, Forge. Lower = closer to the cage.

Output: presentation/figures/results/fig-size.pdf
"""

from __future__ import annotations

import os
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

matplotlib.rcParams["pdf.fonttype"] = 42

OUTPUT = os.path.join(
    os.path.dirname(__file__), "..", "figures", "results", "fig-size.pdf"
)

METHODS = ["voltage", "voltage-rl", "Forge"]
RATIOS = [1.72, 1.66, 1.23]

COLOR_NAVY = "#1f3a5f"
COLOR_FORGE = "#8c3b3b"
COLOR_EDGE = "#1a1a1a"
COLOR_SPINE = "#888888"
COLOR_REF_LINE = "#8c8c8c"
COLOR_REF_LABEL = "#6b6b6b"


def main() -> None:
    figure_fn = cast(Callable[..., Figure], plt.figure)
    fig = figure_fn(figsize=(7.5, 4.6), facecolor="#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#ffffff")

    colors = [COLOR_FORGE if m == "Forge" else COLOR_NAVY for m in METHODS]

    bars = ax.bar(
        METHODS,
        RATIOS,
        color=colors,
        edgecolor=COLOR_EDGE,
        linewidth=0.6,
        width=0.5,
    )

    # Value labels above each bar (comma as decimal separator, trailing ×)
    for bar, val in zip(bars, RATIOS):
        label = f"{val:.2f}".replace(".", ",") + "×"
        _ = ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=12,
            color="#1a1a1a",
        )

    # Horizontal dashed reference line at y = 1.0
    _ = ax.axhline(y=1.0, color=COLOR_REF_LINE, linestyle="--", linewidth=1.2, zorder=0)
    _ = ax.text(
        -0.42,
        1.03,
        "klietka (1×)",
        ha="left",
        va="bottom",
        fontsize=10,
        color=COLOR_REF_LABEL,
    )

    _ = ax.set_ylim(0, 1.95)
    _ = ax.set_ylabel("veľkosť grafu / Moore's bound", fontsize=13)
    _ = ax.tick_params(axis="x", labelsize=13)
    _ = ax.tick_params(axis="y", labelsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_SPINE)
    ax.spines["bottom"].set_color(COLOR_SPINE)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(
        OUTPUT,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor="#ffffff",
    )
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()

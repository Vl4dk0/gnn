from __future__ import annotations

import os
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

matplotlib.rcParams["pdf.fonttype"] = 42

OUTPUT = os.path.join(
    os.path.dirname(__file__), "..", "figures", "results", "fig-pred-results.pdf"
)

ARCHITECTURES = ["SAGE", "GPS-GIN", "GIN", "Loopy", "GCN"]
DEGREE_ACC = [1.000, 0.998, 0.887, 0.512, 0.382]
GIRTH_ACC = [0.119, 0.068, 0.055, 0.510, 0.227]

COLOR_DARK = "#1a1a1a"
COLOR_DEGREE = "#1f3a5f"
COLOR_GIRTH = "#8c3b3b"
COLOR_SPINE = "#888888"


def main() -> None:
    figure_fn = cast(Callable[..., Figure], plt.figure)
    fig = figure_fn(figsize=(8.5, 4.6), facecolor="#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#ffffff")

    x = np.arange(len(ARCHITECTURES))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        DEGREE_ACC,
        width,
        color=COLOR_DEGREE,
        edgecolor=COLOR_DARK,
        linewidth=0.6,
        label="stupeň",
    )
    bars2 = ax.bar(
        x + width / 2,
        GIRTH_ACC,
        width,
        color=COLOR_GIRTH,
        edgecolor=COLOR_DARK,
        linewidth=0.6,
        label="obvod",
    )

    for bar, val in zip(bars1, DEGREE_ACC):
        _ = ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    for bar, val in zip(bars2, GIRTH_ACC):
        _ = ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    _ = ax.set_ylim(0, 1.12)
    _ = ax.set_ylabel("presnosť predikcie", fontsize=13)
    _ = ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    _ = ax.tick_params(axis="both", labelsize=13)
    _ = ax.set_xticks(x)
    _ = ax.set_xticklabels(ARCHITECTURES, fontsize=13)

    _ = ax.legend(loc="upper right", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_SPINE)
    ax.spines["bottom"].set_color(COLOR_SPINE)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    _ = fig.savefig(
        OUTPUT,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor="#ffffff",
    )
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()

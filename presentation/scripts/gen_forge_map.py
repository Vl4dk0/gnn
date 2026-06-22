"""Generate a 2-D trade-off map for the Forge pipeline defense slide.

Positions three graph-construction approaches (voltage, klasické, Forge) on
axes of graph size vs. reach, arguing that Forge lands in the desirable region
(small size, good reach). This is a schematic conceptual figure, not a data plot.

Output: presentation/figures/forge/forge-map.pdf
"""

from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# Palette
BG = "#ffffff"
TEXT_DARK = "#1a1a1a"
GREY_MED = "#6b6b6b"
GREY_FADED = "#9a9a9a"
GREY_SPINE = "#888888"
GREY_DASH = "#bbbbbb"
GREY_ARROW = "#aaaaaa"
GREY_REGION = "#e8e8e8"


def make_figure() -> Figure:
    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=(8.0, 5.0), facecolor=BG)
    ax: Axes = fig.add_subplot(111, facecolor=BG)

    # --- Good corner shaded region (drawn first, behind everything) ----------
    rect = Rectangle(
        (0.7, 0.58),
        1.35 - 0.7,
        1.0 - 0.58,
        linewidth=0,
        edgecolor="none",
        facecolor=GREY_REGION,
        alpha=0.55,
        zorder=1,
    )
    _ = ax.add_patch(rect)

    text_fn = cast(Callable[..., object], ax.text)

    # --- Cage reference vertical dashed line ---------------------------------
    plot_fn = cast(Callable[..., object], ax.plot)
    _ = plot_fn(
        [1.0, 1.0],
        [0.0, 1.0],
        color=GREY_DASH,
        linewidth=1.2,
        linestyle="--",
        zorder=2,
    )
    _ = text_fn(
        1.0,
        0.03,
        "klietka",
        ha="center",
        va="bottom",
        fontsize=10,
        color=GREY_FADED,
        zorder=3,
    )

    # --- Composition arrows (behind points) ----------------------------------
    annotate_fn = cast(Callable[..., object], ax.annotate)

    # From voltage to Forge
    _ = annotate_fn(
        "",
        xy=(1.23, 0.62),
        xytext=(2.0, 0.85),
        arrowprops=dict(
            arrowstyle="-|>",
            color=GREY_ARROW,
            lw=1.1,
            connectionstyle="arc3,rad=-0.2",
        ),
        zorder=3,
    )

    # From klasické to Forge
    _ = annotate_fn(
        "",
        xy=(1.23, 0.62),
        xytext=(1.0, 0.35),
        arrowprops=dict(
            arrowstyle="-|>",
            color=GREY_ARROW,
            lw=1.1,
            connectionstyle="arc3,rad=0.2",
        ),
        zorder=3,
    )

    # --- Scatter points -------------------------------------------------------
    scatter_fn = cast(Callable[..., object], ax.scatter)

    # voltage
    _ = scatter_fn(
        [2.0],
        [0.85],
        s=220,
        color=GREY_MED,
        zorder=10,
        clip_on=False,
    )
    _ = text_fn(
        2.05,
        0.85,
        "voltage",
        ha="left",
        va="center",
        fontsize=12,
        color=TEXT_DARK,
        zorder=11,
    )

    # klasické
    _ = scatter_fn(
        [1.0],
        [0.35],
        s=220,
        color=GREY_MED,
        zorder=10,
        clip_on=False,
    )
    _ = text_fn(
        0.93,
        0.30,
        "klasické",
        ha="right",
        va="top",
        fontsize=12,
        color=TEXT_DARK,
        zorder=11,
    )

    # Forge (largest, on top)
    _ = scatter_fn(
        [1.23],
        [0.62],
        s=340,
        color=TEXT_DARK,
        zorder=12,
        clip_on=False,
    )
    _ = text_fn(
        1.29,
        0.65,
        "Forge",
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=TEXT_DARK,
        zorder=13,
    )

    # --- Axes formatting ------------------------------------------------------
    set_xlabel = cast(Callable[..., object], ax.set_xlabel)
    set_ylabel = cast(Callable[..., object], ax.set_ylabel)
    set_xlim = cast(Callable[..., object], ax.set_xlim)
    set_ylim = cast(Callable[..., object], ax.set_ylim)
    set_xticks = cast(Callable[..., object], ax.set_xticks)
    set_yticks = cast(Callable[..., object], ax.set_yticks)

    _ = set_xlim(0.7, 2.3)
    _ = set_ylim(0.0, 1.0)
    _ = set_xticks([1.0, 1.5, 2.0])
    _ = set_yticks([])

    _ = set_xlabel(
        "veľkosť grafu / Moore's bound",
        fontsize=12,
        color=TEXT_DARK,
    )
    _ = set_ylabel(
        "dosah",
        fontsize=12,
        color=TEXT_DARK,
    )

    # Spine cleanup
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GREY_SPINE)
    ax.spines["bottom"].set_color(GREY_SPINE)
    tick_params_fn = cast(Callable[..., object], ax.tick_params)
    _ = tick_params_fn(colors=GREY_SPINE)

    return fig


def main() -> None:
    out_dir = FIGURES / "forge"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "forge-map.pdf"

    fig = make_figure()

    savefig_fn = cast(Callable[..., object], fig.savefig)
    _ = savefig_fn(
        out_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor=BG,
    )

    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

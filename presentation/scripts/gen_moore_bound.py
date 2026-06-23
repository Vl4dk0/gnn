"""Generate a BFS-ball tree figure illustrating WHY the Moore bound is a lower bound.

Shows the forced distinct vertices for k=3, g=5: root + 3 level-1 neighbours +
6 level-2 grandchildren = 10 vertices minimum - exactly the Petersen graph order.

Output: presentation/figures/petersen/fig-moore.pdf
"""

from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# Palette - same as gen_opening_figures.py
NODE_BASE = "#6b6b6b"
NODE_HI = "#1a1a1a"
EDGE_BASE = "#aaaaaa"
TEXT_DARK = "#1a1a1a"
TEXT_MUTED = "#555555"

NODE_SIZE = 360
EDGE_LW = 1.8


# Node positions. The tree is drawn WIDE (and short) so that, sized by height on
# the slide, it fills the slide width and the tree itself stays large.
LEVEL0: list[tuple[float, float]] = [(0.0, 3.0)]
LEVEL1: list[tuple[float, float]] = [(-2.9, 1.6), (0.0, 1.6), (2.9, 1.6)]
LEVEL2: list[tuple[float, float]] = [
    (-3.9, 0.0),
    (-2.2, 0.0),
    (-0.8, 0.0),
    (0.8, 0.0),
    (2.2, 0.0),
    (3.9, 0.0),
]

# Edges: node indices into ALL_POS (0=root, 1-3=level1, 4-9=level2)
EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (1, 5),
    (2, 6),
    (2, 7),
    (3, 8),
    (3, 9),
]

ALL_POS: list[tuple[float, float]] = LEVEL0 + LEVEL1 + LEVEL2


def draw_tree(ax: Axes) -> None:
    """Draw edges first, then nodes on top."""
    plot = cast(Callable[..., object], ax.plot)

    for u, v in EDGES:
        xu, yu = ALL_POS[u]
        xv, yv = ALL_POS[v]
        _ = plot(
            [xu, xv],
            [yu, yv],
            color=EDGE_BASE,
            linewidth=EDGE_LW,
            zorder=1,
            solid_capstyle="round",
        )

    scatter = cast(Callable[..., object], ax.scatter)
    # Root highlighted
    rx, ry = LEVEL0[0]
    _ = scatter([rx], [ry], s=NODE_SIZE, color=NODE_HI, zorder=2)
    # Level-1 nodes
    xs1 = [p[0] for p in LEVEL1]
    ys1 = [p[1] for p in LEVEL1]
    _ = scatter(xs1, ys1, s=NODE_SIZE, color=NODE_BASE, zorder=2)
    # Level-2 nodes
    xs2 = [p[0] for p in LEVEL2]
    ys2 = [p[1] for p in LEVEL2]
    _ = scatter(xs2, ys2, s=NODE_SIZE, color=NODE_BASE, zorder=2)


def draw_annotations(ax: Axes) -> None:
    """Add level count labels on the left, depth brace on the right, totals below."""
    text = cast(Callable[..., object], ax.text)
    annotate = cast(Callable[..., object], ax.annotate)

    label_x = -4.5

    # Count labels - one number per row
    _ = text(
        label_x,
        3.0,
        "1",
        ha="right",
        va="center",
        fontsize=15,
        color=TEXT_DARK,
        fontweight="bold",
    )
    _ = text(
        label_x,
        1.6,
        "3",
        ha="right",
        va="center",
        fontsize=15,
        color=TEXT_DARK,
        fontweight="bold",
    )
    _ = text(
        label_x,
        0.0,
        "6",
        ha="right",
        va="center",
        fontsize=15,
        color=TEXT_DARK,
        fontweight="bold",
    )

    # Right-side depth brace annotation
    brace_x = 4.6
    brace_top = 3.0
    brace_bot = 0.0
    brace_mid = (brace_top + brace_bot) / 2.0
    tick_len = 0.15

    vline = cast(Callable[..., object], ax.vlines)
    hline = cast(Callable[..., object], ax.hlines)
    _ = vline(brace_x, brace_bot, brace_top, color=TEXT_MUTED, linewidth=1.2, zorder=3)
    _ = hline(
        brace_top,
        brace_x - tick_len,
        brace_x,
        color=TEXT_MUTED,
        linewidth=1.2,
        zorder=3,
    )
    _ = hline(
        brace_bot,
        brace_x - tick_len,
        brace_x,
        color=TEXT_MUTED,
        linewidth=1.2,
        zorder=3,
    )
    _ = annotate(
        "",
        xy=(brace_x, brace_mid),
        xytext=(brace_x + 0.05, brace_mid),
        annotation_clip=False,
    )
    _ = text(
        brace_x + 0.22,
        brace_mid,
        "(g-1)/2 = 2",
        ha="left",
        va="center",
        fontsize=10,
        color=TEXT_MUTED,
        rotation=90,
        rotation_mode="anchor",
    )

    # Total line
    _ = text(
        0.0,
        -0.55,
        "1 + 3 + 6 = 10",
        ha="center",
        va="top",
        fontsize=16,
        color=TEXT_DARK,
        fontweight="bold",
    )


def main() -> None:
    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=(12, 5.0))
    ax: Axes = fig.add_subplot(111)

    draw_tree(ax)
    draw_annotations(ax)

    _ = ax.axis("off")
    _ = ax.set_xlim(-5.2, 5.6)
    _ = ax.set_ylim(-1.1, 3.45)

    out = FIGURES / "petersen" / "fig-moore.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(
        out,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
        facecolor="none",
    )
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

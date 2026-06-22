"""Generate a BFS-ball tree figure illustrating WHY the Moore bound is a lower bound.

Shows the forced distinct vertices for k=3, g=5: root + 3 level-1 neighbours +
6 level-2 grandchildren = 10 vertices minimum — exactly the Petersen graph order.

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

# Palette
NAVY = "#1f3a5f"
EDGE_COLOR = "#aaaaaa"
TEXT_DARK = "#1a1a1a"
TEXT_MUTED = "#555555"

NODE_SIZE = 420
EDGE_LW = 1.8


# Node positions
LEVEL0: list[tuple[float, float]] = [(0.0, 3.0)]
LEVEL1: list[tuple[float, float]] = [(-2.4, 1.6), (0.0, 1.6), (2.4, 1.6)]
LEVEL2: list[tuple[float, float]] = [
    (-3.1, 0.0),
    (-1.7, 0.0),
    (-0.7, 0.0),
    (0.7, 0.0),
    (1.7, 0.0),
    (3.1, 0.0),
]

# Edges: (index into flat node list, index into flat node list)
# Nodes order: 0=root, 1-3=level1, 4-9=level2
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

    # Edges
    for u, v in EDGES:
        xu, yu = ALL_POS[u]
        xv, yv = ALL_POS[v]
        _ = plot(
            [xu, xv],
            [yu, yv],
            color=EDGE_COLOR,
            linewidth=EDGE_LW,
            zorder=1,
            solid_capstyle="round",
        )

    # Nodes
    scatter = cast(Callable[..., object], ax.scatter)
    xs = [p[0] for p in ALL_POS]
    ys = [p[1] for p in ALL_POS]
    _ = scatter(xs, ys, s=NODE_SIZE, color=NAVY, zorder=2)


def draw_annotations(ax: Axes) -> None:
    """Add level count labels on the left and totals below the tree."""
    text = cast(Callable[..., object], ax.text)

    label_x = -4.3

    # Level 0 label
    _ = text(
        label_x,
        3.0,
        "1 koreň",
        ha="right",
        va="center",
        fontsize=14,
        color=TEXT_DARK,
    )

    # Level 1 label
    _ = text(
        label_x,
        1.6,
        "k = 3",
        ha="right",
        va="center",
        fontsize=14,
        color=TEXT_DARK,
    )

    # Level 2 label
    _ = text(
        label_x,
        0.0,
        "k-1 na každého → 6",
        ha="right",
        va="center",
        fontsize=14,
        color=TEXT_DARK,
    )

    # Total line
    _ = text(
        0.0,
        -1.05,
        "1 + 3 + 6 = 10 vrcholov",
        ha="center",
        va="top",
        fontsize=16,
        color=TEXT_DARK,
        fontweight="bold",
    )

    # Subtitle line
    _ = text(
        0.0,
        -1.7,
        "= Moore's bound pre (3,5)",
        ha="center",
        va="top",
        fontsize=13,
        color=TEXT_MUTED,
    )


def main() -> None:
    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=(10, 6.2))
    ax: Axes = fig.add_subplot(111)

    draw_tree(ax)
    draw_annotations(ax)

    _ = ax.axis("off")
    _ = ax.set_xlim(-5.0, 4.2)
    _ = ax.set_ylim(-2.1, 3.5)

    out = FIGURES / "petersen" / "fig-moore.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(
        out,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.12,
        transparent=True,
        facecolor="none",
    )
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

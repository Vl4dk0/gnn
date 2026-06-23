"""Generate small shaped graphs for the closing slide's background.

Nodes are sampled along a shape outline and wired into a cycle (the outline)
plus a few internal chords/veins so each reads as a graph, not just a curve.
Greyscale, transparent background so many faint copies can be scattered behind
the "Ďakujem za pozornosť" closer.

Outputs:
  presentation/figures/petersen/fig-heart.pdf
  presentation/figures/petersen/fig-leaf.pdf
"""

import math
from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# Light greyscale: these are scattered as faint background decoration, so they
# must recede behind the hero Petersen graph and the headline.
NODE_COLOR = "#cccccc"
EDGE_COLOR = "#d8d8d8"

N_NODES = 16
NODE_SIZE = 70
EDGE_LW = 1.8


def heart_point(t: float) -> tuple[float, float]:
    """Parametric heart curve, t in [0, 2*pi)."""
    x = 16.0 * math.sin(t) ** 3
    y = (
        13.0 * math.cos(t)
        - 5.0 * math.cos(2.0 * t)
        - 2.0 * math.cos(3.0 * t)
        - math.cos(4.0 * t)
    )
    return x, y


def leaf_outline(n: int) -> list[tuple[float, float]]:
    """An almond/leaf outline: two quadratic-Bézier sides from base to tip."""
    base = (0.0, -1.0)
    tip = (0.0, 1.0)
    bulge = 0.62
    half = n // 2

    def bezier(c: tuple[float, float], s: float) -> tuple[float, float]:
        x = (1 - s) ** 2 * base[0] + 2 * (1 - s) * s * c[0] + s**2 * tip[0]
        y = (1 - s) ** 2 * base[1] + 2 * (1 - s) * s * c[1] + s**2 * tip[1]
        return x, y

    # Right side base -> tip, then left side tip -> base (closing the cycle).
    right = [bezier((bulge, 0.0), i / half) for i in range(half)]
    left = [bezier((-bulge, 0.0), 1.0 - i / (n - half)) for i in range(n - half)]
    return right + left


def draw_graph(
    pts: list[tuple[float, float]],
    chords: list[tuple[int, int]],
    out_name: str,
    figsize: tuple[float, float],
) -> None:
    """Draw an outline cycle over `pts` plus `chords`, save transparent."""
    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=figsize)
    ax: Axes = fig.add_subplot(111)

    plot = cast(Callable[..., object], ax.plot)
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        _ = plot(
            [x0, x1],
            [y0, y1],
            color=EDGE_COLOR,
            linewidth=EDGE_LW,
            zorder=1,
            solid_capstyle="round",
        )
    for a, b in chords:
        xa, ya = pts[a]
        xb, yb = pts[b]
        _ = plot(
            [xa, xb],
            [ya, yb],
            color=EDGE_COLOR,
            linewidth=EDGE_LW * 0.8,
            zorder=1,
            solid_capstyle="round",
        )

    scatter = cast(Callable[..., object], ax.scatter)
    _ = scatter(
        [p[0] for p in pts],
        [p[1] for p in pts],
        s=NODE_SIZE,
        color=NODE_COLOR,
        zorder=2,
    )

    _ = ax.axis("off")
    _ = ax.set_aspect("equal")

    out = FIGURES / "petersen" / out_name
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


def main() -> None:
    # Heart: outline cycle + symmetric internal chords.
    heart_pts = [
        heart_point(2.0 * math.pi * i / N_NODES + math.pi / 2.0) for i in range(N_NODES)
    ]
    heart_chords = [(2, N_NODES - 2), (4, N_NODES - 4), (6, N_NODES - 6)]
    draw_graph(heart_pts, heart_chords, "fig-heart.pdf", (4.0, 3.6))

    # Leaf: outline cycle + a midrib vein (base->tip) drawn as chords across it.
    leaf_pts = leaf_outline(N_NODES)
    half = N_NODES // 2
    # Indices 0 = base, `half` = tip. Midrib + a couple of side veins.
    leaf_chords = [
        (0, half),
        (half - 2, half + 2),
        (3, N_NODES - 3),
    ]
    draw_graph(leaf_pts, leaf_chords, "fig-leaf.pdf", (3.0, 4.2))


if __name__ == "__main__":
    main()

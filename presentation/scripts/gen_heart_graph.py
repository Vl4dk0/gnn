"""Generate a small heart-shaped graph for the closing slide's background.

Nodes are sampled along the classic heart curve and wired into a cycle (the
outline) plus a few internal chords so it reads as a graph, not just a curve.
Greyscale, transparent background so many faint copies can be scattered behind
the "Ďakujem za pozornosť" closer.

Output: presentation/figures/petersen/fig-heart.pdf
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


def main() -> None:
    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=(4.0, 3.6))
    ax: Axes = fig.add_subplot(111)

    # Sample nodes evenly in the parameter, offset so the cusp/tip sit symmetric.
    pts = [
        heart_point(2.0 * math.pi * i / N_NODES + math.pi / 2.0) for i in range(N_NODES)
    ]

    plot = cast(Callable[..., object], ax.plot)
    # Outline cycle.
    for i in range(N_NODES):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % N_NODES]
        _ = plot(
            [x0, x1],
            [y0, y1],
            color=EDGE_COLOR,
            linewidth=EDGE_LW,
            zorder=1,
            solid_capstyle="round",
        )

    # A few internal chords for graph-like structure (symmetric across the axis).
    chords = [(2, N_NODES - 2), (4, N_NODES - 4), (6, N_NODES - 6)]
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
    _ = ax.set_xlim(-19.0, 19.0)
    _ = ax.set_ylim(-18.0, 15.0)

    out = FIGURES / "petersen" / "fig-heart.pdf"
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

"""Generate the RL action/legality figure for the defense deck.

The agent's action is a vertex pair (u,v): if the edge is missing it is added,
if present it is removed. A move is only legal when it keeps the graph buildable
towards a (k,g)-graph. This figure shows three side-by-side moves on a target of
degree k = 3, girth g = 5:

  1. legal add   — closes a long enough cycle (>= g), degrees stay <= k,
  2. degree veto — an endpoint already has degree k,
  3. girth veto  — the edge would close a cycle shorter than g.

Each panel is locally correct; they are independent vignettes, not one graph.

Output: presentation/figures/rl/rl-rules.pdf (transparent vector PDF).
"""

import math
from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

NODE_FILL = "#6b6b6b"
NODE_EDGE = "#6b6b6b"
EDGE = "#aaaaaa"
GREEN = "#3c7d4a"
RED = "#8c3b3b"
INK = "#1a1a1a"


def pentagon(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    """Five points on a regular pentagon, first point at the top."""
    return [
        (cx + r * math.sin(2 * math.pi * i / 5), cy + r * math.cos(2 * math.pi * i / 5))
        for i in range(5)
    ]


def draw_nodes(ax: Axes, pts: list[tuple[float, float]], radius: float = 0.16) -> None:
    add_patch = cast(Callable[..., object], ax.add_patch)
    for x, y in pts:
        _ = add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=NODE_FILL,
                edgecolor=NODE_EDGE,
                linewidth=1.4,
                zorder=3,
            )
        )


def draw_edge(
    ax: Axes,
    p: tuple[float, float],
    q: tuple[float, float],
    *,
    color: str = EDGE,
    dashed: bool = False,
    width: float = 1.8,
) -> None:
    line = Line2D(
        [p[0], q[0]],
        [p[1], q[1]],
        color=color,
        linewidth=width,
        linestyle="--" if dashed else "-",
        zorder=2 if not dashed else 4,
        solid_capstyle="round",
    )
    add_line = cast(Callable[..., object], ax.add_line)
    _ = add_line(line)


def panel_title(ax: Axes, mark: str, text: str, sub: str, color: str) -> None:
    text_fn = cast(Callable[..., object], ax.text)
    _ = text_fn(
        0.5,
        1.16,
        f"{mark}  {text}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=15,
        color=color,
        fontweight="bold",
    )
    _ = text_fn(
        0.5,
        1.0,
        sub,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.5,
        color=INK,
    )


def setup(ax: Axes) -> None:
    _ = ax.set_xlim(-1.5, 1.5)
    _ = ax.set_ylim(-1.5, 1.5)
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")


def draw_legal(ax: Axes) -> None:
    """A path of 5 vertices; the candidate closes a 5-cycle (legal at g = 5)."""
    pts = pentagon(0.0, 0.0, 1.05)
    # Existing path edges along the pentagon, leaving one gap (the candidate).
    for i in range(4):
        draw_edge(ax, pts[i], pts[i + 1])
    draw_nodes(ax, pts)
    # Candidate: close the pentagon (vertices 4 -> 0), green dashed.
    draw_edge(ax, pts[4], pts[0], color=GREEN, dashed=True, width=2.6)
    panel_title(ax, "✓", "legálny ťah", "uzavrie 5-cyklus", GREEN)
    setup(ax)


def draw_degree_veto(ax: Axes) -> None:
    """Centre vertex already has degree k = 3; a 4th edge is vetoed."""
    c = (0.0, 0.1)
    leaves = [(-0.95, 0.85), (0.95, 0.85), (0.0, -1.05)]
    for lf in leaves:
        draw_edge(ax, c, lf)
    p = (1.05, -0.6)
    draw_nodes(ax, [c, *leaves, p])
    # Candidate from the new vertex p to the already-full centre, red dashed.
    draw_edge(ax, p, c, color=RED, dashed=True, width=2.6)
    text_fn = cast(Callable[..., object], ax.text)
    _ = text_fn(
        c[0],
        c[1] + 0.34,
        "stupeň 3",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=RED,
    )
    panel_title(ax, "✗", "porušený stupeň", "vrchol by mal stupeň > k", RED)
    setup(ax)


def draw_girth_veto(ax: Axes) -> None:
    """A path of 4 vertices; the candidate would close a 4-cycle (< g = 5)."""
    pts = [(-0.9, 0.8), (0.9, 0.8), (0.9, -0.8), (-0.9, -0.8)]
    # Existing path: 0-1-2-3 (three edges).
    for i in range(3):
        draw_edge(ax, pts[i], pts[i + 1])
    draw_nodes(ax, pts)
    # Candidate 3 -> 0 closes a 4-cycle, red dashed.
    draw_edge(ax, pts[3], pts[0], color=RED, dashed=True, width=2.6)
    panel_title(ax, "✗", "porušený obvod", "vznikol by 4-cyklus < g", RED)
    setup(ax)


def main() -> None:
    subplots_fn = cast(Callable[..., tuple[Figure, list[Axes]]], plt.subplots)
    fig, axes = subplots_fn(1, 3, figsize=(12.0, 4.2))

    draw_legal(axes[0])
    draw_degree_veto(axes[1])
    draw_girth_veto(axes[2])

    out = FIGURES / "rl" / "rl-rules.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(
        out,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=True,
        facecolor="none",
    )
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

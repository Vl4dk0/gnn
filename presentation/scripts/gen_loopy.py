"""Generate the Loopy GNN explainer figures for the defense deck.

Produces two separate vector PDFs:
  figures/gnn/fig-loopy-standard.pdf  — standard GNN panel
  figures/gnn/fig-loopy-loopy.pdf     — Loopy model panel
"""

from pathlib import Path
from typing import cast
from collections.abc import Callable
import math

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Embed TrueType fonts in the PDF (type 42) so text stays selectable and sharp.
matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --- Style palette ---
OUTLINE = "#3a3a3a"
FOCUS_OUTLINE = "#1a1a1a"
FADE_OUTLINE = "#dddddd"
FADE_TEXT = "#bbbbbb"
TEXT = "#1a1a1a"
FADE = 0.13
EDGE = "#9a9a9a"

# --- Node feature vectors (printed inside each box) ---
NODE_VECTORS: dict[str, tuple[int, int, int]] = {
    "v": (31, 4, 1),
    "a": (2, 31, 1),
    "b": (0, 3, 31),
    "c": (28, 18, 2),
    "d": (18, 2, 28),
    "e": (3, 28, 16),
    "f": (30, 2, 18),
}

# --- Graph definition ---
NODES = ["v", "a", "b", "c", "d", "e", "f"]
EDGES = [
    ("v", "a"),
    ("a", "b"),
    ("b", "c"),
    ("c", "d"),
    ("d", "v"),
    ("v", "e"),
    ("b", "f"),
]
CYCLE_EDGES = [("v", "a"), ("a", "b"), ("b", "c"), ("c", "d"), ("d", "v")]
CYCLE_NODES = ["v", "a", "b", "c", "d"]
V_NEIGHBORS = ["a", "d", "e"]


def build_graph() -> "nx.Graph[str]":
    g: nx.Graph[str] = nx.Graph()
    g.add_nodes_from(NODES)
    g.add_edges_from(EDGES)
    return g


def fixed_layout() -> dict[str, tuple[float, float]]:
    """Hand-tuned layout: v in center, a-b-c-d-e-f around it."""
    pos: dict[str, tuple[float, float]] = {}
    pos["v"] = (0.0, 0.0)

    r = 1.3
    a_angles = [54, 126, 198, 270]
    names = ["a", "b", "c", "d"]
    for name, ang in zip(names, a_angles):
        rad = math.radians(ang)
        pos[name] = (r * math.cos(rad), r * math.sin(rad))

    rad_e = math.radians(330)
    pos["e"] = (r * 0.95 * math.cos(rad_e), r * 0.95 * math.sin(rad_e))

    bx, by = pos["b"]
    pos["f"] = (bx - 0.55, by + 0.62)

    # The "(r, g, b)" boxes are wide, so scale positions up to give them room.
    scale = 1.7
    return {n: (x * scale, y * scale) for n, (x, y) in pos.items()}


POS = fixed_layout()


def _draw_box(ax: Axes, node: str, *, focus: bool = False, faded: bool = False) -> None:
    """Draw one vertex as a white rounded box with its "(r, g, b)" vector inside."""
    r, gg, b = NODE_VECTORS[node]
    x, y = POS[node]
    if faded:
        edgecolor = FADE_OUTLINE
        linewidth = 1.6
        textcolor = FADE_TEXT
    elif focus:
        edgecolor = FOCUS_OUTLINE
        linewidth = 3.0
        textcolor = TEXT
    else:
        edgecolor = OUTLINE
        linewidth = 1.6
        textcolor = TEXT
    bbox = dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    text_fn = cast(Callable[..., object], ax.text)
    _ = text_fn(
        x,
        y,
        f"({r}, {gg}, {b})",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color=textcolor,
        bbox=bbox,
        zorder=3,
    )


def _finish_axes(ax: Axes) -> None:
    """Equal aspect, hidden spines, generous margins so wide boxes never clip."""
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.autoscale_view()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xpad = (x1 - x0) * 0.22
    ypad = (y1 - y0) * 0.12
    _ = ax.set_xlim(x0 - xpad, x1 + xpad)
    _ = ax.set_ylim(y0 - ypad, y1 + ypad)


def draw_standard_gnn(ax: Axes) -> None:
    """Standard GNN panel: v and direct neighbours highlighted, rest faded."""
    g = build_graph()

    faded_nodes = [n for n in NODES if n not in ["v"] + V_NEIGHBORS]
    faded_edges: list[tuple[str, str]] = [
        e
        for e in EDGES
        if not (
            (e[0] == "v" and e[1] in V_NEIGHBORS)
            or (e[1] == "v" and e[0] in V_NEIGHBORS)
        )
    ]
    emphasized_edges: list[tuple[str, str]] = [("v", "a"), ("v", "d"), ("v", "e")]

    # Edges first, boxes on top.
    _ = nx.draw_networkx_edges(
        g, POS, ax=ax, edgelist=faded_edges, edge_color=EDGE, width=2.0, alpha=FADE
    )
    _ = nx.draw_networkx_edges(
        g, POS, ax=ax, edgelist=emphasized_edges, edge_color=EDGE, width=2.0
    )
    for n in faded_nodes:
        _draw_box(ax, n, faded=True)
    for n in V_NEIGHBORS:
        _draw_box(ax, n)
    _draw_box(ax, "v", focus=True)
    _finish_axes(ax)


def draw_loopy(ax: Axes) -> None:
    """Loopy model panel: 5-cycle through v highlighted bold dark."""
    g = build_graph()

    faded_nodes = [n for n in NODES if n not in CYCLE_NODES]
    faded_edges = [
        e for e in EDGES if e not in CYCLE_EDGES and (e[1], e[0]) not in CYCLE_EDGES
    ]
    cycle_chain = ["a", "b", "c", "d"]

    # Edges first, boxes on top.
    _ = nx.draw_networkx_edges(
        g, POS, ax=ax, edgelist=faded_edges, edge_color=EDGE, width=2.0, alpha=FADE
    )
    _ = nx.draw_networkx_edges(
        g,
        POS,
        ax=ax,
        edgelist=CYCLE_EDGES,
        edge_color="#1a1a1a",
        width=3.2,
    )
    for n in faded_nodes:
        _draw_box(ax, n, faded=True)
    for n in cycle_chain:
        _draw_box(ax, n)
    _draw_box(ax, "v", focus=True)
    _finish_axes(ax)


def gen_standard_gnn() -> None:
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    draw_standard_gnn(ax)

    out = FIGURES / "gnn" / "fig-loopy-standard.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.12,
        transparent=True,
        facecolor="none",
    )
    print(f"saved {out}")
    plt.close(fig)


def gen_loopy() -> None:
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    draw_loopy(ax)

    out = FIGURES / "gnn" / "fig-loopy-loopy.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.12,
        transparent=True,
        facecolor="none",
    )
    print(f"saved {out}")
    plt.close(fig)


def main() -> None:
    gen_standard_gnn()
    gen_loopy()


if __name__ == "__main__":
    main()

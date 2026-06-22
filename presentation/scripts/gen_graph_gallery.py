"""Generate a 2x2 gallery of (3,g)-cage graphs for the defense deck.

Produces figures/results/fig-gallery.pdf — a transparent, vector PDF showing
the classic increasing-girth sequence:

  (3,5)  Petersen graph      10 vertices
  (3,6)  Heawood graph       14 vertices
  (3,7)  McGee graph         24 vertices
  (3,8)  Tutte-Coxeter graph 30 vertices

Each cage is drawn with a circular layout so its rotational symmetry shows.
Every graph is asserted to be 3-regular with the expected girth before drawing.
"""

from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures" / "results"

# --- Palette ------------------------------------------------------------------
NODE_COLOR = "#1f3a5f"
EDGE_COLOR = "#aaaaaa"
CAPTION_COLOR = "#1a1a1a"

# --- Sizing -------------------------------------------------------------------
FIG_W = 10.0
FIG_H = 9.0
EDGE_WIDTH = 1.2

# Node sizes scale down for larger graphs so they stay clean.
NODE_SIZES: dict[str, int] = {
    "petersen": 200,
    "heawood": 160,
    "mcgee": 120,
    "tutte": 110,
}

CAPTION_FONTSIZE = 13


# --- Graph constructors -------------------------------------------------------
def make_petersen() -> "nx.Graph[int]":
    return cast("nx.Graph[int]", nx.petersen_graph())


def make_heawood() -> "nx.Graph[int]":
    return cast("nx.Graph[int]", nx.heawood_graph())


def make_mcgee() -> "nx.Graph[int]":
    return cast("nx.Graph[int]", nx.LCF_graph(24, [12, 7, -7], 8))


def make_tutte() -> "nx.Graph[int]":
    return cast("nx.Graph[int]", nx.LCF_graph(30, [-13, -9, 7, -7, 9, 13], 5))


# --- Verification -------------------------------------------------------------
def compute_girth(graph: "nx.Graph[int]") -> int:
    """Return the girth of graph via the minimum-cycle-basis length."""
    cycles = nx.minimum_cycle_basis(graph)
    if not cycles:
        return 0
    return min(len(c) for c in cycles)


def assert_cage(
    graph: "nx.Graph[int]",
    name: str,
    k: int,
    g: int,
) -> None:
    """Assert graph is k-regular with girth g, and print the result."""
    n = graph.number_of_nodes()
    degrees = [d for _, d in graph.degree()]
    assert all(d == k for d in degrees), (
        f"{name}: not {k}-regular (degrees {set(degrees)})"
    )
    girth = compute_girth(graph)
    assert girth == g, f"{name}: expected girth {g}, got {girth}"
    print(f"VERIFIED  {name:20s}  ({k},{g})-cage  {n} vertices  girth={girth}")


# --- Drawing ------------------------------------------------------------------
def draw_cage(
    ax: Axes,
    graph: "nx.Graph[int]",
    node_size: int,
    caption: str,
) -> None:
    """Draw one cage on ax with circular layout and a caption below."""
    pos = nx.circular_layout(graph)

    _ = nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color=EDGE_COLOR,
        width=EDGE_WIDTH,
    )
    _ = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=NODE_COLOR,
    )

    _ = ax.set_aspect("equal")
    _ = ax.axis("off")

    _ = ax.text(
        0.5,
        -0.03,
        caption,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=CAPTION_FONTSIZE,
        color=CAPTION_COLOR,
    )


# --- Main ---------------------------------------------------------------------
def main() -> None:
    graphs = [
        (make_petersen(), "petersen", 3, 5, "Petersen — (3,5)"),
        (make_heawood(), "heawood", 3, 6, "Heawood — (3,6)"),
        (make_mcgee(), "mcgee", 3, 7, "McGee — (3,7)"),
        (make_tutte(), "tutte", 3, 8, "Tutte–Coxeter — (3,8)"),
    ]

    # Verify all graphs before drawing anything.
    for graph, key, k, g, _ in graphs:
        assert_cage(graph, key, k, g)

    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=(FIG_W, FIG_H))

    for idx, (graph, key, _k, _g, caption) in enumerate(graphs):
        ax: Axes = fig.add_subplot(2, 2, idx + 1)
        draw_cage(ax, graph, NODE_SIZES[key], caption)

    fig.subplots_adjust(hspace=0.08, wspace=0.05)

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "fig-gallery.pdf"
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(
        out,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.15,
        transparent=True,
        facecolor="none",
    )
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

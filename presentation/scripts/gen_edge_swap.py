"""Generate a greyscale vector PDF illustrating a single edge-swap (2-switch).

The figure shows a classic 2-switch on a 6-vertex graph:
  - Nodes A,B,C form a triangle with a pendant node F off C; D is below A, E below B.
  - Removed edges (A-B and D-E): bold dashed near-black lines.
  - Added edges  (A-D and B-E): bold solid near-black lines.
  - Kept edges   (A-C, B-C, C-F): light-grey, normal weight.

No text labels appear in the figure. Visual encoding only:
  dashed dark = removed, solid dark = added.

Output: presentation/figures/refine/edge-swap-cycle.pdf
"""

from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.rcParams["pdf.fonttype"] = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIGURES = Path(__file__).resolve().parents[1] / "figures"
OUT_PATH = FIGURES / "refine" / "edge-swap-cycle.pdf"

# ---------------------------------------------------------------------------
# Palette  (matches gen_opening_figures.py exactly)
# ---------------------------------------------------------------------------

NODE_BASE: str = "#6b6b6b"
EDGE_BASE: str = "#aaaaaa"  # kept / normal edges
EDGE_HI: str = "#1a1a1a"  # swap edges (removed = dashed, added = solid)

NODE_SIZE: int = 275
EDGE_WIDTH_BASE: float = 1.8
EDGE_WIDTH_HI: float = 2.8

FIG_W: float = 5.0  # wide enough for two-panel reading
FIG_H: float = 3.4

# ---------------------------------------------------------------------------
# Graph structure  (same 6-vertex graph as the thesis figure)
#
# Layout rationale: A and D share x=0 (vertical new edge A-D is clean);
# B and E share x=2 (vertical new edge B-E is clean); C/F at intermediate x.
# ---------------------------------------------------------------------------

POSITIONS: dict[str, tuple[float, float]] = {
    "A": (0.0, 2.0),
    "B": (2.0, 2.0),
    "C": (1.0, 0.7),
    "D": (0.0, -0.7),
    "E": (2.0, -0.7),
    "F": (2.9, 0.7),
}

NODES: list[str] = ["A", "B", "C", "D", "E", "F"]

# Edges present in both panels (unchanged by the swap)
KEPT_EDGES: list[tuple[str, str]] = [("A", "C"), ("B", "C"), ("C", "F")]

# The two edges that the 2-switch removes (drawn dashed in the before-panel)
REMOVED_EDGES: list[tuple[str, str]] = [("A", "B"), ("D", "E")]

# The two edges that the 2-switch adds (drawn solid in the after-panel)
ADDED_EDGES: list[tuple[str, str]] = [("A", "D"), ("B", "E")]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _xlim_ylim(ax: Axes) -> None:
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(-0.6, 3.5)
    _ = ax.set_ylim(-1.3, 2.5)


def _draw_nodes(g: nx.Graph[str], ax: Axes) -> None:
    _ = nx.draw_networkx_nodes(
        g,
        POSITIONS,
        ax=ax,
        node_size=NODE_SIZE,
        node_color=NODE_BASE,
    )


def draw_before(ax: Axes) -> None:
    """Left panel: graph with the two edges that will be removed (dashed dark)."""
    g: nx.Graph[str] = nx.Graph()
    _ = g.add_nodes_from(NODES)
    _ = g.add_edges_from(KEPT_EDGES)
    _ = g.add_edges_from(REMOVED_EDGES)

    _ = nx.draw_networkx_edges(
        g,
        POSITIONS,
        ax=ax,
        edgelist=KEPT_EDGES,
        edge_color=EDGE_BASE,
        width=EDGE_WIDTH_BASE,
    )
    _ = nx.draw_networkx_edges(
        g,
        POSITIONS,
        ax=ax,
        edgelist=REMOVED_EDGES,
        edge_color=EDGE_HI,
        width=EDGE_WIDTH_HI,
        style="dashed",
    )
    _draw_nodes(g, ax)
    _xlim_ylim(ax)


def draw_after(ax: Axes) -> None:
    """Right panel: graph after the swap — new edges drawn solid dark."""
    g: nx.Graph[str] = nx.Graph()
    _ = g.add_nodes_from(NODES)
    _ = g.add_edges_from(KEPT_EDGES)
    _ = g.add_edges_from(ADDED_EDGES)

    _ = nx.draw_networkx_edges(
        g,
        POSITIONS,
        ax=ax,
        edgelist=KEPT_EDGES,
        edge_color=EDGE_BASE,
        width=EDGE_WIDTH_BASE,
    )
    _ = nx.draw_networkx_edges(
        g,
        POSITIONS,
        ax=ax,
        edgelist=ADDED_EDGES,
        edge_color=EDGE_HI,
        width=EDGE_WIDTH_HI,
        style="solid",
    )
    _draw_nodes(g, ax)
    _xlim_ylim(ax)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    matplotlib.rcParams["figure.facecolor"] = "none"
    matplotlib.rcParams["axes.facecolor"] = "none"

    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=(FIG_W, FIG_H), facecolor="none")

    ax_before: Axes = fig.add_subplot(1, 2, 1)
    ax_after: Axes = fig.add_subplot(1, 2, 2)

    draw_before(ax_before)
    draw_after(ax_after)

    fig.tight_layout(pad=0.4)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(
        OUT_PATH,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.08,
        transparent=True,
        facecolor="none",
    )
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()

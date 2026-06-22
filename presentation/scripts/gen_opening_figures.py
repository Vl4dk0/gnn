"""Generate the three opening concept figures for the defense presentation.

Produces three vector PDFs into presentation/figures/:

  fig-degree.pdf  — the prism graph (3-regular, 6 vertices), one vertex and its
                    3 incident edges highlighted in near-black.
  fig-girth.pdf   — K_{2,3} (bipartite, girth 4, 5 vertices), the shortest
                    4-cycle highlighted in near-black; remaining elements in
                    mid-grey.
  fig-cage.pdf    — the Petersen graph in its classic outer-pentagon /
                    inner-pentagram layout.

Design: greyscale only, no title text, no axis labels, white background.
Output is vector PDF (sharp at any width); all layouts are deterministic.
"""

import math
from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Embed TrueType fonts in the PDF (type 42) so text stays selectable and sharp.
matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --- Palette (greyscale graph drawings; deck chrome stays grey). Shared
#     meaning across every graph figure: base vertices = mid grey, highlighted
#     vertices / edges / cycles = near-black, ordinary edges = light grey. ---
BG = "#ffffff"
NODE_BASE = "#6b6b6b"
EDGE_BASE = "#aaaaaa"
NODE_HI = "#1a1a1a"
EDGE_HI = "#1a1a1a"
LABEL_ON_BASE = "#f0f0f0"
LABEL_ON_HI = "#f0f0f0"

# --- Consistent sizing across all three figures ---
NODE_SIZE = 260
# Petersen-family node size: shared by all Petersen figures (plain and numbered)
# so they look consistent side-by-side.
CAGE_NODE_SIZE = 290
EDGE_WIDTH_BASE = 1.8
EDGE_WIDTH_HI = 2.8
FIG_W = 4.0  # inches (single figure, square-ish)
FIG_H = 3.6


# ---------------------------------------------------------------------------
# fig-degree.png  —  prism graph (K_3 □ K_2), 3-regular, 6 vertices
# ---------------------------------------------------------------------------

# Vertices: top triangle 0,1,2 and bottom triangle 3,4,5.
# Prism edges: top ring + bottom ring + three vertical rungs.
PRISM_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 0),
    (3, 4),
    (4, 5),
    (5, 3),
    (0, 3),
    (1, 4),
    (2, 5),
]

# Fixed positions: top triangle rotated so vertex 0 is at the top.
#   top ring at radius 1.0, bottom ring at radius 0.58 (inner).
#   Rotated so top-0 is up, and the layout is wide enough to be uncluttered.
_PRISM_R_OUT = 1.0
_PRISM_R_IN = 0.55


def _angle(i: int, n: int, offset_deg: float = 90.0) -> float:
    return math.radians(offset_deg - i * 360.0 / n)


PRISM_POS: dict[int, tuple[float, float]] = {
    i: (_PRISM_R_OUT * math.cos(_angle(i, 3)), _PRISM_R_OUT * math.sin(_angle(i, 3)))
    for i in range(3)
}
PRISM_POS.update(
    {
        i + 3: (
            _PRISM_R_IN * math.cos(_angle(i, 3)),
            _PRISM_R_IN * math.sin(_angle(i, 3)),
        )
        for i in range(3)
    }
)

# Vertex 0 is the highlighted vertex; its 3 incident edges are (0,1),(2,0),(0,3).
DEGREE_HI_NODE = 0
DEGREE_HI_EDGES: list[tuple[int, int]] = [(0, 1), (2, 0), (0, 3)]
DEGREE_BASE_EDGES = [e for e in PRISM_EDGES if e not in DEGREE_HI_EDGES]


def draw_degree(ax: Axes) -> None:
    g: nx.Graph[int] = nx.Graph()
    g.add_edges_from(PRISM_EDGES)

    node_colors = [NODE_HI if v == DEGREE_HI_NODE else NODE_BASE for v in g.nodes()]

    _ = nx.draw_networkx_edges(
        g,
        PRISM_POS,
        ax=ax,
        edgelist=DEGREE_BASE_EDGES,
        edge_color=EDGE_BASE,
        width=EDGE_WIDTH_BASE,
    )
    _ = nx.draw_networkx_edges(
        g,
        PRISM_POS,
        ax=ax,
        edgelist=DEGREE_HI_EDGES,
        edge_color=EDGE_HI,
        width=EDGE_WIDTH_HI,
    )
    _ = nx.draw_networkx_nodes(
        g,
        PRISM_POS,
        ax=ax,
        node_size=NODE_SIZE,
        node_color=node_colors,
    )

    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(-1.3, 1.3)
    _ = ax.set_ylim(-0.9, 1.3)


# ---------------------------------------------------------------------------
# fig-girth.png  —  same prism layout as fig-degree, but with one edge removed
# from each triangle so no 3-cycle survives. The shortest cycle is then a
# 4-cycle (girth 4), highlighted in near-black.
# ---------------------------------------------------------------------------
#
# Start from the prism (top triangle 0,1,2; bottom triangle 3,4,5; rungs
# 0-3,1-4,2-5). The only 3-cycles are the two triangles. Removing edge (1,2)
# from the top and (4,5) from the bottom destroys both triangles, leaving a
# graph whose girth is 4. The new shortest 4-cycle 0-1-4-3-0 is highlighted.
GIRTH_REMOVED_EDGES: set[tuple[int, int]] = {(1, 2), (4, 5)}
GIRTH_EDGES: list[tuple[int, int]] = [
    e for e in PRISM_EDGES if e not in GIRTH_REMOVED_EDGES
]

GIRTH_POS = PRISM_POS

# Shortest cycle after the removals: 0-1-4-3-0 (length 4).
GIRTH_HI_NODES: set[int] = {0, 1, 4, 3}
GIRTH_HI_EDGES: list[tuple[int, int]] = [(0, 1), (1, 4), (3, 4), (0, 3)]
GIRTH_BASE_EDGES = [e for e in GIRTH_EDGES if e not in GIRTH_HI_EDGES]


def draw_girth(ax: Axes) -> None:
    g: nx.Graph[int] = nx.Graph()
    g.add_edges_from(GIRTH_EDGES)

    node_colors = [NODE_HI if v in GIRTH_HI_NODES else NODE_BASE for v in g.nodes()]

    _ = nx.draw_networkx_edges(
        g,
        GIRTH_POS,
        ax=ax,
        edgelist=GIRTH_BASE_EDGES,
        edge_color=EDGE_BASE,
        width=EDGE_WIDTH_BASE,
    )
    _ = nx.draw_networkx_edges(
        g,
        GIRTH_POS,
        ax=ax,
        edgelist=GIRTH_HI_EDGES,
        edge_color=EDGE_HI,
        width=EDGE_WIDTH_HI,
    )
    _ = nx.draw_networkx_nodes(
        g,
        GIRTH_POS,
        ax=ax,
        node_size=NODE_SIZE,
        node_color=node_colors,
    )

    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(-1.3, 1.3)
    _ = ax.set_ylim(-0.9, 1.3)


# ---------------------------------------------------------------------------
# fig-cage.png  —  Petersen graph in outer-pentagon / inner-pentagram layout
# ---------------------------------------------------------------------------

PETERSEN_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 0),
    (5, 7),
    (7, 9),
    (9, 6),
    (6, 8),
    (8, 5),
    (0, 5),
    (1, 6),
    (2, 7),
    (3, 8),
    (4, 9),
]


def petersen_pos() -> dict[int, tuple[float, float]]:
    """Outer-pentagon (0..4) + inner-pentagram (5..9), vertex 0 at top."""
    pos: dict[int, tuple[float, float]] = {}
    for i in range(5):
        a = math.radians(90.0 - i * 72.0)
        pos[i] = (math.cos(a), math.sin(a))
        pos[i + 5] = (0.45 * math.cos(a), 0.45 * math.sin(a))
    return pos


PETERSEN_POS = petersen_pos()


def draw_cage(ax: Axes) -> None:
    g: nx.Graph[int] = nx.Graph()
    g.add_edges_from(PETERSEN_EDGES)

    _ = nx.draw_networkx_edges(
        g,
        PETERSEN_POS,
        ax=ax,
        edgelist=PETERSEN_EDGES,
        edge_color=EDGE_BASE,
        width=EDGE_WIDTH_BASE,
    )
    _ = nx.draw_networkx_nodes(
        g,
        PETERSEN_POS,
        ax=ax,
        node_size=CAGE_NODE_SIZE,
        node_color=NODE_BASE,
    )

    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(-1.25, 1.25)
    _ = ax.set_ylim(-1.25, 1.25)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def save(fig: Figure, name: str) -> None:
    # Graph drawings (node-link) are saved with no background (transparent PDF).
    (FIGURES / name).parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(
        FIGURES / name,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.08,
        transparent=True,
        facecolor="none",
    )
    print(f"saved {FIGURES / name}")


def make_fig() -> tuple[Figure, Axes]:
    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=(FIG_W, FIG_H), facecolor=BG)
    ax: Axes = fig.add_subplot(1, 1, 1)
    _ = ax.set_facecolor(BG)
    return fig, ax


def verify_girth() -> None:
    """Assert the fig-girth graph has no 3-cycle and girth exactly 4."""
    g: nx.Graph[int] = nx.Graph()
    g.add_edges_from(GIRTH_EDGES)
    triangles = [c for c in nx.cycle_basis(g) if len(c) == 3]
    shortest = min(len(c) for c in nx.minimum_cycle_basis(g))
    assert not any(
        len({a, b, c_}) == 3
        and g.has_edge(a, b)
        and g.has_edge(b, c_)
        and g.has_edge(a, c_)
        for a in g.nodes()
        for b in g.nodes()
        for c_ in g.nodes()
    ), "fig-girth still contains a 3-cycle"
    assert not triangles, f"unexpected 3-cycle in cycle basis: {triangles}"
    assert shortest == 4, f"fig-girth girth is {shortest}, expected 4"
    print(f"fig-girth verified: no 3-cycle, girth = {shortest}")


# ---------------------------------------------------------------------------
# petersen-base.pdf  —  plain Petersen graph, all greyscale
# ---------------------------------------------------------------------------


def draw_petersen_base(ax: Axes) -> None:
    g: nx.Graph[int] = nx.Graph()
    g.add_edges_from(PETERSEN_EDGES)

    _ = nx.draw_networkx_edges(
        g,
        PETERSEN_POS,
        ax=ax,
        edgelist=PETERSEN_EDGES,
        edge_color=EDGE_BASE,
        width=EDGE_WIDTH_BASE,
    )
    _ = nx.draw_networkx_nodes(
        g,
        PETERSEN_POS,
        ax=ax,
        node_size=CAGE_NODE_SIZE,
        node_color=NODE_BASE,
    )

    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(-1.25, 1.25)
    _ = ax.set_ylim(-1.25, 1.25)


# ---------------------------------------------------------------------------
# petersen-degree.pdf  —  vertex 0 and its 3 incident edges highlighted
# ---------------------------------------------------------------------------

PETERSEN_DEGREE_HI_NODE = 0
PETERSEN_DEGREE_HI_EDGES: list[tuple[int, int]] = [(0, 1), (4, 0), (0, 5)]
PETERSEN_DEGREE_BASE_EDGES = [
    e for e in PETERSEN_EDGES if e not in PETERSEN_DEGREE_HI_EDGES
]


def draw_petersen_degree(ax: Axes) -> None:
    g: nx.Graph[int] = nx.Graph()
    g.add_edges_from(PETERSEN_EDGES)

    node_colors = [
        NODE_HI if v == PETERSEN_DEGREE_HI_NODE else NODE_BASE for v in g.nodes()
    ]

    _ = nx.draw_networkx_edges(
        g,
        PETERSEN_POS,
        ax=ax,
        edgelist=PETERSEN_DEGREE_BASE_EDGES,
        edge_color=EDGE_BASE,
        width=EDGE_WIDTH_BASE,
    )
    _ = nx.draw_networkx_edges(
        g,
        PETERSEN_POS,
        ax=ax,
        edgelist=PETERSEN_DEGREE_HI_EDGES,
        edge_color=EDGE_HI,
        width=EDGE_WIDTH_HI,
    )
    _ = nx.draw_networkx_nodes(
        g,
        PETERSEN_POS,
        ax=ax,
        node_size=CAGE_NODE_SIZE,
        node_color=node_colors,
    )

    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(-1.25, 1.25)
    _ = ax.set_ylim(-1.25, 1.25)


# ---------------------------------------------------------------------------
# petersen-girth.pdf  —  outer 5-cycle 0-1-2-3-4-0 highlighted
# ---------------------------------------------------------------------------

PETERSEN_GIRTH_HI_NODES: set[int] = {0, 1, 2, 3, 4}
PETERSEN_GIRTH_HI_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 0),
]
PETERSEN_GIRTH_BASE_EDGES = [
    e for e in PETERSEN_EDGES if e not in PETERSEN_GIRTH_HI_EDGES
]


def draw_petersen_girth(ax: Axes) -> None:
    g: nx.Graph[int] = nx.Graph()
    g.add_edges_from(PETERSEN_EDGES)

    node_colors = [
        NODE_HI if v in PETERSEN_GIRTH_HI_NODES else NODE_BASE for v in g.nodes()
    ]

    _ = nx.draw_networkx_edges(
        g,
        PETERSEN_POS,
        ax=ax,
        edgelist=PETERSEN_GIRTH_BASE_EDGES,
        edge_color=EDGE_BASE,
        width=EDGE_WIDTH_BASE,
    )
    _ = nx.draw_networkx_edges(
        g,
        PETERSEN_POS,
        ax=ax,
        edgelist=PETERSEN_GIRTH_HI_EDGES,
        edge_color=EDGE_HI,
        width=EDGE_WIDTH_HI,
    )
    _ = nx.draw_networkx_nodes(
        g,
        PETERSEN_POS,
        ax=ax,
        node_size=CAGE_NODE_SIZE,
        node_color=node_colors,
    )

    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(-1.25, 1.25)
    _ = ax.set_ylim(-1.25, 1.25)


# ---------------------------------------------------------------------------
# petersen-numbered.pdf  —  Petersen graph with vertices numbered 1..10
# ---------------------------------------------------------------------------

# Use the shared Petersen node size; numbers still fit with a slight font reduction.
PETERSEN_NUMBERED_NODE_SIZE = CAGE_NODE_SIZE


def draw_petersen_numbered(ax: Axes) -> None:
    g: nx.Graph[int] = nx.Graph()
    g.add_edges_from(PETERSEN_EDGES)

    _ = nx.draw_networkx_edges(
        g,
        PETERSEN_POS,
        ax=ax,
        edgelist=PETERSEN_EDGES,
        edge_color=EDGE_BASE,
        width=EDGE_WIDTH_BASE,
    )
    _ = nx.draw_networkx_nodes(
        g,
        PETERSEN_POS,
        ax=ax,
        node_size=PETERSEN_NUMBERED_NODE_SIZE,
        node_color=NODE_BASE,
    )
    # Labels: internal nx vertex ids are 0..9; display as 1..10.
    labels = {v: str(v + 1) for v in g.nodes()}
    _ = nx.draw_networkx_labels(
        g,
        PETERSEN_POS,
        ax=ax,
        labels=labels,
        font_size=6,
        font_color=LABEL_ON_BASE,
        font_weight="bold",
    )

    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(-1.25, 1.25)
    _ = ax.set_ylim(-1.25, 1.25)


# ---------------------------------------------------------------------------
# dodecahedron-numbered.pdf  —  dodecahedral graph (3-regular, girth 5, 20 v)
#   Layout: the classic crossing-free Schlegel diagram (planar embedding):
#     outer pentagon (5 v)  -> middle ring (10 v)  -> inner pentagon (5 v)
#   Each outer vertex sends one spoke inward to an alternating middle-ring
#   vertex; each remaining middle-ring vertex sends one spoke to the inner
#   pentagon. Coordinates are derived from NetworkX dodecahedral_graph()'s
#   fixed labelling so the drawing is deterministic and has zero edge crossings.
# ---------------------------------------------------------------------------

DODECAHEDRON_NODE_SIZE = CAGE_NODE_SIZE


def dodecahedron_pos() -> dict[int, tuple[float, float]]:
    """Crossing-free nested-pentagon Schlegel layout for the dodecahedron.

    Three concentric shells of NetworkX dodecahedral_graph():
      outer pentagon (5 v)  : 0, 1, 2, 3, 19            (radius 1.00)
      middle ring (10 v)    : 10,9,8,7,6,5,4,17,18,11   (radius 0.62)
      inner pentagon (5 v)  : 13,14,15,16,12            (radius 0.30)
    The middle ring alternates outer-spoke vertices (even index, aligned under
    an outer vertex) and inner-spoke vertices (odd index, aligned over an inner
    vertex). The inner pentagon is rotated so each vertex sits radially beneath
    its middle-ring neighbour, which removes all edge crossings.
    """
    pos: dict[int, tuple[float, float]] = {}

    def place(nodes: list[int], radius: float, rot_rad: float) -> None:
        k = len(nodes)
        for idx, v in enumerate(nodes):
            a = math.pi / 2.0 + rot_rad + 2.0 * math.pi * idx / k
            pos[v] = (radius * math.cos(a), radius * math.sin(a))

    outer = [0, 1, 2, 3, 19]
    middle = [10, 9, 8, 7, 6, 5, 4, 17, 18, 11]
    inner = [13, 14, 15, 16, 12]

    place(outer, 1.00, 0.0)
    # Offset the 10-cycle by half a step so even-index (outer-spoke) vertices
    # tuck just inside each outer vertex.
    place(middle, 0.62, -math.pi / 10.0)
    # Rotate the inner pentagon to line up with the odd-index middle vertices.
    place(inner, 0.30, math.pi / 10.0)
    return pos


def draw_dodecahedron_numbered(ax: Axes) -> None:
    g: nx.Graph[int] = cast("nx.Graph[int]", nx.dodecahedral_graph())
    pos = dodecahedron_pos()

    edges: list[tuple[int, int]] = list(g.edges())
    _ = nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        edgelist=edges,
        edge_color=EDGE_BASE,
        width=EDGE_WIDTH_BASE,
    )
    _ = nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_size=DODECAHEDRON_NODE_SIZE,
        node_color=NODE_BASE,
    )
    labels: dict[int, str] = {v: str(v + 1) for v in g.nodes()}
    _ = nx.draw_networkx_labels(
        g,
        pos,
        ax=ax,
        labels=labels,
        font_size=5.5,
        font_color=LABEL_ON_BASE,
        font_weight="bold",
    )

    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    margin = 0.18
    _ = ax.set_xlim(-1.0 - margin, 1.0 + margin)
    _ = ax.set_ylim(-1.0 - margin, 1.0 + margin)


def main() -> None:
    plt.rcParams["figure.facecolor"] = BG
    plt.rcParams["axes.facecolor"] = BG

    verify_girth()

    fig_d, ax_d = make_fig()
    draw_degree(ax_d)
    save(fig_d, "degree-girth/fig-degree.pdf")
    plt.close(fig_d)

    fig_g, ax_g = make_fig()
    draw_girth(ax_g)
    save(fig_g, "degree-girth/fig-girth.pdf")
    plt.close(fig_g)

    fig_c, ax_c = make_fig()
    draw_cage(ax_c)
    save(fig_c, "_unused/fig-cage.pdf")
    plt.close(fig_c)

    fig_pb, ax_pb = make_fig()
    draw_petersen_base(ax_pb)
    save(fig_pb, "petersen/petersen-base.pdf")
    plt.close(fig_pb)

    fig_pd, ax_pd = make_fig()
    draw_petersen_degree(ax_pd)
    save(fig_pd, "petersen/petersen-degree.pdf")
    plt.close(fig_pd)

    fig_pg, ax_pg = make_fig()
    draw_petersen_girth(ax_pg)
    save(fig_pg, "petersen/petersen-girth.pdf")
    plt.close(fig_pg)

    fig_pn, ax_pn = make_fig()
    draw_petersen_numbered(ax_pn)
    save(fig_pn, "petersen/petersen-numbered.pdf")
    plt.close(fig_pn)

    fig_dn, ax_dn = make_fig()
    draw_dodecahedron_numbered(ax_dn)
    save(fig_dn, "petersen/dodecahedron-numbered.pdf")
    plt.close(fig_dn)


if __name__ == "__main__":
    main()

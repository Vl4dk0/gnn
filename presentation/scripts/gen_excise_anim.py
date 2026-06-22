"""Generate the excision step figures for the defense deck.

Self-contained (matplotlib + networkx only). Produces a small set of CLEAN,
individually-centered, transparent greyscale vector PDFs into
presentation/figures/excision/, each meant to occupy one cell of a 6-cell grid
on a single slide.

The worked example (the (3, 5) case, same as the frontend /excise demo):

  Start graph = the dodecahedron (20 vertices, 3-regular, girth 5).
  k = 3, g = 5, so the excision depth is the Moore radius
    d = floor((g - 1) / 2) = 2.
  Root the breadth-first tree at vertex 0 and grow it to depth 2: this is a tree
  of 1 + 3 + 6 = 10 vertices, the ball {0,1,2,3,8,9,10,11,18,19}. Delete it. The
  six survivors that touched the tree, {4,6,7,12,13,17}, fall from degree 3 to
  degree 2 (the deficient vertices). Stitch them back to degree 3 with the three
  girth-safe edges (4,13), (6,12), (7,17): each joins a pair at distance >= g - 1
  in the reduced graph, so no cycle shorter than g is created. The result is a
  10-vertex 3-regular girth-5 graph isomorphic to the Petersen graph, the
  (3, 5)-cage. The Moore bound for (3, 5) is 10, so it cannot shrink further.

Layouts reused verbatim from gen_opening_figures.py: the crossing-free
nested-pentagon (Schlegel) dodecahedron layout and the canonical Petersen
outer-pentagon / inner-pentagram layout. Every figure is drawn with equal aspect
and symmetric x/y limits so the graph sits centered in its canvas, with a
consistent node size (CAGE_NODE_SIZE) across all five.

Figures (all transparent greyscale PDFs, no titles/captions):

  exc-dodeca.pdf    plain dodecahedron (the starting (3, 5)-graph), all grey.
  exc-bfs.pdf       same dodecahedron, the depth-2 BFS tree from root 0
                    highlighted (tree vertices dark, tree edges bold dark, root
                    ringed); everything else grey.
  exc-removed.pdf   the 10 survivors after the tree is cut, in their original
                    positions; the 6 deficient vertices ringed dark.
  exc-stitched.pdf  the survivors with the 3 new stitch edges added bold dark.
  exc-petersen.pdf  the result redrawn cleanly as the canonical Petersen graph.
"""

import math
from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Embed TrueType fonts in the PDF (type 42) so any text stays selectable/sharp.
matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures" / "excision"

# --- Palette (shared with gen_opening_figures.py) ----------------------------
BG = "#ffffff"
NODE_BASE = "#6b6b6b"
EDGE_BASE = "#aaaaaa"
NODE_HI = "#1a1a1a"
EDGE_HI = "#1a1a1a"

# --- Consistent sizing across all five figures -------------------------------
CAGE_NODE_SIZE = 290
EDGE_WIDTH_BASE = 1.8
EDGE_WIDTH_HI = 2.8
FIG_W = 4.0  # inches (single figure, square-ish)
FIG_H = 3.6

# Outline widths for ringed (root / deficient) vertices and their ring colour.
RING_LW = 2.6
RING_COLOR = NODE_HI

K = 3
G_TARGET = 5
DEPTH = (G_TARGET - 1) // 2  # Moore radius d = floor((g - 1) / 2) = 2

# --- The excision example (root 0 on the dodecahedron, verified at runtime) ---
ROOT = 0
REMOVED: set[int] = {0, 1, 2, 3, 8, 9, 10, 11, 18, 19}
DEFICIENT: set[int] = {4, 6, 7, 12, 13, 17}
STITCH_EDGES: list[tuple[int, int]] = [(4, 13), (6, 12), (7, 17)]

# --- Canonical Petersen edges (matches nx.petersen_graph()) ------------------
PETERSEN_EDGES: list[tuple[int, int]] = [
    (0, 1), (0, 4), (0, 5), (1, 2), (1, 6), (2, 3), (2, 7), (3, 4),
    (3, 8), (4, 9), (5, 7), (5, 8), (6, 8), (6, 9), (7, 9),
]  # fmt: skip


DODECAHEDRAL_EDGES: list[tuple[int, int]] = [
    (0, 1), (0, 10), (0, 19), (1, 2), (1, 8), (2, 3), (2, 6), (3, 4),
    (3, 19), (4, 5), (4, 17), (5, 6), (5, 15), (6, 7), (7, 8), (7, 14),
    (8, 9), (9, 10), (9, 13), (10, 11), (11, 12), (11, 18), (12, 13),
    (12, 16), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19),
]  # fmt: skip


def dodecahedral_graph() -> "nx.Graph[int]":
    graph: "nx.Graph[int]" = nx.Graph()
    graph.add_edges_from(DODECAHEDRAL_EDGES)
    return graph


def petersen_graph() -> "nx.Graph[int]":
    graph: "nx.Graph[int]" = nx.Graph()
    graph.add_edges_from(PETERSEN_EDGES)
    return graph


# --- Crossing-free nested-pentagon (Schlegel) dodecahedron layout ------------
# Reused verbatim from gen_opening_figures.dodecahedron_pos().
def dodecahedron_pos() -> dict[int, tuple[float, float]]:
    """Crossing-free nested-pentagon Schlegel layout for the dodecahedron."""
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
    place(middle, 0.62, -math.pi / 10.0)
    place(inner, 0.30, math.pi / 10.0)
    return pos


def petersen_pos() -> dict[int, tuple[float, float]]:
    """Outer-pentagon (0..4) + inner-pentagram (5..9), vertex 0 at top."""
    pos: dict[int, tuple[float, float]] = {}
    for i in range(5):
        a = math.radians(90.0 - i * 72.0)
        pos[i] = (math.cos(a), math.sin(a))
        pos[i + 5] = (0.45 * math.cos(a), 0.45 * math.sin(a))
    return pos


DODECA_POS = dodecahedron_pos()
PETERSEN_POS = petersen_pos()

# Symmetric limits so each graph sits centered in its canvas (cf. opening figs).
DODECA_LIM = 1.20
PETERSEN_LIM = 1.25


# --- BFS tree from the root --------------------------------------------------
def bfs_tree_edges(
    graph: "nx.Graph[int]", root: int, depth: int
) -> list[tuple[int, int]]:
    """Parent-link edges of the BFS tree grown from root to the given depth
    (same-level horizontal edges excluded)."""
    level = {root: 0}
    tree: list[tuple[int, int]] = []
    frontier = [root]
    for _ in range(depth):
        nxt: list[int] = []
        for u in frontier:
            for w in graph.neighbors(u):
                if w not in level:
                    level[w] = level[u] + 1
                    tree.append((u, w))
                    nxt.append(w)
        frontier = nxt
    return tree


# --- Drawing scaffolding -----------------------------------------------------
def make_fig(lim: float) -> tuple[Figure, Axes]:
    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=(FIG_W, FIG_H), facecolor=BG)
    ax: Axes = fig.add_subplot(1, 1, 1)
    _ = ax.set_facecolor(BG)
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(-lim, lim)
    _ = ax.set_ylim(-lim, lim)
    return fig, ax


def save(fig: Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
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
    plt.close(fig)


def draw_graph(
    ax: Axes,
    graph: "nx.Graph[int]",
    pos: dict[int, tuple[float, float]],
    *,
    base_edges: list[tuple[int, int]],
    hi_edges: list[tuple[int, int]],
    hi_nodes: set[int],
    ringed_nodes: set[int],
) -> None:
    """Draw a graph in the shared greyscale style.

    base_edges  ordinary edges, light grey.
    hi_edges    highlighted edges (tree / stitch), bold dark.
    hi_nodes    vertices filled dark (NODE_HI); the rest filled mid grey.
    ringed_nodes  vertices drawn with a dark ring (root / deficient markers).
    """
    if base_edges:
        _ = nx.draw_networkx_edges(
            graph, pos, ax=ax,
            edgelist=base_edges, edge_color=EDGE_BASE, width=EDGE_WIDTH_BASE,
        )  # fmt: skip
    if hi_edges:
        _ = nx.draw_networkx_edges(
            graph, pos, ax=ax,
            edgelist=hi_edges, edge_color=EDGE_HI, width=EDGE_WIDTH_HI,
        )  # fmt: skip

    nodes = list(graph.nodes())
    node_colors = [NODE_HI if v in hi_nodes else NODE_BASE for v in nodes]
    edge_colors = [
        RING_COLOR if v in ringed_nodes else node_colors[i] for i, v in enumerate(nodes)
    ]
    line_widths = [RING_LW if v in ringed_nodes else 0.0 for v in nodes]
    _ = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        nodelist=nodes,
        node_size=CAGE_NODE_SIZE,
        node_color=node_colors,
        edgecolors=edge_colors,
        linewidths=line_widths,
    )


# --- The five figures --------------------------------------------------------
def fig_dodeca() -> None:
    """exc-dodeca.pdf: plain dodecahedron, all grey."""
    g = dodecahedral_graph()
    fig, ax = make_fig(DODECA_LIM)
    draw_graph(
        ax, g, DODECA_POS,
        base_edges=list(g.edges()), hi_edges=[],
        hi_nodes=set(), ringed_nodes=set(),
    )  # fmt: skip
    save(fig, "exc-dodeca.pdf")


def fig_bfs() -> None:
    """exc-bfs.pdf: depth-2 BFS tree from root 0 highlighted on the dodecahedron."""
    g = dodecahedral_graph()
    tree = bfs_tree_edges(g, ROOT, DEPTH)
    tree_set = {frozenset(e) for e in tree}
    base_edges = [e for e in g.edges() if frozenset(e) not in tree_set]
    fig, ax = make_fig(DODECA_LIM)
    draw_graph(
        ax, g, DODECA_POS,
        base_edges=base_edges, hi_edges=tree,
        hi_nodes=REMOVED, ringed_nodes={ROOT},
    )  # fmt: skip
    save(fig, "exc-bfs.pdf")


def survivor_graph() -> "nx.Graph[int]":
    """Dodecahedron with the removed BFS ball deleted (original vertex ids)."""
    g = dodecahedral_graph()
    g.remove_nodes_from(REMOVED)
    return g


def centered_pos(graph: "nx.Graph[int]") -> dict[int, tuple[float, float]]:
    """Survivor positions in the original layout, recentered on their bounding
    box so the cut-down graph sits in the middle of its canvas (the full
    dodecahedron is symmetric, but a survivor subgraph is not)."""
    nodes = list(graph.nodes())
    xs = [DODECA_POS[v][0] for v in nodes]
    ys = [DODECA_POS[v][1] for v in nodes]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    return {v: (DODECA_POS[v][0] - cx, DODECA_POS[v][1] - cy) for v in nodes}


# Uniform padding (fraction of the survivors' half-extent) so the cut-down
# graph fills its canvas at the same visual scale as the other figures. Matched
# to the Petersen figure's headroom (lim 1.25 over node half-extent 1.0) so the
# 10-vertex survivor graph fills and its node markers look the same size.
SURVIVOR_MARGIN = 0.25


def survivor_lim(pos: dict[int, tuple[float, float]]) -> float:
    """Symmetric half-extent that tightly frames the recentered survivors with a
    small uniform margin, so they fill the canvas like the dodecahedron does."""
    half = max(max(abs(x), abs(y)) for x, y in pos.values())
    return half * (1.0 + SURVIVOR_MARGIN)


def fig_removed() -> None:
    """exc-removed.pdf: survivors after the cut, deficient vertices ringed."""
    g = survivor_graph()
    pos = centered_pos(g)
    fig, ax = make_fig(survivor_lim(pos))
    draw_graph(
        ax, g, pos,
        base_edges=list(g.edges()), hi_edges=[],
        hi_nodes=set(), ringed_nodes=DEFICIENT,
    )  # fmt: skip
    save(fig, "exc-removed.pdf")


def fig_stitched() -> None:
    """exc-stitched.pdf: survivors with the 3 stitch edges added, bold dark."""
    g = survivor_graph()
    g.add_edges_from(STITCH_EDGES)
    stitch_set = {frozenset(e) for e in STITCH_EDGES}
    base_edges = [e for e in g.edges() if frozenset(e) not in stitch_set]
    pos = centered_pos(g)
    fig, ax = make_fig(survivor_lim(pos))
    draw_graph(
        ax, g, pos,
        base_edges=base_edges, hi_edges=STITCH_EDGES,
        hi_nodes=set(), ringed_nodes=DEFICIENT,
    )  # fmt: skip
    save(fig, "exc-stitched.pdf")


def stitched_graph() -> "nx.Graph[int]":
    g = survivor_graph()
    g.add_edges_from(STITCH_EDGES)
    return g


def fig_petersen() -> None:
    """exc-petersen.pdf: the stitched result redrawn as the canonical Petersen."""
    g = stitched_graph()
    petersen = petersen_graph()
    matcher = nx.algorithms.isomorphism.GraphMatcher(g, petersen)
    assert matcher.is_isomorphic(), "stitched graph is not isomorphic to Petersen"
    mapping = cast(dict[int, int], matcher.mapping)
    pos = {v: PETERSEN_POS[mapping[v]] for v in g.nodes()}
    fig, ax = make_fig(PETERSEN_LIM)
    draw_graph(
        ax, g, pos,
        base_edges=list(g.edges()), hi_edges=[],
        hi_nodes=set(), ringed_nodes=set(),
    )  # fmt: skip
    save(fig, "exc-petersen.pdf")


# --- Verification ------------------------------------------------------------
def verify() -> None:
    """Assert the dodecahedron is 3-regular girth 5 and the stitched result is
    isomorphic to the Petersen graph, with the prompt's exact removed/deficient
    sets and stitch edges."""
    g = dodecahedral_graph()
    assert all(d == K for _, d in g.degree()), "dodecahedron not 3-regular"
    girth = min(len(c) for c in nx.minimum_cycle_basis(g))
    assert girth == G_TARGET, f"dodecahedron girth {girth}, expected {G_TARGET}"

    # The removed ball is exactly the depth-2 BFS ball from the root.
    reach = cast(
        dict[int, int],
        nx.single_source_shortest_path_length(g, ROOT, cutoff=DEPTH),
    )
    assert set(reach) == REMOVED, f"removed ball mismatch: {sorted(reach)}"

    surv = survivor_graph()
    deficient = {v for v in surv.nodes() if surv.degree(v) < K}
    assert deficient == DEFICIENT, f"deficient mismatch: {sorted(deficient)}"

    stitched = stitched_graph()
    degs = {stitched.degree(v) for v in stitched.nodes()}
    assert degs == {K}, f"stitched graph not {K}-regular: degrees {degs}"
    sg = min(len(c) for c in nx.minimum_cycle_basis(stitched))
    assert sg == G_TARGET, f"stitched graph girth {sg}, expected {G_TARGET}"
    iso = nx.is_isomorphic(stitched, petersen_graph())
    assert iso, "stitched graph not isomorphic to Petersen"
    msg = (
        f"VERIFIED: dodecahedron 3-regular girth {girth}; stitched result "
        + f"10-vertex 3-regular girth {sg}, Petersen={iso}; depth {DEPTH}"
    )
    print(msg)


def main() -> None:
    plt.rcParams["figure.facecolor"] = BG
    plt.rcParams["axes.facecolor"] = BG

    verify()

    fig_dodeca()
    fig_bfs()
    fig_removed()
    fig_stitched()
    fig_petersen()


if __name__ == "__main__":
    main()

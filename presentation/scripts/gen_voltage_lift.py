"""Generate the voltage-lift -> Petersen build-up figures for the defense deck.

Self-contained (matplotlib + networkx for the isomorphism check, no ai.cage
import). Produces transparent greyscale vector PDFs into
presentation/figures/voltage/.

The exact example ("Two loops + Z5 -> Petersen"):
  Base voltage graph B: group Gamma = Z5, two base vertices 0 and 1, three
  "edges":
    - self-loop on 0, voltage 1
    - self-loop on 1, voltage 2
    - connecting arc 0 -> 1, voltage 0
  Lift: each base vertex v has 5 copies (v, g) for g in 0..4. A base edge u->w
  with voltage a creates, for every g in Z5, the edge (u, g)-(w, (g + a) % 5).
    - self-loop on 0, voltage 1: (0,g)-(0,g+1)  -> outer pentagon
    - self-loop on 1, voltage 2: (1,g)-(1,g+2)  -> inner pentagram
    - arc 0->1,     voltage 0: (0,g)-(1,g)      -> 5 spokes
  Result: 10 vertices, 15 edges, 3-regular, girth 5 = the Petersen graph.

Figures produced:
  vlift-base.pdf      — the 2-vertex base voltage graph, drawn cleanly.
  vlift-labeled.pdf   — the full Petersen lift, every vertex labelled inside.
  vlift-step1.pdf     — two-panel: lift with outer pentagon added | base edge 1.
  vlift-step2.pdf     — two-panel: lift with inner pentagram added | base edge 2.
  vlift-step3.pdf     — two-panel: lift with spokes added (full Petersen) | edge 3.
"""

from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle, FancyArrowPatch

# Embed TrueType fonts (type 42) so text stays selectable and sharp.
matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --- Greyscale palette (shared deck graph palette) ---
BG = "#ffffff"
NODE_BASE = "#6b6b6b"
EDGE_BASE = "#aaaaaa"
NODE_HI = "#1a1a1a"
EDGE_HI = "#1a1a1a"
LABEL_ON_BASE = "#f0f0f0"
VOLT = "#3a3a3a"  # voltage labels on the base graph

# Edge shades for the step composites.
NEW_EDGE = EDGE_HI  # newly added edges this step (bold, dark)
OLD_EDGE = EDGE_BASE  # previously added edges (faded grey)

N = 5  # group Z5

NEW_LW = 3.2  # bold new edges
OLD_LW = 1.8  # faded old edges
CAGE_NODE_SIZE = 290  # shared Petersen-family node size (matplotlib scatter units)
# Slightly larger nodes for the labelled lift so "0,0" fits inside.
LIFT_NODE_SIZE = 560
# Bigger still for the step composites, where the Petersen panel dominates.
STEP_NODE_SIZE = 1100
STEP_NEW_LW = 4.4  # bold new edges in the (larger) step composites
STEP_OLD_LW = 2.4  # faded old edges in the step composites

LiftVertex = tuple[int, int]
LiftEdge = tuple[LiftVertex, LiftVertex]


# ---------------------------------------------------------------------------
# Save / figure helpers (transparent vector PDF)
# ---------------------------------------------------------------------------


def save(fig: Figure, name: str) -> None:
    (FIGURES / name).parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(
        FIGURES / name,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.06,
        transparent=True,
        facecolor="none",
    )
    print(f"saved {FIGURES / name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Canonical Petersen layout for the lift
#   (0, g) on an OUTER pentagon, (1, g) on an INNER pentagram, radially aligned.
#   g = 0..4, vertex (0,0) at the top, going clockwise to match petersen_pos().
# ---------------------------------------------------------------------------

OUTER_R = 1.0
INNER_R = 0.45


def angle(g: int) -> float:
    return np.deg2rad(90.0 - 72.0 * g)


def lift_pos() -> dict[LiftVertex, tuple[float, float]]:
    pos: dict[LiftVertex, tuple[float, float]] = {}
    for g in range(N):
        a = angle(g)
        pos[(0, g)] = (OUTER_R * float(np.cos(a)), OUTER_R * float(np.sin(a)))
        pos[(1, g)] = (INNER_R * float(np.cos(a)), INNER_R * float(np.sin(a)))
    return pos


def outer_edges() -> list[LiftEdge]:
    # self-loop on 0, voltage 1: (0,g)-(0,(g+1)%5)  -> outer pentagon
    return [((0, g), (0, (g + 1) % N)) for g in range(N)]


def inner_edges() -> list[LiftEdge]:
    # self-loop on 1, voltage 2: (1,g)-(1,(g+2)%5)  -> inner pentagram
    return [((1, g), (1, (g + 2) % N)) for g in range(N)]


def spoke_edges() -> list[LiftEdge]:
    # arc 0->1, voltage 0: (0,g)-(1,g)  -> spokes
    return [((0, g), (1, g)) for g in range(N)]


# ---------------------------------------------------------------------------
# Isomorphism verification
# ---------------------------------------------------------------------------


def build_lift_graph() -> "nx.Graph[LiftVertex]":
    g: nx.Graph[LiftVertex] = nx.Graph()
    g.add_edges_from(outer_edges() + inner_edges() + spoke_edges())
    return g


def verify_petersen() -> None:
    g = build_lift_graph()
    assert g.number_of_nodes() == 10, f"lift has {g.number_of_nodes()} nodes, want 10"
    assert g.number_of_edges() == 15, f"lift has {g.number_of_edges()} edges, want 15"
    petersen: nx.Graph[LiftVertex] = cast("nx.Graph[LiftVertex]", nx.petersen_graph())
    assert nx.is_isomorphic(g, petersen), "constructed lift is NOT the Petersen graph"
    print("verified: lift is isomorphic to the Petersen graph (10 v, 15 e)")


# ---------------------------------------------------------------------------
# Lift drawing primitives (labelled-inside nodes, Petersen-family style)
# ---------------------------------------------------------------------------


def _key(v: LiftVertex) -> str:
    """String key 'v,g' used as the networkx node id (also the label text)."""
    return f"{v[0]},{v[1]}"


def _str_pos(
    pos: dict[LiftVertex, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    return {_key(v): xy for v, xy in pos.items()}


def draw_lift_edges(
    ax: Axes,
    pos: dict[LiftVertex, tuple[float, float]],
    edges: list[LiftEdge],
    color: str,
    lw: float,
) -> None:
    if not edges:
        return
    g: nx.Graph[str] = nx.Graph()
    edgelist = [(_key(u), _key(w)) for u, w in edges]
    g.add_edges_from(edgelist)
    _ = nx.draw_networkx_edges(
        g,
        _str_pos(pos),
        ax=ax,
        edgelist=edgelist,
        edge_color=color,
        width=lw,
    )


def draw_lift_nodes(
    ax: Axes,
    pos: dict[LiftVertex, tuple[float, float]],
    node_size: float,
    font_size: float,
) -> None:
    """Draw the 10 lift vertices as filled circles with the label INSIDE."""
    g: nx.Graph[str] = nx.Graph()
    g.add_nodes_from(_key(v) for v in pos)
    spos = _str_pos(pos)
    _ = nx.draw_networkx_nodes(
        g,
        spos,
        ax=ax,
        node_size=node_size,
        node_color=NODE_BASE,
    )
    labels = {_key(v): _key(v) for v in pos}
    _ = nx.draw_networkx_labels(
        g,
        spos,
        ax=ax,
        labels=labels,
        font_size=font_size,
        font_color=LABEL_ON_BASE,
        font_weight="bold",
    )


def setup_lift_ax(ax: Axes) -> None:
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(-1.25, 1.25)
    _ = ax.set_ylim(-1.25, 1.25)


# ---------------------------------------------------------------------------
# vlift-labeled.pdf  —  the full Petersen lift, labelled inside every node
# ---------------------------------------------------------------------------


def gen_labeled() -> None:
    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=(5.0, 5.0), facecolor=BG)
    ax: Axes = fig.add_subplot(1, 1, 1)
    pos = lift_pos()
    all_edges = outer_edges() + inner_edges() + spoke_edges()
    draw_lift_edges(ax, pos, all_edges, EDGE_BASE, 1.8)
    draw_lift_nodes(ax, pos, LIFT_NODE_SIZE, font_size=8.5)
    setup_lift_ax(ax)
    save(fig, "voltage/vlift-labeled.pdf")


# ---------------------------------------------------------------------------
# vlift-nodes.pdf  —  the 10 lift vertices only, no edges. Same layout
#   positions as the step composites' left panel, so the animation is
#   continuous: the nodes appear first, then edges grow between them.
# ---------------------------------------------------------------------------


def gen_nodes() -> None:
    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=(5.4, 5.4), facecolor=BG)
    ax: Axes = fig.add_subplot(1, 1, 1)
    pos = lift_pos()
    # No edges at all: just the 10 labelled circles.
    draw_lift_nodes(ax, pos, LIFT_NODE_SIZE, font_size=9.5)
    setup_lift_ax(ax)
    save(fig, "voltage/vlift-nodes.pdf")


# ---------------------------------------------------------------------------
# vlift-base.pdf  —  the small 2-vertex base voltage graph, drawn cleanly
# ---------------------------------------------------------------------------

BASE_NODE_R = 0.22


def _draw_self_loop(
    ax: Axes,
    node: tuple[float, float],
    direction: tuple[float, float],
    color: str,
    z: int,
) -> None:
    """Draw a clear circular self-loop bulging away from ``node``.

    The loop is a near-full-circle Arc whose centre is offset from the node
    along ``direction`` (a unit vector). A small arrowhead at the loop's
    closing gap shows orientation. The loop diameter is roughly the node
    diameter, so it reads unmistakably as a self-loop.
    """
    dx, dy = direction
    norm = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / norm, dy / norm

    loop_r = BASE_NODE_R * 1.05  # loop radius ~ node radius
    # Centre the loop so its near edge touches the node and it bulges outward.
    cx = node[0] + ux * (BASE_NODE_R + loop_r * 0.65)
    cy = node[1] + uy * (BASE_NODE_R + loop_r * 0.65)

    base_ang = np.degrees(np.arctan2(uy, ux))
    # Draw most of the circle, leaving a small gap on the side facing the node
    # (base_ang + 180) so the arrowhead can sit in the gap.
    gap = 32.0
    theta1 = base_ang + 180.0 + gap / 2.0
    theta2 = base_ang + 180.0 - gap / 2.0 + 360.0
    loop = Arc(
        (cx, cy),
        2.0 * loop_r,
        2.0 * loop_r,
        angle=0.0,
        theta1=theta1,
        theta2=theta2,
        linewidth=2.4,
        color=color,
        zorder=z,
    )
    _ = ax.add_patch(loop)

    # Arrowhead: a tiny tangent segment at the gap edge (theta2 end).
    end_ang = np.radians(theta2)
    ex = cx + loop_r * np.cos(end_ang)
    ey = cy + loop_r * np.sin(end_ang)
    # Tangent direction (clockwise closing toward the gap).
    tx = np.sin(end_ang)
    ty = -np.cos(end_ang)
    head = FancyArrowPatch(
        (ex - tx * 0.06, ey - ty * 0.06),
        (ex + tx * 0.02, ey + ty * 0.02),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.4,
        color=color,
        zorder=z,
    )
    _ = ax.add_patch(head)


def _draw_base_template(
    ax: Axes,
    *,
    highlight: str | None,
    ylim: tuple[float, float] = (-0.95, 0.95),
    show_voltages: bool = True,
) -> None:
    """Draw base graph B. ``highlight`` in {None,'loop0','loop1','arc'}.

    When ``highlight`` is None every edge is drawn dark (standalone base
    figure). Otherwise the named edge is dark and the others are faded grey.
    """
    p0 = (-1.0, 0.0)
    p1 = (1.0, 0.0)

    def col(name: str) -> str:
        if highlight is None:
            return EDGE_HI
        return EDGE_HI if highlight == name else EDGE_BASE

    def vcol(name: str) -> str:
        if highlight is None:
            return VOLT
        return VOLT if highlight == name else EDGE_BASE

    def zb(name: str) -> int:
        return 4 if (highlight == name or highlight is None) else 2

    # Connecting arc 0 -> 1, voltage 0.
    arc = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>" if show_voltages else "-",
        mutation_scale=16,
        shrinkA=26,
        shrinkB=26,
        linewidth=2.4,
        color=col("arc"),
        zorder=zb("arc"),
    )
    _ = ax.add_patch(arc)
    if show_voltages:
        _ = ax.text(
            0.0,
            0.14,
            "0",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color=vcol("arc"),
            zorder=zb("arc"),
        )

    # Self-loop on 0, voltage 1 (bulges up-left, clear of the 0->1 arc).
    _draw_self_loop(ax, p0, (-1.0, 0.45), col("loop0"), zb("loop0"))
    if show_voltages:
        _ = ax.text(
            p0[0] - 0.92,
            p0[1] + 0.34,
            "1",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=vcol("loop0"),
            zorder=zb("loop0"),
        )

    # Self-loop on 1, voltage 2 (bulges up-right, clear of the 0->1 arc).
    _draw_self_loop(ax, p1, (1.0, 0.45), col("loop1"), zb("loop1"))

    if show_voltages:
        _ = ax.text(
            p1[0] + 0.92,
            p1[1] + 0.34,
            "2",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=vcol("loop1"),
            zorder=zb("loop1"),
        )

    # Base nodes (always dark grey so they read as graph vertices).
    for (x, y), lbl in ((p0, "0"), (p1, "1")):
        circ = Circle(
            (x, y),
            BASE_NODE_R,
            facecolor=NODE_BASE,
            edgecolor="none",
            zorder=5,
        )
        _ = ax.add_patch(circ)
        _ = ax.text(
            x,
            y,
            lbl,
            ha="center",
            va="center",
            fontsize=17,
            color=LABEL_ON_BASE,
            fontweight="bold",
            zorder=6,
        )

    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(-2.0, 2.0)
    _ = ax.set_ylim(*ylim)


def gen_base() -> None:
    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=(6.2, 3.0), facecolor=BG)
    ax: Axes = fig.add_subplot(1, 1, 1)
    _draw_base_template(ax, highlight=None)
    save(fig, "voltage/vlift-base.pdf")


def gen_base_no_voltage() -> None:
    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=(6.2, 3.0), facecolor=BG)
    ax: Axes = fig.add_subplot(1, 1, 1)
    _draw_base_template(ax, highlight=None, show_voltages=False)
    save(fig, "voltage/vlift-base-no-voltage.pdf")


# ---------------------------------------------------------------------------
# vlift-step{1,2,3}.pdf  —  two-panel composites (lift-so-far | base edge)
# ---------------------------------------------------------------------------


def gen_step(
    name: str,
    *,
    old: list[LiftEdge],
    new: list[LiftEdge],
    highlight: str,
) -> None:
    fig_fn = cast(Callable[..., Figure], plt.figure)
    # Larger overall figure; the Petersen (left) panel dominates and a clear
    # horizontal gap separates it from the small base-graph (right) panel.
    fig: Figure = fig_fn(figsize=(13.5, 6.2), facecolor=BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.32)
    ax_lift: Axes = fig.add_subplot(gs[0, 0])
    ax_base: Axes = fig.add_subplot(gs[0, 1])

    pos = lift_pos()
    draw_lift_edges(ax_lift, pos, old, OLD_EDGE, STEP_OLD_LW)
    draw_lift_edges(ax_lift, pos, new, NEW_EDGE, STEP_NEW_LW)
    draw_lift_nodes(ax_lift, pos, STEP_NODE_SIZE, font_size=12.0)
    setup_lift_ax(ax_lift)

    # Square-ish ylim so the wide base graph fills (not collapses into) the panel.
    _draw_base_template(ax_base, highlight=highlight, ylim=(-1.9, 1.9))

    save(fig, name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# vlift-concept.pdf  —  text-free "small 3-regular graph -> bigger 3-regular
#   graph" illustration. Concrete example: base = K4 (3-regular), group Z2,
#   voltage 1 on a perfect matching of K4 and 0 on the remaining edges. The
#   Z2 lift has 8 vertices, is 3-regular, simple and connected.
# ---------------------------------------------------------------------------

# K4 vertices 0..3. A perfect matching: {(0,1), (2,3)}; the other four edges
# carry voltage 0.
K4_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
]
K4_MATCHING: set[tuple[int, int]] = {(0, 1), (2, 3)}


def concept_base_graph() -> "nx.Graph[int]":
    g: nx.Graph[int] = nx.Graph()
    g.add_edges_from(K4_EDGES)
    return g


def concept_lift_graph() -> "nx.Graph[int]":
    """Z2 lift of K4 with voltage 1 on the matching, 0 elsewhere.

    Vertex (v, s) for v in 0..3 and s in {0,1} is encoded as v * 2 + s.
    A base edge (u, w) with voltage a yields, for each s in Z2, the edge
    (u, s)-(w, (s + a) % 2).
    """
    g: nx.Graph[int] = nx.Graph()
    g.add_nodes_from(range(8))
    for u, w in K4_EDGES:
        a = 1 if (u, w) in K4_MATCHING else 0
        for s in range(2):
            _ = g.add_edge(u * 2 + s, w * 2 + ((s + a) % 2))
    return g


def verify_concept_lift() -> None:
    g = concept_lift_graph()
    assert g.number_of_nodes() == 8, f"concept lift has {g.number_of_nodes()} nodes"
    assert all(d == 3 for _, d in g.degree()), "concept lift is not 3-regular"
    assert sum(1 for u, w in g.edges() if u == w) == 0, "concept lift has self-loops"
    assert nx.is_connected(g), "concept lift is not connected"
    # K4 itself is 3-regular and simple (sanity).
    base = concept_base_graph()
    assert all(d == 3 for _, d in base.degree()), "base K4 is not 3-regular"
    print("verified: concept lift has 8 vertices, 3-regular, simple, connected")


def _draw_plain_graph(
    ax: Axes,
    g: "nx.Graph[int]",
    pos: dict[int, tuple[float, float]],
) -> None:
    _ = nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        edge_color=EDGE_BASE,
        width=1.8,
    )
    _ = nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_size=CAGE_NODE_SIZE,
        node_color=NODE_BASE,
    )
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")


def gen_concept() -> None:
    verify_concept_lift()

    base = concept_base_graph()
    lift = concept_lift_graph()

    # Base K4 laid out as a square (deterministic, clearly 3-regular).
    base_pos: dict[int, tuple[float, float]] = {
        0: (-0.7, 0.7),
        1: (0.7, 0.7),
        2: (0.7, -0.7),
        3: (-0.7, -0.7),
    }
    # Lift on a circular layout: clean, symmetric, no crossings emphasised.
    raw = cast(
        dict[int, tuple[float, float]],
        nx.circular_layout(lift),
    )
    lift_pos_concept: dict[int, tuple[float, float]] = {
        v: (float(xy[0]), float(xy[1])) for v, xy in raw.items()
    }

    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=(9.2, 4.4), facecolor=BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.45, 1.25], wspace=0.05)
    ax_base: Axes = fig.add_subplot(gs[0, 0])
    ax_arrow: Axes = fig.add_subplot(gs[0, 1])
    ax_lift: Axes = fig.add_subplot(gs[0, 2])

    _draw_plain_graph(ax_base, base, base_pos)
    _ = ax_base.set_xlim(-1.1, 1.1)
    _ = ax_base.set_ylim(-1.1, 1.1)

    _draw_plain_graph(ax_lift, lift, lift_pos_concept)
    _ = ax_lift.set_xlim(-1.25, 1.25)
    _ = ax_lift.set_ylim(-1.25, 1.25)

    # Middle panel: a single big arrow, no text.
    _ = ax_arrow.axis("off")
    _ = ax_arrow.set_xlim(0.0, 1.0)
    _ = ax_arrow.set_ylim(0.0, 1.0)
    arrow = FancyArrowPatch(
        (0.1, 0.5),
        (0.9, 0.5),
        arrowstyle="-|>",
        mutation_scale=26,
        linewidth=3.0,
        color=NODE_BASE,
    )
    _ = ax_arrow.add_patch(arrow)

    save(fig, "voltage/vlift-concept.pdf")


def main() -> None:
    plt.rcParams["figure.facecolor"] = BG
    plt.rcParams["axes.facecolor"] = BG

    verify_petersen()

    gen_base()
    gen_base_no_voltage()
    gen_nodes()
    gen_labeled()
    gen_concept()

    outer = outer_edges()
    inner = inner_edges()
    spokes = spoke_edges()

    # Step 1: add the outer pentagon (self-loop on 0, voltage 1).
    gen_step("voltage/vlift-step1.pdf", old=[], new=outer, highlight="loop0")
    # Step 2: pentagon now grey, add the inner pentagram (self-loop on 1, voltage 2).
    gen_step("voltage/vlift-step2.pdf", old=outer, new=inner, highlight="loop1")
    # Step 3: outer + inner grey, add the spokes (arc 0->1, voltage 0) -> Petersen.
    gen_step(
        "voltage/vlift-step3.pdf",
        old=outer + inner,
        new=spokes,
        highlight="arc",
    )


if __name__ == "__main__":
    main()

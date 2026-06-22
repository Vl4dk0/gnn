"""Generate the GNN message-passing figures for the defense deck.

Produces transparent vector PDFs into presentation/figures/ around ONE fixed
9-node graph with a fixed layout. Node fills are RGB colors (the feature
vector IS the color). Chrome (outlines, arrows, halos) is dark grey.

Message-passing mechanics (slide 1):
  mp-graph-nums.pdf — full graph, COLORLESS (white fill), RGB vector inside node.
  mp-graph.pdf      — same graph, nodes colored by feature vector.
  mp-highlight.pdf  — vertex v and its 3 neighbours highlighted (halo + ring).
  mp-aggregate.pdf  — focused view of v and 3 neighbours, message arrows into v.

Mean diffusion (slide 2): closed-neighbourhood synchronous mean. Shares the
mechanics initial colors; reuses mp-graph-nums.pdf for the numbers frame.
Three colored snapshots shown side by side (rounds 0, 2, 8).
  diff-mean-0.pdf / diff-mean-2.pdf / diff-mean-8.pdf

Sum diffusion (slide 3): closed-neighbourhood synchronous sum, channel-dominant
dark init (a few red/green/blue-leaning nodes), snapshots rounds 0, 2, 8.
  diff-sum-nums.pdf / diff-sum-0.pdf / diff-sum-2.pdf / diff-sum-8.pdf

Every "-nums" figure draws white nodes with a dark-grey outline and the node's
three RGB numbers legible inside. The matching colored figure fills the same
nodes with that RGB value. All equations stay in the SLIDE TEXT, never baked
into the images.
"""

from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

# Embed TrueType fonts in the PDF (type 42) so text stays selectable and sharp.
matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --- Chrome palette (everything that is NOT a feature-vector fill) ---
OUTLINE = "#3a3a3a"  # thin dark-grey node outline
HALO = "#5a5a5a"  # highlight ring around v + neighbours
NODE_HALO_DARK = "#1a1a1a"  # bold dark halo around the focus vertex v
ARROW = "#1a1a1a"  # message arrows
EDGE = "#9a9a9a"  # graph edges

NODE_SIZE = 1700  # colored graphs (no text, so smaller + spread out)
NODE_SIZE_FOCUS = 2400  # aggregate focus frame
NUMS_SPREAD = 2.6  # layout scale for the numbers frame (room for wide labels)
FADE = 0.13  # alpha for de-emphasised (non-selected) chrome and fills

# --- Shared fixed graph -------------------------------------------------------
# Nine nodes, average degree ~3, v has exactly three neighbours (n1, n2, n3).
V = "v"
N1, N2, N3 = "n1", "n2", "n3"
NODES = [V, N1, N2, N3, "p", "q", "r", "s", "t"]
EDGES = [
    (V, N1),
    (V, N2),
    (V, N3),
    (N1, "p"),
    (N1, "q"),
    (N2, "q"),
    (N2, "r"),
    (N3, "r"),
    (N3, "s"),
    ("p", "t"),
    ("q", "t"),
    ("r", "s"),
    ("s", "t"),
    ("p", "q"),
]


def build_graph() -> "nx.Graph[str]":
    g: nx.Graph[str] = nx.Graph()
    g.add_nodes_from(NODES)
    g.add_edges_from(EDGES)
    return g


def fixed_layout() -> dict[str, tuple[float, float]]:
    """Deterministic spring layout from a fixed seed, spread well apart."""
    g = build_graph()
    # Larger k spreads nodes apart so big circles do not overlap or crowd.
    pos = nx.spring_layout(g, seed=7, k=2.2, iterations=300)
    return {n: (float(p[0]), float(p[1])) for n, p in pos.items()}


POS = fixed_layout()

# --- Initial feature vectors (RGB, 0-255) for the message-passing scene -------
# v starts red and turns olive after mean aggregation, so the update is visible.
# v's three neighbours are clearly different colors (green, blue, yellow); the
# other five keep distinct, well-saturated seeded colors. These are ALSO the
# mean-diffusion init colors.
MP_COLORS: dict[str, tuple[int, int, int]] = {
    V: (220, 100, 60),  # red -> becomes olive (135, 145, 105)
    N1: (60, 200, 80),  # green
    N2: (60, 80, 220),  # blue
    N3: (200, 200, 60),  # yellow
    "p": (240, 200, 40),
    "q": (40, 200, 220),
    "r": (220, 60, 220),
    "s": (250, 150, 40),
    "t": (120, 80, 220),
}


def rgb_hex(c: tuple[float, float, float]) -> str:
    r, g, b = (int(max(0, min(255, round(x)))) for x in c)
    return f"#{r:02x}{g:02x}{b:02x}"


# --- Diffusion ----------------------------------------------------------------
def closed_neighbourhood_matrix(g: "nx.Graph[str]") -> tuple[np.ndarray, list[str]]:
    """Adjacency + self-loop matrix A (rows/cols ordered by `order`)."""
    order = NODES
    idx = {n: i for i, n in enumerate(order)}
    n = len(order)
    a = np.eye(n, dtype=float)
    for u, w in g.edges():
        a[idx[u], idx[w]] = 1.0
        a[idx[w], idx[u]] = 1.0
    return a, order


def diffuse(init: np.ndarray, a: np.ndarray, rounds: int, mode: str) -> np.ndarray:
    """Synchronous closed-neighbourhood diffusion in float (no clamping)."""
    x = init.copy()
    for _ in range(rounds):
        if mode == "mean":
            deg = a.sum(axis=1, keepdims=True)
            x = (a @ x) / deg
        elif mode == "sum":
            x = a @ x
        else:
            raise ValueError(mode)
    return x


# --- Drawing helpers ----------------------------------------------------------
def _finish(ax: Axes, margin: float = 0.08) -> None:
    """Equal aspect, no axes, tight margins so the drawing fills the canvas.

    A small margin still keeps node circles from clipping; combined with
    savefig(bbox_inches="tight") the graph occupies most of the figure box.
    """
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.margins(margin)
    _ = ax.autoscale_view()


def draw_graph(
    ax: Axes,
    colors: dict[str, str],
    *,
    node_size: int = NODE_SIZE,
) -> None:
    """Plain colored graph: every node filled with its RGB color."""
    g = build_graph()
    _ = nx.draw_networkx_edges(
        g, POS, ax=ax, edgelist=EDGES, edge_color=EDGE, width=2.0
    )
    fills = [colors[n] for n in NODES]
    _ = nx.draw_networkx_nodes(
        g,
        POS,
        ax=ax,
        nodelist=NODES,
        node_size=node_size,
        node_color=fills,
        edgecolors=OUTLINE,
        linewidths=1.4,
    )
    _finish(ax)


def draw_highlight(ax: Axes, colors: dict[str, str]) -> None:
    """Neighbourhood of v in full color, everything else faded to near-nothing.

    v is the most prominent: larger, with a bold dark halo so the eye lands on
    it first; its three neighbours stay full color; all other nodes and every
    edge not inside the chosen neighbourhood fade to very light grey.
    """
    g = build_graph()
    sel = {V, N1, N2, N3}
    sel_edges = [(V, N1), (V, N2), (V, N3)]
    other_edges = [e for e in EDGES if e not in sel_edges]
    others = [n for n in NODES if n not in sel]

    # Faded background: distant edges + non-selected nodes (very light).
    _ = nx.draw_networkx_edges(
        g, POS, ax=ax, edgelist=other_edges, edge_color=EDGE, width=1.6, alpha=FADE
    )
    _ = nx.draw_networkx_nodes(
        g,
        POS,
        ax=ax,
        nodelist=others,
        node_size=NODE_SIZE,
        node_color=[colors[n] for n in others],
        edgecolors=OUTLINE,
        linewidths=1.0,
        alpha=FADE,
    )
    # Neighbourhood edges at full strength.
    _ = nx.draw_networkx_edges(
        g, POS, ax=ax, edgelist=sel_edges, edge_color=EDGE, width=2.4
    )
    # The three neighbours, full color.
    _ = nx.draw_networkx_nodes(
        g,
        POS,
        ax=ax,
        nodelist=[N1, N2, N3],
        node_size=NODE_SIZE,
        node_color=[colors[n] for n in (N1, N2, N3)],
        edgecolors=OUTLINE,
        linewidths=1.6,
    )
    # Bold dark halo behind v, then v itself, larger and most prominent.
    _ = nx.draw_networkx_nodes(
        g,
        POS,
        ax=ax,
        nodelist=[V],
        node_size=int(NODE_SIZE * 2.05),
        node_color="none",
        edgecolors=NODE_HALO_DARK,
        linewidths=5.0,
    )
    _ = nx.draw_networkx_nodes(
        g,
        POS,
        ax=ax,
        nodelist=[V],
        node_size=int(NODE_SIZE * 1.45),
        node_color=[colors[V]],
        edgecolors=NODE_HALO_DARK,
        linewidths=2.4,
    )
    _finish(ax)


def numbers_layout() -> dict[str, tuple[float, float]]:
    """Well-separated layout for the numbers frame so the wide labels never overlap.

    The wide "(a, b, c)" boxes need generous spacing, which the shared layout
    (seed 7) does not fully guarantee. This standalone frame uses a deterministic
    spring layout with a larger repulsion (k) and the seed that maximises the
    minimum pairwise node distance, then scales it out. The colored frames keep
    the shared spring layout, so the parts that animate stay continuous.
    """
    g = build_graph()
    pos = nx.spring_layout(g, seed=17, k=3.0, iterations=400)
    return {
        n: (float(p[0]) * NUMS_SPREAD, float(p[1]) * NUMS_SPREAD)
        for n, p in pos.items()
    }


def draw_graph_numbers(ax: Axes, vectors: dict[str, tuple[int, int, int]]) -> None:
    """Colorless graph: white rounded labels + dark outline + "(a, b, c)" inside.

    Uses a rounded box per node (not a circle) so the one-line vector fits
    legibly; edges connect the box centres. Wider layout prevents overlap.
    """
    pos = numbers_layout()
    g = build_graph()
    _ = nx.draw_networkx_edges(
        g, pos, ax=ax, edgelist=EDGES, edge_color=EDGE, width=2.0
    )
    bbox = dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor=OUTLINE,
        linewidth=1.6,
    )
    for n in NODES:
        r, gg, b = vectors[n]
        x, y = pos[n]
        _ = ax.text(
            x,
            y,
            f"({r}, {gg}, {b})",
            ha="center",
            va="center",
            fontsize=12.0,
            color="#1a1a1a",
            bbox=bbox,
            zorder=3,
        )
    # The rounded label boxes extend well beyond the node centres (which is what
    # autoscale sees), so add generous margins (more horizontally) to guarantee
    # every box is fully inside the axis limits and nothing is clipped.
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.autoscale_view()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xpad = (x1 - x0) * 0.22
    ypad = (y1 - y0) * 0.12
    _ = ax.set_xlim(x0 - xpad, x1 + xpad)
    _ = ax.set_ylim(y0 - ypad, y1 + ypad)


def make_fig(w: float = 6.0, h: float = 5.4) -> tuple[Figure, Axes]:
    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=(w, h))
    ax: Axes = fig.add_subplot(1, 1, 1)
    return fig, ax


def save(fig: Figure, name: str) -> None:
    (FIGURES / name).parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(
        FIGURES / name,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
        facecolor="none",
    )
    print(f"saved {FIGURES / name}")
    plt.close(fig)


# --- Slide 1: mechanics -------------------------------------------------------
def gen_mechanics() -> None:
    base_colors = {n: rgb_hex(MP_COLORS[n]) for n in NODES}

    # mp-graph-nums: colorless graph, "(a, b, c)" labels inside each node.
    # (Shared with the mean-diffusion slide as its "numbers" frame.)
    fig, ax = make_fig(w=7.6, h=6.4)
    draw_graph_numbers(ax, MP_COLORS)
    save(fig, "gnn/mp-graph-nums.pdf")

    # mp-graph: full graph colored by feature vector (== mean diffusion round 0).
    fig, ax = make_fig()
    draw_graph(ax, base_colors)
    save(fig, "gnn/mp-graph.pdf")

    # mp-highlight: v + 3 neighbours in full color, everything else faded.
    fig, ax = make_fig()
    draw_highlight(ax, base_colors)
    save(fig, "gnn/mp-highlight.pdf")

    # mp-aggregate: ONLY v + its 3 neighbours, clean message arrows into v.
    # v is drawn with its UPDATED color (mean) so the change is visible.
    fig, ax = make_fig()
    _draw_aggregate(ax, base_colors)
    save(fig, "gnn/mp-aggregate.pdf")


def _draw_aggregate(ax: Axes, base_colors: dict[str, str]) -> None:
    """Focus frame: v and its 3 neighbours only, arrows from each neighbour to v.

    Arrows are offset by the node radius at both ends so they start at the
    neighbour boundary and stop at v's boundary (no overshoot, no doubled edge
    underneath, consistent style).
    """
    g = build_graph()
    focus = [V, N1, N2, N3]

    result = diffuse_v_mean()  # (135, 145, 105), olive
    fills = {n: base_colors[n] for n in focus}
    fills[V] = rgb_hex(result)

    # Approximate node radius in data units from the marker area (points^2).
    # area = pi * r_pt^2  ->  r_pt; convert via axis scale after limits are set.
    # Simpler: draw nodes first, then size arrows by a fixed data-space gap that
    # matches the chosen node sizes / layout spacing.
    _ = nx.draw_networkx_nodes(
        g,
        POS,
        ax=ax,
        nodelist=[N1, N2, N3],
        node_size=NODE_SIZE_FOCUS,
        node_color=[fills[n] for n in (N1, N2, N3)],
        edgecolors=OUTLINE,
        linewidths=1.8,
    )
    _ = nx.draw_networkx_nodes(
        g,
        POS,
        ax=ax,
        nodelist=[V],
        node_size=int(NODE_SIZE_FOCUS * 1.25),
        node_color=[fills[V]],
        edgecolors=NODE_HALO_DARK,
        linewidths=2.2,
    )

    # Clean message arrows: shrink endpoints by node radius (in points) so the
    # arrow starts at the neighbour edge and the head stops at v's edge.
    r_nb = (NODE_SIZE_FOCUS / 3.14159) ** 0.5  # marker radius in points
    r_v = ((NODE_SIZE_FOCUS * 1.25) / 3.14159) ** 0.5
    for nb in (N1, N2, N3):
        arrow = FancyArrowPatch(
            POS[nb],
            POS[V],
            arrowstyle="-|>",
            mutation_scale=20,
            color=ARROW,
            linewidth=2.4,
            shrinkA=r_nb + 3,
            shrinkB=r_v + 4,
        )
        _ = ax.add_patch(arrow)

    # Only the focus nodes drive the extent (with margin so nothing clips).
    xs = [POS[n][0] for n in focus]
    ys = [POS[n][1] for n in focus]
    pad = 0.42
    _ = ax.set_xlim(min(xs) - pad, max(xs) + pad)
    _ = ax.set_ylim(min(ys) - pad, max(ys) + pad)
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")


def diffuse_v_mean() -> tuple[float, float, float]:
    """Mean of v and its three neighbours' colors.

    With the message-passing init this is (135, 145, 105): v goes red -> olive,
    a visible change.
    """
    cols = [MP_COLORS[V], MP_COLORS[N1], MP_COLORS[N2], MP_COLORS[N3]]
    arr = np.array(cols, dtype=float)
    m = arr.mean(axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))


# --- Slides 2 & 3: diffusion --------------------------------------------------
def gen_mean_diffusion() -> None:
    # Uses the SAME channel-dominant init colors as the sum slide (SUM_COLORS),
    # so the two diffusion examples start identically and the audience can
    # compare mean vs sum directly. Each snapshot is rescaled the same way as
    # the sum slide (largest channel value across all vertices -> 255) so the
    # low-magnitude init is visible and round 0 matches diff-sum-0 exactly.
    # The "numbers" frame and round-0 frame are shared with the sum slide
    # (diff-sum-nums / diff-sum-0). Mean keeps values bounded and converges to a
    # single shared color (oversmoothing).
    g = build_graph()
    a, order = closed_neighbourhood_matrix(g)
    init = np.array([SUM_COLORS[n] for n in order], dtype=float)
    for rounds in (0, 2, 8):
        x = diffuse(init, a, rounds, "mean")
        peak = float(x.max())
        if peak > 0:
            x = x / peak * 255.0
        colors = {order[i]: rgb_hex(tuple(x[i])) for i in range(len(order))}
        fig, ax = make_fig()
        draw_graph(ax, colors)
        save(fig, f"gnn/diff-mean-{rounds}.pdf")


# Channel-dominant init for the SUM slide: three red-leaning, three
# green-leaning, three blue-leaning nodes, each with small per-node variation
# so all nine are distinct. Channels span 0..31 (~255/8): the dominant channel
# reaches ~31 while off-channels stay low. Closed-neighbourhood SUM grows by
# ~(deg+1) ~ 5x per round, so it saturates fast; the three snapshots tell the
# "sum blows up / saturates" story:
#   round 0 -> clearly colored (max 31),
#   round 2 -> brightening towards saturation,
#   round 8 -> fully saturated (every channel pinned to 255).
SUM_COLORS: dict[str, tuple[int, int, int]] = {
    V: (31, 4, 1),  # red-dominant
    N1: (28, 2, 3),
    N2: (24, 5, 0),
    N3: (2, 31, 1),  # green-dominant
    "p": (4, 27, 3),
    "q": (1, 23, 2),
    "r": (0, 3, 31),  # blue-dominant
    "s": (3, 1, 28),
    "t": (1, 4, 24),
}


def gen_sum_diffusion() -> None:
    g = build_graph()
    a, order = closed_neighbourhood_matrix(g)
    init = np.array([SUM_COLORS[n] for n in order], dtype=float)

    # diff-sum-nums: colorless graph, round-0 (a, b, c) labels inside each node.
    fig, ax = make_fig(w=7.6, h=6.4)
    draw_graph_numbers(ax, SUM_COLORS)
    save(fig, "gnn/diff-sum-nums.pdf")

    for rounds in (0, 2, 8):
        x = diffuse(init, a, rounds, "sum")
        # Closed-neighbourhood sum grows ~(deg+1)x per round, so raw values blow
        # far past 255 and every channel clamps to white. Instead rescale each
        # snapshot: the largest single channel value across all vertices maps to
        # 255 and every other value scales by the same ratio. This preserves the
        # relative color of every node so the converged structure stays visible.
        peak = float(x.max())
        if peak > 0:
            x = x / peak * 255.0
        colors = {order[i]: rgb_hex(tuple(x[i])) for i in range(len(order))}
        fig, ax = make_fig()
        draw_graph(ax, colors)
        save(fig, f"gnn/diff-sum-{rounds}.pdf")


def main() -> None:
    gen_mechanics()
    gen_mean_diffusion()
    gen_sum_diffusion()


if __name__ == "__main__":
    main()

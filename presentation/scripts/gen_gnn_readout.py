"""Generate the GNN read-out figure for the defense deck.

Closes the message-passing story: after message passing every vertex carries a
VECTOR (shown as a number tuple, the same values that produced the diff-sum
colours), and those vectors are the input to a neural network. The figure shows
the graph with vertices drawn AS VECTORS (white boxes with the (r,g,b) tuple, not
colours) on the left, a bold arrow, and a small feed-forward network on the right.

The graph, layout, sum-init and diffusion are reproduced deterministically here so
the vectors match the diff-sum figures from gen_gnn_figures. We display round 2:
the vectors are already distinct (sum aggregation oversmooths to near-identical
vectors by round 8), which reads clearly as "every vertex has its own vector".

Output: presentation/figures/gnn/gnn-readout.pdf (transparent vector PDF).
"""

from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch

matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --- Shared fixed graph (mirrors gen_gnn_figures) -----------------------------
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

# Channel-dominant sum-init (identical to gen_gnn_figures.SUM_COLORS).
SUM_COLORS: dict[str, tuple[int, int, int]] = {
    V: (31, 4, 1),
    N1: (28, 2, 3),
    N2: (24, 5, 0),
    N3: (2, 31, 1),
    "p": (4, 27, 3),
    "q": (1, 23, 2),
    "r": (0, 3, 31),
    "s": (3, 1, 28),
    "t": (1, 4, 24),
}

DIFFUSION_ROUNDS = 2  # round 2 vectors are distinct (round 8 oversmooths)

# Chrome.
EDGE = "#9a9a9a"
OUTLINE = "#3a3a3a"
NN_NODE = "#4a6572"  # steel fill for the network units
NN_EDGE = "#c4ced4"  # thin connection lines
ARROW = "#1a1a1a"


def build_graph() -> "nx.Graph[str]":
    g: nx.Graph[str] = nx.Graph()
    g.add_nodes_from(NODES)
    g.add_edges_from(EDGES)
    return g


def round_vectors() -> dict[str, tuple[int, int, int]]:
    """Sum-diffusion vectors at DIFFUSION_ROUNDS, rescaled to 0..255 integers."""
    order = NODES
    idx = {n: i for i, n in enumerate(order)}
    a = np.eye(len(order), dtype=float)
    for u, w in EDGES:
        a[idx[u], idx[w]] = 1.0
        a[idx[w], idx[u]] = 1.0
    x = np.array([SUM_COLORS[n] for n in order], dtype=float)
    for _ in range(DIFFUSION_ROUNDS):
        x = a @ x
    peak = float(x.max())
    if peak > 0:
        x = x / peak * 255.0
    return {
        order[i]: (
            int(round(float(x[i][0]))),
            int(round(float(x[i][1]))),
            int(round(float(x[i][2]))),
        )
        for i in range(len(order))
    }


def graph_positions(
    cx: float, cy: float, half: float
) -> dict[str, tuple[float, float]]:
    """Well-spread spring layout (room for the wide number boxes), centred at
    (cx,cy) and scaled UNIFORMLY to fit `half`."""
    pos = nx.spring_layout(build_graph(), seed=17, k=3.0, iterations=400)
    p = {n: (float(v[0]), float(v[1])) for n, v in pos.items()}
    xs = [q[0] for q in p.values()]
    ys = [q[1] for q in p.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span = max(maxx - minx, maxy - miny)
    scale = 2.0 * half / span
    ox = (minx + maxx) / 2.0
    oy = (miny + maxy) / 2.0
    return {
        n: (cx + (q[0] - ox) * scale, cy + (q[1] - oy) * scale) for n, q in p.items()
    }


def draw_vector_graph(ax: Axes, pos: dict[str, tuple[float, float]]) -> None:
    """Draw the graph with each vertex shown AS A VECTOR (white box + tuple)."""
    g = build_graph()
    vectors = round_vectors()
    _ = nx.draw_networkx_edges(
        g, pos, ax=ax, edgelist=EDGES, edge_color=EDGE, width=2.0
    )
    bbox = dict(
        boxstyle="round,pad=0.3", facecolor="white", edgecolor=OUTLINE, linewidth=1.5
    )
    text_fn = cast(Callable[..., object], ax.text)
    for n in NODES:
        r, gg, b = vectors[n]
        x, y = pos[n]
        _ = text_fn(
            x,
            y,
            f"({r}, {gg}, {b})",
            ha="center",
            va="center",
            fontsize=10.5,
            color="#1a1a1a",
            bbox=bbox,
            zorder=3,
        )


def draw_network(
    ax: Axes, x_left: float, layers: list[int], y_center: float, span: float
) -> None:
    """Draw a small fully-connected feed-forward net: columns of circles + lines."""
    col_gap = 1.7
    radius = 0.24
    coords: list[list[tuple[float, float]]] = []
    for li, count in enumerate(layers):
        cx = x_left + li * col_gap
        if count == 1:
            ys = [y_center]
        else:
            ys = [y_center + span / 2 - i * span / (count - 1) for i in range(count)]
        coords.append([(cx, y) for y in ys])

    plot_fn = cast(Callable[..., object], ax.plot)
    for li in range(len(coords) - 1):
        for x_a, y_a in coords[li]:
            for x_b, y_b in coords[li + 1]:
                _ = plot_fn(
                    [x_a, x_b], [y_a, y_b], color=NN_EDGE, linewidth=0.8, zorder=2
                )

    add_patch = cast(Callable[..., object], ax.add_patch)
    for col in coords:
        for cx, cy in col:
            _ = add_patch(
                Circle(
                    (cx, cy),
                    radius,
                    facecolor=NN_NODE,
                    edgecolor="#2f4250",
                    linewidth=1.0,
                    zorder=3,
                )
            )


def main() -> None:
    figure_fn = cast(Callable[..., Figure], plt.figure)
    fig = figure_fn(figsize=(14.0, 6.0))
    ax = fig.add_subplot(1, 1, 1)

    # --- Left: the graph with vertices drawn as VECTORS ----------------------
    pos = graph_positions(cx=3.9, cy=3.0, half=2.7)
    draw_vector_graph(ax, pos)

    # --- Middle: bold arrow ---------------------------------------------------
    add_patch = cast(Callable[..., object], ax.add_patch)
    _ = add_patch(
        FancyArrowPatch(
            (7.9, 3.0),
            (9.2, 3.0),
            arrowstyle="-|>",
            mutation_scale=30,
            color=ARROW,
            linewidth=3.0,
        )
    )

    # --- Right: feed-forward network -----------------------------------------
    draw_network(ax, x_left=10.0, layers=[4, 6, 4, 1], y_center=3.0, span=4.2)

    _ = ax.set_xlim(-0.3, 15.8)
    _ = ax.set_ylim(0.0, 6.0)
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")

    out = FIGURES / "gnn" / "gnn-readout.pdf"
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

"""Presentation-specific greyscale variants of the reused thesis figures.

The thesis figures (thesis/scripts/*.py, thesis/figures/*.png) use a blue/orange
palette. The defense deck is strictly achromatic, so this script re-draws the
same six figures in the greyscale website palette and exports them as vector PDF.

Outputs into presentation/figures/ (the filenames the deck references):

  fig01-message-passing.pdf   <- thesis mpnn_layer (one message-passing layer)
  fig-voltage-base.pdf        <- base K_4 with voltages in Z_3 (transparent)
  fig-voltage-lift.pdf        <- its lift: 12 vertices, 3-regular, girth 4 (transparent)
  fig05-2swap.pdf             <- thesis edge_swap_move (degree-preserving 2-swap)
  fig07-3swap.pdf             <- thesis three_swap (degree-preserving 3-swap)
  fig09-dodeca-petersen.pdf   <- thesis petersen_excision (dodecahedron -> Petersen)
  fig10-degree-vs-girth.pdf   <- thesis node_prediction (degree vs. girth accuracy)

The drawing logic mirrors the thesis originals; only the palette changes (no
hue anywhere) and the export format is vector PDF. The thesis originals are NOT
modified. Layouts are deterministic; the excision is verified at runtime to be
the (3,5)-cage (Petersen) before drawing.
"""

import math
import sys
from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai.cage.voltage.base_graphs import BaseGraph
from ai.cage.voltage.groups import cyclic_group
from ai.cage.voltage.lift import build_lift

# Embed TrueType fonts in the PDF (type 42) so text stays selectable and sharp.
matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"
BG = "#ffffff"

# ---------------------------------------------------------------------------
# Greyscale palette for the graph drawings (deck chrome stays grey too). Each
# role maps a thesis colour to a grey tone, keeping the original figure's
# two-emphasis hierarchy: base vertices light grey, highlighted vertices
# near-black, primary emphasis near-black, secondary emphasis dark grey.
# (fig10-degree-vs-girth is an explainer chart and uses its own muted palette.)
# ---------------------------------------------------------------------------
COLORS: dict[str, str] = {
    "node": "#dcdcdc",  # light grey vertex fill
    "node_alt": "#1a1a1a",  # highlighted vertex fill (near-black)
    "edge": "#aaaaaa",  # ordinary edge / outline grey
    "accent": "#1a1a1a",  # primary emphasis (was orange) -> near-black
    "accent2": "#5a5a5a",  # secondary emphasis (was green) -> dark grey
    "label": "#1a1a1a",  # dark text/labels
    "label_light": "#f0f0f0",  # light text on dark fills
}
INK = "#1a1a1a"
MUTED = "#555555"
LINE = "#d0d0d0"
NODE_SIZE = 520
BASE_NODE_SIZE = 900


def label_color(fill: str) -> str:
    """Readable label colour (dark or light) for a given vertex fill."""
    h = fill.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return COLORS["label"] if luminance > 0.5 else COLORS["label_light"]


def save(fig: Figure, name: str, **kwargs: object) -> None:
    (FIGURES / name).parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(FIGURES / name, format="pdf", facecolor=BG, **kwargs)
    print(f"saved {FIGURES / name}")


# ===========================================================================
# fig01-message-passing.pdf  — one message-passing layer (three panels)
# ===========================================================================
MP_NEIGHBOUR_ANGLE: list[float] = [60.0, 150.0, 240.0, 330.0]
MP_NEIGHBOUR_RADIUS = 1.35
MP_NODES: list[str] = ["v", "u1", "u2", "u3", "u4"]
MP_LABELS: dict[str, str] = {
    "v": r"$v$",
    "u1": r"$u_1$",
    "u2": r"$u_2$",
    "u3": r"$u_3$",
    "u4": r"$u_4$",
}
MP_EDGES: list[tuple[str, str]] = [("v", "u1"), ("v", "u2"), ("v", "u3"), ("v", "u4")]
CELL_W = 0.16
CELL_H = 0.16
N_CELLS = 3


def mp_node_positions() -> dict[str, tuple[float, float]]:
    pos: dict[str, tuple[float, float]] = {"v": (0.0, 0.0)}
    for name, ang in zip(["u1", "u2", "u3", "u4"], MP_NEIGHBOUR_ANGLE):
        a = math.radians(ang)
        pos[name] = (
            MP_NEIGHBOUR_RADIUS * math.cos(a),
            MP_NEIGHBOUR_RADIUS * math.sin(a),
        )
    return pos


def draw_vector(ax: Axes, origin: tuple[float, float], fill: str, edge: str) -> None:
    add_patch = cast(Callable[..., object], ax.add_patch)
    x0, y0 = origin
    for i in range(N_CELLS):
        _ = add_patch(
            Rectangle(
                (x0 + i * CELL_W, y0),
                CELL_W,
                CELL_H,
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.1,
            )
        )


def mp_draw_nodes(ax: Axes, pos: dict[str, tuple[float, float]]) -> None:
    g: nx.Graph[str] = nx.Graph()
    g.add_nodes_from(MP_NODES)
    g.add_edges_from(MP_EDGES)
    fill = {n: (COLORS["node_alt"] if n == "v" else COLORS["node"]) for n in MP_NODES}
    _ = nx.draw_networkx_edges(
        g, pos, ax=ax, edgelist=MP_EDGES, edge_color=COLORS["edge"], width=1.6
    )
    _ = nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_size=NODE_SIZE,
        node_color=[fill[n] for n in MP_NODES],
        edgecolors=COLORS["label"],
        linewidths=1.4,
    )
    _ = nx.draw_networkx_labels(
        g, pos, labels=MP_LABELS, ax=ax, font_color=COLORS["label"], font_size=10
    )


def mp_vector_origin(p: tuple[float, float]) -> tuple[float, float]:
    return (p[0] + 0.18, p[1] - 0.30)


def mp_panel_states(ax: Axes) -> None:
    pos = mp_node_positions()
    mp_draw_nodes(ax, pos)
    for name in MP_NODES:
        fill = COLORS["node_alt"] if name == "v" else COLORS["node"]
        draw_vector(ax, mp_vector_origin(pos[name]), fill, COLORS["edge"])
    set_title = cast(Callable[..., object], ax.set_title)
    _ = set_title("Hidden states", fontsize=11, color=INK)
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(-2.0, 2.2)
    _ = ax.set_ylim(-2.0, 2.0)


def mp_panel_messages(ax: Axes) -> None:
    pos = mp_node_positions()
    mp_draw_nodes(ax, pos)
    add_patch = cast(Callable[..., object], ax.add_patch)
    text = cast(Callable[..., object], ax.text)
    for idx, name in enumerate(["u1", "u2", "u3", "u4"]):
        sx, sy = pos[name]
        tx, ty = pos["v"]
        dx, dy = tx - sx, ty - sy
        d = math.hypot(dx, dy)
        ux, uy = dx / d, dy / d
        start = (sx + ux * 0.32, sy + uy * 0.32)
        end = (tx - ux * 0.36, ty - uy * 0.36)
        _ = add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=16,
                color=COLORS["accent"],
                linewidth=2.0,
                shrinkA=0.0,
                shrinkB=0.0,
            )
        )
        if idx == 0:
            mx = (start[0] + end[0]) / 2.0
            my = (start[1] + end[1]) / 2.0
            _ = text(
                mx + 0.12,
                my + 0.12,
                r"$m$",
                ha="center",
                va="center",
                color=COLORS["accent"],
                fontsize=11,
                fontweight="bold",
            )
    set_title = cast(Callable[..., object], ax.set_title)
    _ = set_title("Messages", fontsize=11, color=INK)
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(-2.0, 2.2)
    _ = ax.set_ylim(-2.0, 2.0)


def mp_draw_box(
    ax: Axes, center: tuple[float, float], w: float, h: float, label: str
) -> None:
    add_patch = cast(Callable[..., object], ax.add_patch)
    text = cast(Callable[..., object], ax.text)
    x0, y0 = center[0] - w / 2.0, center[1] - h / 2.0
    _ = add_patch(
        Rectangle(
            (x0, y0),
            w,
            h,
            facecolor=COLORS["node"],
            edgecolor=COLORS["label"],
            linewidth=1.6,
        )
    )
    _ = text(
        center[0],
        center[1],
        label,
        ha="center",
        va="center",
        color=COLORS["label"],
        fontsize=12,
        fontweight="bold",
    )


def mp_arrow(
    ax: Axes, start: tuple[float, float], end: tuple[float, float], color: str
) -> None:
    add_patch = cast(Callable[..., object], ax.add_patch)
    _ = add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            linewidth=1.8,
            shrinkA=2.0,
            shrinkB=2.0,
        )
    )


def mp_panel_aggregate_update(ax: Axes) -> None:
    text = cast(Callable[..., object], ax.text)
    msg_x = 0.3
    msg_ys = [2.7, 2.1, 1.5, 0.9]
    for y in msg_ys:
        draw_vector(ax, (msg_x, y), COLORS["node"], COLORS["accent"])
    _ = text(
        msg_x + 1.5 * CELL_W,
        msg_ys[0] + CELL_H + 0.22,
        "messages",
        ha="center",
        va="bottom",
        color=COLORS["accent"],
        fontsize=9,
        fontstyle="italic",
    )
    agg_center = (2.0, 1.8)
    mp_draw_box(ax, agg_center, 0.95, 0.7, "AGG")
    for y in msg_ys:
        mp_arrow(
            ax,
            (msg_x + N_CELLS * CELL_W + 0.05, y + CELL_H / 2.0),
            (agg_center[0] - 0.55, agg_center[1]),
            COLORS["accent"],
        )
    mvt = (3.55, 1.8)
    mp_arrow(
        ax,
        (agg_center[0] + 0.55, agg_center[1]),
        (mvt[0] - 0.05, mvt[1]),
        COLORS["edge"],
    )
    draw_vector(ax, (mvt[0], mvt[1] - CELL_H / 2.0), COLORS["node"], COLORS["edge"])
    _ = text(
        mvt[0] + 1.5 * CELL_W,
        mvt[1] + CELL_H,
        r"$m_v^t$",
        ha="center",
        va="bottom",
        color=COLORS["edge"],
        fontsize=11,
    )
    old = (3.55, 0.35)
    draw_vector(ax, (old[0], old[1] - CELL_H / 2.0), COLORS["node_alt"], COLORS["edge"])
    _ = text(
        old[0] + 1.5 * CELL_W,
        old[1] - CELL_H,
        r"$h_v^{t-1}$",
        ha="center",
        va="top",
        color=COLORS["edge"],
        fontsize=11,
    )
    upd_center = (5.2, 1.1)
    mp_draw_box(ax, upd_center, 0.95, 0.7, "UPD")
    mp_arrow(
        ax,
        (mvt[0] + N_CELLS * CELL_W + 0.05, mvt[1]),
        (upd_center[0] - 0.55, upd_center[1] + 0.18),
        COLORS["edge"],
    )
    mp_arrow(
        ax,
        (old[0] + N_CELLS * CELL_W + 0.05, old[1]),
        (upd_center[0] - 0.55, upd_center[1] - 0.18),
        COLORS["edge"],
    )
    new = (6.75, 1.1)
    mp_arrow(
        ax,
        (upd_center[0] + 0.55, upd_center[1]),
        (new[0] - 0.05, new[1]),
        COLORS["accent2"],
    )
    # new state in light grey fill with dark-grey outline (the "updated" emphasis)
    draw_vector(ax, (new[0], new[1] - CELL_H / 2.0), "#eeeeee", COLORS["accent2"])
    _ = text(
        new[0] + 1.5 * CELL_W,
        new[1] + CELL_H,
        r"$h_v^t$",
        ha="center",
        va="bottom",
        color=COLORS["accent2"],
        fontsize=11,
        fontweight="bold",
    )
    set_title = cast(Callable[..., object], ax.set_title)
    _ = set_title("Aggregate and update", fontsize=11, color=INK)
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(-0.1, 7.8)
    _ = ax.set_ylim(-0.4, 3.4)


def make_message_passing() -> None:
    figure = cast(Callable[..., Figure], plt.figure)
    fig: Figure = figure(figsize=(12, 4), facecolor=BG)
    ax1: Axes = fig.add_subplot(1, 3, 1)
    ax2: Axes = fig.add_subplot(1, 3, 2)
    ax3: Axes = fig.add_subplot(1, 3, 3)
    mp_panel_states(ax1)
    mp_panel_messages(ax2)
    mp_panel_aggregate_update(ax3)
    fig.tight_layout()
    save_transparent(
        fig, "_unused/fig01-message-passing.pdf", bbox_inches="tight", pad_inches=0.1
    )
    plt.close(fig)


# ===========================================================================
# fig-voltage-base.pdf / fig-voltage-lift.pdf  — base K_4 over Z_3 and its lift
# ===========================================================================
K4_EDGES: list[tuple[int, int]] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
K4_VOLTAGES: list[int] = [0, 0, 0, 1, 1, 1]
GROUP_ORDER = 3
TRI_ANGLE: dict[int, float] = {1: 90.0, 2: 210.0, 3: 330.0}
BASE_RADIUS = 1.0
FLOOR_ANGLE: list[float] = [90.0, 210.0, 330.0]
FLOOR_CENTER_RADIUS = 2.3
COPY_RADIUS = 0.95


def local_offset(v: int, r: float) -> tuple[float, float]:
    if v == 0:
        return (0.0, 0.0)
    a = math.radians(TRI_ANGLE[v])
    return (r * math.cos(a), r * math.sin(a))


def k4_base() -> BaseGraph:
    bg = BaseGraph(num_nodes=4)
    for u, v in K4_EDGES:
        _ = bg.add_edge(u, v)
    return bg


def vl_base_pos() -> dict[int, tuple[float, float]]:
    return {v: local_offset(v, BASE_RADIUS) for v in range(4)}


def vl_draw_base(ax: Axes) -> None:
    pos = vl_base_pos()
    dg: nx.DiGraph[int] = nx.DiGraph()
    _ = dg.add_nodes_from(range(4))
    for (u, v), volt in zip(K4_EDGES, K4_VOLTAGES):
        _ = dg.add_edge(u, v, volt=volt)

    free = [(u, v) for (u, v), w in zip(K4_EDGES, K4_VOLTAGES) if w != 0]
    tree = [(u, v) for (u, v), w in zip(K4_EDGES, K4_VOLTAGES) if w == 0]

    _ = nx.draw_networkx_edges(
        dg,
        pos,
        ax=ax,
        edgelist=tree,
        edge_color=COLORS["edge"],
        width=1.6,
        arrowsize=14,
        node_size=BASE_NODE_SIZE,
        connectionstyle="arc3,rad=0.0",
    )
    _ = nx.draw_networkx_edges(
        dg,
        pos,
        ax=ax,
        edgelist=free,
        edge_color=COLORS["accent"],
        width=2.4,
        arrowsize=16,
        node_size=BASE_NODE_SIZE,
        connectionstyle="arc3,rad=0.0",
    )
    _ = nx.draw_networkx_nodes(
        dg,
        pos,
        ax=ax,
        node_size=BASE_NODE_SIZE,
        node_color=COLORS["node"],
        edgecolors=COLORS["label"],
    )
    _ = nx.draw_networkx_labels(dg, pos, ax=ax, font_color=COLORS["label"])

    text = cast(Callable[..., object], ax.text)
    for (u, v), volt in zip(K4_EDGES, K4_VOLTAGES):
        mx = (pos[u][0] + pos[v][0]) / 2.0
        my = (pos[u][1] + pos[v][1]) / 2.0
        color = COLORS["accent"] if volt != 0 else COLORS["edge"]
        _ = text(
            mx,
            my,
            str(volt),
            ha="center",
            va="center",
            color=color,
            fontsize=11,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
        )
    _ = ax.set_xlim(-1.5, 1.5)
    _ = ax.set_ylim(-1.5, 1.5)
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")


def vl_lift_pos() -> dict[int, tuple[float, float]]:
    pos: dict[int, tuple[float, float]] = {}
    for h in range(GROUP_ORDER):
        fa = math.radians(FLOOR_ANGLE[h])
        cx = FLOOR_CENTER_RADIUS * math.cos(fa)
        cy = FLOOR_CENTER_RADIUS * math.sin(fa)
        for v in range(4):
            ox, oy = local_offset(v, COPY_RADIUS)
            pos[v * GROUP_ORDER + h] = (cx + ox, cy + oy)
    return pos


def vl_draw_lift(ax: Axes) -> None:
    group = cyclic_group(GROUP_ORDER)
    lifted = build_lift(k4_base(), group, K4_VOLTAGES)
    pos = vl_lift_pos()

    in_floor: list[tuple[int, int]] = []
    cross: list[tuple[int, int]] = []
    for a, b in lifted.edges:
        if a % GROUP_ORDER == b % GROUP_ORDER:
            in_floor.append((a, b))
        else:
            cross.append((a, b))

    _ = nx.draw_networkx_edges(
        lifted,
        pos,
        ax=ax,
        edgelist=cross,
        edge_color="#bbbbbb",
        width=0.9,
        arrows=True,
        arrowstyle="-",
        connectionstyle="arc3,rad=0.16",
    )
    _ = nx.draw_networkx_edges(
        lifted,
        pos,
        ax=ax,
        edgelist=in_floor,
        edge_color=COLORS["edge"],
        width=1.4,
    )

    fill_of = {
        n: (COLORS["node_alt"] if n % GROUP_ORDER == 0 else COLORS["node"])
        for n in lifted.nodes
    }
    _ = nx.draw_networkx_nodes(
        lifted,
        pos,
        ax=ax,
        node_size=720,
        node_color=[fill_of[n] for n in lifted.nodes],
        edgecolors=COLORS["label"],
    )
    labels = {n: f"{n // GROUP_ORDER},{n % GROUP_ORDER}" for n in lifted.nodes}
    for lc in {label_color(c) for c in fill_of.values()}:
        sub = {n: labels[n] for n in lifted.nodes if label_color(fill_of[n]) == lc}
        _ = nx.draw_networkx_labels(
            lifted, pos, labels=sub, ax=ax, font_size=8, font_color=lc
        )

    text = cast(Callable[..., object], ax.text)
    for h in range(GROUP_ORDER):
        fa = math.radians(FLOOR_ANGLE[h])
        cx = FLOOR_CENTER_RADIUS * math.cos(fa)
        cy = FLOOR_CENTER_RADIUS * math.sin(fa)
        _ = text(
            cx,
            cy + COPY_RADIUS + 0.7,
            f"copy $h={h}$",
            ha="center",
            va="center",
            color=COLORS["accent2"],
            fontsize=9,
            fontstyle="italic",
        )

    _ = ax.set_xlim(-4.0, 4.0)
    _ = ax.set_ylim(-4.0, 4.2)
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")


def save_transparent(fig: Figure, name: str, **kwargs: object) -> None:
    """Save a graph drawing with no background (transparent PDF)."""
    (FIGURES / name).parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(FIGURES / name, format="pdf", transparent=True, facecolor="none", **kwargs)
    print(f"saved {FIGURES / name}")


def make_voltage_base() -> None:
    figure = cast(Callable[..., Figure], plt.figure)
    fig: Figure = figure(figsize=(5.2, 5.2), facecolor="none")
    ax: Axes = fig.add_subplot(1, 1, 1)
    vl_draw_base(ax)
    fig.tight_layout()
    save_transparent(fig, "voltage/fig-voltage-base.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def make_voltage_lift() -> None:
    figure = cast(Callable[..., Figure], plt.figure)
    fig: Figure = figure(figsize=(6.0, 6.0), facecolor="none")
    ax: Axes = fig.add_subplot(1, 1, 1)
    vl_draw_lift(ax)
    fig.tight_layout()
    save_transparent(fig, "voltage/fig-voltage-lift.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ===========================================================================
# fig05-2swap.pdf and fig07-3swap.pdf  — degree-preserving swaps
# ===========================================================================
def swap_panel(
    ax: Axes,
    positions: dict[str, tuple[float, float]],
    nodes: list[str],
    edges: list[tuple[str, str]],
    edge_color: str,
    linestyle: str,
    edge_width: float,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    g: nx.Graph[str] = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    _ = nx.draw_networkx_nodes(
        g,
        positions,
        ax=ax,
        node_size=NODE_SIZE,
        node_color=COLORS["node"],
        edgecolors=COLORS["label"],
        linewidths=1.4,
    )
    _ = nx.draw_networkx_labels(
        g, positions, ax=ax, font_color=COLORS["label"], font_size=9
    )
    _ = nx.draw_networkx_edges(
        g,
        positions,
        ax=ax,
        edgelist=edges,
        edge_color=edge_color,
        width=edge_width,
        style=linestyle,
    )
    set_title = cast(Callable[..., object], ax.set_title)
    _ = set_title(title, fontsize=10, color=INK)
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(*xlim)
    _ = ax.set_ylim(*ylim)


def make_2swap() -> None:
    positions: dict[str, tuple[float, float]] = {
        "A": (0.0, 1.0),
        "B": (1.0, 1.0),
        "C": (1.0, 0.0),
        "D": (0.0, 0.0),
    }
    nodes = ["A", "B", "C", "D"]
    figure = cast(Callable[..., Figure], plt.figure)
    fig: Figure = figure(figsize=(8, 4), facecolor=BG)
    ax_before: Axes = fig.add_subplot(1, 2, 1)
    ax_after: Axes = fig.add_subplot(1, 2, 2)
    swap_panel(
        ax_before,
        positions,
        nodes,
        [("A", "B"), ("C", "D")],
        COLORS["accent"],
        "dashed",
        2.2,
        "Before: edges A–B, C–D",
        (-0.4, 1.4),
        (-0.4, 1.4),
    )
    swap_panel(
        ax_after,
        positions,
        nodes,
        [("A", "D"), ("B", "C")],
        COLORS["accent2"],
        "solid",
        2.6,
        "After: edges A–D, B–C\n(degrees unchanged)",
        (-0.4, 1.4),
        (-0.4, 1.4),
    )
    fig.tight_layout()
    save_transparent(fig, "refine/fig05-2swap.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_3swap() -> None:
    positions: dict[str, tuple[float, float]] = {
        "A": (0.0, 1.0),
        "C": (1.0, 1.0),
        "E": (2.0, 1.0),
        "B": (0.0, 0.0),
        "D": (1.0, 0.0),
        "F": (2.0, 0.0),
    }
    nodes = ["A", "B", "C", "D", "E", "F"]
    figure = cast(Callable[..., Figure], plt.figure)
    fig: Figure = figure(figsize=(9, 4), facecolor=BG)
    ax_before: Axes = fig.add_subplot(1, 2, 1)
    ax_after: Axes = fig.add_subplot(1, 2, 2)
    swap_panel(
        ax_before,
        positions,
        nodes,
        [("A", "B"), ("C", "D"), ("E", "F")],
        COLORS["accent"],
        "dashed",
        2.2,
        "Before: edges A–B, C–D, E–F",
        (-0.5, 2.5),
        (-0.5, 1.5),
    )
    swap_panel(
        ax_after,
        positions,
        nodes,
        [("A", "C"), ("B", "E"), ("D", "F")],
        COLORS["accent2"],
        "solid",
        2.6,
        "After: 3-swap — A–C, B–E, D–F\n(all degrees unchanged)",
        (-0.5, 2.5),
        (-0.5, 1.5),
    )
    fig.tight_layout()
    save_transparent(fig, "refine/fig07-3swap.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# ===========================================================================
# fig09-dodeca-petersen.pdf  — depth-2 excision: dodecahedron -> Petersen
# ===========================================================================
K = 3
G_TARGET = 5
DEPTH = (G_TARGET - 1) // 2
EX_NODE_SIZE = 250

DODECAHEDRAL_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 10),
    (0, 19),
    (1, 2),
    (1, 8),
    (2, 3),
    (2, 6),
    (3, 4),
    (3, 19),
    (4, 5),
    (4, 17),
    (5, 6),
    (5, 15),
    (6, 7),
    (7, 8),
    (7, 14),
    (8, 9),
    (9, 10),
    (9, 13),
    (10, 11),
    (11, 12),
    (11, 18),
    (12, 13),
    (12, 16),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (17, 18),
    (18, 19),
]
PETERSEN_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 4),
    (0, 5),
    (1, 2),
    (1, 6),
    (2, 3),
    (2, 7),
    (3, 4),
    (3, 8),
    (4, 9),
    (5, 7),
    (5, 8),
    (6, 8),
    (6, 9),
    (7, 9),
]


def dodecahedral_graph() -> "nx.Graph[int]":
    graph: "nx.Graph[int]" = nx.Graph()
    graph.add_edges_from(DODECAHEDRAL_EDGES)
    return graph


def petersen_graph() -> "nx.Graph[int]":
    graph: "nx.Graph[int]" = nx.Graph()
    graph.add_edges_from(PETERSEN_EDGES)
    return graph


def girth_safe_repair(
    H: "nx.Graph[int]", deficit: dict[int, int], g: int
) -> "nx.Graph[int] | None":
    open_deficit = {v: c for v, c in deficit.items() if c > 0}
    if not open_deficit:
        return H

    def partners(u: int) -> list[int]:
        dist = cast(dict[int, int], nx.single_source_shortest_path_length(H, u))
        result: list[int] = []
        for v in open_deficit:
            if v == u or H.has_edge(u, v):
                continue
            if dist.get(v, math.inf) >= g - 1:
                result.append(v)
        return result

    pivot = min(open_deficit, key=lambda x: len(partners(x)))
    options = partners(pivot)
    if not options:
        return None
    for v in options:
        _ = H.add_edge(pivot, v)
        reduced = dict(open_deficit)
        reduced[pivot] -= 1
        reduced[v] -= 1
        solved = girth_safe_repair(H, reduced, g)
        if solved is not None:
            return solved
        H.remove_edge(pivot, v)
    return None


def build_excision() -> tuple[
    "nx.Graph[int]", int, set[int], set[int], "nx.Graph[int]", list[tuple[int, int]]
]:
    base = dodecahedral_graph()
    petersen = petersen_graph()
    for root in base.nodes():
        reach = cast(
            dict[int, int],
            nx.single_source_shortest_path_length(base, root, cutoff=DEPTH),
        )
        ball = set(reach.keys())
        if len(ball) != 10:
            continue
        bfs_tree = nx.bfs_tree(base, root, depth_limit=DEPTH)
        if not nx.is_tree(bfs_tree):
            continue
        leftover = base.copy()
        leftover.remove_nodes_from(ball)
        before = {frozenset(e) for e in leftover.edges()}
        deficit = {
            v: K - leftover.degree(v)
            for v in leftover.nodes()
            if leftover.degree(v) < K
        }
        deficient = set(deficit.keys())
        repaired = girth_safe_repair(leftover.copy(), deficit, G_TARGET)
        if repaired is None:
            continue
        added = sorted(
            cast(tuple[int, int], tuple(sorted(e)))
            for e in repaired.edges()
            if frozenset(e) not in before
        )
        regular = all(repaired.degree(v) == K for v in repaired.nodes())
        basis = nx.minimum_cycle_basis(repaired)
        girth = min(len(c) for c in basis)
        iso = nx.is_isomorphic(repaired, petersen)
        if repaired.number_of_nodes() == 10 and regular and girth == G_TARGET and iso:
            print(
                f"verified: root={root} regular={regular}"
                + f" girth={girth} isomorphic_to_petersen={iso}"
            )
            return base, root, ball, deficient, repaired, added
    raise RuntimeError("no root yielded a Petersen-isomorphic excision")


SCHLEGEL_RINGS: list[tuple[list[int], list[float], float]] = [
    ([0, 19, 3, 2, 1], [0.0, 1.0, 2.0, 3.0, 4.0], 1.0),
    ([10, 18, 4, 6, 8], [0.0, 1.0, 2.0, 3.0, 4.0], 0.66),
    ([11, 17, 5, 7, 9], [0.5, 1.5, 2.5, 3.5, 4.5], 0.40),
    ([12, 16, 15, 14, 13], [0.5, 1.5, 2.5, 3.5, 4.5], 0.16),
]


def schlegel_pos() -> dict[int, tuple[float, float]]:
    pos: dict[int, tuple[float, float]] = {}
    for vertices, indices, radius in SCHLEGEL_RINGS:
        for v, idx in zip(vertices, indices):
            a = math.radians(90.0 - idx * 72.0)
            pos[v] = (radius * math.cos(a), radius * math.sin(a))
    return pos


def bfs_tree_edges(
    graph: "nx.Graph[int]", root: int, depth: int
) -> set[frozenset[int]]:
    level = {root: 0}
    tree: set[frozenset[int]] = set()
    frontier = [root]
    for _ in range(depth):
        nxt: list[int] = []
        for u in frontier:
            for w in graph.neighbors(u):
                if w not in level:
                    level[w] = level[u] + 1
                    tree.add(frozenset((u, w)))
                    nxt.append(w)
        frontier = nxt
    return tree


def ex_petersen_pos() -> dict[int, tuple[float, float]]:
    pos: dict[int, tuple[float, float]] = {}
    for i in range(5):
        a = math.radians(90.0 - i * 72.0)
        pos[i] = (math.cos(a), math.sin(a))
        pos[i + 5] = (0.5 * math.cos(a), 0.5 * math.sin(a))
    return pos


def ex_draw_base(
    ax: Axes,
    base: "nx.Graph[int]",
    root: int,
    ball: set[int],
    deficient: set[int],
) -> None:
    pos = schlegel_pos()
    ball_fill = "#7a7a7a"  # mid grey for the radius-d tree being removed

    tree = bfs_tree_edges(base, root, DEPTH)
    tree_edges: list[tuple[int, int]] = []
    inside_horizontal: list[tuple[int, int]] = []
    cut_edges: list[tuple[int, int]] = []
    outside_edges: list[tuple[int, int]] = []
    for e in base.edges():
        edge = (e[0], e[1])
        both_in = edge[0] in ball and edge[1] in ball
        if frozenset(edge) in tree:
            tree_edges.append(edge)
        elif both_in:
            inside_horizontal.append(edge)
        elif edge[0] in ball or edge[1] in ball:
            cut_edges.append(edge)
        else:
            outside_edges.append(edge)

    _ = nx.draw_networkx_edges(
        base, pos, ax=ax, edgelist=outside_edges, edge_color=COLORS["edge"], width=1.8
    )
    _ = nx.draw_networkx_edges(
        base,
        pos,
        ax=ax,
        edgelist=inside_horizontal + cut_edges,
        edge_color="#b5b5b5",
        width=1.5,
        style="dashed",
    )
    _ = nx.draw_networkx_edges(
        base, pos, ax=ax, edgelist=tree_edges, edge_color=ball_fill, width=2.0
    )

    fill_of: dict[int, str] = {}
    for v in base.nodes():
        if v == root:
            fill_of[v] = COLORS["accent"]
        elif v in ball:
            fill_of[v] = ball_fill
        else:
            fill_of[v] = COLORS["node"]

    _ = nx.draw_networkx_nodes(
        base,
        pos,
        ax=ax,
        node_size=EX_NODE_SIZE,
        node_color=[fill_of[v] for v in base.nodes()],
        edgecolors=COLORS["label"],
        linewidths=1.0,
    )
    _ = nx.draw_networkx_nodes(
        base,
        pos,
        ax=ax,
        nodelist=sorted(deficient),
        node_size=EX_NODE_SIZE,
        node_color="none",
        edgecolors=COLORS["accent2"],
        linewidths=1.8,
    )
    text_fn = cast(Callable[..., Text], ax.text)
    for v in base.nodes():
        x, y = pos[v]
        _ = text_fn(
            x,
            y,
            str(v),
            ha="center",
            va="center",
            fontsize=7.5,
            color=label_color(fill_of[v]),
        )
    _ = ax.set_xlim(-1.18, 1.18)
    _ = ax.set_ylim(-1.18, 1.18)
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")


def ex_draw_petersen(
    ax: Axes,
    repaired: "nx.Graph[int]",
    added: list[tuple[int, int]],
    deficient: set[int],
) -> None:
    petersen = petersen_graph()
    matcher = nx.algorithms.isomorphism.GraphMatcher(repaired, petersen)
    if not matcher.is_isomorphic():
        raise RuntimeError("repaired graph is not isomorphic to Petersen")
    mapping = cast(dict[int, int], matcher.mapping)
    canonical_pos = ex_petersen_pos()
    pos = {v: canonical_pos[mapping[v]] for v in repaired.nodes()}

    added_set = {frozenset(e) for e in added}
    repair_edges = [
        cast(tuple[int, int], tuple(e))
        for e in repaired.edges()
        if frozenset(e) in added_set
    ]
    other_edges = [
        cast(tuple[int, int], tuple(e))
        for e in repaired.edges()
        if frozenset(e) not in added_set
    ]

    _ = nx.draw_networkx_edges(
        repaired, pos, ax=ax, edgelist=other_edges, edge_color=COLORS["edge"], width=1.8
    )
    _ = nx.draw_networkx_edges(
        repaired,
        pos,
        ax=ax,
        edgelist=repair_edges,
        edge_color=COLORS["accent"],
        width=2.0,
    )
    fill_of = {
        v: (COLORS["node_alt"] if v in deficient else COLORS["node"])
        for v in repaired.nodes()
    }
    _ = nx.draw_networkx_nodes(
        repaired,
        pos,
        ax=ax,
        node_size=EX_NODE_SIZE,
        node_color=[fill_of[v] for v in repaired.nodes()],
        edgecolors=COLORS["label"],
        linewidths=1.0,
    )
    text_fn = cast(Callable[..., Text], ax.text)
    for v in repaired.nodes():
        x, y = pos[v]
        _ = text_fn(
            x,
            y,
            str(v),
            ha="center",
            va="center",
            fontsize=7.5,
            color=label_color(fill_of[v]),
        )
    _ = ax.set_xlim(-1.18, 1.18)
    _ = ax.set_ylim(-1.18, 1.18)
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")


def make_excision() -> None:
    base, root, ball, deficient, repaired, added = build_excision()
    figure = cast(Callable[..., Figure], plt.figure)
    fig: Figure = figure(figsize=(6.4, 3.3), facecolor=BG)
    ax_base: Axes = fig.add_subplot(1, 2, 1)
    ax_petersen: Axes = fig.add_subplot(1, 2, 2)
    ex_draw_base(ax_base, base, root, ball, deficient)
    ex_draw_petersen(ax_petersen, repaired, added, deficient)
    fig.tight_layout()
    save_transparent(
        fig, "excision/fig09-dodeca-petersen.pdf", bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)


# ===========================================================================
# fig10-degree-vs-girth.pdf  — per-architecture degree vs. girth accuracy
# ===========================================================================
# Source: thesis chapters/09-results.typ, tab:results-degree + tab:results-mincycle.
ARCH_DATA: list[tuple[str, float, float]] = [
    ("SAGE", 1.000, 0.119),
    ("GPS-GIN", 0.998, 0.068),
    ("GPS-SAGE", 0.980, 0.068),
    ("GIN", 0.887, 0.055),
    ("Loopy", 0.512, 0.510),
    ("GPS-GCN", 0.457, 0.311),
    ("GCN", 0.382, 0.227),
]


def make_degree_vs_girth() -> None:
    archs = [d[0] for d in ARCH_DATA]
    degree = [d[1] for d in ARCH_DATA]
    girth = [d[2] for d in ARCH_DATA]
    x = np.arange(len(archs))
    width = 0.36

    figure = cast(Callable[..., Figure], plt.figure)
    fig: Figure = figure(figsize=(8, 4.2), facecolor=BG)
    ax: Axes = fig.add_subplot(1, 1, 1)
    _ = ax.set_facecolor(BG)

    # Muted, serious palette: slate navy vs. brick red.
    bars_deg = ax.bar(
        x - width / 2,
        degree,
        width,
        label="Degree",
        color="#1f3a5f",
        edgecolor="#16293f",
        linewidth=0.7,
    )
    bars_mc = ax.bar(
        x + width / 2,
        girth,
        width,
        label="Minimum-cycle",
        color="#8c3b3b",
        edgecolor="#5e2828",
        linewidth=0.7,
    )

    _ = ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#ededed", linewidth=0.7)
    _ = ax.set_xticks(x)
    _ = ax.set_xticklabels(archs, fontsize=9, color=INK)
    _ = ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _ = ax.set_yticklabels(
        ["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color=MUTED
    )
    _ = ax.set_ylim(0, 1.12)
    _ = ax.set_ylabel("Per-node accuracy", fontsize=10, color=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=MUTED)
    _ = ax.legend(fontsize=9, framealpha=0.8)

    for group in (bars_deg, bars_mc):
        for bar in group:
            h = bar.get_height()
            _ = ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.015,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=MUTED,
            )

    fig.tight_layout()
    save(fig, "degree-girth/fig10-degree-vs-girth.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# ===========================================================================
# fig-refine-before.pdf / fig-refine-after.pdf — 2-swap: before and after
# ===========================================================================
REFINE_BEFORE_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 0),  # triangle (short cycle)
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 5),
    (3, 6),
    (4, 6),
    (4, 7),
    (5, 7),
    (6, 7),
]
REFINE_SHORT_CYCLE: list[tuple[int, int]] = [(0, 1), (1, 2), (2, 0)]
REFINE_SWAP_REMOVE: list[tuple[int, int]] = [(1, 2), (4, 6)]
REFINE_SWAP_ADD: list[tuple[int, int]] = [(1, 6), (2, 4)]

_REFINE_SHORT_CYCLE_FS: set[frozenset[int]] = {frozenset(e) for e in REFINE_SHORT_CYCLE}
_REFINE_SWAP_REMOVE_FS: set[frozenset[int]] = {frozenset(e) for e in REFINE_SWAP_REMOVE}


def make_refine_before() -> None:
    g: nx.Graph[int] = nx.Graph()
    g.add_nodes_from(range(8))
    g.add_edges_from(REFINE_BEFORE_EDGES)
    pos = nx.circular_layout(g)
    figure = cast(Callable[..., Figure], plt.figure)
    fig: Figure = figure(figsize=(4.5, 4.5), facecolor="none")
    ax: Axes = fig.add_subplot(1, 1, 1)
    normal_edges = [
        e for e in REFINE_BEFORE_EDGES if frozenset(e) not in _REFINE_SHORT_CYCLE_FS
    ]
    _ = nx.draw_networkx_edges(
        g, pos, ax=ax, edgelist=normal_edges, edge_color=COLORS["edge"], width=1.8
    )
    _ = nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        edgelist=REFINE_SHORT_CYCLE,
        edge_color=COLORS["accent"],
        width=2.2,
        style="dashed",
    )
    _ = nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_size=NODE_SIZE,
        node_color=COLORS["node"],
        edgecolors=COLORS["label"],
        linewidths=1.4,
    )
    _ = nx.draw_networkx_labels(g, pos, ax=ax, font_color=COLORS["label"], font_size=9)
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    fig.tight_layout()
    save_transparent(fig, "refine/fig-refine-before.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_refine_after() -> None:
    after_edges_kept = [
        e for e in REFINE_BEFORE_EDGES if frozenset(e) not in _REFINE_SWAP_REMOVE_FS
    ]
    g_ref: nx.Graph[int] = nx.Graph()
    g_ref.add_nodes_from(range(8))
    g_ref.add_edges_from(REFINE_BEFORE_EDGES)  # for consistent layout
    pos = nx.circular_layout(g_ref)
    figure = cast(Callable[..., Figure], plt.figure)
    fig: Figure = figure(figsize=(4.5, 4.5), facecolor="none")
    ax: Axes = fig.add_subplot(1, 1, 1)
    _ = nx.draw_networkx_edges(
        g_ref,
        pos,
        ax=ax,
        edgelist=REFINE_SWAP_REMOVE,
        edge_color="#cccccc",
        width=1.4,
        style="dashed",
    )
    _ = nx.draw_networkx_edges(
        g_ref,
        pos,
        ax=ax,
        edgelist=after_edges_kept,
        edge_color=COLORS["edge"],
        width=1.8,
    )
    _ = nx.draw_networkx_edges(
        g_ref,
        pos,
        ax=ax,
        edgelist=REFINE_SWAP_ADD,
        edge_color=COLORS["accent"],
        width=2.4,
    )
    _ = nx.draw_networkx_nodes(
        g_ref,
        pos,
        ax=ax,
        node_size=NODE_SIZE,
        node_color=COLORS["node"],
        edgecolors=COLORS["label"],
        linewidths=1.4,
    )
    _ = nx.draw_networkx_labels(
        g_ref, pos, ax=ax, font_color=COLORS["label"], font_size=9
    )
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    fig.tight_layout()
    save_transparent(fig, "refine/fig-refine-after.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# ===========================================================================
# fig-excise-before.pdf / fig-excise-after.pdf — separate excision panels
# ===========================================================================
def make_excise_before() -> None:
    base, root, ball, deficient, _repaired, _added = build_excision()
    figure = cast(Callable[..., Figure], plt.figure)
    fig: Figure = figure(figsize=(4.5, 4.5), facecolor="none")
    ax: Axes = fig.add_subplot(1, 1, 1)
    ex_draw_base(ax, base, root, ball, deficient)
    fig.tight_layout()
    save_transparent(fig, "excision/fig-excise-before.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_excise_after() -> None:
    _base, _root, _ball, deficient, repaired, added = build_excision()
    figure = cast(Callable[..., Figure], plt.figure)
    fig: Figure = figure(figsize=(4.5, 4.5), facecolor="none")
    ax: Axes = fig.add_subplot(1, 1, 1)
    ex_draw_petersen(ax, repaired, added, deficient)
    fig.tight_layout()
    save_transparent(fig, "excision/fig-excise-after.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ===========================================================================
def main() -> None:
    plt.rcParams["figure.facecolor"] = BG
    plt.rcParams["axes.facecolor"] = BG
    make_message_passing()
    make_voltage_base()
    make_voltage_lift()
    make_2swap()
    make_3swap()
    make_excision()
    make_refine_before()
    make_refine_after()
    make_excise_before()
    make_excise_after()
    make_degree_vs_girth()


if __name__ == "__main__":
    main()

"""Generate four message-passing step PNGs for the docs page."""

import matplotlib

matplotlib.use("Agg")

import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "frontend" / "public" / "static"

# ── shared style (matches kg-graph-terms.svg) ─────────────────────────
BG = "#1A1A1A"
NODE_CENTER = "#3f6f8f"
NODE_NEIGHBOR = "#2f2f2f"
NODE_BORDER = "#cfcfcf"
EDGE_COLOR = "#8a8a8a"
ARROW_COLOR = "#4ade80"
TEXT_COLOR = "#d8d8d8"
TEXT_SUB = "#a9a9a9"
FONT = "sans-serif"
DPI = 200
NODE_LW = 3
EDGE_LW = 4

FIG_W, FIG_H = 6.0, 4.5

CENTER = (0, 0)
NEIGHBORS = {
    "w₁": (-1, 1),
    "w₂": (1, 1),
    "w₃": (-1, -1),
    "w₄": (1, -1),
}
NEIGHBOR_VALS = {"w₁": 0.25, "w₂": 0.10, "w₃": 0.40, "w₄": 0.05}
CENTER_VAL = 0.30
UPDATED_CENTER = 0.57
UPDATED_NEIGHBORS = {"w₁": 0.44, "w₂": 0.19, "w₃": 0.62, "w₄": 0.28}


def _make_fig():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _save(fig, name):
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(OUT / name, dpi=DPI, facecolor=BG)
    plt.close(fig)


def _draw_node(ax, pos, label, value=None, radius=0.28, is_center=False, fontsize=None):
    """Draw a circle with text inside. If value is given, show label+value on two lines."""
    color = NODE_CENTER if is_center else NODE_NEIGHBOR
    circle = plt.Circle(pos, radius, fc=color, ec=NODE_BORDER, lw=NODE_LW, zorder=4)
    ax.add_patch(circle)
    if fontsize is None:
        fontsize = 15 if is_center else 12
    if value is not None:
        ax.text(
            *pos,
            f"{label}\n{value:.2f}",
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=TEXT_COLOR,
            fontfamily=FONT,
            zorder=5,
            linespacing=1.3,
        )
    else:
        ax.text(
            *pos,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=TEXT_COLOR,
            fontfamily=FONT,
            zorder=5,
        )


def _draw_edge(ax, p1, p2, color=EDGE_COLOR, lw=EDGE_LW):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, zorder=1)


def _draw_arrow(
    ax, start, end, color=ARROW_COLOR, lw=2, radius_a=0.28, radius_b=0.28, gap=0.06
):
    """Draw arrow from edge of circle A to edge of circle B, with a gap."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist
    x0 = start[0] + ux * (radius_a + gap)
    y0 = start[1] + uy * (radius_a + gap)
    x1 = end[0] - ux * (radius_b + gap)
    y1 = end[1] - uy * (radius_b + gap)
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="->,head_width=5,head_length=4",
        color=color,
        lw=lw,
        zorder=3,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=1,
    )
    ax.add_patch(arrow)


# ── Step 1: Node states ──────────────────────────────────────────────
def step1():
    fig, ax = _make_fig()
    for name, pos in NEIGHBORS.items():
        _draw_edge(ax, CENTER, pos)
    _draw_node(ax, CENTER, "v", CENTER_VAL, radius=0.32, is_center=True)
    for name, pos in NEIGHBORS.items():
        _draw_node(ax, pos, name, NEIGHBOR_VALS[name])
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.6, 1.6)
    _save(fig, "mpnn-step-1.png")


# ── Step 2: Messages sent to center ──────────────────────────────────
def step2():
    fig, ax = _make_fig()
    for name, pos in NEIGHBORS.items():
        _draw_arrow(ax, pos, CENTER, radius_a=0.28, radius_b=0.32, gap=0.08)
    _draw_node(ax, CENTER, "v", radius=0.32, is_center=True)
    for name, pos in NEIGHBORS.items():
        _draw_node(ax, pos, name, None)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.6, 1.6)
    _save(fig, "mpnn-step-2.png")


# ── Step 3: Aggregate + Update ────────────────────────────────────────
def step3():
    fig, ax = _make_fig()
    msg_labels = ["m₁", "m₂", "m₃", "m₄"]
    msg_ys = [1.2, 0.4, -0.4, -1.2]
    msg_x = -1.8
    agg_pos = (0, 0)
    out_pos = (1.8, 0)

    # message nodes
    for label, y in zip(msg_labels, msg_ys):
        _draw_node(ax, (msg_x, y), label, radius=0.22, fontsize=12)

    # aggregation node
    agg_circle = plt.Circle(
        agg_pos, 0.45, fc="#355f78", ec=NODE_BORDER, lw=NODE_LW, zorder=4
    )
    ax.add_patch(agg_circle)
    ax.text(
        0,
        0.1,
        "Σ + U",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
        fontfamily=FONT,
        zorder=5,
    )
    ax.text(
        0,
        -0.18,
        "(hᵗᵥ, mᵥ)",
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT_SUB,
        fontfamily=FONT,
        zorder=5,
    )

    # output node
    out_circle = plt.Circle(
        out_pos, 0.35, fc=NODE_CENTER, ec=NODE_BORDER, lw=NODE_LW, zorder=4
    )
    ax.add_patch(out_circle)
    ax.text(
        out_pos[0],
        out_pos[1] + 0.07,
        "v",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
        fontfamily=FONT,
        zorder=5,
    )
    ax.text(
        out_pos[0],
        out_pos[1] - 0.14,
        "hᵗ⁺¹",
        ha="center",
        va="center",
        fontsize=11,
        color=TEXT_COLOR,
        fontfamily=FONT,
        zorder=5,
    )

    # arrows from messages to aggregator
    for y in msg_ys:
        _draw_arrow(ax, (msg_x, y), agg_pos, radius_a=0.22, radius_b=0.45, gap=0.08)

    # arrow from aggregator to output
    _draw_arrow(ax, agg_pos, out_pos, radius_a=0.45, radius_b=0.35, gap=0.08)

    ax.set_xlim(-2.4, 2.5)
    ax.set_ylim(-1.7, 1.7)
    _save(fig, "mpnn-step-3.png")


# ── Step 4: Layer t → Layer t+1 ──────────────────────────────────────
def step4():
    fig, ax = _make_fig()
    offset_l = -1.6
    offset_r = 1.6
    spread = 0.7

    def draw_cluster(ax, cx, vals_center, vals_neighbors):
        c = (cx, 0)
        nb_pos = {
            "w₁": (cx - spread, spread),
            "w₂": (cx + spread, spread),
            "w₃": (cx - spread, -spread),
            "w₄": (cx + spread, -spread),
        }
        for name, pos in nb_pos.items():
            _draw_edge(ax, c, pos, lw=3)
        _draw_node(ax, c, "v", vals_center, radius=0.28, is_center=True, fontsize=12)
        for name, pos in nb_pos.items():
            val = vals_neighbors[name]
            # Small neighbor: just value, no label
            _draw_node(ax, pos, f"{val:.2f}", radius=0.18, fontsize=7)

    # Layer labels
    ax.text(
        offset_l,
        1.3,
        "Layer t",
        ha="center",
        fontsize=13,
        color=TEXT_SUB,
        fontfamily=FONT,
        fontweight="bold",
    )
    ax.text(
        offset_r,
        1.3,
        "Layer t+1",
        ha="center",
        fontsize=13,
        color=TEXT_SUB,
        fontfamily=FONT,
        fontweight="bold",
    )

    draw_cluster(ax, offset_l, CENTER_VAL, NEIGHBOR_VALS)
    draw_cluster(ax, offset_r, UPDATED_CENTER, UPDATED_NEIGHBORS)

    # big arrow between clusters
    _draw_arrow(
        ax, (offset_l, 0), (offset_r, 0), lw=2.5, radius_a=0.28, radius_b=0.28, gap=0.08
    )

    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-1.3, 1.6)
    _save(fig, "mpnn-step-4.png")


if __name__ == "__main__":
    step1()
    step2()
    step3()
    step4()
    print(f"Wrote 4 PNGs to {OUT}")

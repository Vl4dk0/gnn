"""Generate a two-panel figure illustrating why GNNs cannot predict girth.

Panel A: A hexagon (6-cycle, girth g=6) with vertex v highlighted. The
L=2 hop receptive field is shown as a translucent disk that covers v and
its two hops in each direction — but does NOT reach the antipodal vertex
closing the cycle.  The unreachable arc is drawn in bronze to signal the
structural blindness.

Panel B: Two bars — degree accuracy 100 % (navy, local, solved) vs girth
accuracy 51 % (bronze, global, best model).

Output: presentation/figures/gnn/fig-girth-receptive.pdf
"""

from pathlib import Path
from typing import Callable, cast

import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle

matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# ---------------------------------------------------------------------------
# Palette (matches house style)
# ---------------------------------------------------------------------------
NAVY = "#1f3a5f"
STEEL = "#4a6572"
BRONZE = "#8c3b3b"
GREY = "#888888"
EDGE_GREY = "#d0d0d0"
TEXT_DARK = "#1a1a1a"
TEXT_MUTED = "#555555"
WHITE = "#ffffff"


# ---------------------------------------------------------------------------
# Panel A helpers
# ---------------------------------------------------------------------------

N_CYCLE = 6
# Vertices arranged clockwise starting at the top
# Index 0 = top ("v"), indices go clockwise
RADIUS = 1.0


def hex_pos(i: int, n: int = N_CYCLE) -> tuple[float, float]:
    """Return (x, y) for vertex i of a regular n-gon, top at index 0."""
    angle = math.pi / 2 - 2 * math.pi * i / n
    return math.cos(angle) * RADIUS, math.sin(angle) * RADIUS


ALL_POS: list[tuple[float, float]] = [hex_pos(i) for i in range(N_CYCLE)]

# For g=6, floor(g/2) = 3, so L=2 < 3 means the 2-hop ball doesn't see vertex 3
# (the antipodal vertex).  The 2-hop neighborhood of v=0 is {0, 1, 5, 2, 4}.
# Vertex 3 is the only one outside.
CENTER_IDX = 0  # highlighted vertex v
L_HOPS = 2
VISIBLE_IDX: set[int] = {0, 1, 5, 2, 4}  # 2-hop ball of vertex 0
BLIND_IDX: set[int] = {3}  # outside the receptive field


def draw_hexagon(ax: Axes) -> None:
    """Draw the g=6 cycle with shaded receptive field and bronze blind arc."""
    plot = cast(Callable[..., object], ax.plot)
    scatter = cast(Callable[..., object], ax.scatter)
    text = cast(Callable[..., object], ax.text)

    # --- Receptive-field background disk (translucent, centred at v) ---------
    # The 2-hop ball of v=0 spans vertices 0,1,2,4,5.  The convex hull of
    # those five points fits well inside a circle of radius ~1.15*RADIUS.
    disk = Circle(
        (ALL_POS[CENTER_IDX][0], ALL_POS[CENTER_IDX][1]),
        radius=1.18,
        color=STEEL,
        alpha=0.13,
        zorder=1,
        linewidth=0,
    )
    add_patch = cast(Callable[..., object], ax.add_patch)
    _ = add_patch(disk)

    # Dashed border of the disk
    disk_border = Circle(
        (ALL_POS[CENTER_IDX][0], ALL_POS[CENTER_IDX][1]),
        radius=1.18,
        fill=False,
        edgecolor=STEEL,
        linewidth=1.4,
        linestyle="--",
        zorder=2,
    )
    _ = add_patch(disk_border)

    # --- Edges ----------------------------------------------------------------
    # Edges in the visible ball: grey
    # The arc that v can't see (edge 2-3 and 3-4): bronze
    def edge_color(u: int, v_idx: int) -> str:
        if u in BLIND_IDX or v_idx in BLIND_IDX:
            return BRONZE
        return EDGE_GREY

    for i in range(N_CYCLE):
        j = (i + 1) % N_CYCLE
        xi, yi = ALL_POS[i]
        xj, yj = ALL_POS[j]
        _ = plot(
            [xi, xj],
            [yi, yj],
            color=edge_color(i, j),
            linewidth=2.2,
            zorder=3,
            solid_capstyle="round",
        )

    # --- Nodes ----------------------------------------------------------------
    for i, (x, y) in enumerate(ALL_POS):
        if i == CENTER_IDX:
            color = NAVY
            sz = 280
        elif i in BLIND_IDX:
            color = BRONZE
            sz = 200
        else:
            color = STEEL
            sz = 200
        _ = scatter(x, y, s=sz, color=color, zorder=4)

    # --- Vertex labels --------------------------------------------------------
    # Only label v (vertex 0)
    vx, vy = ALL_POS[CENTER_IDX]
    _ = text(
        vx - 0.04,
        vy + 0.14,
        "v",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color=NAVY,
        zorder=5,
    )

    # Label the blind vertex
    bx, by = ALL_POS[3]
    _ = text(
        bx,
        by - 0.15,
        "neviditeľný",
        ha="center",
        va="top",
        fontsize=9,
        color=BRONZE,
        zorder=5,
    )

    # --- Annotation: "L-hop dosah" inside the disk ---------------------------
    _ = text(
        -0.55,
        0.5,
        "L-hop dosah",
        ha="center",
        va="center",
        fontsize=10,
        color=STEEL,
        fontstyle="italic",
        zorder=5,
    )

    # --- Inequality label at bottom of disk -----------------------------------
    _ = text(
        0.0,
        -1.52,
        r"L < $\lfloor$g/2$\rfloor$",
        ha="center",
        va="top",
        fontsize=13,
        color=TEXT_DARK,
        zorder=5,
    )

    # "girth g=6" label for the cycle
    _ = text(
        0.0,
        -1.75,
        "girth g = 6",
        ha="center",
        va="top",
        fontsize=10,
        color=TEXT_MUTED,
        zorder=5,
    )


# ---------------------------------------------------------------------------
# Panel B helpers
# ---------------------------------------------------------------------------

BAR_LABELS = ["Stupeň\n(lokálny)", "Obvod\n(globálny)"]
BAR_VALUES = [100.0, 51.0]
BAR_COLORS = [NAVY, BRONZE]


def draw_bars(ax: Axes) -> None:
    """Draw the two accuracy bars."""
    bar_fn = cast(Callable[..., object], ax.bar)
    text = cast(Callable[..., object], ax.text)

    _ = bar_fn(
        BAR_LABELS,
        BAR_VALUES,
        color=BAR_COLORS,
        width=0.5,
        zorder=2,
    )

    # Value labels inside bars
    for i, val in enumerate(BAR_VALUES):
        _ = text(
            i,
            val - 5,
            f"{val:.0f} %",
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=WHITE,
            zorder=3,
        )

    set_ylim = cast(Callable[..., object], ax.set_ylim)
    _ = set_ylim(0, 115)

    set_ylabel = cast(Callable[..., object], ax.set_ylabel)
    _ = set_ylabel("presnosť (%)", fontsize=12)

    tick_params = cast(Callable[..., object], ax.tick_params)
    _ = tick_params(labelsize=11)

    set_yticks = cast(Callable[..., object], ax.set_yticks)
    _ = set_yticks([0, 25, 50, 75, 100])

    for side in ("top", "right"):
        spine = ax.spines[side]
        set_vis = cast(Callable[..., object], spine.set_visible)
        _ = set_vis(False)

    set_facecolor = cast(Callable[..., object], ax.set_facecolor)
    _ = set_facecolor(WHITE)

    set_title = cast(Callable[..., object], ax.set_title)
    _ = set_title("Lokálne sa dá,\nglobálne nie", fontsize=12, color=TEXT_DARK, pad=8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    subplots_fn = cast(
        Callable[..., tuple[Figure, list[Axes]]],
        plt.subplots,
    )
    fig, axes = subplots_fn(
        1,
        2,
        figsize=(9, 4.5),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )
    ax_a: Axes = axes[0]
    ax_b: Axes = axes[1]

    set_facecolor_fig = cast(Callable[..., object], fig.patch.set_facecolor)
    _ = set_facecolor_fig(WHITE)

    # --- Panel A --------------------------------------------------------------
    _ = ax_a.axis("off")
    set_xlim = cast(Callable[..., object], ax_a.set_xlim)
    _ = set_xlim(-1.85, 1.85)
    set_ylim = cast(Callable[..., object], ax_a.set_ylim)
    _ = set_ylim(-1.95, 1.35)

    draw_hexagon(ax_a)

    set_title_a = cast(Callable[..., object], ax_a.set_title)
    _ = set_title_a(
        "Receptívne pole GNN (L = 2 vrstvy) vs. girth g = 6",
        fontsize=12,
        color=TEXT_DARK,
        pad=10,
    )

    # Panel A legend entries (manual patches)
    legend_handles = [
        mpatches.Patch(color=NAVY, label="vrchol v"),
        mpatches.Patch(color=STEEL, alpha=0.55, label="L-hop dosah (viditeľné)"),
        mpatches.Patch(color=BRONZE, label="mimo dosahu (neviditeľné)"),
    ]
    legend_fn = cast(Callable[..., object], ax_a.legend)
    _ = legend_fn(
        handles=legend_handles,
        loc="upper right",
        fontsize=9,
        frameon=False,
    )

    # --- Panel B --------------------------------------------------------------
    draw_bars(ax_b)

    # --- Overall title --------------------------------------------------------
    suptitle = cast(Callable[..., object], fig.suptitle)
    _ = suptitle(
        "Obvod je globálny — lokálne videnie nestačí",
        fontsize=14,
        fontweight="bold",
        color=TEXT_DARK,
        y=1.02,
    )

    tight_layout = cast(Callable[..., object], fig.tight_layout)
    _ = tight_layout()

    out = FIGURES / "gnn" / "fig-girth-receptive.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(
        out,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor=WHITE,
    )
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

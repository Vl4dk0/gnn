"""Generate the data-driven defense figures (greyscale, website palette).

Produces three vector PDFs into presentation/figures/:

  fig-architectures.pdf  — grouped bar chart, per-architecture accuracy on the
                           degree task vs. the minimum-cycle (girth) task. Real
                           numbers from the thesis results tables.
  fig-results.pdf        — coverage heatmap over the 22 (k,g) targets x methods,
                           cells shaded by who solves what. The payoff visual.

Design: greyscale only, white background, deterministic, vector PDF.
"""

from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# Embed TrueType fonts in the PDF (type 42) so text stays selectable and sharp.
matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --- Palette. Explainer charts (fig-architectures, fig-results) use a muted,
#     serious palette (slate navy / dark bronze / a dark sequential colormap).
#     Deck chrome stays grey. ---
BG = "#ffffff"
INK = "#1a1a1a"
MUTED = "#555555"
LINE = "#d0d0d0"
# Muted explainer-chart accents (fig-architectures bars).
NAVY = "#1f3a5f"
BRONZE = "#8c3b3b"


# ===========================================================================
# fig-architectures.png — degree vs. girth accuracy, grouped bars
# ===========================================================================
# (architecture, degree_acc, mincycle_acc), ordered by degree_acc desc.
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


def draw_architectures() -> Figure:
    archs = [d[0] for d in ARCH_DATA]
    degree = [d[1] for d in ARCH_DATA]
    girth = [d[2] for d in ARCH_DATA]

    x = np.arange(len(archs))
    width = 0.38

    figure = cast(Callable[..., Figure], plt.figure)
    fig: Figure = figure(figsize=(9.2, 4.4), facecolor=BG)
    ax: Axes = fig.add_subplot(1, 1, 1)
    _ = ax.set_facecolor(BG)

    bars_d = ax.bar(
        x - width / 2,
        degree,
        width,
        label="Stupeň",
        color=NAVY,
        edgecolor="#16293f",
        linewidth=0.6,
    )
    bars_g = ax.bar(
        x + width / 2,
        girth,
        width,
        label="Obvod",
        color=BRONZE,
        edgecolor="#5e2828",
        linewidth=0.6,
    )

    _ = ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#ededed", linewidth=0.8)
    _ = ax.set_xticks(x)
    _ = ax.set_xticklabels(archs, fontsize=11, color=INK)
    _ = ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _ = ax.set_yticklabels(
        ["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10, color=MUTED
    )
    _ = ax.set_ylim(0, 1.14)
    _ = ax.set_ylabel("Presnosť (per vrchol)", fontsize=11, color=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=MUTED)
    _ = ax.legend(fontsize=11, framealpha=0.9, loc="upper right")

    for group in (bars_d, bars_g):
        for bar in group:
            h = bar.get_height()
            _ = ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.018,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=MUTED,
            )

    fig.tight_layout()
    return fig


# ===========================================================================
# fig-results.png — coverage heatmap: methods x (k,g) targets
# ===========================================================================
# Targets in the exact row order of the thesis coverage table (tab:results-time
# in chapters/09-results.typ). Ordered roughly by Moore size / difficulty.
TARGETS: list[str] = [
    "(5,3)",
    "(6,3)",
    "(7,3)",
    "(3,5)",
    "(5,4)",
    "(6,4)",
    "(3,6)",
    "(7,4)",
    "(4,5)",
    "(3,7)",
    "(4,6)",
    "(5,5)",
    "(3,8)",
    "(6,5)",
    "(5,6)",
    "(3,9)",
    "(7,5)",
    "(4,7)",
    "(3,10)",
    "(6,6)",
    "(4,8)",
    "(5,7)",
]

# Three targets nobody solves within the 60 s budget (blank in every column).
UNSOLVED_BY_ALL: set[str] = {"(7,5)", "(4,7)", "(5,7)"}

# Columns = methods (display order). Solved sets from tab:results-time.
# The thesis splits voltage into v-rl / v-girth / v-tabu / v-alg; here the three
# classical voltage producers are merged into one "voltage" column (their union)
# while voltage-rl keeps its own column, since the deck contrasts those two.
METHODS: list[str] = [
    "A*",
    "RandomWalk",
    "Bruteforce",
    "directRL",
    "voltage",
    "voltage-rl",
    "Forge",
]

SOLVED: dict[str, set[str]] = {
    "A*": {
        "(5,3)",
        "(6,3)",
        "(7,3)",
        "(3,5)",
        "(5,4)",
        "(6,4)",
        "(3,6)",
        "(7,4)",
        "(3,7)",
        "(4,6)",
        "(3,8)",
        "(5,6)",
    },
    "RandomWalk": {
        "(5,3)",
        "(6,3)",
        "(7,3)",
        "(3,5)",
        "(5,4)",
        "(6,4)",
        "(3,6)",
        "(7,4)",
        "(4,6)",
        "(3,8)",
        "(5,6)",
        "(6,6)",
    },
    "Bruteforce": {"(3,6)", "(4,6)", "(3,8)"},
    "directRL": {"(5,3)", "(6,3)", "(3,5)", "(5,4)", "(3,6)"},
    # voltage family (girth/tabu/algebraic combined reach)
    "voltage": {
        "(5,3)",
        "(6,3)",
        "(7,3)",
        "(3,5)",
        "(5,4)",
        "(6,4)",
        "(3,6)",
        "(7,4)",
        "(4,5)",
        "(3,7)",
        "(4,6)",
        "(5,5)",
        "(3,8)",
        "(5,6)",
    },
    "voltage-rl": {
        "(5,3)",
        "(6,3)",
        "(7,3)",
        "(3,5)",
        "(5,4)",
        "(6,4)",
        "(3,6)",
        "(7,4)",
        "(4,5)",
        "(3,7)",
        "(4,6)",
        "(5,5)",
        "(3,8)",
        "(5,6)",
        "(6,5)",
        "(3,9)",
        "(3,10)",
        "(6,6)",
        "(4,8)",
    },
    "Forge": {
        "(5,3)",
        "(6,3)",
        "(7,3)",
        "(3,5)",
        "(5,4)",
        "(6,4)",
        "(3,6)",
        "(7,4)",
        "(4,5)",
        "(3,7)",
        "(4,6)",
    },
}


def draw_results() -> Figure:
    n_t = len(TARGETS)
    n_m = len(METHODS)
    solved_grid: list[list[bool]] = [[t in SOLVED[m] for m in METHODS] for t in TARGETS]

    figure = cast(Callable[..., Figure], plt.figure)
    fig: Figure = figure(figsize=(9.4, 6.4), facecolor=BG)
    ax: Axes = fig.add_subplot(1, 1, 1)
    _ = ax.set_facecolor(BG)

    # Navy sequential colormap (light -> deep navy), coherent with the deck's
    # dark-blue accent. A solved cell is filled with a tone whose depth tracks
    # the target's difficulty rank, so solved cells stand out against the
    # neutral light-grey unsolved cells while keeping a sober look.
    cmap = plt.get_cmap("Blues")
    for i in range(n_t):
        # depth in [0.12, 0.95] by difficulty (top rows easy, bottom rows hard)
        depth = 0.12 + 0.83 * (i / max(n_t - 1, 1))
        solved_color = cmap(depth)
        for j in range(n_m):
            solved = solved_grid[i][j]
            _ = ax.add_patch(
                Rectangle(
                    (j, n_t - 1 - i),
                    1,
                    1,
                    facecolor=(solved_color if solved else "#ececed"),
                    edgecolor="#ffffff",
                    linewidth=1.5,
                )
            )

    _ = ax.set_xlim(0, n_m)
    _ = ax.set_ylim(0, n_t)
    _ = ax.set_xticks([j + 0.5 for j in range(n_m)])
    _ = ax.set_xticklabels(METHODS, fontsize=9, color=INK, rotation=55, ha="left")
    _ = ax.set_yticks([n_t - 1 - i + 0.5 for i in range(n_t)])
    _ = ax.set_yticklabels(TARGETS, fontsize=9.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # Compress the 22-row grid vertically (cells wider than tall) so the figure
    # is landscape enough to fit the slide content band without clipping the
    # column headers. A square aspect would make the figure far too tall.
    _ = ax.set_aspect(0.42)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Targets nobody solves within 60 s are fully empty rows. Tag each with a
    # side note so the row order can follow the thesis table faithfully (the
    # three unsolved targets are not contiguous at the bottom).
    for i, t in enumerate(TARGETS):
        if t in UNSOLVED_BY_ALL:
            y = n_t - 1 - i + 0.5
            _ = ax.annotate(
                "nikto do 60 s",
                xy=(n_m + 0.25, y),
                fontsize=8.5,
                color=MUTED,
                va="center",
                ha="left",
                annotation_clip=False,
            )

    _ = ax.set_xlim(0, n_m + 2.4)
    fig.tight_layout()
    return fig


# ===========================================================================
# fig-forge.pdf — the three-stage Forge pipeline as a compact thumbnail
# ===========================================================================
# Mirrors the inline Forge diagram on the deck's Forge slide, exported as a
# small static figure for the conclusion slide. Masculine explainer palette.
STEEL = "#4a6572"
STEELFILL = "#eef1f3"
FORGE_STAGES: list[tuple[str, str, str]] = [
    ("Producer", "voltage", NAVY),
    ("Refinement", "oprava", STEEL),
    ("Excision", "veľkosť", NAVY),
]


def draw_forge() -> Figure:
    figure = cast(Callable[..., Figure], plt.figure)
    fig: Figure = figure(figsize=(7.2, 1.9), facecolor=BG)
    ax: Axes = fig.add_subplot(1, 1, 1)
    _ = ax.set_facecolor(BG)

    box_w, box_h, gap = 1.9, 1.0, 0.9
    pitch = box_w + gap
    for i, (title, sub, edge) in enumerate(FORGE_STAGES):
        x = i * pitch
        _ = ax.add_patch(
            FancyBboxPatch(
                (x, 0.0),
                box_w,
                box_h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=STEELFILL,
                edgecolor=edge,
                linewidth=1.6,
            )
        )
        _ = ax.text(
            x + box_w / 2,
            box_h * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=12,
            color=INK,
            fontweight="bold",
        )
        _ = ax.text(
            x + box_w / 2,
            box_h * 0.27,
            sub,
            ha="center",
            va="center",
            fontsize=9,
            color=MUTED,
        )
        if i < len(FORGE_STAGES) - 1:
            _ = ax.add_patch(
                FancyArrowPatch(
                    (x + box_w, box_h / 2),
                    (x + box_w + gap, box_h / 2),
                    arrowstyle="-|>",
                    mutation_scale=16,
                    color=BRONZE,
                    linewidth=2.0,
                    shrinkA=0,
                    shrinkB=0,
                )
            )

    _ = ax.set_xlim(-0.2, len(FORGE_STAGES) * pitch - gap + 0.2)
    _ = ax.set_ylim(-0.3, box_h + 0.3)
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    fig.tight_layout()
    return fig


# ===========================================================================
def save(fig: Figure, name: str) -> None:
    (FIGURES / name).parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(
        FIGURES / name,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor=BG,
    )
    print(f"saved {FIGURES / name}")


def save_transparent(fig: Figure, name: str) -> None:
    """Save with no background (transparent PDF), for graph drawings / diagrams."""
    (FIGURES / name).parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(
        FIGURES / name,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.12,
        transparent=True,
        facecolor="none",
    )
    print(f"saved {FIGURES / name}")


def main() -> None:
    plt.rcParams["figure.facecolor"] = BG
    plt.rcParams["axes.facecolor"] = BG
    plt.rcParams["font.size"] = 11

    f1 = draw_architectures()
    save(f1, "architectures/fig-architectures.pdf")
    plt.close(f1)

    f3 = draw_results()
    save(f3, "results/fig-results.pdf")
    plt.close(f3)

    f4 = draw_forge()
    save_transparent(f4, "forge/fig-forge.pdf")
    plt.close(f4)


if __name__ == "__main__":
    main()

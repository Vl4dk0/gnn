"""Generate the edge-swap refinement animation frames for the defense deck.

Self-contained (matplotlib + networkx only). Produces transparent greyscale
vector PDFs into presentation/figures/refine/.

The exact example (same as the frontend /refine demo):
  A 3-regular graph on 14 vertices (0..13), girth 3, target girth 5.
  A 2-switch removes two independent edges (u,v),(x,y) on four distinct
  vertices and adds (u,x),(v,y), preserving every degree. We use 2-switches to
  destroy short cycles and raise the girth to 5.

Deterministic greedy 2-switch sequence (verified, reproducible) that raises
this graph's girth to 5 in two swaps:
  swap 0: remove (0,2),(12,13)  add (0,12),(2,13)   cost 10 -> 2
  swap 1: remove (0,7),(4,9)    add (0,4),(7,9)      cost  2 -> 0
The final graph is 3-regular with girth 5 (verified at generation time).

ONE fixed layout for the 14 vertices is used across ALL frames.

Frames:
  refine-0.pdf      — starting graph; one short cycle (triangle 0-2-7) bold/dark.
  refine-1a.pdf     — swap 1: the two removed edges drawn bold + dashed.
  refine-1b.pdf     — after swap 1: the two new edges bold solid, rest grey.
  refine-2a.pdf     — swap 2: the two removed edges drawn bold + dashed.
  refine-2b.pdf     — after swap 2: the two new edges bold solid, rest grey.
  refine-final.pdf  — final graph, no highlights, girth 5.
"""

from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Embed TrueType fonts so text stays selectable and sharp.
matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures" / "refine"

# --- Greyscale palette (deck graph palette) ---
INK = "#1a1a1a"  # bold/dark highlighted edges
GREY_EDGE = "#bdbdbd"  # ordinary (unhighlighted) edges
NODE_FILL = "#ffffff"
NODE_OUTLINE = "#3a3a3a"
LABEL = "#1a1a1a"

GREY_LW = 1.6
BOLD_LW = 3.6
NODE_SIZE = 360.0

# --- The example graph -------------------------------------------------------
START_EDGES: list[tuple[int, int]] = [
    (0, 2), (0, 5), (0, 7), (1, 5), (1, 9), (1, 13), (2, 7), (2, 8),
    (3, 4), (3, 6), (3, 8), (4, 7), (4, 9), (5, 11), (6, 9), (6, 12),
    (8, 11), (10, 11), (10, 12), (10, 13), (12, 13),
]  # fmt: skip

# Deterministic verified 2-switch sequence (removed, added) per swap.
SWAPS: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = [
    ([(0, 2), (12, 13)], [(0, 12), (2, 13)]),
    ([(0, 7), (4, 9)], [(0, 4), (7, 9)]),
]

# A short cycle present in the start graph that the first swap destroys.
HIGHLIGHT_CYCLE: list[int] = [0, 2, 7]


def _ekey(e: tuple[int, int]) -> tuple[int, int]:
    return (min(e), max(e))


def make_fig() -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    # Explicit limits that bound every node ([-1, 1]) plus padding, so no
    # vertex, circle or label is ever clipped at a frame edge.
    _ = ax.set_xlim(-AXIS_LIM, AXIS_LIM)
    _ = ax.set_ylim(-AXIS_LIM, AXIS_LIM)
    return fig, ax


def save(fig: Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
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
    plt.close(fig)


def compute_layout() -> dict[int, tuple[float, float]]:
    """One fixed deterministic layout for all 14 vertices, reused everywhere.

    A tuned spring layout, then rescaled so every node sits inside [-1, 1] on
    both axes (centered, symmetric). Combined with the explicit axis limits in
    ``make_fig`` this guarantees no node is ever clipped at a frame edge.
    """
    G = nx.Graph()
    G.add_nodes_from(range(14))
    G.add_edges_from(START_EDGES)
    raw = nx.spring_layout(G, seed=7, k=1.1, iterations=400)

    xs = [float(p[0]) for p in raw.values()]
    ys = [float(p[1]) for p in raw.values()]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0 or 1.0

    pos: dict[int, tuple[float, float]] = {}
    for n, p in raw.items():
        pos[int(n)] = ((float(p[0]) - cx) / half, (float(p[1]) - cy) / half)
    return pos


POS = compute_layout()

# Drawing extent: nodes live in [-1, 1]; pad so circles + labels never clip.
AXIS_LIM = 1.18


def draw_nodes(ax: Axes) -> None:
    for n, (x, y) in POS.items():
        ax.scatter(
            [x], [y],
            s=NODE_SIZE, facecolors=NODE_FILL, edgecolors=NODE_OUTLINE,
            linewidths=1.4, zorder=3,
        )  # fmt: skip
        ax.text(
            x, y, str(n),
            ha="center", va="center", fontsize=9.0, color=LABEL, zorder=4,
        )  # fmt: skip


def draw_edge(
    ax: Axes,
    e: tuple[int, int],
    *,
    color: str,
    lw: float,
    dashed: bool = False,
) -> None:
    (x0, y0), (x1, y1) = POS[e[0]], POS[e[1]]
    ax.plot(
        [x0, x1], [y0, y1],
        color=color, linewidth=lw, zorder=2,
        linestyle=(0, (5, 3.5)) if dashed else "-",
        solid_capstyle="round", dash_capstyle="round",
    )  # fmt: skip


def cycle_edges(cycle: list[int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for i in range(len(cycle)):
        out.add(_ekey((cycle[i], cycle[(i + 1) % len(cycle)])))
    return out


def render(
    name: str,
    edges: list[tuple[int, int]],
    *,
    bold_solid: set[tuple[int, int]] | None = None,
    bold_dashed: set[tuple[int, int]] | None = None,
) -> None:
    """Render a frame: grey edges, with given sets drawn bold (solid/dashed)."""
    bold_solid = bold_solid or set()
    bold_dashed = bold_dashed or set()
    fig, ax = make_fig()
    # Grey base edges first.
    for e in edges:
        k = _ekey(e)
        if k in bold_solid or k in bold_dashed:
            continue
        draw_edge(ax, k, color=GREY_EDGE, lw=GREY_LW)
    # Bold edges on top.
    for k in bold_solid:
        draw_edge(ax, k, color=INK, lw=BOLD_LW)
    for k in bold_dashed:
        draw_edge(ax, k, color=INK, lw=BOLD_LW, dashed=True)
    draw_nodes(ax)
    save(fig, name)


def verify_final(edges: list[tuple[int, int]]) -> None:
    G = nx.Graph()
    G.add_nodes_from(range(14))
    G.add_edges_from(edges)
    degs = set(dict(G.degree()).values())
    girth = min((len(c) for c in nx.simple_cycles(G, length_bound=8)), default=None)
    assert degs == {3}, f"final graph not 3-regular: degrees {degs}"
    assert girth == 5, f"final graph girth {girth}, expected 5"
    print(f"VERIFIED final graph: 3-regular, girth {girth}")


def main() -> None:
    # Frame 0: start graph, one short cycle highlighted bold/dark.
    render("refine-0.pdf", START_EDGES, bold_solid=cycle_edges(HIGHLIGHT_CYCLE))

    edges = list(START_EDGES)
    for i, (removed, added) in enumerate(SWAPS, start=1):
        rem = {_ekey(e) for e in removed}
        add = {_ekey(e) for e in added}
        # frame Na: removed edges bold + dashed.
        render(f"refine-{i}a.pdf", edges, bold_dashed=rem)
        # apply swap.
        edges = [_ekey(e) for e in edges if _ekey(e) not in rem]
        edges.extend(sorted(add))
        # frame Nb: newly added edges bold solid.
        render(f"refine-{i}b.pdf", edges, bold_solid=add)

    # Final frame: no highlights.
    render("refine-final.pdf", edges)
    verify_final(edges)


if __name__ == "__main__":
    main()

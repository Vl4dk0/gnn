"""Generate the RL edge-building animation figures for the defense deck.

Animates, step by step, how a reinforcement-learning agent would construct the
(3,4)-cage by adding edges one at a time. The (3,4)-cage is K_{3,3}, bipartite
with parts A = {0, 4, 5} and B = {1, 2, 3}, 3-regular with girth 4.

Output: one composite PDF per step into presentation/figures/rl/, named
rl-step-0.pdf ... rl-step-9.pdf. Each composite is a wide two-panel figure:
  LEFT  = the graph after adding the first k edges (last edge bold/dark),
  RIGHT = an "Akcie" action panel listing candidate add/remove actions, with
          the chosen action highlighted and triangle/degree-blocked actions
          disabled (light grey, strikethrough, with a Slovak reason).

Design: greyscale, vector PDF, transparent background, fonttype 42.
"""

from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.text import Text

# Embed TrueType fonts in the PDF (type 42) so text stays selectable and sharp.
matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --- Palette (shared with the other greyscale graph figures) -----------------
BG = "#ffffff"
NODE_BASE = "#6b6b6b"
EDGE_BASE = "#aaaaaa"
NODE_HI = "#1a1a1a"
EDGE_HI = "#1a1a1a"
LABEL_ON_BASE = "#f0f0f0"

# Action-panel greys.
TEXT_DARK = "#1a1a1a"
TEXT_DISABLED = "#b5b5b5"
BOX_FILL = "#e8e8e8"
BOX_EDGE = "#1a1a1a"

NODE_SIZE = 290
EDGE_WIDTH_BASE = 1.8
EDGE_WIDTH_HI = 3.0

# --- The (3,4)-cage = K_{3,3} -------------------------------------------------
# Bipartite parts: A on the left column (x = -1), B on the right column (x = +1).
PART_A: list[int] = [0, 4, 5]
PART_B: list[int] = [1, 2, 3]

# Two vertical columns, spread in y so nodes do not crowd.
RL_POS: dict[int, tuple[float, float]] = {
    0: (-1.0, 1.0),
    4: (-1.0, 0.0),
    5: (-1.0, -1.0),
    1: (1.0, 1.0),
    2: (1.0, 0.0),
    3: (1.0, -1.0),
}

# Edge-add order (9 steps).
EDGE_ORDER: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (4, 1),
    (4, 2),
    (4, 3),
    (5, 1),
    (5, 2),
    (5, 3),
]


# ---------------------------------------------------------------------------
# Graph panel
# ---------------------------------------------------------------------------


def draw_graph(ax: Axes, step: int) -> None:
    """Draw the (3,4)-cage after adding the first `step` edges.

    The most-recently-added edge is bold/dark; earlier edges are normal grey.
    All six vertices are always drawn (isolated at step 0), labelled inside.
    """
    g: nx.Graph[int] = nx.Graph()
    g.add_nodes_from(PART_A + PART_B)

    added = EDGE_ORDER[:step]
    last_edge = added[-1] if added else None
    base_edges = [e for e in added if e != last_edge]
    g.add_edges_from(added)

    if base_edges:
        _ = nx.draw_networkx_edges(
            g,
            RL_POS,
            ax=ax,
            edgelist=base_edges,
            edge_color=EDGE_BASE,
            width=EDGE_WIDTH_BASE,
        )
    if last_edge is not None:
        _ = nx.draw_networkx_edges(
            g,
            RL_POS,
            ax=ax,
            edgelist=[last_edge],
            edge_color=EDGE_HI,
            width=EDGE_WIDTH_HI,
        )
    _ = nx.draw_networkx_nodes(
        g,
        RL_POS,
        ax=ax,
        nodelist=PART_A + PART_B,
        node_size=NODE_SIZE,
        node_color=NODE_BASE,
    )
    labels = {v: str(v) for v in g.nodes()}
    _ = nx.draw_networkx_labels(
        g,
        RL_POS,
        ax=ax,
        labels=labels,
        font_size=9,
        font_color=LABEL_ON_BASE,
        font_weight="bold",
    )

    _ = ax.set_aspect("equal")
    _ = ax.axis("off")
    _ = ax.set_xlim(-1.6, 1.6)
    _ = ax.set_ylim(-1.5, 1.5)


# ---------------------------------------------------------------------------
# Action panel logic
# ---------------------------------------------------------------------------


def closes_triangle(g: "nx.Graph[int]", u: int, v: int) -> bool:
    """Adding edge (u, v) closes a 3-cycle iff u and v share a neighbour."""
    return bool(set(g.neighbors(u)) & set(g.neighbors(v)))


def candidate_add_lines(
    g: "nx.Graph[int]", chosen: tuple[int, int] | None
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], str]]]:
    """Split potential A-B add actions into (valid, disabled-with-reason).

    Valid   = an A-B edge whose endpoints are below degree 3 and whose addition
              would not close a triangle.
    Disabled = blocked because an endpoint already has degree 3, or because it
               would close a triangle (a cycle shorter than girth 4). Triangle
               candidates include same-part pairs (e.g. two B-vertices sharing a
               common A-neighbour), which are exactly the moves a girth-4 target
               must forbid.
    The chosen edge is excluded from both lists (rendered separately).
    """
    valid: list[tuple[int, int]] = []
    disabled: list[tuple[tuple[int, int], str]] = []

    # A-B candidates: the only edges the bipartite target actually wants.
    for a in PART_A:
        for b in PART_B:
            e = (a, b)
            if g.has_edge(a, b) or e == chosen:
                continue
            if g.degree(a) >= 3 or g.degree(b) >= 3:
                disabled.append((e, "stupeň 3"))
            elif closes_triangle(g, a, b):
                disabled.append((e, "trojuholník"))
            else:
                valid.append(e)

    # Same-part pairs that would close a triangle: never valid for a girth-4
    # target, but worth surfacing as a disabled move so the agent's constraint
    # is visible (e.g. (1,2) once both touch vertex 0).
    for part in (PART_B, PART_A):
        for i, u in enumerate(part):
            for w in part[i + 1 :]:
                e = (u, w)
                if g.has_edge(u, w) or e == chosen:
                    continue
                if closes_triangle(g, u, w):
                    disabled.append((e, "trojuholník"))

    return valid, disabled


def fmt_edge(e: tuple[int, int]) -> str:
    return f"({e[0]},{e[1]})"


def _strike_through(ax: Axes, t: Text) -> None:
    """Draw a horizontal strike line across a rendered text object.

    Measures the text's pixel bounding box via the figure renderer, converts it
    to axes coordinates, and draws a light-grey line through the vertical middle.
    """
    canvas = cast(object, ax.figure.canvas)
    draw = cast(Callable[[], None], getattr(canvas, "draw", lambda: None))
    draw()  # ensure a renderer exists so the text extent is measurable
    get_extent = cast(Callable[[], object], t.get_window_extent)
    bbox_px = get_extent()
    x0_px = cast(float, getattr(bbox_px, "x0"))
    y0_px = cast(float, getattr(bbox_px, "y0"))
    x1_px = cast(float, getattr(bbox_px, "x1"))
    y1_px = cast(float, getattr(bbox_px, "y1"))
    inv = ax.transAxes.inverted()
    transform = cast(
        Callable[[tuple[float, float]], "tuple[float, float]"], inv.transform
    )
    x0, y0 = transform((x0_px, y0_px))
    x1, y1 = transform((x1_px, y1_px))
    ymid = (y0 + y1) / 2.0
    plot = cast(Callable[..., object], ax.plot)
    _ = plot(
        [x0, x1],
        [ymid, ymid],
        transform=ax.transAxes,
        color=TEXT_DISABLED,
        linewidth=1.1,
        solid_capstyle="butt",
    )


def draw_action_panel(ax: Axes, step: int) -> None:
    """Render the 'Akcie' panel for the given step (top-aligned, <= 7 lines).

    Rules:
      - chosen "pridaj (u,v)" highlighted (dark bold, light box, '>' marker),
      - a couple of other valid "pridaj (u,v)" options (dark normal),
      - at least one disabled add option (light grey, strikethrough, reason),
      - one "odober (u,v)" remove option once any edge exists.
    """
    _ = ax.axis("off")
    _ = ax.set_xlim(0.0, 1.0)
    _ = ax.set_ylim(0.0, 1.0)
    text = cast(Callable[..., object], ax.text)

    present = EDGE_ORDER[:step]
    g: nx.Graph[int] = nx.Graph()
    g.add_nodes_from(PART_A + PART_B)
    g.add_edges_from(present)

    chosen = EDGE_ORDER[step - 1] if step >= 1 else None
    valid, disabled = candidate_add_lines(g, chosen)

    # Removable edges: anything already present except the chosen edge.
    removable = [e for e in present if e != chosen]

    # Sort disabled: triangle first, then degree.
    disabled_sorted = sorted(disabled, key=lambda d: 0 if d[1] == "trojuholník" else 1)

    # Build the ordered list of exactly 5 action entries.
    # Each entry is one of:
    #   ("chosen", edge)
    #   ("valid", edge)
    #   ("remove", edge)
    #   ("disabled", edge, reason)
    entries: list[tuple[object, ...]] = []

    if chosen is not None:
        entries.append(("chosen", chosen))

    # Reserve a slot for "odober" whenever earlier edges exist, so it always
    # appears when the graph is non-empty (spec requirement).
    # Remaining budget for valid adds after chosen + odober slot.
    odober_reserved = 1 if removable else 0
    valid_budget = 5 - (1 if chosen is not None else 0) - odober_reserved

    # Fill with valid adds (excluding chosen), up to the budget.
    for e in valid:
        if len(entries) - (1 if chosen is not None else 0) >= valid_budget:
            break
        entries.append(("valid", e))

    # Insert the one remove option now (guaranteed slot).
    if removable:
        entries.append(("remove", removable[-1]))

    # Fill remaining with disabled options.
    for e, reason in disabled_sorted:
        if len(entries) >= 5:
            break
        entries.append(("disabled", e, reason))

    # If still short (e.g. very late steps), add more remove options.
    remove_idx = 1
    while len(entries) < 5:
        if remove_idx < len(removable):
            e = removable[-(remove_idx + 1)]
            entries.append(("remove", e))
            remove_idx += 1
        else:
            # Fallback: repeat first disabled if nothing else available.
            if disabled_sorted:
                e2, r2 = disabled_sorted[0]
                entries.append(("disabled", e2, r2))
            else:
                break

    # Heading.
    x = 0.06
    y = 0.94
    dy = 0.115
    _ = text(
        x,
        y,
        "Akcie",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=TEXT_DARK,
        family="monospace",
    )
    y -= dy * 1.25

    line_fs = 11.5

    # Render exactly the 5 entries.
    for entry in entries[:5]:
        kind = entry[0]
        if kind == "chosen":
            e = cast(tuple[int, int], entry[1])
            label = f"► pridaj {fmt_edge(e)}"
            _ = text(
                x,
                y,
                label,
                ha="left",
                va="top",
                fontsize=line_fs,
                fontweight="bold",
                color=TEXT_DARK,
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=BOX_FILL,
                    edgecolor=BOX_EDGE,
                    linewidth=1.0,
                ),
            )
            y -= dy * 1.15
        elif kind == "valid":
            e = cast(tuple[int, int], entry[1])
            _ = text(
                x,
                y,
                f"  pridaj {fmt_edge(e)}",
                ha="left",
                va="top",
                fontsize=line_fs,
                color=TEXT_DARK,
                family="monospace",
            )
            y -= dy
        elif kind == "remove":
            e = cast(tuple[int, int], entry[1])
            _ = text(
                x,
                y,
                f"  odober {fmt_edge(e)}",
                ha="left",
                va="top",
                fontsize=line_fs,
                color=TEXT_DARK,
                family="monospace",
            )
            y -= dy
        elif kind == "disabled":
            e = cast(tuple[int, int], entry[1])
            reason = cast(str, entry[2])
            line = f"  pridaj {fmt_edge(e)}  ✗ ({reason})"
            t = cast(
                Text,
                text(
                    x,
                    y,
                    line,
                    ha="left",
                    va="top",
                    fontsize=line_fs,
                    color=TEXT_DISABLED,
                    family="monospace",
                ),
            )
            _strike_through(ax, t)
            y -= dy


# ---------------------------------------------------------------------------
# Composite + entry point
# ---------------------------------------------------------------------------


def save(fig: Figure, name: str) -> None:
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


def make_composite(step: int) -> Figure:
    fig_fn = cast(Callable[..., Figure], plt.figure)
    fig: Figure = fig_fn(figsize=(9.0, 5.0))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.05], wspace=0.05, figure=fig)
    ax_graph: Axes = fig.add_subplot(gs[0, 0])
    ax_panel: Axes = fig.add_subplot(gs[0, 1])
    draw_graph(ax_graph, step)
    draw_action_panel(ax_panel, step)
    return fig


def verify_target() -> None:
    """Assert the final graph is 3-regular with girth 4."""
    g: nx.Graph[int] = nx.Graph()
    g.add_nodes_from(PART_A + PART_B)
    g.add_edges_from(EDGE_ORDER)
    degrees = {d for _, d in g.degree()}
    assert degrees == {3}, f"final graph is not 3-regular: degrees {degrees}"
    girth = min(len(c) for c in nx.minimum_cycle_basis(g))
    assert girth == 4, f"final graph girth is {girth}, expected 4"
    print(f"(3,4)-cage verified: 3-regular, girth = {girth}")


def main() -> None:
    plt.rcParams["figure.facecolor"] = BG
    plt.rcParams["axes.facecolor"] = BG

    verify_target()

    for step in range(0, 10):
        fig = make_composite(step)
        save(fig, f"rl/rl-step-{step}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()

"""Generate the quality-vs-breadth tradeoff scatter plot for the defense deck.

X axis = graph size / Moore bound (1 = cage), Y axis = percentage of the 22
targets a method solves. Each method is one point; voltage-rl is highlighted.

Output: presentation/figures/results/fig-tradeoff.pdf
"""

from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --- Data --------------------------------------------------------------------

# (method_name, x = size / Moore, y = percent of 22 targets solved, color, s).
# Percentages are solved_targets / 22 rounded: 19->86, 14->64, 12->55, 11->50,
# 5->23, 3->14.
POINTS: list[tuple[str, float, float, str, int]] = [
    ("voltage-rl", 1.66, 86, "#1f3a5f", 360),
    ("voltage", 1.72, 64, "#4a6572", 220),
    ("A*", 1.10, 55, "#4a6572", 220),
    ("RandomWalk", 1.00, 55, "#4a6572", 220),
    ("Forge", 1.23, 50, "#4a6572", 220),
    ("directRL", 1.11, 23, "#4a6572", 220),
    ("Bruteforce", 1.74, 14, "#4a6572", 220),
]


# --- Main --------------------------------------------------------------------


def main() -> None:
    subplots_fn = cast(Callable[..., tuple[Figure, Axes]], plt.subplots)
    fig, ax = subplots_fn(figsize=(8.5, 6))

    set_facecolor_fig = cast(Callable[..., object], fig.patch.set_facecolor)
    _ = set_facecolor_fig("#ffffff")
    set_facecolor_ax = cast(Callable[..., object], ax.set_facecolor)
    _ = set_facecolor_ax("#ffffff")

    # Shaded "desired corner" rectangle (small size + high reach), behind all.
    add_patch = cast(Callable[..., object], ax.add_patch)
    _ = add_patch(
        Rectangle(
            (0.9, 58),
            0.45,
            42,
            facecolor="#e8edf2",
            alpha=0.6,
            zorder=0,
        )
    )

    text_fn = cast(Callable[..., object], ax.text)

    # Dashed vertical line at x=1.0 (cage boundary).
    axvline_fn = cast(Callable[..., object], ax.axvline)
    _ = axvline_fn(x=1.0, color="#8c8c8c", linestyle="--", linewidth=1.5, zorder=1)

    _ = text_fn(
        1.005,
        2.0,
        "klietka",
        fontsize=11,
        color="#8c8c8c",
        va="bottom",
        ha="left",
        zorder=2,
    )

    # Plot each method point.
    scatter_fn = cast(Callable[..., object], ax.scatter)

    for _name, x, y, color, s in POINTS:
        _ = scatter_fn(x, y, color=color, s=s, zorder=3)

    # Label each point, with offsets for A* and RandomWalk which overlap.
    for name, x, y, _color, _s in POINTS:
        if name == "A*":
            offset_x = 0.02
            offset_y = 2.5
            ha = "left"
            va = "bottom"
        elif name == "RandomWalk":
            offset_x = 0.02
            offset_y = -2.5
            ha = "left"
            va = "top"
        else:
            offset_x = 0.02
            offset_y = 2.2
            ha = "left"
            va = "bottom"

        _ = text_fn(
            x + offset_x,
            y + offset_y,
            name,
            fontsize=12,
            ha=ha,
            va=va,
            zorder=4,
        )

    set_xlabel = cast(Callable[..., object], ax.set_xlabel)
    _ = set_xlabel("veľkosť grafu / Moore's bound", fontsize=14)

    set_ylabel = cast(Callable[..., object], ax.set_ylabel)
    _ = set_ylabel("vyriešené ciele (%)", fontsize=14)

    set_xlim = cast(Callable[..., object], ax.set_xlim)
    _ = set_xlim(0.85, 1.95)

    set_ylim = cast(Callable[..., object], ax.set_ylim)
    _ = set_ylim(0, 100)

    set_yticks = cast(Callable[..., object], ax.set_yticks)
    _ = set_yticks([0, 20, 40, 60, 80, 100])

    tick_params_fn = cast(Callable[..., object], ax.tick_params)
    _ = tick_params_fn(labelsize=14)

    for side in ("top", "right"):
        spine = ax.spines[side]
        set_vis = cast(Callable[..., object], spine.set_visible)
        _ = set_vis(False)

    out = FIGURES / "results" / "fig-tradeoff.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(
        out,
        format="pdf",
        bbox_inches="tight",
        facecolor="#ffffff",
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()

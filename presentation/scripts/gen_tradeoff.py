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
    ("voltage-rl", 1.66, 86, "#1f3a5f", 520),
    ("voltage", 1.72, 64, "#4a6572", 330),
    ("A*", 1.10, 55, "#4a6572", 330),
    ("Forge", 1.23, 50, "#4a6572", 330),
    ("RL", 1.11, 23, "#4a6572", 330),
    ("Bruteforce", 1.74, 14, "#4a6572", 330),
]


# --- Main --------------------------------------------------------------------


def main() -> None:
    subplots_fn = cast(Callable[..., tuple[Figure, Axes]], plt.subplots)
    fig, ax = subplots_fn(figsize=(13.0, 5.0))

    set_facecolor_fig = cast(Callable[..., object], fig.patch.set_facecolor)
    _ = set_facecolor_fig("#ffffff")
    set_facecolor_ax = cast(Callable[..., object], ax.set_facecolor)
    _ = set_facecolor_ax("#ffffff")

    # Shaded "desired corner" (small size + high reach). Hatched + light blue so
    # it survives a washed-out projector. It starts at x = 1.0 because a graph can
    # never be smaller than its Moore bound.
    add_patch = cast(Callable[..., object], ax.add_patch)
    _ = add_patch(
        Rectangle(
            (1.0, 58),
            0.35,
            42,
            facecolor="#d6e8f7",
            edgecolor="#4a78a8",
            hatch="///",
            linewidth=1.0,
            alpha=0.55,
            zorder=0,
        )
    )

    text_fn = cast(Callable[..., object], ax.text)

    # Plot each method point.
    scatter_fn = cast(Callable[..., object], ax.scatter)

    for _name, x, y, color, s in POINTS:
        _ = scatter_fn(x, y, color=color, s=s, zorder=3)

    # Label each point.
    for name, x, y, _color, _s in POINTS:
        _ = text_fn(
            x + 0.018,
            y + 3.0,
            name,
            fontsize=18,
            ha="left",
            va="bottom",
            zorder=4,
        )

    set_xlabel = cast(Callable[..., object], ax.set_xlabel)
    _ = set_xlabel("veľkosť grafu / Moore's bound", fontsize=22)

    set_ylabel = cast(Callable[..., object], ax.set_ylabel)
    _ = set_ylabel("vyriešené ciele (%)", fontsize=22)

    set_xlim = cast(Callable[..., object], ax.set_xlim)
    _ = set_xlim(1.0, 2.0)

    set_ylim = cast(Callable[..., object], ax.set_ylim)
    _ = set_ylim(0, 100)

    set_yticks = cast(Callable[..., object], ax.set_yticks)
    _ = set_yticks([0, 20, 40, 60, 80, 100])

    tick_params_fn = cast(Callable[..., object], ax.tick_params)
    _ = tick_params_fn(labelsize=19)

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

"""Generate the tau-threshold and policy-comparison figure for the defense deck.

Panel A: Conceptual schematic of tau tuning — success-per-candidate vs.
         throughput/coverage as tau varies, with the chosen operating point.
Panel B: Grouped bar chart comparing classical vs. GNN policy success rates
         for Refinement and Excision.

Output: presentation/figures/results/fig-tau-policy.pdf
"""

from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

NAVY = "#1f3a5f"
STEEL = "#4a6572"
BRONZE = "#8c3b3b"
GREY = "#888888"
LINES = "#d0d0d0"


def _panel_a(ax: Axes) -> None:
    """Schematic illustration of tau threshold tuning."""

    set_facecolor = cast(Callable[..., object], ax.set_facecolor)
    _ = set_facecolor("#ffffff")

    x = np.linspace(0, 1, 300)
    y1 = 0.9 * np.exp(-3.0 * x)  # úspešnosť na kandidáta — decreasing
    y2 = 0.85 * (1.0 - np.exp(-4.0 * x))  # priepustnosť / pokrytie — increasing

    plot_fn = cast(Callable[..., list[object]], ax.plot)
    _ = plot_fn(x, y1, color=NAVY, linewidth=2.2, label="úspešnosť na kandidáta")
    _ = plot_fn(x, y2, color=STEEL, linewidth=2.2, label="priepustnosť / pokrytie")

    axvspan_fn = cast(Callable[..., object], ax.axvspan)
    _ = axvspan_fn(0.1, 0.35, alpha=0.12, color=STEEL, zorder=0)

    axvline_fn = cast(Callable[..., object], ax.axvline)
    _ = axvline_fn(0.2, color=BRONZE, linewidth=1.5, linestyle="--", zorder=2)

    text_fn = cast(Callable[..., object], ax.text)
    _ = text_fn(
        0.205,
        0.04,
        "zvolené τ = 0,2",
        fontsize=10,
        color=BRONZE,
        va="bottom",
        ha="left",
    )

    set_xlabel = cast(Callable[..., object], ax.set_xlabel)
    _ = set_xlabel("prah defective fraction τ", fontsize=12)

    set_ylabel = cast(Callable[..., object], ax.set_ylabel)
    _ = set_ylabel("—", fontsize=12)

    set_yticks = cast(Callable[..., object], ax.set_yticks)
    _ = set_yticks([])

    set_xlim = cast(Callable[..., object], ax.set_xlim)
    _ = set_xlim(0.0, 1.0)

    set_ylim = cast(Callable[..., object], ax.set_ylim)
    _ = set_ylim(0.0, 1.0)

    set_title = cast(Callable[..., object], ax.set_title)
    _ = set_title("Ladenie prahu τ", fontsize=13)

    legend_fn = cast(Callable[..., object], ax.legend)
    _ = legend_fn(fontsize=10, loc="center right")

    for side in ("top", "right"):
        spine = ax.spines[side]
        set_vis = cast(Callable[..., object], spine.set_visible)
        _ = set_vis(False)


def _panel_b(ax: Axes) -> None:
    """Grouped bar chart: classical vs. GNN policy for Refinement and Excision."""

    set_facecolor = cast(Callable[..., object], ax.set_facecolor)
    _ = set_facecolor("#ffffff")

    width = 0.35

    # Refinement group (x=0): two bars offset by ±0.2. These are REAL measured
    # success rates. Excision is handled with an honest text note (no invented
    # percentage), since there is no measured excision success number.
    bar_fn = cast(Callable[..., object], ax.bar)
    _ = bar_fn(
        0.0 - 0.2,
        75,
        width=width,
        color=NAVY,
        label="Klasická politika",
        zorder=3,
    )
    _ = bar_fn(
        0.0 + 0.2,
        51,
        width=width,
        color=BRONZE,
        label="GNN politika",
        zorder=3,
    )

    text_fn = cast(Callable[..., object], ax.text)

    # Value labels above Refinement bars
    _ = text_fn(
        0.0 - 0.2,
        77,
        "75%",
        ha="center",
        va="bottom",
        fontsize=10,
        color=NAVY,
    )
    _ = text_fn(
        0.0 + 0.2,
        53,
        "51%",
        ha="center",
        va="bottom",
        fontsize=10,
        color=BRONZE,
    )

    # Honest excision note: learned repair equals classical, no invented number.
    _ = text_fn(
        0.0,
        -14,
        "Excízia: naučené zošitie = klasické (žiadny zisk)",
        ha="center",
        va="top",
        fontsize=10,
        color=GREY,
    )

    set_ylabel = cast(Callable[..., object], ax.set_ylabel)
    _ = set_ylabel("úspešnosť (%)", fontsize=12)

    set_ylim = cast(Callable[..., object], ax.set_ylim)
    _ = set_ylim(0, 100)

    set_yticks = cast(Callable[..., object], ax.set_yticks)
    _ = set_yticks([0, 25, 50, 75, 100])

    set_xticks = cast(Callable[..., object], ax.set_xticks)
    _ = set_xticks([0.0 - 0.2, 0.0 + 0.2])

    set_xticklabels = cast(Callable[..., object], ax.set_xticklabels)
    _ = set_xticklabels(["Klasická", "GNN"], fontsize=11)

    set_title = cast(Callable[..., object], ax.set_title)
    _ = set_title("Naučená politika neprekonáva klasiku", fontsize=13)

    legend_fn = cast(Callable[..., object], ax.legend)
    _ = legend_fn(fontsize=10)

    for side in ("top", "right"):
        spine = ax.spines[side]
        set_vis = cast(Callable[..., object], spine.set_visible)
        _ = set_vis(False)


def main() -> None:
    subplots_fn = cast(Callable[..., tuple[Figure, tuple[Axes, Axes]]], plt.subplots)
    fig, (ax_a, ax_b) = subplots_fn(1, 2, figsize=(9.5, 4.5))

    set_facecolor_fig = cast(Callable[..., object], fig.patch.set_facecolor)
    _ = set_facecolor_fig("#ffffff")

    suptitle_fn = cast(Callable[..., object], fig.suptitle)
    _ = suptitle_fn("Prah τ a prínos učenia", fontsize=15)

    _panel_a(ax_a)
    _panel_b(ax_b)

    tight_layout_fn = cast(Callable[..., None], plt.tight_layout)
    tight_layout_fn()

    out = FIGURES / "results" / "fig-tau-policy.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(out, format="pdf", bbox_inches="tight", facecolor="#ffffff")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

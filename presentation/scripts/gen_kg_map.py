"""Generate the (k,g) parameter-plane scatter plot for the defense deck.

For each (k,g) target, plots a marker at (g, k) coloured by which method
family can reach it within the time budget.

Output: presentation/figures/results/fig-kg-map.pdf
"""

from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --- Data --------------------------------------------------------------------

KLASICKE: list[tuple[int, int]] = [
    (5, 3),
    (6, 3),
    (7, 3),
    (3, 5),
    (5, 4),
    (6, 4),
    (3, 6),
    (7, 4),
    (3, 7),
    (4, 6),
    (3, 8),
    (5, 6),
    (6, 6),
]
VOLTAGE: list[tuple[int, int]] = [(4, 5), (5, 5)]
LEN_VOLTAGE_RL: list[tuple[int, int]] = [(6, 5), (3, 9), (3, 10), (4, 8)]
NIKTO: list[tuple[int, int]] = [(7, 5), (4, 7), (5, 7)]


# --- Main --------------------------------------------------------------------


def main() -> None:
    subplots_fn = cast(Callable[..., tuple[Figure, Axes]], plt.subplots)
    fig, ax = subplots_fn(figsize=(8.5, 6))

    set_facecolor_fig = cast(Callable[..., object], fig.patch.set_facecolor)
    _ = set_facecolor_fig("#ffffff")
    set_facecolor_ax = cast(Callable[..., object], ax.set_facecolor)
    _ = set_facecolor_ax("#ffffff")

    scatter_fn = cast(Callable[..., object], ax.scatter)

    # klasicke
    xs_k = [g for _k, g in KLASICKE]
    ys_k = [k for k, _g in KLASICKE]
    _ = scatter_fn(
        xs_k,
        ys_k,
        marker="o",
        color="#1f3a5f",
        s=180,
        zorder=3,
        label="klasické",
    )

    # voltage
    xs_v = [g for _k, g in VOLTAGE]
    ys_v = [k for k, _g in VOLTAGE]
    _ = scatter_fn(
        xs_v,
        ys_v,
        marker="s",
        color="#4a6572",
        s=180,
        zorder=3,
        label="voltage",
    )

    # len_voltage_rl
    xs_r = [g for _k, g in LEN_VOLTAGE_RL]
    ys_r = [k for k, _g in LEN_VOLTAGE_RL]
    _ = scatter_fn(
        xs_r,
        ys_r,
        marker="*",
        color="#8c3b3b",
        s=420,
        zorder=3,
        label="len voltage-rl",
    )

    # nikto
    xs_n = [g for _k, g in NIKTO]
    ys_n = [k for k, _g in NIKTO]
    _ = scatter_fn(
        xs_n,
        ys_n,
        marker="x",
        color="#b5b5b5",
        s=120,
        zorder=3,
        label="nikto do 60 s",
    )

    set_xlabel = cast(Callable[..., object], ax.set_xlabel)
    _ = set_xlabel("obvod g", fontsize=14)

    set_ylabel = cast(Callable[..., object], ax.set_ylabel)
    _ = set_ylabel("stupeň k", fontsize=14)

    set_xticks = cast(Callable[..., object], ax.set_xticks)
    _ = set_xticks(list(range(3, 11)))

    set_yticks = cast(Callable[..., object], ax.set_yticks)
    _ = set_yticks(list(range(3, 8)))

    grid_fn = cast(Callable[..., object], ax.grid)
    _ = grid_fn(True, linestyle="--", alpha=0.3)

    tick_params_fn = cast(Callable[..., object], ax.tick_params)
    _ = tick_params_fn(labelsize=14)

    legend_fn = cast(Callable[..., object], ax.legend)
    _ = legend_fn(fontsize=14)

    for side in ("top", "right"):
        spine = ax.spines[side]
        set_vis = cast(Callable[..., object], spine.set_visible)
        _ = set_vis(False)

    out = FIGURES / "results" / "fig-kg-map.pdf"
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

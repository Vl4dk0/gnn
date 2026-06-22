"""Generate the frontier heatmap showing which method finds the smallest graph per (k,g).

Rows = degree k (3..7), columns = girth g (3..10).
Each cell is coloured by the group that achieved the smallest successful
Moore ratio: grey = classical (A*/randomwalk/bruteforce), navy = forge/voltage,
white/hatched = nobody succeeded.

Output: presentation/figures/results/fig-frontier.pdf
"""

import csv
from pathlib import Path
from typing import Callable, cast

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.rcParams["pdf.fonttype"] = 42

FIGURES = Path(__file__).resolve().parents[1] / "figures"
DATA = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "runs"
    / "2026-05-31_23-37-36"
    / "cage.csv"
)

# Moore bounds per (k, g)
MOORE: dict[tuple[int, int], int] = {
    (3, 5): 10,
    (3, 6): 14,
    (3, 7): 22,
    (3, 8): 30,
    (3, 9): 46,
    (3, 10): 62,
    (4, 5): 17,
    (4, 6): 26,
    (4, 7): 53,
    (4, 8): 80,
    (5, 3): 6,
    (5, 4): 10,
    (5, 5): 26,
    (5, 6): 42,
    (5, 7): 106,
    (6, 3): 7,
    (6, 4): 12,
    (6, 5): 37,
    (6, 6): 62,
    (7, 3): 8,
    (7, 4): 14,
    (7, 5): 50,
}

CLASSICAL = {"astar", "bruteforce", "randomwalk"}

COLOR_CLASSICAL = "#b8b8b8"
COLOR_FORGE = "#1f3a5f"
COLOR_NONE = "#ffffff"

K_VALUES = [3, 4, 5, 6, 7]
G_VALUES = [3, 4, 5, 6, 7, 8, 9, 10]


def is_forge_voltage(method: str) -> bool:
    return "forge" in method or "voltage" in method


# Row type from CSV (after parsing)
type CsvRow = dict[str, str]


def load_csv(path: Path) -> list[CsvRow]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def classify_cell(rows: list[CsvRow], k: int, g: int) -> tuple[str, float | None]:
    """Return (category, best_ratio) for a (k,g) cell.

    category is one of: 'classical', 'forge', 'none'.
    best_ratio is the Moore ratio of the winning group, or None if nobody won.
    """
    moore = MOORE.get((k, g))
    if moore is None:
        return ("none", None)

    # Group successful rows by approach
    best_per_method: dict[str, float] = {}
    for row in rows:
        if int(row["target_k"]) != k or int(row["target_g"]) != g:
            continue
        if row["success"].strip().lower() != "true":
            continue
        n_str = row.get("n_nodes", "").strip()
        if not n_str:
            continue
        try:
            n = float(n_str)
        except ValueError:
            continue
        method = row["approach"].strip()
        if method not in best_per_method or n < best_per_method[method]:
            best_per_method[method] = n

    classical_best: float | None = None
    forge_best: float | None = None

    for method, min_n in best_per_method.items():
        # Skip directRL/rl for leader purposes
        if method in ("rl", "directrl", "direct_rl"):
            continue
        ratio = min_n / moore
        if method in CLASSICAL:
            if classical_best is None or ratio < classical_best:
                classical_best = ratio
        elif is_forge_voltage(method):
            if forge_best is None or ratio < forge_best:
                forge_best = ratio

    if classical_best is None and forge_best is None:
        return ("none", None)

    # Ties: classical wins
    if classical_best is not None and forge_best is not None:
        if classical_best <= forge_best:
            return ("classical", classical_best)
        else:
            return ("forge", forge_best)
    elif classical_best is not None:
        return ("classical", classical_best)
    else:
        return ("forge", forge_best)


def main() -> None:
    rows = load_csv(DATA)

    approaches = sorted({r["approach"].strip() for r in rows})
    print("Approaches in CSV:", approaches)

    subplots_fn = cast(Callable[..., tuple[Figure, Axes]], plt.subplots)
    fig, ax = subplots_fn(figsize=(8, 5))

    set_facecolor_fig = cast(Callable[..., object], fig.patch.set_facecolor)
    _ = set_facecolor_fig("#ffffff")
    set_facecolor_ax = cast(Callable[..., object], ax.set_facecolor)
    _ = set_facecolor_ax("#ffffff")

    n_rows = len(K_VALUES)
    n_cols = len(G_VALUES)

    add_patch = cast(Callable[..., object], ax.add_patch)
    text_fn = cast(Callable[..., object], ax.text)

    for ri, k in enumerate(K_VALUES):
        for ci, g in enumerate(G_VALUES):
            row_y = n_rows - 1 - ri

            # Cells not in the target set: draw light grey N/A block
            if (k, g) not in MOORE:
                rect = mpatches.FancyBboxPatch(
                    (ci, row_y),
                    1,
                    1,
                    boxstyle="square,pad=0",
                    facecolor="#e8e8e8",
                    edgecolor="#cccccc",
                    linewidth=0.5,
                )
                _ = add_patch(rect)
                continue

            category, best_ratio = classify_cell(rows, k, g)

            if category == "none":
                rect = mpatches.FancyBboxPatch(
                    (ci, row_y),
                    1,
                    1,
                    boxstyle="square,pad=0",
                    facecolor=COLOR_NONE,
                    edgecolor="#aaaaaa",
                    linewidth=1.0,
                )
                _ = add_patch(rect)
                # Hatch pattern drawn on top
                hatch_rect = mpatches.FancyBboxPatch(
                    (ci, row_y),
                    1,
                    1,
                    boxstyle="square,pad=0",
                    facecolor="none",
                    edgecolor="#cccccc",
                    linewidth=0.5,
                    hatch="////",
                )
                _ = add_patch(hatch_rect)
            else:
                color = COLOR_CLASSICAL if category == "classical" else COLOR_FORGE
                rect = mpatches.FancyBboxPatch(
                    (ci, row_y),
                    1,
                    1,
                    boxstyle="square,pad=0",
                    facecolor=color,
                    edgecolor="#888888",
                    linewidth=0.5,
                )
                _ = add_patch(rect)

                if best_ratio is not None:
                    ratio_str = f"{best_ratio:.2f}"
                    text_color = "#ffffff" if category == "forge" else "#1a1a1a"
                    _ = text_fn(
                        ci + 0.5,
                        row_y + 0.5,
                        ratio_str,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=text_color,
                        fontweight="bold",
                    )

            debug_ratio = f"{best_ratio:.3f}" if best_ratio is not None else "—"
            print(f"  ({k},{g}): {category:10s}  ratio={debug_ratio}")

    # Axes cosmetics
    set_xlim = cast(Callable[..., object], ax.set_xlim)
    _ = set_xlim(0, n_cols)

    set_ylim = cast(Callable[..., object], ax.set_ylim)
    _ = set_ylim(0, n_rows)

    set_xticks = cast(Callable[..., object], ax.set_xticks)
    _ = set_xticks([i + 0.5 for i in range(n_cols)])

    set_xticklabels = cast(Callable[..., object], ax.set_xticklabels)
    _ = set_xticklabels([str(gv) for gv in G_VALUES], fontsize=12)

    set_yticks = cast(Callable[..., object], ax.set_yticks)
    _ = set_yticks([i + 0.5 for i in range(n_rows)])

    set_yticklabels = cast(Callable[..., object], ax.set_yticklabels)
    _ = set_yticklabels([str(kv) for kv in reversed(K_VALUES)], fontsize=12)

    set_xlabel = cast(Callable[..., object], ax.set_xlabel)
    _ = set_xlabel("obvod  g", fontsize=13)

    set_ylabel = cast(Callable[..., object], ax.set_ylabel)
    _ = set_ylabel("stupeň  k", fontsize=13)

    set_title = cast(Callable[..., object], ax.set_title)
    _ = set_title("Kto nájde najmenší graf?", fontsize=14, pad=10)

    tick_params_fn = cast(Callable[..., object], ax.tick_params)
    _ = tick_params_fn(length=0)

    for side in ("top", "right", "bottom", "left"):
        spine = ax.spines[side]
        set_vis = cast(Callable[..., object], spine.set_visible)
        _ = set_vis(False)

    # Legend
    legend_handles = [
        mpatches.Patch(
            facecolor=COLOR_CLASSICAL,
            edgecolor="#888888",
            label="Klasick\xe9 (A*) — najmenš\xed",
        ),
        mpatches.Patch(
            facecolor=COLOR_FORGE,
            edgecolor="#888888",
            label="Forge — najmenš\xed",
        ),
        mpatches.Patch(
            facecolor=COLOR_NONE,
            edgecolor="#aaaaaa",
            hatch="////",
            label="nikto",
        ),
    ]
    legend_fn = cast(Callable[..., object], ax.legend)
    _ = legend_fn(
        handles=legend_handles,
        loc="lower right",
        fontsize=10,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    out = FIGURES / "results" / "fig-frontier.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    savefig = cast(Callable[..., None], fig.savefig)
    savefig(
        out,
        format="pdf",
        bbox_inches="tight",
        facecolor="#ffffff",
    )
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

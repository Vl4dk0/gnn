import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

matplotlib.use("Agg")

rng = np.random.default_rng(42)

# Active vertices: indices 0-4, fixed positions
active_pos = np.array(
    [
        [2.0, 5.0],  # 0
        [3.0, 3.5],  # 1  -> labeled x
        [4.5, 4.5],  # 2
        [5.5, 3.0],  # 3  -> labeled y
        [4.0, 2.0],  # 4
    ]
)

# Available vertices: indices 5-15, scattered
n_avail = 11
avail_pos = rng.uniform([0.5, 0.5], [9.5, 6.5], size=(n_avail, 2))

all_pos = np.vstack([active_pos, avail_pos])
n_total = len(all_pos)

# Edges between active vertices (partial path)
active_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_aspect("equal")
ax.axis("off")

# Draw edges between active vertices
for u, v in active_edges:
    ax.plot(
        [all_pos[u, 0], all_pos[v, 0]],
        [all_pos[u, 1], all_pos[v, 1]],
        color="#888888",
        linewidth=1.5,
        zorder=1,
    )

# Dashed candidate edge: vertex 1 (x) to vertex 3 (y)
ax.plot(
    [all_pos[1, 0], all_pos[3, 0]],
    [all_pos[1, 1], all_pos[3, 1]],
    color="#555555",
    linewidth=1.5,
    linestyle="--",
    zorder=2,
)

# Draw available vertices (indices 5-15)
for i in range(5, n_total):
    circle = plt.Circle(
        all_pos[i],
        0.12,
        color="#cccccc",
        ec="#aaaaaa",
        linewidth=0.8,
        zorder=3,
    )
    ax.add_patch(circle)

# Draw active vertices (indices 0-4)
for i in range(5):
    circle = plt.Circle(
        all_pos[i],
        0.15,
        color="#555555",
        ec="#333333",
        linewidth=1.0,
        zorder=4,
    )
    ax.add_patch(circle)

# Labels: x=vertex1, y=vertex3, z=vertex8 (available, index 8 in all_pos = avail index 3)
labels = {1: "x", 3: "y", 8: "z"}
for idx, label in labels.items():
    ax.text(
        all_pos[idx, 0] + 0.22,
        all_pos[idx, 1] + 0.22,
        label,
        fontsize=10,
        color="#444444",
        ha="left",
        va="bottom",
        zorder=5,
    )

ax.set_xlim(-0.3, 10.3)
ax.set_ylim(-0.3, 7.3)

fig.patch.set_alpha(0)
ax.set_facecolor("none")

import os

os.makedirs("/Users/vladko/School/GNN/presentation/figures/rl", exist_ok=True)

fig.savefig(
    "/Users/vladko/School/GNN/presentation/figures/rl/rl-pool.pdf",
    bbox_inches="tight",
    transparent=True,
)
fig.savefig(
    "/Users/vladko/School/GNN/presentation/figures/rl/rl-pool.png",
    bbox_inches="tight",
    transparent=True,
    dpi=150,
)
print("Saved rl-pool.pdf and rl-pool.png")

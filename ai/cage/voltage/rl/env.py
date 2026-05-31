"""RL environment for voltage assignment on base graphs.

The agent assigns voltage labels (group elements) to edges of a small base
graph, one edge at a time.  After all edges are labeled, the voltage graph
lift is evaluated for girth.

Key advantage over the edge-by-edge CageConstructionEnv:
  - Every produced graph is **automatically k-regular** by construction.
  - Action space is |group| (typically 20-100) instead of O(n^2).
  - Episodes are only num_edges steps long (5-15) instead of 500-1000.
  - The agent focuses 100% on maximising girth.
"""

from __future__ import annotations

import random

import torch
from torch_geometric.data import Data  # pyright: ignore[reportMissingTypeStubs]

from ai.cage.voltage.base_graphs import (
    BaseGraph,
    dumbbell,
    prism_base,
    cubic_multigraph_4nodes,
)
from ai.cage.voltage.groups import (
    FiniteGroup,
    cyclic_group,
    dihedral_group,
    direct_product,
    semidirect_product_cyclic,
)
from ai.cage.voltage.lift import build_lift, verify_lift
from ai.utils.device import get_preferred_device
from backend.utils.graph_utils import moore_bound


# (k, g, base_graph_factory, group_factory) configurations
# Each gives a k-regular lift; the agent picks voltages.
TARGET_CONFIGS: list[tuple[int, int]] = [
    (3, 5),  # mb=10
    (3, 6),  # mb=14
    (3, 7),  # mb=22
    (3, 8),  # mb=30
    (3, 9),  # mb=46
    (3, 10),  # mb=70
    (3, 11),  # mb=112
    (3, 12),  # mb=126
]


class VoltageAssignmentEnv:
    """RL environment where the agent assigns voltages to base-graph edges.

    State : base graph with partially assigned voltages (PyG Data).
    Action: integer in [0, group_order) — the group element for the next edge.
    Reward: potential-based shaping after each step + terminal girth bonus.

    The lift is always k-regular, so no regularity reward is needed.
    """

    SUCCESS_REWARD: float = 20.0
    GOOD_GIRTH_REWARD: float = 5.0  # girth >= g_target but not beating record
    STEP_REWARD: float = 0.0  # neutral per-step reward

    k: int
    g_target: int
    base: BaseGraph
    group: FiniteGroup
    voltages: list[int]  # assigned so far
    current_edge_idx: int  # which edge to assign next
    num_edges: int
    device: torch.device

    # Progressive curriculum
    configs: list[tuple[int, int]]
    unlocked: int

    def __init__(
        self,
        k: int = 3,
        g_target: int = 5,
        randomize: bool = True,
    ):
        self.default_k: int = k
        self.default_g: int = g_target
        self.k = k
        self.g_target = g_target
        self.randomize: bool = randomize

        self.configs = TARGET_CONFIGS
        self.unlocked = 1

        self.base = dumbbell(3)
        self.group = cyclic_group(5)
        self.voltages: list[int] = []
        self.current_edge_idx = 0
        self.num_edges = 0

        self.device = torch.device(get_preferred_device())

        # Tracking
        self.episode_count: int = 0
        self.success_history: dict[tuple[int, int], list[bool]] = {}

    def get_action_dim(self) -> int:
        """Number of possible actions = group order."""
        return self.group.order

    def get_input_dim(self) -> int:
        """Node feature dimension."""
        # [degree/max_k, node_idx/num_nodes, progress, norm_k, norm_g, group_order_norm]
        return 6

    def _pick_base_and_group(self) -> tuple[BaseGraph, FiniteGroup]:
        """Select a base graph and group for the current (k, g) target.

        Uses configurations known to be achievable from our search experiments:
        - dumbbell(3): works well for girth 5-6 with larger cyclic groups
        - cubic_4nodes: works for girth 7-10
        - prism: works for girth 8-12

        Groups are sampled from a diverse pool: cyclic (Z_n), dihedral (D_m),
        direct products (Z_a x Z_b), and semidirect products (Z_n ⋊ Z_m).
        Cyclic-only was the likely blocker for (8,5)/(9,5)/(10,5)/(11,6):
        their known records require non-cyclic groups.
        """
        mb = moore_bound(self.k, self.g_target)

        if self.k == 3:
            if self.g_target <= 6:
                bases = [dumbbell(3)]
                min_order = max(7, mb // 2)
                max_order = max(min_order + 5, (2 * mb) // 2)
            elif self.g_target <= 10:
                bases = [cubic_multigraph_4nodes(), prism_base()]
                n_base = random.choice([4, 6])
                min_order = max(5, mb // n_base)
                max_order = max(min_order + 5, (2 * mb) // n_base)
            else:
                bases = [prism_base()]
                min_order = max(10, mb // 6)
                max_order = max(min_order + 5, (2 * mb) // 6)
        else:
            bases = [dumbbell(self.k)]
            n_base = 2
            min_order = max(5, mb // n_base)
            max_order = max(min_order + 5, (2 * mb) // n_base)

        base = random.choice(bases)
        g_order = random.randint(min_order, max_order)

        # Build a pool of groups at this order and sample one uniformly.
        # Cyclic is always available; dihedral, direct-product, and semidirect
        # variants are added when a valid construction exists at this order.
        pool: list[FiniteGroup] = [cyclic_group(g_order)]

        # Dihedral D_m has order 2m.
        if g_order >= 4 and g_order % 2 == 0:
            pool.append(dihedral_group(g_order // 2))

        # Direct product Z_a x Z_b where a * b == g_order, a < b, both >= 2.
        for a in range(2, g_order):
            if g_order % a == 0:
                b = g_order // a
                if b > a:
                    pool.append(direct_product(cyclic_group(a), cyclic_group(b)))

        # Semidirect product Z_n ⋊_phi Z_m with n * m == g_order, phi != 1,
        # phi^m ≡ 1 (mod n). One phi per (n, m) suffices.
        for n in range(3, g_order):
            if g_order % n == 0:
                m = g_order // n
                if m >= 2:
                    for phi in range(2, n):
                        if pow(phi, m, n) == 1:
                            pool.append(semidirect_product_cyclic(n, m, phi))
                            break

        group = random.choice(pool)
        return base, group

    def reset(self) -> Data:
        """Reset: pick a new (k, g) target, base graph, and group."""
        if self.randomize:
            pool = self.configs[: self.unlocked]
            # Uniform weighting across unlocked stages
            chosen = random.choice(pool)
            self.k, self.g_target = chosen
        else:
            self.k = self.default_k
            self.g_target = self.default_g

        self.base, self.group = self._pick_base_and_group()
        self.num_edges = self.base.num_undirected_edges()
        self.voltages = []
        self.current_edge_idx = 0

        return self._get_obs()

    def _get_obs(self) -> Data:
        """Build observation: base graph with partial voltage assignment."""
        n = self.base.num_nodes
        input_dim = self.get_input_dim()
        x = torch.zeros((n, input_dim), dtype=torch.float, device=self.device)

        max_k = max(self.k, 4)
        progress = self.current_edge_idx / max(self.num_edges, 1)

        for v in range(n):
            deg = self.base.degree(v)
            x[v, 0] = deg / max_k
            x[v, 1] = v / max(n - 1, 1)
            x[v, 2] = progress
            x[v, 3] = self.k / 7.0
            x[v, 4] = self.g_target / 14.0
            x[v, 5] = self.group.order / 80.0

        # Edge index from base graph arcs
        src_list: list[int] = []
        dst_list: list[int] = []
        edge_attr_list: list[list[float]] = []

        for arc in self.base.arcs:
            src_list.append(arc.src)
            dst_list.append(arc.dst)

            # Edge features: [voltage / group_order, is_assigned]
            fwd_id = arc.arc_id
            # Find if this arc's undirected edge is assigned
            assigned = False
            voltage_norm = 0.0
            for edge_pos, und_fwd_id in enumerate(self.base.undirected_edge_ids):
                if (
                    und_fwd_id == fwd_id
                    or self.base.arcs[und_fwd_id].reverse_id == fwd_id
                ):
                    if edge_pos < len(self.voltages):
                        assigned = True
                        voltage_norm = self.voltages[edge_pos] / max(
                            self.group.order - 1, 1
                        )
                    break

            edge_attr_list.append([voltage_norm, 1.0 if assigned else 0.0])

        if src_list:
            edge_index = torch.tensor(
                [src_list, dst_list], dtype=torch.long, device=self.device
            )
            edge_attr = torch.tensor(
                edge_attr_list, dtype=torch.float, device=self.device
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0, 2), dtype=torch.float, device=self.device)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)

    def step(
        self, action: int
    ) -> tuple[Data, float, bool, dict[str, int | float | bool | str]]:
        """Assign voltage `action` to the next unassigned edge."""
        # Clamp action to valid range
        action = action % self.group.order

        self.voltages.append(action)
        self.current_edge_idx += 1

        done = self.current_edge_idx >= self.num_edges
        reward = self.STEP_REWARD

        info: dict[str, int | float | bool | str] = {
            "k": self.k,
            "g": self.g_target,
            "edge_idx": self.current_edge_idx - 1,
            "voltage": action,
            "group_order": self.group.order,
            "base_nodes": self.base.num_nodes,
            "num_edges": self.num_edges,
            "success": False,
        }

        if done:
            # Build and verify the actual lift
            lifted = build_lift(self.base, self.group, self.voltages)
            props = verify_lift(lifted, self.k, self.g_target)
            girth_val = props["girth"]
            girth_int = int(girth_val) if isinstance(girth_val, int) else 0
            lift_order = self.base.num_nodes * self.group.order

            info["girth"] = girth_int
            info["lift_order"] = lift_order
            info["done_reason"] = "complete"
            info["is_connected"] = bool(props["is_connected"])
            info["is_k_regular"] = bool(props["is_k_regular"])

            # Reward only for valid lifts (connected + k-regular)
            if props["is_valid_kg"]:
                reward = float(girth_int) + self.SUCCESS_REWARD
                info["success"] = True
            elif props["is_k_regular"] and props["is_connected"]:
                # Regular and connected but girth too low — reward girth
                reward = float(girth_int)
            else:
                # Degenerate lift (disconnected or not regular) — penalise
                reward = -1.0

            self.episode_count += 1
        else:
            # No intermediate reward — episodes are short (3-9 steps),
            # PPO with GAE handles credit assignment fine.
            info["done_reason"] = "step"

        obs = self._get_obs()
        return obs, reward, done, info

    def report_result(self, success: bool) -> bool:
        """Record episode outcome; return True if a new stage was unlocked."""
        pair = (self.k, self.g_target)
        if pair not in self.success_history:
            self.success_history[pair] = []
        self.success_history[pair].append(success)

        # Keep only last 20
        if len(self.success_history[pair]) > 20:
            self.success_history[pair] = self.success_history[pair][-20:]

        if self.unlocked < len(self.configs):
            newest = self.configs[self.unlocked - 1]
            history = self.success_history.get(newest, [])
            if len(history) >= 8 and sum(history[-8:]) / 8 >= 0.5:
                self.unlocked += 1
                return True
        return False

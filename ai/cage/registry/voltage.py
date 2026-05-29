"""Voltage graph lift search for cage generation.

Constructs (k,g)-graphs by searching for voltage assignments on small base
graphs.  The lift is automatically k-regular by construction, so the search
focuses entirely on maximising girth.

Uses tabu search with random restarts over base graph + group combinations.
"""

import random
import time

import networkx as nx

from ai.cage.voltage.base_graphs import (
    BaseGraph,
    dumbbell,
    cubic_multigraph_4nodes,
    prism_base,
)
from ai.cage.voltage.cycle_analysis import (
    compute_lift_girth,
    count_short_identity_walks,
)
from ai.cage.voltage.groups import FiniteGroup, cyclic_group
from ai.cage.voltage.lift import build_lift, verify_lift
from ai.cage.voltage.supervised.model import GirthPredictor, load_girth_predictor
from ai.cage.voltage.supervised.search import beam_search
from backend.utils.graph_utils import is_k_regular, moore_bound

BEAM_PROBE_INTERVAL = 50


class VoltageSearchGenerator:
    """Cage generator using voltage graph lift search.

    Each step() call tries one voltage assignment (either a tabu move or a
    random restart).  When a valid (k,g)-graph is found, it becomes the
    current graph and is_complete is set.
    """

    k: int
    g: int
    graph: nx.Graph[int]
    step_count: int
    is_complete: bool
    success: bool
    start_time: float

    _base: BaseGraph
    _group: FiniteGroup
    _voltages: list[int]
    _best_girth: int
    _tabu: dict[tuple[int, int], int]
    _free_indices: list[int]
    _configs: list[tuple[BaseGraph, FiniteGroup, str]]
    _config_idx: int
    _restarts: int
    _model: GirthPredictor | None
    _model_id: str | None
    _feat_cycle_lengths: list[int] | None
    _feat_rwpe_dim: int

    def __init__(self, k: int, g: int, model_id: str | None = None) -> None:
        self.k = k
        self.g = g
        self.step_count = 0
        self.is_complete = False
        self.success = False
        self.start_time = 0.0
        self._best_girth = 0
        self._tabu = {}
        self._restarts = 0
        self._model_id = model_id
        self._feat_cycle_lengths = None
        self._feat_rwpe_dim = 0
        if model_id is not None:
            model, feat_cl, feat_rwpe = load_girth_predictor(model_id)
            self._model = model
            self._feat_cycle_lengths = feat_cl
            self._feat_rwpe_dim = feat_rwpe
        else:
            self._model = None

        # Build candidate (base_graph, group) configs
        self._configs = _build_configs(k, g)
        self._config_idx = 0

        # Start with first config
        base, group, _name = self._configs[0]
        self._base = base
        self._group = group
        self._free_indices = base.free_edge_indices()

        # Initialize graph with Moore bound nodes (placeholder until we find something)
        mb = moore_bound(k, g)
        self.graph = nx.Graph()
        for i in range(mb):
            self.graph.add_node(i)

        # Random initial voltage assignment (zeros on tree, random on free)
        m = base.num_undirected_edges()
        self._voltages = [0] * m
        for idx in self._free_indices:
            self._voltages[idx] = random.randint(0, self._group.order - 1)

    def elapsed_time(self) -> float:
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    def is_regular(self) -> bool:
        return is_k_regular(self.graph, self.k)

    def step(self) -> None:
        """One search step: try a tabu move or random restart."""
        if self.start_time == 0:
            self.start_time = time.time()
        self.step_count += 1

        if self.is_complete:
            return

        # Current cost: number of short identity-voltage walks
        current_cost = count_short_identity_walks(
            self._base, self._group, self._voltages, self.g
        )

        if current_cost == 0:
            # Might be a solution — verify
            girth = compute_lift_girth(
                self._base, self._group, self._voltages, max_girth=2 * self.g
            )
            if isinstance(girth, int) and girth >= self.g:
                lifted = build_lift(self._base, self._group, self._voltages)
                props = verify_lift(lifted, self.k, self.g)
                if props["is_valid_kg"]:
                    self.graph = lifted
                    self.is_complete = True
                    self.success = True
                    self._best_girth = girth
                    return

            # Degenerate — random restart
            self._random_restart()
            self.graph = build_lift(self._base, self._group, self._voltages)
            return

        # Tabu move: try changing one free-edge voltage
        best_move_cost = current_cost + 1
        best_edge = -1
        best_val = -1

        for edge_idx in self._free_indices:
            old_val = self._voltages[edge_idx]
            for new_val in range(self._group.order):
                if new_val == old_val:
                    continue
                is_tabu = self._tabu.get((edge_idx, old_val), -1) > self.step_count

                self._voltages[edge_idx] = new_val
                new_cost = count_short_identity_walks(
                    self._base, self._group, self._voltages, self.g
                )
                self._voltages[edge_idx] = old_val

                if new_cost < best_move_cost and (not is_tabu or new_cost == 0):
                    best_move_cost = new_cost
                    best_edge = edge_idx
                    best_val = new_val

        if best_edge >= 0 and best_move_cost <= current_cost:
            old_val = self._voltages[best_edge]
            self._voltages[best_edge] = best_val
            self._tabu[(best_edge, old_val)] = self.step_count + 10
        else:
            # Stuck — random restart with next config
            self._random_restart()

        # ML-guided beam-search probe: when a girth predictor is loaded,
        # periodically run a full beam search on the current (base, group)
        # config. Cheap one-shot — beam_search is sequential and bounded.
        if self._model is not None and self.step_count % BEAM_PROBE_INTERVAL == 0:
            volts_b, girth_b = beam_search(
                self._base,
                self._group,
                self.k,
                self.g,
                model=self._model,
                beam_width=20,
                verbose=False,
                feat_cycle_lengths=self._feat_cycle_lengths,
                feat_rwpe_dim=self._feat_rwpe_dim,
            )
            if isinstance(girth_b, int) and girth_b >= self.g and volts_b is not None:
                lifted = build_lift(self._base, self._group, volts_b)
                props = verify_lift(lifted, self.k, self.g)
                if props["is_valid_kg"]:
                    self._voltages = volts_b
                    self.graph = lifted
                    self.is_complete = True
                    self.success = True
                    self._best_girth = girth_b
                    return

        # Check intermediate girth periodically
        if self.step_count % 20 == 0:
            girth = compute_lift_girth(
                self._base, self._group, self._voltages, max_girth=2 * self.g
            )
            if isinstance(girth, int) and girth > self._best_girth:
                self._best_girth = girth
                lifted = build_lift(self._base, self._group, self._voltages)
                props = verify_lift(lifted, self.k, self.g)
                if props["is_valid_kg"]:
                    self.graph = lifted
                    self.is_complete = True
                    self.success = True
                elif props["is_k_regular"] and props["is_connected"]:
                    # Valid graph, just not high enough girth yet — show it
                    self.graph = lifted

        if not self.is_complete:
            self.graph = build_lift(self._base, self._group, self._voltages)

    def _random_restart(self) -> None:
        """Restart with a new random voltage assignment, possibly new config."""
        self._restarts += 1
        self._tabu.clear()

        # Cycle through configs every few restarts
        if self._restarts % 3 == 0 and len(self._configs) > 1:
            self._config_idx = (self._config_idx + 1) % len(self._configs)
            base, group, _name = self._configs[self._config_idx]
            self._base = base
            self._group = group
            self._free_indices = base.free_edge_indices()

        m = self._base.num_undirected_edges()
        self._voltages = [0] * m
        for idx in self._free_indices:
            self._voltages[idx] = random.randint(0, self._group.order - 1)


def _build_configs(k: int, g: int) -> list[tuple[BaseGraph, FiniteGroup, str]]:
    """Build candidate (base_graph, group) configurations for target (k, g)."""
    mb = moore_bound(k, g)
    configs: list[tuple[BaseGraph, FiniteGroup, str]] = []

    if k == 3:
        bases: list[tuple[BaseGraph, str]] = [
            (dumbbell(3), "dumbbell"),
        ]
        if g >= 7:
            bases.append((cubic_multigraph_4nodes(), "cubic4"))
        if g >= 8:
            bases.append((prism_base(), "prism"))

        for base, bname in bases:
            n_base = base.num_nodes
            min_order = max(5, mb // n_base)
            max_order = min(80, max(min_order + 10, (3 * mb) // n_base))
            # Pick a few group orders spread across the range
            for n in range(
                min_order, max_order + 1, max(1, (max_order - min_order) // 5)
            ):
                group = cyclic_group(n)
                configs.append((base, group, f"{bname}+Z_{n}"))
    else:
        base = dumbbell(k)
        min_order = max(5, mb // 2)
        max_order = min(80, max(min_order + 10, (3 * mb) // 2))
        for n in range(min_order, max_order + 1, max(1, (max_order - min_order) // 5)):
            configs.append((base, cyclic_group(n), f"dumbbell+Z_{n}"))

    if not configs:
        configs.append((dumbbell(k), cyclic_group(max(5, mb // 2)), "fallback"))

    return configs

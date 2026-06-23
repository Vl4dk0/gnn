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
from ai.cage.voltage.groups import (
    FiniteGroup,
    cyclic_group,
    dihedral_group,
    direct_product,
    semidirect_product_cyclic,
)
from ai.cage.voltage.lift import build_lift, verify_lift
from ai.cage.voltage.supervised.model import GirthPredictor, load_girth_predictor
from ai.cage.voltage.supervised.search import beam_search
from backend.utils.graph_utils import is_k_regular, moore_bound

# Default beam-probe interval when a girth predictor is loaded.
# Lowered from 50 to 5 so the probe fires on short runs (many targets solve
# in <50 steps, so the old value meant it never fired).  beam_search is a
# bounded one-shot, but it IS proportional to beam_width * group.order * m,
# so very small values add per-step overhead on large groups.  5 is a
# reasonable default; expose as a constructor parameter for benchmarking.
BEAM_PROBE_INTERVAL = 5

# Default beam width for the ML-guided beam-search probe.
DEFAULT_BEAM_WIDTH = 20

# Default tabu tenure: how many steps a reverted (edge, value) move stays tabu.
DEFAULT_TABU_TENURE = 10


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
    candidates_evaluated: int
    is_complete: bool
    success: bool
    start_time: float

    best_near_miss: nx.Graph[int] | None
    best_near_miss_girth: int

    # Harvest mode (off by default; used by the forge producer).  When enabled
    # every k-regular, connected lift the search computes is appended to
    # ``harvested`` as ``(graph, girth)`` for a downstream consumer to drain.
    harvest: bool
    harvested: list[tuple[nx.Graph[int], int]]

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
    _kind: str
    _beam_probe_interval: int
    _beam_width: int
    _tabu_tenure: int

    def __init__(
        self,
        k: int,
        g: int,
        model_id: str | None = None,
        *,
        harvest: bool = False,
        beam_probe_interval: int = BEAM_PROBE_INTERVAL,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        tabu_tenure: int = DEFAULT_TABU_TENURE,
    ) -> None:
        self.k = k
        self.g = g
        self.step_count = 0
        self.is_complete = False
        self.success = False
        self.start_time = 0.0
        self.candidates_evaluated = 0
        self.best_near_miss = None
        self.best_near_miss_girth = 0
        self.harvest = harvest
        self.harvested = []
        self._best_girth = 0
        self._tabu = {}
        self._restarts = 0
        self._model_id = model_id
        self._feat_cycle_lengths = None
        self._feat_rwpe_dim = 0
        self._kind = "girth"
        self._beam_probe_interval = beam_probe_interval
        self._beam_width = beam_width
        self._tabu_tenure = tabu_tenure
        if model_id is not None:
            model, feat_cl, feat_rwpe, kind = load_girth_predictor(model_id)
            self._model = model
            self._feat_cycle_lengths = feat_cl
            self._feat_rwpe_dim = feat_rwpe
            self._kind = kind
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

    def _record_near_miss(self, lifted: nx.Graph[int], girth: int) -> None:
        """Keep the highest-girth k-regular, connected lift seen so far.

        The caller has already verified *lifted* is k-regular and connected with
        the given integer *girth* (< g).  This best near-miss is what the forge
        cascade hands to the refine stage when the voltage search cannot directly
        reach girth g.
        """
        if girth > self.best_near_miss_girth:
            self.best_near_miss_girth = girth
            self.best_near_miss = lifted

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
        self.candidates_evaluated += 1

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
                    if self.harvest:
                        # Harvest the hit and keep searching: the forge producer
                        # owns the lifecycle, so we never self-complete.
                        self.harvested.append((lifted, girth))
                    else:
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
                self.candidates_evaluated += 1
                self._voltages[edge_idx] = old_val

                if new_cost < best_move_cost and (not is_tabu or new_cost == 0):
                    best_move_cost = new_cost
                    best_edge = edge_idx
                    best_val = new_val

        if best_edge >= 0 and best_move_cost <= current_cost:
            old_val = self._voltages[best_edge]
            self._voltages[best_edge] = best_val
            self._tabu[(best_edge, old_val)] = self.step_count + self._tabu_tenure
        else:
            # Stuck — random restart with next config
            self._random_restart()

        # ML-guided beam-search probe: when a girth predictor is loaded,
        # periodically run a full beam search on the current (base, group)
        # config. Cheap one-shot — beam_search is sequential and bounded.
        if self._model is not None and self.step_count % self._beam_probe_interval == 0:
            volts_b, girth_b = beam_search(
                self._base,
                self._group,
                self.k,
                self.g,
                model=self._model,
                beam_width=self._beam_width,
                verbose=False,
                feat_cycle_lengths=self._feat_cycle_lengths,
                feat_rwpe_dim=self._feat_rwpe_dim,
                kind=self._kind,
            )
            if isinstance(girth_b, int) and volts_b is not None:
                lifted = build_lift(self._base, self._group, volts_b)
                props = verify_lift(lifted, self.k, self.g)
                if girth_b >= self.g and props["is_valid_kg"]:
                    # Full solution found via beam: mark complete (or harvest).
                    self._voltages = volts_b
                    self.graph = lifted
                    if self.harvest:
                        self.harvested.append((lifted, girth_b))
                    else:
                        self.is_complete = True
                        self.success = True
                        self._best_girth = girth_b
                        return
                elif (
                    props["is_k_regular"]
                    and props["is_connected"]
                    and girth_b > self._best_girth
                ):
                    # Warm-start adoption: the beam found a valid k-regular,
                    # connected lift whose girth beats our running best even
                    # though it does not yet reach g.  Adopt the new voltage
                    # assignment as the tabu starting point and clear the tabu
                    # list — the move history no longer applies to the new
                    # assignment.
                    self._voltages = volts_b
                    self._best_girth = girth_b
                    self.graph = lifted
                    self._tabu.clear()
                    self._record_near_miss(lifted, girth_b)

        # Check intermediate girth periodically.  In harvest mode we probe on
        # EVERY step (the forge producer wants a steady stream of lifts), and
        # collect every k-regular, connected lift; otherwise we keep the
        # original best-girth-gated behaviour so plain voltage is unaffected.
        if self.harvest or self.step_count % 20 == 0:
            girth = compute_lift_girth(
                self._base, self._group, self._voltages, max_girth=2 * self.g
            )
            if isinstance(girth, int) and (self.harvest or girth > self._best_girth):
                lifted = build_lift(self._base, self._group, self._voltages)
                props = verify_lift(lifted, self.k, self.g)
                if props["is_valid_kg"]:
                    if not self.harvest:
                        self._best_girth = girth
                        self.graph = lifted
                        self.is_complete = True
                        self.success = True
                    else:
                        self.graph = lifted
                        self.harvested.append((lifted, girth))
                elif props["is_k_regular"] and props["is_connected"]:
                    # Valid graph, just not high enough girth yet — show it and
                    # remember it as the best near-miss for a later refine handoff.
                    self.graph = lifted
                    if not self.harvest:
                        self._best_girth = girth
                    self._record_near_miss(lifted, girth)
                    if self.harvest:
                        self.harvested.append((lifted, girth))

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


def _candidate_groups_for_order_range(
    min_order: int,
    max_order: int,
) -> list[tuple[FiniteGroup, str]]:
    """Return a diverse set of groups whose order is in [min_order, max_order].

    Includes cyclic groups (Z_n), dihedral groups (D_m, order 2m), direct
    products of two cyclic groups (Z_a x Z_b), and semidirect products
    Z_n ⋊ Z_m — the same variety used by the supervised meta-search and
    the known records for hard targets like (8,5)/(9,5)/(10,5)/(11,6).
    """
    groups: list[tuple[FiniteGroup, str]] = []

    for n in range(min_order, max_order + 1):
        groups.append((cyclic_group(n), f"Z_{n}"))

    # Dihedral D_m has order 2m — include when 2m is in range.
    for m in range(max(2, min_order // 2), max_order // 2 + 1):
        order = 2 * m
        if min_order <= order <= max_order:
            groups.append((dihedral_group(m), f"D_{m}"))

    # Direct products Z_a x Z_b (a <= b, both >= 2, order in range, a != b to
    # avoid duplicating Z_n for prime powers).
    for a in range(2, max_order):
        for b in range(a + 1, max_order):
            order = a * b
            if order < min_order:
                continue
            if order > max_order:
                break
            groups.append(
                (
                    direct_product(cyclic_group(a), cyclic_group(b)),
                    f"Z_{a}xZ_{b}",
                )
            )

    # Semidirect products Z_n ⋊_phi Z_m where phi generates a non-trivial
    # action (phi != 1, phi^m == 1 mod n).  One phi per (n, m) is enough.
    for n in range(3, max_order):
        for m in range(2, max_order):
            order = n * m
            if order < min_order:
                continue
            if order > max_order:
                break
            for phi in range(2, n):
                if pow(phi, m, n) == 1:
                    groups.append(
                        (
                            semidirect_product_cyclic(n, m, phi),
                            f"Z_{n}rtZ_{m}",
                        )
                    )
                    break  # one phi per (n, m) pair is sufficient

    return groups


def _build_configs(k: int, g: int) -> list[tuple[BaseGraph, FiniteGroup, str]]:
    """Build candidate (base_graph, group) configurations for target (k, g).

    A lift has exactly ``base.num_nodes * group.order`` vertices, known before
    searching.  A (k,g)-graph needs at least ``moore_bound(k, g)`` vertices, so
    any config whose lift order is below the Moore bound can NEVER yield a valid
    (k,g)-graph; we skip those outright rather than waste search on them.

    Groups are diverse: cyclic, dihedral, direct products Z_a x Z_b, and
    semidirect products.  The cyclic-only restriction was identified as the
    likely blocker for hard targets such as (8,5)/(9,5)/(10,5)/(11,6) whose
    known records use non-cyclic groups (e.g. SmallGroup(80,32)).
    """
    mb = moore_bound(k, g)
    configs: list[tuple[BaseGraph, FiniteGroup, str]] = []

    def _add(base: BaseGraph, group: FiniteGroup, name: str) -> None:
        # Skip sub-Moore lifts: they cannot reach girth g at any voltage.
        if base.num_nodes * group.order < mb:
            return
        configs.append((base, group, name))

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
            for group, gname in _candidate_groups_for_order_range(min_order, max_order):
                _add(base, group, f"{bname}+{gname}")
    else:
        base = dumbbell(k)
        min_order = max(5, mb // 2)
        # Window above the Moore-bound minimum: covers the "could beat the
        # record" band (known record group orders sit ~7-12 above min_order for
        # (8,5)/(9,5)/(10,5)/(11,6)) without a richer-group config explosion or
        # the old fixed-80 ceiling that excluded high-k targets like (11,6).
        max_order = min_order + 30
        for group, gname in _candidate_groups_for_order_range(min_order, max_order):
            _add(base, group, f"dumbbell+{gname}")

    if not configs:
        # Last resort: a dumbbell lift sized at or above the Moore bound.
        n_base = 2
        order = max(5, -(-mb // n_base))  # ceil(mb / n_base)
        configs.append((dumbbell(k), cyclic_group(order), "fallback"))

    return configs

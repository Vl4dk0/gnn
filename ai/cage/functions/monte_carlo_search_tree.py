"""
Monte Carlo Tree Search for cage graph generation (Baseline 2).

Proper MCTS with:
- Tree structure to remember explored states
- UCB1 selection for exploration/exploitation balance
- Simulation/rollout from leaf nodes
- Backpropagation to update statistics
"""

import random
import time
import math
import random
import networkx as nx
from typing import Any

from backend.utils.graph_utils import (
    compute_girth,
    moore_bound,
    moore_hoffman_upper_bound,
    is_k_regular,
    can_add_edge_preserving_girth,
    score_graph_quality,
)


class MCTSNode:
    """Node in the Monte Carlo Search Tree."""

    graph: nx.Graph
    k: int
    g: int
    parent: "MCTSNode | None"
    action: Any
    children: list["MCTSNode"]
    visits: int
    value: float
    untried_actions: list[tuple[str, Any]]
    is_terminal: bool

    def __init__(
        self,
        graph: nx.Graph,
        k: int,
        g: int,
        parent: "MCTSNode | None" = None,
        action: Any = None,
    ):
        self.graph = graph
        self.k = k
        self.g = g
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children = []
        self.visits = 0
        self.value = 0.0

        # Generate untried actions
        self.untried_actions = self._get_legal_actions()
        self.is_terminal = is_k_regular(self.graph, self.k)

    def _get_legal_actions(self) -> list[tuple[str, Any]]:
        """Generate legal actions from current state."""
        actions: list[tuple[str, Any]] = []
        nodes = list(self.graph.nodes())
        num_nodes = len(nodes)

        # Limit max nodes to avoid infinite expansion
        mb = moore_bound(self.k, self.g)
        max_nodes = int(mb * 1.5)

        # 1. Add Edge
        for i, u in enumerate(nodes):
            if self.graph.degree(u) >= self.k:  # type: ignore
                continue
            for v in nodes[i + 1 :]:
                if self.graph.degree(v) >= self.k:  # type: ignore
                    continue
                if not self.graph.has_edge(u, v):
                    if can_add_edge_preserving_girth(self.graph, u, v, self.g):
                        actions.append(("add_edge", (u, v)))

        # 2. Add Node (only if we need more nodes)
        if num_nodes < max_nodes:
            actions.append(("add_node", None))

        random.shuffle(actions)
        return actions

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = 1.414) -> "MCTSNode | None":
        """Select best child using UCT formula."""
        if not self.children:
            return None

        choices_weights = [
            (child.value / child.visits)
            + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]


class MCTSGenerator:
    """Monte Carlo Tree Search for Cage Generation."""

    k: int
    g: int
    root: MCTSNode
    graph: nx.Graph
    step_count: int
    is_complete: bool
    success: bool
    start_time: float
    iterations: int

    def __init__(self, k: int, g: int):
        self.k = k
        self.g = g

        # Start with Moore bound vertices
        mb = moore_bound(k, g)
        initial_graph = nx.Graph()
        for i in range(mb):
            initial_graph.add_node(i)

        self.root = MCTSNode(initial_graph, k, g)
        self.graph = initial_graph

        self.step_count = 0
        self.is_complete = False
        self.success = False
        self.start_time = time.time()
        self.iterations = 0

    def elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time

    def is_regular(self) -> bool:
        """Check if graph is k-regular."""
        return is_k_regular(self.graph, self.k)

    def step(self) -> None:
        """Execute one MCTS iteration."""
        self.step_count += 1

        # 1. Selection
        node = self.select_node()

        # 2. Expansion
        if not node.is_terminal and not node.is_fully_expanded():
            node = self.expand(node)

        # 3. Simulation
        score = self.simulate(node)

        # 4. Backpropagation
        self.backpropagate(node, score)

        # Update best graph found so far (from root's best child)
        best_child = self.root.best_child(c_param=0)  # Exploitation only
        if best_child:
            self.graph = best_child.graph

            # Check success
            if is_k_regular(self.graph, self.k):
                if compute_girth(self.graph) == self.g:
                    self.is_complete = True
                    self.success = True

    def select_node(self) -> MCTSNode:
        """Select a promising node to expand."""
        node = self.root
        while node.is_fully_expanded() and not node.is_terminal:
            child = node.best_child()
            if child:
                node = child
            else:
                break
        return node

    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expand a leaf node."""
        action = node.untried_actions.pop()
        action_type, action_data = action

        new_graph = node.graph.copy()
        if action_type == "add_edge":
            u, v = action_data
            new_graph.add_edge(u, v)
        elif action_type == "add_node":
            new_id = len(new_graph.nodes())
            new_graph.add_node(new_id)

        child_node = MCTSNode(new_graph, self.k, self.g, parent=node, action=action)
        node.children.append(child_node)
        return child_node

    def simulate(self, node: MCTSNode) -> float:
        """Simulate a random rollout from the node."""
        # Instead of full random rollout (too slow), we use a heuristic score
        # + partial random steps

        curr_graph = node.graph.copy()

        # Do a few random steps
        steps = 5
        for _ in range(steps):
            if is_k_regular(curr_graph, self.k):
                break

            nodes = list(curr_graph.nodes())
            # Try to add edge
            added = False
            random.shuffle(nodes)

            for i, u in enumerate(nodes):
                if curr_graph.degree(u) >= self.k:  # type: ignore
                    continue
                for v in nodes[i + 1 :]:
                    if curr_graph.degree(v) >= self.k:  # type: ignore
                        continue
                    if not curr_graph.has_edge(u, v):
                        if can_add_edge_preserving_girth(curr_graph, u, v, self.g):
                            curr_graph.add_edge(u, v)
                            added = True
                            break
                if added:
                    break

            if not added:
                # Add node
                new_id = len(curr_graph.nodes())
                curr_graph.add_node(new_id)

        return score_graph_quality(curr_graph, self.k, self.g)

    def backpropagate(self, node: MCTSNode | None, score: float) -> None:
        """Backpropagate the score up the tree."""
        while node is not None:
            node.visits += 1
            node.value += score
            node = node.parent

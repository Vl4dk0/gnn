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
import copy
import networkx as nx

from utils.graph_utils import (
    compute_girth,
    moore_bound,
    is_valid_cage,
    can_add_edge_preserving_girth,
    is_k_regular
)


class MCTSNode:
    """Node in the MCTS tree representing a graph state."""
    
    def __init__(self, graph, k, g, parent=None, action=None):
        self.graph = graph
        self.k = k
        self.g = g
        self.parent = parent
        self.action = action
        
        self.children = {}
        self.visits = 0
        self.value = 0.0
        
        self.untried_actions = self._get_possible_actions()
        self.is_terminal = False
        
        if is_k_regular(graph, k):
            girth = compute_girth(graph)
            if girth == g:
                self.is_terminal = True
    
    def _get_possible_actions(self):
        """Get list of possible actions from this state."""
        actions = []
        nodes = list(self.graph.nodes())  # type: ignore
        
        # Always consider adding a vertex
        actions.append(('add_vertex', None, None))
        
        # Prioritize edge additions - try many combinations
        if len(nodes) >= 2:
            # Try more edge combinations to encourage graph building
            for _ in range(min(50, len(nodes) * len(nodes))):
                u, v = random.sample(nodes, 2)
                if not self.graph.has_edge(u, v):  # type: ignore
                    if self.graph.degree(u) < self.k and self.graph.degree(v) < self.k:  # type: ignore
                        if can_add_edge_preserving_girth(self.graph, u, v, self.g):
                            actions.append(('add_edge', u, v))
        
        # Consider removing edges (but less frequently)
        edges = list(self.graph.edges())  # type: ignore
        for u, v in random.sample(edges, min(3, len(edges))):
            actions.append(('remove_edge', u, v))
        
        # Consider removing vertices (least frequently)
        if len(nodes) > 0:
            for node in random.sample(nodes, min(2, len(nodes))):
                actions.append(('remove_vertex', node, None))
        
        return actions
    
    def is_fully_expanded(self):
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param=1.4):
        """Select best child using UCB1 formula."""
        choices_weights = []
        for child in self.children.values():
            if child.visits == 0:
                weight = float('inf')
            else:
                exploit = child.value / child.visits
                explore = c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
                weight = exploit + explore
            choices_weights.append((child, weight))
        
        return max(choices_weights, key=lambda x: x[1])[0]
    
    def expand(self):
        """Expand node by trying an untried action."""
        if not self.untried_actions:
            return None
        
        action = self.untried_actions.pop(random.randrange(len(self.untried_actions)))
        
        new_graph = copy.deepcopy(self.graph)
        action_type, param1, param2 = action
        
        if action_type == 'add_vertex':
            new_id = max(new_graph.nodes(), default=-1) + 1  # type: ignore
            new_graph.add_node(new_id)  # type: ignore
        elif action_type == 'add_edge':
            new_graph.add_edge(param1, param2)  # type: ignore
        elif action_type == 'remove_edge':
            if new_graph.has_edge(param1, param2):  # type: ignore
                new_graph.remove_edge(param1, param2)  # type: ignore
        elif action_type == 'remove_vertex':
            if param1 in new_graph.nodes():  # type: ignore
                new_graph.remove_node(param1)  # type: ignore
        
        child_node = MCTSNode(new_graph, self.k, self.g, parent=self, action=action)
        self.children[action] = child_node
        return child_node
    
    def update(self, reward):
        """Update node statistics after simulation."""
        self.visits += 1
        self.value += reward


class MCTSGenerator:
    """Monte Carlo Tree Search generator for cage graphs."""
    
    def __init__(self, k, g, max_iterations=1000, c_param=0.5, reroot_interval=5):
        self.k = k
        self.g = g
        self.max_iterations = max_iterations
        self.c_param = c_param  # Lower = more exploitation, less exploration
        self.reroot_interval = reroot_interval  # Move root forward every N steps
        
        initial_graph = nx.Graph()  # type: ignore
        self.root = MCTSNode(initial_graph, k, g)
        
        self.step_count = 0
        self.is_complete = False
        self.success = False
        self.start_time = time.time()
        self.best_graph = None
    
    @property
    def graph(self):
        """Return the best graph found so far."""
        if self.best_graph is not None:
            return self.best_graph
        return self.root.graph
    
    def elapsed_time(self):
        """Get elapsed time since start."""
        return time.time() - self.start_time
    
    def is_regular(self):
        """Check if current best graph is regular."""
        if self.best_graph is None:
            return False
        return is_k_regular(self.best_graph, self.k)
    
    def step(self):
        """Execute one MCTS iteration."""
        self.step_count += 1
        
        # Selection: Navigate to a promising leaf node
        node = self._select(self.root)
        
        # Expansion: Try a new action if possible
        if not node.is_terminal and not node.is_fully_expanded():
            node = node.expand()
            if node is None:
                return
        
        # Simulation: Evaluate the node
        reward = self._simulate(node)
        
        # Backpropagation: Update statistics
        self._backpropagate(node, reward)
        
        # Update best graph: Always track the best node found so far
        if self.best_graph is None:
            self.best_graph = node.graph
        else:
            current_score = self._evaluate_graph(self.best_graph)
            node_score = self._evaluate_graph(node.graph)
            if node_score > current_score:
                self.best_graph = node.graph
        
        # Check for success
        if node.is_terminal:
            self.is_complete = True
            self.success = True
        
        # Periodically move root forward to commit to best path
        if self.step_count % self.reroot_interval == 0 and len(self.root.children) > 0:
            # Find the child with highest value (best average reward)
            best_child = max(
                self.root.children.values(),
                key=lambda c: (c.value / c.visits) if c.visits > 0 else 0
            )
            # Only reroot if the child is actually better than current root
            if self._evaluate_graph(best_child.graph) > self._evaluate_graph(self.root.graph):
                best_child.parent = None
                self.root = best_child
    
    def _select(self, node):
        """Select a leaf node using UCB1."""
        while not node.is_terminal and node.is_fully_expanded():
            if len(node.children) == 0:
                break
            node = node.best_child(self.c_param)
        return node
    
    def _simulate(self, node):
        """Simulate random playout from node."""
        if node.is_terminal:
            return 1.0
        
        temp_graph = copy.deepcopy(node.graph)
        
        for _ in range(10):
            action_type = random.choice(['add_vertex', 'add_edge', 'remove_edge', 'remove_vertex'])
            nodes = list(temp_graph.nodes())  # type: ignore
            
            if action_type == 'add_vertex':
                new_id = max(temp_graph.nodes(), default=-1) + 1  # type: ignore
                temp_graph.add_node(new_id)  # type: ignore
            
            elif action_type == 'add_edge' and len(nodes) >= 2:
                u, v = random.sample(nodes, 2)
                if not temp_graph.has_edge(u, v):  # type: ignore
                    if temp_graph.degree(u) < self.k and temp_graph.degree(v) < self.k:  # type: ignore
                        if can_add_edge_preserving_girth(temp_graph, u, v, self.g):
                            temp_graph.add_edge(u, v)  # type: ignore
            
            elif action_type == 'remove_edge':
                edges = list(temp_graph.edges())  # type: ignore
                if edges:
                    u, v = random.choice(edges)
                    temp_graph.remove_edge(u, v)  # type: ignore
            
            elif action_type == 'remove_vertex' and len(nodes) > 0:
                node_to_remove = random.choice(nodes)
                temp_graph.remove_node(node_to_remove)  # type: ignore
            
            if is_k_regular(temp_graph, self.k):
                girth = compute_girth(temp_graph)
                if girth == self.g:
                    return 1.0
        
        return self._evaluate_graph(temp_graph)
    
    def _evaluate_graph(self, graph):
        """Evaluate how close a graph is to being a valid cage."""
        num_nodes = len(graph.nodes())  # type: ignore
        if num_nodes == 0:
            return 0.0
        
        mb = moore_bound(self.k, self.g)
        num_edges = len(graph.edges())  # type: ignore
        
        # Regularity score: proportion of nodes at correct degree
        nodes_at_k = sum(1 for node in graph.nodes() if graph.degree(node) == self.k)  # type: ignore
        regularity = nodes_at_k / num_nodes
        
        # Size score: reward getting closer to Moore bound
        size_score = min(num_nodes / mb, 1.0)
        
        # Edge density: reward having edges (k-regular graph should have n*k/2 edges)
        expected_edges = (num_nodes * self.k) / 2
        if expected_edges > 0:
            edge_score = min(num_edges / expected_edges, 1.0)
        else:
            edge_score = 0.0
        
        # Girth score: more lenient during construction
        girth = compute_girth(graph)
        if girth == float('inf'):
            # No cycles yet - give credit based on graph progress
            girth_score = 0.4 + 0.3 * edge_score
        elif girth < self.g:
            # Penalize but don't completely kill the score
            girth_score = 0.1
        elif girth == self.g:
            # Perfect girth
            girth_score = 1.0
        else:
            # Girth too large
            girth_score = 0.6
        
        # Weighted combination: strongly favor building up the graph
        return 0.3 * regularity + 0.3 * size_score + 0.3 * edge_score + 0.1 * girth_score
    
    def _backpropagate(self, node, reward):
        """Backpropagate reward up the tree."""
        while node is not None:
            node.update(reward)
            node = node.parent

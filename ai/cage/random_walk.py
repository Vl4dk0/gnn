"""
Random Walk for cage graph generation.

Smart probabilistic construction starting from Moore bound structure:
- Initialize with Moore bound nodes
- Analyze graph state (degree distribution, edge deficit)
- Probabilistically select actions based on current state
- Adaptively adjust probabilities to escape local optima
"""

import random
import time
import networkx as nx

from utils.graph_utils import (
    compute_girth,
    moore_bound,
    moore_hoffman_upper_bound,
    can_add_edge_preserving_girth,
    is_k_regular
)


class RandomWalkGenerator:
    """Random walk cage generator starting from Moore bound structure."""
    
    def __init__(self, k, g):
        self.k = k
        self.g = g
        self.mb = moore_bound(k, g)
        self.upper_bound = moore_hoffman_upper_bound(k, g)
        
        # Initialize graph with Moore bound nodes
        self.graph = nx.Graph()  # type: ignore
        for i in range(self.mb):
            self.graph.add_node(i)  # type: ignore
        
        self.step_count = 0
        self.is_complete = False
        self.success = False
        self.start_time = time.time()
        
        # Track expected edges for k-regularity
        self.target_edges = (self.graph.number_of_nodes() * k) // 2  # type: ignore
    
    def elapsed_time(self):
        """Get elapsed time since start."""
        return time.time() - self.start_time
    
    def is_regular(self):
        """Check if graph is k-regular."""
        return is_k_regular(self.graph, self.k)
    
    def step(self):
        """Execute one construction step."""
        self.step_count += 1
        
        # Check if we've exceeded the upper bound
        if self.graph.number_of_nodes() > self.upper_bound:  # type: ignore
            self.is_complete = True
            self.success = False
            return
        
        # Check if we've found a valid cage
        if is_k_regular(self.graph, self.k):
            girth = compute_girth(self.graph)
            if girth == self.g:
                self.is_complete = True
                self.success = True
                return
        
        # Get current state
        num_nodes = self.graph.number_of_nodes()  # type: ignore
        num_edges = self.graph.number_of_edges()  # type: ignore
        self.target_edges = (num_nodes * self.k) // 2
        
        # Calculate degree distribution
        nodes = list(self.graph.nodes())  # type: ignore
        low_degree_nodes = [n for n in nodes if self.graph.degree(n) < self.k]  # type: ignore
        high_degree_nodes = [n for n in nodes if self.graph.degree(n) > self.k]  # type: ignore
        correct_degree_nodes = [n for n in nodes if self.graph.degree(n) == self.k]  # type: ignore
        
        # Decision probabilities based on current state
        edge_deficit = self.target_edges - num_edges
        
        # Calculate action probabilities based on state
        # More flexibility to restructure when stuck
        
        if len(high_degree_nodes) > 0:
            # If we have nodes with degree > k, strongly prefer removing edges
            action_probs = {'remove_edge_high': 0.8, 'remove_edge': 0.15, 'remove_vertex': 0.05}
        
        elif edge_deficit > 0 and len(low_degree_nodes) >= 2:
            # Need more edges - but still allow restructuring
            action_probs = {
                'add_edge': 0.75,
                'remove_edge': 0.10,
                'add_vertex': 0.10 if num_nodes < self.mb else 0.0,
                'remove_vertex': 0.05 if num_nodes > self.mb else 0.0
            }
        
        elif edge_deficit < 0:
            # Too many edges - remove or restructure
            action_probs = {
                'remove_edge': 0.7,
                'remove_vertex': 0.2 if num_nodes > self.mb else 0.0,
                'add_vertex': 0.1 if num_nodes < self.mb else 0.0
            }
        
        elif len(low_degree_nodes) == 1:
            # Odd node out - need to restructure
            action_probs = {
                'remove_edge': 0.5,
                'remove_vertex': 0.25,
                'add_vertex': 0.15 if num_nodes < self.mb else 0.0,
                'add_edge': 0.1
            }
        
        else:
            # General case or stuck - balanced exploration
            action_probs = {
                'remove_edge': 0.4,
                'add_edge': 0.25,
                'remove_vertex': 0.2 if num_nodes > self.mb else 0.1,
                'add_vertex': 0.15 if num_nodes < self.mb else 0.05
            }
        
        # Normalize probabilities
        total = sum(action_probs.values())
        if total > 0:
            action_probs = {k: v/total for k, v in action_probs.items()}
        
        # Choose action based on probabilities
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        action = random.choices(actions, weights=probs, k=1)[0]
        
        # Execute chosen action
        if action == 'add_edge':
            self._add_edge_between_low_degree(low_degree_nodes)
        elif action == 'remove_edge_high':
            self._remove_edge_from_high_degree(high_degree_nodes)
        elif action == 'remove_edge':
            self._remove_random_edge()
        elif action == 'add_vertex':
            self._add_vertex()
        elif action == 'remove_vertex':
            self._remove_vertex()
    
    def _add_edge_between_low_degree(self, low_degree_nodes):
        """Try to add an edge between two low-degree nodes."""
        # Try multiple pairs to find valid edge
        attempts = min(50, len(low_degree_nodes) * len(low_degree_nodes))
        
        for _ in range(attempts):
            u, v = random.sample(low_degree_nodes, 2)
            
            if not self.graph.has_edge(u, v):  # type: ignore
                # Check girth constraint
                if can_add_edge_preserving_girth(self.graph, u, v, self.g):
                    self.graph.add_edge(u, v)  # type: ignore
                    return
    
    def _remove_edge_from_high_degree(self, high_degree_nodes):
        """Remove an edge from a high-degree node."""
        node = random.choice(high_degree_nodes)
        neighbors = list(self.graph.neighbors(node))  # type: ignore
        
        if neighbors:
            neighbor = random.choice(neighbors)
            self.graph.remove_edge(node, neighbor)  # type: ignore
    
    def _remove_random_edge(self):
        """Remove a random edge."""
        edges = list(self.graph.edges())  # type: ignore
        if edges:
            u, v = random.choice(edges)
            self.graph.remove_edge(u, v)  # type: ignore
    
    def _add_vertex(self):
        """Add a new vertex."""
        new_id = max(self.graph.nodes(), default=-1) + 1  # type: ignore
        self.graph.add_node(new_id)  # type: ignore
        self.target_edges = (self.graph.number_of_nodes() * self.k) // 2  # type: ignore
    
    def _remove_vertex(self):
        """Remove a random vertex (preferably low degree)."""
        nodes = list(self.graph.nodes())  # type: ignore
        if nodes:
            # Prefer removing nodes with degree 0 or 1
            low_deg = [n for n in nodes if self.graph.degree(n) <= 1]  # type: ignore
            if low_deg:
                node = random.choice(low_deg)
            else:
                node = random.choice(nodes)
            self.graph.remove_node(node)  # type: ignore
            self.target_edges = (self.graph.number_of_nodes() * self.k) // 2  # type: ignore

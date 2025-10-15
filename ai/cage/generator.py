"""
Monte Carlo graph generation for cage graphs.

Smart incremental validation approach.
"""

import random
import time
import networkx as nx

from utils.graph_utils import compute_girth, moore_bound, is_valid_cage


class CageGenerator:
    """Monte Carlo generator for cage graphs with efficient validation."""
    
    def __init__(self, k, g):
        self.k = k
        self.g = g
        self.graph = nx.Graph()  # type: ignore
        self.step_count = 0
        self.is_complete = False
        self.success = False
        self.start_time = time.time()
        
        # Efficient regularity tracking
        # Count how many nodes have degree == k
        self.nodes_with_correct_degree = 0
    
    def elapsed_time(self):
        """Get elapsed time since start."""
        return time.time() - self.start_time
        
    def add_vertex(self):
        """Add a new isolated vertex to the graph."""
        new_id = max(self.graph.nodes(), default=-1) + 1  # type: ignore
        self.graph.add_node(new_id)  # type: ignore
        # New node has degree 0, not k, so doesn't contribute to regularity
        return new_id
    
    def can_add_edge(self, u, v):
        """
        Check if edge (u,v) can be added without violating constraints.
        
        Rules:
        1. Edge doesn't already exist
        2. Neither u nor v would exceed degree k
        3. Adding edge wouldn't create cycle < g
        """
        # Check if edge exists
        if self.graph.has_edge(u, v):  # type: ignore
            return False
        
        # Check degree constraints
        if self.graph.degree(u) >= self.k or self.graph.degree(v) >= self.k:  # type: ignore
            return False
        
        # Check girth constraint - smart incremental check
        # Only need to find shortest cycle through the new edge
        if not self._would_preserve_girth(u, v):
            return False
        
        return True
    
    def _would_preserve_girth(self, u, v):
        """
        Check if adding edge (u,v) would preserve girth >= g.
        
        Smart approach: Find shortest path from v to u in current graph.
        If path exists and path_length + 1 < g, adding edge creates too-small cycle.
        """
        if u == v:
            # Self-loop creates cycle of length 1
            return self.g <= 1
        
        try:
            # Find shortest path from v to u (excluding the direct edge)
            path_length = nx.shortest_path_length(self.graph, u, v)  # type: ignore
            # Adding edge would create cycle of length path_length + 1
            cycle_length = path_length + 1
            return cycle_length >= self.g
        except nx.NetworkXNoPath:  # type: ignore
            # No path exists, adding edge won't create a cycle
            return True
    
    def add_edge(self, u, v):
        """Add edge (u,v) to graph and update regularity count."""
        old_deg_u = self.graph.degree(u)  # type: ignore
        old_deg_v = self.graph.degree(v)  # type: ignore
        
        # Update regularity count (remove old contributions)
        if old_deg_u == self.k:
            self.nodes_with_correct_degree -= 1
        if old_deg_v == self.k and u != v:
            self.nodes_with_correct_degree -= 1
        
        # Add edge
        self.graph.add_edge(u, v)  # type: ignore
        
        # Update regularity count (add new contributions)
        new_deg_u = self.graph.degree(u)  # type: ignore
        new_deg_v = self.graph.degree(v)  # type: ignore
        
        if new_deg_u == self.k:
            self.nodes_with_correct_degree += 1
        if new_deg_v == self.k and u != v:
            self.nodes_with_correct_degree += 1
    
    def remove_edge(self, u, v):
        """Remove edge (u,v) and update regularity count."""
        if not self.graph.has_edge(u, v):  # type: ignore
            return
        
        old_deg_u = self.graph.degree(u)  # type: ignore
        old_deg_v = self.graph.degree(v)  # type: ignore
        
        # Update regularity count (remove old contributions)
        if old_deg_u == self.k:
            self.nodes_with_correct_degree -= 1
        if old_deg_v == self.k and u != v:
            self.nodes_with_correct_degree -= 1
        
        # Remove edge
        self.graph.remove_edge(u, v)  # type: ignore
        
        # Update regularity count (add new contributions)
        new_deg_u = self.graph.degree(u)  # type: ignore
        new_deg_v = self.graph.degree(v)  # type: ignore
        
        if new_deg_u == self.k:
            self.nodes_with_correct_degree += 1
        if new_deg_v == self.k and u != v:
            self.nodes_with_correct_degree += 1
    
    def remove_vertex(self, node):
        """Remove vertex and all its edges, update regularity count."""
        if node not in self.graph.nodes():  # type: ignore
            return
        
        # Get neighbors before removal
        neighbors = list(self.graph.neighbors(node))  # type: ignore
        deg_node = self.graph.degree(node)  # type: ignore
        
        # Update regularity count for node being removed
        if deg_node == self.k:
            self.nodes_with_correct_degree -= 1
        
        # Update regularity count for neighbors (they'll lose an edge)
        for neighbor in neighbors:
            old_deg = self.graph.degree(neighbor)  # type: ignore
            if old_deg == self.k:
                self.nodes_with_correct_degree -= 1
            if int(old_deg) - 1 == self.k:  # Will become regular after edge removal  # type: ignore
                self.nodes_with_correct_degree += 1
        
        # Remove node
        self.graph.remove_node(node)  # type: ignore
    
    def is_regular(self):
        """Check if all nodes have degree k."""
        num_nodes = len(self.graph.nodes())  # type: ignore
        return num_nodes > 0 and self.nodes_with_correct_degree == num_nodes
    
    def monte_carlo_step(self):
        """
        Execute one Monte Carlo step for graph generation.
        
        Operations (with probabilities):
        1. ADD_VERTEX - higher probability if below Moore bound
        2. ADD_EDGE - main construction operation
        3. REMOVE_EDGE - small probability for exploration
        4. REMOVE_VERTEX - very small probability for major restructuring
        """
        self.step_count += 1
        
        num_nodes = len(self.graph.nodes())  # type: ignore
        mb = moore_bound(self.k, self.g)
        
        # Define operation probabilities based on current state
        if num_nodes < mb:
            # Need more vertices
            probs = {'add_vertex': 0.6, 'add_edge': 0.35, 'remove_edge': 0.04, 'remove_vertex': 0.01}
        elif num_nodes == mb:
            # At target size, focus on edges
            probs = {'add_vertex': 0.1, 'add_edge': 0.7, 'remove_edge': 0.15, 'remove_vertex': 0.05}
        else:
            # Too many vertices, consider removing
            probs = {'add_vertex': 0.05, 'add_edge': 0.5, 'remove_edge': 0.2, 'remove_vertex': 0.25}
        
        # Choose operation
        operation = random.choices(
            list(probs.keys()),
            weights=list(probs.values()),
            k=1
        )[0]
        
        # Execute operation
        if operation == 'add_vertex':
            self.add_vertex()
            
        elif operation == 'add_edge':
            # Try to add a random valid edge
            nodes = list(self.graph.nodes())  # type: ignore
            if len(nodes) < 2:
                # Not enough nodes, add one instead
                self.add_vertex()
            else:
                # Try random edges until we find a valid one (max attempts)
                max_attempts = 20
                for _ in range(max_attempts):
                    u, v = random.sample(nodes, 2)
                    if self.can_add_edge(u, v):
                        self.add_edge(u, v)
                        break
                        
        elif operation == 'remove_edge':
            edges = list(self.graph.edges())  # type: ignore
            if edges:
                u, v = random.choice(edges)
                self.remove_edge(u, v)
                
        elif operation == 'remove_vertex':
            nodes = list(self.graph.nodes())  # type: ignore
            if nodes:
                node = random.choice(nodes)
                self.remove_vertex(node)
        
        # Check if we should terminate
        # Complete if: regular + correct girth + reasonable number of steps
        if self.is_regular() and num_nodes >= mb:
            girth = compute_girth(self.graph)  # type: ignore
            if girth >= self.g:
                self.is_complete = True
                self.success = True
        
        # Also terminate if too many steps
        if self.step_count >= 500:
            self.is_complete = True
            self.success = is_valid_cage(self.graph, self.k, self.g)  # type: ignore

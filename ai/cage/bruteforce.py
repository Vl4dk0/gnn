"""
Bruteforce backtracking search for cage generation.

This algorithm:
1. Only performs additive operations (add edge, add vertex)
2. At each step, generates ALL possible valid actions
3. Scores each action by quality (how close to a valid cage)
4. Explores actions in greedy order (best first)
5. Backtracks when stuck and tries next best option
6. No deletion needed - purely forward search with backtracking
"""

import time
import copy
import networkx as nx
from typing import List, Tuple, Optional

from utils.graph_utils import (
    compute_girth,
    moore_bound,
    moore_hoffman_upper_bound,
    is_k_regular,
    can_add_edge_preserving_girth,
    score_graph_quality
)


class BruteforceGenerator:
    """Bruteforce backtracking search for cage graphs."""
    
    def __init__(self, k, g):
        self.k = k
        self.g = g
        self.mb = moore_bound(k, g)
        self.upper_bound = moore_hoffman_upper_bound(k, g)
        
        # Start with empty graph
        self.graph = nx.Graph()
        
        # Search state
        self.search_stack = []  # Stack of (graph, actions_remaining)
        self.step_count = 0
        self.is_complete = False
        self.success = False
        self.start_time = time.time()
        
        # Initialize with first state
        initial_actions = self._get_all_actions(self.graph)
        self.search_stack.append((self.graph.copy(), initial_actions))
    
    def elapsed_time(self):
        """Get elapsed time since start."""
        return time.time() - self.start_time
    
    def is_regular(self):
        """Check if graph is k-regular."""
        return is_k_regular(self.graph, self.k)
    
    def step(self):
        """Execute one search step - try next action or backtrack."""
        self.step_count += 1
        
        # Check if we've exceeded the upper bound
        if self.graph.number_of_nodes() > self.upper_bound:  # type: ignore
            # This path is invalid - backtrack
            if self.search_stack:
                self.search_stack.pop()
                if self.search_stack:
                    self.graph = self.search_stack[-1][0].copy()
            else:
                self.is_complete = True
                self.success = False
            return
        
        # Check if current graph is a valid cage
        if is_k_regular(self.graph, self.k):
            girth = compute_girth(self.graph)
            if girth == self.g:
                self.is_complete = True
                self.success = True
                return
        
        # Check if we should continue exploring
        if not self.search_stack:
            # No more options to explore - failed
            self.is_complete = True
            self.success = False
            return
        
        # Get current state
        current_graph, actions_remaining = self.search_stack[-1]
        
        if not actions_remaining:
            # No more actions at this level - backtrack
            self.search_stack.pop()
            if self.search_stack:
                self.graph = self.search_stack[-1][0].copy()
            return
        
        # Try next best action
        action = actions_remaining.pop()
        action_type, action_data = action
        
        # Apply action to create new graph
        new_graph = current_graph.copy()
        
        if action_type == 'add_edge':
            u, v = action_data
            new_graph.add_edge(u, v)
        elif action_type == 'add_vertex':
            new_id = len(new_graph.nodes())  # type: ignore
            new_graph.add_node(new_id)
        
        # Update current graph
        self.graph = new_graph.copy()
        
        # Generate new actions for this state
        new_actions = self._get_all_actions(new_graph)
        
        # Push new state onto stack
        self.search_stack.append((new_graph.copy(), new_actions))

    def _get_all_actions(self, graph: nx.Graph) -> List[Tuple[str, any]]: # type: ignore
        """
        Generate all possible valid actions for the current graph state.
        Returns sorted list of (action_type, action_data) tuples.
        """
        actions = []
        
        num_nodes = len(graph.nodes()) # type: ignore
        
        # Option 1: Add a new vertex (always valid, but only if we need more)
        # Don't add too many vertices (heuristic: stop at 2x Moore bound)
        if num_nodes < self.mb * 2:
            score = self._score_add_vertex(graph)
            actions.append((score, 'add_vertex', None))
        
        # Option 2: Add edges between existing nodes
        nodes = list(graph.nodes())
        
        for i, u in enumerate(nodes):
            # Skip if u already has degree >= k
            if graph.degree(u) >= self.k: # type: ignore
                continue
                
            for v in nodes[i+1:]:  # Only consider u < v to avoid duplicates
                # Skip if v already has degree >= k
                if graph.degree(v) >= self.k: # type: ignore
                    continue
                
                # Skip if edge already exists
                if graph.has_edge(u, v):
                    continue
                
                # Check if adding edge would violate girth constraint
                if can_add_edge_preserving_girth(graph, u, v, self.g):
                    score = self._score_add_edge(graph, u, v)
                    actions.append((score, 'add_edge', (u, v)))
        
        # Sort actions by score (ascending - lower scores first)
        # We'll pop from the end, so best (highest score) will be at the end
        actions.sort(key=lambda x: x[0])
        
        # Return only action type and data (remove score)
        return [(action_type, action_data) for _, action_type, action_data in actions]
    
    def _score_add_vertex(self, graph: nx.Graph) -> float:
        """Score the action of adding a new vertex."""
        num_nodes = len(graph.nodes()) # type: ignore
        
        # Prefer adding vertices if we're below Moore bound
        if num_nodes < self.mb:
            return 1.0  # High priority
        elif num_nodes == self.mb:
            return 0.5  # Medium priority
        else:
            # Penalize adding too many vertices
            excess = num_nodes - self.mb
            return max(0.0, 0.3 - 0.1 * excess)
    
    def _score_add_edge(self, graph: nx.Graph, u: int, v: int) -> float:
        """
        Score the action of adding edge (u, v).
        Uses graph quality score after the edge is added.
        """
        # Create temporary graph with edge added
        temp_graph = graph.copy()
        temp_graph.add_edge(u, v)
        
        # Use existing quality scoring function
        base_score = score_graph_quality(temp_graph, self.k, self.g)
        
        # Bonus: prefer edges that balance degree distribution
        deg_u_after = temp_graph.degree(u) # type: ignore
        deg_v_after = temp_graph.degree(v) # type: ignore

        # Bonus if both nodes get closer to k
        balance_bonus = 0.0
        if deg_u_after == self.k:
            balance_bonus += 0.2
        if deg_v_after == self.k:
            balance_bonus += 0.2
        
        return base_score + balance_bonus

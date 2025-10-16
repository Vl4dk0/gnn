"""
A* / Best-First Search for cage generation.

This algorithm:
1. Starts with Moore bound vertices and no edges
2. Uses a priority queue to explore most promising graphs first
3. Generates all valid successor states (add edge or add vertex)
4. Scores each graph by "probability of success" (how likely it leads to a cage)
5. Uses hash table to avoid revisiting duplicate states
6. Optional: Uses Weisfeiler-Lehman hash for approximate isomorphism checking
"""

import time
import heapq
import networkx as nx
from typing import Set, Tuple, Optional

from utils.graph_utils import (
    compute_girth,
    moore_bound,
    moore_hoffman_upper_bound,
    is_k_regular,
    can_add_edge_preserving_girth,
    score_graph_quality
)


def graph_hash(G: nx.Graph) -> str:
    """
    Compute a hash for the graph to detect duplicates.
    
    Uses Weisfeiler-Lehman graph hash which is:
    - Fast O(n log n)
    - Good at distinguishing non-isomorphic graphs
    - Not perfect but works well in practice
    
    For exact isomorphism checking, we'd need to use VF2 algorithm
    which is exponential in worst case.
    """
    # Weisfeiler-Lehman graph hash (fast heuristic)
    try:
        return nx.weisfeiler_lehman_graph_hash(G)
    except:
        # Fallback to simple edge-based hash if WL fails
        edges = sorted([(min(u, v), max(u, v)) for u, v in G.edges()])
        return f"n{G.number_of_nodes()}_e{G.number_of_edges()}_{hash(tuple(edges))}"


class AStarGenerator:
    """A* search for cage graphs using priority queue."""
    
    def __init__(self, k, g):
        self.k = k
        self.g = g
        self.mb = moore_bound(k, g)
        self.upper_bound = moore_hoffman_upper_bound(k, g)
        
        # Start with Moore bound vertices and no edges
        initial_graph = nx.Graph()
        for i in range(self.mb):
            initial_graph.add_node(i)
        
        # Priority queue: (negative_score, counter, graph)
        # We use negative score because heapq is a min-heap
        self.counter = 0  # Tie-breaker for equal scores
        self.pq = []
        
        # Hash set to track visited states
        self.visited_hashes: Set[str] = set()
        
        # Current best graph (the one we'll return)
        self.graph = initial_graph.copy()
        
        # Add initial state to queue
        initial_score = self._score_graph(initial_graph)
        heapq.heappush(self.pq, (-initial_score, self.counter, initial_graph))
        self.counter += 1
        
        # Tracking
        self.step_count = 0
        self.is_complete = False
        self.success = False
        self.start_time = time.time()
        self.explored_states = 0
        self.duplicates_skipped = 0
    
    def elapsed_time(self):
        """Get elapsed time since start."""
        return time.time() - self.start_time
    
    def is_regular(self):
        """Check if graph is k-regular."""
        return is_k_regular(self.graph, self.k)
    
    def step(self):
        """
        Execute one A* step:
        1. Pop best graph from priority queue
        2. Check if it's a valid cage (goal state)
        3. Generate all valid successors
        4. Add new successors to queue (if not visited)
        """
        self.step_count += 1
        
        # Check if queue is empty
        if not self.pq:
            self.is_complete = True
            self.success = False
            return
        
        # Pop best candidate from priority queue
        neg_score, _, current_graph = heapq.heappop(self.pq)
        score = -neg_score
        
        # Update current graph for visualization
        self.graph = current_graph.copy()
        self.explored_states += 1
        
        # Check if we've exceeded the upper bound
        if current_graph.number_of_nodes() > self.upper_bound:
            # Skip this path - it can't lead to a valid cage
            return
        
        # Check if this is a valid cage (goal state)
        if is_k_regular(current_graph, self.k):
            girth = compute_girth(current_graph)
            if girth == self.g:
                self.is_complete = True
                self.success = True
                return
        
        # Generate all valid successors
        successors = self._generate_successors(current_graph)
        
        # Add successors to priority queue (if not already visited)
        for succ_graph in successors:
            # Compute hash to check for duplicates
            graph_h = graph_hash(succ_graph)
            
            if graph_h in self.visited_hashes:
                self.duplicates_skipped += 1
                continue
            
            # Mark as visited
            self.visited_hashes.add(graph_h)
            
            # Score and add to queue
            succ_score = self._score_graph(succ_graph)
            heapq.heappush(self.pq, (-succ_score, self.counter, succ_graph))
            self.counter += 1
    
    def _generate_successors(self, graph: nx.Graph) -> list:
        """
        Generate all valid successor graphs.
        
        Two types of actions:
        1. Add an edge between two existing vertices (if it preserves girth)
        2. Add a new vertex
        """
        successors = []
        nodes = list(graph.nodes())
        num_nodes = len(nodes)
        
        # Action 1: Add edges between existing vertices
        for i, u in enumerate(nodes):
            # Skip if u already has degree >= k
            if graph.degree(u) >= self.k:  # type: ignore
                continue
            
            for v in nodes[i+1:]:
                # Skip if v already has degree >= k
                if graph.degree(v) >= self.k:  # type: ignore
                    continue
                
                # Skip if edge already exists
                if graph.has_edge(u, v):
                    continue
                
                # Check if adding edge preserves girth constraint
                if can_add_edge_preserving_girth(graph, u, v, self.g):
                    new_graph = graph.copy()
                    new_graph.add_edge(u, v)
                    successors.append(new_graph)
        
        # Action 2: Add a new vertex (only if we haven't exceeded reasonable limit)
        # Heuristic: don't add more than 2x Moore bound vertices
        if num_nodes < self.mb * 2:
            new_graph = graph.copy()
            new_id = num_nodes  # Next sequential ID
            new_graph.add_node(new_id)
            successors.append(new_graph)
        
        return successors
    
    def _score_graph(self, graph: nx.Graph) -> float:
        """
        Score a graph based on how likely it is to lead to a valid cage.
        
        Higher score = more promising = explored first.
        
        Scoring components:
        1. Base quality score (regularity, size, girth)
        2. Progress towards goal (how close to being k-regular)
        3. Efficiency (prefer smaller graphs if possible)
        """
        num_nodes = len(graph.nodes())  # type: ignore
        num_edges = len(graph.edges())  # type: ignore
        
        # Base quality score
        base_score = score_graph_quality(graph, self.k, self.g)
        
        # Progress towards k-regularity
        if num_nodes > 0:
            regular_nodes = sum(1 for n in graph.nodes() if graph.degree(n) == self.k)  # type: ignore
            regularity_progress = regular_nodes / num_nodes
        else:
            regularity_progress = 0.0
        
        # Size preference: prefer graphs closer to Moore bound
        if num_nodes > 0:
            size_score = 1.0 - abs(num_nodes - self.mb) / max(self.mb, num_nodes)
        else:
            size_score = 0.0
        
        # Edge density: prefer graphs with appropriate number of edges
        # Target: n*k/2 edges for k-regular graph
        target_edges = (num_nodes * self.k) // 2
        if target_edges > 0:
            edge_score = 1.0 - abs(num_edges - target_edges) / max(target_edges, num_edges)
        else:
            edge_score = 0.0
        
        # Girth check: strong bonus if girth is exactly g
        current_girth = compute_girth(graph)
        if current_girth == self.g:
            girth_score = 1.0
        elif current_girth > self.g:
            girth_score = 0.8  # Good but not perfect
        elif current_girth == float('inf'):
            girth_score = 0.5  # No cycles yet
        else:
            girth_score = 0.0  # Girth too small - bad
        
        # Weighted combination
        final_score = (
            0.3 * base_score +
            0.3 * regularity_progress +
            0.15 * size_score +
            0.15 * edge_score +
            0.1 * girth_score
        )
        
        return final_score

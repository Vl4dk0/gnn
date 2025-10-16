"""Cage graph generation package."""

from .random_walk import RandomWalkGenerator
from .bruteforce import BruteforceGenerator
from .astar import AStarGenerator
from .monte_carlo_search_tree import MCTSGenerator

# Backwards compatibility: CageGenerator is RandomWalkGenerator
CageGenerator = RandomWalkGenerator

__all__ = ['CageGenerator', 'RandomWalkGenerator', 'BruteforceGenerator', 'AStarGenerator', 'MCTSGenerator']

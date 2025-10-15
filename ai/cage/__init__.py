"""Cage graph generation package."""

from .random_walk import RandomWalkGenerator
from .monte_carlo_search_tree import MCTSGenerator
# from .greedy import GreedyGenerator  # To be implemented
# from .beam_search import BeamSearchGenerator  # Future
# from .backtracking import BacktrackingGenerator  # Future

# Backwards compatibility: CageGenerator is RandomWalkGenerator (current default)
CageGenerator = RandomWalkGenerator

__all__ = ['CageGenerator', 'RandomWalkGenerator', 'MCTSGenerator']

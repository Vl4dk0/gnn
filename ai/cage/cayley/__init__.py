"""Cayley graph construction and search for (k, g)-cages.

A Cayley graph Gamma(G, S) of a finite group G with symmetric generating set
S = S^{-1} (no identity) has |G| vertices and is automatically |S|-regular and
vertex-transitive. The girth is the length of the shortest non-trivial
relation s_{i_1} ... s_{i_l} = e with no immediate cancellations.

This module mirrors the layout of ai/cage/voltage/: groups, search algorithms,
data generation, and an ML girth predictor that guides search.
"""

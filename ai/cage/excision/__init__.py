"""Tree excision + GNN repair for (k,g)-cage construction.

Based on the technique from Exoo, Goedgebeur, Jooken, Stubbe, Van den Eede (2025)
arXiv:2511.07247: BFS-remove a depth-d tree from a known (k,g)-graph, then
repair degree-deficient boundary vertices to produce a smaller valid (k,g)-graph.
"""

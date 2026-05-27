"""Post-lift tabu refinement for (k,g)-cage construction.

Modules:
    swaps    — 2-switch and 3-switch enumeration/application
    cost     — short-cycle cost function
    tabu     — TabuRefiner with optional GNN score_fn
    move_oracle — GNN Δcost predictor (MoveOracle)
    data_gen    — supervised dataset generation for move oracle
    train       — CLI training entry-point for MoveOracle
"""

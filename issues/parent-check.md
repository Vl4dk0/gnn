# parent-check

**Súbor:** `backend/utils/graph_utils.py:186-188`  
**Závažnosť:** LOW

## Popis

`_compute_girth_bfs` identifikuje rodiča podľa identity uzla, nie hrany:

```python
if neighbor == parent:
    continue
```

V multigrafoch (viacero hrán medzi dvoma uzlami) by sa preskočili VŠETKY hrany k rodičovskému uzlu, čím by sa premeškali 2-cykly z paralelných hrán.

## Dôsledok

Pre `nx.Graph` (simple graph bez paralelných hrán) nie je dopad žiadny. Problém nastane ak by sa funkcia zavolala s `nx.MultiGraph`. Chýba ochrana na vstupe.

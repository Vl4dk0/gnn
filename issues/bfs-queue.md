# bfs-queue

**Súbor:** `backend/utils/graph_utils.py:182`  
**Závažnosť:** MEDIUM

## Popis

`_compute_girth_bfs` používa zoznam ako frontu a volá `pop(0)`, čo je O(n):

```python
node, parent, dist = queue.pop(0)   # O(n) — každý prvok sa posúva
```

Pre BFS je správna štruktúra `collections.deque` s `popleft()` v O(1).

Rovnaký problém existuje v `ai/cage/voltage/base_graphs.py:93` (`spanning_tree_edge_indices`).

## Dôsledok

Girth výpočet je O(V² × E) namiesto O(V × (V+E)). Pre grafy s V~120 je to ~120× pomalší BFS na každý štartový uzol.

## Oprava

```python
from collections import deque

queue: deque[tuple[int, int, int]] = deque([(start_node, -1, 0)])
# ...
node, parent, dist = queue.popleft()
```

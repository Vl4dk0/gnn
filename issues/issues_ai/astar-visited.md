# astar-visited

**Súbor:** `ai/cage/functions/astar.py:89-91`  
**Závažnosť:** MEDIUM

## Popis

Počiatočný graf je pridaný do priority queue, ale NIE do `visited_hashes`:

```python
heapq.heappush(self.pq, (-initial_score, self.counter, initial_graph))
# visited_hashes zostáva prázdne — initial_graph chýba
```

Nástupníci sa pridávajú do `visited_hashes` pri objavení (riadok 163). Ak by nejaký vzdialený nástupník vygeneroval počiatočný stav, preskúmal by sa znova.

## Dôsledok

Pre A* kde sa hrany iba pridávajú (nie odoberajú) je dopad nulový — počiatočný stav sa nedá dosiahnuť spätne. Pre obojsmerný priestor stavov by to spôsobovalo cykly.

## Oprava

```python
initial_hash = graph_hash(initial_graph)
self.visited_hashes.add(initial_hash)
heapq.heappush(self.pq, (-initial_score, self.counter, initial_graph))
```

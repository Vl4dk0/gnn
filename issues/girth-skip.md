# girth-skip

**Súbor:** `ai/cage/voltage/search.py:77, 120`  
**Závažnosť:** LOW

## Popis

`exhaustive_search` aj `random_search` preskakujú priradenia kde `compute_lift_girth` vrátil `math.inf`:

```python
girth = compute_lift_girth(base, group, volts, max_girth=2 * g_target)
if isinstance(girth, float) or girth <= best_girth:
    continue
```

Lift s girth > 2×g_target je valídny (girth ≥ g_target), ale preskočí sa.

## Dôsledok

V praxi nízky dopad — connected k-regular lift s tak veľkým girthom je extrémne zriedkavý pri náhodnom vzorkovaní. Latentná logická chyba.

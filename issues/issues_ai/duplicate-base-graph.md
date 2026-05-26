# duplicate-base-graph

**Súbor:** `ai/cage/voltage/search.py:398-404`  
**Závažnosť:** LOW

## Popis

Funkcia `_candidate_bases(k)` pri $k=3$ pridá do zoznamu kandidátskych bázických grafov `dumbbell(3)` dvakrát. Prvýkrát ho pridá v `if k == 3:` bloku a následne ho bezpodmienečne pridá znova o riadok nižšie:

```python
    if k == 3:
        bases.append(("dumbbell(3)", dumbbell(3)))
        # ...
    bases.append((f"dumbbell({k})", dumbbell(k)))
```

## Dôsledok

Pri spustení `meta_search` pre $k=3$ sa vyhľadávanie nad konfiguráciami dumbbell(3) spúšťa duplicitne. To spôsobuje zbytočné výpočtové plytvanie a predlžuje celkový čas vyhľadávania.

## Oprava

Upraviť bezpodmienečné pridávanie tak, aby sa `dumbbell(k)` pridával iba vtedy, ak $k \neq 3$:

```python
    if k == 3:
        bases.append(("dumbbell(3)", dumbbell(3)))
        bases.append(("cubic_4nodes", cubic_multigraph_4nodes()))
        bases.append(("prism", prism_base()))
        bases.append(("petersen_base", moebius_kantor_base()))
    else:
        bases.append((f"dumbbell({k})", dumbbell(k)))
```

# tabu-discarded

**Súbor:** `ai/cage/voltage/search.py:453`  
**Závažnosť:** HIGH

## Popis

V `_search_one_config` tabu hľadanie nikdy nenahrádza výsledok náhodného hľadania:

```python
if best is None or lift_size < int(str(best.get("order", 999999))):
    best = candidate
```

`lift_size` a `best["order"]` sú vždy rovnaké — oba pochádzajú z tej istej dvojice `(base, group)`, teda `lift_size = base.num_nodes * group.order`. Porovnanie `lift_size < lift_size` je vždy `False`.

## Dôsledok

Tabu výsledok je zahodený vždy keď náhodné hľadanie niečo nájde, aj keby tabu dosiahlo vyšší girth. Tabu hľadanie je v rámci jednej `(base, group)` kombinácie efektívne zbytočné.

## Oprava

```python
if best is None or int(cast(int, candidate["girth"])) > int(cast(int, best["girth"])):
    best = candidate
```

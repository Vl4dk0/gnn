# cayley-group-data-inf-label-zero

**Súbor:** `ai/cage/cayley/group_data_gen.py:117, 123`  
**Závažnosť:** HIGH

## Popis

`best_achievable_girth` zahodí `float('inf')` girth z oboch zdrojov (random_search aj tabu_search) cez `isinstance(..., int)`. Skupina, ktorej najlepší dosiahnuteľný girth presahuje `2*g_target`, dostane label `best = 0` — identický s prípadom "vôbec sa nič nenašlo".

```python
# Riadok 117:
if isinstance(girth_r, int) and girth_r > best:
    best = girth_r   # ← float('inf') tu vypadne

# Riadok 123:
if gens_t is not None and isinstance(girth_t, int) and girth_t > best:
    best = girth_t   # ← float('inf') z tabu_search tu vypadne
```

Skupina s veľmi dobrým girthom (> `2*g_target`) sa teda správa rovnako ako zlá skupina — `GroupPromisePredictor` dostáva nesprávne trénovacie dáta.

## Dôsledok

Trénovacie dáta pre `GroupPromisePredictor` obsahujú mislabeled príklady: skupiny schopné produkovať veľmi vysoký girth sú označené rovnako ako skupiny bez potenciálu. Degraduje presnosť predikcie a znižuje efektivitu group-guided search.

## Oprava

```python
# Riadok 117:
if girth_r is not None and (isinstance(girth_r, int) or girth_r == float('inf')):
    effective = int(2 * g_target + 1) if girth_r == float('inf') else int(girth_r)
    if effective > best:
        best = effective

# Riadok 123:
if gens_t is not None:
    effective_t = int(2 * g_target + 1) if girth_t == float('inf') else (int(girth_t) if isinstance(girth_t, int) else 0)
    if effective_t > best and nx.is_connected(cay) and cay.degree(0) == k:
        best = effective_t
```

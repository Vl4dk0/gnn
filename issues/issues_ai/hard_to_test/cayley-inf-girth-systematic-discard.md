# cayley-inf-girth-systematic-discard

**Súbory:** `ai/cage/cayley/search.py:69, 179-192, 312, 403`  
**Závažnosť:** HIGH

## Popis

`cayley_girth` vracia `float('inf')` keď je skutočný girth > `2*g_target`. Takýto Cayleyho graf je platná `(k,g)`-klietka (girth ≥ g_target), ale všetci volajúci ho systematicky odmietajú cez `isinstance(girth, int)`.

**Riadok 69 — `random_search`:**
```python
if isinstance(girth, float) or girth <= best_girth:
    continue  # float('inf') vždy preskočený — platná klietka zahodená
```

**Riadky 179-192 — `tabu_search`:**
```python
girth = cayley_girth(group, gens, max_girth=2 * g_target)
if (
    isinstance(girth, int)   # ← float('inf') tu padne → podmienka False
    and girth >= g_target
    ...
):
    return gens, girth
current_cost = 1   # ← nastaví sa aj pri inf-girth úspechu!
```

**Riadky 312 a 403 — `search_one_group` a ML-refinement:**
```python
if gens_t is not None and isinstance(girth_t, int) and girth_t >= g_target:
    ...   # ← inf-girth výsledok ticho zahodený

if gens_t is None or not isinstance(girth_t, int) or girth_t < g_target:
    continue   # ← inf-girth tu vyradený
```

## Dôsledok

Platné `(k,g)`-klietky s girthom > `2*g_target` sa nikdy nevykonajú ani nenahlása. Celé vetvy priestoru hľadania sú ignorované. Chyba tvorí sústredený reťazec: nájdená klietka sa zahodí vo `random_search`, `tabu_search` ju ignoruje ako neúspech, `search_one_group` ju zahodí, ML-refinement ju preskočí.

## Oprava

Vždy skontrolovať `float('inf')` pred porovnaním `isinstance(girth, int)`:

```python
girth_val = float('inf') if isinstance(girth, float) else girth

# random_search:
if girth_val <= best_girth:
    continue

# tabu_search:
if (girth_val == float('inf') or girth_val >= g_target) and nx.is_connected(cay) ...:
    return gens, girth

# search_one_group:
if gens_t is not None and (girth_t == float('inf') or (isinstance(girth_t, int) and girth_t >= g_target)):
    ...
```

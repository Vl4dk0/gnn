# rl-env-girth-check-uses-cycle-basis

**Súbor:** `ai/cage/rl/env.py:153-160`  
**Závažnosť:** HIGH

## Popis

Metóda `_is_girth_satisfied_active()` v `CageConstructionEnv` kontroluje, či aktívny subgraf spĺňa požiadavku na girth, pomocou `nx.cycle_basis()`:

```python
def _is_girth_satisfied_active(self) -> bool:
    sg = self._active_subgraph()
    if sg.number_of_edges() == 0:
        return True
    cycles = nx.cycle_basis(sg)
    if not cycles:
        return True
    return min(len(c) for c in cycles) >= self.g
```

## Chyba

`nx.cycle_basis(G)` vracia *bázu cyklov* (minimum spanning cycle basis) — nie zoznam **všetkých** cyklov a nie nevyhnutne najkratší cyklus grafu. Báza cyklov pokrýva nezávislú množinu cyklov, ale jej najkratší prvok môže byť **dlhší** ako skutočný girth grafu.

Konkrétny príklad: Graf s trojuholníkom (3-cyklus) a ďalšou štruktúrou môže mať v báze cykly dĺžky 4, 5, ak trojuholník vznikol kombináciou cyklov bázy. `nx.cycle_basis` nie je garantovaný vrátiť girth ako najkratší prvok.

Dôsledok: `_is_girth_satisfied_active()` môže vrátiť `True` (girth OK), keď v grafe existuje kratší cyklus ako `self.g`. To znamená:
- Odmena `SATISFY_BONUS` sa udelí nesprávne
- Podmienka úspechu (`is_k_regular and girth_ok`) sa vyhodnotí nesprávne
- Agent dostáva **falošnú pozitívnu odmenu** za nevalídne stavy

## Existujúca oprava

`backend/utils/graph_utils.py` obsahuje správnu `compute_girth()` funkciu (BFS-based). Tej by sa mal použiť:

```python
from backend.utils.graph_utils import compute_girth

def _is_girth_satisfied_active(self) -> bool:
    sg = self._active_subgraph()
    if sg.number_of_edges() == 0:
        return True
    g = compute_girth(sg)
    return isinstance(g, float) or g >= self.g
```

## Kontext

Funkcia `compute_girth` z `graph_utils.py` (riadky 136–198) implementuje BFS-based výpočet girthu, ktorý je presný. Používa ju aj `is_valid_cage()` — kanonická validácia. RL env by mala byť konzistentná s tou validáciou.


## Test

Proven by [`test_cycle_basis_disagrees_on_known_bad_graph`](../../tests/test_issues_ai.py) (test fails → bug confirmed).

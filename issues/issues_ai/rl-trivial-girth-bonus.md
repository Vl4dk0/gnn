# rl-trivial-girth-bonus

**Súbor:** `ai/cage/rl/env.py:153-160`  
**Závažnosť:** MEDIUM

## Popis

`_is_girth_satisfied_active()` vracia `True` pre prázdny graf alebo strom (žiadne cykly), čo triviálne udeľuje `SATISFY_BONUS` na začiatku každej epizódy.

```python
# Riadok 153-160:
def _is_girth_satisfied_active(self) -> bool:
    sg = self._active_subgraph()
    if sg.number_of_edges() == 0:
        return True   # ← prázdny graf → girth "ok"
    cycles = nx.cycle_basis(sg)
    if not cycles:
        return True   # ← strom → girth "ok"
    return min(len(c) for c in cycles) >= self.g
```

V kombinácii s odmenovaním:
```python
if valid_action and (girth_ok or is_k_regular):
    reward += SATISFY_BONUS
```

Agent dostáva bonus za každú hranu pridanú do prázdneho grafu na začiatku epizódy, kde `girth_ok = True` triviálne.

## Dôsledok

Zbytočná odmena v počiatočných stavoch môže zavádzať tréning — agent sa môže naučiť preferovať začiatočné kroky kvôli zaručenému bonusu namiesto zmysluplného budovania grafu.

## Oprava

```python
def _is_girth_satisfied_active(self) -> bool:
    sg = self._active_subgraph()
    if sg.number_of_edges() == 0:
        return False   # prázdny graf nemá žiaduce vlastnosti
    cycles = nx.cycle_basis(sg)
    if not cycles:
        return False   # strom tiež nespĺňa podmienku girthu
    return min(len(c) for c in cycles) >= self.g
```


## Test

Proven by [`test_rl_env_girth_unsatisfied_for_empty_graph`](../../tests/test_issues_ai.py) (test fails → bug confirmed).

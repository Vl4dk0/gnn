# mcts-terminal-ignores-girth

**Súbor:** `ai/cage/functions/monte_carlo_search_tree.py:60`  
**Závažnosť:** MEDIUM

## Popis

`MCTSNode.is_terminal` je nastavený na `True` ak je graf `k`-regulárny, bez ohľadu na to či má správny girth. K-regulárny graf so zlým girthom (napr. trojuholník pri hľadaní g=6) je označený ako terminálny, MCTS ho prestane expandovať a nikdy nenájde klietku cez tento uzol.

```python
# Riadok 60 — chybný:
self.is_terminal = is_k_regular(self.graph, self.k)

# Riadok 159 — dôsledok:
if not node.is_terminal and not node.is_fully_expanded():
    node = self.expand(node)  # Preskočené pre k-regulárne s zlým girthom!
```

## Dôsledok

MCTS predčasne ukončí expanziu pre grafy, ktoré sú k-regulárne ale nespĺňajú podmienku girthu. Mnohé vetvy priestoru stavov zostanú neprebádané, čo znižuje šancu nájsť klietku.

## Oprava

```python
from backend.utils.graph_utils import compute_girth

self.is_terminal = (
    is_k_regular(self.graph, self.k)
    and compute_girth(self.graph) == self.g
)
```


## Test

Proven by [`test_mcts_terminal_requires_correct_girth`](../../tests/test_issues_ai.py) (test fails → bug confirmed).

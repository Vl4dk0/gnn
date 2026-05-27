# mcts-zero-division-visits

**Súbor:** `ai/cage/functions/monte_carlo_search_tree.py:99-102`  
**Závažnosť:** HIGH

## Popis

`best_child()` nechráni pred `child.visits == 0` ani `self.visits == 0`. UCB1 výpočet:

```python
choices_weights = [
    (child.value / child.visits)                                        # ZeroDivisionError
    + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))  # math domain error
    for child in self.children
]
```

`math.log(0)` hodí `ValueError: math domain error`. `child.value / 0` hodí `ZeroDivisionError`. Ak root uzol nebol ešte navštívený (`self.visits == 0`), nastane chyba pri prvom volaní `best_child`.

## Dôsledok

`ValueError` alebo `ZeroDivisionError` pri volaní `best_child()` ak niektoré dieťa ešte nemá zaznamenanú návštevu — čo môže nastať pri implementačných zmenách alebo edge cases.

## Oprava

```python
def best_child(self, c_param: float = 1.414) -> "MCTSNode | None":
    if not self.children:
        return None
    choices_weights = [
        (child.value / max(1, child.visits))
        + c_param * math.sqrt(
            2 * math.log(max(1, self.visits)) / max(1, child.visits)
        )
        for child in self.children
    ]
    return self.children[choices_weights.index(max(choices_weights))]
```


## Test

Proven by [`test_mcts_best_child_safe_with_zero_visits`](../../tests/test_issues_ai.py) (test fails → bug confirmed).

# voltage-mislabel

**Súbor:** `ai/cage/voltage/data_gen.py:71-76`  
**Závažnosť:** MEDIUM

## Popis

Rovnaký labeling bug ako `cayley-mislabel.md`, tentoraz pre voltage girth prediktor:

```python
# Label — infinite girth means degenerate lift
if isinstance(girth, float):
    girth_int = 0
    girth_class = 0   # ← potenciálne nesprávne
```

Komentár tvrdí, že nekonečný girth znamená degenerovaný lift. To nie je vždy pravda: `compute_lift_girth` vracia `math.inf` keď girth > `max_girth = 2×g_target`, čo môže byť valídny lift s veľkým girthom (≥ g_target).

## Dôsledok

Praktický dopad je malý — výskyt takýchto vzoriek pri náhodnom vzorkovaní je veľmi nízky. Logika a komentár sú však nesprávne a môžu spôsobiť problémy pri cielenom generovaní dát.

## Oprava

```python
if isinstance(girth, float):
    girth_int = 2 * g_target   # dolný odhad
    girth_class = 1             # girth > 2×g_target ≥ g_target
```


## Test

Proven by [`test_base_graph_to_pyg_labels_inf_as_positive`](../../tests/test_issues_ai.py) (test fails → bug confirmed).

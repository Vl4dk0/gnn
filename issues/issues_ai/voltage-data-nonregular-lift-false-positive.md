# voltage-data-nonregular-lift-false-positive

**Súbor:** `ai/cage/voltage/data_gen.py:204-215`  
**Závažnosť:** MEDIUM

## Popis

Generátor dát nevynucuje, že náhodný voltage assignment produkuje k-regulárny lift. Ak dve hrany dumbbell(k) dostanú rovnaké napätie, alebo bouquet(k//2) dostane napätie 0 (identita), lift sa stane neregulárnym — ale sample dostane `girth_class=1` (falošný pozitív) ak je vypočítaný girth dostatočne veľký.

```python
# data_gen.py:204:
volt = [rng.randint(0, group.order - 1) for _ in range(n_edges)]
# Neskontroluje sa: má lift skutočne stupeň k?
girth = compute_lift_girth(base, group, volt, max_girth=2 * g_target)
data = base_graph_to_pyg(base, volt, group, k, g_target, girth)
# Mislabeled: girth_class=1 aj keď lift nie je k-regulárny
```

Konkrétny prípad: `dumbbell(3)`, `Z_5`, napätia `[2, 2, 1]` → lift je 2-regulárny (nie 3-regulárny), girth=10, `girth_class=1` — falošný pozitív.

## Dôsledok

Trénovacie dáta voltage girth prediktora obsahujú falošné pozitívy — neregulárne lifty označené ako klietky. Degraduje presnosť klasifikácie.

## Oprava

```python
from ai.cage.voltage.lift import build_lift
from backend.utils.graph_utils import is_k_regular

lift_graph = build_lift(base, group, volt)
if not is_k_regular(lift_graph, k):
    continue   # preskočiť neregulárny lift
```


## Test

Proven by [`test_voltage_dataset_does_not_label_nonregular_lift_as_positive`](../../tests/test_issues_ai.py) (test fails → bug confirmed).

# random-walk-sample-underflow

**Súbor:** `ai/cage/functions/random_walk.py:174-175`  
**Závažnosť:** HIGH

## Popis

`_add_edge_between_low_degree` volá `random.sample(low_degree_nodes, 2)` bez kontroly, či zoznam obsahuje aspoň 2 prvky. V bloku `else` (General case) má akcia `add_edge` pravdepodobnosť 0.25 bez ohľadu na veľkosť `low_degree_nodes`.

```python
# Riadok 174-175:
for _ in range(attempts):
    u, v = random.sample(low_degree_nodes, 2)  # ValueError ak len < 2!
```

Situácia nastane napríklad keď má low-degree iba jeden uzol — čo je možné v počiatočnom stave alebo pri grafoch s vysokým stupňom.

## Dôsledok

`ValueError: Sample larger than population or is negative` — crash pri generovaní.

## Oprava

```python
def _add_edge_between_low_degree(self, low_degree_nodes: list[int]) -> None:
    if len(low_degree_nodes) < 2:
        return
    # ... zvyšok funkcie
```

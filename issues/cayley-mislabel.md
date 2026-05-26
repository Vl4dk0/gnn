# cayley-mislabel

**Súbor:** `ai/cage/cayley/data_gen.py:88-93`  
**Závažnosť:** HIGH

## Popis

Keď `cayley_girth` vracia `float('inf')` (girth > 2×g_target + 2), vzorka sa chybne označí ako negatívna:

```python
if isinstance(girth, float):   # float('inf')
    girth_int = 0
    girth_class = 0   # ← má byť 1
```

Cayleho graf s girth > 2×g_target má určite girth ≥ g_target — je to pozitívna vzorka. Namiesto toho dostáva `girth_class = 0` a regresia cieľ `girth_int = 0`.

## Dôsledok

Trénovacie dáta pre Cayley girth prediktor obsahujú mislabeled pozitívne vzorky. Degraduje F1 skóre klasifikácie a kalibruje regresnú hlavu na nezmyselné nulové hodnoty.

## Oprava

```python
if isinstance(girth, float):
    girth_int = 2 * g_target + 2   # dolný odhad skutočného girthu
    girth_class = 1                 # girth > max_girth ≥ g_target
```

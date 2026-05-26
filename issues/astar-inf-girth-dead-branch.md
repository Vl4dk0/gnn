# astar-inf-girth-dead-branch

**Súbor:** `ai/cage/functions/astar.py:257-260`  
**Závažnosť:** HIGH

## Popis

Vetva `elif current_girth == float("inf")` je mŕtvy kód — nikdy sa nevykoná. Podmienka `current_girth > self.g` je pravda pre `float("inf")` (keďže `inf > akékoľvek číslo`), takže stav "graf bez cyklov" (žiadne hrany) dostane skóre `0.8` namiesto správneho `0.5`.

```python
# Súčasný kód — chybný:
if current_girth == self.g:
    girth_score = 1.0
elif current_girth > self.g:
    girth_score = 0.8   # ← float("inf") padá sem, nie pod elif nižšie
elif current_girth == float("inf"):
    girth_score = 0.5   # ← NIKDY sa nevykoná
else:
    girth_score = 0.0
```

## Dôsledok

Prázdny graf (žiadne hrany, girth = inf) dostane skóre `0.8` (dobré skóre), čo je vyššie ako zámer `0.5` (neutrálne). A* bude preferovať stavy s nedostatočnými hranami ako dobré počiatočné stavy, čo degraduje kvalitu heuristiky.

## Oprava

```python
if current_girth == float("inf"):
    girth_score = 0.5
elif current_girth == self.g:
    girth_score = 1.0
elif current_girth > self.g:
    girth_score = 0.8
else:
    girth_score = 0.0
```

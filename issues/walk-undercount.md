# walk-undercount

**Súbor:** `ai/cage/voltage/cycle_analysis.py:174-175`  
**Závažnosť:** MEDIUM

## Popis

`count_short_identity_walks` zastaví rozvíjanie chôdze po nájdení uzavretého okruhu:

```python
if node == start and net_v == identity and length >= 3:
    count += 1
    continue   # ← nerozvíja ďalej
```

Dlhšie uzavreté chôdze, ktoré sú dostupné len predĺžením tejto uzavretej chôdze, sa nespočítajú.

## Dôsledok

Funkcia správne vracia `0` práve vtedy, keď neexistujú krátke okruhy (každý premešknutý dlhší okruh má kratší prefix, ktorý je zarátaný). Teda logika nula/nenula je správna.

Avšak pre nenulové hodnoty je počet **podhodnotený** — tabu hľadanie horšie rozlišuje medzi "málo porušeniami" a "veľa porušeniami", čo degraduje kvalitu zoradenia krokov.

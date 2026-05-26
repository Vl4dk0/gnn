# wrong-bound

**Súbor:** `backend/utils/graph_utils.py:244-269`  
**Závažnosť:** LOW

## Popis

`moore_hoffman_upper_bound` používa nestandardný vzorec bez citácie:

```
N(k,g) ≤ 2(k-1)^(g-2)    if g is odd
N(k,g) ≤ 4(k-1)^(g-3)    if g is even
```

Toto nie je štandardný Moore-Hoffmanov horný odhad z literatúry. Správny odhad pre klietky je odlišný. Funkcia sa nazýva "Moore-Hoffman upper bound" ale vzorec nie je doložený odkazom.

## Dôsledok

Výsledok je potenciálne nesprávny a môže uvádzať do omylu pri použití v rozhraní (frontend zobrazuje `upper_bound` z tohto výpočtu).

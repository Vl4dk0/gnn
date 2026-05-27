# cayley-tabu-marks-wrong-element

**Súbor:** `ai/cage/cayley/search.py:241`  
**Závažnosť:** MEDIUM

## Popis

Tabu zoznam označí `s_old` (práve **odstránený** generátor) namiesto `s_new` (práve **pridaný** generátor). `s_old` sa v množine generátorov nenachádza počas celej tabu periódy, takže záznam `tabu[s_old]` nikdy nevyvolá blokovanie — anti-cyklický mechanizmus efektívne nefunguje.

```python
# Riadok 241:
new_cost, _kind, new_gens, s_old = chosen
tabu[s_old] = it + tabu_tenure   # ← označí odstránený prvok, nie pridaný!
```

Bezprostredné zvrátenie ťahu (odstrániť `s_new`, znova pridať `s_old`) nie je zakázané. Tabu search sa môže ľubovoľne cyklovať.

## Dôsledok

Tabu list neprevádza svoju základnú funkciu — zabraňovanie cyklom. Search môže oscilovať medzi dvoma stavmi bez konvergencie.

## Oprava

```python
prev_gens = set(gens)
gens = new_gens
added = set(gens) - prev_gens
if added:
    s_new = next(iter(added))
    tabu[s_new] = it + tabu_tenure   # zakáže návrat k s_new
```

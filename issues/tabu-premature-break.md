# tabu-premature-break

**Súbor:** `ai/cage/voltage/search.py:211, 236`  
**Závažnosť:** HIGH

## Popis

Tabu search inicializuje `best_move_cost = current_cost + 1` (horšie ako aktuálny stav) a vykoná `break` ak nenájde pohyb s nižšou hodnotou. To je fundamentálne nesprávne — správny tabu search **vždy** vyberie nejaký pohyb (aj zhoršujúci), aby unikol lokálnemu minimu. Toto je celý zmysel tabu zoznamu.

```python
# Riadok 211 — chybné:
best_move_cost = current_cost + 1  # worse than current

# Riadok 231:
if new_cost < best_cost or (not is_tabu and new_cost < best_move_cost):
    # ← nikdy nevyberie pohyby horšie ako current_cost

# Riadok 236:
if best_move_edge < 0:
    break  # ← predčasné ukončenie pri stagnácii
```

Aktuálna implementácia sa správa ako hill-climbing so zastavením pri stagnácii, nie ako skutočný tabu search.

## Dôsledok

Search sa zasekne pri lokálnej plošine a ukončí sa predčasne, hoci by mohol pokračovať tisíce iterácií ďalej a nájsť klietku. Klietky sa nenájdu v mnohých prípadoch kde by ich správny tabu search objavil.

## Oprava

```python
best_move_cost = math.inf   # akceptovať akýkoľvek pohyb, nielen zlepšujúce

# podmienka výberu pohybu ostáva — aspiration criterion
if new_cost < best_cost or (not is_tabu and new_cost < best_move_cost):
    ...

if best_move_edge < 0:
    break  # teraz len keď SÚ VŠETKY pohyby tabu a žiaden nesplní aspiráciu
```

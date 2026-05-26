# thesis-specialist-predictor-phantom-column

**Súbor:** `thesis/chapters/07-experiments-results.typ:19-44, 129, 145`  
**Závažnosť:** CRITICAL

## Popis

Thesis v kapitole 7 popisuje a prezentuje tri varianty voltage-girth prediktora podľa miery zdieľania parametrov:

- **A: cubic only** – trénovaný iba na kubických cieľoch $(3, g)$
- **B: fixed girth** – jeden model pre pevnú hodnotu girthú, zdieľaný cez viac stupňov
- **C: unified** – jeden model pre všetky $(k, g)$ dvojice

Tabula F1 skóre (riadky 33–41) porovnáva varianty A, B a C.  
Tabula úspešnosti generátorov (riadok 129) obsahuje stĺpec **`Volt.+S`** ("specialist") odlišný od **`Volt.+U`** ("unified").  
Popis v riadku 145 hovorí: *"Volt.+S uses the matching per-girth specialist predictor when available"*.

## Nesúlad s kódom

Kód obsahuje **iba jeden typ prediktora**: `girth_predictor_unified` v `ai/trained/voltage_girth/`. Pravidlá repozitára (AGENTS.md) explicitne zakazujú per-`(k,g)` ani per-`g` prediktory:

> "The voltage girth predictor is always a single (k,g)-independent model… Never re-create per-(k,g) or per-g predictors."

Špecialisti **(variant A, B, Volt.+S)** nikdy reálne neexistovali vo validovanej podobe – boli vytvorené v staršej iterácii, rozhodnuté nepoužívať, a z repozitára odobrané. Napriek tomu thesis stále reportuje ich výsledky (F1 tabuľka, stĺpec Volt.+S) akoby šlo o platný experiment.

## Dôsledok

Thesis prezentuje experimentálne výsledky pre modely, ktoré v repozitári neexistujú a *neboli nikdy určené na použitie*. Čitateľ nemôže replikovať experiment "Volt.+S". F1 tabuľka variantov A/B/C je zavádzajúca, pretože iba C (unified) zodpovedá kódu.

## Oprava

1. Odstrániť zo thesis tabuľku porovnania variantov A/B/C (alebo ju presunúť do appendixu s poznámkou, že ide o archívne experimentovanie).
2. Premenovať stĺpec `Volt.+S` na niečo, čo zodpovedá realite, alebo ho odstrániť.
3. Popis v caption riadok 145 opraviť tak, aby nespomínal neexistujúce specialist prediktory.

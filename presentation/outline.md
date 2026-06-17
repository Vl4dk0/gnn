# Osnova obhajobovej prezentácie (12 min)

**Práca:** *Machine Learning for Generation of Graph of Given Degree and Girth*
**Autor:** Vladimír Jančár — FMFI UK Bratislava, 2026 · Školiteľ: Mgr. Ján Pastorek

**Ústredná línia (red thread):** *„Učenie navádza, exaktné algoritmy rozhodujú."*
Stupeň grafu sa dá naučiť/vynútiť, girth sa musí hľadať exaktne — a každá metóda
je odpoveďou na to, kde nakresliť hranicu medzi učením a exaktným výpočtom.

> Cieľová stopáž: ~12 min hovorenia + rezerva na otázky. ~13 obsahových slidov
> (≈1/min) + titulka + obsah + záver + backup slidy na otázky (nepočítajú sa).

---

## Časový rozpočet po slidoch

| # | Slide | Čas | Kľúčové posolstvo | Vizuál |
|---|---|---|---|---|
| 1 | **Titulka** | — | Názov, autor, školiteľ, fakulta | logo UK |
| 2 | **Obsah / Outline** | 0:15 | 4 sekcie: Problém → Diagnostika → Metódy → Výsledky | — |
| 3 | **Problém: (k,g)-grafy a cages** | 1:00 | k-regulárny graf s girth ≥ g; najmenší = *cage*; klasický extremálny problém, rekordy stále zlepšované ručne + počítačom | obrázok Petersen/cage |
| 4 | **Prečo to ML nezvládne ako čierna skrinka** | 1:30 | MP-GNN ≤ 1-WL; **girth dokázateľne nevypočítateľný** GNN-kou [Garg]; RL funguje len keď priestor ťahov nesie štruktúru | fig01/fig02 message passing + receptive field |
| 5 | **Preformulovanie otázky** | 0:30 | Nie „vyrob cage", ale **naveď, kde má hľadanie hľadať** — exaktný test rozhodne o platnosti | jednoduchá schéma „propose → accept" |
| 6 | **Diagnostika: čo sa GNN dá naučiť** | 2:00 | **Stupeň vyriešený** (SAGE, 4 900 parametrov → accuracy 1.000); **min-cyklus/girth odolá všetkému** (najlepšie Loopy 0.510). → girth musí riešiť exaktné hľadanie | **fig10** + Tab. 1/2 |
| 7 | **Rebrík metód (prehľad)** | 0:30 | Rastúca štruktúra: Direct RL → Voltage lifts → Refinement & Excision → Forge | ikonky 4 krokov |
| 8 | **Direct RL** | 1:00 | Stav = čiastočný graf, akcia = pridaj/zruš hranu, PPO; potrebuje **curriculum** (štart (3,5)) + **shaped reward**, inak nevyrobí ani (3,5)-graf; kvadratický priestor → nešk áluje | — / malá schéma |
| 9 | **Voltage lifts** | 1:30 | k-regularita je **štrukturálna**; girth sa číta z malého base grafu cez *net voltage* (girth = min \|W\|·ord(s)); **gauge** vynuluje hrany kostry → hľadá sa len pár voltageov (K₄: 3 namiesto 6) | **fig03** + **fig04** |
| 10 | **Refinement & Excision** | 1:30 | 2-/3-swap opravia girth pri zachovaní stupňa; **excision** odoberie strom Moore-polomeru ⌊(g−1)/2⌋ a zošije deficientné vrcholy → dodekaéder → **Petersen = (3,5)-cage**, polovičná veľkosť; učená oprava + exaktný backtracking | **fig09** (+ fig08, fig05/07 backup) |
| 11 | **Forge** | 1:00 | Kompozícia: producer (reach) + refinement (oprava near-miss) + excision (veľkosť); odovzdanie cez **defective fraction ≤ τ**; round-robin, producer vymeniteľný | schéma pipeline |
| 12 | **Výsledky: pokrytie vs. veľkosť** | 1:30 | **22 cieľov**; **voltage-rl najširší dosah** (jediný rieši (3,9),(3,10),(4,8),(6,6); (6,5) na 57 %); voltage rodina **~2× Moore**; klasické size-optimal ale úzke; (4,7),(5,7),(7,5) nikto | Tab. 3/4 (výber) |
| 13 | **Forge: výsledok + ablácia** | 1:00 | Tabu producenti ~0.46 úspech, **~1.23× Moore** (a cage presne na najmenších); RL producer najsilnejší samostatne, ale najslabší vo Forge (chce diverzitu); ablácia: bez excision 1.51→2.06 | Tab. 5/6 |
| 14 | **Záver** | 1:00 | Stupeň vyriešený, girth nie; trade-off pokrytie↔veľkosť; **učené komponenty zriedka prekonajú neučené** — a to *je* zistenie; učenie navádza, exaktný test rozhodne | 3 bullet pointy |
| 15 | **Future work + Ďakujem** | 0:30 | Co-adaptácia stupňov Forge, učenie root/depth excision, girth-špecializovaný model; bežiaci rekordný search (8,5),(9,5),(10,5),(11,6) | focus slide |

**Súčet hovorenia:** ≈ 12:15 (s rezervou na prechody dolaďuj slidy 4, 6, 9, 10).

---

## Backup slidy (na otázky — nepočítajú sa do 12 min)

- **B1 — Gauge transformácia podrobne** (prečo možno vynulovať hrany kostry) — §6.3
- **B2 — Net-voltage girth formula + K₄/Z₃ príklad** (trojuholník → 9-cyklus) — fig04, §6.3
- **B3 — Curriculum & potential-based reward** (Alg. 2/3) — §5.3
- **B4 — Excision detail** (tree removal + stitching, fig08) — §7.2
- **B5 — Tabuľky 1–2 naplno** (architektúry: GCN/SAGE/GIN/GINE/GPS/Loopy)
- **B6 — Tabuľky 3–6 naplno** (čas, veľkosť, Forge, ablácia)
- **B7 — Eval setup** (PERUN, 60 s/pokus, 128 workerov) — §9 intro
- **B8 — Prečo Erdős–Rényi tréning property-predictorov** — §4.1

## Anticipované otázky komisie (kde je odpoveď)
1. Ak GNN nevie girth, načo GNN? → §3.2 + Ch 11 (GNN navádza, nerozhoduje)
2. Učené zriedka vyhrá — kde je prínos? → Ch 11 (učená oprava v excision; RL ako zdroj diverzných liftov)
3. Ako voltage lift garantuje k-regularitu / girth bez veľkého grafu? → §6.2–6.3
4. Čo je gauge a prečo nuluje kostru? → §6.3
5. Prečo poradie producer→refinement→excision a čo je τ? → §8.1
6. Prečo RL najlepší sám, ale najhorší vo Forge? → §9.3 + Ch 11
7. Čo znamená ~1.23×/~2× Moore a našli ste rekord? → §9.2–9.3, Ch 10 (zatiaľ nie)
8. Prečo curriculum a shaping pre direct RL? → §5.3–5.4

(Plné poznámky: `thesis_notes.md` · podklad: `thesis_md/thesis.md`)

# Spec obhajobovej prezentácie (~12 min, slovensky)

**Práca:** *Machine Learning for Generation of Graph of Given Degree and Girth*
**Autor:** Vladimír Jančár — FMFI UK Bratislava, 2026 · Mgr. Ján Pastorek
**Súbor:** `presentation/slides.typ` (Touying + metropolis, reskinnuté do štýlu webu)

Toto je záväzný spec. Každý subagent ho prečíta pred prácou. Fakty si overuj v
`thesis/chapters.typ` (text bakalárky). Figúry sú v `presentation/figures/`,
generátory v `presentation/scripts/`.

---

## ŠTÝL OBSAHU v2 — POVINNÉ (toto je najdôležitejšie)

Toto NIE je dokument, je to vizuálna opora pre rozprávanie. Tvrdé pravidlá:

- **MINIMUM TEXTU.** Žiadny „bullet-point monster". Na slide patrí pár slov, nie odseky.
- **ŽIADNE odrážkové zoznamy** ako hlavný obsah. Namiesto bulletov **súvislý text**
  (krátke vety/fragmenty), ktorý prichádza **postupne cez `#pause`** (1 myšlienka =
  1 reveal). Hovorené slovo nesie detail, slide ho len podčiarkuje.
- **NA KAŽDOM SLIDE VIZUALIZÁCIA.** Každý obsahový slide má dominantný obrázok, diagram,
  graf alebo schému toho, o čom sa hovorí. „VŽDY SA TO DÁ." Ak vizualizácia neexistuje,
  vygeneruj ju (matplotlib/networkx do `presentation/figures/`, generátor do
  `presentation/scripts/`, šedá paleta webu).
- Pomer: obrázok dominuje (~60–70 % plochy), text je minoritný sprievod.
- **ŽIADNE jednoslovné slidy.** Slide s jediným slovom (napr. section divider „Problém",
  „Metódy") je zbytočný → ODSTRÁNIŤ. Nepoužívaj `=` section dividery, ktoré generujú
  samostatnú takmer prázdnu stránku. Členenie nech plynie cez obsah, nie cez nadpisové
  medzistránky. (Výnimka: koncový `#focus-slide` „Ďakujem za pozornosť" ostáva.)
- **Na otázky školiteľa a oponenta ZATIAĽ NEODPOVEDÁME.** Otázkové slidy teraz vynechaj
  (alebo nechaj len holú otázku bez odpovede). Žiadne vymyslené odpovede.

## Vodiace princípy (Herout: „čo chcem počuť na obhajobe")

- Z každého slidu musí kričať **ČO SOM SPRAVIL** a **AKÝ JE PRÍNOS**. Slovesný rámec:
  *skúsil som, navrhol som, naprogramoval som, spustil som, odmeral som, upravil som,
  zistil som.*
- Obrázky/demá namiesto bulletov, kdekoľvek sa dá (vzorec sa ráta ako obrázok).
- Teória zrezaná na minimum; komisia odbor pozná.
- **Žiadny „Outline"/agenda slide.** Otvor obrázkom zo svojej práce.
- Záver = **presne 3 vety**, nič nové, na jednom slide.
- Zákon 2t: hovoríš 2× dlhšie, než plánuješ → tvrdo rež.
- Tón: pokojná istota, nie teatrálnosť.
- **Slovo „backup" sa nikde nepoužíva.** Doplnkové slidy = sekcia **Dodatok**.

## Vizualizácia na každom slide (plán — dogeneruj chýbajúce)

| Slide | Vizualizácia | Stav |
|---|---|---|
| GNN (4 revealov) | reveal 1: `gnns_aigen.png` (prehľad); revealov 2–4: `fig-gnn-step1/2/3.pdf` (okolie → správy → update) | existuje |
| Definície (stupeň + obvod) | `fig-degree.pdf` + `fig-girth.pdf`, vertikálna deliaca čiara 1.8 pt, oba panely statické (bez animácie) | existuje |
| (k,g)-grafy a klietky | `fig-cage.png` (Petersen) | existuje |
| Stupeň vs. obvod | `fig10-degree-vs-girth.png` | existuje |
| Architektúry | NOVÁ: porovnanie architektúr (stupeň vs. obvod presnosť), bar/heatmap z Tab. 1–2 | **dogenerovať** |
| Loopy GNN | NOVÁ: malá schéma, ako Loopy „vidí" cyklus (cesta uzavretá cez vrchol) | **dogenerovať** |
| Priamy RL | NOVÁ: schéma stav → akcia (pridaj/zruš hranu) → graf, prípadne sekvencia stavby | **dogenerovať** |
| Voltage lifts | `fig03-voltage-lift-k4z3.png` + live demo | existuje |
| Refinement | `fig05-2swap.png` / `fig07-3swap.png` + live demo | existuje |
| Excision | `fig09-dodeca-petersen.png` + live demo | existuje |
| Forge | NOVÁ: pipeline diagram producer → refinement → excision s mini-grafmi | **dogenerovať** |
| Výsledky | NOVÁ: vizualizácia pokrytie vs. veľkosť, ideálne (k,g) mriežka/heatmap kto rieši čo + pomer k Moore | **dogenerovať** |
| Záver (3 vety) | 3 riadky, každá veta má vlastný obrázok: (1) `fig-architectures`, (2) `fig-forge` (pipeline), (3) `fig-results` | existuje (reuse) |

Signpost slide „Ako sa hľadajú dnes" zruš, ak by mal byť textový/jednoslovný.

---

## Vizuálna identita (plný reskin do štýlu vladimirjancar.sk, light mode)

Achromatická šedá paleta, žiadny odtieň. Tokeny z `frontend/src/index.css`:

| Účel | Hodnota |
|---|---|
| pozadie (page) | `#f0f0f0` |
| pozadie (slide/panel) | `#ffffff` |
| primárny text | `#1a1a1a` |
| tlmený text | `#555555` |
| dim text | `#888888` |
| akcent (nadpisy/zvýraznenie) | `#5a5a5a` |
| akcent hover/tmavší | `#4a4a4a` |
| čiary/okraje | `#d0d0d0` |

- **Nadpisy:** font **Fraunces** (serif, editorial). Ak nie je v `typst fonts`,
  stiahnuť TTF do `presentation/fonts/` a kompilovať s `--font-path presentation/fonts`.
- **Text:** systémový sans (napr. "Helvetica Neue"/"Arial"/sans-serif fallback).
- **Akcenty/kód:** monospace.
- Header sekcií NIE tmavomodrý — prerobiť do šedej (header pruh `#1a1a1a` alebo biely
  s tmavým textom + Fraunces; zladiť s webom). Section divider má len nadpis + statickú
  čiaru `#d0d0d0` (žiadny progress bar v strede — to už je vyriešené cez
  `section-slide-no-bar`). Dolný progress bar nech je šedý (`#5a5a5a` na `#d0d0d0`).
- **Pozadie GRAPH/NEURL/NTWRK** na každom slide: dlaždicový vzor slov `GRAPH`, `NEURL`,
  `NTWRK` (vynechané samohlásky), monospace ~11pt, farba `#1a1a1a` pri ~5 % opacity,
  stĺpcový krok ~50pt, riadkový ~20pt, diagonála cez index `(row+col) % 3`. Veľmi
  jemné, nesmie rušiť čitateľnosť textu.

---

## Štruktúra slidov

### Titulka
- Minimalistická (FMFI to vyžaduje). Názov, meno, pod menom menej výrazne školiteľ,
  fakulta, dátum. Žiadne „Bachelor's Thesis Defense". Hneď prejsť ďalej.

### Otvorenie — strojové učenie najprv (názov práce začína ML)
1. **Čo sú GNN — 4 revealov na jednom slide** — reveal 1: prehľadový obrázok GNN
   (`gnns_aigen.png`, šírka ~88 %, celý slide). Revealov 2–4: 3-krokový message passing
   v grid rozložení: (2) vrchol a jeho **susedia** (`fig-gnn-step1`), (3) susedia
   **posielajú správy** (`fig-gnn-step2`), (4) vrchol **aktualizuje stav** (`fig-gnn-step3`)
   + záver „Lokálne a permutačne invariantné." Žiadny samostatný hero slide.
2. **Definície** — dva statické panely (žiadna animácia), vertikálna deliaca čiara (1.8 pt,
   `#bbbbbb`). Ľavý: `fig-degree.pdf`, „Graf je *k-regulárny*, ak každý jeho vrchol má
   rovnaký počet susedov." Pravý: `fig-girth.pdf`, „Obvod grafu g = najkratší cyklus v
   grafe." Oba panely viditeľné súčasne, bez `#pause`.
3. **(k,g)-grafy a klietky** — obrázok (3,5)-grafu = Petersen. k-regulárny + obvod ≥ g;
   najmenší taký = **klietka**; klasický extremálny problém, rekordy sa stále zlepšujú.

### Jadro — „čo som spravil"
5. **Rozdelil som problém: stupeň vs. obvod** — stupeň vyriešený (SAGE, **4 900 param →
   accuracy 1.000**); **obvod odoláva každej architektúre** (najlepšie Loopy ~0.510), lebo
   GNN nevie spoľahlivo rozpoznávať cykly → toto je tá ťažká časť (prvé zistenie).
   `fig10-degree-vs-girth.png`.
6. **Skúšal som rôzne architektúry + parametre** — GCN, GraphSAGE, GIN, GINE, GPS
   (graf transformer), Loopy; parametre (skryté dim, počet vrstiev, agregácia). Čo je na
   čo najlepšie a prečo (sum agregácia drží stupeň; normalizovaná ho skrýva).
7. **Loopy GNN** — cycle-aware architektúra, najlepšia na obvod; prečo vidí cykly.
8. **Môj prvý nápad: nech graf postaví GNN samo** — stav = čiastočný graf, akcie =
   pridaj/zruš hranu, tréning **RL (PPO)** — netreba dataset. Potrebuje curriculum
   (štart (3,5)) + shaped reward; akčný priestor rastie ~C(n,2). **Nefungovalo dobre →
   preto som sa pozrel, ako sa (k,g)-grafy hľadajú dnes.**
9. **Ako sa (k,g)-grafy hľadajú dnes** — signpost: voltage lifts → refinement → excision.
10. **Voltage lifts** — k-regularita je štrukturálna (každý lift k-regulárneho base je
    k-regulárny); girth sa číta z malého base grafu. 🔗 **live demo: vladimirjancar.sk/lift**.
    `fig03-voltage-lift-k4z3.png`.
11. **Refinement** — vezmi skoro-dobrý graf a uprav ho degree-preserving 2-/3-swapmi
    (tabu search). 🔗 **live demo: vladimirjancar.sk/refine**. `fig05-2swap.png`/`fig07-3swap.png`.
12. **Excision** — zmenši priveľký graf: odober strom Moore-polomeru ⌊(g−1)/2⌋ a zošij
    deficientné vrcholy. Dodekaéder → **Petersen = (3,5)-klietka**, polovičná veľkosť.
    🔗 **live demo: vladimirjancar.sk/excise**. `fig09-dodeca-petersen.png`.
13. **Forge** — poskladal som producer (voltage) + refinement (oprava near-miss) +
    excision (veľkosť) do pipeline; odovzdanie cez defective fraction ≤ τ, producer
    vymeniteľný.
14. **Výsledky** — naprieč 22 cieľmi: **A\* dáva najmenšie** grafy kde funguje;
    **voltage-RL má najširší dosah** (jediný rieši (3,9),(3,10),(4,8),(6,6); (6,5) na 57 %);
    **Forge najvyváženejší (~0.46 úspech, ~1.23× Moore)**; (4,7),(5,7),(7,5) nikto do 60 s.
    Čestne: učené komponenty zriedka prekonajú neučené — a to *je* zistenie.
15. **Záver — 3 vety** (pripravené dopredu) + prínos. **3 riadky, každá veta má vlastný
    obrázok** (malý obrázok + veta), odhaľované postupne cez `#pause`:
    (1) Rozdelil som problém na stupeň a obvod; stupeň sa GNN naučí presne, obvod je tá
    ťažká, exaktná časť → `fig-architectures.pdf`. (2) Postavil som a porovnal viacero
    konštrukčných metód (RL, voltage lifts, refinement, excision) a zložil ich do pipeline
    Forge → `fig-forge.pdf` (pipeline diagram). (3) Vznikol trade-off pokrytie↔veľkosť,
    Forge je najvyváženejší; učenie má miesto popri exaktnom hľadaní, nie namiesto neho →
    `fig-results.pdf`.
16. **Ďakujem** — focus slide.

### Otázky komisie (na konci, 1 slide na otázku)
17. **Školiteľ Q1** — prečo A\* (najmenšie) a voltage-RL (najširšie) prekonali Forge?
    (Forge optimalizuje vyváženosť, nie jeden extrém; producer kvalita vs. diverzita.)
18. **Školiteľ Q2** — prečo 60 s budget; ako vieme, že ML nepotrebovalo viac času na
    konvergenciu? (Tréning vs. inferencia/hľadanie; budget je férový a fixný pre všetky.)
19. **Oponent Q1** — aké pozičné/štruktúrne príznaky pridať, aby model lepšie zachytil
    globálnu cyklickú štruktúru (girth)? (Napr. pozičné enkódovania, počty cyklov, vzdialenosti.)
20. **Oponent Q2** — ako systematicky ladiť defective fraction τ a ako experimentálne
    oddeliť prínos diverzity kandidátov od kvality naučenej politiky? (Ablácie, fixovanie
    producenta.)

### Dodatok (referenčné slidy pre hlbšie otázky — NIE „backup")
- Net-voltage girth vzorec + K₄/Z₃ príklad (trojuholník → 9-cyklus). `fig04-voltage-girth.png`.
- Gauge transformácia (prečo možno vynulovať hrany kostry).
- Excision detail (tree removal + stitching). `fig08-excision-tree.png`.
- Eval setup: PERUN (TUKE), 60 s/pokus, 128 paralelných workerov; property-predictory
  trénované na Erdős–Rényi grafoch (čerstvé per krok → žiadna memorizácia).
- Plné tabuľky 1–6 (architektúry, čas, veľkosť, Forge, ablácie).

---

## Manifest figúr

**Znovupoužiť (existujú):** fig01-message-passing, fig03-voltage-lift-k4z3,
fig04-voltage-girth, fig05-2swap, fig07-3swap, fig08-excision-tree, fig09-dodeca-petersen,
fig10-degree-vs-girth.

**Vygenerovať nové** (do `presentation/figures/`, generátor v `presentation/scripts/`,
štýl ako `thesis/scripts`, šedá paleta webu, čisté pozadie):
- `fig-degree.png` — 3-regulárny graf, zvýraznený jeden vrchol a jeho 3 hrany.
- `fig-girth.png` — graf s jasne viditeľným najkratším cyklom dĺžky 4 alebo 5.
- `fig-cage.png` — (3,5)-graf = Petersenov graf, pekné symetrické rozloženie.
- `fig-gnn-step1/2/3.pdf` — 3-krokový message passing (okolie → agregácia správ →
  update stavu), šedá paleta, koherentná sekvencia (`gen_gnn_figures.py`). Nahrádzajú
  jeden zlúčený `fig01-message-passing` na GNN slide.
- `fig-forge.pdf` — kompaktný thumbnail Forge pipeline (producer → refinement →
  excision), maskulínna paleta diagramov, pre Záver (`gen_method_figures.py`).

**Heatmap `fig-results.pdf`** (`gen_method_figures.py`): stĺpce premenované
`dRL → directRL`, `RW → RandomWalk`, `BF → Bruteforce` (ostatné A\*, voltage,
voltage-rl, Forge nezmenené); rotácia popisov 55° aby dlhšie názvy nekolidovali.

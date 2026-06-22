# Presentation — Bachelor's Thesis Defense

A 12-minute defense deck for *Machine Learning for Generation of Graph of Given
Degree and Girth* (Vladimír Jančár, FMFI UK Bratislava, 2026), built in
**Typst** with the **Touying** framework (`metropolis` theme).

## Files

| File | What it is |
|---|---|
| `slides.typ` | **The presentation source** (Touying + metropolis). Edit this. |
| `slides.pdf` | Compiled deck (committed for convenience). |
| `outline.md` | The **outline / osnova**: slide-by-slide plan with a per-slide time budget and backup slides. |
| `figures/` | Hero figures extracted from the thesis, used by the slides. |
| `thesis_notes.md` | Chapter-by-chapter analysis: anchor points, navigation map, slide-candidate figures, likely defense questions. |
| `typst_guide.md` | How to author great Typst presentations (Touying vs Polylux, full examples, gotchas). |
| `thesis_md/` | The whole thesis converted to Markdown (+ extracted images) — the source material. |

## Build

Needs the [Typst](https://github.com/typst/typst) CLI (≥ 0.13).

```bash
typst compile --font-path fonts slides.typ slides.pdf      # one-shot
typst watch  --font-path fonts slides.typ slides.pdf       # live preview while editing
```

`--font-path fonts` is required — the deck uses **Fraunces** (static TTFs in `fonts/`
downloaded from the [official Fraunces release](https://github.com/undercasetype/Fraunces/releases)).
Without it Typst cannot find Fraunces and falls back to a default serif.

On first compile, Typst downloads the `touying` package from the `@preview`
registry (needs internet once). The deck uses the `metropolis` theme reskinned to
the achromatic greyscale palette of vladimirjancar.sk (light mode).

### Incremental reveals & handout

Slides use `#pause` for incremental reveals, so the PDF has more pages than
logical slides (each pause adds a sub-page). A `handout` toggle is wired in:

```bash
typst compile --input handout=true slides.typ handout.pdf
```

(Collapsing sub-slides requires matching `touying`/`typst` versions; on a version
mismatch the toggle is simply ignored and you get the full build.)

## Speaker notes & timing

Every content slide carries a `#speaker-note[...]` with the target time stamp
(e.g. `~1:30`). Total ≈ 12 minutes. See `outline.md` for the full time budget
and the backup-slide / Q&A plan.

## How `thesis_md/` was produced

`thesis/main.pdf` was converted to Markdown with `pymupdf4llm` (figures extracted
to `thesis_md/images/`), then read chapter-by-chapter to produce `thesis_notes.md`.

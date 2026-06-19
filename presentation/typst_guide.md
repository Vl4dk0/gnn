# Building an Academic Defense Slide Deck in Typst

A practical, code-heavy guide for authoring a **12-minute bachelor's thesis defense**
(math / CS topic: graph theory + machine learning) with **Touying**, the leading Typst
presentation framework. Verified against current package versions (June 2026).

---

## 1. Recommendation: Use **Touying**

**Use Touying with the `metropolis` theme.** Clear and opinionated.

| Criterion | Touying | Polylux | Built-in Typst |
| --- | --- | --- | --- |
| Maintenance / momentum (2025/26) | Actively developed, de-facto standard | Older, less active | None — Typst has no slide primitives |
| Incremental reveals (`#pause`) | Fast (no `counter`/`locate` penalty) | Slower (counter-based) | N/A |
| Built-in professional themes | Simple, Metropolis, University, Dewdrop, Aqua, Stargazer | Few | N/A |
| Math-animation, CeTZ/Fletcher diagram support | First-class | Limited | N/A |
| Speaker notes (pdfpc / second screen) | Yes (`#speaker-note`) | Yes | No |
| Academic third-party themes | Many (e.g. `touying-unistra-pristine`) | Few | N/A |

**Why Touying for *this* deck:** you need clean math typesetting, a couple of diagrams
(graphs / GNN architecture), figures with captions, incremental reveals to pace a live
talk, and a restrained academic look. Touying's `metropolis` theme delivers all of this
with minimal code, compiles in well under a second, and Typst's native math syntax is far
nicer to write than LaTeX/Beamer. Touying actually inherited part of its API from Polylux,
so you lose nothing by skipping Polylux. There is **no built-in Typst slide system** — a
package is mandatory.

> If your university has an official Touying theme (search Typst Universe for your
> institution), prefer it for branding. Otherwise `metropolis` is the safe, professional
> default. A polished academic third-party option is `touying-unistra-pristine`.

---

## 2. Setup

### Install the Typst CLI

```bash
# macOS
brew install typst

# Arch
pacman -S typst

# Windows
winget install --id Typst.Typst

# Any platform: download a release binary
# https://github.com/typst/typst/releases
# or via cargo:
cargo install --locked typst-cli
```

Verify:

```bash
typst --version   # expect 0.13.x or newer
```

### Compile to PDF

```bash
# one-shot compile
typst compile slides.typ slides.pdf

# live preview: recompiles on every save (great while authoring)
typst watch slides.typ slides.pdf
```

Packages from the `@preview` registry are downloaded and cached automatically on first
compile — **no manual install needed**, just an internet connection the first time.

### Version-pinned imports (copy these exact lines)

```typ
#import "@preview/touying:0.7.4": *      // slide framework (latest)
#import themes.metropolis: *             // clean academic theme

// Optional, only if you draw diagrams:
#import "@preview/cetz:0.5.2"            // low-level drawing canvas
#import "@preview/fletcher:0.5.9": diagram, node, edge   // graph/arrow diagrams
```

**Always pin versions** (the `:0.7.4` part). Unpinned imports can silently break when a
package publishes a new release. Touying 0.7.x has no breaking import changes; if you see a
guide using `0.5.x` or `0.6.x`, the import line is the same but APIs differ slightly — stay
on `0.7.4`.

---

## 3. Complete minimal working example

Save as `slides.typ`, then `typst compile slides.typ slides.pdf`. This produces a title
slide, a section divider, a bullet slide, a math slide, a figure slide, and a two-column
slide.

```typ
#import "@preview/touying:0.7.4": *
#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  // Footer left shows the institution on every content slide:
  footer: self => self.info.institution,
  config-info(
    title: [Graph Neural Networks for Combinatorial Optimization],
    subtitle: [Learning to Solve Graph Problems],
    author: [Jane Doe],
    date: datetime.today(),
    institution: [University of Example · Dept. of Computer Science],
    // logo: image("logo.png", height: 1.5cm),   // uncomment with a real path
  ),
)

// Optional: nicer fonts (only if installed on the machine).
#set text(font: "Fira Sans", size: 22pt)
#show math.equation: set text(font: "Fira Math")

// ---------- TITLE SLIDE ----------
#title-slide()

// ---------- OUTLINE ----------
#slide(title: [Outline])[
  #components.adaptive-columns(outline(title: none, indent: 1em))
]

// ---------- SECTION DIVIDER ----------
= Background

// ---------- BULLET SLIDE ----------
== Problem Setting

We study supervised learning on graphs $G = (V, E)$.

- Nodes carry feature vectors $bold(x)_v in RR^d$.
- Edges encode pairwise structure.
- Goal: predict a graph-level label $y in {0, 1}$.

#pause

Classical heuristics ignore learned structure — we want a *trainable* model.

// ---------- MATH SLIDE ----------
== Message Passing

A graph neural network updates node embeddings layer by layer:

$ bold(h)_v^((k)) = phi.alt(
  bold(h)_v^((k-1)),
  #h(0.2em) plus.circle.big_(u in cal(N)(v)) psi(bold(h)_v^((k-1)), bold(h)_u^((k-1)))
) $

where $cal(N)(v)$ is the neighbourhood of $v$, $plus.circle.big$ is a permutation-invariant
aggregator, and $phi.alt, psi$ are learned MLPs.

// ---------- FIGURE SLIDE ----------
= Method

== Architecture

#figure(
  // Replace the rect with: image("figures/architecture.png", width: 70%)
  rect(width: 70%, height: 4cm, stroke: 0.5pt, inset: 1em)[
    #align(center + horizon)[GNN architecture diagram goes here]
  ],
  caption: [Three message-passing layers followed by mean pooling and a linear head.],
)

// ---------- TWO-COLUMN SLIDE ----------
== Results

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    *Setup*
    - 3 GNN layers, hidden dim 64
    - Adam, lr $= 10^(-3)$
    - 5-fold cross-validation
  ],
  [
    *Outcome*
    #figure(
      rect(width: 100%, height: 3cm, stroke: 0.5pt)[
        #align(center + horizon)[accuracy plot]
      ],
      caption: [Test accuracy vs. baseline.],
    )
  ],
)

// ---------- CLOSING ----------
#focus-slide[
  Thank you — questions welcome.
]
```

---

## 4. Key features for academic talks

### Slide structure (heading-driven)

Touying maps Typst headings to slide structure:

- `= Section Title` → creates a **section divider slide** (and an outline entry).
- `== Slide Title` → starts a **new content slide** with that title in the header.
- Use the explicit `#slide(title: [...])[ ... ]` form when you need a slide with no heading
  (e.g. the outline) or custom options like `repeat:` for animations.

### Incremental reveals

```typ
== Pacing a live talk

First point appears immediately.
#pause
Second point appears on the next click.
#pause
Third point appears after that.
```

- `#pause` — reveal everything after it on the next subslide; content before stays.
- `#meanwhile` — reset so the next block animates *in parallel* with the previous one.
- `#uncover("2-")[...]` — show on subslides 2 onward, **reserving space** (no reflow).
- `#only("2-")[...]` — show only on subslide 2+, **without** reserving space.
- `#alternatives[a][b][c]` — swap content across consecutive subslides.

Range syntax: `"2-"` (from 2), `"-3"` (up to 3), `"2,4"` (only those), `"2-4"` (range).

Callback style (when many elements share animation state):

```typ
#slide(repeat: 3, self => [
  #let (uncover, only) = (self.methods.uncover, self.methods.only)
  Always visible.
  #uncover(self)("2-")[Appears on click 2.]
  #only(self)("3")[Only on click 3.]
])
```

> Keep animations sparing in a defense — a couple of `#pause`es per slide to control
> attention, not flashy choreography.

### Sections and outline

- Every `= Section` adds an entry to the table of contents.
- Render the outline with Typst's native `outline()` inside a slide (see the example).
- Metropolis shows a thin **progress bar** in the footer automatically
  (`footer-progress: true` by default) so the audience sees how far along you are.

### Math (Typst math syntax — no `$$`/`\`)

Inline: `$bold(x)_v in RR^d$`. Display: wrap with spaces inside `$ ... $`:

```typ
$ cal(L) = -1/N sum_(i=1)^N y_i log hat(y)_i $
```

Cheatsheet for thesis math:

| You want | Typst |
| --- | --- |
| Blackboard bold $\mathbb{R}$ | `RR`, `NN`, `ZZ`, `QQ`, `CC` |
| Bold vector | `bold(x)` |
| Subscript / superscript | `x_v`, `x^2`, `x_v^((k))` |
| Greek | `alpha`, `phi.alt`, `Sigma` |
| Sum / big operator | `sum_(i=1)^N`, `product`, `union.big` |
| Neighbourhood set | `cal(N)(v)` |
| Fraction | `1/N` or `frac(a, b)` |
| Norm | `norm(x)`, `abs(x)` |
| Hat / bar | `hat(y)`, `macron(x)` |
| Aligned multi-line | `$ a &= b \ &= c $` |

Numbered equations: `#set math.equation(numbering: "(1)")` near the top, then reference with
a `<label>` and `@label`.

### Figures with captions

```typ
#figure(
  image("figures/graph.png", width: 60%),
  caption: [A 5-node example graph and its adjacency matrix.],
)
```

Center is the default in slides. Control size with `width: 60%` / `height: 4cm`. To
reference a figure, attach `<fig-graph>` after it and cite `@fig-graph`.

### Two-column layouts

The robust, alignment-friendly choice is `#grid` (preferred over `#columns` for slides):

```typ
#grid(
  columns: (1fr, 1fr),   // or (2fr, 1fr) for asymmetric
  gutter: 1.5em,
  [ left content ],
  [ right content ],
)
```

Touying also offers `#components.side-by-side[a][b][c]` as a quick equal-split helper.

### Footer with author / page number

Metropolis already puts a **slide counter** at the footer-right and a progress bar across
the bottom. Set the footer-left text via the theme's `footer:` parameter:

```typ
#show: metropolis-theme.with(
  footer: self => self.info.institution,           // or: self => self.info.author
  // footer: self => [#self.info.author · #self.info.title],
  ...
)
```

### Speaker notes

```typ
== A slide
Visible content.
#speaker-note[Remember to mention the ablation study and the runtime numbers.]
```

Notes are hidden in the normal PDF. To use them live:

```typ
// Show notes on a second screen (e.g. with pympress / pdfpc):
#show: metropolis-theme.with(
  ..config-common(show-notes-on-second-screen: right),
  ...
)
```

Then present with a tool like **pympress** or **pdfpc** that supports dual-screen notes.
(In Touying 0.7.3+, `#speaker-note[]` always attaches to the slide *above* it.)

---

## 5. Theming

### Recommended look: `metropolis` (restrained, academic)

The default is a dark teal + orange accent. For a defense, consider **muting the accent** to
a calm academic blue and keeping a light background:

```typ
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-colors(
    primary: rgb("#1f4e79"),        // deep university blue (accent / headers)
    primary-light: rgb("#cfd8e3"),
    secondary: rgb("#23373b"),      // dark slate (text)
    neutral-lightest: rgb("#ffffff"),
    neutral-dark: rgb("#23373b"),
    neutral-darkest: rgb("#0f1c1f"),
  ),
  config-info( ... ),
)
```

Two professional, non-flashy palettes:

- **Academic blue:** primary `#1f4e79`, text `#23373b`, background `#ffffff`.
- **Forest / slate:** primary `#2e5d4b`, text `#1d2b2b`, background `#fafafa`.

Other clean themes worth a look if metropolis isn't to taste: **`themes.simple`**
(minimalist), **`themes.university`** (built-in, branded headers/footers), or the academic
universe theme **`touying-unistra-pristine`**.

### University branding on the title slide

```typ
#show: metropolis-theme.with(
  config-info(
    title: [Thesis Title],
    subtitle: [Bachelor's Thesis Defense],
    author: [Jane Doe — Supervisor: Prof. X. Y.],
    date: datetime.today().display("[month repr:long] [day], [year]"),
    institution: [University of Example · Faculty of Science],
    logo: image("figures/university-logo.svg", height: 1.6cm),
  ),
)
#title-slide()
```

`title-slide()` renders title, subtitle, author, date, institution and the logo
automatically. Put the logo file in a `figures/` folder beside `slides.typ`.

---

## 6. Best-practice tips for a 12-minute defense

- **Slide count: ~12–15 content slides** (roughly 1 per minute, plus title + outline +
  thank-you). Budget extra *backup* slides after the thank-you slide for likely questions
  (ablations, proofs, hyperparameters) — they don't count toward the 12 minutes.
- **One idea per slide.** A slide should make a single point you can state in a sentence.
- **≤ 6 bullet lines, ≤ ~10 words each.** The slide is scaffolding for *you*, not a
  document. Never paste paragraphs.
- **Figures over text.** For a graph-theory/ML talk, a drawn example graph, a GNN
  architecture diagram, and a results plot communicate far more than prose. Aim for a
  visual on most slides.
- **Show one key equation, not a derivation.** Put the message-passing update or loss on
  screen; keep full derivations in backup slides.
- **Rough structure for 12 min:** Motivation (1–2) → Problem statement (1) → Background /
  related work (1–2) → Method (3–4) → Experiments & results (2–3) → Conclusion + future
  work (1) → Thank you (1).
- **Use `#pause` to control attention**, e.g. reveal a result *after* you've set up the
  question — not to add visual noise.
- **Readable from the back:** body text ≥ 20pt (the example uses 22pt). Test the PDF
  fullscreen and step back from the monitor.
- **End on a strong "Conclusion," not "Thank You."** A final slide that restates your
  contribution in 3 bullets is what the committee stares at during questions.

---

## 7. Pitfalls / gotchas

- **Image paths are relative to the `.typ` file**, not your shell's working directory.
  Keep assets in a `figures/` subfolder and reference `image("figures/x.png")`. A wrong
  path gives `file not found`.
- **Pin every package version.** Unpinned `@preview` imports can break across releases.
  Mixing a `0.5.x`-era tutorial with `touying:0.7.4` causes confusing "unknown
  function/variable" errors — match the docs to your pinned version.
- **Fonts must be installed locally.** `#set text(font: "Fira Sans")` silently falls back if
  the font is missing. Run `typst fonts` to list available fonts; embed custom fonts with
  `typst compile --font-path ./fonts slides.typ`. For portability, stick to common fonts
  (e.g. `Libertinus Serif`, `New Computer Modern`) or ship the font files.
- **Fira Math** (used for `#show math.equation`) is a separate font — omit that line if it's
  not installed, or math will fall back.
- **`#pause` only works inside a slide context.** Using it at the top level (outside a
  `==` heading slide or `#slide[...]`) does nothing or errors.
- **Headings *are* slides.** Don't use `=`/`==` for in-slide subheadings — they'll spawn new
  slides. Use `*bold*` or `#text(weight: "bold")[...]` for emphasis within a slide.
- **`datetime.today()` needs no network**, but `.display(...)` requires a valid format
  string; a malformed format string is a compile error.
- **SVG logos** render crisply and scale better than PNG for university crests — prefer SVG
  when available.
- **First compile is slower** (it downloads packages and builds the font cache). Subsequent
  compiles and `typst watch` are near-instant.
- **CeTZ/Fletcher version coupling:** Fletcher depends on a specific CeTZ version. If you
  import both, let Fletcher pull its own CeTZ rather than pinning a conflicting one, or you
  may get version-mismatch errors.

---

## Quick reference card

```typ
#import "@preview/touying:0.7.4": *
#import themes.metropolis: *
#show: metropolis-theme.with(aspect-ratio: "16-9", config-info( title: [...], author: [...] ))
#title-slide()
= Section          // section divider + outline entry
== Slide title     // content slide
#pause             // incremental reveal
$ ... $            // display math
#figure(image("figures/x.png", width: 60%), caption: [...])
#grid(columns: (1fr,1fr), gutter: 1em, [left], [right])
#speaker-note[...] // hidden note
#focus-slide[Thank you]
```

```bash
typst watch slides.typ slides.pdf     # author with live preview
typst compile slides.typ slides.pdf   # final build
```

---

## Sources

- Touying — Typst Universe: https://typst.app/universe/package/touying/
- Touying GitHub (README, latest 0.7.4): https://github.com/touying-typ/touying
- Touying changelog: https://github.com/touying-typ/touying/blob/main/changelog.md
- Touying — Getting Started: https://touying-typ.github.io/docs/start
- Touying — Introduction / features: https://touying-typ.github.io/docs/intro
- Touying — Metropolis theme: https://touying-typ.github.io/docs/themes/metropolis
- Touying theme source (metropolis.typ): https://github.com/touying-typ/touying/blob/main/themes/metropolis.typ
- Touying vs Polylux (Typst Forum): https://forum.typst.app/t/touying-vs-polylux/1703
- Academic slides example (Typst Forum): https://forum.typst.app/t/academic-slides-in-typst-an-actual-use-case-example-using-touying-and-touying-unistra-pristine/4606
- touying-unistra-pristine (academic theme): https://typst.app/universe/package/touying-unistra-pristine/
- Typst grid / side-by-side layout: https://forum.typst.app/t/what-is-the-preferred-way-of-doing-side-by-side-content/3672 · https://typst.app/docs/reference/layout/grid/
- CeTZ — Typst Universe: https://typst.app/universe/package/cetz/
- Fletcher — Typst Universe: https://typst.app/universe/package/fletcher/

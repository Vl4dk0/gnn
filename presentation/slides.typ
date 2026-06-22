// =============================================================================
//  Bachelor's thesis defense — 12-minute deck
//  "Machine Learning for Generation of Graph of Given Degree and Girth"
//  Vladimir Jancar — Comenius University Bratislava, 2026
//
//  Build:   typst compile --font-path fonts slides.typ slides.pdf
//  Watch:   typst watch --font-path fonts slides.typ slides.pdf
//
//  Fonts:   Fraunces static TTFs (family "Fraunces 9pt") live in ./fonts/,
//           extracted from the official release:
//           https://github.com/undercasetype/Fraunces/releases (UnderCaseType_Fraunces_1.000.zip,
//           "Fonts - Desktop/static/ttf/Fraunces9pt-*.ttf"). Build with --font-path fonts.
//
//  Framework: Touying + metropolis theme, reskinned to vladimirjancar.sk light-mode palette.
//  Figures live in ./figures/ (paths are relative to THIS file).
// =============================================================================

#import "@preview/touying:0.7.4": *
#import themes.metropolis: *
#import "@preview/fletcher:0.5.8": diagram, edge, node

// Handout mode collapses #pause subslides to their final state (one page per slide).
// Enable for previews/printing with:  typst compile --input handout=true slides.typ
#let handout = sys.inputs.at("handout", default: "false") == "true"

// =============================================================================
//  BACKGROUND TEXTURE
//  Tiled GRAPH / NEURL / NTWRK text, monospace ~11pt, diagonal stripes via
//  (row+col) % 3 word assignment. Color: #1a1a1a at ~5% opacity — whisper-subtle.
//  Column pitch: 50pt, row pitch: 20pt. No rotation.
// =============================================================================
#let graph-texture-background = {
  let col-pitch = 44pt
  let row-pitch = 17pt
  let words = ("GRAPH", "NEURL", "NTWRK")
  // Slide is 254mm × 142.875mm at 16:9 (Touying default pt: 358.08pt × 201.42pt approx).
  // Overshoot generously so the tile covers the full page regardless of exact dims.
  let n-cols = 30
  let n-rows = 60
  let texture-color = rgb("#1a1a1a").transparentize(92%)
  place(
    dx: 0pt,
    dy: 0pt,
    {
      for r in range(n-rows) {
        for c in range(n-cols) {
          let word = words.at(calc.rem(r + c, 3))
          place(
            dx: c * col-pitch,
            dy: r * row-pitch,
            text(
              font: "Courier New",
              size: 11pt,
              fill: texture-color,
              word,
            ),
          )
        }
      }
    },
  )
}

// =============================================================================
//  SECTION DIVIDER — no mid-slide progress bar
//  Keeps only the section heading + a static grey line (#d0d0d0).
//  The running progress bar stays at the bottom of content slides only.
// =============================================================================
#let section-slide-no-bar(
  config: (:),
  level: 1,
  numbered: true,
  body,
) = touying-slide-wrapper(self => {
  let slide-body = {
    graph-texture-background
    set std.align(horizon)
    show: pad.with(20%)
    set text(size: 1.5em)
    stack(
      dir: ttb,
      spacing: 0.8em,
      text(
        font: "Fraunces 9pt",
        fill: self.colors.neutral-darkest,
        utils.display-current-heading(
          level: level,
          numbered: numbered,
          style: auto,
        ),
      ),
      line(length: 100%, stroke: 1pt + rgb("#d0d0d0")),
    )
    text(self.colors.neutral-dark, body)
  }
  self = utils.merge-dicts(self, config-page(
    fill: self.colors.neutral-lightest,
  ))
  touying-slide(self: self, config: config, slide-body)
})

// =============================================================================
//  WITHIN-SLIDE REVEAL INDICATOR (footer-left)
//  A small row of dots showing how deep we are in the CURRENT slide's reveals:
//  filled (ink) up to and including the current subslide, hollow (line color)
//  for the remaining reveals. Hidden on single-reveal slides.
//
//  Touying exposes both values on `self` during footer rendering (the footer is
//  evaluated per-subslide via call-or-display(self, ...)):
//    self.subslide — current subslide index (1-based)
//    self.repeat   — total number of subslides for the current slide (resolved,
//                    works for #pause auto-detection AND #slide(repeat: n, ..))
//  This mirrors how the theme's progress bar / nav use these fields.
// =============================================================================
#let reveal-dots(self) = {
  let total = self.at("repeat", default: 1)
  let current = self.at("subslide", default: 1)
  // Only show on multi-reveal slides.
  if total == none or total <= 1 {
    return none
  }
  let ink = self.colors.neutral-darkest
  let linecol = self.colors.primary-light
  let d = 0.45em
  set align(horizon)
  stack(
    dir: ltr,
    spacing: 0.35em,
    ..range(1, total + 1).map(i => box(
      width: d,
      height: d,
      radius: 50%,
      fill: if i <= current { ink } else { none },
      stroke: if i <= current { none } else { 0.5pt + linecol },
    )),
  )
}

// =============================================================================
//  THEME — greyscale editorial palette matching vladimirjancar.sk light mode
//  Tokens from frontend/src/index.css:
//    primary (#5a5a5a) = accent/headings
//    primary-light (#d0d0d0) = lines/borders
//    secondary (#1a1a1a) = header bar background
//    neutral-lightest (#ffffff) = slide/panel background
//    neutral-dark (#1a1a1a) = body text
//    neutral-darkest (#1a1a1a) = strongest text
//  Page fill: #f0f0f0 (outermost background, matches --color-bg0).
//  Progress bar: #5a5a5a on #d0d0d0 — grey on light-grey.
// =============================================================================
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  // Footer-left: within-slide reveal dots (additional to the bottom progress bar
  // and the right-side slide counter, both of which are left untouched).
  footer: self => reveal-dots(self),
  config-common(handout: handout, new-section-slide-fn: section-slide-no-bar),
  config-colors(
    primary: rgb("#5a5a5a"),
    primary-light: rgb("#d0d0d0"),
    secondary: rgb("#1a1a1a"),
    neutral-lightest: rgb("#ffffff"),
    neutral-dark: rgb("#1a1a1a"),
    neutral-darkest: rgb("#1a1a1a"),
  ),
  config-page(
    fill: rgb("#f0f0f0"),
    // Inject the texture behind every content slide via the page background.
    background: graph-texture-background,
  ),
  config-info(
    title: {
      text(
        font: "Fraunces 9pt",
      )[Strojové učenie pre generáciu grafov\ s daným stupňom a obvodom]
      linebreak()
      v(0.3em)
      text(
        font: "Fraunces 9pt",
        size: 0.5em,
        fill: rgb("#888888"),
      )[Machine Learning for Generation of Graphs of Given Degree and Girth]
    },
    author: [
      Vladimír Jančár
      #linebreak()
      #text(size: 0.75em, fill: rgb("#888888"))[Mgr. Ján Pastorek]
    ],
    date: datetime(year: 2026, month: 6, day: 23).display(
      "[day]. [month]. [year]",
    ),
    institution: [Univerzita Komenského v Bratislave],
    // logo: image("figures/uk-logo.svg", height: 1.5cm),  // add a real crest if available
  ),
)

// Body text: Inter (bundled TTF in ./fonts), with system sans fallbacks.
// Typst does not accept CSS generic families like "sans-serif" — omit them.
#set text(size: 22pt, font: ("Inter", "Helvetica Neue", "Arial"))
// Apply Fraunces to all headings (slide titles rendered via == headings, section titles via =).
#show heading: set text(font: "Fraunces 9pt")
// Emphasis: bold is near-black and heavier (never paler than body).
#show strong: set text(fill: rgb("#1a1a1a"), weight: 800)
// Slides don't need "Figure N:" labels; also avoids a counter-convergence warning
// caused by figures living on slides with #pause subslides.
#set figure(numbering: none)

// =============================================================================
//  FLETCHER DIAGRAM HELPERS — greyscale, reused on the RL and Forge slides.
// =============================================================================
#let ink = rgb("#1a1a1a")
#let muted = rgb("#555555")
#let dimcol = rgb("#888888")
#let linecol = rgb("#d0d0d0")

// Diagram accents (explainer diagrams only; deck chrome stays grey).
// Muted, serious palette: slate navy, brick red, steel.
#let dgnavy = rgb("#1f3a5f")
#let dgbronze = rgb("#8c3b3b")
#let dgsteel = rgb("#4a6572")
#let dgsteelfill = rgb("#eef1f3")

// -----------------------------------------------------------------------------
#title-slide()

// =============================================================================
// GNN OPENING SLIDE — ultra-minimal: one hero image, one sentence below.
// =============================================================================

== Grafové neurónové siete

#v(1fr)
#align(center, image("figures/gnn/fig-gnn-overall.png", width: 100%))
#v(1fr)

// =============================================================================
// SLIDE 1 — "Čo sú GNN": mechanics. ONE large centered image per reveal that
// swaps in place, driven by self.subslide (no #pause inside #grid — known
// Touying bug). Numbers-first: r1 colorless vectors, r2 they turn into color.
// Equations live in the slide text, never in the figure.
// =============================================================================
#slide(title: [Grafové neurónové siete], repeat: 5, self => {
  let s = self.subslide

  // Reveals 1-2: side-by-side two-column transformation view (numbers -> colors).
  // The right column space is reserved on reveal 1 so the left does not shift.
  if s <= 3 {
    // Size each column figure by HEIGHT so the tall numbers graph fits in its
    // half without overflowing the page; width then follows the aspect.
    let column(img, cap) = stack(
      dir: ttb,
      spacing: 0.4em,
      image(img, height: 70%),
      text(size: 0.85em, cap),
    )
    v(0.3em)
    grid(
      columns: (1fr, 1fr),
      gutter: 1.6em,
      align: center + bottom,
      align(center, column(
        "figures/gnn/mp-graph-nums.pdf",
        [Každý vrchol má vektor],
      )),
      if s == 2 {
        align(center, column(
          "figures/gnn/mp-graph.pdf",
          [Vektor zobrazíme ako farbu],
        ))
      } else if s == 3 {
        align(center, column(
          "figures/gnn/mp-highlight.pdf",
          [Znázorníme message-passing],
        ))
      } else {
        hide(align(
          center,
          column(
            "figures/gnn/mp-graph.pdf",
            [Vektor zobrazíme ako farbu cez RGB],
          ),
        ))
      },
    )
  } else {
    // Reveals 4-5: scene switch to a single large centered focused graph.
    let src = if s == 4 { "figures/gnn/mp-aggregate-2.pdf" } else {
      "figures/gnn/mp-aggregate.pdf"
    }

    // Reveal 3 has only a caption below, so the graph can be big; reveals 4-5
    // also carry the equation, so a touch smaller.
    let h = if s >= 4 { 52% } else { 80% }

    v(0.15em)
    align(center, image(src, height: h))
    v(0.3em)

    let cap = [Susedné vrcholy pošlú svoje vektory, a použijeme agregčnú funkciu na výpočet nového vektora pre vrchol $v$]
    align(center, text(size: 0.85em, cap))

    let vv = [#box(baseline: 0.18em, rect(
      width: 1em,
      height: 1em,
      fill: rgb(220, 100, 60),
      stroke: 0.6pt + rgb("#3a3a3a"),
    ))]
    let n1 = [#box(baseline: 0.18em, rect(
      width: 1em,
      height: 1em,
      fill: rgb(60, 200, 80),
      stroke: 0.6pt + rgb("#3a3a3a"),
    ))]
    let n2 = [#box(baseline: 0.18em, rect(
      width: 1em,
      height: 1em,
      fill: rgb(60, 80, 220),
      stroke: 0.6pt + rgb("#3a3a3a"),
    ))]
    let n3 = [#box(baseline: 0.18em, rect(
      width: 1em,
      height: 1em,
      fill: rgb(200, 200, 60),
      stroke: 0.6pt + rgb("#3a3a3a"),
    ))]
    let res = [#box(baseline: 0.18em, rect(
      width: 1em,
      height: 1em,
      fill: rgb(135, 145, 105),
      stroke: 0.6pt + rgb("#3a3a3a"),
    ))]

    // Reveals 4 and 5: the exact mean computation on the SLIDE.
    if s == 5 {
      v(0.3em)
      align(center, text(size: 0.85em)[
        $vv v = (220,100,60), space n1 n_1 = (60,200,80), space n2 n_2 = (60,80,220), space n3 n_3 = (200,200,60)$
        #v(0.25em)
        nový $v$ = priemer($v, n_1, n_2, n_3$) = $(540 / 4, 580 / 4, 420 / 4) = (135,145,105)$ $res$
      ])
    }
  }
})

// =============================================================================
// DIFFUSION SLIDE HELPER — same numbers->colors intro as the mechanics slide,
// then the round progression:
//   r1: LEFT = numbers graph ("Vektory ako čísla."), RIGHT half reserved.
//   r2: RIGHT = colored round-0 graph ("Vektory ako farby."), so the two halves
//       side by side show the (a,b,c) -> color transformation.
//   r3: scene switch to the 3-column round progression 0 | 2 | 8.
// No #pause inside the #grid (Touying bug); gated on self.subslide.
// =============================================================================
#let diffusion-slide(title, nums-src, colors0-src, snaps) = {
  slide(title: title, repeat: 3, self => {
    let s = self.subslide

    if s <= 2 {
      // Two-column transformation: numbers (left) | colors (right, reserved r1).
      // Height-based images so the (a,b,c) labels fit at half-width, uncropped.
      let column(img, cap) = stack(
        dir: ttb,
        spacing: 0.4em,
        image(img, height: 70%),
        text(size: 0.85em, cap),
      )
      v(0.3em)
      grid(
        columns: (1fr, 1fr),
        gutter: 1.6em,
        align: center + bottom,
        align(center, column(nums-src, [Vektory ako čísla])),
        if s == 2 {
          align(center, column(colors0-src, [Vektory ako farby]))
        } else {
          hide(align(center, column(colors0-src, [Vektory ako farby])))
        },
      )
    } else {
      // Reveals 3-4: the 3-column round progression (all shown at once).
      let col(idx) = {
        let (src, cap) = snaps.at(idx)
        stack(
          dir: ttb,
          spacing: 0.4em,
          image(src, width: 100%),
          text(size: 0.8em, fill: muted, cap),
        )
      }
      v(0.5em)
      grid(
        columns: (1fr, 1fr, 1fr),
        gutter: 1.4em,
        align: center + bottom,
        col(0), col(1), col(2),
      )
    }
  })
}

// SLIDE 2 — mean diffusion: SAME graph + SAME init colors as the sum slide, so
// the two examples are directly comparable. Mean over the closed neighbourhood
// converges towards one shared color (oversmoothing).
#diffusion-slide(
  [Message passing (mean aggregation)],
  "figures/gnn/diff-sum-nums.pdf",
  "figures/gnn/diff-mean-0.pdf",
  (
    ("figures/gnn/diff-mean-0.pdf", [začiatok]),
    ("figures/gnn/diff-mean-2.pdf", [po 2 kolách]),
    ("figures/gnn/diff-mean-8.pdf", [po 8 kolách]),
  ),
)

// SLIDE 3 — sum diffusion: SAME graph, restarted with channel-dominant feature
// vectors. Sum over the closed neighbourhood grows fast and saturates.
#diffusion-slide(
  [Message passing (sum aggregation)],
  "figures/gnn/diff-sum-nums.pdf",
  "figures/gnn/diff-sum-0.pdf",
  (
    ("figures/gnn/diff-sum-0.pdf", [začiatok]),
    ("figures/gnn/diff-sum-2.pdf", [po 2 kolách]),
    ("figures/gnn/diff-sum-8.pdf", [po 8 kolách]),
  ),
)

// Closes the message-passing story by showing the full pipeline photo again
// (no text; presenter narrates). Same hero image as the opening GNN slide.
== Grafové neurónové siete

#v(1fr)
#align(center, image("figures/gnn/fig-gnn-overall.png", width: 100%))
#v(1fr)

== Regularita a Obvod

#v(0.5em)
#grid(
  columns: (1fr, 2pt, 1fr),
  gutter: 0pt,
  align: top,
  // LEFT — stupeň
  pad(right: 2em)[
    #image("figures/degree-girth/fig-degree.pdf", width: 100%)
    #v(0.4em)
    #text(size: 0.88em)[
      #align(center)[
        Graf je *k-regulárny*, ak každý jeho vrchol má práve *k* susedov
      ]
    ]
  ],
  // DIVIDER — visible grey line between columns
  align(center, line(
    angle: 90deg,
    length: 75%,
    stroke: 1.8pt + rgb("#bbbbbb"),
  )),
  // RIGHT — obvod
  pad(left: 2em)[
    #image("figures/degree-girth/fig-girth.pdf", width: 100%)
    #v(0.4em)
    #text(size: 0.88em)[
      #align(center)[
        Obvod grafu *g* je veľkosť najkratšieho cyklu v grafe
      ]
    ]
  ],
)


#slide(title: [(k,g)-grafy], repeat: 6, self => {
  let s = self.subslide

  let src = if s == 1 {
    "figures/petersen/petersen-base.pdf"
  } else if s == 2 {
    "figures/petersen/petersen-degree.pdf"
  } else if s == 3 {
    "figures/petersen/petersen-girth.pdf"
  } else if s == 4 {
    "figures/petersen/petersen-numbered.pdf"
  } else {
    "figures/petersen/dodecahedron-numbered.pdf"
  }

  grid(
    columns: (1.1fr, 1fr),
    gutter: 2.5em,
    align: horizon,
    align(center, image(src, height: 70%)),
    text(size: 0.9em)[
      #if s <= 3 [
        Toto je (3,5)-graf
        #v(0.5em)
        #if s >= 2 [lebo je 3-regulárny] else [#hide[lebo je 3-regulárny]]
        #v(0.5em)
        #if s >= 3 [a jeho obvod je 5] else [#hide[a jeho obvod je 5]]
      ] else if s == 4 [
        Tento (3,5)-graf je *klietka*,\
        lebo je najmenší možný (3,5)-graf
      ] else [
        Toto je (3,5)-graf
        #v(0.5em)
        #if (
          s >= 6
        ) [Ale *nie je* to klietka] else [#hide[Ale *nie je* to klietka]]
      ]
    ],
  )
})


// Reinforcement learning: the agent builds a (3,4)-graph = K_{3,3} edge by edge.
// Scene 1 is title-only; each later scene swaps in the next build step. Every step
// is one large composite (graph on the left, action space on the right with the
// chosen action boxed, a remove option, and triangle/degree-illegal adds struck out).
#slide(title: [Reinforcement learning], repeat: 12, self => {
  let s = self.subslide
  if s == 1 {
    // What reinforcement learning is: the classic agent-environment loop.
    set align(horizon)
    align(center)[
      #set text(size: 15pt)
      #diagram(
        node-stroke: 1.3pt + linecol,
        node-fill: white,
        node-inset: 10pt,
        spacing: (9em, 4em),
        node((0, 0), [Agent], corner-radius: 5pt),
        node((1, 0), [Prostredie], corner-radius: 5pt),
        edge(
          (0, 0),
          (1, 0),
          "-|>",
          [akcia],
          label-pos: 0.5,
          label-side: left,
          bend: 28deg,
          stroke: 1.2pt + muted,
        ),
        edge(
          (1, 0),
          (0, 0),
          "-|>",
          [stav, odmena],
          label-pos: 0.5,
          label-side: left,
          bend: 28deg,
          stroke: 1.2pt + muted,
        ),
      )
      #v(0.8em)
      #text(
        size: 0.9em,
      )[Agent mení graf akciami a dostáva odmenu/trest a z toho sa učí]
    ]
  } else if s == 2 {
    set align(horizon)
    align(center, text(
      size: 1.5em,
    )[Natrénovaný RL agent by zostrojil (3,4)-graf takto:])
  } else {
    let i = s - 3
    // Height-based so the wide composite fits on one page. No caption.
    align(center, image("figures/rl/rl-step-" + str(i) + ".pdf", height: 100%))
  }
})


== Reinforcement learning: výsledky

#v(1fr)
#align(center)[
  #text(fill: muted)[Výsledky z PERUN behu, dopĺňam.]
  #v(1.2em)
  #text(size: 1.05em)[RL sa podarilo zostrojiť len malé ciele.]
]
#v(1fr)


// Voltage lifts — concept first (small k-regular graph -> bigger one), then the
// concrete Z5 -> Petersen construction. Stepped, one image per reveal.
#slide(title: [Voltage lifts], repeat: 4, self => {
  let s = self.subslide
  let line(body, step) = if s >= step [#body] else [#hide[#body]]
  grid(
    columns: (1.4fr, 1fr),
    gutter: 2em,
    align: horizon,
    align(center, grid(
      columns: (auto, auto),
      align: horizon,
      gutter: 1.2em,
      image("figures/voltage/fig-voltage-base.pdf", height: 70%),
      image("figures/voltage/fig-voltage-lift.pdf", height: 70%),
    )),
    text(size: 0.95em)[
      #line([Zachová k-regularitu a zväčší graf], 1)
      #v(0.7em)
      #line(
        [
          Potrebujeme zvoliť základný graf $G$ a grupu $Gamma$
        ],
        2,
      )
      #v(0.7em)
      #line([Priradiť jej prvky hranám - napätia], 3)
      #v(0.7em)
      #line(
        [
          Výsledný počet vrcholov =
          #v(0em)
          |$Gamma$| × |V(grafu)|
        ],
        4,
      )
    ],
  )
})

// Concrete construction: group Z5, a 2-vertex base graph, lifted to Petersen.
// Each reveal swaps one large image in place (no #pause inside a grid).
#slide(title: [Voltage lifts], repeat: 7, self => {
  let s = self.subslide

  let src = if s == 1 {
    "figures/voltage/vlift-base-no-voltage.pdf"
  } else if s == 2 {
    "figures/voltage/vlift-base-no-voltage.pdf"
  } else if s == 3 {
    "figures/voltage/vlift-base.pdf"
  } else if s == 4 {
    "figures/voltage/vlift-nodes.pdf"
  } else if s == 5 {
    "figures/voltage/vlift-step1.pdf"
  } else if s == 6 {
    "figures/voltage/vlift-step2.pdf"
  } else {
    "figures/voltage/vlift-step3.pdf"
  }

  let cap = if s == 1 [
    Základný graf
  ] else if s == 2 [
    Zvolená grupa: $Z_5 = {0,1,2,3,4}$
  ] else if s == 3 [
    Zvolené napätia
  ] else if s == 4 [
    Pre každý prvok $Z_5$ jedna kópia vrcholu
  ] else if s == 5 [
    Slučka na 0 dá hrany (0,x) → (0,x+1). Vznikne vonkajší päťuholník
  ] else if s == 6 [
    Slučka na 1 dá hrany (1,x) → (1,x+2). Vznikne vnútorná hviezda
  ] else [
    Hrana 0→1 dá hrany (0,x) → (1,x). Výsledok je Petersenov graf
  ]

  v(0.15em)
  // Size everything by HEIGHT so the image + caption always fit on one page.
  // The step composites are wide (~2.2:1); squarish base/nodes get more height.
  let h = 70%
  align(center, image(src, height: h))
  v(0.4em)
  align(center, text(size: 0.9em, cap))
})


== Voltage lifts a GNN

#grid(
  columns: (1.55fr, 1fr),
  gutter: 2em,
  align: horizon,
  [
    #align(center)[
      #set text(size: 13pt)
      #diagram(
        node-stroke: 1.3pt + linecol,
        node-fill: white,
        node-inset: 8pt,
        spacing: (2.2em, 3.0em),
        node(
          (1, -2),
          align(center)[grupa $Gamma$\ + základný graf],
          corner-radius: 5pt,
        ),
        node(
          (1, -1),
          align(center)[priraď napätia],
          corner-radius: 5pt,
        ),
        node(
          (0, 0),
          align(center)[spočítaj lift],
          corner-radius: 5pt,
        ),
        node(
          (1, 1),
          align(center)[over obvod],
          corner-radius: 5pt,
        ),
        node(
          (1, 2),
          align(center)[hotovo],
          corner-radius: 5pt,
        ),
        node(
          (2, 0),
          align(center)[GNN predikuje obvod\ z napätí],
          corner-radius: 5pt,
          stroke: 1.5pt + ink,
          fill: dgsteelfill,
        ),

        edge((1, -2), (1, -1), "-|>", stroke: 1.2pt + muted),
        edge((1, -1), (0, 0), "-|>", stroke: 1.2pt + muted),
        edge((0, 0), (1, 1), "-|>", stroke: 1.2pt + muted),
        edge((1, 1), (1, 2), "-|>", stroke: 1.2pt + muted),
        edge((1, 1), (2, 0), "-|>", stroke: 1.2pt + muted),
        edge((2, 0), (1, -1), "-|>", stroke: 1.2pt + muted),
      )
    ]
  ],
  [
    #text(size: 0.92em)[
      GNN tu vieme použiť na urýchlenie hľadania vyhovujúceho priradenia napätí

      #pause
      GNN sme učili hádať obvod po lifte podľa priradených napätí

      #pause
      #v(0.5em)
      #text(fill: muted)[Výsledky dopĺňam z behu na PERUN]
    ]
  ],
)


// Edge-swap refinement, stepped one 2-switch at a time. ONE large centered
// image per reveal that swaps in place, driven by self.subslide (no #pause
// inside a grid — known Touying bug). All frames share ONE fixed 14-vertex
// layout. Removed edges read as bold + dashed, added edges as bold + solid.
// Refinement: show the problem graph, then the text-free 2-swap mechanism, then
// the repair example. One large image per reveal swaps in place.
#slide(title: [Refinement], repeat: 5, self => {
  let s = self.subslide

  // Each 2-swap is shown as ONE page: the removal (left) and the addition
  // (right) side by side, both visible at once. No #pause inside the grid
  // (known Touying bug); the plain grid renders both cells simultaneously.
  let pair(left-src, right-src) = grid(
    columns: (1fr, 1fr),
    gutter: 1.4em,
    align: center + bottom,
    image(left-src, height: 80%), image(right-src, height: 80%),
  )

  let cap = if s == 1 [
    Skoro-dobrý graf, ale ostali krátke cykly
  ] else if s == 2 [
    2-swap: zachováme stupeň, vymažeme krátky cyklus
  ] else if s == 3 [
    Prvý 2-swap: odoberieme dve hrany (vľavo), pridáme dve nové (vpravo)
  ] else if s == 4 [
    Druhý 2-swap: odoberieme (vľavo), pridáme (vpravo)
  ] else [
    Po dvoch swapoch má graf obvod 5
  ]

  v(0.1em)
  if s == 3 {
    pair("figures/refine/refine-1a.pdf", "figures/refine/refine-1b.pdf")
  } else if s == 4 {
    pair("figures/refine/refine-2a.pdf", "figures/refine/refine-2b.pdf")
  } else if s == 2 {
    // The 2-swap mechanism figure is wide (before/after).
    align(center, image("figures/refine/edge-swap-cycle.pdf", width: 78%))
  } else {
    let src = if s == 1 {
      "figures/refine/refine-0.pdf"
    } else {
      "figures/refine/refine-final.pdf"
    }
    align(center, image(src, height: 78%))
  }
  v(0.3em)
  align(center, text(size: 0.9em, cap))
})

== Refinement a GNN

#grid(
  columns: (1.5fr, 1fr),
  gutter: 2em,
  align: horizon,
  [
    #align(center)[
      #set text(size: 12pt)
      // With GNN: a scoring step ranks the candidates.
      #diagram(
        node-stroke: 1.3pt + linecol,
        node-fill: white,
        node-inset: 8pt,
        spacing: (2.0em, 2.0em),
        node((0, 0), [Skoro-dobrý\ graf]),
        edge((0, 0), (1, 0), "-|>", stroke: 1.1pt + muted),
        node((1, 0), align(center)[nájdi kandidátov\ na 2-swap]),
        edge((1, 0), (2, 0), "-|>", stroke: 1.1pt + muted),
        node(
          (2, 0),
          align(center)[GNN ohodnotí\ kandidátov],
          stroke: 1.5pt + ink,
          fill: dgsteelfill,
        ),
        edge((2, 0), (3, 0), "-|>", stroke: 1.1pt + muted),
        node((3, 0), align(center)[skúšaj\ najlepších]),
      )
    ]
  ],
  [
    #text(size: 0.92em)[
      GNN navrhuje poradie kandidátov

      #pause
      #v(0.5em)
      #text(fill: muted)[výsledky dopĺňam z behu na PERUN]
    ]
  ],
)


// Excision as a six-cell storyboard: five graph cells fill in one per reveal
// (dodecahedron -> BFS depth 2 -> remove -> stitch -> Petersen), the sixth cell
// carries the running caption. hide() reserves space so the grid never reflows.
#slide(title: [Excision], repeat: 6, self => {
  let s = self.subslide
  let cell(idx, path) = {
    let img = align(center + horizon, image(
      "figures/excision/" + path,
      width: 70%,
    ))
    if s - 1 >= idx { img } else { hide(img) }
  }
  let caps = (
    [Z veľkého (k,g)-grafu spravíme menší (k,g)-graf],
    [Začneme s priveľkým (3,5)-grafom: dodekaéder, 20 vrcholov],
    [Z koreňa vyrastieme BFS strom do hĺbky 2 - Moorov polomer],
    [Strom naraz odstránime],
    [Chýbajúce hrany zošijeme tak, aby obvod ostal 5],
    [Výsledok je Petersenov graf, teda (3,5)-klietka],
  )
  v(0.3em)
  grid(
    columns: (1fr, 1fr, 1fr),
    rows: (44%, 44%),
    gutter: 0.6em,
    cell(1, "exc-dodeca.pdf"),
    cell(2, "exc-bfs.pdf"),
    cell(3, "exc-removed.pdf"),
    cell(4, "exc-stitched.pdf"),
    cell(5, "exc-petersen.pdf"),
    align(center + horizon, text(size: 0.84em, caps.at(s - 1))),
  )
})

== Excision a GNN

#grid(
  columns: (1.5fr, 1fr),
  gutter: 2em,
  align: horizon,
  [
    #align(center)[
      #set text(size: 12pt)
      // With GNN: a learned repair policy proposes the stitching.
      #diagram(
        node-stroke: 1.3pt + linecol,
        node-fill: white,
        node-inset: 8pt,
        spacing: (2.0em, 2.0em),
        node((0, 0), align(center)[veľký\ (k,g)-graf]),
        node((1, 1), align(
          center,
        )[odstráň strom\ do hĺbky d z nejakého koreňa]),
        node(
          (2, 0),
          align(center)[GNN\ navrhne zošitie],
          stroke: 1.5pt + ink,
          fill: dgsteelfill,
        ),
        node((2, 1), align(
          center,
        )[hotovo]),
        edge((0, 0), (1, 1), "-|>", stroke: 1.1pt + muted),
        edge((1, 1), (2, 0), "-|>", stroke: 1.1pt + muted),
        edge((2, 0), (2, 1), "-|>", stroke: 1.1pt + muted),
        edge((2, 0), (0, 0), "-|>", stroke: 1.1pt + muted),
      )
    ]
  ],
  [
    #text(size: 0.92em)[
      GNN navrhuje ktoré hrany vytvoriť.

      #pause
      #v(0.5em)
      #text(fill: muted)[výsledky dopĺňam z behu na PERUN]
    ]
  ],
)


== Forge

#align(center)[
  #set text(size: 14pt)
  #v(0.3em)
  #diagram(
    node-fill: dgsteelfill,
    spacing: (4.5em, 1.2em),
    node(
      (0, 0),
      [Producer\ #text(0.78em, fill: muted)[voltage · dosah]],
      corner-radius: 5pt,
      inset: 10pt,
      stroke: 1.3pt + dgnavy,
    ),
    edge((0, 0), (1, 0), "-|>", stroke: 1.2pt + dgbronze),
    node(
      (1, 0),
      [Refinement\ #text(0.78em, fill: muted)[oprava · platnosť]],
      corner-radius: 5pt,
      inset: 10pt,
      stroke: 1.3pt + dgsteel,
    ),
    edge((1, 0), (2, 0), "-|>", stroke: 1.2pt + dgbronze),
    node(
      (2, 0),
      [Excision\ #text(0.78em, fill: muted)[zmenši · veľkosť]],
      corner-radius: 5pt,
      inset: 10pt,
      stroke: 1.3pt + dgnavy,
    ),
  )
]

#pause

#align(center)[
  #text(
    size: 0.92em,
  )[Odovzdanie cez *defective fraction $<= tau$*, producer je vymeniteľný.]
]


== Výsledky

#grid(
  columns: (1fr, 1.05fr),
  gutter: 2em,
  align: horizon,
  [
    #image("figures/results/fig-results.pdf", width: 86%)
  ],
  [
    #text(size: 0.9em)[
      *voltage-rl*: najširší dosah.

      #pause
      #v(0.35em)
      Klasické: najmenšie tam, kde fungujú.

      #pause
      #v(0.35em)
      Forge: úspech 0,46, veľkosť 1,23× Moore.

      #pause
      #v(0.35em)
      #text(fill: muted)[Učené zriedka prekoná neučené, a to je zistenie.]
    ]
  ],
)


// =============================================================================
//  QUESTION SLIDES — supervisor + opponent. For now ONLY the question text is
//  shown (answers to be added later). One question per slide, before the záver.
// =============================================================================

== Otázka školiteľa (1)

#v(1.2em)
#text(size: 1.05em)[
  Čakal som, že kombinovaná metóda Forge dosiahne najlepšie výsledky. Podľa
  tabuliek vychádza najlepšie A\* pri hľadaní najmenšieho príkladu (k,g)-grafu a
  voltage-RL pri hľadaní aspoň nejakého príkladu. Čo zapríčinilo dané výsledky?
]

== Otázka školiteľa (2)

#v(1.2em)
#text(size: 1.05em)[
  Prečo ste metódy zastavili po 60 sekundách? Ako vieme, či modely so strojovým
  učením nepotrebovali viac času na konvergenciu?
]

== Otázka oponenta (1)

#v(1.2em)
#text(size: 1.05em)[
  V práci ukazujete, že štandardné GNN nedokážu spoľahlivo predikovať obvod. Aké
  štrukturálne alebo pozičné príznaky by ste do modelu pridali, aby lepšie
  zachytil globálnu cyklickú štruktúru grafu?
]

== Otázka oponenta (2)

#v(1.2em)
#text(size: 1.05em)[
  Pipeline Forge kombinuje voltage producer, refinement a excision. Ako by ste
  systematicky ladili prah defective fraction a ako by ste experimentálne
  odlíšili prínos diverzity kandidátov od kvality naučenej politiky?
]


== Záver

#v(1.0em)
// PLACEHOLDER — prepísať. Tri vety na záver.
#text(size: 1.05em)[
  Postavil som postupnosť metód na generovanie (k,g)-grafov a porovnal učené
  verzie s neučenými.

  #v(0.5em)
  Hlavné zistenie je, že učenie navádza hľadanie, ale nenahrádza exaktné overenie.

  #v(0.5em)
  Ďalej by sa dali pridať pozičné a štruktúrne príznaky pre lepšiu predikciu obvodu.
]


// -----------------------------------------------------------------------------
#focus-slide[
  Ďakujem za pozornosť.
]

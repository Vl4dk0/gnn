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
// VECTOR <-> COLOR INTRO — every vertex carries a vector, which we draw as a
// colour. The SAME graph, vectors and colours as the diffusion progression that
// follows (diff-sum round 0), so the example is consistent end to end.
// =============================================================================
#slide(title: [Grafové neurónové siete], self => {
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
      "figures/gnn/diff-sum-nums.pdf",
      [Každý vrchol má vektor],
    )),
    align(center, column(
      "figures/gnn/diff-sum-0.pdf",
      [Zobrazme ako farby (podľa RGB)],
    )),
  )
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
// SUM DIFFUSION PROGRESSION — repeating SUM aggregation over the same graph.
// One step further than the mechanics slide: do it for every vertex, round
// after round. Vectors grow and saturate; after 8 rounds each vector encodes
// the neighbourhood up to depth 8. Three snapshots side by side (0, 2, 8).
#slide(title: [Grafové neurónové siete], self => {
  let col(src, cap) = stack(
    dir: ttb,
    spacing: 0.4em,
    image(src, width: 100%),
    text(size: 0.8em, fill: muted, cap),
  )
  v(0.5em)
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1.4em,
    align: center + bottom,
    col("figures/gnn/diff-sum-0.pdf", [začiatok]),
    col("figures/gnn/diff-sum-2.pdf", [po 2 kolách]),
    col("figures/gnn/diff-sum-8.pdf", [po 8 kolách]),
  )
})

// READ-OUT — after message passing each vertex carries a vector; those vectors
// are the input to a neural network. Size by HEIGHT so the wide composite never
// overflows onto a phantom page.
== Grafové neurónové siete

#v(1fr)
#align(center, image("figures/gnn/gnn-readout.pdf", height: 62%))
#v(0.7em)
#align(
  center,
  text(
    size: 0.9em,
  )[Po message-passingu nesie každý vrchol vektor svojho okolia, ktorý je vstupom do neurónovej siete],
)
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


// (k,g)-graphs: start with the dodecahedron (a (3,5)-graph: 3-regular, girth 5),
// note it is NOT the cage, then reveal the Petersen graph as the actual cage.
// The Moore's-bound slide that follows explains WHY Petersen is the smallest.
#slide(title: [(k,g)-grafy], repeat: 5, self => {
  let s = self.subslide

  let src = if s == 1 {
    "figures/petersen/dodecahedron-base.pdf"
  } else if s == 2 {
    "figures/petersen/dodecahedron-degree.pdf"
  } else if s == 3 {
    "figures/petersen/dodecahedron-girth.pdf"
  } else if s == 4 {
    "figures/petersen/dodecahedron-base.pdf"
  } else {
    "figures/petersen/petersen-base.pdf"
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
        Tento (3,5)-graf ale *nie je* klietka
      ] else [
        Toto je tiež (3,5)-graf
        #v(0.5em)
        a je to *klietka* - najmenší možný (3,5)-graf
      ]
    ],
  )
})


// Moore's bound — quick explainer right after the (k,g)-graph intro. The BFS
// ball tree forces 1 + k + k(k-1) distinct vertices up to radius (g-1)/2, which
// is the smallest a (k,g)-graph can be (the (3,5) tree gives 10 = Petersen).
== Moore's bound

#v(0.3em)
#align(
  center,
  text(size: 0.95em)[Najmenší počet vrcholov, ktorý (k,g)-graf vôbec musí mať],
)
#v(0.3em)
#align(center, image("figures/petersen/fig-moore.pdf", height: 60%))
#v(0.3em)
#align(
  center,
  text(
    size: 0.85em,
    fill: muted,
  )[Z koreňa: k susedov, potom každý ďalší k-1. Do hĺbky (g-1)/2 musia byť všetky rôzne, inak vznikne cyklus kratší ako g],
)


// Reinforcement learning: describe the ENVIRONMENT we built, not a build trace.
//   s1: what RL is (agent-environment loop),
//   s2: the action (toggle an edge on a vertex pair) and when it is legal,
//   s3: the goal and the reward.
#slide(title: [Reinforcement learning], repeat: 3, self => {
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
      )[Agent stavia graf hranu po hrane, za to dostáva odmenu a z nej sa učí]
    ]
  } else if s == 2 {
    // The action and its legality rules, shown on three example moves. The big
    // figure carries the legality (per-panel verdicts); the action line sits
    // below it.
    v(0.2em)
    align(center, image("figures/rl/rl-rules.pdf", height: 74%))
    v(0.6em)
    align(
      center,
      text(size: 0.9em)[
        Akcia je dvojica vrcholov $(u, v)$: ak hrana chýba, pridá sa, inak sa odoberie
      ],
    )
  } else {
    // Goal + reward. Invalid moves are impossible by design (the action mask
    // only offers legal moves), so there is no invalid-move penalty to mention.
    set align(horizon)
    align(center)[
      #text(size: 1.15em)[Cieľ: *k-regulárny graf s obvodom aspoň g*]
      #v(1.0em)
      #text(size: 0.92em)[
        Veľká odmena za hotový graf, priebežne malá odmena za každý vrchol,
        ktorý dosiahne stupeň k, a za pridávané hrany
      ]
      #v(0.8em)
      #text(
        size: 0.85em,
        fill: muted,
      )[Agent dostane pevný počet vrcholov a aktivuje ich hranami]
    ]
  }
})


== Reinforcement learning: výsledky

#v(1fr)
#align(center)[
  #text(
    size: 1.1em,
  )[Priame RL zvládlo len najmenšie ciele: (5,3), (3,5), (6,3) a podobné]
  #v(1.0em)
  #text(
    size: 1.0em,
    fill: muted,
  )[Úspešnosť 8 %, a zo všetkých metód bolo najpomalšie]
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


// Two learned ways we drove the voltage search, one per scene:
//   1) tabu/beam SEARCH over voltage assignments, optionally scored by a GNN
//      girth/tabu predictor (vs the no-GNN algebraic baseline),
//   2) an RL POLICY (voltage_rl) that assigns voltages edge by edge.
#slide(title: [Voltage lifts a GNN], repeat: 2, self => {
  let s = self.subslide
  if s == 1 {
    grid(
      columns: (1.5fr, 1fr),
      gutter: 2em,
      align: horizon,
      align(center)[
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
          node((1, -1), align(center)[priraď napätia], corner-radius: 5pt),
          node((0, 0), align(center)[spočítaj lift], corner-radius: 5pt),
          node((1, 1), align(center)[over obvod], corner-radius: 5pt),
          node((1, 2), align(center)[hotovo], corner-radius: 5pt),
          node(
            (2, 0),
            align(center)[GNN ohodnotí\ napätia],
            corner-radius: 5pt,
            stroke: 1.5pt + ink,
            fill: dgsteelfill,
          ),
          edge((1, -2), (1, -1), "-|>", stroke: 1.2pt + muted),
          edge((1, -1), (0, 0), "-|>", stroke: 1.2pt + muted),
          edge((0, 0), (1, 1), "-|>", stroke: 1.2pt + muted),
          edge((1, 1), (1, 2), "-|>", [$>=$g], stroke: 1.2pt + muted),
          edge((1, 1), (2, 0), "-|>", [$<$g], stroke: 1.2pt + muted),
          edge((2, 0), (1, -1), "-|>", stroke: 1.2pt + muted),
        )
      ],
      text(size: 0.92em)[
        Napätia hľadáme prehľadávaním

        #v(0.4em)
        GNN vie ohodnotiť kandidátov a navádzať hľadanie

        #v(0.6em)
        #text(
          fill: muted,
        )[Algebraické 63 %, s GNN predikciou 59 %. GNN tu nepomohlo.]
      ],
    )
  } else {
    grid(
      columns: (1.5fr, 1fr),
      gutter: 2em,
      align: horizon,
      align(center)[
        #set text(size: 13pt)
        #diagram(
          node-stroke: 1.3pt + linecol,
          node-fill: white,
          node-inset: 8pt,
          spacing: (4.6em, 3.4em),
          node(
            (0, 0),
            align(center)[RL politika\ (GNN)],
            corner-radius: 5pt,
            stroke: 1.5pt + ink,
            fill: dgsteelfill,
          ),
          node(
            (1, 0),
            align(center)[napätie na\ ďalšiu hranu],
            corner-radius: 5pt,
          ),
          node((2, 0), align(center)[po lifte:\ obvod], corner-radius: 5pt),
          edge(
            (0, 0),
            (1, 0),
            "-|>",
            [akcia],
            label-side: left,
            stroke: 1.2pt + muted,
          ),
          edge((1, 0), (2, 0), "-|>", stroke: 1.2pt + muted),
          edge(
            (2, 0),
            (0, 0),
            "-|>",
            [odmena = obvod],
            bend: 34deg,
            stroke: 1.2pt + muted,
          ),
        )
      ],
      text(size: 0.92em)[
        RL politika priraďuje napätia po jednej hrane

        #v(0.4em)
        Odmena = obvod grafu po lifte

        #v(0.6em)
        #text(
          fill: muted,
        )[Najširší dosah zo všetkých metód: 84 %]
      ],
    )
  }
})


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
      #text(
        fill: muted,
      )[Krok bol rýchlejší, no úspešnosť nižšia: 51 % oproti 75 % bez GNN]
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
        edge((2, 0), (2, 1), [správne], "-|>", stroke: 1.1pt + muted),
        edge((2, 0), (0, 0), [nesprávne], "-|>", stroke: 1.1pt + muted),
      )
    ]
  ],
  [
    #text(size: 0.92em)[
      GNN navrhuje ktoré hrany vytvoriť.

      #pause
      #v(0.5em)
      #text(
        fill: muted,
      )[Bez excízie 2,06× Moore, s ňou 1,51×. Naučené zošitie zatiaľ neukázalo zisk]
    ]
  ],
)


// Forge as a stepped story across three scenes:
//   1) each method's weakness (why a single method is not enough),
//   2) the pipeline with a REAL graph flowing through it (reusing the refine and
//      excision figures so it ties back to those slides),
//   3) the defective-fraction gate that routes near-misses.
#slide(title: [Forge], repeat: 3, self => {
  let s = self.subslide

  if s == 1 {
    // WEAKNESS framing: one card per method (strength + weakness).
    set align(horizon)
    let card(name, good, bad) = box(
      stroke: 1.2pt + linecol,
      radius: 6pt,
      inset: 12pt,
      width: 100%,
      stack(
        dir: ttb,
        spacing: 0.7em,
        text(weight: 800, size: 1.0em, name),
        text(size: 0.85em)[#text(fill: dgsteel)[#sym.checkmark] #good],
        text(size: 0.85em, fill: muted)[#sym.crossmark #bad],
      ),
    )
    v(0.3em)
    grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 1.2em,
      align: horizon,
      card([Voltage lifts], [veľký dosah], [grafy ~2× Moore]),
      card([Klasické hľadanie], [veľkosť klietky], [úzky dosah]),
      card([Refinement + Excision], [opraví a zmenší], [samy nič nepostavia]),
    )
    v(1.0em)
    align(center, text(
      size: 0.9em,
    )[Každá metóda je silná inde. Forge ich zreťazí, nech každá robí len to svoje.])
  } else if s == 2 {
    // PIPELINE with a real graph flowing through (reused vector figures).
    let stage(img, name, cap) = stack(
      dir: ttb,
      spacing: 0.4em,
      text(weight: 800, size: 0.85em, name),
      image(img, height: 52%),
      text(size: 0.78em, fill: muted, cap),
    )
    let arrow = align(horizon, text(size: 1.7em, fill: dgbronze)[→])
    v(0.4em)
    grid(
      columns: (1fr, auto, 1fr, auto, 1fr),
      gutter: 0.7em,
      align: center + horizon,
      stage(
        "figures/refine/refine-0.pdf",
        [Producer],
        [lift, často skoro-dobrý],
      ),
      arrow,
      stage("figures/refine/refine-final.pdf", [Refinement], [obvod = g]),
      arrow,
      stage(
        "figures/excision/exc-petersen.pdf",
        [Excision],
        [zmenšený ku klietke],
      ),
    )
  } else {
    // GATE: defective-fraction routing of near-misses, with the definition.
    // The 15pt size is scoped to the diagram box only, so the formula and the
    // caption below keep the full slide text size.
    set align(horizon)
    align(center)[
      #box[
        #set text(size: 15pt)
        #diagram(
          node-stroke: 1.3pt + linecol,
          node-fill: white,
          node-inset: 10pt,
          spacing: (7em, 3.0em),
          node((0, 0), align(center)[skoro-dobrý\ lift], corner-radius: 5pt),
          node(
            (1, 0),
            align(center)[defective\ fraction $<= tau$ ?],
            corner-radius: 5pt,
            stroke: 1.5pt + ink,
            fill: dgsteelfill,
          ),
          node((2, -0.7), align(center)[refinement], corner-radius: 5pt),
          node((2, 0.7), align(center)[zahodiť], corner-radius: 5pt),
          edge((0, 0), (1, 0), "-|>", stroke: 1.2pt + muted),
          edge(
            (1, 0),
            (2, -0.7),
            "-|>",
            [áno],
            label-side: left,
            stroke: 1.2pt + muted,
          ),
          edge(
            (1, 0),
            (2, 0.7),
            "-|>",
            [nie],
            label-side: right,
            stroke: 1.2pt + muted,
          ),
        )
      ]
      #v(0.9em)
      $ "defective fraction" = (|E_"def"|) / (|E|) $
      #v(0.2em)
      #text(
        size: 0.82em,
      )[$E_"def"$ = hrany ležiace na cykle kratšom ako $g$, teda tie, čo kazia obvod]
      #v(0.55em)
      #text(
        size: 0.9em,
      )[Brána pustí na opravu len lifty s malým podielom chybných hrán]
    ]
  }
})


// Results as two full-page, text-free figure scenes: (1) the wide all-methods
// coverage table coloured by target size with a colorbar legend, (2) the
// reach x size performance plane.
#slide(title: [Výsledky], repeat: 2, self => {
  let s = self.subslide
  set align(center + horizon)
  if s == 1 {
    image("figures/results/fig-results.pdf", width: 100%)
  } else {
    image("figures/results/fig-tradeoff.pdf", height: 86%)
  }
})


// =============================================================================
//  QUESTION SLIDES — supervisor + opponent. For now ONLY the question text is
//  shown (answers to be added later). One question per slide, before the záver.
// =============================================================================

#slide(title: [Otázka školiteľa (1)], repeat: 2, self => {
  let s = self.subslide
  if s == 1 {
    set align(horizon)
    text(size: 1.05em)[
      Čakal som, že kombinovaná metóda Forge dosiahne najlepšie výsledky. Podľa
      tabuliek vychádza najlepšie A\* pri hľadaní najmenšieho príkladu (k,g)-grafu a
      voltage-RL pri hľadaní aspoň nejakého príkladu. Čo zapríčinilo dané výsledky?
    ]
  } else {
    v(0.1em)
    align(center, image("figures/results/fig-frontier.pdf", height: 60%))
    v(0.3em)
    align(center, text(size: 0.74em)[
      A\* je presné len na ľahkých cieľoch. Voltage-RL má dosah, no grafy okolo 2× Moore.
      Na *ťažkej hranici*, kde A\* nič nevráti, je *Forge* najmenší platný graf (okolo
      1,23× Moore). Nevyhráva stĺpec, lebo stĺpce sa vyhrávajú na ľahkých cieľoch.
    ])
  }
})

#slide(title: [Otázka školiteľa (2)], repeat: 2, self => {
  let s = self.subslide
  if s == 1 {
    set align(horizon)
    text(size: 1.05em)[
      Prečo ste metódy zastavili po 60 sekundách? Ako vieme, či modely so strojovým
      učením nepotrebovali viac času na konvergenciu?
    ]
  } else {
    v(0.1em)
    align(center, image("figures/results/fig-time-budget.pdf", height: 60%))
    v(0.3em)
    align(center, text(size: 0.74em)[
      Začal som na 5 minútach, no výsledok bol vždy rovnaký: čo sa podarí, podarí sa do
      minúty, inak ani za päť. Preto 1 minúta, *rovnaký rozpočet pre všetky metódy*.
      Učenie modelov je *offline pri tréningu*, nie počas hľadania.
    ])
  }
})

#slide(title: [Otázka oponenta (1)], repeat: 2, self => {
  let s = self.subslide
  if s == 1 {
    set align(horizon)
    text(size: 1.05em)[
      V práci ukazujete, že štandardné GNN nedokážu spoľahlivo predikovať obvod. Aké
      štrukturálne alebo pozičné príznaky by ste do modelu pridali, aby lepšie
      zachytil globálnu cyklickú štruktúru grafu?
    ]
  } else {
    v(0.1em)
    align(center, image("figures/gnn/fig-girth-receptive.pdf", height: 60%))
    v(0.2em)
    align(center, text(size: 0.74em)[
      Problém je *reprezentačný*: MPNN nevie spočítať obvod (Garg, 2020), 1-WL nevidí
      dlhé cykly. Riešením je pridať príznaky: *(1) RWSE*, *(2) počty cyklov (GSN)*,
      *(3) spektrálne kódovanie*. Náš model *Loopy* už ide týmto smerom.
    ])
  }
})

#slide(title: [Otázka oponenta (2)], repeat: 2, self => {
  let s = self.subslide
  if s == 1 {
    set align(horizon)
    text(size: 1.05em)[
      Pipeline Forge kombinuje voltage producer, refinement a excision. Ako by ste
      systematicky ladili prah defective fraction a ako by ste experimentálne
      odlíšili prínos diverzity kandidátov od kvality naučenej politiky?
    ]
  } else {
    v(0.1em)
    align(center, image("figures/results/fig-tau-policy.pdf", height: 60%))
    v(0.2em)
    align(center, text(size: 0.74em)[
      Prah $tau$ je kompromis výťažok/náklad, ladený sweepom cez validačné ciele
      (zvolené 0,2). Prínos *diverzity* a *politiky* oddelíme 2×2 abláciou. Doterajšie
      dáta: zisk je zo *šírky hľadania*, nie z naučenej politiky.
    ])
  }
})


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

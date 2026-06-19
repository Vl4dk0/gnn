// =============================================================================
//  Bachelor's thesis defense — 12-minute deck
//  "Machine Learning for Generation of Graph of Given Degree and Girth"
//  Vladimir Jancar — Comenius University Bratislava, 2026
//
//  Build:   typst compile slides.typ slides.pdf
//  Author:  typst watch slides.typ slides.pdf   (live preview)
//
//  Framework: Touying + metropolis theme (see typst_guide.md for the why/how).
//  Figures live in ./figures/ (paths are relative to THIS file).
// =============================================================================

#import "@preview/touying:0.7.4": *
#import themes.metropolis: *

// Handout mode collapses #pause subslides to their final state (one page per slide).
// Enable for previews/printing with:  typst compile --input handout=true slides.typ
#let handout = sys.inputs.at("handout", default: "false") == "true"

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  ..config-common(handout: handout),
  footer: self => self.info.institution,
  // Calm academic-blue palette (muted, defense-appropriate):
  config-colors(
    primary: rgb("#1f4e79"),
    primary-light: rgb("#cfd8e3"),
    secondary: rgb("#23373b"),
    neutral-lightest: rgb("#ffffff"),
    neutral-dark: rgb("#23373b"),
    neutral-darkest: rgb("#0f1c1f"),
  ),
  config-info(
    title: [Machine Learning for Generation of\ Graphs of Given Degree and Girth],
    subtitle: [Bachelor's Thesis Defense],
    author: [Vladimír Jančár — Supervisor: Mgr. Ján Pastorek],
    date: datetime.today().display("[month repr:long] [day], [year]"),
    institution: [Comenius University Bratislava · FMFI],
    // logo: image("figures/uk-logo.svg", height: 1.5cm),  // add a real crest if available
  ),
)

// Fonts are optional; Touying falls back gracefully if these are missing.
#set text(size: 22pt)
// Slides don't need "Figure N:" labels; also avoids a counter-convergence warning
// caused by figures living on slides with #pause subslides.
#set figure(numbering: none)

// -----------------------------------------------------------------------------
#title-slide()

// -----------------------------------------------------------------------------
#slide(title: [Outline])[
  #components.adaptive-columns(outline(title: none, indent: 1em))
  #speaker-note[~12 min. Red thread: "learning guides, exact algorithms decide."]
]

// =============================================================================
= The Problem
// =============================================================================

== $(k, g)$-graphs and cages

A *$(k,g)$-graph* is a graph that is

- *$k$-regular* — every vertex has degree exactly $k$, and
- has *girth $>= g$* — its shortest cycle has length at least $g$.

#pause

The smallest such graph is a *cage*, of order $n(k,g)$.

#pause

A classical *extremal problem*: best known orders are still improved today by a
mix of hand constructions and large computer searches.

#speaker-note[~1:00 — Keep it concrete: draw/point to the Petersen graph as the (3,5)-cage.
The whole talk is about building these as small as possible.]

== Why ML cannot be a black box here

#grid(
  columns: (1.15fr, 1fr),
  gutter: 1.2em,
  [
    Two defining constraints, two very different difficulties:

    - *Degree* is *local* — a one-hop count.
    - *Girth* is *global* — needs paths that leave and return.

    #pause

    Hard limits:
    - Message-passing GNN expressivity $<=$ 1-WL.
    - *Girth is provably not computable by an MP-GNN* (Garg et al.).
    - RL only works where the move space already carries structure.
  ],
  [
    #figure(
      image("figures/fig02-receptive-field.png", width: 100%),
      caption: [Depth = how far information travels; a global property like girth
        slips past a fixed receptive field.],
    )
  ],
)

#speaker-note[~1:30 — This is the crux: don't hand ML the whole job. The impossibility
result for girth is the strongest single argument and likely a question target.]

== Reframing the question

#align(center + horizon)[
  #text(size: 28pt)[
    Not *"produce a cage."*

    #v(0.4em)
    Rather: *learning guides where the search looks,*\
    *exact algorithms decide which graphs are valid.*
  ]
]

#pause

#align(center)[A learned component *ranks / proposes*; an exact check *accepts*.]

#speaker-note[~0:30 — State the thesis in one sentence. Everything after is a variation
on where to draw the learning-vs-exact boundary.]

// =============================================================================
= Diagnostic: What Can a GNN Learn?
// =============================================================================

== Degree is learnable, girth is not

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    Probe the two defining constraints directly as per-node predictions
    (trained on fresh random graphs, exact labels):

    - *Degree* — *solved*: SAGE with *4,900 params* reaches *accuracy 1.000*.
    - *Min-cycle / girth* — *resists every architecture*: best is the
      cycle-aware *Loopy GNN at 0.510* (mislabels ~half).

    #pause

    *Consequence (drives the whole design):*
    degree gets *built into the construction*;
    girth gets handled by *explicit exact search*.
  ],
  [
    #figure(
      image("figures/fig10-degree-vs-girth.png", width: 100%),
      caption: [The split: degree is recovered exactly; the minimum cycle is not.],
    )
  ],
)

#speaker-note[~2:00 — The empirical justification for the architecture. Tables 1-2 are
backup. Land the line: "this is why degree is enforced and girth is searched."]

// =============================================================================
= Construction Methods
// =============================================================================

== A ladder of increasing structure

#align(center + horizon)[
  #grid(
    columns: (1fr,),
    row-gutter: 0.7em,
    [*1. Direct RL* — edit edges under a learned policy #h(0.4em) _(least structure)_],
    [*2. Voltage lifts* — regularity built in, only girth searched],
    [*3. Refinement & Excision* — repair near-misses, then shrink],
    [*4. Forge* — compose a producer with refinement + excision #h(0.4em) _(most structure)_],
  )
]

#speaker-note[~0:30 — Signpost slide. Each method fixes the previous one's flaw.]

== 1. Direct RL

State = partial graph · Action = add/remove a vertex pair · trained with *PPO*.

- Illegal moves are *pruned* (degree cap, would-be short cycle, disconnection).
- Needs a *curriculum* (start at $(3,5)$, unlock harder targets) —
  *without it the agent never even makes a $(3,5)$-graph*.
- Needs a *potential-based shaped reward* to learn at all.

#pause

*Limits:* regularity is not guaranteed, and the action space grows
*quadratically* $binom(n, 2)$ → does not scale. → motivates structure.

#speaker-note[~1:00 — The honest "too little structure" baseline. Sets up voltage lifts.]

== 2. Voltage lifts

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    Idea: *$k$-regularity becomes structural* — every lift of a $k$-regular
    base graph is $k$-regular. Only choice that affects girth: the
    *voltage assignment* on a small base.

    #pause

    Read girth *off the base* via net voltages:
    $ "girth(lift)" = min_W abs(W) dot "ord"(s) $

    *Gauge* freedom zeroes spanning-tree voltages → search only non-tree
    edges (e.g. $K_4$: *3 voltages instead of 6*).
  ],
  [
    #figure(
      image("figures/fig03-voltage-lift-k4z3.png", width: 92%),
      caption: [$K_4$ over $ZZ_3$: a 12-vertex, 3-regular connected lift.],
    )
  ],
)

#speaker-note[~1:30 — Widest reach but ~2x Moore bound. fig04 (triangle wraps to a
9-cycle) is the backup for "girth on the base".]

== 3. Refinement & Excision

#grid(
  columns: (1fr, 1.05fr),
  gutter: 1.2em,
  [
    *Refinement* — fix a near-miss lift without discarding it:
    degree-preserving *2-/3-swaps* via tabu search, weighting short cycles
    exponentially. Learned evaluator *ranks* swaps; exact check decides.

    #pause

    *Excision* — make a valid graph *smaller*:
    remove a tree of *Moore radius* $floor((g-1)/2)$, then stitch deficient
    vertices at distance $>= g-1$. Learned $(k,g)$-independent repair policy
    + *exact backtracking* fallback.
  ],
  [
    #figure(
      image("figures/fig09-dodeca-petersen.png", width: 100%),
      caption: [Excision: dodecahedral graph → *Petersen graph*, the
        $(3,5)$-cage — half the order.],
    )
  ],
)

#speaker-note[~1:30 — fig09 is the hero visual. fig08 (tree removal) and fig05/07
(swaps) are backups.]

== 4. Forge

Compose the methods so each does only what it does well:

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  align: center + horizon,
  [*Producer*\ (voltage method)\ → _reach_],
  [*Refinement*\ rescue near-misses\ → _validity_],
  [*Excision*\ shrink to cage\ → _size_],
)

#pause

- Near-miss hand-off via the *defective fraction $<= tau$* (scales with size).
- Ordering is forced; stages run *round-robin*; producer is *interchangeable*.

#speaker-note[~1:00 — "Reach from one stage, size from another." Producer pluggability
enables the ablation on the results slide.]

// =============================================================================
= Results
// =============================================================================

== Coverage vs. size, across 22 targets

- *voltage-rl* has by far the *widest reach* — the only method solving
  $(3,9), (3,10), (4,8), (6,6)$, and $(6,5)$ at 57% of seeds.
- *Classical* methods are *size-optimal but narrow*; direct RL is weakest.
- *No method* solves $(4,7), (5,7), (7,5)$ within the 60 s budget.

#pause

- *Size:* the voltage family lands at *~2× the Moore bound*; classical methods
  are smallest where they work at all.

#speaker-note[~1:30 — This is the coverage-vs-size trade-off. Full Tables 3-4 are backup.]

== Forge: the balanced answer

- Tabu-based producers: *~0.46 success, ~1.23× Moore bound* —
  and *meet the cage exactly on the smallest targets*.
- The RL producer is the *strongest standalone* policy yet the *weakest inside
  Forge*: the pipeline wants a *diverse* stream of near-cage lifts.

#pause

- *Ablation:* remove excision → size *1.51 → 2.06* (excision is what shrinks);
  remove refinement → success *0.31 → 0.62* but larger graphs.

#speaker-note[~1:00 — Tables 5-6. Punchline coming: learned rarely beats non-learned.]

// =============================================================================
= Conclusion
// =============================================================================

== What we learned

- *Degree* essentially *solved* (tiny model, perfect); *girth resists every
  architecture* → handled by exact search.
- Dominant pattern: a *coverage-vs-size trade-off*; *Forge is the most balanced*
  (~1.23× Moore, keeps much of the reach).

#pause

- *The honest finding:* *learned components rarely beat their non-learned
  counterparts* — and that *is* the result.
- Learning earns a place *beside* structured search: the *learned excision
  repair*, and *RL as a diverse lift source* in Forge.

#align(center)[*Learning guides where the search looks; exact algorithms decide validity.*]

#speaker-note[~1:00 — End on the contribution, not "thank you". This is what the
committee reads during questions.]

== Future work

- *Co-adapt* Forge's three stages (currently tuned in isolation).
- *Learn* the excision root and depth; train the producer for *diversity*.
- A *girth-specialised, degree-agnostic* model (degree is free under lifts).
- Ongoing parallel *record search* on open targets $(8,5),(9,5),(10,5),(11,6)$.

#speaker-note[~0:30 — Record search: no record-beating graph found yet; honest and ongoing.]

#focus-slide[
  Thank you — questions welcome.
]

// =============================================================================
//  BACKUP SLIDES (after the thank-you; do not count toward the 12 minutes)
// =============================================================================

= Backup

== Girth on the base graph ($K_4 / ZZ_3$)

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    Net voltage of a reduced closed walk $W$:
    - net $=$ identity → cycle of length $abs(W)$;
    - else → cycle of length $abs(W) dot "ord"(s)$.

    $ "girth(lift)" = min_W abs(W) dot "ord"(s) $

    In $K_4 / ZZ_3$: triangles (order 3) wrap into a *9-cycle*; the 4-cycle
    survives → *girth 4*. No assignment reaches girth 5.
  ],
  [
    #figure(
      image("figures/fig04-voltage-girth.png", width: 100%),
      caption: [Base cycles lift to short cycles; the surviving 4-cycle fixes girth.],
    )
  ],
)

== Excision: tree removal + stitching

#figure(
  image("figures/fig08-excision-tree.png", width: 62%),
  caption: [A depth-2 BFS tree removed from the dodecahedral graph leaves deficient
    boundary vertices, reconnected only at distance $>= g-1$.],
)

== Degree-preserving swaps

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    #figure(image("figures/fig05-2swap.png", width: 95%), caption: [2-swap: rewire two
      disjoint edges; degrees preserved, a triangle can be broken.])
  ],
  [
    #figure(image("figures/fig07-3swap.png", width: 95%), caption: [3-swap: reaches
      configurations no 2-swap can.])
  ],
)

== Message passing

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    $ bold(h)_v^((k)) = phi.alt(
        bold(h)_v^((k-1)), #h(0.2em)
        plus.big_(u in cal(N)(v)) psi(bold(h)_v^((k-1)), bold(h)_u^((k-1)))
      ) $

    Same local update at every vertex → size-independent and
    permutation-equivariant. Sum aggregation keeps counts (degree);
    normalized aggregation hides them.
  ],
  [
    #figure(image("figures/fig01-message-passing.png", width: 100%),
      caption: [One message-passing layer: aggregate then update.])
  ],
)

== Evaluation setup

- Trained on the *PERUN* supercomputer (TUKE).
- *60-second* budget per attempt; single node (256 cores), *128 parallel workers*
  (256 oversubscribes the cores → collapse).
- Architectures probed: GCN, GraphSAGE, GIN, GINE, GPS (graph transformer),
  Loopy (cycle-aware).
- Property predictors trained on *Erdős–Rényi* graphs (fresh per step → no
  memorization), with exact labels.

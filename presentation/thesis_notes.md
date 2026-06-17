# Thesis Analysis & Defense Prep Notes

**Thesis:** *Machine Learning for Generation of Graph of Given Degree and Girth*
**Author:** Vladimír Jančár — Comenius University Bratislava, 2026
**Supervisor:** Mgr. Ján Pastorek (advisor RNDr. Jozef Šiška, PhD.)
**Source:** `/home/user/gnn/presentation/thesis_md/thesis.md` (1140 lines)
**Code:** github.com/Vl4dk0/gnn • Site: vladimirjancar.sk • Compute: PERUN supercomputer (TUKE)

---

## 0. One-line thesis statement

Can GNNs / RL help build small **(k,g)-graphs** (k-regular, girth ≥ g) and **cages** (the smallest such graphs)? Neither works as a black box: a message-passing GNN provably *cannot detect girth* (bounded by 1-WL; Garg et al.), and RL only works on extremal-graph search where moves already carry structure. So the **real question is whether learning can *guide where the search looks* while exact algorithms decide which graphs are valid** — "learning ranks/proposes, exact check accepts."

---

## 1. PER-CHAPTER SUMMARY

### Ch 1 — Introduction (p.1)
- Two ML paradigms motivate the work: **GNNs** (learn structure representations, e.g. AlphaFold) and **RL** (search constructions directly, e.g. Wagner [1] disproving combinatorics conjectures).
- Target objects: **(k,g)-graphs** = every vertex degree k, every cycle length ≥ g. Smallest = **cage**, order n(k,g); a classical extremal problem still improved by hand + computer search [3].
- **Why neither transfers as a black box**: MP-GNN expressivity ≤ 1-WL [4], and girth is provably beyond it [5]; RL succeeds only when the move space has structure.
- Reframed question: not "produce a cage by itself" but **"guide where the search looks while exact algorithms verify validity."**
- Recurring lesson from successful cage constructions: **constrain the search space mathematically, then explore only the remaining freedom.** Thesis follows a progression of increasing structure.

### Ch 2 — Background and Definitions (p.2-5)
- **2.1 Graphs:** degree, neighborhood, k-regular, walk/path/cycle, reduced closed walk, **girth** (= shortest cycle, ∞ if acyclic), **active subgraph** (non-isolated vertices).
- **2.2 (k,g)-graphs & cages:** (k,g)-cage = smallest k-regular graph of girth g; order n(k,g). Search tests `girth(G) ≥ g`. **Moore bound** = theoretical lower bound on order + practical target size; Moore graphs (meeting it) are rare; practical problem = close gap between Moore bound and best published upper bounds.
- **2.3 Groups:** group axioms, abelian, order, element order; constructions use finite groups, mostly **cyclic Z_n**.
- **2.4 Tabu search:** local-search metaheuristic; **tabu list** = short-term memory forbidding recent moves so it can escape local minima without oscillating.
- **2.5 GNNs:** same local update at every vertex → size-independent + permutation-equivariant. Message passing (aggregate + update). Key params: #layers (receptive field), hidden dim, aggregation rule (sum keeps counts; normalized hides them). **Local properties (degree) easy, global (girth) hard.** Grohe [4]: MP-GNN ≈ color refinement / 1-WL. Architectures used: **GCN, GraphSAGE, GIN, GINE (edge-aware), GPS (graph transformer), Loopy GNN (cycle-aware)**.
- **2.6 RL:** agent/state/action/reward/policy/episode/return. Policies trained with **PPO** [13] (clipped policy-gradient, stable updates).

### Ch 3 — Related Work (p.6-8)
- **3.1 Classical/computational cage construction:** Dynamic Cage Survey [2]. Successful methods **impose structure first, search the rest**: incidence graphs of projective planes/generalized quadrangles (girth 6/8), generalized hexagons (girth 12), Cayley graphs, **Ramanujan graphs** [14]. **Voltage lifts** [15] reduce O(n²) adjacency choices to a small voltage assignment on the base. Local search (hill climbing, tabu over edge swaps) and **excision** [16] (Balaban-style tree removal). Exoo et al. [3] chain voltage-lifts + tabu + excision → **eleven new upper bounds in one pipeline**. Lesson: improvements come from *chaining* structured methods. These are the **non-learning baselines** — a learned method must beat a *structured* search, not just random edge editing.
- **3.2 GNNs and their limits:** NeuroSAT [17] = NN guides symbolic search without replacing the solver. Garg/Jegelka/Jaakkola [5] **prove MP-GNNs cannot compute girth/diameter** — exactly the cage regime (high girth, near-vertex-transitive, locally tree-like) where L-layer GNNs collapse to near-identical embeddings. Two responses: **subgraph/motif-count features** [18] and **cycle-aware hierarchies (Loopy)** [12]. Degree can be enforced by construction → motivates **learned scoring/move-ranking, not replacing verification.** Wagner [1] / RLGT [19] / Freire et al. [20] (RL on LDPC code components where action space carries structure).

### Ch 4 — Predicting Graph Properties with GNNs (p.9-10)
- Two controlled probes for the two defining constraints: **degree** (local one-hop count) and **minimum-cycle** (shortest cycle through each vertex, 0 if none — global, needs paths that leave and return).
- **4.1 Data:** dynamically generated random **Erdős–Rényi** graphs (fresh per step → no memorization). Labels by exact algorithms. Input features intentionally trivial (constant 1-D or 4-D: normalized index + random coords/scalar) → force learning from structure.
- **4.2 Implementation:** exact min-cycle label (Algorithm 1): for each incident edge, remove it, find shortest path back to v → cycle of that length + 1; min over neighbors; 0 if none.

### Ch 5 — Construction Methods / Direct RL (p.11-15)
- Construction is a **search problem** not a single prediction: hold a partial object, choose a modification, check satisfiability, repeat.
- **5.1 From guided search to RL:** supervised A*-style heuristic fails because labeling partial states is ill-posed (negatives are hard — a "bad" partial graph may still be extendable). **RL learns from complete construction attempts** instead of a dataset → "a learned version of heuristic search." Action pruning guarantees *legality* but **not completion** (unlike algebraic constructions).
- **5.2 Direct RL:** state = partial graph; action = pick a vertex pair (add if absent, remove if present); trained with **PPO**. No precomputed dataset needed.
- Pruned (illegal) actions never shown to agent: can't add if endpoint at degree k; can't add if it closes a cycle < g; can't remove if it disconnects active subgraph; can't remove if it drops below Moore-bound count.
- **5.3 Curriculum + reward:** order pairs by Moore bound, unlock next after solving ≥ 4 of last 8 episodes; **start at (3,5)**. **Without curriculum the agent never even produces a (3,5)-graph; with it, reaches (3,5) and (3,6).** **Potential-based shaping reward** [21]: per-step Φ(s')−Φ(s) rewards moving toward k-regularity on Moore-bound vertices + large terminal bonus. (Algorithms 2 & 3.)
- **5.4 Limits:** regularity not guaranteed (must be learned/enforced); action space grows **quadratically** (C(n,2)). → motivates voltage lifts.

### Ch 6 — Voltage Graph Lifts (p.16-22)
- Core idea: **k-regularity becomes structural** — every lift of a k-regular base is k-regular. Control reduces to choosing **group labels on edges of a small base graph**.
- **6.1 Three (and only three) choices:** finite group Γ (sets #copies = |Γ|), edge orientation (bookkeeping; reversing arc ↔ inverse voltage), and **voltage assignment α (the only thing affecting girth)**. Vertices = |V(B)|·|Γ|; degree inherited.
- **6.2 Building the lift:** vertex set V(B)×Γ; arc u→v voltage a gives edge (u,h)–(v,ha). Uses **right multiplication**. Example **K₄ over Z₃ = 12 vertices, 3-regular, connected** (Figure 3). **Regularity proof:** each of k arcs at u gives exactly one lift edge per layer.
- **6.3 Computing girth on the base (no lift built):** net voltage of a reduced closed walk W; if net = identity → cycle of length |W|; else → cycle of length |W|·ord(s). **girth(lift) = min over W of |W|·ord(s).** K₄/Z₃ example: triangles (net 1, order 3) wrap into a 9-cycle; the 4-cycle 0–2–3–1 (net 0) survives → **girth 4** (no assignment on K₄/Z₃ reaches girth 5) (Figure 4). Cheap exact cost: count reduced closed walks shorter than g with identity net voltage; zero ⟺ girth ≥ g.
- **Reducing the search space (gauge):** exhaustive search scales |Γ|^m (m = base edges). **Gauge transformation** (relabel copies of a vertex) can't change girth → can zero all **spanning-tree** edges (|V|−1 of them); only **non-tree edges** (m − |V| + 1) carry real freedom. K₄: 6 edges → only **3 voltages searched** instead of 6.
- **6.4 Search methods:** exhaustive (only when |Γ|^m small), random (baseline, sometimes works), **tabu search** (fix tree voltages, change one non-tree at a time, minimize #short identity-voltage walks, [3]-style), **GNN-guided beam search** (girth-predictor scores partial assignments — predictor detailed Ch 9), **meta-search** over base graphs (dumbbell, 4-node cubic, prism, Petersen-like) and groups (cyclic, dihedral, direct/semidirect products). **RL voltage env:** one assignment = one episode, episode length = #base edges (much shorter than direct editing); uses **GINE** encoder or **GPS+GINE** (needs edge features), actor head over group elements, critic for PPO.
- **6.5 Implementation:** Algorithm 4 (lift_girth via walk enumeration on base), Algorithm 5 (tabu over free voltages).

### Ch 7 — Refinement and Excision (p.23-30)
- Two methods that act on an **already-constructed k-regular graph**.
- **7.1 Edge-swap refinement:** a lift may fall short of girth g (few short cycles survive); rather than discard, **edit a few edges to remove short cycles while preserving degree.**
  - **2-swap** (Figure 5): two disjoint edges A–B, C–D → A–D, B–C; degrees preserved; can break a triangle (Figure 6).
  - **3-swap** (Figure 7): three disjoint edges rewired; reaches configs no 2-swap can.
  - **Search:** cost = weighted count of cycles shorter than g, **shorter cycles weighted exponentially more** (triangle outweighs any longer cycle); reaches 0 exactly at girth target. **Tabu search**; escalate 2-swap → 3-swap when progress flattens (3-swaps O(|E|³) vs 2-swaps O(|E|²), so occasional).
  - **Learned evaluator:** GNN embeds graph **once per iteration**, predicts cost change for each candidate swap from its 4 endpoint embeddings + graph summary, **batched**. Uses cycle-count features + random-walk positional encodings (break symmetry on regular graphs). **Estimate only ranks — exact verification decides.** Same division of labor throughout.
- **7.2 Excision:** make a valid graph **smaller** → push order toward cage/Moore bound. Balaban [22] originated it ((3,12)-cage → (3,11)-cage). **Removing a tree:** BFS tree to depth d = ⌊(g−1)/2⌋ (the **Moore radius**, deepest acyclic radius), delete it; boundary vertices drop below degree k = **deficient vertices** (Figure 8: dodecahedral, depth-2 tree of 10 vertices). **Stitching:** reconnect deficient vertices, edge allowed only at distance ≥ g−1 (keeps girth) (Figure 9: dodecahedral → **Petersen graph = (3,5)-cage, half the order**). **Search:** try each vertex as root, accept first valid smaller graph, restart; bottoms out at a cage. **Learned policy** = the hard combinatorial core: GNN actor-critic (PPO) picks next girth-safe edge; vertex features = degree, deficiency, target k & g, short-cycle counts, RW positional encodings. **Policy is (k,g)-independent** (k,g as features → one policy serves all targets). On dead end → **exact backtracking takes over.**
- **7.3 Implementation:** Algorithm 6 (refinement tabu w/ learned ranking), Algorithm 7 (girth-safe repair: greedy policy + backtracking fallback).

### Ch 8 — Forge (p.31-33)
- **Composition** so each method does only what it does well: producer supplies **reach**, refinement **rescues** near-miss lifts, excision recovers **size-optimality**.
- **Three-stage cooperative pipeline:** (1) **producer** = any Ch 6 voltage method (algebraic tabu, two GNN-guided variants, or RL voltage policy — interchangeable); (2) **refinement** (Ch 7 2-/3-swaps); (3) **excision**.
- **8.1 When to refine a lift:** be **generous** about what the producer passes on (many lifts come out 1–2 below g). A lift already at girth g → straight to excision. For near-misses, **girth alone is a poor judge** (one triangle drives girth to 3). Use **defective fraction** = (# edges on a cycle < g) / total edges, in [0,1]. Refine only if fraction ≤ threshold **τ**; drop if above (refinement repairs, doesn't rebuild). Fraction scales with size → same τ works across targets. **Ordering is forced**: excision needs a valid graph (last), refinement closes the gap (middle), producer emits near-misses (first). Stages run **concurrently in a round-robin loop**; first valid shrunk graph wins. Producer interchangeable + each repair stage toggleable → controlled **ablation** of which stage earns its keep.
- **8.2 Implementation:** Algorithm 8 (forge pipeline).

### Ch 9 — Results (p.34-38)
- **Setup:** trained on PERUN; eval **60-second budget per attempt**, single node (256 cores), **128 parallel workers** (256 oversubscribes → collapse). Higher accuracy/success better; lower MAE/time/size better.
- **9.1 What architectures predict:**
  - **Degree solved** (Table 1): **SAGE 4,900 params → accuracy 1.000, MAE 0.00**; GPS–GIN 0.998. GCN variants weakest (normalization discards the count). Loopy moderate but numerically unstable (MAE blows past 10⁵).
  - **Minimum cycle solved by nothing** (Table 2): best = cycle-aware **Loopy 0.510** accuracy, MAE 1.45; ranking **inverts** (SAGE/GIN strong on degree are worst here). → **girth not recoverable as a supervised per-node label → must be handled by explicit search.** (Figure 10.)
- **9.2 Constructing (k,g)-graphs:** **22 targets, 100 seeds each, 60 s/attempt.** Columns: v-rl (RL voltage), v-girth / v-tabu / v-alg (voltage guided by girth predictor / tabu-cost predictor / algebraic), A* , RW (random walk), BF (brute force), dRL (direct RL), forge.
  - **voltage-rl has by far the widest reach** — only method solving (3,9), (3,10), (4,8), (6,6); only one to crack **(6,5) at 57% of seeds**.
  - **Classical A*/RW**: fast, cage-size where they work, but only easy low-girth targets; fail past a moderate point. **Brute force solves almost nothing. Direct RL** solves only trivial targets, slowest by far.
  - **GNN-guided voltage variants collapse earlier than voltage-rl and algebraic** → per-node guidance does **not extend reach.**
  - **Size:** classical methods smallest where they work; voltage family **~2× Moore bound**; forge **much closer to cage** but solves fewer targets than voltage-rl. **No method solves (4,7), (5,7), (7,5)** within budget. Reference Moore bounds: (3,5)=10, (6,5)=37, (3,10)=62, (5,7)=106.
- **9.3 Forge** (Tables 5 & 6):
  - Table 5 (per producer): three tabu-based producers tie ~**0.46 success, ~1.23× Moore** (voltage-tabu 1.22, voltage-algebraic 1.23, voltage-girth 1.24), **all beating the RL producer (0.31 success, 1.51×)**. The widest-reach standalone policy is the **weakest producer in Forge** — pipeline needs a *varied stream of near-cage lifts* a deterministic policy doesn't supply.
  - Table 6 (ablation on voltage-rl producer): full 0.31/1.51; **no refinement → 0.62/1.58** (refinement costs coverage for the rl producer); **no excision → 0.37/2.06** (excision is what shrinks). Forge recovers near-1.23× Moore size while keeping much of voltage reach.

### Ch 10 — Future Work (p.39)
- **Co-adapt** Forge's three stages (currently tuned in isolation). Learn the **excision root and depth** (currently a fixed sweep). Train producer for **diversity** not single-shot validity. Scale to larger unreached targets.
- **Record-graph search:** Forge run as long parallel PERUN search against open targets **(8,5), (9,5), (10,5), (11,6)**, banking every valid graph by girth. **To date no record-beating graph found; ongoing.**
- **Girth-specialised, degree-agnostic model:** voltage lift is k-regular by construction → model never needs to enforce/observe degree; fixing g pins a rigid local tree to radius ⌊(g−1)/2⌋. A model fixing g and never seeing k could specialise.

### Ch 11 — Conclusion (p.40-41)
- Restates the narrow question and the progression of increasing structure.
- **Degree essentially solved (tiny model, perfect); min-cycle resists every architecture (best still mislabels ~half).**
- Dominant pattern = **coverage vs. size trade-off**: voltage reaches most but ~2× Moore; classical size-optimal but narrow; direct RL weakest by a wide margin; **Forge most balanced (~1.23× Moore, keeps much reach)**; largest targets unreached.
- **Value of learning is conditional:** GNN-guided voltage producers **match** the algebraic search on size but **don't extend reach** (solve no new targets, no smaller graphs, collapse slightly earlier on hardest). A learned method merely matching its baseline still says something real — the algebraic structure leaves little room for per-node guidance.
- **Clearest learning contributions:** (1) the **learned repair inside excision** (guides boundary reconnection, exact backtracking guarantees validity), and (2) the **RL producer as a useful source of lifts in Forge** (even if tabu producers score higher).
- **Headline answer:** learning did **not replace** structured search; it **guides and densifies** the search whose validity exact algorithms still decide — "a learned component ranks or proposes and an exact check accepts."

---

## 2. KEY ANCHOR POINTS ("dôležité záchytné body") — MUST appear in defense

1. **The object & problem:** (k,g)-graph = k-regular with girth ≥ g; smallest = **cage**, order n(k,g); a classical extremal problem, best orders still improved by hand + computer search.
2. **Why it's hard for ML (two impossibility facts):** MP-GNN expressivity ≤ 1-WL [4], and **girth provably not computable by MP-GNN** [5]; RL only works where the move space has structure. → can't be a black box.
3. **The central thesis question / philosophy:** *learning guides where the search looks; exact algorithms decide validity* — "ranks/proposes vs. accepts." Constrain the space mathematically, search only the remaining freedom.
4. **Property-prediction result — the diagnostic:** **degree is solved** (SAGE, 4,900 params, accuracy 1.000) but **min-cycle/girth resists every architecture** (best = Loopy 0.510, mislabels ~half) → **girth must be handled by explicit exact search, not a learned label.**
5. **Four construction methods, increasing structure:** (a) **Direct RL** (edge editing, weakest), (b) **Voltage Lifts** (k-regularity structural, only girth searched), (c) **Refinement & Excision** (repair + shrink), (d) **Forge** (composition).
6. **Direct RL** needs **curriculum** (start (3,5)) + **potential-based shaped reward** to learn at all; without curriculum it never even makes a (3,5)-graph; quadratic action space → doesn't scale.
7. **Voltage lift core idea:** k-regularity is automatic; girth read off the **base graph** via net voltages (girth = min |W|·ord(s)); **gauge freedom** zeroes spanning-tree voltages → search only non-tree edges (K₄: 3 instead of 6).
8. **Refinement** = degree-preserving 2-/3-swaps via tabu, weighted short-cycle cost, learned evaluator **ranks only**, exact check decides.
9. **Excision** = remove a Moore-radius tree (depth ⌊(g−1)/2⌋), stitch deficient vertices at distance ≥ g−1; learned (k,g)-independent repair policy + exact backtracking fallback. Example: dodecahedron → Petersen = (3,5)-cage.
10. **Forge** = producer + refinement + excision; "near-miss" hand-off via **defective fraction ≤ τ**; ordering forced; round-robin parallel; producer interchangeable.
11. **Headline numbers — coverage:** **22 targets**; **voltage-rl widest reach** (only one to solve (3,9),(3,10),(4,8),(6,6) and (6,5) at 57%); no method solves (4,7),(5,7),(7,5).
12. **Headline numbers — size:** voltage methods **~2× Moore bound**; classical size-optimal but narrow; **Forge ~1.23× Moore** (tabu producers 1.22–1.24) and **meets the cage exactly on the smallest targets**.
13. **The honest finding:** **learned components rarely beat their non-learned counterparts** — GNN-guided voltage producers match algebraic on size, don't extend reach; RL voltage policy is strongest standalone yet weakest Forge producer (Forge wants diversity).
14. **Where learning genuinely helps:** learned excision repair, and the RL policy as a diverse lift source inside Forge.
15. **Ablation evidence:** removing excision → size 1.51→2.06 (excision shrinks); removing refinement → success 0.31→0.62 but bigger (refinement trades coverage for size on rl producer).

---

## 3. KEY FIGURES / TABLES (slide candidates)

| Item | File / location | Shows | Slide use |
|---|---|---|---|
| **Moore bound formula** | `images/main.pdf-0012-15.png` (§2.2, p.2) | The lower-bound formula on order | Define the benchmark |
| **Figure 1** — message-passing layer | `main.pdf-0014-00/01.png` (p.4) | One MP layer aggregate+update | GNN intro |
| **Figure 2** — depth & receptive field | `main.pdf-0014-04.png` (p.4) | Why layers = how far info travels | Why girth (global) is hard |
| **Figure 3** — voltage lift K₄ over Z₃ | `main.pdf-0027-00.png` (p.16) | 12-vertex 3-regular lift | Explain voltage construction |
| **Figure 4** — base cycles → girth | `main.pdf-0028-03.png` (p.18) | Triangle wraps to 9-cycle, 4-cycle survives → girth 4 | Net-voltage / girth-on-base |
| **Figure 5/6** — 2-swap | `main.pdf-0033-05.png`, `main.pdf-0034-00.png` (p.23-24) | Degree-preserving swap, breaking a triangle | Refinement |
| **Figure 7** — 3-swap | `main.pdf-0034-04.png` (p.24) | 3-edge rewire | Refinement escalation |
| **Figure 8** — tree removal (dodecahedral) | `main.pdf-0037-00.png` (p.26) | Depth-2 tree removed, 6 deficient vertices | Excision step 1 |
| **Figure 9** — dodecahedral → Petersen | `main.pdf-0037-03.png` (p.27) | Excision to (3,5)-cage, half the order | Excision payoff (great visual) |
| **Figure 10** — degree vs min-cycle accuracy | `main.pdf-0045-03.png` (p.35) | The split: degree learnable, girth not | The diagnostic result |
| **Table 1** — degree prediction | §9.1, p.34 | SAGE 1.000 @ 4,900 params | Degree solved |
| **Table 2** — min-cycle prediction | §9.1, p.35 | Loopy 0.510 best | Girth not solved |
| **Table 3** — mean time to solve | §9.2, p.36 | Coverage map (blank = unsolved) | Reach comparison |
| **Table 4** — mean graph size \|V\|/\|E\| | §9.2, p.37 | Size comparison | ~2× Moore vs cage |
| **Table 5** — Forge by producer | §9.3, p.38 | tabu producers ~0.46 / 1.23× | Forge headline |
| **Table 6** — stage ablation | §9.3, p.38 | excision shrinks, refinement trades coverage | Justify pipeline |

Recommended hero slides: **Figure 9** (excision visual), **Figure 4** (voltage girth), **Figure 10 + Tables 1/2** (the diagnostic), **Table 5** (Forge result).

---

## 4. NAVIGATION MAP ("ako sa orientovať v práci")

| Topic | Page | Section |
|---|---|---|
| Abstract (all numbers in one place) | ii | Abstract |
| Problem statement & guiding question | 1 | Ch 1 |
| Definitions: girth, (k,g)-graph, cage, **Moore bound** | 2 | §2.1–2.2 |
| Groups / cyclic Z_n | 3 | §2.3 |
| Tabu search | 3 | §2.4 |
| GNN basics, 1-WL limitation, architecture list | 3–5 | §2.5 |
| RL / PPO basics | 5 | §2.6 |
| Classical cage methods, voltage lifts, excision (lit) | 6–7 | §3.1 |
| **GNN impossibility for girth** [5] | 7 | §3.2 |
| Property prediction (degree vs min-cycle) setup | 9 | Ch 4 |
| Exact min-cycle label algorithm | 9–10 | §4.2 / Alg 1 |
| Why supervised A* fails → RL | 11 | §5.1 |
| Direct RL formulation, action pruning | 12 | §5.2 |
| **Curriculum + shaped reward** | 12–13 | §5.3 |
| Direct RL limits (quadratic, no regularity guarantee) | 13 | §5.4 |
| Legal-edits & reward algorithms | 14–15 | Alg 2,3 |
| Voltage lift: 3 parameters | 16 | §6.1 |
| Building the lift, regularity proof, K₄/Z₃ example | 16–17 | §6.2 |
| **Girth on the base graph (net voltage)** | 17–18 | §6.3 |
| **Gauge transformation / spanning-tree zeroing** | 18–19 | §6.3 |
| Search methods (exhaustive, tabu, GNN-beam, RL) | 20 | §6.4 |
| lift_girth + voltage tabu algorithms | 20–22 | Alg 4,5 |
| 2-swap / 3-swap / refinement cost & learned evaluator | 23–25 | §7.1 |
| **Excision: tree removal + stitching + learned policy** | 26–28 | §7.2 |
| Refinement & repair algorithms | 29–30 | Alg 6,7 |
| **Forge pipeline + defective fraction τ** | 31–32 | Ch 8, §8.1 |
| Forge algorithm | 33 | Alg 8 |
| Eval setup (60s, 128 workers, PERUN) | 34 | Ch 9 intro |
| **Tables 1–2 + Fig 10 (prediction results)** | 34–35 | §9.1 |
| **Tables 3–4 (construction: time & size, 22 targets)** | 36–37 | §9.2 |
| **Tables 5–6 (Forge results + ablation)** | 38 | §9.3 |
| Future work (record search, girth-specialised model) | 39 | Ch 10 |
| Conclusion (the conditional value of learning) | 40–41 | Ch 11 |
| Bibliography (key refs [1] Wagner, [3] Exoo, [5] Garg, [15] Exoo-Jajcay voltage) | 42–43 | — |

---

## 5. DEFENSE Q&A RISKS (likely examiner questions + where the answer lives)

1. **"If a GNN provably can't detect girth, why use GNNs at all?"**
   → §3.2 (p.7) + Ch 11 (p.40). Answer: GNNs aren't asked to *verify* girth; exact algorithms do that. GNNs **rank/propose** moves inside a structured search. The diagnostic (Tables 1–2) confirms this is the right split.

2. **"Your learned components rarely beat the non-learned ones — what's the contribution then?"**
   → Ch 11 (p.40, "value of learning is conditional"). A learned method *matching* a structured baseline shows the algebraic structure already captures most of the signal; the genuine wins are the **learned excision repair** and the **RL producer as a diverse lift source in Forge**. Also honesty/negative results have scientific value.

3. **"How does the voltage lift guarantee k-regularity, and how do you get girth without building the big graph?"**
   → §6.2 (regularity proof, p.16-17) + §6.3 (net-voltage girth formula `min |W|·ord(s)`, p.17-18). Figures 3 & 4.

4. **"What is the gauge transformation and why can you fix tree voltages to zero?"**
   → §6.3 "Reducing the search space" (p.18-19). Gauge = relabeling vertex copies; doesn't change net voltage of any cycle; only a cycle-free edge set (spanning tree) can be zeroed → search non-tree edges only.

5. **"Why is Forge ordered producer → refinement → excision, and what is τ (the defective fraction)?"**
   → §8.1 (p.31-32). Ordering forced by what each stage needs; τ = fraction of edges on a short cycle, routes near-misses to refinement, scales with size.

6. **"Why does the RL voltage policy have the widest standalone reach but is the worst producer inside Forge?"**
   → §9.3 (p.38) + Ch 11 (p.40). Forge benefits from a **diverse stream of near-cage lifts**; the deterministic-leaning RL policy doesn't supply variety. Table 5 + Table 6 ablation.

7. **"What does '~1.23× Moore bound' / '~2× Moore bound' mean, and did you find any record graphs?"**
   → Abstract (ii), §9.2-9.3 (p.37-38), Ch 10 (p.39). Voltage ~2×, Forge ~1.23× and meets cage exactly on smallest targets. **No record-beating graph found yet**; record search on (8,5),(9,5),(10,5),(11,6) is ongoing.

8. **"Why curriculum learning and reward shaping for direct RL — and why is direct RL still the weakest?"**
   → §5.3 (p.12-13, without curriculum never makes a (3,5)-graph) + §5.4 (p.13, quadratic action space, regularity not guaranteed) + §9.2 (solves only trivial targets, slowest).

9. **(Possible) "Why Erdős–Rényi graphs for training the property predictors, not (k,g)-graphs?"**
   → §4.1 (p.9). Dynamically sampled fresh graphs prevent memorization and cheaply expose many structures; labels are exact.

10. **(Possible) "Why 128 workers, not more?"**
   → Ch 9 intro (p.34). At 256 workers, unpinned numerical-library threads oversubscribe 256 cores under a wall-clock budget → fewer iterations per attempt → collapse.

---

## NARRATIVE ARC (for the 12-minute talk)

The thesis tells a single, disciplined story: **"learning guides, exact algorithms decide."**

1. **Hook the problem (1 min):** (k,g)-graphs and cages — a one-line definition that hides a very hard extremal problem still being chipped at by hand + computers.
2. **The obstacle (1.5 min):** ML can't be a black box here — a message-passing GNN *provably cannot even detect girth* (1-WL limit), and RL only works where moves carry structure. So reframe: don't ask ML to produce a cage; ask it to **guide where a search looks** while exact checks verify.
3. **The diagnostic (2 min):** Probe the two defining constraints. **Degree is trivially learnable** (SAGE, ~5k params, perfect); **girth/min-cycle resists everything** (best 0.51). This *empirically justifies* the whole architecture: degree gets built into the construction, girth gets handled by exact search. (Figure 10, Tables 1–2.)
4. **The method ladder (4 min)** — increasing structure, each fixing the previous one's flaw:
   - **Direct RL:** pure edge editing — needs curriculum + shaped reward just to make a (3,5)-graph; quadratic, weakest. Lesson: too little structure.
   - **Voltage lifts:** make k-regularity *structural*; search only girth via net voltages on a tiny base graph (gauge trick shrinks it further). Widest reach but **~2× Moore bound**. (Figures 3–4.)
   - **Refinement & excision:** repair near-misses (degree-preserving swaps) and shrink toward the cage (dodecahedron → Petersen). Excision's hard core gets a learned repair policy + exact backtracking. (Figure 9.)
   - **Forge:** compose all three — producer for reach, refinement to rescue near-misses, excision for size — reaching **~1.23× Moore and the cage exactly on the smallest targets**. (Tables 5–6.)
5. **The honest punchline (2 min):** Across **22 targets**, the trade-off is coverage vs. size; **learned components rarely beat their structured counterparts** — and that *is* the finding. Learning earns a place *beside* structured search (learned excision repair, RL as a diverse lift source), sharpening *where* it looks without owning *whether* it's correct.
6. **Close (0.5 min):** Future work — co-adapt Forge's stages, learn excision root/depth, a girth-specialised degree-agnostic model; an ongoing parallel record search on four open cages.

**The single storyline to follow:** *the constraint that defines the object (degree vs. girth) splits cleanly into "learnable/enforceable" and "must be searched exactly," and every method in the thesis is an answer to where on that line to draw the boundary between learning and exact computation — Forge being the best-balanced answer.*

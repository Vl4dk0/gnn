# Thesis References Forbidden Per-Girth Specialist Predictors in Results Table
## Source of the issue
- `thesis/chapters/07-experiments-results.typ` (Lines 19-24, 33-34, 124-146)
- `AGENTS.md` (Validation Rules section)

## Definition of the issue
The repository rules in `AGENTS.md` explicitly forbid the use and reference of per-girth or per-(k,g) specialist predictors (e.g., `girth_predictor_k*_g*`, `girth_predictor_g*_multik`), noting that these were a mistake from earlier iterations and have been entirely removed.

However, the thesis currently references these forbidden specialist predictors in Chapter 7. Specifically, it describes "Variant B" as a model that "shares one model across several degrees for a fixed girth value" and reports its F1 scores under `[B: fixed girth]`. Additionally, the Generator Comparison table includes a `Volt.+S` column, explicitly identified in the caption as "the matching per-girth specialist predictor when available," thus contradicting the repository's rules.

To resolve this issue, the thesis must be updated to eliminate all mentions of Variant B and the per-girth specialist predictors. This involves removing the description and F1 scores of Variant B, deleting the `Volt.+S` column from the comparison table, and updating all associated prose and captions to only discuss the correct models: Variant A (the cubic-only unified predictor) and Variant C (the full unified predictor).

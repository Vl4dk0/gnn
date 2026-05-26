# Thesis References Per-Girth Specialist Predictors That No Longer Exist in the Codebase

**Source/Occurrence:**
- `thesis/chapters/07-experiments-results.typ` (Lines 19-24, 129-145)
- `AGENTS.md` (Validation Rules section)

**Explanation:**
The thesis text (lines 19-24) explicitly describes evaluating "three degrees of parameter sharing": a model trained only on cubic targets (A), a model sharing parameters across degrees for a fixed girth value (B), and the unified model (C). The results table (lines 26-44) reports F1 scores for all three variants.

The thesis currently references models (Variant A, B, Volt.+S specialist predictors) that no longer exist in the codebase and cannot be reproduced.

Furthermore, the generator comparison table (line 129) references `Volt.+S` described as "the matching per-girth specialist predictor when available" — a per-g model that is no longer present in the repository.

The thesis table caption (line 145) still refers to these specialist predictors as if they are valid and available, creating a direct contradiction with the current code state.

**Actionable Steps:**
1. In `thesis/chapters/07-experiments-results.typ`, revise lines 19-24 to remove mention of the three-variant comparison (A, B, C). Only the unified model is valid.
2. Remove the first table (lines 26-44) describing per-girth F1 scores for variants A and B, as these correspond to models that should not exist.
3. In the generator comparison table (lines 124-146), remove the `Volt.+S` column entirely and update the success counts/caption accordingly.
4. Update the surrounding prose to no longer reference "specialist predictors" or the per-girth sharing scheme.

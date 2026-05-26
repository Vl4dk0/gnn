# Invalid Girth Predictor Model ID in Help Text

## Source of the issue
- Codebase: `ai/cage/voltage/search.py` (Line 613)
- Rules: `AGENTS.md` (Validation Rules section, regarding unified girth predictors)

## Definition of the issue
In `ai/cage/voltage/search.py`, the argument parser help text for `--model-id` uses the example `girth_predictor_k3_g7`. However, `AGENTS.md` explicitly states: "The voltage girth predictor is always a single (k, g)-independent model... Earlier a mistaken agent trained per-(k,g) and per-g predictors (girth_predictor_k*_g*, girth_predictor_g*_multik); those were never intended, confused everyone, and have been removed. Never re-create per-(k,g) or per-g predictors."

Thus, providing `girth_predictor_k3_g7` as an example contradicts the unified model design and misleads users into using the deprecated naming convention.

To fix this, the `help` string for the `--model-id` argument around line 613 should be updated to reflect the unified predictor (e.g., `"Girth predictor model_id (e.g. girth_predictor_v1) — enables beam search"`).

# Voltage Edge Embedding Has Fixed Max Size That May Not Cover All Groups in Search

**Source/Occurrence:**
- `ai/cage/voltage/model.py` (Lines 51, 60, 116, `GirthPredictor.__init__` and `forward`)
- `ai/cage/voltage/search.py` (Lines 362-391, `_candidate_groups_for_target`)

**Explanation:**
The `GirthPredictor`'s edge voltage embedding layer is initialized as:

```python
self.edge_embed = nn.Embedding(max_group_order, hidden_dim)
```

The default `max_group_order = 200` is used when loading a saved model (line 191 of `model.py`), and during inference the voltage index is clamped to `[0, max_group_order - 1]`.

However, `_candidate_groups_for_target` in `search.py` can generate groups with orders up to `max_order` (default 100), and the `generate_dataset` function in `data_gen.py` uses `max_group_order=60` by default. If a model is trained with `max_group_order=60` but then used for beam search over groups with order up to 100 (or any order > 60), any voltage value in the range [60, order-1] will be silently clamped to 59, producing an incorrect embedding for that voltage element.

This silent clamping means the model receives incorrect input features for high-voltage-index elements without any error or warning, causing the beam search to produce meaningless scores for those configurations.

**Actionable Steps:**
1. During beam search inference in `search.py`, before calling `model(data)`, check that `group.order <= model.max_group_order` and skip or warn if not.
2. Save `max_group_order` into `info.json` during training (currently it appears under `training` dict) and ensure `load_girth_predictor` reads this value correctly so it is not silently defaulted to 200.
3. Consider normalizing voltage indices to `[0, 1]` as a continuous feature rather than using a fixed-size embedding, which would generalize across group orders.

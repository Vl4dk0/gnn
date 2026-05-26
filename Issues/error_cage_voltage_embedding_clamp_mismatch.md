# Voltage Edge Embedding Has Fixed Max Size That May Not Cover All Groups in Search
## Source of the issue
- `ai/cage/voltage/model.py` (Lines 51, 60, 116, `GirthPredictor.__init__` and `forward` function)
- `ai/cage/voltage/search.py` (Lines 362-391, `_candidate_groups_for_target` function)
- `ai/cage/voltage/data_gen.py` (`generate_dataset` function)

## Definition of the issue
The `GirthPredictor` model utilizes an embedding layer to encode edge voltage labels, initialized with a fixed maximum vocabulary size via `self.edge_embed = nn.Embedding(max_group_order, hidden_dim)`. When the model performs a forward pass, the input voltage indices are clamped to a maximum value of `[0, max_group_order - 1]` to prevent index-out-of-bounds errors.

During training data generation, `max_group_order` defaults to 60. However, the inference function `_candidate_groups_for_target` in `search.py` can generate candidate groups with orders up to 100. Furthermore, when loading a saved model, `max_group_order` is typically defaulted to 200 rather than properly retrieving the value from the loaded state or `info.json`. 

Because of this mismatch, if a model trained with `max_group_order=60` is used in a beam search evaluating group elements up to order 100, any voltage value $v \ge 60$ will be silently clamped to 59. The model will receive incorrect and indistinguishable embeddings for entirely different voltage elements, which disrupts the predictor's logic and causes beam search to produce meaningless scores without throwing any warnings. 

To resolve this issue, the beam search script must strictly enforce `group.order <= model.max_group_order` before inference. The saved model state should accurately preserve `max_group_order` via `info.json` to prevent arbitrary default fallbacks. Alternatively, the embedding layer could be replaced by continuous normalized features `[0, 1]` to elegantly accommodate variable group orders.

# get_min_cycle Labels Self-Loop Vertices as Cycle-Length 1, But Thesis Targets Simple Graphs

## Source of the issue
- `ai/min_cycle/functions/graph_service.py` (Lines 37-38, `get_min_cycle`)
- `backend/utils/graph_utils.py` (Lines 62-100, `generate_random_graph`; Line 152-153, `compute_girth`)
- `thesis/chapters/04-preparatory-tasks.typ` (Lines 23-26, Data Generation)

## Definition of the issue
The thesis (lines 23-26) states: "Both tasks use dynamically generated random graphs ... new Erdos-Renyi graphs are sampled during training." This implies the training graphs are standard Erdős-Rényi graphs. However, `generate_random_graph` in `backend/utils/graph_utils.py` (line 63) has `self_loop_prob=0.1` — meaning **10% of nodes in each generated graph receive a self-loop**. This is not an Erdős-Rényi graph.

As a direct consequence, `get_min_cycle` returns `1` for any vertex with a self-loop (line 37-38 in `graph_service.py`), treating self-loops as 1-cycles. The model is therefore trained to predict `1.0` for self-loop vertices. `compute_girth` in `graph_utils.py` (line 152-153) also returns `1` when any node has a self-loop, which is consistent with `get_min_cycle`.

However, the thesis definition (lines 14-18) refers to "the length of the shortest cycle containing that vertex" in the context of simple graphs for the $(k,g)$-graph construction task. Self-loops (1-cycles) have no meaningful role in cage construction, as cages are simple graphs. The presence of self-loops also inflates the proportion of label=1 in training data, biasing the model toward predicting short cycles even on simple graphs.

The thesis does not mention self-loops in the data generation description, creating a mismatch between the written methodology and the actual data pipeline.

To fix this, `backend/utils/graph_utils.py` should change the default `self_loop_prob=0.0` (or remove the self-loop addition entirely) to match the Erdős-Rényi generation described in the thesis. `ai/min_cycle/train.py` and `ai/degree/train.py` should ensure the call to `generate_random_graph` does not pass a non-zero `self_loop_prob`. Alternatively, the thesis should be updated to explicitly state whether self-loops are included in training graphs.

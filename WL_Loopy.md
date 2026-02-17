# Context: Weisfeiler and Leman Go Loopy (r-lWL & r-lGIN)

**Paper Title:** Weisfeiler and Leman Go Loopy: A New Hierarchy for Graph Representational Learning  
**Authors:** Raffaele Paolino, Sohir Maskey, Pascal Welke, Gitta Kutyniok  
**Conference:** NeurIPS 2024  
**Source Material:** Uploaded PDF ("WL_Loopy.pdf") + Web Search  
**Official Repository:** [https://github.com/RPaolino/loopy](https://github.com/RPaolino/loopy)

---

## 1. Executive Summary
The paper introduces **r-loopy Weisfeiler-Leman (r-lWL)**, a hierarchy of graph isomorphism tests, and a corresponding GNN architecture called **r-loopy Graph Isomorphism Network (r-lGIN)**.

**Core Problem:** Standard Message Passing Neural Networks (MPNNs) are bounded by the 1-WL test and cannot detect cycles (e.g., distinguishing a 3-cycle from a set of independent edges). High-order k-WL methods are computationally expensive.
**Solution:** r-lWL enhances node updates by aggregating information not just from direct neighbors, but from **simple paths of length `r`** connecting distinct neighbors.
**Capabilities:**
* Can subgraph-count cycles up to length `r + 2`.
* Can homomorphism-count **cactus graphs** (graphs where every edge belongs to at most one cycle).
* Strictly more expressive than 1-WL; incomparable to k-WL for fixed k.

---

## 2. Theoretical Definitions

### 2.1 r-Neighborhood
The core data structure is the **r-neighborhood** $\mathcal{N}_r(v)$.
For a node $v$, $\mathcal{N}_r(v)$ is the set of all **simple paths** of length $r$ between two *distinct* neighbors of $v$, excluding $v$ itself.
* **Simple Path:** A sequence of nodes where no node is repeated.
* **Constraint:** The endpoints of the path must be in $\mathcal{N}(v)$ (direct neighbors of $v$).
* **Formal Definition:**
    $$\mathcal{N}_r(v) := \{ p = (p_1, \dots, p_{r+1}) \mid \{v, p_1\} \in E, \{v, p_{r+1}\} \in E, p \text{ is simple}, v \notin p \}$$
* **Base Case:** $\mathcal{N}_0(v) = \mathcal{N}(v)$ (standard direct neighbors).

### 2.2 r-lWL Color Refinement
The node color $c^{(t+1)}(v)$ is updated by hashing:
1.  The previous color $c^{(t)}(v)$.
2.  The multiset of colors of direct neighbors (standard WL).
3.  The multisets of colors of paths in $\mathcal{N}_k(v)$ for $k=1 \dots r$.

---

## 3. Model Architecture: r-lGIN

The **r-lGIN** is the neural implementation of r-lWL. It generalizes the GIN (Graph Isomorphism Network) update rule.

### 3.1 The Update Rule (Equation 4)
The node feature update for layer $t+1$ is defined as:

$$h_v^{(t+1)} = \text{MLP} \left( (1 + \epsilon) h_v^{(t)} + \sum_{u \in \mathcal{N}(v)} h_u^{(t)} + \sum_{k=1}^r \sum_{p \in \mathcal{N}_k(v)} \text{GIN}_{path}(p) \right)$$

* **Term 1:** Self-loop node feature.
* **Term 2:** Standard aggregation of direct neighbors.
* **Term 3:** Aggregation of **path embeddings**.

### 3.2 Path Embedding ($\text{GIN}_{path}$)
Paths are treated as independent graph objects (or sequences) and processed to generate a single embedding vector per path.
* The paper suggests using a **GIN** on the path graph itself to maximize expressivity.
* To reduce parameters, the path-processing GIN can be shared across all path lengths $k$.
* **Implementation Detail (from Web/GitHub):** The official code uses `torch.nn.functional.conv3d` with kernel `[1, 0, 1]` to pass messages on paths, as only consecutive nodes in the path are linked.

### 3.3 Complexity
* **Time Complexity:** $\mathcal{O}(|E| + \sum_{v \in V} \sum_{k=1}^r k |\mathcal{N}_k(v)|)$.
* For sparse graphs (e.g., molecular graphs), the number of paths is manageable.
* For dense graphs, path computation is expensive ($\mathcal{O}(N d^r)$).

---

## 4. Implementation Details for Coding Agent

### 4.1 Preprocessing (Critical Step)
Before training, you must precompute the `r-neighborhoods`.

**Algorithm Strategy:**
1.  **Cycle Finding:** Use `networkx.simple_cycles(G)` to find all simple cycles in the graph.
2.  **Path Extraction:**
    * Iterate through each cycle.
    * For every node $v$ in the cycle, the remainder of the cycle forms a path between two of $v$'s neighbors.
    * Store these paths grouped by length $k$ (where path length is `cycle_len - 2`).
    * *Note:* This efficiently generates paths because the relevant paths for cycle counting are segments of cycles.
3.  **Lazy Loading (Optional):** For large datasets, compute cyclic permutations on the fly during the forward pass to avoid OOM (Out Of Memory).

### 4.2 Architecture Components
* **Encoder:** Linear embedding for node and edge features (atomic types, bond types).
* **Path Encoder:** A separate GNN (e.g., GINEConv or standard GIN) that processes the sequence of node features in a path `p` to produce a vector $h_p$.
* **Main GNN Layers:**
    * Input: Graph node features $H$, List of Paths $P$.
    * Operation:
        1.  Compute standard neighbor sum: $A \cdot H$.
        2.  Compute path embeddings: Run `PathEncoder` on all $p \in P$.
        3.  Aggregate path embeddings: Sum $h_p$ for all $p$ centered at $v$.
        4.  Combine: `MLP( (1+eps)H + NeighborSum + PathSum )`.
* **Readout:** Sum pooling over all nodes.

### 4.3 Hyperparameters (Benchmarks)
**Dataset: ZINC12K**
* **r (Path Length):** 5 (Key parameter)
* **Hidden Dimension:** 64
* **Layers:** 3
* **Dropout:** 0
* **Learning Rate:** 0.001 (with ReduceLROnPlateau)
* **Epochs:** 1000
* **Batch Size:** 64

**Dataset: ZINC250K**
* **Hidden Dimension:** 256
* **Layers:** 4

### 4.4 Software Stack
* **Language:** Python
* **DL Framework:** PyTorch
* **Graph Library:** PyTorch Geometric (PyG)
* **Preprocessing:** NetworkX (for cycle extraction)

---

## 5. Empirical Results & Capabilities
Use these metrics to validate the implementation correctness.

1.  **Synthetic Tests:**
    * **Cycles:** Can perfectly distinguish cycles up to length $r+2$.
    * **Cactus Graphs:** Can homomorphism-count all cactus graphs.
    * **Strongly Regular Graphs:** Outperforms 3-WL on the BREC dataset (distinguishes 257/400 pairs vs 3-WL's 0/50).

2.  **Real-World (ZINC12K):**
    * **MAE:** ~0.072 (with $r=5$).
    * **Comparison:** Significantly outperforms standard GIN (~0.163) and GCN (~0.321).
    * **Training Time:** ~10s/epoch (fast) vs 3-WLGNN ~330s/epoch.

## 6. Citations & References
* **Primary:** Paolino, R., Maskey, S., Welke, P., & Kutyniok, G. (2024). *Weisfeiler and Leman Go Loopy: A New Hierarchy for Graph Representational Learning*. NeurIPS.
* **GitHub Implementation:** [RPaolino/loopy](https://github.com/RPaolino/loopy)

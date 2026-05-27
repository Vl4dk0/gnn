import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { useHighlight } from "../../hooks/useHighlight";

export const DocsArchitecturePage = () => {
  useHighlight();

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Model Architectures
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          Architecture choice matters because it changes the message passing{" "}
          <code>update rule</code>: how neighbor signals are aggregated, how self-information is
          mixed in, and which structural distinctions survive.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">At a glance</h2>
        <div className="overflow-x-auto">
          <table className="mt-2 w-full text-sm text-textMuted">
            <thead>
              <tr className="border-b border-line2 text-left text-textSub">
                <th className="py-2 pr-4 font-semibold">Architecture</th>
                <th className="py-2 pr-4 font-semibold">Aggregation</th>
                <th className="py-2 pr-4 font-semibold">Scope</th>
                <th className="py-2 font-semibold">Key strength</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">GCN</td>
                <td className="py-2 pr-4">Degree-normalized mean</td>
                <td className="py-2 pr-4">Local</td>
                <td className="py-2">Stable, simple baseline</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">GraphSAGE</td>
                <td className="py-2 pr-4">Learned mean</td>
                <td className="py-2 pr-4">Local</td>
                <td className="py-2">Inductive, generalizes to unseen nodes</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">GIN</td>
                <td className="py-2 pr-4">Sum + MLP</td>
                <td className="py-2 pr-4">Local</td>
                <td className="py-2">Maximally expressive (1-WL equivalent)</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">GPS</td>
                <td className="py-2 pr-4">Local conv + global attention</td>
                <td className="py-2 pr-4">Global</td>
                <td className="py-2">Captures long-range dependencies</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 font-medium text-textMain">Loopy</td>
                <td className="py-2 pr-4">Sum + r-neighborhood paths</td>
                <td className="py-2 pr-4">Local (cycle-aware)</td>
                <td className="py-2">Detects cycles up to length r+2</td>
              </tr>
            </tbody>
          </table>
        </div>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          GCN - Graph Convolution Network
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          GCN uses degree-normalized aggregation: each neighbor's message is scaled by the inverse
          square root of both the sender's and receiver's degree. This normalization makes training
          stable but can dampen raw counting signals — the model struggles to distinguish a node
          with 3 neighbors from one with 5 if the normalized values end up similar.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">
            {`# ai/models/gcn.py (simplified)
class GCN_GNN(BaseGNN):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x`}
          </code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          GraphSAGE - Sampling and Aggregation
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          GraphSAGE was designed as an inductive architecture: it learns <code>aggregator
          functions</code> rather than fixed node embeddings. This means it can generalize to
          unseen nodes and dynamic graphs without retraining. In practice it uses a learned mean
          aggregation followed by concatenation with the node's own features.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">
            {`# ai/models/sage.py
class SAGE_GNN(BaseGNN):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        # Notice aggr="add". This sum aggregation preserves neighbor count information,
        # making it very suitable for degree-related predictions.
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr="add"))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr="add"))
            
        self.convs.append(SAGEConv(hidden_dim, output_dim, aggr="add"))`}
          </code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          GIN - Graph Isomorphism Network
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          GIN uses sum aggregation followed by an MLP with a learnable self-weight term. It is
          provably as powerful as the <code>1-WL test</code> (Weisfeiler-Leman graph isomorphism
          test) — a classical algorithm that distinguishes graphs by iteratively comparing
          neighborhood multisets. In practice, this means GIN can distinguish any pair of graphs
          that 1-WL can distinguish, making it one of the most expressive standard GNN
          architectures.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">
            {`# ai/models/gin.py
class GIN_GNN(BaseGNN):
    def _make_gin_conv(self, in_dim: int, out_dim: int) -> GINConv:
        """Create a GINConv layer with 2-layer MLP."""
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        # train_eps=True enables the learnable (1 + eps) self-loop weight
        return GINConv(mlp, train_eps=True)`}
          </code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          GPS - General, Powerful, Scalable
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          GPS is a graph transformer that combines local message passing with global attention. Each
          layer runs a standard GNN convolution (GCN, GIN, or SAGE) in parallel with a multi-head
          attention mechanism over all nodes, then merges the results. This gives every node a{" "}
          <code>global view</code> of the graph in a single layer — unlike purely local
          architectures that need many layers stacked to propagate information across distant nodes.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">
            {`# ai/models/gps.py
class GPS_GNN(BaseGNN):
    def __init__(self, ... conv_type="gin", heads=4):
        # GPS layers
        for _ in range(num_layers):
            inner_conv = self._make_inner_conv(hidden_dim)
            layer = GPSConv(
                channels=hidden_dim,
                conv=inner_conv,
                heads=heads,
                dropout=dropout,
                act="relu",
            )
            self.gps_layers.append(layer)

    def forward(self, data: Data) -> torch.Tensor:
        # Each GPS layer combines local conv + global attention
        for layer in self.gps_layers:
            x = layer(x, edge_index, batch=data.batch)`}
          </code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Loopy</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Loopy adds a second structural channel on top of ordinary message passing by incorporating{" "}
          <code>r-neighborhood</code> path and cycle information. For each node, it finds all
          simple cycles up to length <code>r+2</code> that pass through it, and processes them
          with dedicated path convolutions. This gives the model explicit cycle awareness — standard
          GNNs provably cannot count cycles, but Loopy can. The trade-off is that it needs
          sufficient hidden dimensions to leverage this extra information effectively.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">
            {`# ai/models/loopy.py (simplified PathConv + Scatter)
for L in range(self.r + 1):
    key = f"loopyN{L}" # Precomputed r-neighborhood paths
    paths = data[key]
    
    # Process node embeddings along each path
    path_embeddings = x[paths[1:]]
    contribution = path_conv(path_embeddings, path_atomic)
    
    # Aggregate contributions back to center nodes
    center_nodes = paths[0]
    node_contribution = scatter(contribution, center_nodes, reduce="sum")
    
    r_contribution = r_contribution + (1 + self.r_eps[L]) * node_contribution`}
          </code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Primary sources</h2>
        <ol className="ml-5 mt-1.5 list-decimal space-y-2 text-textMuted">
          <li>
            Kipf and Welling, GCN:
            <a className="text-textMain" href="https://arxiv.org/abs/1609.02907">
              {" "}
              arXiv 1609.02907
            </a>
          </li>
          <li>
            Hamilton et al., GraphSAGE:
            <a className="text-textMain" href="https://arxiv.org/abs/1706.02216">
              {" "}
              arXiv 1706.02216
            </a>
          </li>
          <li>
            Xu et al., GIN:
            <a className="text-textMain" href="https://openreview.net/forum?id=ryGs6iA5Km">
              {" "}
              OpenReview ryGs6iA5Km
            </a>
          </li>
          <li>
            Rampasek et al., GraphGPS:
            <a
              className="text-textMain"
              href="https://proceedings.neurips.cc/paper_files/paper/2022/hash/6c1f4b5a7d7d5d8a9a7b4a9f99cc5a0a-Abstract-Conference.html"
            >
              {" "}
              NeurIPS 2022
            </a>
          </li>
          <li>
            Paolino et al., Weisfeiler and Leman Go Loopy:
            <a className="text-textMain" href="https://openreview.net/forum?id=9O2sVnEHor">
              {" "}
              OpenReview 9O2sVnEHor
            </a>
          </li>
        </ol>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/docs/gnns" label="How GNNs work" direction="back" />
        <DocsNextButton href="/docs/module-degree" label="Degree prediction" />
      </div>
    </DocsLayout>
  );
};

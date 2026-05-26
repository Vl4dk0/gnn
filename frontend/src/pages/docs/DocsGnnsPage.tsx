import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { useHighlight } from "../../hooks/useHighlight";

export const DocsGnnsPage = () => {
  useHighlight();

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          How GNNs work
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          Graph neural networks update each node by exchanging messages with its neighbors. That
          mechanism is the core difference from standard neural networks, which operate on fixed
          vectors without an explicit notion of graph structure.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">What enters the model?</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The thesis focuses on three parts: <code>degree prediction</code>,{" "}
          <code>minimum-cycle prediction</code>, and <code>(k,g)-graph generation</code>.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The first two are node-level regression tasks, so each graph is converted into node
          features, connectivity, and one scalar target per node. We use the PyTorch Geometric <code>Data</code> object to represent this:
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">
            {`# A graph is represented as a single Data object in PyTorch Geometric
data = Data(
    x=node_features,         # Shape: [num_nodes, num_node_features]
    edge_index=edge_index,   # Shape: [2, num_edges] (COO format)
    y=targets,               # Shape: [num_nodes] (for node-level tasks)
    batch=batch_vector       # Shape: [num_nodes] (maps node to graph in a batch)
)`}
          </code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Message passing and readout
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          A <code>forward pass</code> has two stages: <code>message passing</code> and{" "}
          <code>readout</code>
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          During <code>message passing</code>, each node combines its current hidden state with
          aggregated neighbor information.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          During <code>readout</code>, the final hidden state is mapped to the output for that node.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Because the prediction tasks are <code>node-level</code>, the readout stays local: each
          node ends with <code>one scalar prediction</code> rather than a single graph-wide label.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">One message-passing update</h2>
        <div className="mt-2.5 rounded-lg border-2 border-line2 bg-bg1 p-4">
          <p className="text-sm font-semibold uppercase tracking-[0.7px] text-textDim">Message</p>
          <p className="mt-1 font-mono text-[0.95rem] text-textMain">
            m<sub>v</sub>
            <sup>(t)</sup> = AGG<sub>{"{w in N(v)}"}</sub> φ<sub>t</sub>(h<sub>v</sub>
            <sup>(t)</sup>, h<sub>w</sub>
            <sup>(t)</sup>, e<sub>vw</sub>)
          </p>
          <p className="mt-4 text-sm font-semibold uppercase tracking-[0.7px] text-textDim">
            Update
          </p>
          <p className="mt-1 font-mono text-[0.95rem] text-textMain">
            h<sub>v</sub>
            <sup>(t+1)</sup> = ψ<sub>t</sub>(h<sub>v</sub>
            <sup>(t)</sup>, m<sub>v</sub>
            <sup>(t)</sup>)
          </p>
        </div>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">One layer, step by step</h2>
        <div className="grid grid-cols-[repeat(2,minmax(0,1fr))] gap-4 max-[760px]:grid-cols-1">
          {[
            { step: 1, caption: "Each node holds a hidden state (feature vector). Here v = 0.30 and its neighbors w₁–w₄ hold their own values." },
            { step: 2, caption: "Every neighbor sends a message to v along its edge. Messages are computed from the sender's current hidden state." },
            { step: 3, caption: "All incoming messages are aggregated (Σ) and combined with v's own state via an update function U to produce the new hidden state hᵗ⁺¹." },
            { step: 4, caption: "After one full layer, every node has updated its state. Stacking more layers lets each node see further into the graph." },
          ].map(({ step, caption }) => (
            <figure key={step} className="rounded-[10px] p-0">
              <img
                src={`/static/mpnn-step-${step}.png`}
                alt={`Message-passing step ${step}`}
                className="block w-full rounded-[10px]"
              />
              <figcaption className="mt-2 text-sm text-textSub">{caption}</figcaption>
            </figure>
          ))}
        </div>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Why sum aggregation is used here
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          In this thesis the architectures use <code>sum aggregation</code> because it is the most
          expressive standard choice for multiset inputs. With a sufficiently powerful MLP, sum
          aggregation can be <code>injective</code> over multisets — meaning two different
          neighborhoods will always produce different aggregated values. Mean and max aggregation
          lose this property: mean collapses <code>{"{1,1,1}"}</code> and{" "}
          <code>{"{2,2,2}"}</code> if they share the same average, and max ignores everything
          except the largest value.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          GNN Hyperparameters and their effects
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          When training these models, several key parameters dictate how well the model learns:
        </p>
        <ul className="mt-2.5 ml-5 list-disc space-y-3 text-base leading-[1.7] text-textMuted">
          <li>
            <strong className="text-textMain">Number of layers (l):</strong> Determines the <code>receptive field</code>. With 4 layers, a node gathers information from neighbors up to 4 hops away. Too few layers means the node can't "see" far enough (critical for minimum cycle prediction). Too many layers can cause <code>over-smoothing</code>, where all node embeddings become indistinguishable.
          </li>
          <li>
            <strong className="text-textMain">Hidden dimension (h):</strong> The size of the node embeddings at each layer (e.g., 32, 64, 128). This is the model's <code>capacity</code>. Larger dimensions allow the model to memorize complex graph structures, but risk overfitting on simpler tasks.
          </li>
          <li>
            <strong className="text-textMain">Learning Rate:</strong> Controls the step size during gradient descent (typically 0.001).
          </li>
          <li>
            <strong className="text-textMain">Dropout:</strong> Randomly zeros out elements of the hidden embeddings during training (e.g., 0.2) to prevent overfitting, forcing the model to learn general structural rules.
          </li>
        </ul>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Citations</h2>
        <ol className="ml-5 mt-1.5 list-decimal space-y-2 text-textMuted">
          <li>
            Gilmer et al., Neural Message Passing for Quantum Chemistry:
            <a className="text-textMain" href="https://arxiv.org/abs/1704.01212">
              {" "}
              arXiv 1704.01212
            </a>
          </li>
          <li>
            Xu et al., How Powerful Are Graph Neural Networks:
            <a className="text-textMain" href="https://openreview.net/forum?id=ryGs6iA5Km">
              {" "}
              OpenReview ryGs6iA5Km
            </a>
          </li>
        </ol>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/" label="Overview" direction="back" />
        <DocsNextButton href="/docs/architecture" label="Architectures" />
      </div>
    </DocsLayout>
  );
};

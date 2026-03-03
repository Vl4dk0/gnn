import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { useFeatureFlags } from "../../hooks/useFeatureFlags";
import { useHighlight } from "../../hooks/useHighlight";

export const DocsTrainingPage = () => {
  const features = useFeatureFlags();
  useHighlight();

  return (
    <DocsLayout currentPath="/docs/training.html" featureActive={features}>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Try it yourself: a minimal degree predictor
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          This is a compact local tutorial without API or UI. It trains one GNN for node-degree
          regression on random graphs, then prints how many node predictions are exactly correct.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          <code>requirements.txt</code>
        </h2>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-3.5">
          <code className="language-plaintext">{`torch
torch-geometric
networkx`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          <code>train.py</code>
        </h2>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-3.5">
          <code className="language-python">{`import random
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class DegreeSAGE(torch.nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.c1 = SAGEConv(1, hidden, aggr="add")
        self.c2 = SAGEConv(hidden, hidden, aggr="add")
        self.c3 = SAGEConv(hidden, 1, aggr="add")

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.relu(self.c1(x, ei))
        x = F.relu(self.c2(x, ei))
        x = self.c3(x, ei)
        return x.squeeze(-1)


def make_graph(n_min=5, n_max=12, p_min=0.15, p_max=0.6):
    n = random.randint(n_min, n_max)
    p = random.uniform(p_min, p_max)
    g = nx.erdos_renyi_graph(n, p)

    edge_index = []
    for u, v in g.edges():
        edge_index.append([u, v])
        edge_index.append([v, u])

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    x = torch.ones((n, 1), dtype=torch.float)
    y = torch.tensor([g.degree(i) for i in range(n)], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)


def train(epochs=2000, graphs_per_epoch=20):
    model = DegreeSAGE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for _ in range(graphs_per_epoch):
            d = make_graph()
            pred = model(d)
            loss = F.mse_loss(pred, d.y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        if ep % 200 == 0:
            print(f"epoch {ep:4d} | loss {total / graphs_per_epoch:.4f}")

    torch.save(model.state_dict(), "degree_sage.pt")


if __name__ == "__main__":
    train()`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          <code>main.py</code>
        </h2>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-3.5">
          <code className="language-python">{`import torch
from train import DegreeSAGE, make_graph


def evaluate(num_graphs=100):
    model = DegreeSAGE()
    model.load_state_dict(torch.load("degree_sage.pt", map_location="cpu"))
    model.eval()

    total_correct = 0
    total_nodes = 0

    with torch.no_grad():
        for _ in range(num_graphs):
            d = make_graph()
            pred = torch.round(model(d))
            total_correct += int((pred == d.y).sum().item())
            total_nodes += d.y.numel()

    acc = 100.0 * total_correct / max(total_nodes, 1)
    print(f"Node-level exact accuracy: {acc:.2f}%")


if __name__ == "__main__":
    evaluate()`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">What to try next</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Once this works, swap the label from degree to min-cycle, keep the same structure, and
          compare how accuracy changes across architectures.
        </p>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/docs/module-cage.html" label="Cage generation" direction="back" />
        <DocsNextButton href="/" label="Overview" />
      </div>
    </DocsLayout>
  );
};

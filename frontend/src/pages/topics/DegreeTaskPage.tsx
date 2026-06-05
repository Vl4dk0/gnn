import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { EditorLinks } from "../../components/docs/EditorLinks";
import { useHighlight } from "../../hooks/useHighlight";

export const DegreeTaskPage = () => {
  useHighlight();

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Degree subproblem
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          The first subproblem: can a GNN tell a vertex's degree? k-regularity is one half of the
          definition. Hence this is important.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Why degree, and why first</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Every vertex in a (k,g)-graph must have degree exactly <code>k</code>. A construction
          model that cannot reliably read degree from a partial graph will never produce a k-regular
          result.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The task is regression: given a graph, predict the degree of every vertex as a continuous
          value. The label is the true degree, loss is mean squared error. A well-trained model
          should reach near-zero MSE because one round of message passing is sufficient to count a
          vertex's neighbours.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Training step</h2>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`def train_step(model: BaseGNN, optimizer: torch.optim.Optimizer, data: Data) -> float:
    optimizer.zero_grad()
    out: torch.Tensor = model(data).squeeze()       # one scalar per node
    loss: torch.Tensor = F.mse_loss(out, data.y)    # data.y holds the true degrees
    loss.backward()
    optimizer.step()
    return loss.item()`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Play with degree prediction models in this editor.
          <br />
          You can choose prediction models in settings (gear icon in the top right corner of the
          editor)
        </p>
        <EditorLinks
          links={[
            {
              href: "/degree",
              label: "Open"
            }
          ]}
        />
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/" direction="back" label="Overview" />
        <DocsNextButton href="/min-cycle-task" label="Min-cycle subproblem" />
      </div>
    </DocsLayout>
  );
};

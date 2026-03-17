import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { useHighlight } from "../../hooks/useHighlight";

export const DocsModuleCagePage = () => {
  useHighlight();

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Cage generation
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          The prediction tasks are preparation. The long-term objective is to help search for{" "}
          <code>(k,g)-graphs</code>.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it interactively</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The generation page lets you launch search for a <code>(k,g)-graph</code>, inspect
          intermediate states, and follow the progress of a running attempt.
        </p>
        <a href="/cage" className="ui-button-outline ui-surface-link mt-3">
          Open generation editor
        </a>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Why a search-based approach came first
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          A direct constructive search is the most natural baseline. If the goal is to build a graph
          that satisfies regularity and girth constraints, then a heuristic search procedure
          provides a straightforward place to score partial states and expand promising ones.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`# ai/cage/functions/astar.py
for succ_graph in successors:
    graph_h = graph_hash(succ_graph)
    if graph_h in self.visited_hashes:
        self.duplicates_skipped += 1
        continue

    self.visited_hashes.add(graph_h)
    succ_score = self._score_graph(succ_graph)

    heapq.heappush(self.pq, (-succ_score, self.counter, succ_graph))`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          What broke in the guided-A* idea
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The weak point was not the search loop itself. The weak point was creating a{" "}
          <code>good dataset</code>
          for training. A guide model needs informative examples of partial states, especially
          examples <code>close to the decision boundary</code> between useful and useless
          expansions.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Finding those situations (training examples) turned out to be harder than expected,
          eventually deemed <code>infeasible</code> at this point.
        </p>
      </DocsCard>

      <DocsCard>
        <figure className="rounded-[10px] p-0">
          <img
            src="/static/boundary-examples.png"
            alt="Illustration of examples near a decision boundary"
            className="block w-full rounded-lg border border-line2 bg-bg1"
          />
          <figcaption className="mt-2 text-[0.9rem] leading-[1.55] text-textMuted">
            States close to the boundary carry the strongest learning signal, but they are also the
            hardest examples to collect.
          </figcaption>
        </figure>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Why PPO became more attractive
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          Reinforcement learning avoids part of that dataset problem by learning from interaction
          rather than from a fixed archive of labeled search states.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The PPO path reframes the task as sequential graph editing with rewards for moving toward
          valid constructions. I let the model explore the space of graph edits, and it learns to
          prefer actions that lead to better states.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The reward structure was developed iteratively. I knew <code>instinctively</code> that
          there should be a positive reward for adding an edge, I was sure I wanted the model to run
          indefinitely, so I added an action to remove an edge. To disable model from adding and
          removing the same edge over and over, I knew that the reward for{" "}
          <code>removing an edge</code> should be <code>less than</code>{" "}
          <code>-add_edge_reward</code>. I wanted to help model understand what I want from it by
          giving it a small extra rewards for satisfying regularity and girth constraints, but
          again, those rewards couldn't be too high, to avoid the model coming close to such states
          and then adding and removing the same edge over and over for{" "}
          <code>positive score trade</code>.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          I also had to solve the issue with <code>disconnecting</code> the graph by{" "}
          <code>removing an edge</code>. At first, I gave it <code>invalid action</code> penalty,
          but that was too hard for the model to understand and it never stopped trying to
          disconnect the graph. Then I simply <code>disallowed</code> disconnecting actions by
          <code>removing</code> them from the <code>action space</code>. Similarly, I removed an
          option to add an edge s.t. the girth constraint would be violated, simply to help the
          model.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          You can determine whether the girth constraint would be validated with simple O(n)
          algorithm. If added edge <code>breaks</code> the girth contraint, the <code>new</code>{" "}
          shortest cycle must <code>include</code> that edge. So we can just check the{" "}
          <code>shortest path</code> between two endpoints of that edge (before it was added), and
          if it's shorter than <code>g-1</code>, then adding that edge would create a cycle shorter
          than <code>g</code>. So this step wouldn't add much computational overhead, but it helps
          filtering out a large portion of invalid actions.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          I started training by giving the model <code>random</code> k and g values and gave it{" "}
          <code>5000</code> steps to find a solution. That turned out to take too long{" "}
          <code>without</code> any significant <code>learning signal</code>. So I switched to
          <code>curriculum</code> learning, starting with easier tasks. I <code>ordered</code> (k,g)
          pairs by <code>moores bound</code> and started off by giving the model <code>2000</code>{" "}
          steps to find a solution for the first pair. Only once it achieved at least{" "}
          <code>50%</code> success rate on last 8 episodes, I allowed next pair to be sampled. The
          sample rate is always <code>3:1</code> in favor of the new pair. The <code>1</code> in{" "}
          <code>3:1</code> represents all previous pairs mixed. So after <code>4</code> pairs are
          unlocked, there is <code>75%</code> chance to get 4th pair, <code>18.75%</code> chance to
          get 3rd pair, <code>~4.7%</code> change to get 2nd pair and <code>~1.6%</code> chance to
          get the 1st pair.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The final reward structure is as follows:
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">
            {`# ai/cage/rl/env.py (constants)
SCORE_FLOOR: float = -20.0  # do not go beneath this
SUCCESS_REWARD: float = 50.0  # when found the correct thing
INVALID_PENALTY: float = -0.005  # if you make an invalid action
ADD_REWARD: float = 0.01  # add edge
REMOVE_PENALTY: float = -0.1  # remove edge
SATISFY_BONUS: float = 0.05  # if graph became k-regular or girth became correct`}
          </code>
        </pre>
      </DocsCard>

      <DocsCard>
        <p className="text-base leading-[1.7] text-textMuted">
          Currenlty, the PPO-based approach is still being trained, results so far are promising. An
          RL-agent is able to find small <code>(3,5)</code> and <code>(3,6)</code> graphs fast and
          consistently.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          I will start training the model for speed, reducing its number of steps to find a
          solution, once it can reliably find solutions for all <code>(k,g)</code> pairs with
          moore's bound up to
          <code>30</code>.
        </p>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/docs/module-assessment" label="Assessment" direction="back" />
        <DocsNextButton href="/docs/training" label="Do It Yourself" />
      </div>
    </DocsLayout>
  );
};

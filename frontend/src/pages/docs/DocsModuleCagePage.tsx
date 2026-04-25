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
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Curriculum learning and the initial reward structure
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
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
          The initial reward structure was as follows:
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">
            {`# ai/cage/rl/env.py — initial reward constants
SCORE_FLOOR: float = -20.0      # do not go beneath this
SUCCESS_REWARD: float = 50.0    # when found the correct thing
INVALID_PENALTY: float = -0.005 # if you make an invalid action
ADD_REWARD: float = 0.01        # add edge
REMOVE_PENALTY: float = -0.1    # remove edge
SATISFY_BONUS: float = 0.05     # if graph became k-regular or girth became correct

# With these rewards, the agent achieved only ~1.6% success rate on (3,5)-cage.`}
          </code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Iterating on the reward
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The reward structure above looked reasonable on paper, but in practice the agent barely
          learned anything. After <code>100k</code> training steps, the success rate on the easiest
          cage <code>(3,5)</code> was just <code>1.6%</code>. The core problem was that the reward
          signal was almost entirely <code>sparse</code>: the agent received <code>+0.01</code> per
          edge addition and a <code>+50</code> jackpot on success that almost never fired. There was
          no directional signal telling it whether it was getting <code>closer</code> to a valid
          cage.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The <code>10:1</code> asymmetry between the remove penalty (<code>-0.1</code>) and the
          add reward (<code>+0.01</code>) was also crippling. Cage construction inherently requires
          backtracking — you add an edge, realize it blocks k-regularity somewhere, and need to undo
          it. But the agent was terrified of removing edges, so it just kept adding until it got
          stuck in a dead-end state for the remaining <code>~1985</code> steps of a 2000-step
          episode.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The fix was <code>progress-based reward shaping</code>. I defined a potential function
          that measures how close the current graph is to a valid cage — combining{" "}
          <code>regularity progress</code> (fraction of nodes with correct degree) and{" "}
          <code>edge progress</code> (edges placed vs target). Each step, the agent receives the
          change in potential as a bonus: positive when making progress, negative when regressing.
          I also reduced the remove penalty to <code>-0.02</code>, shortened episodes to{" "}
          <code>500</code> steps, and increased the entropy coefficient for more exploration.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">
            {`# ai/cage/rl/env.py — updated constants
REMOVE_PENALTY: float = -0.02  # was -0.1

# Progress-based shaping added to step():
# Initially used: reward += 0.99 * potential(s') - potential(s)
# Fixed to:       reward += potential(s') - potential(s)
# where potential = 5.0 * (0.6 * regularity_score + 0.4 * edge_progress)

# Also changed:
# episode_steps: 2000 -> 500
# entropy_coef:  0.01 -> 0.05`}
          </code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          The gamma discount trap
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The first version of the shaping used a <code>discount factor</code>{" "}
          <code>{"γ=0.99"}</code> in the formula, following the theoretical framework from Ng et
          al. (1999): <code>{"reward += γ·φ(s') - φ(s)"}</code>. This is mathematically guaranteed
          to preserve the optimal policy. In practice, it created a disaster.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Over a 500-step episode, the cumulative shaped reward telescopes to{" "}
          <code>{"0.99·φ(T) - φ(0) + (-0.01)·Σφ(t)"}</code>. That last term is a{" "}
          <code>hidden per-step penalty</code> for maintaining high potential. Once the agent built
          a graph close to a valid cage (high potential) but couldn't finish it, it bled{" "}
          <code>-0.04</code> reward <code>every step</code> just for existing at high potential.
          Over 450 steps: <code>-18.0</code> cumulative penalty. The agent learned to{" "}
          <code>avoid building good graphs</code> because it couldn't complete them — the exact
          opposite of the intended behavior.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The fix was simple: drop the discount and use <code>{"φ(s') - φ(s)"}</code> directly. The
          cumulative shaped reward then equals <code>{"φ(T) - φ(0)"}</code> — always non-negative,
          no per-step penalty. After this change, both GIN and GPS models maintained positive
          success rates throughout training instead of collapsing to zero.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Reward iteration summary
        </h2>
        <div className="overflow-x-auto">
          <table className="mt-2 w-full text-sm text-textMuted">
            <thead>
              <tr className="border-b border-line2 text-left text-textSub">
                <th className="py-2 pr-4 font-semibold">Version</th>
                <th className="py-2 pr-4 font-semibold">Key change</th>
                <th className="py-2 font-semibold">(3,5) success</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">v1: Sparse only</td>
                <td className="py-2 pr-4">+0.01 per edge, +50 on success, remove = -0.1</td>
                <td className="py-2">1.6%</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">{"v2: Shaping (γ=0.99)"}</td>
                <td className="py-2 pr-4">{"+ potential shaping, remove → -0.02, episodes → 500"}</td>
                <td className="py-2">{"23% peak → collapsed to 0%"}</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 font-medium text-textMain">v3: Undiscounted</td>
                <td className="py-2 pr-4">{"Drop γ, use φ(s') - φ(s) directly"}</td>
                <td className="py-2">30-60%, stable</td>
              </tr>
            </tbody>
          </table>
        </div>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Training on PERUN supercomputer
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          To scale up training, I moved to the{" "}
          <a
            href="https://www.hpc.tuke.sk/en/perun-supercomputer-info"
            className="text-accent1 hover:underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            PERUN supercomputer
          </a>{" "}
          at TUKE, equipped with NVIDIA H200 GPUs. This allowed training multiple model variants in
          parallel for extended periods.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          For cage generation, I tested <code>GIN</code> (hidden 64 and 96) and{" "}
          <code>GPS</code> (Graph Transformer with GIN convolution) at various hidden dimensions
          and attention head counts. GPS consistently outperformed GIN — the attention mechanism
          captures <code>global graph structure</code> in a single layer, which helps the agent
          reason about girth constraints that depend on distant parts of the graph. Plain GIN
          only propagates information locally, hop by hop. Scaling up GPS to <code>h128</code>{" "}
          with <code>8 attention heads</code> produced the best results by a wide margin.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Results so far</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The best performing model is <code>GPS h128</code> with 8 attention heads and GIN
          convolution. It achieved over <code>30%</code> success rate on the{" "}
          <code>(3,5)</code>-cage and over <code>20%</code> on <code>(3,6)</code>, unlocking
          three progressive stages including <code>(4,5)</code>. The model reached these
          milestones within the first 20 minutes of training — dramatically faster than smaller
          variants.
        </p>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-sm text-textMuted">
            <thead>
              <tr className="border-b border-line2 text-left text-textSub">
                <th className="py-2 pr-4 font-semibold">Model</th>
                <th className="py-2 pr-4 font-semibold">(3,5) success</th>
                <th className="py-2 pr-4 font-semibold">(3,6) success</th>
                <th className="py-2 pr-4 font-semibold">Stages unlocked</th>
                <th className="py-2 font-semibold">Best reward</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">GPS h128, 8 heads</td>
                <td className="py-2 pr-4">30.6%</td>
                <td className="py-2 pr-4">21.7%</td>
                <td className="py-2 pr-4">3 (incl. (4,5))</td>
                <td className="py-2">42.64</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">GPS h64, 4 heads</td>
                <td className="py-2 pr-4">9.7%</td>
                <td className="py-2 pr-4">3.5%</td>
                <td className="py-2 pr-4">2</td>
                <td className="py-2">52.75</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">GIN h64</td>
                <td className="py-2 pr-4">17.1%</td>
                <td className="py-2 pr-4">1.9%</td>
                <td className="py-2 pr-4">3</td>
                <td className="py-2">50.18</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 font-medium text-textMain">GIN h32 (no shaping)</td>
                <td className="py-2 pr-4">1.6%</td>
                <td className="py-2 pr-4">-</td>
                <td className="py-2 pr-4">0</td>
                <td className="py-2">28.41</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-base leading-[1.7] text-textMuted">
          Training continues on PERUN, and these models may achieve even better results with
          longer runs.
        </p>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/docs/module-assessment" label="Assessment" direction="back" />
        <DocsNextButton href="/docs/training" label="Do It Yourself" />
      </div>
    </DocsLayout>
  );
};

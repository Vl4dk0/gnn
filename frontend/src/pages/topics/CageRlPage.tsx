import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { EditorLinks } from "../../components/docs/EditorLinks";
import { useHighlight } from "../../hooks/useHighlight";

export const CageRlPage = () => {
  useHighlight();

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Reinforcement learning
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          Build a (k,g)-graph edge by edge with a learned policy. Illegal moves are masked before
          the policy sees them. A potential-based reward tracks progress toward a valid cage.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">State, action, and masking</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The <strong>state</strong> is the partially built graph. The <strong>action</strong> is
          choosing an unordered vertex pair and adding or removing that edge. Before the policy
          scores actions, illegal moves are masked out: an edge cannot be added if it would push a
          vertex past degree <code>k</code>, create a cycle shorter than <code>g</code>, or
          disconnect the graph below the Moore-bound size.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The <strong>reward</strong> uses potential-based shaping: the per-step reward is{" "}
          <code>Φ(s′) − Φ(s)</code>, where <code>Φ</code> measures how close the current graph
          is to being k-regular on Moore-bound-many vertices. A large terminal bonus is added when
          the graph is a valid (k,g)-graph. Shaping keeps the reward dense enough for learning
          without biasing the optimal policy. A curriculum advances to harder (k,g) targets once
          the agent succeeds on at least half of its recent episodes.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Environment step</h2>
        <p className="mb-2.5 text-base leading-[1.7] text-textMuted">
          Legality check and reward shaping happen inside <code>env.step</code>.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`# ai/cage/rl/env.py
def step(
    self, action_idx: int
) -> tuple[Data, float, bool, dict[str, int | str | bool | list[int] | float]]:
    self.current_step += 1
    reward = 0.0
    done = False

    u, v = self.idx_to_edge(action_idx)

    if self.graph.has_edge(u, v):
        if self._can_remove_edge(u, v):
            self.graph.remove_edge(u, v)
            reward += self.REMOVE_PENALTY
        else:
            reward += self.INVALID_PENALTY
    else:
        if self._can_add_edge(u, v):
            self.graph.add_edge(u, v)
            reward += self.ADD_REWARD
        else:
            reward += self.INVALID_PENALTY

    # Potential-based shaping: reward = Phi(s') - Phi(s)
    new_potential = self._potential()
    reward += new_potential - self._prev_potential
    self._prev_potential = new_potential

    girth_ok = self._is_girth_satisfied_active()
    is_k_regular = self._is_k_regular_active()
    if girth_ok and is_k_regular:
        reward += self.SUCCESS_REWARD
        done = True

    self.episode_score += reward
    return self._get_obs(), reward, done, {}`}</code>
        </pre>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          This is where the girth constraint actually lives.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">{`# ai/cage/rl/env.py
def _can_add_edge(self, u: int, v: int) -> bool:
    if self.graph.has_edge(u, v):
        return False
    if not self._can_add_by_active_rule(u, v):
        return False
    if self.graph.degree(u) >= self.k or self.graph.degree(v) >= self.k:
        return False
    # Adding (u, v) closes a cycle of length shortest_path(u, v) + 1.
    # That cycle must be at least g, so we need shortest_path(u, v) + 1 >= g.
    try:
        path_len = int(nx.shortest_path_length(self.graph, u, v))
        return (path_len + 1) >= self.g
    except nx.NetworkXNoPath:
        return True   # u and v are disconnected, no cycle can be closed`}</code>
        </pre>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it</h2>
        <p className="mb-3 text-base leading-[1.7] text-textMuted">
          Open the cage editor and choose the RL agent to watch the policy build a graph step by
          step from an empty graph to a valid (k,g)-graph.
        </p>
        <EditorLinks
          links={[
            {
              href: "/cage?from=rl",
              label: "Open the cage editor",
              description: "Watch the RL agent build a (k,g)-graph edge by edge.",
            },
          ]}
        />
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/cage/astar" direction="back" label="A* + backtracking" />
        <DocsNextButton href="/cage/voltage" label="Voltage lifts" />
      </div>
    </DocsLayout>
  );
};

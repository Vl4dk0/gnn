import { DocsCard, DocsHero } from "../../components/docs/DocsCard";
import { DocsLayout } from "../../components/docs/DocsLayout";
import { DocsNextButton } from "../../components/docs/DocsNextButton";
import { useHighlight } from "../../hooks/useHighlight";

export const DocsVoltagePage = () => {
  useHighlight();

  return (
    <DocsLayout>
      <DocsHero>
        <h1 className="mb-2.5 text-[clamp(1.7rem,3.1vw,2.4rem)] font-bold leading-[1.22] text-textMain">
          Voltage graph lifts
        </h1>
        <p className="text-base leading-[1.7] text-textMuted">
          A fundamentally different approach to cage construction — instead of building graphs
          edge by edge, use algebraic structure to generate large regular graphs from tiny
          blueprints.
        </p>
      </DocsHero>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">Try it interactively</h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The cage generation editor supports voltage-lifted graphs. Launch a search and watch
          the agent assign voltages to a base graph in real time.
        </p>
        <a href="/cage" className="ui-button-outline ui-surface-link mt-3">
          Open generation editor
        </a>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Why edge-by-edge construction hits a wall
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The RL agent from the previous chapter builds graphs one edge at a time. It must
          simultaneously learn <code>k-regularity</code> (every node has exactly k edges) and{" "}
          <code>girth</code> (no short cycles). The action space is{" "}
          <code>{"O(n²)"}</code> possible edge pairs, episodes are 500 steps long, and the agent
          barely cracked the <code>(4,5)</code>-cage at 0.1% success rate after days of training.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The core problem: the agent is fighting two constraints at once in a massive search
          space. But there is an algebraic shortcut that eliminates one constraint entirely and
          shrinks the search space by orders of magnitude.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          The voltage lift idea
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          A voltage graph lift takes three ingredients: a tiny <code>base graph</code> (2-6
          nodes), a finite <code>group</code> (like the cyclic group Z_n), and a{" "}
          <code>voltage assignment</code> — one group element per edge of the base graph.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The lift produces a large graph with <code>base_nodes × |group|</code> vertices. For
          each edge in the base graph with voltage <code>α</code>, the lift connects vertex{" "}
          <code>(u, g)</code> to vertex <code>(w, g·α)</code> for every group element{" "}
          <code>g</code>. The key property: if the base graph is k-regular, the lift is{" "}
          <code>always k-regular</code>. Regularity is guaranteed by construction — the agent
          only needs to optimize girth.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          Think of it as designing a blueprint and stamping it out, rather than placing bricks
          one by one. A 2-node base graph with 3 parallel edges and a cyclic group of order 7
          produces a 14-vertex 3-regular graph in one step.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          How girth maps to algebra
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The girth of the lifted graph equals the length of the shortest <code>closed walk</code>
          {" "}in the base graph whose net voltage product equals the <code>identity element</code>.
          This can be checked on the tiny base graph without ever building the full lift — just
          enumerate short walks and multiply voltages along the way.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">
            {`# ai/cage/voltage/cycle_analysis.py (simplified)
def count_short_identity_walks(base: BaseGraph, group: FiniteGroup, voltages, g_target):
    # We explore all walks up to length g_target in the tiny base graph
    # and keep a running product of the edge voltages.
    # If the net voltage is the identity element, we found a short cycle in the lift!
    identity = group.identity()
    count = 0
    for start in range(base.num_nodes):
        frontier = [...] # init with neighbors
        for depth in range(1, g_target):
            next_frontier = []
            for node, net_v, length, prev_arc in frontier:
                if node == start and net_v == identity and length >= 3:
                    count += 1
                    continue
                # explore further...
                new_v = group.mult(net_v, arc_voltage[arc_id])
                next_frontier.append((next_node, new_v, length + 1, arc_id))
            frontier = next_frontier
    return count`}
          </code>
        </pre>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          For example, a <code>dumbbell(3)</code> base (2 nodes, 3 parallel edges) with the
          cyclic group <code>Z_7</code> and voltages <code>[4, 2, 1]</code> produces a lift with
          14 vertices and girth 6. This is the Heawood graph — the optimal{" "}
          <code>(3,6)</code>-cage.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Comparison with edge-by-edge
        </h2>
        <div className="overflow-x-auto">
          <table className="mt-2 w-full text-sm text-textMuted">
            <thead>
              <tr className="border-b border-line2 text-left text-textSub">
                <th className="py-2 pr-4 font-semibold">Aspect</th>
                <th className="py-2 pr-4 font-semibold">Edge-by-edge RL</th>
                <th className="py-2 font-semibold">Voltage lifts</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">Action space</td>
                <td className="py-2 pr-4">{"O(n²)"} node pairs (100-1000)</td>
                <td className="py-2">|group| elements (20-100)</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">Episode length</td>
                <td className="py-2 pr-4">500 steps</td>
                <td className="py-2">3-15 steps</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">Regularity</td>
                <td className="py-2 pr-4">Must learn (hard constraint)</td>
                <td className="py-2">Automatic by construction</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">Girth check</td>
                <td className="py-2 pr-4">Build full graph, find cycles</td>
                <td className="py-2">Algebraic check on base (fast)</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 font-medium text-textMain">Scaling</td>
                <td className="py-2 pr-4">Linear (add nodes one by one)</td>
                <td className="py-2">Exponential (2 nodes → thousands)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          RL on voltage assignments
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          The same PPO framework from the edge-by-edge approach works here, but the task is
          radically simpler. The agent assigns one group element per edge of the base graph —
          episodes are <code>3-9 steps</code> instead of 500. A GINEConv encoder processes the
          base graph with partial voltage assignments, and an MLP actor head outputs a
          distribution over group elements.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          I added <code>progress-based shaping</code> here too: after each voltage assignment,
          the environment checks whether the partial assignment already creates short identity
          walks (girth violations). If a new assignment introduces violations, the agent receives
          a penalty proportional to the number of new violations. This gives per-step directional
          signal even though the final girth can only be evaluated when all edges are assigned.
        </p>
        <pre className="mt-2.5 overflow-x-auto rounded-lg border-2 border-line2 bg-bg1 p-1.5">
          <code className="language-python">
            {`# ai/cage/voltage/rl_env.py — per-step shaping
violations = count_short_identity_walks(base, group, partial_voltages, g_target)
new_violations = violations - prev_violations
if new_violations > 0:
    reward -= new_violations * 0.5  # penalize creating short cycles`}
          </code>
        </pre>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The results were immediate: the voltage RL agent unlocked <code>6 out of 8</code>{" "}
          progressive stages within <code>10 hours</code> of training, reaching{" "}
          <code>(3,10)</code>-cages (Moore bound 70). The edge-by-edge agent struggled to crack{" "}
          <code>(4,5)</code> (Moore bound 17) after 48 hours.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Search approaches
        </h2>
        <p className="text-base leading-[1.7] text-textMuted">
          RL is not the only way to find good voltage assignments.{" "}
          <code>Tabu search</code> iteratively improves an assignment by changing one voltage at a
          time, always picking the change that reduces the number of girth violations the most. A{" "}
          <code>tabu list</code> prevents the algorithm from undoing recent changes, forcing it to
          explore new regions of the search space instead of cycling between the same states.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          The algebraic cost function (<code>count_short_identity_walks</code>) is cheap and
          exact — it operates on the tiny base graph, not the full lift. This makes tabu search
          extremely fast: thousands of configurations tested per second. Random search over
          voltage assignments is also effective for smaller groups and was enough to find the
          optimal <code>(3,5)</code> and <code>(3,6)</code> cages.
        </p>
        <p className="mt-2.5 text-base leading-[1.7] text-textMuted">
          ML and search complement each other: RL explores adaptively and learns patterns across
          episodes, while search explores exhaustively within a single (base, group) configuration.
          Running both in parallel covers different parts of the solution space.
        </p>
      </DocsCard>

      <DocsCard>
        <h2 className="mb-2.5 text-[1.3rem] font-bold text-textMain">
          Results and outlook
        </h2>
        <div className="overflow-x-auto">
          <table className="mt-2 w-full text-sm text-textMuted">
            <thead>
              <tr className="border-b border-line2 text-left text-textSub">
                <th className="py-2 pr-4 font-semibold">Method</th>
                <th className="py-2 pr-4 font-semibold">Highest girth reached</th>
                <th className="py-2 font-semibold">Notes</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">Voltage RL (GINE h64)</td>
                <td className="py-2 pr-4">(3,10) — 6/8 stages</td>
                <td className="py-2">10 hours of training</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">Voltage RL (GINE h128)</td>
                <td className="py-2 pr-4">(3,8) — 4/8 stages</td>
                <td className="py-2">Slower convergence, needs more training</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-4 font-medium text-textMain">Meta search (random + tabu)</td>
                <td className="py-2 pr-4">(3,6) — optimal cage</td>
                <td className="py-2">Heawood graph found in minutes</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 font-medium text-textMain">Edge-by-edge RL (best)</td>
                <td className="py-2 pr-4">(4,5) — 0.1% success</td>
                <td className="py-2">48 hours of training</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-base leading-[1.7] text-textMuted">
          The voltage approach is a qualitative leap over edge-by-edge construction. However, it
          has a known limitation: with <code>cyclic groups</code> (which are abelian), there is a
          ceiling around girth 12 for certain base graphs. Reaching higher girths requires{" "}
          <code>non-abelian groups</code> — dihedral, semidirect products, or matrix groups like
          SL(2,p). These are implemented but dramatically expand the search space. Training
          continues on the PERUN supercomputer with both RL and search approaches.
        </p>
      </DocsCard>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-5">
        <DocsNextButton href="/docs/module-cage" label="Cage generation" direction="back" />
        <DocsNextButton href="/docs/cayley" label="Cayley graphs" />
      </div>
    </DocsLayout>
  );
};

/* ============================================================
   Track 7 — Reinforcement Learning and Alignment
   ============================================================ */

import { t, key, intuition, warn, hist, mathnote, viz, deriv, code, quiz, paper, book, course, blog, video, demo, codeRef } from './_helpers.js';

export default [

/* ---------------------------------------------------------- */
{
  id: 'rl-mdp',
  title: 'MDPs, Value, and the Bellman Equation',
  sub: 'Learning from consequences rather than answers.',
  mins: 26, level: 'core',
  prereq: ['math-probability'],
  tags: ['RL', 'MDP', 'dynamic programming'],
  sections: [
    t(`## A different learning problem

Supervised learning gives you the right answer for each input. RL gives you a **reward** — a scalar saying how well
things went, often much later, without telling you which action caused it.

Three difficulties follow, and they are what make RL hard:

- **Credit assignment.** Which of the last 200 moves lost the game?
- **Exploration.** You only learn about actions you take, so you must sometimes act suboptimally to discover anything.
- **Non-stationarity.** Your data distribution depends on your policy, which is changing.`),

    t(`## Formalization

A **Markov decision process** is $(\\mathcal{S}, \\mathcal{A}, P, R, \\gamma)$: states, actions, transition
probabilities $P(s'\\mid s,a)$, rewards $R(s,a)$, and a discount factor $\\gamma\\in[0,1)$.

The **Markov property** — the future depends on the present state alone, not the history — is what makes any of this
tractable. It is also frequently false, and much of practical RL is about engineering a state representation that
makes it approximately true.

A **policy** $\\pi(a\\mid s)$ maps states to action distributions. The goal is to maximize expected discounted return:

$$G_t = \\sum_{k=0}^{\\infty}\\gamma^k r_{t+k}$$

$\\gamma$ does two jobs: it encodes preference for sooner rewards, and it keeps the sum finite. At $\\gamma = 0.99$ a
reward 100 steps away is worth 37% of an immediate one.`),

    t(`## Value functions and Bellman

$$V^\\pi(s) = \\mathbb{E}_\\pi[G_t \\mid s_t = s], \\qquad Q^\\pi(s,a) = \\mathbb{E}_\\pi[G_t\\mid s_t=s, a_t=a]$$

"How good is this state (or this action here), assuming I follow $\\pi$ afterwards?"

The **Bellman equation** is the recursive consistency condition that makes value computable:

$$V^\\pi(s) = \\sum_a \\pi(a\\mid s)\\sum_{s'}P(s'\\mid s,a)\\big[R(s,a) + \\gamma V^\\pi(s')\\big]$$

And for the optimal policy, the **Bellman optimality equation**:

$$V^*(s) = \\max_a \\sum_{s'}P(s'\\mid s,a)\\big[R(s,a)+\\gamma V^*(s')\\big]$$`),

    viz('value-iteration'),

    t(`Press play and watch value propagate outward from the goal, one square per sweep. **That is credit assignment
happening** — information about a distant reward diffusing backward through the state space.

The update is a contraction mapping with modulus $\\gamma$, so it converges to a unique fixed point regardless of
initialization. That guarantee is why value iteration always works when you know $P$ and $R$.`),

    t(`Things worth trying in that figure:

- **Lower $\\gamma$** to 0.6: distant rewards stop mattering and the policy far from the goal becomes indifferent.
- **Raise the slip probability**: the optimal policy starts routing *away* from the tiles adjacent to the −1 trap,
  even though that path is longer. It is buying safety margin against its own unreliability.
- **Make the living reward more negative**: the agent becomes impatient and accepts riskier shortcuts.`),

    key(`Value iteration requires knowing $P$ and $R$ — a **model** of the environment. You almost never have one.

Everything in the next lesson is about the model-free case: estimating values from sampled experience instead of
computing them from known dynamics.`),

    code('Value iteration and policy iteration', `import numpy as np

# 4x4 gridworld: goal at (3,3)=+1, trap at (3,1)=-1, walls
W, H = 4, 4
walls = {(1,1), (1,2)}
goal, trap = (3,3), (3,1)
ACTIONS = [(0,1),(1,0),(0,-1),(-1,0)]
ARROW = "^>v<"

def states():
    return [(x,y) for x in range(W) for y in range(H) if (x,y) not in walls]

def step(s, a, slip=0.2):
    """Returns [(prob, next_state, reward)]."""
    if s in (goal, trap): return [(1.0, s, 0.0)]
    out = []
    for ai, p in [(a, 1-slip), ((a+1)%4, slip/2), ((a+3)%4, slip/2)]:
        nx, ny = s[0]+ACTIONS[ai][0], s[1]+ACTIONS[ai][1]
        ns = (nx,ny) if (0<=nx<W and 0<=ny<H and (nx,ny) not in walls) else s
        r = 1.0 if ns == goal else (-1.0 if ns == trap else -0.04)
        out.append((p, ns, r))
    return out

def value_iteration(gamma=0.95, tol=1e-10):
    V = {s: 0.0 for s in states()}
    for it in range(1000):
        delta = 0
        for s in states():
            if s in (goal, trap): continue
            best = max(sum(p*(r + gamma*V[ns]) for p,ns,r in step(s,a))
                       for a in range(4))
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < tol:
            return V, it+1
    return V, 1000

V, iters = value_iteration()
print(f"converged in {iters} sweeps\\n")
for y in reversed(range(H)):
    row = ""
    for x in range(W):
        s = (x,y)
        if s in walls: row += "  ####  "
        elif s == goal: row += "   +1   "
        elif s == trap: row += "   -1   "
        else:
            a = max(range(4), key=lambda a: sum(p*(r+0.95*V[(nx,ny)])
                    for p,(nx,ny),r in step(s,a)))
            row += f" {V[s]:+.2f}{ARROW[a]} "
    print(row)

print("\\neffect of the discount factor on the value of the start state:")
for g in [0.5, 0.9, 0.95, 0.99]:
    Vg, _ = value_iteration(g)
    print(f"  gamma={g:.2f}: V(0,0)={Vg[(0,0)]:+.4f}   "
          f"reward 10 steps away is worth {g**10:.3f}")`),

    quiz('Why does the optimal policy avoid tiles next to the trap when the slip probability is high?',
      ['Expected value accounts for the chance of slipping into the −1 square, so a longer safer path can be worth more',
       'The agent is programmed to be cautious',
       'The trap has a larger reward magnitude than the goal',
       'Value iteration has not converged'],
      0,
      'The Bellman backup takes an expectation over the transition distribution. Standing next to the trap with a 20% chance of slipping sideways means a 10% chance of falling in, which drags down that state\'s value. The policy is not "cautious" as a design choice — it is maximizing expected return, and expected return already prices in the agent\'s own unreliability.'),
  ],
  refs: [
    book('Reinforcement Learning: An Introduction', 'Sutton & Barto', 2018, 'http://incompleteideas.net/book/the-book-2nd.html', 'Free PDF. The textbook — clear, complete, and still the right place to start.'),
    course('RL Course', 'David Silver (DeepMind/UCL)', 2015, 'https://www.davidsilver.uk/teaching/', 'The lecture series everyone learns from.'),
    course('Spinning Up in Deep RL', 'OpenAI', 2018, 'https://spinningup.openai.com/', 'Practical, with clean reference implementations.'),
    book('Algorithms for Reinforcement Learning', 'Csaba Szepesvári', 2010, 'https://sites.ualberta.ca/~szepesva/rlbook.html', 'Short and rigorous, if you want the theory.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'rl-model-free',
  title: 'Q-Learning, Exploration, and Deep RL',
  sub: 'Learning from experience when you have no model of the world.',
  mins: 26, level: 'core',
  prereq: ['rl-mdp'],
  tags: ['Q-learning', 'DQN', 'exploration'],
  sections: [
    t(`## Temporal difference learning

Without $P$ and $R$ you cannot compute the Bellman backup — but you can *sample* it. Take an action, observe
$(s, a, r, s')$, and move your estimate toward what you saw:

$$Q(s,a) \\leftarrow Q(s,a) + \\alpha\\Big[\\underbrace{r + \\gamma\\max_{a'}Q(s',a')}_{\\text{TD target}} - Q(s,a)\\Big]$$

The bracketed quantity is the **TD error** — the surprise. Learning is proportional to surprise, which turns out to
match dopamine signaling in the brain remarkably well.`),

    viz('q-learning'),

    key(`**Q-learning is off-policy.** It explores with one policy (say ε-greedy) but learns the value of acting
*greedily*, because of the $\\max_{a'}$ in the target. So the arrows converge to the optimal policy even while the
agent is still wandering randomly.

**SARSA** is the on-policy sibling: it uses $Q(s',a')$ for the action actually taken. It learns the value of the policy
it is following, including its exploration — which makes it more conservative near cliffs, because it accounts for the
chance of exploring off one.`),

    t(`## Exploration vs exploitation`),

    viz('bandit'),

    t(`The bandit strips the problem to its core. **Regret** is what you lost by not always pulling the best arm.

- **Greedy** locks onto whichever arm got lucky first. Regret grows linearly — it can be wrong forever.
- **ε-greedy** fixes that but wastes ε of every pull on arms it already knows are bad.
- **UCB** is *optimistic in proportion to uncertainty*: $\\bar{x}_i + \\sqrt{2\\ln t / n_i}$. Under-explored arms get a
  bonus that shrinks as you learn, so exploration self-extinguishes. Logarithmic regret.
- **Thompson sampling** draws from the posterior and acts greedily on the draw. Also logarithmic, often better in
  practice, and trivially handles context.

In deep RL, exploration is much harder — the state space is enormous and ε-greedy is nearly useless in sparse-reward
environments. Approaches include intrinsic curiosity (reward prediction error), count-based bonuses on learned state
representations, and noise injected into parameters rather than actions.`),

    t(`## Deep Q-networks

Replace the Q-table with a neural network. Naively this diverges, and DQN's contribution was the three stabilizers
that make it work:

1. **Experience replay.** Store transitions in a buffer and train on random samples. Breaks the temporal correlation
   that violates i.i.d. assumptions, and reuses each transition many times.
2. **Target network.** Compute the TD target with a frozen copy of the network, synced every $N$ steps. Without it,
   the target moves as you update — you are chasing your own tail.
3. **Reward clipping** to $[-1,1]$, so one hyperparameter set works across games with wildly different reward scales.

**Double DQN** fixes the overestimation bias: $\\max_{a'}Q(s',a')$ is biased upward because the max of noisy estimates
exceeds the true max. Decouple action *selection* from *evaluation* — pick with the online network, evaluate with the
target.

The **deadly triad** — function approximation + bootstrapping + off-policy learning — can diverge in theory, and does
in practice. All three are present in DQN. It works anyway, with care.`),

    code('Q-learning, SARSA, and the exploration strategies', `import numpy as np
rng = np.random.default_rng(0)

# cliff walking: the classic that separates Q-learning from SARSA
W, H = 12, 4
START, GOAL = (0,0), (11,0)
CLIFF = {(x,0) for x in range(1,11)}
ACTIONS = [(0,1),(1,0),(0,-1),(-1,0)]

def step(s, a):
    nx, ny = s[0]+ACTIONS[a][0], s[1]+ACTIONS[a][1]
    if not (0 <= nx < W and 0 <= ny < H): nx, ny = s
    if (nx,ny) in CLIFF: return START, -100, False
    return (nx,ny), (0 if (nx,ny)==GOAL else -1), (nx,ny)==GOAL

def run(algo, episodes=500, alpha=0.5, eps=0.1, gamma=1.0):
    Q = np.zeros((W,H,4))
    returns = []
    for ep in range(episodes):
        s, total, done = START, 0, False
        a = rng.integers(4) if rng.random()<eps else Q[s].argmax()
        for _ in range(200):
            ns, r, done = step(s, a)
            na = rng.integers(4) if rng.random()<eps else Q[ns].argmax()
            target = r + gamma*(0 if done else
                     (Q[ns].max() if algo=="q" else Q[ns][na]))
            Q[s][a] += alpha*(target - Q[s][a])
            s, a, total = ns, na, total+r
            if done: break
        returns.append(total)
    return Q, np.mean(returns[-100:])

for algo in ["q", "sarsa"]:
    Q, avg = run(algo)
    path, s = [], START
    for _ in range(30):
        path.append(s)
        s, _, done = step(s, Q[s].argmax())
        if done or s == START: break
    print(f"{algo.upper():6s} avg return (last 100 eps): {avg:7.1f}   "
          f"greedy path height: {max(p[1] for p in path)}")
print("\\nQ-learning learns the optimal (cliff-edge) path but performs worse ONLINE,")
print("because epsilon-greedy exploration keeps knocking it off the cliff.")
print("SARSA learns a safer path because it accounts for its own exploration.\\n")

# --- bandit strategies ---
def bandit(strategy, arms=(0.3,0.5,0.55,0.2), T=2000):
    K = len(arms); n = np.zeros(K); s = np.zeros(K); regret = 0
    for t in range(1, T+1):
        if strategy == "greedy":
            k = np.argmax(np.where(n>0, s/np.maximum(n,1), 1.0))
        elif strategy == "eps":
            k = rng.integers(K) if rng.random()<0.1 else np.argmax(np.where(n>0, s/np.maximum(n,1), 1.0))
        elif strategy == "ucb":
            k = np.argmax(np.where(n>0, s/np.maximum(n,1) + np.sqrt(2*np.log(t)/np.maximum(n,1)), 1e9))
        else:
            k = np.argmax([rng.beta(s[i]+1, n[i]-s[i]+1) for i in range(K)])
        r = rng.random() < arms[k]
        n[k] += 1; s[k] += r
        regret += max(arms) - arms[k]
    return regret

print("cumulative regret over 2000 pulls (lower is better):")
for st in ["greedy", "eps", "ucb", "thompson"]:
    print(f"  {st:10s} {np.mean([bandit(st) for _ in range(20)]):7.1f}")`),

    quiz('Why does DQN need a separate target network?',
      ['Without it the TD target shifts every update, so the network chases a moving target and can diverge',
       'To reduce memory usage',
       'To enable exploration',
       'To handle continuous action spaces'],
      0,
      'The target $r + \\gamma\\max_{a\'}Q(s\',a\')$ is computed with the same network being updated. Changing the weights changes the target, which changes the gradient, which changes the weights — an unstable feedback loop. Freezing a copy for $N$ steps gives a stationary regression target within each window, turning the problem into something like supervised learning.'),
  ],
  refs: [
    paper('Human-level control through deep reinforcement learning', 'Mnih et al.', 2015, 'https://www.nature.com/articles/nature14236', 'DQN on Atari. The paper that started deep RL.'),
    paper('Deep Reinforcement Learning with Double Q-learning', 'van Hasselt, Guez & Silver', 2015, 'https://arxiv.org/abs/1509.06461', 'Fixes the overestimation bias.'),
    paper('Rainbow: Combining Improvements in Deep RL', 'Hessel et al.', 2017, 'https://arxiv.org/abs/1710.02298', 'A careful ablation of six DQN improvements. Model of how to do this well.'),
    book('Reinforcement Learning: An Introduction, Ch. 6', 'Sutton & Barto', 2018, 'http://incompleteideas.net/book/the-book-2nd.html', 'TD learning, and the cliff-walking example above.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'rl-policy-gradient',
  title: 'Policy Gradients and PPO',
  sub: 'Optimize the policy directly — the family that RLHF uses.',
  mins: 26, level: 'advanced',
  prereq: ['rl-model-free'],
  tags: ['policy gradient', 'PPO', 'actor-critic'],
  sections: [
    t(`## Why not just optimize the policy?

Value methods learn $Q$ and act greedily. That struggles with continuous or very large action spaces (the $\\max$ is
itself an optimization), and it cannot represent a stochastic optimal policy.

Policy gradient methods parameterize $\\pi_\\theta(a\\mid s)$ and do gradient ascent on expected return.`),

    deriv('The policy gradient theorem', `We want $\\nabla_\\theta J(\\theta)$ where $J(\\theta) = \\mathbb{E}_{\\tau\\sim\\pi_\\theta}[R(\\tau)]$. The difficulty is that
the *distribution* depends on $\\theta$. Use the log-derivative identity $\\nabla p = p\\,\\nabla\\log p$:

$$\\nabla_\\theta J = \\nabla_\\theta\\int p_\\theta(\\tau)R(\\tau)\\,d\\tau
= \\int p_\\theta(\\tau)\\,\\nabla_\\theta\\log p_\\theta(\\tau)\\,R(\\tau)\\,d\\tau
= \\mathbb{E}_{\\tau}\\big[\\nabla_\\theta\\log p_\\theta(\\tau)\\,R(\\tau)\\big]$$

Now, $p_\\theta(\\tau) = p(s_0)\\prod_t \\pi_\\theta(a_t\\mid s_t)P(s_{t+1}\\mid s_t,a_t)$. Taking the log turns it into a sum,
and **every term without $\\theta$ vanishes under the gradient** — including the transition dynamics. So:

$$\\nabla_\\theta J = \\mathbb{E}\\left[\\sum_t \\nabla_\\theta\\log\\pi_\\theta(a_t\\mid s_t)\\; R(\\tau)\\right]$$

We never needed to know the environment. That is the theorem's power: a model-free gradient estimate from samples
alone. ∎`),

    viz('policy-gradient'),

    t(`## Variance is the enemy

The REINFORCE estimator is unbiased and has enormous variance. Two standard reductions:

**Causality.** An action cannot affect past rewards, so replace $R(\\tau)$ with the reward-to-go
$\\sum_{t'\\ge t}\\gamma^{t'-t}r_{t'}$.

**Baselines.** Subtract any function of state:

$$\\nabla_\\theta J = \\mathbb{E}\\big[\\nabla_\\theta\\log\\pi_\\theta(a_t\\mid s_t)\\,(G_t - b(s_t))\\big]$$

This stays unbiased — $\\mathbb{E}[\\nabla\\log\\pi \\cdot b(s)] = 0$ because $\\mathbb{E}[\\nabla\\log\\pi]=0$ — and can
massively reduce variance. Toggle the baseline in the figure and watch the trajectory smooth out.

The best baseline is $V(s)$, giving the **advantage** $A(s,a) = Q(s,a)-V(s)$: "how much better than average was this
action?" Learning $V$ alongside $\\pi$ is **actor–critic**. **GAE** interpolates between low-variance/high-bias and
high-variance/low-bias advantage estimates with a parameter $\\lambda$.`),

    t(`## PPO

Policy gradient is on-policy — after one update, your collected data is stale. But throwing away a batch after one
gradient step is enormously wasteful.

You can reuse it with importance sampling, weighting by the ratio $r_t(\\theta) = \\pi_\\theta(a_t|s_t)/\\pi_{\\text{old}}(a_t|s_t)$.
The problem: if the policy drifts far, that ratio explodes and the estimate becomes garbage.

TRPO solved this with a hard KL trust region and second-order optimization. **PPO** solves it with one line:

$$\\mathcal{L}^{\\text{CLIP}} = \\mathbb{E}\\Big[\\min\\big(r_t(\\theta)\\hat A_t,\\ \\text{clip}(r_t(\\theta),1-\\epsilon,1+\\epsilon)\\hat A_t\\big)\\Big]$$`),

    viz('ppo-clip'),

    t(`Flatten the objective outside $[1-\\epsilon, 1+\\epsilon]$ and there is nothing to gain from drifting. Note the
asymmetry: with positive advantage the gradient dies once $r > 1+\\epsilon$ (no reward for pushing further); with
negative advantage it dies below $1-\\epsilon$ (you cannot crush the action to zero in one update).

Simple, first-order, and it replaced TRPO almost entirely.

**GRPO** (used by DeepSeek-R1) drops the value network altogether: sample $G$ responses per prompt and use the
group's mean reward as the baseline. For LLM RL, where a value head over token sequences is expensive and hard to
train, this is a substantial simplification.`),

    code('REINFORCE with and without a baseline', `import numpy as np
rng = np.random.default_rng(0)

# a 1-D continuous control problem
def reward(a):
    return np.exp(-((a-1.7)**2)/0.6) + 0.55*np.exp(-((a+1.6)**2)/0.5)

def train(use_baseline, batch=8, lr=0.05, sigma=0.8, steps=400):
    theta, baseline = -1.5, 0.0
    traj = []
    for t in range(steps):
        a = theta + rng.normal(0, sigma, batch)
        r = reward(a)
        adv = r - (baseline if use_baseline else 0.0)
        grad = (adv * (a - theta) / sigma**2).mean()      # d log pi / d theta
        theta += lr * grad
        baseline = 0.9*baseline + 0.1*r.mean()
        traj.append(theta)
    return np.array(traj)

for label, bl in [("no baseline ", False), ("baseline    ", True)]:
    for B in [1, 8, 64]:
        tr = train(bl, batch=B)
        # measure jitter over the last 100 steps
        print(f"{label} batch={B:3d}: final theta {tr[-1]:+.3f}  "
              f"step-to-step jitter {np.abs(np.diff(tr[-100:])).mean():.5f}")
print("\\nA baseline reduces variance without changing the fixed point.\\n")

# --- PPO clipping, on a single step ---
def ppo_obj(ratio, adv, eps=0.2):
    return np.minimum(ratio*adv, np.clip(ratio, 1-eps, 1+eps)*adv)

print("PPO clipped objective (eps=0.2):")
print(f"{'ratio':>7} {'A=+1':>8} {'A=-1':>8}  effect")
for r in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    pos, neg = ppo_obj(r, 1.0), ppo_obj(r, -1.0)
    note = ""
    if r > 1.2: note = "clipped for A>0 -> no gradient"
    elif r < 0.8: note = "clipped for A<0 -> no gradient"
    print(f"{r:7.1f} {pos:8.2f} {neg:8.2f}  {note}")`),

    quiz('Why does subtracting a baseline from the return leave the policy gradient unbiased?',
      ['E[∇log π(a|s)] = 0 for any fixed state, so the baseline term contributes zero in expectation',
       'The baseline is chosen to be small',
       'It introduces a small bias that is acceptable in practice',
       'Because the baseline is learned alongside the policy'],
      0,
      '$\\mathbb{E}_{a\\sim\\pi}[\\nabla\\log\\pi(a|s)] = \\int \\pi \\nabla\\log\\pi = \\int \\nabla\\pi = \\nabla\\int\\pi = \\nabla 1 = 0$. So $\\mathbb{E}[\\nabla\\log\\pi \\cdot b(s)] = b(s)\\cdot 0 = 0$ for any $b$ that does not depend on the action. You get a free variance reduction with no bias — one of the genuinely elegant results in RL.'),
  ],
  refs: [
    paper('Policy Gradient Methods for RL with Function Approximation', 'Sutton et al.', 1999, 'https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation', 'The theorem.'),
    paper('Proximal Policy Optimization Algorithms', 'Schulman et al.', 2017, 'https://arxiv.org/abs/1707.06347', 'PPO. Short, practical, and now ubiquitous.'),
    paper('High-Dimensional Continuous Control Using GAE', 'Schulman et al.', 2015, 'https://arxiv.org/abs/1506.02438', 'Generalized advantage estimation.'),
    paper('DeepSeekMath: Pushing the Limits of Mathematical Reasoning', 'Shao et al.', 2024, 'https://arxiv.org/abs/2402.03300', 'Introduces GRPO — policy gradient without a value network.'),
    blog('The 37 Implementation Details of PPO', 'Huang et al.', 2022, 'https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/', 'The gap between the paper and a working implementation. Sobering and essential.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'rl-rlhf',
  title: 'RLHF, DPO, and Reasoning Models',
  sub: 'How a text predictor becomes an assistant — and then a reasoner.',
  mins: 28, level: 'advanced',
  prereq: ['rl-policy-gradient', 'llm-finetuning'],
  tags: ['RLHF', 'DPO', 'alignment'],
  sections: [
    t(`## The problem SFT cannot solve

Supervised fine-tuning imitates demonstrations, so it is capped by demonstration quality. And for many things people
care about — helpfulness, tone, refusing appropriately — it is far easier for a human to **compare** two outputs than
to write the ideal one.

RLHF turns comparisons into a training signal.`),

    viz('rlhf-pipeline'),

    t(`## The reward model

Collect pairs where a human picked $y^+$ over $y^-$ for prompt $x$. Train a scalar reward model with the
Bradley–Terry objective:

$$\\mathcal{L} = -\\log\\sigma\\big(r_\\phi(x,y^+) - r_\\phi(x,y^-)\\big)$$

That is logistic regression on the score *difference*. The reward model is usually the SFT model with the unembedding
replaced by a scalar head.`),

    t(`## The RL stage

Optimize the policy against the reward model, with a KL leash back to the SFT model:

$$\\max_\\theta\\ \\mathbb{E}_{y\\sim\\pi_\\theta}\\big[r_\\phi(x,y)\\big] - \\beta\\,D_{\\text{KL}}\\big(\\pi_\\theta\\,\\|\\,\\pi_{\\text{SFT}}\\big)$$

The KL term is not a technicality — it is load-bearing.`),

    warn(`**The reward model is a proxy, and proxies get hacked.** Goodhart's law in its purest form: an unconstrained
policy will find inputs where $r_\\phi$ is high and actual quality is not. Observed failures include exploiting
formatting quirks, producing text that is confidently wrong (raters like confidence), and degenerate outputs that score
absurdly high.

The KL penalty keeps the policy in the region where the reward model was trained and is therefore still trustworthy.
Too small a $\\beta$ and you get reward hacking; too large and nothing changes. It is one of the most consequential
hyperparameters in the pipeline.

The related, well-documented side effect: RLHF **damages calibration**. Base models report probabilities that track
accuracy reasonably well; RLHF'd models are systematically overconfident, because raters prefer confident-sounding
answers regardless of correctness.`),

    t(`## DPO: skip the reward model

The RLHF objective has a closed-form optimal policy:

$$\\pi^*(y\\mid x) \\propto \\pi_{\\text{ref}}(y\\mid x)\\exp\\!\\left(\\frac{r(x,y)}{\\beta}\\right)$$

Invert it — express $r$ in terms of $\\pi^*$ — and substitute back into the Bradley–Terry loss. The reward model
cancels out entirely:

$$\\mathcal{L}_{\\text{DPO}} = -\\log\\sigma\\!\\left(\\beta\\log\\frac{\\pi_\\theta(y^+|x)}{\\pi_{\\text{ref}}(y^+|x)} - \\beta\\log\\frac{\\pi_\\theta(y^-|x)}{\\pi_{\\text{ref}}(y^-|x)}\\right)$$

**The language model is its own reward model.** No separate model, no sampling loop, no RL — just supervised learning
on preference pairs. Dramatically simpler and now the common default for open models.

The tradeoffs are real: DPO is off-policy (it trains on a fixed preference dataset rather than the policy's own
samples), and online variants that regenerate preferences with the current policy tend to outperform it. Related
methods — IPO, KTO, ORPO, SimPO — vary the loss and the reference handling.`),

    t(`## RLVR and reasoning models

The newest and most consequential shift. Instead of a learned reward model, use a **verifier**: run the unit tests,
check the final answer, compile the proof.

Verifiable rewards cannot be hacked in the way learned rewards can — the tests either pass or they do not. That lets
you crank up RL far more aggressively.

DeepSeek-R1 demonstrated something striking: with pure RL on verifiable math and code problems, and no reasoning
demonstrations at all, models **spontaneously learn to produce long chains of thought**, to backtrack, and to
self-verify. Response length grows over training as the model discovers that thinking longer pays.

Structurally this is a different bet than RLHF: RLHF shapes *behavior* against human taste; RLVR improves *capability*
against ground truth. It is the axis frontier labs are pushing hardest, and it connects directly to test-time compute
scaling.`),

    code('Bradley-Terry, DPO, and reward hacking', `import numpy as np
rng = np.random.default_rng(0)

# --- reward model training on preferences ---
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))

d = 6
true_r = rng.normal(size=d)                     # the "true" quality direction
def gen_pair():
    a, b = rng.normal(size=d), rng.normal(size=d)
    return (a, b) if a@true_r > b@true_r else (b, a)

w = np.zeros(d)
for step in range(4000):
    pos, neg = gen_pair()
    p = sigmoid(w@pos - w@neg)
    g = (1-p)                                    # d/dw of -log sigmoid(diff)
    w += 0.05 * g * (pos - neg)
print(f"reward model cosine with truth: "
      f"{w@true_r/(np.linalg.norm(w)*np.linalg.norm(true_r)):.4f}")

# --- reward hacking: optimize the PROXY without a KL leash ---
print("\\noptimizing the learned proxy, unconstrained:")
x = rng.normal(size=d)
ref = x.copy()
for beta in [0.0, 0.1, 0.5, 2.0]:
    x = ref.copy()
    for _ in range(300):
        grad = w - beta*(x - ref)                # proxy gradient minus KL pull
        x += 0.05*grad
    print(f"  beta={beta:4.1f}: proxy reward {x@w:8.3f}   "
          f"TRUE reward {x@true_r:8.3f}   drift {np.linalg.norm(x-ref):6.2f}")
print("  ^ with beta=0 the proxy score soars while true quality lags -> reward hacking")

# --- DPO loss ---
def dpo_loss(logp_pos, logp_neg, ref_pos, ref_neg, beta=0.1):
    return -np.log(sigmoid(beta*((logp_pos-ref_pos) - (logp_neg-ref_neg))))

print("\\nDPO loss as the policy learns to prefer the chosen response:")
for shift in [-1.0, 0.0, 1.0, 3.0]:
    print(f"  policy favours chosen by {shift:+.1f} nats -> "
          f"loss {dpo_loss(shift, 0.0, 0.0, 0.0, beta=1.0):.4f}")`),

    quiz('Why does the RLHF objective include a KL penalty to the SFT model?',
      ['The reward model is only accurate near its training distribution; the penalty keeps the policy where the proxy is still valid',
       'To speed up convergence',
       'To prevent the model from forgetting its pretraining data',
       'It is required for the policy gradient to be unbiased'],
      0,
      'The reward model was trained on outputs from roughly the SFT distribution. Far from there, its scores are extrapolations and often nonsense — and an optimizer will find exactly those regions, because that is where the proxy is highest. The KL term bounds the drift, keeping the policy inside the region where the proxy still tracks real quality. It is Goodhart mitigation, expressed as a constraint.'),
  ],
  refs: [
    paper('Training language models to follow instructions with human feedback', 'Ouyang et al.', 2022, 'https://arxiv.org/abs/2203.02155', 'InstructGPT. The full RLHF pipeline.'),
    paper('Direct Preference Optimization', 'Rafailov et al.', 2023, 'https://arxiv.org/abs/2305.18290', 'DPO. The derivation is worth working through by hand.'),
    paper('DeepSeek-R1', 'DeepSeek-AI', 2025, 'https://arxiv.org/abs/2501.12948', 'Reasoning emerging from pure RL on verifiable rewards, documented in detail.'),
    paper('Constitutional AI: Harmlessness from AI Feedback', 'Bai et al.', 2022, 'https://arxiv.org/abs/2212.08073', 'RLAIF — replacing human preference labels with model-generated ones against a written constitution.'),
    paper('Scaling Laws for Reward Model Overoptimization', 'Gao, Schulman & Hilton', 2022, 'https://arxiv.org/abs/2210.10760', 'Quantifies exactly how proxy reward and true reward diverge as you optimize.'),
  ],
},

];

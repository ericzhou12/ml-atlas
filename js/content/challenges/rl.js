/* ============================================================
   Challenges — track 7, RL & alignment

   One entry per lesson id: { title, prompt, starter, solution, hint?, explain? }
   Starters carry their own PASS/FAIL check so the learner gets a verdict
   without leaving the lab.

   EDITOR NOTE: these are JS template literals, so an unescaped backtick
   silently terminates the string. Escape inline code in prose.
   ============================================================ */

export default {

'rl-mdp': {
  title: 'Value iteration on a gridworld',
  prompt: `Implement the Bellman optimality backup and iterate to convergence. Then show how the discount factor and
the slip probability change the optimal policy near the trap.`,
  hint: 'V(s) = max_a sum over outcomes of prob * (reward + gamma * V(s_next)).',
  starter: `import numpy as np

W, H = 4, 4
walls = {(1,1), (1,2)}
goal, trap = (3,3), (3,1)
ACTIONS = [(0,1),(1,0),(0,-1),(-1,0)]
ARROW = "^>v<"

states = [(x,y) for x in range(W) for y in range(H) if (x,y) not in walls]

def step(s, a, slip=0.2):
    if s in (goal, trap): return [(1.0, s, 0.0)]
    out = []
    for ai, p in [(a, 1-slip), ((a+1)%4, slip/2), ((a+3)%4, slip/2)]:
        nx, ny = s[0]+ACTIONS[ai][0], s[1]+ACTIONS[ai][1]
        ns = (nx,ny) if (0<=nx<W and 0<=ny<H and (nx,ny) not in walls) else s
        r = 1.0 if ns == goal else (-1.0 if ns == trap else -0.04)
        out.append((p, ns, r))
    return out

def value_iteration(gamma=0.95, slip=0.2, tol=1e-10):
    V = {s: 0.0 for s in states}
    for it in range(1000):
        delta = 0.0
        for s in states:
            if s in (goal, trap): continue
            # TODO: Bellman optimality backup
            pass
        if delta < tol: return V, it+1
    return V, 1000

V, iters = value_iteration()
print(f"converged in {iters} sweeps")
for gamma in [0.5, 0.9, 0.99]:
    Vg, _ = value_iteration(gamma)
    print(f"  gamma={gamma:.2f}: V(0,0)={Vg[(0,0)]:+.4f}")
assert value_iteration(0.99)[0][(0,0)] > value_iteration(0.5)[0][(0,0)], \\
    "a patient agent should value the start square more highly"
print("\\nPASS")`,
  solution: `import numpy as np

W, H = 4, 4
walls = {(1,1), (1,2)}
goal, trap = (3,3), (3,1)
ACTIONS = [(0,1),(1,0),(0,-1),(-1,0)]
ARROW = "^>v<"
states = [(x,y) for x in range(W) for y in range(H) if (x,y) not in walls]

def step(s, a, slip=0.2):
    if s in (goal, trap): return [(1.0, s, 0.0)]
    out = []
    for ai, p in [(a, 1-slip), ((a+1)%4, slip/2), ((a+3)%4, slip/2)]:
        nx, ny = s[0]+ACTIONS[ai][0], s[1]+ACTIONS[ai][1]
        ns = (nx,ny) if (0<=nx<W and 0<=ny<H and (nx,ny) not in walls) else s
        r = 1.0 if ns == goal else (-1.0 if ns == trap else -0.04)
        out.append((p, ns, r))
    return out

def value_iteration(gamma=0.95, slip=0.2, tol=1e-10):
    V = {s: 0.0 for s in states}
    for it in range(1000):
        delta = 0.0
        for s in states:
            if s in (goal, trap): continue
            best = max(sum(p*(r + gamma*V[ns]) for p,ns,r in step(s,a,slip))
                       for a in range(4))
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < tol: return V, it+1
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
            a = max(range(4), key=lambda a: sum(p*(r+0.95*V[ns])
                    for p,ns,r in step(s,a)))
            row += f" {V[s]:+.2f}{ARROW[a]} "
    print(row)

print()
for gamma in [0.5, 0.9, 0.99]:
    Vg, _ = value_iteration(gamma)
    print(f"  gamma={gamma:.2f}: V(0,0)={Vg[(0,0)]:+.4f}  "
          f"(reward 10 steps away worth {gamma**10:.3f})")

assert value_iteration(0.99)[0][(0,0)] > value_iteration(0.5)[0][(0,0)], \\
    "a patient agent should value the start square more highly"
print("\\nPASS")`,
  explain: `The policy arrows are the interesting output. Near the trap the agent does not take the shortest route — it
detours, because value iteration has propagated the trap's $-1$ backwards through the transition model, so the
squares *next to* the trap are worth less than their distance from the goal alone would suggest. Nobody coded
that; it fell out of the Bellman backup taking a max over actions of "immediate reward plus discounted value of
where you land".

The discount sweep is the second half. $\gamma$ is not a tuning knob for convergence speed — it is a statement
about how far ahead the agent cares. At $\gamma = 0.5$ a reward ten steps away is worth $0.5^{10} \approx 0.001$,
which is to say nothing at all, and the agent behaves as if the goal did not exist. At $\gamma = 0.99$ the same
reward is worth 0.90 and the whole grid points toward it. Changing $\gamma$ changes what problem you are solving,
not how well you solve it.`,
},
'rl-model-free': {
  title: 'Q-learning vs SARSA on the cliff',
  prompt: `Implement both TD updates on cliff-walking. Q-learning should find the optimal cliff-edge path but score
worse online; SARSA should learn a safer detour. Explain why before you run it.`,
  hint: 'Q-learning uses max Q(s2, .) in the target; SARSA uses Q(s2, a2) for the action actually taken.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

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
            # TODO: target differs between q-learning and sarsa
            target = 0.0
            Q[s][a] += alpha*(target - Q[s][a])
            s, a, total = ns, na, total+r
            if done: break
        returns.append(total)
    return Q, np.mean(returns[-100:])

heights = {}
for algo in ["q", "sarsa"]:
    Q, avg = run(algo)
    s, path = START, []
    for _ in range(30):
        path.append(s)
        s, _, done = step(s, Q[s].argmax())
        if done or s == START: break
    heights[algo] = max(p[1] for p in path)
    print(f"{algo.upper():6s} avg return {avg:7.1f}   max path height {heights[algo]}")

assert heights["sarsa"] > heights["q"], "SARSA should learn a path further from the cliff edge"
print("\\nPASS")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

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
            bootstrap = 0 if done else (Q[ns].max() if algo=="q" else Q[ns][na])
            Q[s][a] += alpha*(r + gamma*bootstrap - Q[s][a])
            s, a, total = ns, na, total+r
            if done: break
        returns.append(total)
    return Q, np.mean(returns[-100:])

heights = {}
for algo in ["q", "sarsa"]:
    Q, avg = run(algo)
    s, path = START, []
    for _ in range(30):
        path.append(s)
        s, _, done = step(s, Q[s].argmax())
        if done or s == START: break
    heights[algo] = max(p[1] for p in path)
    print(f"{algo.upper():6s} avg return {avg:7.1f}   max path height {heights[algo]}")
print("\\nQ-learning is off-policy: it learns the optimal path but keeps falling off")
print("the cliff while exploring. SARSA accounts for its own exploration and detours.")
assert heights["sarsa"] > heights["q"], "SARSA should learn a path further from the cliff edge"
print("PASS")`,
  explain: `This is the classic cliff-walking result, and the point is that neither algorithm is wrong.

Q-learning's update uses $\max_a Q(s', a)$ — the value of the *best* action available next — regardless of what
the exploring policy will actually do. So it learns the value of the optimal path, which runs right along the
cliff edge, and its learned policy walks that edge. But while it is still exploring, an $\epsilon$-greedy slip
near the edge means falling off, so its **online** return is worse.

SARSA's update uses $Q(s', a')$ where $a'$ is the action it *actually took*, exploration included. That makes the
squares next to the cliff genuinely dangerous in its estimates, because sometimes it does fall in — so it learns
a path one row further from the edge. It is learning the value of the policy it is really following, rather than
the value of a policy it never quite executes.

The names for this are on-policy (SARSA) and off-policy (Q-learning), and the choice matters most where mistakes
during learning are expensive — which is why the distinction gets a lot of attention in robotics and very little
in simulation.`,
},
'rl-policy-gradient': {
  title: 'REINFORCE, a baseline, and the PPO clip',
  prompt: `Implement the policy gradient for a Gaussian policy and show a baseline reduces variance without changing
the fixed point. Then implement the clipped surrogate and verify the gradient dies outside the trust region.`,
  hint: 'For a Gaussian policy, d log pi / d theta = (a - theta) / sigma^2.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

reward = lambda a: np.exp(-((a-1.7)**2)/0.6) + 0.55*np.exp(-((a+1.6)**2)/0.5)

def train(use_baseline, batch=8, lr=0.05, sigma=0.8, steps=400):
    theta, baseline = -1.5, 0.0
    traj = []
    for t in range(steps):
        a = theta + rng.normal(0, sigma, batch)
        r = reward(a)
        adv = r - (baseline if use_baseline else 0.0)
        # TODO: grad = mean(adv * dlogpi), then theta += lr*grad
        grad = 0.0
        theta += lr*grad
        baseline = 0.9*baseline + 0.1*r.mean()
        traj.append(theta)
    return np.array(traj)

for label, bl in [("no baseline", False), ("baseline   ", True)]:
    for B in [1, 8, 64]:
        tr = train(bl, batch=B)
        print(f"{label} batch={B:3d}: theta {tr[-1]:+.3f}  "
              f"jitter {np.abs(np.diff(tr[-100:])).mean():.5f}")

def ppo_obj(ratio, adv, eps=0.2):
    # TODO
    return 0.0

print("\\nPPO clipped objective (eps=0.2):")
for r in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    print(f"  ratio {r:.1f}: A=+1 -> {ppo_obj(r,1.0):+.2f}   A=-1 -> {ppo_obj(r,-1.0):+.2f}")
assert abs(ppo_obj(2.0, 1.0) - ppo_obj(1.5, 1.0)) < 1e-9, \\
    "past the clip the objective must be flat, so the gradient is zero"
assert abs(ppo_obj(1.0, 1.0) - ppo_obj(1.1, 1.0)) > 1e-9, \\
    "inside the trust region the objective must still respond"
print("\\nPASS")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

reward = lambda a: np.exp(-((a-1.7)**2)/0.6) + 0.55*np.exp(-((a+1.6)**2)/0.5)

def train(use_baseline, batch=8, lr=0.05, sigma=0.8, steps=400):
    theta, baseline = -1.5, 0.0
    traj = []
    for t in range(steps):
        a = theta + rng.normal(0, sigma, batch)
        r = reward(a)
        adv = r - (baseline if use_baseline else 0.0)
        grad = (adv * (a - theta) / sigma**2).mean()
        theta += lr*grad
        baseline = 0.9*baseline + 0.1*r.mean()
        traj.append(theta)
    return np.array(traj)

for label, bl in [("no baseline", False), ("baseline   ", True)]:
    for B in [1, 8, 64]:
        tr = train(bl, batch=B)
        print(f"{label} batch={B:3d}: theta {tr[-1]:+.3f}  "
              f"jitter {np.abs(np.diff(tr[-100:])).mean():.5f}")

def ppo_obj(ratio, adv, eps=0.2):
    return min(ratio*adv, np.clip(ratio, 1-eps, 1+eps)*adv)

print("\\nPPO clipped objective (eps=0.2):")
for r in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    note = ""
    if r > 1.2: note = "  clipped for A>0"
    elif r < 0.8: note = "  clipped for A<0"
    print(f"  ratio {r:.1f}: A=+1 -> {ppo_obj(r,1.0):+.2f}   A=-1 -> {ppo_obj(r,-1.0):+.2f}{note}")

assert abs(ppo_obj(2.0, 1.0) - ppo_obj(1.5, 1.0)) < 1e-9, \\
    "past the clip the objective must be flat, so the gradient is zero"
assert abs(ppo_obj(0.5, -1.0) - ppo_obj(0.4, -1.0)) < 1e-9, \\
    "for a negative advantage the clip binds below 0.8, so both are flat"
assert abs(ppo_obj(1.0, 1.0) - ppo_obj(1.1, 1.0)) > 1e-9, \\
    "inside the trust region the objective must still respond"
print("\\nPASS")`,
  explain: `Part 1 is the reason baselines are standard. Subtracting a baseline leaves the *expected* gradient unchanged —
it multiplies a term whose expectation is zero — while cutting the variance substantially. That is a rare kind of
free lunch: the same estimator, the same fixed point, less noise. Without it, REINFORCE gets a large positive
push for every action in a good episode, including the bad actions that happened to be in it.

Part 2 is what the PPO clip actually does, and the table repays reading carefully. Look at the ratio-2.0 row: the
advantage is positive, the new policy already makes this action twice as likely as the old one did, and the
objective stops rewarding further increases — it is flat, so the gradient is **zero**. The same happens on the
other side for negative advantages at ratio 0.5.

That is the whole safety mechanism. Policy gradient estimates are computed under the old policy, so they stop
being valid once the new policy has moved far from it. The clip does not penalise moving too far; it simply stops
paying you for it, which removes the incentive without adding a term that could fight the objective.`,
},
'rl-rlhf': {
  title: 'Train a reward model, then optimize hard enough to break it',
  prompt: `1. **Fit the reward model.** Implement the Bradley–Terry update: given a preferred and a rejected sample,
   push the preferred one's score up relative to the other. Check that the learned direction matches the truth.
2. **Then over-optimize it.** Optimize against the learned reward with a KL penalty of varying strength, and
   watch two numbers: the **proxy** score the reward model reports, and the **true** quality it was standing in
   for. Only the first is available to a real training run; the second is here so you can see what happens.
3. **Predict the shape of both columns** before running. They do not move together.`,
  hint: 'The Bradley–Terry gradient for a linear reward $r(x) = \\mathbf{w}\\cdot\\mathbf{x}$ is $(1-\\sigma(r_{\\text{pos}} - r_{\\text{neg}}))(\\mathbf{x}_{\\text{pos}} - \\mathbf{x}_{\\text{neg}})$ — the update is largest when the model currently has the pair backwards.',
  starter: `import numpy as np
rng = np.random.default_rng(0)
sigmoid = lambda z: 1/(1+np.exp(-np.clip(z, -30, 30)))

d = 6
true_dir = rng.normal(size=d); true_dir /= np.linalg.norm(true_dir)

def true_quality(x):
    """What we actually care about. Rises along true_dir, but only up to a point --
       past that, the output has become extreme and quality falls away again."""
    return x @ true_dir - 0.12*(x @ x)

def gen_pair():
    """Two ordinary outputs, labelled by which a human would prefer."""
    a, b = rng.normal(size=d), rng.normal(size=d)
    return (a, b) if true_quality(a) > true_quality(b) else (b, a)

# ---------- 1. fit the reward model from preferences only ----------
w = np.zeros(d)
for step in range(6000):
    pos, neg = gen_pair()
    # TODO: Bradley-Terry step -- move w so that pos scores above neg
    pass

cos = float(w @ true_dir / (np.linalg.norm(w) + 1e-12))
print(f"learned reward direction vs the truth: cosine {cos:.4f}")
assert cos > 0.9, "the reward model should recover the direction humans were judging by"
print("PASS -- the reward model looks excellent\\n")

# ---------- 2. now optimize against it ----------
ref = -1.5*true_dir + 0.3*rng.normal(size=d)      # the starting policy: mediocre
print(f"starting true quality: {true_quality(ref):.2f}")
print(f"the best true quality any output could reach: {true_quality(true_dir/0.24):.2f}\\n")

print(f"{'KL beta':>8} {'proxy score':>13} {'TRUE quality':>14} {'drift from ref':>16}")
for beta in [0.0, 0.1, 0.3, 0.6, 1.0, 2.0, 5.0]:
    x = ref.copy()
    for _ in range(400):
        x = x + 0.02 * (w - beta*(x - ref))        # climb the proxy, pulled back toward ref
    print(f"{beta:8.1f} {x @ w:13.2f} {true_quality(x):14.2f} {np.linalg.norm(x-ref):16.2f}")

def optimize(beta):
    x = ref.copy()
    for _ in range(400): x = x + 0.02*(w - beta*(x - ref))
    return x @ w, true_quality(x)

assert optimize(0.0)[0] > optimize(0.6)[0], "less constraint should always score higher on the proxy"
assert optimize(0.0)[1] < optimize(0.6)[1], "yet be much worse on what actually matters"
print("\\nPASS")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)
sigmoid = lambda z: 1/(1+np.exp(-np.clip(z, -30, 30)))

d = 6
true_dir = rng.normal(size=d); true_dir /= np.linalg.norm(true_dir)

def true_quality(x):
    return x @ true_dir - 0.12*(x @ x)

def gen_pair():
    a, b = rng.normal(size=d), rng.normal(size=d)
    return (a, b) if true_quality(a) > true_quality(b) else (b, a)

w = np.zeros(d)
for step in range(6000):
    pos, neg = gen_pair()
    p = sigmoid(w @ pos - w @ neg)
    w = w + 0.05 * (1 - p) * (pos - neg)

cos = float(w @ true_dir / (np.linalg.norm(w) + 1e-12))
print(f"learned reward direction vs the truth: cosine {cos:.4f}")
assert cos > 0.9
print("PASS -- the reward model looks excellent\\n")

ref = -1.5*true_dir + 0.3*rng.normal(size=d)
print(f"starting true quality: {true_quality(ref):.2f}")
print(f"the best true quality any output could reach: {true_quality(true_dir/0.24):.2f}\\n")

print(f"{'KL beta':>8} {'proxy score':>13} {'TRUE quality':>14} {'drift from ref':>16}")
for beta in [0.0, 0.1, 0.3, 0.6, 1.0, 2.0, 5.0]:
    x = ref.copy()
    for _ in range(400):
        x = x + 0.02 * (w - beta*(x - ref))
    print(f"{beta:8.1f} {x @ w:13.2f} {true_quality(x):14.2f} {np.linalg.norm(x-ref):16.2f}")

def optimize(beta):
    x = ref.copy()
    for _ in range(400): x = x + 0.02*(w - beta*(x - ref))
    return x @ w, true_quality(x)

assert optimize(0.0)[0] > optimize(0.6)[0]
assert optimize(0.0)[1] < optimize(0.6)[1]
print("\\nPASS")`,
  explain: `Start with what a real training run would see. The reward model is *excellent* — cosine 0.99 with the
direction humans were actually judging by — and the proxy column rises monotonically as you relax the KL penalty.
Every number available to you says that $\\beta = 0$ is the best setting.

Now read the column you would not have. True quality peaks around $\\beta = 0.6$, close to the best any output
could achieve, and at $\\beta = 0$ it has fallen to about **−58**: far worse than the mediocre policy you started
from. The proxy score at that setting is over 90. The optimizer did exactly what it was asked and destroyed the
thing you cared about.

The mechanism is worth being precise about, because it is not that the reward model is bad. It was fit on
ordinary outputs, where quality really does rise along the direction it learned, and it learned that faithfully. What it never
saw is anything far from ordinary — so it has no idea about the $-0.12\\lVert x\\rVert^2$ term, and confidently
extrapolates a straight line into territory where the truth curves away. Look at the drift column: the failures
are exactly the runs that travelled furthest from the reference.

That is what the KL penalty in RLHF is for. It is not a regularizer in the usual sense of preventing overfitting;
it is a statement that **the reward model is only trustworthy near the distribution it was trained on**, and a
mechanism for staying there. And it explains the shape of the practical problem: too much penalty and the model
never improves, too little and it optimizes into nonsense, with no signal available at training time to tell you
which side you are on. Watching the KL divergence itself, rather than the reward, is the standard defence.`,
},

};

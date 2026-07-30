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
    print(f"  gamma={gamma:.2f}: V(0,0)={Vg[(0,0)]:+.4f}")`,
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
          f"(reward 10 steps away worth {gamma**10:.3f})")`,
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

for algo in ["q", "sarsa"]:
    Q, avg = run(algo)
    s, path = START, []
    for _ in range(30):
        path.append(s)
        s, _, done = step(s, Q[s].argmax())
        if done or s == START: break
    print(f"{algo.upper():6s} avg return {avg:7.1f}   max path height {max(p[1] for p in path)}")`,
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

for algo in ["q", "sarsa"]:
    Q, avg = run(algo)
    s, path = START, []
    for _ in range(30):
        path.append(s)
        s, _, done = step(s, Q[s].argmax())
        if done or s == START: break
    print(f"{algo.upper():6s} avg return {avg:7.1f}   max path height {max(p[1] for p in path)}")
print("\\nQ-learning is off-policy: it learns the optimal path but keeps falling off")
print("the cliff while exploring. SARSA accounts for its own exploration and detours.")`,
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
    print(f"  ratio {r:.1f}: A=+1 -> {ppo_obj(r,1.0):+.2f}   A=-1 -> {ppo_obj(r,-1.0):+.2f}")`,
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
    print(f"  ratio {r:.1f}: A=+1 -> {ppo_obj(r,1.0):+.2f}   A=-1 -> {ppo_obj(r,-1.0):+.2f}{note}")`,
},

'rl-rlhf': {
  title: 'Train a reward model, then hack it',
  prompt: `Fit a Bradley-Terry reward model on preference pairs, then optimize against it with and without a KL
penalty. Show that unconstrained optimization drives the proxy up while true quality lags.`,
  hint: 'The Bradley-Terry loss is -log sigmoid(r(chosen) - r(rejected)).',
  starter: `import numpy as np
rng = np.random.default_rng(0)
sigmoid = lambda z: 1/(1+np.exp(-np.clip(z,-30,30)))

d = 6
true_r = rng.normal(size=d)                    # the true quality direction

def gen_pair():
    a, b = rng.normal(size=d), rng.normal(size=d)
    return (a, b) if a@true_r > b@true_r else (b, a)

w = np.zeros(d)
for step in range(4000):
    pos, neg = gen_pair()
    # TODO: gradient of -log sigmoid(w@pos - w@neg), step with lr=0.05
    pass

cos = w@true_r / (np.linalg.norm(w)*np.linalg.norm(true_r) + 1e-12)
print(f"reward model cosine with truth: {cos:.4f}")

print("\\noptimizing the learned proxy:")
ref = rng.normal(size=d)
for beta in [0.0, 0.1, 0.5, 2.0]:
    x = ref.copy()
    for _ in range(300):
        # TODO: ascend w, minus beta*(x - ref) as the KL pull
        pass
    print(f"  beta={beta:4.1f}: proxy {x@w:8.3f}   TRUE {x@true_r:8.3f}   "
          f"drift {np.linalg.norm(x-ref):6.2f}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)
sigmoid = lambda z: 1/(1+np.exp(-np.clip(z,-30,30)))

d = 6
true_r = rng.normal(size=d)

def gen_pair():
    a, b = rng.normal(size=d), rng.normal(size=d)
    return (a, b) if a@true_r > b@true_r else (b, a)

w = np.zeros(d)
for step in range(4000):
    pos, neg = gen_pair()
    p = sigmoid(w@pos - w@neg)
    w += 0.05 * (1-p) * (pos - neg)            # d/dw of -log sigmoid(diff)

cos = w@true_r / (np.linalg.norm(w)*np.linalg.norm(true_r) + 1e-12)
print(f"reward model cosine with truth: {cos:.4f}")

print("\\noptimizing the learned proxy:")
ref = rng.normal(size=d)
for beta in [0.0, 0.1, 0.5, 2.0]:
    x = ref.copy()
    for _ in range(300):
        x += 0.05 * (w - beta*(x - ref))
    print(f"  beta={beta:4.1f}: proxy {x@w:8.3f}   TRUE {x@true_r:8.3f}   "
          f"drift {np.linalg.norm(x-ref):6.2f}")
print("\\nAt beta=0 the proxy score soars while true quality lags -- reward hacking.")`,
},

};

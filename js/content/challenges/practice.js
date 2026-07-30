/* ============================================================
   Challenges — track 11, practice

   EDITOR NOTE: these are JS template literals, so an unescaped backtick
   silently terminates the string. Escape inline code in prose.
   ============================================================ */

export default {

'pr-debugging': {
  title: 'Build the five-check diagnostic harness',
  prompt: `Implement the checks that catch most real bugs, in order: initial loss equals log K, overfit a single batch,
sweep the learning rate, inspect per-layer gradient norms, and gradient-check against finite differences.`,
  hint: 'If check 2 fails you have a bug, not a tuning problem. Eight examples are trivially memorizable.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def make_model(n_in=20, n_hidden=32, n_out=3, seed=0):
    r = np.random.default_rng(seed)
    return [(r.normal(0, np.sqrt(2/a), (a,b)), np.zeros(b))
            for a,b in [(n_in,n_hidden), (n_hidden,n_out)]]

def forward(P, X):
    h = np.maximum(0, X @ P[0][0] + P[0][1])
    return h @ P[1][0] + P[1][1], h

def loss_and_grads(P, X, y):
    logits, h = forward(P, X)
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    d = p.copy(); d[np.arange(len(y)), y] -= 1; d /= len(y)
    dh = (d @ P[1][0].T) * (h > 0)
    return loss, [(X.T @ dh, dh.sum(0)), (h.T @ d, d.sum(0))]

X = rng.normal(size=(512, 20)); y = rng.integers(0, 3, 512)
X[np.arange(512), y % 20] += 2.0

# CHECK 1: initial loss should be log(K)
P = make_model()
l0, _ = loss_and_grads(P, X, y)
print(f"CHECK 1  initial loss {l0:.4f}  expected {np.log(3):.4f}  "
      f"{'PASS' if abs(l0-np.log(3)) < 0.15 else 'FAIL'}")

# CHECK 2: TODO -- overfit X[:8], y[:8] and assert the loss goes below 0.01

# CHECK 3: TODO -- sweep lr over 1e-4 .. 10 and report where it diverges

# CHECK 4: TODO -- print per-layer gradient norms

# CHECK 5: TODO -- gradient-check one weight against a central difference`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def make_model(n_in=20, n_hidden=32, n_out=3, seed=0):
    r = np.random.default_rng(seed)
    return [(r.normal(0, np.sqrt(2/a), (a,b)), np.zeros(b))
            for a,b in [(n_in,n_hidden), (n_hidden,n_out)]]

def forward(P, X):
    h = np.maximum(0, X @ P[0][0] + P[0][1])
    return h @ P[1][0] + P[1][1], h

def loss_and_grads(P, X, y):
    logits, h = forward(P, X)
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    d = p.copy(); d[np.arange(len(y)), y] -= 1; d /= len(y)
    dh = (d @ P[1][0].T) * (h > 0)
    return loss, [(X.T @ dh, dh.sum(0)), (h.T @ d, d.sum(0))]

X = rng.normal(size=(512, 20)); y = rng.integers(0, 3, 512)
X[np.arange(512), y % 20] += 2.0

P = make_model()
l0, _ = loss_and_grads(P, X, y)
print(f"CHECK 1  initial loss {l0:.4f}  expected {np.log(3):.4f}  "
      f"{'PASS' if abs(l0-np.log(3)) < 0.15 else 'FAIL'}")

P = make_model()
for _ in range(600):
    l, g = loss_and_grads(P, X[:8], y[:8])
    P = [(W-0.1*gW, b-0.1*gb) for (W,b),(gW,gb) in zip(P,g)]
print(f"CHECK 2  loss on 8 examples: {l:.6f}  "
      f"{'PASS' if l < 0.01 else 'FAIL - bug, not tuning'}")

print("\\nCHECK 3  learning rate sweep:")
for lr in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    P = make_model()
    for _ in range(100):
        l, g = loss_and_grads(P, X, y)
        P = [(W-lr*gW, b-lr*gb) for (W,b),(gW,gb) in zip(P,g)]
        if not np.isfinite(l): break
    print(f"  lr={lr:<7} {'diverged' if not np.isfinite(l) else f'loss {l:.4f}'}")

P = make_model()
_, g = loss_and_grads(P, X, y)
print("\\nCHECK 4  gradient norms per layer:")
for i, (gW, gb) in enumerate(g):
    print(f"  layer {i}: |dW| {np.linalg.norm(gW):.5f}  |db| {np.linalg.norm(gb):.5f}")

P = make_model()
_, g = loss_and_grads(P, X[:16], y[:16])
eps, i, j = 1e-5, 3, 7
P[0][0][i,j] += eps; lp, _ = loss_and_grads(P, X[:16], y[:16])
P[0][0][i,j] -= 2*eps; lm, _ = loss_and_grads(P, X[:16], y[:16])
P[0][0][i,j] += eps
num = (lp - lm) / (2*eps)
rel = abs(num - g[0][0][i,j]) / (abs(num) + 1e-12)
print(f"\\nCHECK 5  analytic {g[0][0][i,j]:.8f}  numeric {num:.8f}  "
      f"rel err {rel:.2e}  {'PASS' if rel < 1e-4 else 'FAIL'}")`,
  explain: 'Checks 1, 2 and 5 between them catch most real defects in hand-written models: a misconfigured head, a broken gradient path, and a wrong derivative.',
},

'pr-experiments': {
  title: 'Random vs grid search, and how many seeds you need',
  prompt: `Show that random search beats grid search at equal budget when only a couple of hyperparameters matter. Then
determine how many seeds are needed to distinguish a true 1% improvement from noise.`,
  hint: 'Grid search wastes its budget on dimensions that do not matter; random search gets fine resolution on the ones that do.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def objective(h):
    """Only h[0] and h[1] matter. h[2:] are decoys."""
    return (np.exp(-((h[0]-0.3)**2)/0.02)
            + 0.7*np.exp(-((h[1]-0.7)**2)/0.05)
            + 0.02*rng.normal())

BUDGET = 81

# TODO: grid search -- 3 values per dimension over 4 dimensions
best_grid = -np.inf

# TODO: random search -- BUDGET random 5-vectors
best_random = -np.inf

print(f"budget = {BUDGET} evaluations")
print(f"  grid   best {best_grid:.4f}  (3 distinct values per dimension)")
print(f"  random best {best_random:.4f}  ({BUDGET} distinct values per dimension)")

def train_run(seed, effect=0.0):
    r = np.random.default_rng(seed)
    return 0.75 + effect + r.normal(0, 0.025)

print("\\ncomparing two methods where the true difference is 1%:")
for n_seeds in [1, 3, 10, 30]:
    a = [train_run(s, 0.00) for s in range(n_seeds)]
    b = [train_run(s+1000, 0.01) for s in range(n_seeds)]
    # TODO: compute the difference and a standard error, and judge significance
    print(f"  {n_seeds:2d} seed(s):")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def objective(h):
    return (np.exp(-((h[0]-0.3)**2)/0.02)
            + 0.7*np.exp(-((h[1]-0.7)**2)/0.05)
            + 0.02*rng.normal())

BUDGET = 81

best_grid = -np.inf
pts = np.linspace(0.1, 0.9, 3)
for a in pts:
    for b in pts:
        for c in pts:
            for d in pts:
                best_grid = max(best_grid, objective([a,b,c,d,0.5]))

best_random = max(objective(rng.random(5)) for _ in range(BUDGET))

print(f"budget = {BUDGET} evaluations")
print(f"  grid   best {best_grid:.4f}  (3 distinct values per dimension)")
print(f"  random best {best_random:.4f}  ({BUDGET} distinct values per dimension)")

def train_run(seed, effect=0.0):
    r = np.random.default_rng(seed)
    return 0.75 + effect + r.normal(0, 0.025)

print("\\ncomparing two methods where the true difference is 1%:")
for n_seeds in [1, 3, 10, 30]:
    a = [train_run(s, 0.00) for s in range(n_seeds)]
    b = [train_run(s+1000, 0.01) for s in range(n_seeds)]
    diff = np.mean(b) - np.mean(a)
    if n_seeds == 1:
        print(f"  {n_seeds:2d} seed(s): diff {diff:+.4f}  cannot assess")
        continue
    se = np.sqrt(np.var(a, ddof=1)/n_seeds + np.var(b, ddof=1)/n_seeds)
    verdict = "significant" if abs(diff) > 2*se else "NOT significant"
    print(f"  {n_seeds:2d} seed(s): diff {diff:+.4f}  se {se:.4f}  {verdict}")`,
  explain: 'With one seed you cannot tell a 1% improvement from noise at all. This is why single-number comparisons in papers are weak evidence, and why reporting seed variance matters more than another decimal place.',
},

'pr-reading': {
  title: 'Estimate whether a reported improvement is real',
  prompt: `Given a benchmark result, work out whether the claimed gain survives the trial count. Then simulate the
tuning-effort asymmetry: what happens when the baseline gets a smaller hyperparameter search than the new method.`,
  hint: 'Selecting the max over k trials is biased upward. More search budget alone produces an apparent improvement.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def wilson(s, n, z=1.96):
    p = s/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0,c-h), min(1,c+h)

print("a paper reports 2 points on a 1000-example benchmark:")
lo1, hi1 = wilson(870, 1000)
lo2, hi2 = wilson(890, 1000)
print(f"  87.0%: [{lo1:.3f}, {hi1:.3f}]")
print(f"  89.0%: [{lo2:.3f}, {hi2:.3f}]")
print(f"  {'overlapping -> weak evidence' if hi1 > lo2 else 'separated'}")

print("\\ntuning-effort asymmetry -- identical methods, different search budgets:")
TRUE = 0.80
for baseline_tries, method_tries in [(1, 1), (1, 20), (5, 50), (20, 20)]:
    # TODO: each try is TRUE + noise; take the max of each, average over 2000 runs
    print(f"  baseline {baseline_tries:3d} tries vs method {method_tries:3d} tries:")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def wilson(s, n, z=1.96):
    p = s/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0,c-h), min(1,c+h)

print("a paper reports 2 points on a 1000-example benchmark:")
lo1, hi1 = wilson(870, 1000); lo2, hi2 = wilson(890, 1000)
print(f"  87.0%: [{lo1:.3f}, {hi1:.3f}]")
print(f"  89.0%: [{lo2:.3f}, {hi2:.3f}]")
print(f"  {'overlapping -> weak evidence' if hi1 > lo2 else 'separated -> plausible'}")

print("\\ntuning-effort asymmetry -- IDENTICAL methods, different search budgets:")
TRUE, NOISE = 0.80, 0.02
for bt, mt in [(1, 1), (1, 20), (5, 50), (20, 20)]:
    gaps = []
    for _ in range(2000):
        base = max(TRUE + rng.normal(0, NOISE) for _ in range(bt))
        meth = max(TRUE + rng.normal(0, NOISE) for _ in range(mt))
        gaps.append(meth - base)
    print(f"  baseline {bt:3d} tries vs method {mt:3d} tries: "
          f"apparent gain {np.mean(gaps)*100:+.2f} points")
print("\\nThe methods are the same. The entire 'improvement' is search budget.")`,
  explain: 'This is the single most common way ML results mislead — and it is invisible unless the paper states its search budget for both arms. Ask for it.',
},

};

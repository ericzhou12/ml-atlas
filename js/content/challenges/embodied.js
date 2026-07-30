/* ============================================================
   Challenges — track 10, physical & embodied intelligence

   EDITOR NOTE: these are JS template literals, so an unescaped backtick
   silently terminates the string. Escape inline code in prose.
   ============================================================ */

export default {

'emb-why-robots': {
  title: 'Make behavior cloning drift, then stop it',
  prompt: `Clone an expert controller, roll it out, and watch the deviation compound. Then add noise during data
collection (a cheap stand-in for recovery demonstrations) and show the drift stops.`,
  hint: 'The policy is only accurate on states the expert visited. Widening that distribution is the fix.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

reference = lambda t: np.sin(t * 1.1) * 0.9 + 0.15 * t
expert_action = lambda t, y: np.clip((reference(t + 0.25) - y) * 2.2, -1.6, 1.6)

def collect(n_demos, noise=0.0):
    X, A = [], []
    for _ in range(n_demos):
        y = reference(0) + rng.normal(0, noise)
        for t in np.arange(0, 9, 0.12):
            a = expert_action(t, y)
            X.append([t, y]); A.append(a)
            # TODO: step the state forward, adding noise*rng.normal() to the action
    return np.array(X), np.array(A)

feats = lambda X: np.c_[X[:,0], X[:,1], X[:,1]**2, np.ones(len(X))]
fit = lambda X, A: np.linalg.lstsq(feats(X), A, rcond=None)[0]

def rollout(w, steps=75):
    y, dev = reference(0), []
    for t in np.arange(0, 9, 0.12)[:steps]:
        a = feats(np.array([[t, y]]))[0] @ w
        y += a * 0.12
        dev.append(abs(y - reference(t)))
    return np.array(dev)

print(f"{'demos':>7} {'collection noise':>18} {'final dev':>11} {'mean dev':>10}")
for n, noise in [(2, 0.0), (2, 0.15), (10, 0.0), (10, 0.15), (50, 0.15)]:
    d = rollout(fit(*collect(n, noise)))
    print(f"{n:7d} {noise:18.2f} {d[-1]:11.4f} {d.mean():10.4f}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

reference = lambda t: np.sin(t * 1.1) * 0.9 + 0.15 * t
expert_action = lambda t, y: np.clip((reference(t + 0.25) - y) * 2.2, -1.6, 1.6)

def collect(n_demos, noise=0.0):
    X, A = [], []
    for _ in range(n_demos):
        y = reference(0) + rng.normal(0, noise)
        for t in np.arange(0, 9, 0.12):
            a = expert_action(t, y)
            X.append([t, y]); A.append(a)
            y += (a + rng.normal(0, noise)) * 0.12
    return np.array(X), np.array(A)

feats = lambda X: np.c_[X[:,0], X[:,1], X[:,1]**2, np.ones(len(X))]
fit = lambda X, A: np.linalg.lstsq(feats(X), A, rcond=None)[0]

def rollout(w, steps=75):
    y, dev = reference(0), []
    for t in np.arange(0, 9, 0.12)[:steps]:
        a = feats(np.array([[t, y]]))[0] @ w
        y += a * 0.12
        dev.append(abs(y - reference(t)))
    return np.array(dev)

print(f"{'demos':>7} {'collection noise':>18} {'final dev':>11} {'mean dev':>10}")
for n, noise in [(2, 0.0), (2, 0.15), (10, 0.0), (10, 0.15), (50, 0.15)]:
    d = rollout(fit(*collect(n, noise)))
    print(f"{n:7d} {noise:18.2f} {d[-1]:11.4f} {d.mean():10.4f}")

w = fit(*collect(3, 0.0))
d = rollout(w, steps=75)
print("\\ndeviation vs horizon (no recovery data):")
for h in [10, 20, 40, 75]:
    print(f"  T={h:3d}: {d[h-1]:.4f}   quadratic scaling would predict "
          f"{d[9]*(h/10)**2:.4f}")`,
  explain: 'Noise during collection widens the covered state distribution, so the policy has seen states near where it drifts. That is why teleoperators are told to make mistakes and correct them.',
},

'emb-policies': {
  title: 'Show why MSE regression fails on multimodal demonstrations',
  prompt: `Two demonstrations solve the task equally well but differently. Show that the MSE-optimal prediction is an
action nobody demonstrated, and that a mixture density head recovers both modes.`,
  hint: 'Fit a 2-component GMM with EM to the demonstrated actions and compare its samples to the mean.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def demo_actions(n, p_up=0.5):
    return np.where(rng.random(n) < p_up, 1.0, -1.0) + rng.normal(0, 0.12, n)

A = demo_actions(4000)
print("demonstrations are bimodal at +1 and -1")
print(f"  mean {A.mean():+.4f}   <- MSE regression predicts THIS")
print(f"  fraction of demos near the mean (|a|<0.4): {(np.abs(A)<0.4).mean():.4f}")
print("  i.e. the MSE-optimal action is one nobody ever demonstrated.\\n")

def fit_gmm(A, iters=60):
    mu = np.array([-0.5, 0.5]); sd = np.array([0.5, 0.5]); pi = np.array([0.5, 0.5])
    for _ in range(iters):
        # TODO: E-step (responsibilities), then M-step (pi, mu, sd)
        pass
    return pi, mu, sd

pi, mu, sd = fit_gmm(A)
print("mixture density head:")
for k in range(2):
    print(f"  component {k}: weight {pi[k]:.3f}  mean {mu[k]:+.3f}  sd {sd[k]:.3f}")

print("\\naction chunking shortens the effective horizon:")
for k in [1, 8, 20, 50]:
    print(f"  chunk k={k:3d}: {400//k:4d} decisions over 400 steps   "
          f"(T/k)^2 = {(400/k)**2:9.0f}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def demo_actions(n, p_up=0.5):
    return np.where(rng.random(n) < p_up, 1.0, -1.0) + rng.normal(0, 0.12, n)

A = demo_actions(4000)
print("demonstrations are bimodal at +1 and -1")
print(f"  mean {A.mean():+.4f}   <- MSE regression predicts THIS")
print(f"  fraction near the mean (|a|<0.4): {(np.abs(A)<0.4).mean():.4f}\\n")

def fit_gmm(A, iters=60):
    mu = np.array([-0.5, 0.5]); sd = np.array([0.5, 0.5]); pi = np.array([0.5, 0.5])
    for _ in range(iters):
        r = pi * np.exp(-((A[:,None]-mu)**2)/(2*sd**2)) / sd
        r /= r.sum(1, keepdims=True)
        Nk = r.sum(0)
        pi = Nk / len(A)
        mu = (r * A[:,None]).sum(0) / Nk
        sd = np.sqrt((r * (A[:,None]-mu)**2).sum(0) / Nk) + 1e-6
    return pi, mu, sd

pi, mu, sd = fit_gmm(A)
print("mixture density head recovers both modes:")
for k in range(2):
    print(f"  component {k}: weight {pi[k]:.3f}  mean {mu[k]:+.3f}  sd {sd[k]:.3f}")

samples = np.where(rng.random(4000) < pi[0],
                   rng.normal(mu[0], sd[0], 4000), rng.normal(mu[1], sd[1], 4000))
print(f"  sampled actions near the mean: {(np.abs(samples)<0.4).mean():.4f}  <- correctly rare")

print("\\naction chunking shortens the effective horizon:")
for k in [1, 8, 20, 50]:
    print(f"  chunk k={k:3d}: {400//k:4d} decisions over 400 steps   "
          f"(T/k)^2 = {(400/k)**2:9.0f}")`,
},

'emb-vla': {
  title: 'Action tokenization, and why it caps control frequency',
  prompt: `Implement discretized action tokens and measure quantization error at different bin counts. Then compute
the control frequency for token decoding versus a flow-matching action expert.`,
  hint: 'Token decoding costs one forward pass per action dimension; a flow expert emits a whole chunk in a few passes.',
  starter: `import numpy as np

def tokenize(action, n_bins=256, lo=-1.0, hi=1.0):
    # TODO: clip, map to [0, n_bins-1], round to int
    return np.zeros(len(action), dtype=int)

def detokenize(tokens, n_bins=256, lo=-1.0, hi=1.0):
    # TODO: inverse
    return np.zeros(len(tokens))

action = np.array([0.137, -0.442, 0.891, 0.0, -0.213, 0.667, 1.0])
tok = tokenize(action)
rec = detokenize(tok)
print("action    ", np.round(action, 4))
print("tokens    ", tok)
print("recovered ", np.round(rec, 4))
err = np.abs(action-rec).max()
print(f"max quantization error: {err:.5f}")
assert err < 0.01, "256 bins over [-1,1] should give error below 0.004"
print("PASS\\n")

for bins in [64, 256, 1024]:
    res = 1.0/(bins-1)
    print(f"  {bins:5d} bins -> resolution {res:.5f}  ({res*1000:.2f} mm over a 1 m range)")

print("\\ncontrol frequency:")
for name, forwards, per_forward, chunk in [
    ("token decoding, 7 dims, 1 step", 7, 0.020, 1),
    ("token decoding, 7 dims, chunk 10", 70, 0.020, 10),
    ("flow expert, 4 integration steps", 4, 0.012, 50)]:
    lat = forwards * per_forward
    print(f"  {name:36s} {lat*1000:6.0f} ms -> {chunk/lat:6.1f} Hz")`,
  solution: `import numpy as np

def tokenize(action, n_bins=256, lo=-1.0, hi=1.0):
    a = np.clip(action, lo, hi)
    return np.round((a - lo) / (hi - lo) * (n_bins - 1)).astype(int)

def detokenize(tokens, n_bins=256, lo=-1.0, hi=1.0):
    return tokens / (n_bins - 1) * (hi - lo) + lo

action = np.array([0.137, -0.442, 0.891, 0.0, -0.213, 0.667, 1.0])
tok = tokenize(action); rec = detokenize(tok)
print("action    ", np.round(action, 4))
print("tokens    ", tok)
print("recovered ", np.round(rec, 4))
err = np.abs(action-rec).max()
print(f"max quantization error: {err:.5f}")
assert err < 0.01
print("PASS\\n")

for bins in [64, 256, 1024]:
    res = 1.0/(bins-1)
    print(f"  {bins:5d} bins -> resolution {res:.5f}  ({res*1000:.2f} mm over 1 m)")

print("\\ncontrol frequency:")
for name, forwards, per_forward, chunk in [
    ("token decoding, 7 dims, 1 step", 7, 0.020, 1),
    ("token decoding, 7 dims, chunk 10", 70, 0.020, 10),
    ("flow expert, 4 integration steps", 4, 0.012, 50)]:
    lat = forwards * per_forward
    print(f"  {name:36s} {lat*1000:6.0f} ms -> {chunk/lat:6.1f} Hz")
print("\\nAutoregressive decoding costs a forward pass PER DIMENSION. That is the")
print("difference between 5 Hz and 50 Hz control.")`,
},

'emb-world-models': {
  title: 'Watch an imagined rollout diverge, then replan',
  prompt: `Learn a linear dynamics model from limited data and measure how its rollout error grows with horizon. Then
add MPC-style replanning from real observations and show the error stops compounding.`,
  hint: 'Compounding is why Dreamer imagines ~15 steps and bootstraps a value function beyond that.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def true_step(x, v, a):
    nv = 0.98*v + 0.1*a - 0.02*x
    return x + nv, nv

def learn_model(n_real):
    X, Y = [], []
    x, v = rng.normal(), 0.0
    for _ in range(n_real):
        a = rng.normal()
        nx, nv = true_step(x, v, a)
        X.append([x, v, a]); Y.append([nx, nv])
        x, v = nx, nv
        if abs(x) > 6: x, v = rng.normal(), 0.0
    # TODO: least-squares fit mapping [x, v, a] -> [next_x, next_v]
    return np.zeros((3, 2))

def rollout_error(W, horizon, replan_every=None):
    errs = []
    for _ in range(200):
        x = v = mx = mv = 0.0
        for t in range(horizon):
            a = rng.normal()
            x, v = true_step(x, v, a)
            # TODO: step the MODEL forward; if replan_every is set and
            #       t % replan_every == 0, reset (mx, mv) to the true state
            errs.append(abs(mx - x))
    return np.mean(errs)

print(f"{'real transitions':>18} {'H=1':>9} {'H=5':>9} {'H=15':>9} {'H=50':>9}")
for n in [50, 200, 1000, 5000]:
    W = learn_model(n)
    print(f"{n:18d}" + "".join(f"{rollout_error(W,h):9.4f}" for h in [1,5,15,50]))`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def true_step(x, v, a):
    nv = 0.98*v + 0.1*a - 0.02*x
    return x + nv, nv

def learn_model(n_real):
    X, Y = [], []
    x, v = rng.normal(), 0.0
    for _ in range(n_real):
        a = rng.normal()
        nx, nv = true_step(x, v, a)
        X.append([x, v, a]); Y.append([nx, nv])
        x, v = nx, nv
        if abs(x) > 6: x, v = rng.normal(), 0.0
    return np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)[0]

def rollout_error(W, horizon, replan_every=None):
    errs = []
    for _ in range(200):
        x = v = mx = mv = 0.0
        for t in range(horizon):
            a = rng.normal()
            px, pv = x, v
            x, v = true_step(x, v, a)
            if replan_every and t % replan_every == 0:
                mx, mv = px, pv
            mx, mv = np.array([mx, mv, a]) @ W
            errs.append(abs(mx - x))
    return np.mean(errs)

print(f"{'real transitions':>18} {'H=1':>9} {'H=5':>9} {'H=15':>9} {'H=50':>9}")
for n in [50, 200, 1000, 5000]:
    W = learn_model(n)
    print(f"{n:18d}" + "".join(f"{rollout_error(W,h):9.4f}" for h in [1,5,15,50]))

W = learn_model(1000)
print("\\nwith MPC replanning every 5 steps:")
print(f"{'':18s}" + "".join(f"{rollout_error(W,h,5):9.4f}" for h in [1,5,15,50]))

print("\\nto get 1M policy-training steps:")
print(f"  model-free on a real robot at 10 Hz : {1e6/10/3600:8.1f} hours")
print(f"  model-based, 5k real transitions    : {5e3/10/3600:8.2f} hours")`,
},

'emb-benchmarks': {
  title: 'How many trials does a robot evaluation actually need?',
  prompt: `Implement the Wilson interval and apply it to typical robotics trial counts. Determine whether a reported
85% over 20 trials is distinguishable from 75%, and find the n that would separate them.`,
  hint: 'Robot papers commonly report 10-20 trials. Compute the interval width before believing any comparison.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def wilson(successes, n, z=1.96):
    # TODO
    return (0.0, 1.0)

print("what a reported success rate actually tells you:")
print(f"{'trials':>7} {'observed':>10} {'95% CI':>22} {'width':>9}")
for n in [10, 20, 50, 100, 500]:
    s = round(0.85 * n)
    lo, hi = wilson(s, n)
    print(f"{n:7d} {s/n:10.1%} {f'[{lo:.1%}, {hi:.1%}]':>22} {(hi-lo)*100:8.1f}pp")

print("\\ncan you distinguish two policies?")
for n in [20, 50, 200]:
    a = wilson(round(0.75*n), n)
    b = wilson(round(0.85*n), n)
    print(f"  n={n:3d}: 75% [{a[0]:.2f},{a[1]:.2f}] vs 85% [{b[0]:.2f},{b[1]:.2f}]  "
          f"{'overlapping' if a[1] > b[0] else 'separated'}")

# TODO: find the smallest n where the two intervals no longer overlap`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def wilson(successes, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = successes / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0.0, centre-half), min(1.0, centre+half)

print("what a reported success rate actually tells you:")
print(f"{'trials':>7} {'observed':>10} {'95% CI':>22} {'width':>9}")
for n in [10, 20, 50, 100, 500]:
    s = round(0.85 * n)
    lo, hi = wilson(s, n)
    print(f"{n:7d} {s/n:10.1%} {f'[{lo:.1%}, {hi:.1%}]':>22} {(hi-lo)*100:8.1f}pp")

print("\\ncan you distinguish two policies?")
for n in [20, 50, 200]:
    a = wilson(round(0.75*n), n); b = wilson(round(0.85*n), n)
    print(f"  n={n:3d}: 75% [{a[0]:.2f},{a[1]:.2f}] vs 85% [{b[0]:.2f},{b[1]:.2f}]  "
          f"{'overlapping' if a[1] > b[0] else 'separated'}")

for n in range(20, 2000, 10):
    if wilson(round(0.75*n), n)[1] < wilson(round(0.85*n), n)[0]:
        print(f"\\nintervals separate at n = {n} trials per condition")
        break

print("\\nA typical robotics paper reports 10-20 trials. Treat cross-paper")
print("comparisons of success rates as essentially uninformative.")`,
  explain: '17 of 20 gives roughly [64%, 95%]. That is compatible with a strong policy and a mediocre one, and it certainly does not separate 85% from 75% — before you even account for different hardware, scenes, and reset protocols.',
},

};

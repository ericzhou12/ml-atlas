/* ============================================================
   Challenges — track 2, classical ML

   One entry per lesson id: { title, prompt, starter, solution, hint?, explain? }
   Starters carry their own PASS/FAIL check so the learner gets a verdict
   without leaving the lab.

   EDITOR NOTE: these are JS template literals, so an unescaped backtick
   silently terminates the string. Escape inline code in prose.
   ============================================================ */

export default {

'ml-framing': {
  title: 'Watch empirical risk lie to you',
  prompt: `Fit a model on $n$ points and compare its training error to its error on a huge fresh sample. Plot the gap
against $n$. How much data do you need before the training error is trustworthy?`,
  hint: 'Training error is optimistically biased because you chose the parameters to minimize exactly that quantity.',
  starter: `import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
true_f = lambda x: 1.5*x - 0.4
NOISE = 0.5

def sample(n):
    x = rng.uniform(-3, 3, n)
    return x, true_f(x) + rng.normal(0, NOISE, n)

def fit_and_measure(n, deg=5):
    x, y = sample(n)
    # TODO: fit a degree-\`deg\` polynomial, return (train_mse, test_mse)
    return 0.0, 0.0

ns, gaps = [], []
for n in [8, 12, 20, 50, 100, 500, 2000]:
    tr, te = fit_and_measure(n)
    ns.append(n); gaps.append(te - tr)
    print(f"n={n:5d}  train {tr:.4f}  test {te:.4f}  gap {te-tr:+.4f}")
print(f"\\nirreducible noise floor: {NOISE**2:.4f}")`,
  solution: `import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
true_f = lambda x: 1.5*x - 0.4
NOISE = 0.5

def sample(n):
    x = rng.uniform(-3, 3, n)
    return x, true_f(x) + rng.normal(0, NOISE, n)

def fit_and_measure(n, deg=5):
    x, y = sample(n)
    w = np.polyfit(x, y, deg)
    xt, yt = sample(50_000)
    tr = np.mean((np.polyval(w, x) - y) ** 2)
    te = np.mean((np.polyval(w, xt) - yt) ** 2)
    return tr, te

ns, gaps = [], []
for n in [8, 12, 20, 50, 100, 500, 2000]:
    tr, te = fit_and_measure(n)
    ns.append(n); gaps.append(te - tr)
    print(f"n={n:5d}  train {tr:.4f}  test {te:.4f}  gap {te-tr:+.4f}")
print(f"\\nirreducible noise floor: {NOISE**2:.4f}")

plt.semilogx(ns, gaps, "o-")
plt.axhline(0, ls="--", lw=1)
plt.xlabel("training points"); plt.ylabel("test - train MSE")
plt.title("the optimism of empirical risk")`,
},

'ml-linear-regression': {
  title: 'Three ways to solve least squares, and one that breaks',
  prompt: `Solve the same regression with the normal equations, \`lstsq\`, and gradient descent. Then add a
near-duplicate column and watch one of them fall apart.`,
  hint: 'Forming $X^{\\mathsf T}X$ squares the condition number — compare `np.linalg.cond` before and after.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

n, d = 100, 5
X = rng.normal(size=(n, d))
w_true = np.array([2.0, -1.0, 0.5, 0.0, 3.0])
y = X @ w_true + rng.normal(0, 0.3, n)

def solve_normal(X, y):
    # TODO: (X^T X)^-1 X^T y
    return np.zeros(X.shape[1])

def solve_gd(X, y, lr=0.01, steps=5000):
    # TODO: gradient descent on ||Xw - y||^2 / n
    return np.zeros(X.shape[1])

print("truth  :", w_true)
print("normal :", solve_normal(X, y).round(4))
print("lstsq  :", np.linalg.lstsq(X, y, rcond=None)[0].round(4))
print("gd     :", solve_gd(X, y).round(4))

# --- now make it ill-conditioned ---
Xb = X.copy()
Xb[:, 1] = Xb[:, 0] + 1e-7 * rng.normal(size=n)
print(f"\\ncond(X)     = {np.linalg.cond(Xb):.3e}")
print(f"cond(X^T X) = {np.linalg.cond(Xb.T @ Xb):.3e}   <- squared")
print("normal :", solve_normal(Xb, y).round(3))
print("lstsq  :", np.linalg.lstsq(Xb, y, rcond=None)[0].round(3))`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

n, d = 100, 5
X = rng.normal(size=(n, d))
w_true = np.array([2.0, -1.0, 0.5, 0.0, 3.0])
y = X @ w_true + rng.normal(0, 0.3, n)

def solve_normal(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

def solve_gd(X, y, lr=0.01, steps=5000):
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        w -= lr * 2 * X.T @ (X @ w - y) / len(y)
    return w

print("truth  :", w_true)
print("normal :", solve_normal(X, y).round(4))
print("lstsq  :", np.linalg.lstsq(X, y, rcond=None)[0].round(4))
print("gd     :", solve_gd(X, y).round(4))

Xb = X.copy()
Xb[:, 1] = Xb[:, 0] + 1e-7 * rng.normal(size=n)
print(f"\\ncond(X)     = {np.linalg.cond(Xb):.3e}")
print(f"cond(X^T X) = {np.linalg.cond(Xb.T @ Xb):.3e}   <- squared")
print("normal :", solve_normal(Xb, y).round(3))
print("lstsq  :", np.linalg.lstsq(Xb, y, rcond=None)[0].round(3))`,
  explain: 'With two near-identical columns, `lstsq` (which uses SVD) stays sane while the normal equations produce enormous coefficients that cancel. This is why you should never write `inv(X.T @ X)` in production code.',
},

'ml-overfitting': {
  title: 'Reproduce the bias-variance decomposition numerically',
  prompt: `Fit the same model to many independent datasets, then estimate bias² and variance directly from the
definitions and check that they sum to the expected test error minus noise.`,
  hint: 'Bias² is $(\\overline{\\hat f}(x) - f(x))^2$; variance is the spread of $\\hat f(x)$ around its own mean.',
  starter: `import numpy as np

rng = np.random.default_rng(0)
truth = lambda x: np.sin(2.2*x)*1.3 + 0.35*x
NOISE, N_DATA, N_REP = 0.35, 12, 300
grid = np.linspace(-2.4, 2.4, 60)

def fit_many(deg):
    preds = []
    for _ in range(N_REP):
        x = rng.uniform(-2.5, 2.5, N_DATA)
        y = truth(x) + rng.normal(0, NOISE, N_DATA)
        w = np.polyfit(x, y, deg)
        preds.append(np.polyval(w, grid))
    return np.array(preds)      # (N_REP, len(grid))

print(f"{'deg':>4} {'bias^2':>9} {'variance':>10} {'+noise':>9} {'= total':>9}")
for deg in [0, 1, 3, 6, 9, 11]:
    P = fit_many(deg)
    # TODO: compute bias2 and var from P and truth(grid)
    bias2, var = 0.0, 0.0
    print(f"{deg:4d} {bias2:9.4f} {var:10.4f} {NOISE**2:9.4f} {bias2+var+NOISE**2:9.4f}")`,
  solution: `import numpy as np

rng = np.random.default_rng(0)
truth = lambda x: np.sin(2.2*x)*1.3 + 0.35*x
NOISE, N_DATA, N_REP = 0.35, 12, 300
grid = np.linspace(-2.4, 2.4, 60)

def fit_many(deg):
    preds = []
    for _ in range(N_REP):
        x = rng.uniform(-2.5, 2.5, N_DATA)
        y = truth(x) + rng.normal(0, NOISE, N_DATA)
        preds.append(np.polyval(np.polyfit(x, y, deg), grid))
    return np.array(preds)

print(f"{'deg':>4} {'bias^2':>9} {'variance':>10} {'+noise':>9} {'= total':>9}")
for deg in [0, 1, 3, 6, 9, 11]:
    P = fit_many(deg)
    mean_pred = P.mean(0)
    bias2 = np.mean((mean_pred - truth(grid)) ** 2)
    var = np.mean(P.var(0))
    print(f"{deg:4d} {bias2:9.4f} {var:10.4f} {NOISE**2:9.4f} {bias2+var+NOISE**2:9.4f}")`,
  explain: 'Bias falls monotonically with degree while variance climbs, and their sum has a minimum in the middle. Notice how violently variance explodes past degree 9 with only 12 data points.',
},

'ml-regularization': {
  title: 'Implement Lasso with coordinate descent',
  prompt: `Write the soft-thresholding update and use it to recover a sparse signal. Verify Lasso produces *exactly*
zero coefficients while Ridge does not.`,
  hint: 'The one-coordinate solution is $\\mathrm{sign}(\\rho)\\max(|\\rho|-\\lambda, 0)/z$ — the `max(·, 0)` is where zeros come from.',
  starter: `import numpy as np
rng = np.random.default_rng(4)

n, d = 60, 20
X = rng.normal(size=(n, d))
X = (X - X.mean(0)) / X.std(0)          # standardize before regularizing!
w_true = np.zeros(d); w_true[[0, 3, 7]] = [2.5, -1.8, 1.2]
y = X @ w_true + rng.normal(0, 0.5, n)

def ridge(lam):
    return np.linalg.solve(X.T @ X + lam*np.eye(d), X.T @ y)

def lasso(lam, iters=500):
    w = np.zeros(d)
    XtX, Xty = X.T @ X, X.T @ y
    for _ in range(iters):
        for j in range(d):
            rho = Xty[j] - XtX[j] @ w + XtX[j, j] * w[j]
            # TODO: soft-threshold rho by lam, divide by XtX[j,j]
            w[j] = 0.0
    return w

for lam in [1.0, 5.0, 20.0]:
    wr, wl = ridge(lam), lasso(lam)
    print(f"lam={lam:5.1f}  ridge zeros: {np.sum(np.abs(wr)<1e-8):2d}/{d}   "
          f"lasso zeros: {np.sum(np.abs(wl)<1e-8):2d}/{d}")

wl = lasso(5.0)
found = sorted(np.argsort(-np.abs(wl))[:3])
print(f"\\nlasso top-3 features: {found}   truth: [0, 3, 7]")
print("PASS" if found == [0, 3, 7] else "FAIL")`,
  solution: `import numpy as np
rng = np.random.default_rng(4)

n, d = 60, 20
X = rng.normal(size=(n, d))
X = (X - X.mean(0)) / X.std(0)
w_true = np.zeros(d); w_true[[0, 3, 7]] = [2.5, -1.8, 1.2]
y = X @ w_true + rng.normal(0, 0.5, n)

def ridge(lam):
    return np.linalg.solve(X.T @ X + lam*np.eye(d), X.T @ y)

def lasso(lam, iters=500):
    w = np.zeros(d)
    XtX, Xty = X.T @ X, X.T @ y
    for _ in range(iters):
        for j in range(d):
            rho = Xty[j] - XtX[j] @ w + XtX[j, j] * w[j]
            w[j] = np.sign(rho) * max(abs(rho) - lam, 0.0) / XtX[j, j]
    return w

for lam in [1.0, 5.0, 20.0]:
    wr, wl = ridge(lam), lasso(lam)
    print(f"lam={lam:5.1f}  ridge zeros: {np.sum(np.abs(wr)<1e-8):2d}/{d}   "
          f"lasso zeros: {np.sum(np.abs(wl)<1e-8):2d}/{d}")

wl = lasso(5.0)
found = sorted(np.argsort(-np.abs(wl))[:3])
print(f"\\nlasso top-3 features: {found}   truth: [0, 3, 7]")
print("PASS" if found == [0, 3, 7] else "FAIL")`,
},

'ml-logistic': {
  title: 'Logistic regression, and watching ‖w‖ run away',
  prompt: `Implement the gradient and train on separable data with no regularization. Watch $\\|w\\|$ grow without
bound, then add an L2 penalty and confirm it stops.`,
  hint: 'The gradient is $\\frac1n X^{\\mathsf T}(\\sigma(Xw) - y)$ — remember not to penalize the intercept.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

# perfectly separable data
n = 200
X = np.vstack([rng.normal([-2, -2], 0.5, (n//2, 2)),
               rng.normal([ 2,  2], 0.5, (n//2, 2))])
y = np.r_[np.zeros(n//2), np.ones(n//2)]
Xb = np.column_stack([np.ones(n), X])

def sigmoid(z): return 1/(1+np.exp(-np.clip(z, -500, 500)))

def train(l2, steps=20000, lr=0.5):
    w = np.zeros(3)
    for t in range(steps):
        p = sigmoid(Xb @ w)
        # TODO: gradient of mean BCE + l2 penalty on w[1:] only
        grad = np.zeros(3)
        w -= lr * grad
        if t in (100, 1000, 5000, 19999):
            loss = -np.mean(y*np.log(p+1e-12) + (1-y)*np.log(1-p+1e-12))
            print(f"  step {t:6d}  loss {loss:.6f}  |w| {np.linalg.norm(w[1:]):8.3f}")
    return w

print("no regularization:");  train(0.0)
print("\\nwith L2 = 0.01:");  train(0.01)`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

n = 200
X = np.vstack([rng.normal([-2, -2], 0.5, (n//2, 2)),
               rng.normal([ 2,  2], 0.5, (n//2, 2))])
y = np.r_[np.zeros(n//2), np.ones(n//2)]
Xb = np.column_stack([np.ones(n), X])

def sigmoid(z): return 1/(1+np.exp(-np.clip(z, -500, 500)))

def train(l2, steps=20000, lr=0.5):
    w = np.zeros(3)
    for t in range(steps):
        p = sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) / n + l2 * np.r_[0.0, w[1:]]
        w -= lr * grad
        if t in (100, 1000, 5000, 19999):
            loss = -np.mean(y*np.log(p+1e-12) + (1-y)*np.log(1-p+1e-12))
            print(f"  step {t:6d}  loss {loss:.6f}  |w| {np.linalg.norm(w[1:]):8.3f}")
    return w

print("no regularization:");  train(0.0)
print("\\nwith L2 = 0.01:");  train(0.01)`,
  explain: 'On separable data the log-loss has no minimum — scaling $w$ up always improves it, so $\\|w\\|\\to\\infty$ and the loss creeps toward 0 forever. L2 gives the objective a finite optimum. The *direction* it diverges along converges to the max-margin separator, which is a neat connection to SVMs.',
},

'ml-trees-ensembles': {
  title: 'Grow a regression stump, then boost it',
  prompt: `Write a depth-1 regression tree that picks the best split by variance reduction, then boost a hundred of
them on residuals. Compare the learning rate's effect on overfitting.`,
  hint: 'Each new stump is fit to the *residuals* $y - F_{m-1}(x)$, and added with a shrinkage factor.',
  starter: `import numpy as np
rng = np.random.default_rng(14)

truth = lambda x: np.sin(x*1.6)*1.2 + 0.3*x
x = rng.uniform(-2.5, 2.5, 60)
y = truth(x) + rng.normal(0, 0.25, 60)
xt = np.linspace(-2.5, 2.5, 500); yt = truth(xt)

def fit_stump(x, resid):
    """Return (threshold, left_value, right_value) minimizing squared error."""
    best = None
    for thr in np.quantile(x, np.linspace(0.05, 0.95, 30)):
        L, R = x <= thr, x > thr
        if L.sum() < 2 or R.sum() < 2: continue
        # TODO: compute left/right means and the total squared error
        err = np.inf; lm = rm = 0.0
        if best is None or err < best[0]:
            best = (err, thr, lm, rm)
    return best[1], best[2], best[3]

def boost(M, lr):
    base = y.mean()
    pred = np.full_like(y, base)
    stumps = []
    for _ in range(M):
        thr, lm, rm = fit_stump(x, y - pred)
        stumps.append((thr, lm, rm))
        pred += lr * np.where(x <= thr, lm, rm)
    F = lambda q: base + sum(lr*np.where(q <= t, l, r) for t, l, r in stumps)
    return F

for M, lr in [(10, 0.3), (100, 0.3), (100, 1.0), (400, 1.0)]:
    F = boost(M, lr)
    print(f"M={M:4d} lr={lr:.1f}  train {np.mean((F(x)-y)**2):.4f}  "
          f"test {np.mean((F(xt)-yt)**2):.4f}")`,
  solution: `import numpy as np
rng = np.random.default_rng(14)

truth = lambda x: np.sin(x*1.6)*1.2 + 0.3*x
x = rng.uniform(-2.5, 2.5, 60)
y = truth(x) + rng.normal(0, 0.25, 60)
xt = np.linspace(-2.5, 2.5, 500); yt = truth(xt)

def fit_stump(x, resid):
    best = None
    for thr in np.quantile(x, np.linspace(0.05, 0.95, 30)):
        L, R = x <= thr, x > thr
        if L.sum() < 2 or R.sum() < 2: continue
        lm, rm = resid[L].mean(), resid[R].mean()
        err = ((resid[L]-lm)**2).sum() + ((resid[R]-rm)**2).sum()
        if best is None or err < best[0]:
            best = (err, thr, lm, rm)
    return best[1], best[2], best[3]

def boost(M, lr):
    base = y.mean()
    pred = np.full_like(y, base)
    stumps = []
    for _ in range(M):
        thr, lm, rm = fit_stump(x, y - pred)
        stumps.append((thr, lm, rm))
        pred += lr * np.where(x <= thr, lm, rm)
    return lambda q: base + sum(lr*np.where(q <= t, l, r) for t, l, r in stumps)

for M, lr in [(10, 0.3), (100, 0.3), (100, 1.0), (400, 1.0)]:
    F = boost(M, lr)
    print(f"M={M:4d} lr={lr:.1f}  train {np.mean((F(x)-y)**2):.4f}  "
          f"test {np.mean((F(xt)-yt)**2):.4f}")`,
  explain: 'Small learning rate with many trees generalizes better than few trees with a large one — the same shrinkage principle as in gradient descent. At `lr=1.0, M=400` train error keeps falling while test error rises: boosting *can* overfit, unlike random forests.',
},

'ml-svm-knn': {
  title: 'Hinge loss vs log loss, and the support vectors',
  prompt: `Train a linear SVM by subgradient descent on the hinge loss. Count how many points end up as support
vectors, then delete a non-support point and confirm the solution is unchanged.`,
  hint: 'The hinge subgradient is $-y\\mathbf{x}$ when the margin $y f(\\mathbf{x}) < 1$, and zero otherwise.',
  starter: `import numpy as np
rng = np.random.default_rng(2)

n = 80
X = np.vstack([rng.normal([-1.5, -1], 0.7, (n//2, 2)),
               rng.normal([ 1.5,  1], 0.7, (n//2, 2))])
y = np.r_[-np.ones(n//2), np.ones(n//2)]

def train_svm(X, y, C=1.0, steps=4000):
    w, b = np.zeros(2), 0.0
    for t in range(steps):
        lr = 0.05 / (1 + t*0.002)
        margin = y * (X @ w + b)
        # TODO: subgradient of  0.5||w||^2 + C * mean(max(0, 1 - margin))
        gw = np.zeros(2); gb = 0.0
        w -= lr * gw; b -= lr * gb
    return w, b

w, b = train_svm(X, y)
margins = y * (X @ w + b)
sv = margins <= 1.0 + 1e-2
print(f"margin width 2/|w| = {2/np.linalg.norm(w):.4f}")
print(f"support vectors: {sv.sum()} / {n}")

# delete a NON-support point and refit
keep = np.ones(n, bool); keep[np.argmax(margins)] = False
w2, b2 = train_svm(X[keep], y[keep])
print(f"\\nw before {w.round(4)}  after {w2.round(4)}")
print("PASS" if np.allclose(w, w2, atol=0.05) else "FAIL")`,
  solution: `import numpy as np
rng = np.random.default_rng(2)

n = 80
X = np.vstack([rng.normal([-1.5, -1], 0.7, (n//2, 2)),
               rng.normal([ 1.5,  1], 0.7, (n//2, 2))])
y = np.r_[-np.ones(n//2), np.ones(n//2)]

def train_svm(X, y, C=1.0, steps=4000):
    w, b = np.zeros(2), 0.0
    for t in range(steps):
        lr = 0.05 / (1 + t*0.002)
        margin = y * (X @ w + b)
        active = margin < 1
        gw = w - C * (X[active] * y[active, None]).sum(0) / len(y)
        gb = -C * y[active].sum() / len(y)
        w -= lr * gw; b -= lr * gb
    return w, b

w, b = train_svm(X, y)
margins = y * (X @ w + b)
sv = margins <= 1.0 + 1e-2
print(f"margin width 2/|w| = {2/np.linalg.norm(w):.4f}")
print(f"support vectors: {sv.sum()} / {n}")

keep = np.ones(n, bool); keep[np.argmax(margins)] = False
w2, b2 = train_svm(X[keep], y[keep])
print(f"\\nw before {w.round(4)}  after {w2.round(4)}")
print("PASS" if np.allclose(w, w2, atol=0.05) else "FAIL")`,
  explain: 'Removing the point furthest inside its own margin changes nothing — only the support vectors determine the solution. That sparsity is what makes the kernel trick affordable at prediction time.',
},

'ml-unsupervised': {
  title: 'k-means, and how much the initialization matters',
  prompt: `Implement Lloyd's algorithm, then run it from 20 random initializations on the same data and report the
spread of final inertias. Add k-means++ seeding and compare.`,
  hint: 'k-means++ picks each new centre with probability proportional to squared distance from the nearest existing centre.',
  starter: `import numpy as np
rng = np.random.default_rng(2)

X = np.vstack([rng.normal([0,0], 0.4, (120,2)),
               rng.normal([3,3], 0.4, (120,2)),
               rng.normal([0,3], 0.4, (120,2)),
               rng.normal([3,0], 0.4, (120,2))])

def lloyd(X, C, iters=100):
    for _ in range(iters):
        # TODO: assign points to nearest centre, then move centres to the mean
        pass
    d = ((X[:,None] - C[None])**2).sum(-1)
    a = d.argmin(1)
    return C, a, ((X - C[a])**2).sum()

def init_random(X, k, r):
    return X[r.choice(len(X), k, replace=False)]

def init_pp(X, k, r):
    C = [X[r.integers(len(X))]]
    for _ in range(k-1):
        d2 = ((X[:,None] - np.array(C)[None])**2).sum(-1).min(1)
        C.append(X[r.choice(len(X), p=d2/d2.sum())])
    return np.array(C)

for name, init in [("random", init_random), ("k-means++", init_pp)]:
    scores = [lloyd(X, init(X, 4, np.random.default_rng(s)))[2] for s in range(20)]
    print(f"{name:10s} inertia: best {min(scores):7.2f}  worst {max(scores):7.2f}  "
          f"spread {max(scores)-min(scores):7.2f}")`,
  solution: `import numpy as np
rng = np.random.default_rng(2)

X = np.vstack([rng.normal([0,0], 0.4, (120,2)),
               rng.normal([3,3], 0.4, (120,2)),
               rng.normal([0,3], 0.4, (120,2)),
               rng.normal([3,0], 0.4, (120,2))])

def lloyd(X, C, iters=100):
    C = C.copy()
    for _ in range(iters):
        d = ((X[:,None] - C[None])**2).sum(-1)
        a = d.argmin(1)
        for j in range(len(C)):
            if (a == j).any(): C[j] = X[a == j].mean(0)
    d = ((X[:,None] - C[None])**2).sum(-1)
    a = d.argmin(1)
    return C, a, ((X - C[a])**2).sum()

def init_random(X, k, r): return X[r.choice(len(X), k, replace=False)]

def init_pp(X, k, r):
    C = [X[r.integers(len(X))]]
    for _ in range(k-1):
        d2 = ((X[:,None] - np.array(C)[None])**2).sum(-1).min(1)
        C.append(X[r.choice(len(X), p=d2/d2.sum())])
    return np.array(C)

for name, init in [("random", init_random), ("k-means++", init_pp)]:
    scores = [lloyd(X, init(X, 4, np.random.default_rng(s)))[2] for s in range(20)]
    print(f"{name:10s} inertia: best {min(scores):7.2f}  worst {max(scores):7.2f}  "
          f"spread {max(scores)-min(scores):7.2f}")`,
  explain: 'Random seeding produces a wide spread of final inertias — k-means converges reliably to a *local* optimum that depends entirely on where it started. k-means++ narrows the spread substantially, which is why it is the default in every library.',
},

'ml-evaluation': {
  title: 'Put a confidence interval on an accuracy',
  prompt: `Implement the Wilson score interval and use it to answer: with 20 test examples, can you distinguish a
model that scores 85% from one that scores 75%? Then find how many you would need.`,
  hint: 'Also try the bootstrap: resample your predictions with replacement 1000 times and take percentiles.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def wilson(successes, n, z=1.96):
    """95% confidence interval for a proportion."""
    # TODO
    return (0.0, 1.0)

print(f"{'n':>6} {'observed':>10} {'95% CI':>24} {'width':>9}")
for n in [10, 20, 50, 200, 1000]:
    s = round(0.85 * n)
    lo, hi = wilson(s, n)
    print(f"{n:6d} {s/n:10.1%} {f'[{lo:.1%}, {hi:.1%}]':>24} {(hi-lo)*100:8.1f}pp")

lo85, hi85 = wilson(17, 20)
lo75, hi75 = wilson(15, 20)
print(f"\\n85% on n=20: [{lo85:.3f}, {hi85:.3f}]")
print(f"75% on n=20: [{lo75:.3f}, {hi75:.3f}]")
print("overlapping ->", "cannot distinguish" if hi75 > lo85 else "distinguishable")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def wilson(successes, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = successes / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0.0, centre-half), min(1.0, centre+half)

print(f"{'n':>6} {'observed':>10} {'95% CI':>24} {'width':>9}")
for n in [10, 20, 50, 200, 1000]:
    s = round(0.85 * n)
    lo, hi = wilson(s, n)
    print(f"{n:6d} {s/n:10.1%} {f'[{lo:.1%}, {hi:.1%}]':>24} {(hi-lo)*100:8.1f}pp")

lo85, hi85 = wilson(17, 20); lo75, hi75 = wilson(15, 20)
print(f"\\n85% on n=20: [{lo85:.3f}, {hi85:.3f}]")
print(f"75% on n=20: [{lo75:.3f}, {hi75:.3f}]")
print("overlapping ->", "cannot distinguish" if hi75 > lo85 else "distinguishable")

for n in range(20, 2000, 20):
    if wilson(round(0.75*n), n)[1] < wilson(round(0.85*n), n)[0]:
        print(f"\\nneed n >= {n} for the intervals to separate")
        break`,
},

'ml-generative-discriminative': {
  title: 'Find the crossover between naive Bayes and logistic regression',
  prompt: `Train both on the same data at increasing sample sizes and find the $n$ at which the discriminative model
overtakes the generative one. Ng & Jordan predicted this crossover exists — locate it.`,
  hint: 'Generative wins at small $n$ (strong assumptions act as a prior); discriminative wins as $n$ grows.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

d = 30
mu0, mu1 = rng.normal(0, 1, d), rng.normal(0.8, 1, d)

def make(n):
    y = rng.integers(0, 2, n)
    return rng.normal(np.where(y[:,None], mu1, mu0), 1.0), y

def naive_bayes(Xtr, ytr, Xte):
    # TODO: fit per-class means and stds, score with the Gaussian log-likelihood
    return np.zeros(len(Xte), dtype=int)

def logistic(Xtr, ytr, Xte, steps=3000):
    Xb = np.c_[np.ones(len(Xtr)), Xtr]; w = np.zeros(d+1)
    for _ in range(steps):
        p = 1/(1+np.exp(-np.clip(Xb @ w, -500, 500)))
        w -= 0.5 * (Xb.T @ (p - ytr)/len(ytr) + 0.01*np.r_[0, w[1:]])
    return (np.c_[np.ones(len(Xte)), Xte] @ w > 0).astype(int)

Xte, yte = make(4000)
print(f"{'n':>6} {'naive Bayes':>13} {'logistic':>10}")
for n in [20, 50, 100, 300, 1000, 3000]:
    Xtr, ytr = make(n)
    nb = (naive_bayes(Xtr, ytr, Xte) == yte).mean()
    lr = (logistic(Xtr, ytr, Xte) == yte).mean()
    flag = "  <- crossover" if lr > nb else ""
    print(f"{n:6d} {nb:13.3f} {lr:10.3f}{flag}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

d = 30
mu0, mu1 = rng.normal(0, 1, d), rng.normal(0.8, 1, d)

def make(n):
    y = rng.integers(0, 2, n)
    return rng.normal(np.where(y[:,None], mu1, mu0), 1.0), y

def naive_bayes(Xtr, ytr, Xte):
    m0, m1 = Xtr[ytr==0].mean(0), Xtr[ytr==1].mean(0)
    s0 = Xtr[ytr==0].std(0) + 1e-6
    s1 = Xtr[ytr==1].std(0) + 1e-6
    ll0 = -(((Xte-m0)**2)/(2*s0**2) + np.log(s0)).sum(1)
    ll1 = -(((Xte-m1)**2)/(2*s1**2) + np.log(s1)).sum(1)
    return (ll1 > ll0).astype(int)

def logistic(Xtr, ytr, Xte, steps=3000):
    Xb = np.c_[np.ones(len(Xtr)), Xtr]; w = np.zeros(d+1)
    for _ in range(steps):
        p = 1/(1+np.exp(-np.clip(Xb @ w, -500, 500)))
        w -= 0.5 * (Xb.T @ (p - ytr)/len(ytr) + 0.01*np.r_[0, w[1:]])
    return (np.c_[np.ones(len(Xte)), Xte] @ w > 0).astype(int)

Xte, yte = make(4000)
print(f"{'n':>6} {'naive Bayes':>13} {'logistic':>10}")
for n in [20, 50, 100, 300, 1000, 3000]:
    Xtr, ytr = make(n)
    nb = (naive_bayes(Xtr, ytr, Xte) == yte).mean()
    lr = (logistic(Xtr, ytr, Xte) == yte).mean()
    print(f"{n:6d} {nb:13.3f} {lr:10.3f}{'  <- crossover' if lr > nb else ''}")`,
},

};

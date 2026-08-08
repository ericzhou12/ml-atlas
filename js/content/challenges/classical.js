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
  title: 'Watch empirical risk lie to you, twice',
  prompt: `Two ways the number on your screen can be better than the truth.

1. **Fitting.** \`fit_and_measure(n)\` should fit a model on $n$ points and return its error on those same
   points alongside its error on a large fresh sample. Fill it in, then read how the gap behaves as $n$ grows
   and as the model gets more flexible.
2. **Selecting.** The second block never trains on the test set at all — it just picks whichever of $K$ honest
   models scores best on it. Watch the reported score climb with $K$ while the *fresh* score of the same chosen
   model does not. **Predict what happens before running it.**`,
  hint: '`np.polyfit(x, y, deg)` returns coefficients; `np.polyval(w, x)` evaluates them. For part 2, nothing needs implementing — read the two columns and compare.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

true_f = lambda x: 1.5*x - 0.4
NOISE = 0.5

def sample(n):
    x = rng.uniform(-3, 3, n)
    return x, true_f(x) + rng.normal(0, NOISE, n)

def fit_and_measure(n, deg):
    """Fit a degree-deg polynomial on n points; return (error on those points,
       error on 50,000 fresh ones)."""
    x, y = sample(n)
    # TODO
    return 0.0, 0.0

print("degree 1 (two parameters):")
for n in [8, 12, 20, 50, 200, 2000]:
    tr, te = fit_and_measure(n, 1)
    print(f"  n={n:5d}   train {tr:.4f}   fresh {te:.4f}   gap {te-tr:+.4f}")

print("\\ndegree 8 (nine parameters), same data sizes:")
for n in [12, 20, 50, 200, 2000]:
    tr, te = fit_and_measure(n, 8)
    print(f"  n={n:5d}   train {tr:.4f}   fresh {te:.4f}   gap {te-tr:+.4f}")

print(f"\\nirreducible noise floor: {NOISE**2:.4f}")
tr, te = fit_and_measure(2000, 1)
assert abs(te - NOISE**2) < 0.05, "with plenty of data the honest error should approach the noise floor"
assert fit_and_measure(12, 8)[0] < NOISE**2, "a flexible model on 12 points should beat the noise floor on its own data"
print("PASS\\n")

# ---------- part 2: choosing on the test set, without ever training on it ----------
# 400 candidate classifiers that are all, provably, exactly 50% accurate:
# each one just guesses at random. Then pick the "best" one by test accuracy.
rng2 = np.random.default_rng(1)
N_TEST, K = 500, 400
y_test    = rng2.integers(0, 2, N_TEST)
guesses   = rng2.integers(0, 2, (K, N_TEST))
test_acc  = (guesses == y_test).mean(axis=1)

print("   K   best-of-K test accuracy   the SAME model, on fresh data")
for k in [1, 5, 25, 100, 400]:
    best = int(np.argmax(test_acc[:k]))
    y_fresh = rng2.integers(0, 2, 100_000)
    fresh   = (rng2.integers(0, 2, 100_000) == y_fresh).mean()
    print(f"{k:5d}             {test_acc[best]:.4f}                    {fresh:.4f}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

true_f = lambda x: 1.5*x - 0.4
NOISE = 0.5

def sample(n):
    x = rng.uniform(-3, 3, n)
    return x, true_f(x) + rng.normal(0, NOISE, n)

def fit_and_measure(n, deg):
    x, y = sample(n)
    w = np.polyfit(x, y, deg)
    xt, yt = sample(50_000)
    tr = np.mean((np.polyval(w, x) - y) ** 2)
    te = np.mean((np.polyval(w, xt) - yt) ** 2)
    return tr, te

print("degree 1 (two parameters):")
for n in [8, 12, 20, 50, 200, 2000]:
    tr, te = fit_and_measure(n, 1)
    print(f"  n={n:5d}   train {tr:.4f}   fresh {te:.4f}   gap {te-tr:+.4f}")

print("\\ndegree 8 (nine parameters), same data sizes:")
for n in [12, 20, 50, 200, 2000]:
    tr, te = fit_and_measure(n, 8)
    print(f"  n={n:5d}   train {tr:.4f}   fresh {te:.4f}   gap {te-tr:+.4f}")

print(f"\\nirreducible noise floor: {NOISE**2:.4f}")
tr, te = fit_and_measure(2000, 1)
assert abs(te - NOISE**2) < 0.05
assert fit_and_measure(12, 8)[0] < NOISE**2
print("PASS\\n")

# ---------- part 2: choosing on the test set, without ever training on it ----------
# 400 candidate classifiers that are all, provably, exactly 50% accurate:
# each one just guesses at random. Then pick the "best" one by test accuracy.
rng2 = np.random.default_rng(1)
N_TEST, K = 500, 400
y_test    = rng2.integers(0, 2, N_TEST)
guesses   = rng2.integers(0, 2, (K, N_TEST))
test_acc  = (guesses == y_test).mean(axis=1)

print("   K   best-of-K test accuracy   the SAME model, on fresh data")
for k in [1, 5, 25, 100, 400]:
    best = int(np.argmax(test_acc[:k]))
    y_fresh = rng2.integers(0, 2, 100_000)
    fresh   = (rng2.integers(0, 2, 100_000) == y_fresh).mean()
    print(f"{k:5d}             {test_acc[best]:.4f}                    {fresh:.4f}")`,
  explain: `Part 1: compare the two tables. The degree-1 rows bounce around zero, because a two-parameter line
has essentially nothing to overfit. The degree-8 rows are a different world: at $n=12$ the training error drops
*below* the noise floor — impossible for any honest model, since 0.25 is the error of the true function itself —
and the fresh error is catastrophic. Nine parameters and twelve points means the model spent its freedom
memorising noise. That is [overfitting](#/l/ml-overfitting), arriving a lesson early.

Part 2 is the one worth sitting with. Every one of those 400 classifiers is a coin flip, so every one of them
has a true accuracy of exactly 50% — there is no better model in the pool to find. Yet best-of-400 reports about
**57%**, and it reports it honestly: that model really did get 57% on the test set. The right-hand column, the
same model measured on data that played no part in choosing it, sits at 50% every time.

Nothing improper happened anywhere. No model trained on the test set. The only thing that grew was the number of
candidates allowed to compete, and the maximum of many noisy numbers drifts upward on its own. A seven-point
lead, conjured out of nothing but looking. This is why the rule is to look once.`,
},

'ml-linear-regression': {
  title: 'Check the right angle, then break the coefficients without breaking the predictions',
  prompt: `Three things the lesson claimed. Verify all three.

1. Implement \`solve_normal\` and \`solve_gd\`. Closed form and gradient descent should agree to several decimals,
   because they are solving the same problem.
2. **The right angle.** The lesson said least squares works by dropping a perpendicular: the residual ends up
   orthogonal to every feature column. Fill in \`max_alignment\` and confirm it — every dot product should be
   zero to floating-point dust.
3. **Multicollinearity.** The last block adds a near-duplicate column, then reports the standard errors and the
   prediction error. One of those two explodes and one does not. **Predict which before you run it.**`,
  hint: 'The residual is `y - X @ w`. Orthogonal to every column means `X.T @ residual` is the zero vector. For the standard errors, use $\\sigma^2 (X^{\\mathsf T}X)^{-1}$ with $\\sigma^2$ estimated from the residuals.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

n, d = 100, 5
X = rng.normal(size=(n, d))
w_true = np.array([2.0, -1.0, 0.5, 0.0, 3.0])
y = X @ w_true + rng.normal(0, 0.3, n)

def solve_normal(X, y):
    # TODO: solve X^T X w = X^T y
    return np.zeros(X.shape[1])

def solve_gd(X, y, lr=0.01, steps=8000):
    # TODO: gradient descent. The gradient of ||Xw - y||^2 / n is 2 X^T (Xw - y) / n.
    return np.zeros(X.shape[1])

def max_alignment(X, y, w):
    """The largest |dot product| between a feature column and the residual."""
    # TODO
    return 1.0

w_n  = solve_normal(X, y)
w_ls = np.linalg.lstsq(X, y, rcond=None)[0]
w_gd = solve_gd(X, y)

print("truth  :", w_true)
print("normal :", w_n.round(4))
print("lstsq  :", w_ls.round(4))
print("gd     :", w_gd.round(4))
assert np.abs(w_n - w_ls).max() < 1e-8,  "closed forms must agree"
assert np.abs(w_gd - w_ls).max() < 1e-3, "gradient descent should converge to the same place"

print(f"\\nlargest |column . residual| at the solution : {max_alignment(X, y, w_ls):.2e}")
print(f"the same, for a deliberately wrong w        : {max_alignment(X, y, w_ls + 0.1):.2e}")
assert max_alignment(X, y, w_ls) < 1e-9, "at the optimum the residual must be orthogonal to every column"
print("PASS -- the right angle is real\\n")

# ---------- multicollinearity ----------
def std_errors(X, y, w):
    r = y - X @ w
    sigma2 = (r @ r) / (len(y) - X.shape[1])
    return np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))

Xdup = np.column_stack([X, X[:, 0] + 1e-4 * rng.normal(size=n)])   # a 6th, redundant column

for label, Xi in [("five independent columns", X),
                  ("plus a sixth that nearly copies column 0", Xdup)]:
    wi = np.linalg.lstsq(Xi, y, rcond=None)[0]
    print(f"{label}:")
    print(f"   coefficients   : {np.round(wi, 2)}")
    print(f"   standard errors: {np.round(std_errors(Xi, y, wi), 2)}")
    print(f"   prediction RMSE: {np.sqrt(np.mean((Xi @ wi - y)**2)):.4f}\\n")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

n, d = 100, 5
X = rng.normal(size=(n, d))
w_true = np.array([2.0, -1.0, 0.5, 0.0, 3.0])
y = X @ w_true + rng.normal(0, 0.3, n)

def solve_normal(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

def solve_gd(X, y, lr=0.01, steps=8000):
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        w = w - lr * 2 * X.T @ (X @ w - y) / len(y)
    return w

def max_alignment(X, y, w):
    return np.abs(X.T @ (y - X @ w)).max()

w_n  = solve_normal(X, y)
w_ls = np.linalg.lstsq(X, y, rcond=None)[0]
w_gd = solve_gd(X, y)

print("truth  :", w_true)
print("normal :", w_n.round(4))
print("lstsq  :", w_ls.round(4))
print("gd     :", w_gd.round(4))
assert np.abs(w_n - w_ls).max() < 1e-8
assert np.abs(w_gd - w_ls).max() < 1e-3

print(f"\\nlargest |column . residual| at the solution : {max_alignment(X, y, w_ls):.2e}")
print(f"the same, for a deliberately wrong w        : {max_alignment(X, y, w_ls + 0.1):.2e}")
assert max_alignment(X, y, w_ls) < 1e-9
print("PASS -- the right angle is real\\n")

def std_errors(X, y, w):
    r = y - X @ w
    sigma2 = (r @ r) / (len(y) - X.shape[1])
    return np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))

Xdup = np.column_stack([X, X[:, 0] + 1e-4 * rng.normal(size=n)])   # a 6th, redundant column

for label, Xi in [("five independent columns", X),
                  ("plus a sixth that nearly copies column 0", Xdup)]:
    wi = np.linalg.lstsq(Xi, y, rcond=None)[0]
    print(f"{label}:")
    print(f"   coefficients   : {np.round(wi, 2)}")
    print(f"   standard errors: {np.round(std_errors(Xi, y, wi), 2)}")
    print(f"   prediction RMSE: {np.sqrt(np.mean((Xi @ wi - y)**2)):.4f}\\n")`,
  explain: `Part 2 is the geometric claim, checked. At the least-squares solution every feature column is
orthogonal to the residual — the dot products come out around $10^{-13}$, which is zero as far as float64 is
concerned. Nudge $\\mathbf{w}$ off the optimum and they immediately become large. That is the whole content of
the normal equations: *stop when there is no feature left that correlates with your error.*

Part 3 is multicollinearity, and the numbers are worth reading carefully. Adding one redundant column turns the
weight on column 0 from a sober **1.98** into **+405.7**, with **−403.7** on its near-twin. The standard errors
jump from 0.03 to 277, which is the fit telling you outright that it cannot separate the two.

Now look at the last line of each block: **the prediction error does not change at all.** The *sum* of those two
coefficients is still about 2.0 — the true weight — even though the split between them is arbitrary. So the model
predicts exactly as well as before, while every story you might tell about an individual coefficient has become
worthless. Nothing in the fit fails loudly; only the standard errors give it away. Regularization, in [the next lessons](#/l/ml-regularization), is how you force the fit
to pick a sensible split.`,
},

'ml-overfitting': {
  title: 'Reproduce the bias–variance decomposition, and check that it is really an identity',
  prompt: `The lesson said the three-way split is an exact identity, not an approximation. Test that claim.

The setup fits the same polynomial to 400 independently drawn datasets, so you can see the whole cloud of
models the dartboard picture describes, rather than the single dart you normally get.

1. From that cloud, compute **bias²** — how far the *average* model is from the truth — and **variance** — how
   much the individual models scatter around their own average.
2. The script separately measures the actual expected test error by brute force. Your two numbers plus $\\sigma^2$
   have to match it. The assertion checks that they do.
3. Then read the table and find where the total is smallest.`,
  hint: '`P` has one row per dataset and one column per grid point. `P.mean(0)` is the average model; `P.var(0)` is the per-point variance. Average each over the grid.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

truth = lambda x: np.sin(2.2*x)*1.3 + 0.35*x
NOISE, N_DATA, N_REP = 0.35, 30, 400
grid = np.linspace(-2.2, 2.2, 60)

def fit_many(deg):
    """Fit the same model to N_REP independent datasets. Returns (N_REP, len(grid))."""
    out = []
    for _ in range(N_REP):
        x = rng.uniform(-2.5, 2.5, N_DATA)
        y = truth(x) + rng.normal(0, NOISE, N_DATA)
        out.append(np.polyval(np.polyfit(x, y, deg), grid))
    return np.array(out)

def bias2_and_var(P):
    # TODO: return (bias^2, variance), each averaged over the grid
    return 0.0, 0.0

def measured_test_error(P):
    """Brute force: draw a fresh noisy y at every grid point for every model, and
       average the squared error. This is the left-hand side of the identity."""
    noise = rng.normal(0, NOISE, P.shape)
    return np.mean((truth(grid)[None, :] + noise - P) ** 2)

print(f"{'deg':>4} {'bias^2':>9} {'variance':>10} {'noise':>8} {'sum':>9} {'measured':>10}")
for deg in [0, 1, 2, 3, 5, 7, 9]:
    P = fit_many(deg)
    b2, v = bias2_and_var(P)
    total, meas = b2 + v + NOISE**2, measured_test_error(P)
    print(f"{deg:4d} {b2:9.4f} {v:10.4f} {NOISE**2:8.4f} {total:9.4f} {meas:10.4f}")
    assert abs(total - meas) / meas < 0.05, f"the identity failed at degree {deg}"
print("\\nPASS -- the three parts add up to the measured error at every degree")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

truth = lambda x: np.sin(2.2*x)*1.3 + 0.35*x
NOISE, N_DATA, N_REP = 0.35, 30, 400
grid = np.linspace(-2.2, 2.2, 60)

def fit_many(deg):
    out = []
    for _ in range(N_REP):
        x = rng.uniform(-2.5, 2.5, N_DATA)
        y = truth(x) + rng.normal(0, NOISE, N_DATA)
        out.append(np.polyval(np.polyfit(x, y, deg), grid))
    return np.array(out)

def bias2_and_var(P):
    mean_pred = P.mean(0)
    return np.mean((mean_pred - truth(grid)) ** 2), np.mean(P.var(0))

def measured_test_error(P):
    noise = rng.normal(0, NOISE, P.shape)
    return np.mean((truth(grid)[None, :] + noise - P) ** 2)

print(f"{'deg':>4} {'bias^2':>9} {'variance':>10} {'noise':>8} {'sum':>9} {'measured':>10}")
for deg in [0, 1, 2, 3, 5, 7, 9]:
    P = fit_many(deg)
    b2, v = bias2_and_var(P)
    total, meas = b2 + v + NOISE**2, measured_test_error(P)
    print(f"{deg:4d} {b2:9.4f} {v:10.4f} {NOISE**2:8.4f} {total:9.4f} {meas:10.4f}")
    assert abs(total - meas) / meas < 0.05, f"the identity failed at degree {deg}"
print("\\nPASS -- the three parts add up to the measured error at every degree")`,
  explain: `Read the two right-hand columns first: they agree at every degree, which is the identity from the
derivation confirmed by measurement rather than algebra.

Then read the two left-hand columns against each other. Bias² falls steadily as the degree rises — more
flexibility lets the average model track the truth more closely, from 0.92 down to almost nothing. Variance
climbs the whole way, from 0.03 to nearly 2.0, because with thirty points a degree-9 curve is pulled somewhere
different by every dataset. Their sum bottoms out at **degree 5**, where the true function stops being
misrepresented and the scatter has not yet taken over. Neither column alone would have told you to stop there.

Note what you had to do to compute any of this: fit 400 separate datasets. In real work you have one. That is
why bias and variance are a way of *thinking* about error rather than a diagnostic you can run — and why
cross-validation, which fakes a handful of extra datasets out of the one you have, is the practical stand-in.`,
},

'ml-regularization': {
  title: 'Write the soft-threshold rule, then reproduce both of the lesson\'s warnings',
  prompt: `Implement lasso yourself and use it to check three claims.

1. **The zeros.** Fill in the soft-threshold update from the derivation:
   $w_j = \\text{sign}(\\rho)\\max(|\\rho|-\\lambda, 0)/a$. Confirm lasso produces *exactly* zero coefficients and
   ridge produces none, and that lasso finds the three features that actually matter.
2. **Standardize or else.** The second block multiplies one feature by 1000 — the same model, in different
   units — and refits. Watch what the penalty does to that feature's coefficient.
3. **Correlated features.** The third block duplicates a real feature and refits several times on resampled
   data. Compare how ridge and lasso each divide the credit.`,
  hint: 'The `max(|rho| - lam, 0)` is the whole trick: when the correlation `rho` is smaller than `lam`, the coefficient is not made small — it is made exactly zero. Divide by `XtX[j, j]`, which is the `a` from the derivation.',
  starter: `import numpy as np
rng = np.random.default_rng(4)

n, d = 60, 20
X = rng.normal(size=(n, d))
X = (X - X.mean(0)) / X.std(0)                 # standardize before regularizing
w_true = np.zeros(d); w_true[[0, 3, 7]] = [2.5, -1.8, 1.2]
y = X @ w_true + rng.normal(0, 0.5, n)

def ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam*np.eye(X.shape[1]), X.T @ y)

def lasso(X, y, lam, iters=400):
    w = np.zeros(X.shape[1])
    XtX, Xty = X.T @ X, X.T @ y
    for _ in range(iters):
        for j in range(len(w)):
            rho = Xty[j] - XtX[j] @ w + XtX[j, j] * w[j]
            # TODO: soft-threshold rho by lam, then divide by XtX[j, j]
            w[j] = 0.0
    return w

print("  lambda | ridge zeros | lasso zeros")
for lam in [1.0, 5.0, 20.0]:
    wr, wl = ridge(X, y, lam), lasso(X, y, lam)
    print(f"  {lam:6.1f} |    {np.sum(np.abs(wr)<1e-8):2d}/{d}    |    {np.sum(np.abs(wl)<1e-8):2d}/{d}")

wl = lasso(X, y, 5.0)
found = sorted(int(i) for i in np.argsort(-np.abs(wl))[:3])
print(f"\\nlasso's three largest coefficients: {found}   (the truth: [0, 3, 7])")
assert found == [0, 3, 7], "lasso should recover the support"
assert np.sum(np.abs(ridge(X, y, 5.0)) < 1e-8) == 0, "ridge should never produce an exact zero"
print("PASS\\n")

# ---------- 2. the same model in different units ----------
Xu = X.copy(); Xu[:, 0] *= 1000                # feature 0 now measured in millimetres
wl_std = lasso(X,  y, 5.0)
wl_raw = lasso(Xu, y, 5.0)
print("feature 0, standardized :", f"{wl_std[0]:9.4f}")
print("feature 0, x1000 units  :", f"{wl_raw[0]:9.6f}",
      f"  (x1000 = {wl_raw[0]*1000:.4f}, so the fit is the same...)")
print("total penalty paid, standardized:", f"{np.abs(wl_std).sum():8.3f}")
print("total penalty paid, x1000 units :", f"{np.abs(wl_raw).sum():8.3f}")
print("nonzero coefficients:", np.sum(np.abs(wl_std) > 1e-8), "vs", np.sum(np.abs(wl_raw) > 1e-8))

# ---------- 3. two interchangeable features ----------
# A and B are noisy copies of the same underlying signal, so nothing in the data
# prefers one over the other. Fit twice, swapping only the COLUMN ORDER.
base = rng.normal(size=n)
A = base + 0.02*rng.normal(size=n)
B = base + 0.02*rng.normal(size=n)
yc = X @ w_true + 1.5*base + rng.normal(0, 0.5, n)

print("\\ntwo interchangeable features, fit twice with the columns swapped:")
for order, first, second in [("A then B", A, B), ("B then A", B, A)]:
    Xc = np.column_stack([X, first, second])
    wr, wl2 = ridge(Xc, yc, 5.0), lasso(Xc, yc, 5.0)
    n1, n2 = order.split(" then ")
    print(f"  ordered {order}:  ridge -> {n1}={wr[-2]:5.2f} {n2}={wr[-1]:5.2f}   "
          f"lasso -> {n1}={wl2[-2]:5.2f} {n2}={wl2[-1]:5.2f}")`,
  solution: `import numpy as np
rng = np.random.default_rng(4)

n, d = 60, 20
X = rng.normal(size=(n, d))
X = (X - X.mean(0)) / X.std(0)
w_true = np.zeros(d); w_true[[0, 3, 7]] = [2.5, -1.8, 1.2]
y = X @ w_true + rng.normal(0, 0.5, n)

def ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam*np.eye(X.shape[1]), X.T @ y)

def lasso(X, y, lam, iters=400):
    w = np.zeros(X.shape[1])
    XtX, Xty = X.T @ X, X.T @ y
    for _ in range(iters):
        for j in range(len(w)):
            rho = Xty[j] - XtX[j] @ w + XtX[j, j] * w[j]
            w[j] = np.sign(rho) * max(abs(rho) - lam, 0.0) / XtX[j, j]
    return w

print("  lambda | ridge zeros | lasso zeros")
for lam in [1.0, 5.0, 20.0]:
    wr, wl = ridge(X, y, lam), lasso(X, y, lam)
    print(f"  {lam:6.1f} |    {np.sum(np.abs(wr)<1e-8):2d}/{d}    |    {np.sum(np.abs(wl)<1e-8):2d}/{d}")

wl = lasso(X, y, 5.0)
found = sorted(int(i) for i in np.argsort(-np.abs(wl))[:3])
print(f"\\nlasso's three largest coefficients: {found}   (the truth: [0, 3, 7])")
assert found == [0, 3, 7]
assert np.sum(np.abs(ridge(X, y, 5.0)) < 1e-8) == 0
print("PASS\\n")

Xu = X.copy(); Xu[:, 0] *= 1000
wl_std = lasso(X,  y, 5.0)
wl_raw = lasso(Xu, y, 5.0)
print("feature 0, standardized :", f"{wl_std[0]:9.4f}")
print("feature 0, x1000 units  :", f"{wl_raw[0]:9.6f}",
      f"  (x1000 = {wl_raw[0]*1000:.4f}, so the fit is the same...)")
print("total penalty paid, standardized:", f"{np.abs(wl_std).sum():8.3f}")
print("total penalty paid, x1000 units :", f"{np.abs(wl_raw).sum():8.3f}")
print("nonzero coefficients:", np.sum(np.abs(wl_std) > 1e-8), "vs", np.sum(np.abs(wl_raw) > 1e-8))

# ---------- 3. two interchangeable features ----------
# A and B are noisy copies of the same underlying signal, so nothing in the data
# prefers one over the other. Fit twice, swapping only the COLUMN ORDER.
base = rng.normal(size=n)
A = base + 0.02*rng.normal(size=n)
B = base + 0.02*rng.normal(size=n)
yc = X @ w_true + 1.5*base + rng.normal(0, 0.5, n)

print("\\ntwo interchangeable features, fit twice with the columns swapped:")
for order, first, second in [("A then B", A, B), ("B then A", B, A)]:
    Xc = np.column_stack([X, first, second])
    wr, wl2 = ridge(Xc, yc, 5.0), lasso(Xc, yc, 5.0)
    n1, n2 = order.split(" then ")
    print(f"  ordered {order}:  ridge -> {n1}={wr[-2]:5.2f} {n2}={wr[-1]:5.2f}   "
          f"lasso -> {n1}={wl2[-2]:5.2f} {n2}={wl2[-1]:5.2f}")`,
  explain: `Part 1: the \`max(·, 0)\` in your update is the entire source of the zeros. Ridge, whose one-coordinate
solution is $\\rho/(a+\\lambda)$ with no such clamp, never produces one no matter how large $\\lambda$ gets.

Part 2 is the standardization warning, quantified. Multiplying a feature by 1000 shrinks its coefficient by
1000 to compensate, so the *model is identical* — same predictions, same residuals. But the penalty sums
coefficients, so that feature's contribution to the penalty drops by a factor of 1000, and it is now essentially
free. The number of surviving nonzero coefficients changes as a result. Your regularization strength was
silently set by your choice of units.

Part 3 is the correlated-features case, and the setup is deliberately symmetric: A and B are noisy copies of
the same signal, so no fact about the data prefers either one. Ridge splits the weight evenly — about 0.74 each —
and gives the same answer whichever order the columns arrive in. Lasso piles the weight onto one of them and
pushes the other toward zero, and **the one it picks changes when you swap the column order**. Column order
carries no information whatsoever, so lasso's answer to "which feature matters?" is being decided by something
that is not evidence.

That is the instability elastic net was designed to fix: adding a little L2 to the L1 makes interchangeable
features share the credit instead of fighting over it.`,
},

'ml-logistic': {
  title: 'Train it, watch ‖w‖ run away, and measure the gradient squared error throws away',
  prompt: `Implement the gradient, then use it to check two claims from the lesson.

1. **Separable data diverges.** Fill in \`train\`. With no penalty, the weights should grow without bound while
   the loss creeps toward zero; with a small L2 penalty they should settle.
2. **The vanishing gradient.** The last block takes a single confidently-wrong example ($p = 0.001$ when the
   answer is 1) and computes what cross-entropy and squared error each tell the model to do about it. The
   lesson said one of them says almost nothing. **Predict the ratio before you run it.**

Both losses are attached to the same sigmoid, so any difference comes entirely from the loss.`,
  hint: 'The mean binary cross-entropy gradient is $\\frac1n X^{\\mathsf T}(\\sigma(Xw) - y)$, plus $\\lambda w$ on the non-intercept weights. For the last block, differentiate each loss with respect to the logit $z$: cross-entropy gives $p - y$, and squared error gives $2(p-y)\\,p(1-p)$.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

# two clusters far enough apart that a line separates them perfectly
n = 200
X = np.vstack([rng.normal([-2, -2], 0.5, (n//2, 2)),
               rng.normal([ 2,  2], 0.5, (n//2, 2))])
y = np.r_[np.zeros(n//2), np.ones(n//2)]
Xb = np.column_stack([np.ones(n), X])

def sigmoid(z): return 1/(1+np.exp(-np.clip(z, -500, 500)))

def train(l2, steps=20000, lr=0.5, verbose=True):
    w = np.zeros(3)
    for t in range(steps):
        p = sigmoid(Xb @ w)
        # TODO: mean cross-entropy gradient, plus l2 on w[1:] only (never the intercept)
        grad = np.zeros(3)
        w = w - lr * grad
        if verbose and t in (100, 1000, 5000, 19999):
            loss = -np.mean(y*np.log(p+1e-12) + (1-y)*np.log(1-p+1e-12))
            print(f"  step {t:6d}  loss {loss:.6f}  |w| {np.linalg.norm(w[1:]):9.3f}")
    return w

print("no regularization, on separable data:")
w_free = train(0.0)
print("\\nwith L2 = 0.01:")
w_reg = train(0.01)

assert np.linalg.norm(w_free[1:]) > 2 * np.linalg.norm(w_reg[1:]), \\
    "the unregularized run should end up with noticeably larger weights"
print("\\nsame decision boundary direction?",
      np.allclose(w_free[1:]/np.linalg.norm(w_free[1:]),
                  w_reg[1:]/np.linalg.norm(w_reg[1:]), atol=0.05))
print("PASS\\n")

# ---------- 2. what each loss says about a confidently wrong example ----------
print("  p       cross-entropy grad    squared-error grad     ratio")
for p in [0.5, 0.1, 0.01, 0.001, 0.0001]:
    y_true = 1.0
    g_ce = p - y_true                       # d/dz of cross-entropy
    g_sq = 2*(p - y_true) * p * (1 - p)     # d/dz of squared error, via the sigmoid
    print(f"  {p:<8.4f}  {g_ce:16.6f}  {g_sq:19.9f}  {abs(g_ce/g_sq):9.0f}x")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

n = 200
X = np.vstack([rng.normal([-2, -2], 0.5, (n//2, 2)),
               rng.normal([ 2,  2], 0.5, (n//2, 2))])
y = np.r_[np.zeros(n//2), np.ones(n//2)]
Xb = np.column_stack([np.ones(n), X])

def sigmoid(z): return 1/(1+np.exp(-np.clip(z, -500, 500)))

def train(l2, steps=20000, lr=0.5, verbose=True):
    w = np.zeros(3)
    for t in range(steps):
        p = sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) / n + l2 * np.r_[0.0, w[1:]]
        w = w - lr * grad
        if verbose and t in (100, 1000, 5000, 19999):
            loss = -np.mean(y*np.log(p+1e-12) + (1-y)*np.log(1-p+1e-12))
            print(f"  step {t:6d}  loss {loss:.6f}  |w| {np.linalg.norm(w[1:]):9.3f}")
    return w

print("no regularization, on separable data:")
w_free = train(0.0)
print("\\nwith L2 = 0.01:")
w_reg = train(0.01)

assert np.linalg.norm(w_free[1:]) > 2 * np.linalg.norm(w_reg[1:])
print("\\nsame decision boundary direction?",
      np.allclose(w_free[1:]/np.linalg.norm(w_free[1:]),
                  w_reg[1:]/np.linalg.norm(w_reg[1:]), atol=0.05))
print("PASS\\n")

print("  p       cross-entropy grad    squared-error grad     ratio")
for p in [0.5, 0.1, 0.01, 0.001, 0.0001]:
    y_true = 1.0
    g_ce = p - y_true
    g_sq = 2*(p - y_true) * p * (1 - p)
    print(f"  {p:<8.4f}  {g_ce:16.6f}  {g_sq:19.9f}  {abs(g_ce/g_sq):9.0f}x")`,
  explain: `Part 1: look at the two \`|w|\` columns. The regularized run is finished by step 1000 and does not
move again. The unregularized one is still climbing at step 20,000, and it always will be — on separable data
there is no optimum to reach, because any correct boundary scaled up is more confident and therefore scores
better. The growth is only logarithmic, so it looks tame in a short run; leave it going and the weights pass
any number you name while the loss crawls toward zero. Notice the last line though: the two boundaries point in nearly the same
*direction*. Regularization did not change what the model believes, only how loudly it says it.

Part 2 is the reason cross-entropy exists. At $p = 0.001$ with the true answer 1 — the model is as wrong as it
can be — cross-entropy reports a gradient of $-0.999$, essentially the largest it ever produces. Squared error
reports $-0.002$, about **500 times smaller**, and it keeps shrinking the more wrong the model gets. The
$p(1-p)$ from the sigmoid is still sitting in the squared-error gradient, and in the tail it is nearly zero.
Cross-entropy's logarithms cancel it exactly. A model trained with squared error would be stuck hardest on the
examples it most needs to fix.`,
},

'ml-trees-ensembles': {
  title: 'Boost a hundred stumps, then watch decorrelation lower the variance floor',
  prompt: `Both halves of the lesson, made measurable.

1. **Boosting.** Write \`fit_stump\`, which picks the split minimising squared error, then boost a hundred of
   them on residuals. Compare learning rates and round counts, and find where boosting starts overfitting.
2. **Decorrelation.** The second block builds ensembles two ways — letting each stump see both features, or
   forcing it to use one feature chosen at random — and measures the variance of the ensemble's prediction as
   the number of trees grows. The lesson said averaging cannot go below a floor of $\\rho\\sigma^2$.
   **Predict which of the two ends up with the lower floor, and whether the individual trees got better or
   worse.**`,
  hint: 'For the stump: given a threshold, the best constant on each side is that side\'s *mean*, and the error is the summed squared deviation from those means. For boosting, each new stump is fit to the current residuals $y - F_{m-1}(x)$ and added with a shrinkage factor.',
  starter: `import numpy as np
rng = np.random.default_rng(14)

# ---------- part 1: boosting ----------
truth = lambda x: np.sin(x*1.6)*1.2 + 0.3*x
x = rng.uniform(-2.5, 2.5, 60)
y = truth(x) + rng.normal(0, 0.25, 60)
xt = np.linspace(-2.5, 2.5, 500); yt = truth(xt)

def fit_stump(x, resid):
    """Return (threshold, left_value, right_value) minimising squared error."""
    best = None
    for thr in np.quantile(x, np.linspace(0.05, 0.95, 30)):
        L, R = x <= thr, x > thr
        if L.sum() < 2 or R.sum() < 2: continue
        # TODO: the best constant on each side, and the resulting squared error
        lm = rm = 0.0
        err = np.inf
        if best is None or err < best[0]:
            best = (err, thr, lm, rm)
    return best[1], best[2], best[3]

def boost(M, lr):
    base = y.mean()
    pred = np.full_like(y, base)
    stumps = []
    for _ in range(M):
        thr, lm, rm = fit_stump(x, y - pred)      # fit the CURRENT residuals
        stumps.append((thr, lm, rm))
        pred = pred + lr * np.where(x <= thr, lm, rm)
    return lambda q: base + sum(lr*np.where(q <= t, l, r) for t, l, r in stumps)

print("  rounds   lr      train      test")
for M, lr in [(10, 0.3), (100, 0.3), (400, 0.3), (100, 1.0), (400, 1.0)]:
    F = boost(M, lr)
    print(f"  {M:6d}  {lr:.1f}   {np.mean((F(x)-y)**2):.4f}   {np.mean((F(xt)-yt)**2):.4f}")
assert np.mean((boost(400, 1.0)(x)-y)**2) < np.mean((boost(10, 0.3)(x)-y)**2), \\
    "more rounds must always fit the training data better"
print("\\n(noise floor on train is 0.0625; anything below it is fitting noise)\\n")

# ---------- part 2: why forests hide features from their own trees ----------
n, R, MMAX = 100, 150, 32
f2 = lambda X: np.sin(1.5*X[:, 0]) + 0.8*X[:, 1]
probe = np.array([0.7, -0.4])          # one fixed input to watch

def stump_at(Xb, yb, feats, q):
    best = None
    for j in feats:
        for thr in np.quantile(Xb[:, j], np.linspace(.15, .85, 10)):
            L = Xb[:, j] <= thr
            if L.sum() < 4 or (~L).sum() < 4: continue
            lm, rm = yb[L].mean(), yb[~L].mean()
            e = ((yb[L]-lm)**2).sum() + ((yb[~L]-rm)**2).sum()
            if best is None or e < best[0]: best = (e, j, thr, lm, rm)
    _, j, thr, lm, rm = best
    return lm if q[j] <= thr else rm

for label, n_feat in [("each stump sees both features", 2),
                      ("each stump sees ONE random feature", 1)]:
    P = np.zeros((R, MMAX))
    for r in range(R):
        X = rng.uniform(-2, 2, (n, 2)); yy = f2(X) + rng.normal(0, .3, n)
        for m in range(MMAX):
            idx = rng.integers(0, n, n)                     # bootstrap resample
            feats = [0, 1] if n_feat == 2 else [int(rng.integers(0, 2))]
            P[r, m] = stump_at(X[idx], yy[idx], feats, probe)
    C = np.corrcoef(P.T)
    rho = (C.sum() - np.trace(C)) / (MMAX*(MMAX-1))
    s2 = P[:, 0].var()
    print(f"{label}")
    print(f"   one tree's variance sigma^2 = {s2:.4f},  average correlation rho = {rho:.3f}")
    print(f"   the formula's floor, rho*sigma^2 = {rho*s2:.4f}")
    for M in [1, 2, 4, 8, 16, 32]:
        print(f"     M={M:3d} trees ->  ensemble variance {P[:, :M].mean(1).var():.4f}")
    print()`,
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
        pred = pred + lr * np.where(x <= thr, lm, rm)
    return lambda q: base + sum(lr*np.where(q <= t, l, r) for t, l, r in stumps)

print("  rounds   lr      train      test")
for M, lr in [(10, 0.3), (100, 0.3), (400, 0.3), (100, 1.0), (400, 1.0)]:
    F = boost(M, lr)
    print(f"  {M:6d}  {lr:.1f}   {np.mean((F(x)-y)**2):.4f}   {np.mean((F(xt)-yt)**2):.4f}")
assert np.mean((boost(400, 1.0)(x)-y)**2) < np.mean((boost(10, 0.3)(x)-y)**2)
print("\\n(noise floor on train is 0.0625; anything below it is fitting noise)\\n")

n, R, MMAX = 100, 150, 32
f2 = lambda X: np.sin(1.5*X[:, 0]) + 0.8*X[:, 1]
probe = np.array([0.7, -0.4])

def stump_at(Xb, yb, feats, q):
    best = None
    for j in feats:
        for thr in np.quantile(Xb[:, j], np.linspace(.15, .85, 10)):
            L = Xb[:, j] <= thr
            if L.sum() < 4 or (~L).sum() < 4: continue
            lm, rm = yb[L].mean(), yb[~L].mean()
            e = ((yb[L]-lm)**2).sum() + ((yb[~L]-rm)**2).sum()
            if best is None or e < best[0]: best = (e, j, thr, lm, rm)
    _, j, thr, lm, rm = best
    return lm if q[j] <= thr else rm

for label, n_feat in [("each stump sees both features", 2),
                      ("each stump sees ONE random feature", 1)]:
    P = np.zeros((R, MMAX))
    for r in range(R):
        X = rng.uniform(-2, 2, (n, 2)); yy = f2(X) + rng.normal(0, .3, n)
        for m in range(MMAX):
            idx = rng.integers(0, n, n)
            feats = [0, 1] if n_feat == 2 else [int(rng.integers(0, 2))]
            P[r, m] = stump_at(X[idx], yy[idx], feats, probe)
    C = np.corrcoef(P.T)
    rho = (C.sum() - np.trace(C)) / (MMAX*(MMAX-1))
    s2 = P[:, 0].var()
    print(f"{label}")
    print(f"   one tree's variance sigma^2 = {s2:.4f},  average correlation rho = {rho:.3f}")
    print(f"   the formula's floor, rho*sigma^2 = {rho*s2:.4f}")
    for M in [1, 2, 4, 8, 16, 32]:
        print(f"     M={M:3d} trees ->  ensemble variance {P[:, :M].mean(1).var():.4f}")
    print()`,
  explain: `Part 1: training error falls monotonically with more rounds, exactly as "each tree fits what is still
wrong" predicts — and it sails straight through the noise floor of $0.25^2 = 0.0625$, which is the point at
which it has stopped learning the function and started learning the noise. Test error tells the real story:
it improves, bottoms out, and then climbs. This is the asymmetry from the lesson — **boosting will overfit if
you let it**, so it needs early stopping, while a random forest does not.

Part 2 is the decorrelation argument, measured. Read three numbers from each block:

- **One tree alone gets worse when you hide a feature** — $\\sigma^2$ rises from about 0.41 to about 0.52. You
  crippled the base learner, and it shows.
- **The correlation collapses** — $\\rho$ falls from about 0.21 to about 0.05, because trees forced onto
  different features cannot make the same mistakes.
- **The ensemble ends up better anyway** — at 32 trees the variance is about 0.099 with both features and about
  0.045 with one. Compare each against its own predicted floor $\\rho\\sigma^2$: the first has almost reached
  0.085 and has stopped improving, while the second still has room down to 0.028.

That is the whole design of a random forest in three lines. Weakening each individual tree in a way that makes
their errors independent buys more than it costs, because the part of the variance that averaging can remove is
$(1-\\rho)\\sigma^2/M$ and the part it cannot is $\\rho\\sigma^2$. Attacking $\\rho$ attacks the term that adding
trees can never touch.`,
},

'ml-svm-knn': {
  title: 'Find the support vectors, then catch the kernel trick in the act',
  prompt: `1. **Support vectors.** Train a linear SVM by gradient descent on the hinge loss. Count the support
   vectors, then delete the point *furthest* from the boundary, refit, and confirm nothing moved.
2. **The kernel trick, verified.** The polynomial kernel $k(\\mathbf{x},\\mathbf{z}) = (\\mathbf{x}\\cdot\\mathbf{z}+1)^2$
   claims to be a dot product in a space of quadratic features. Write down that feature map $\\phi$ explicitly
   for two-dimensional inputs and check that $\\phi(\\mathbf{x})\\cdot\\phi(\\mathbf{z})$ really does equal
   $k(\\mathbf{x},\\mathbf{z})$, to the last decimal.
3. Then the last block asks the same question of the RBF kernel, for which no finite $\\phi$ exists — and checks
   that its similarity table is still a legitimate one.`,
  hint: 'For part 1, the hinge term contributes a gradient only for points with $y f(\\mathbf{x}) < 1$. For part 2, expand $(x_1z_1 + x_2z_2 + 1)^2$ by hand and read off which functions of $\\mathbf{x}$ appear — remember the cross terms come with a factor of 2, so the feature needs a $\\sqrt{2}$.',
  starter: `import numpy as np
rng = np.random.default_rng(2)

# ---------- part 1: support vectors ----------
n = 80
X = np.vstack([rng.normal([-1.5, -1], 0.7, (n//2, 2)),
               rng.normal([ 1.5,  1], 0.7, (n//2, 2))])
y = np.r_[-np.ones(n//2), np.ones(n//2)]

def train_svm(X, y, C=1.0, steps=4000):
    w, b = np.zeros(2), 0.0
    for t in range(steps):
        lr = 0.05 / (1 + t*0.002)
        margin = y * (X @ w + b)
        # TODO: gradient of  0.5*||w||^2 + C * mean(max(0, 1 - margin))
        # only the points with margin < 1 contribute to the hinge part
        gw = np.zeros(2); gb = 0.0
        w = w - lr * gw; b = b - lr * gb
    return w, b

w, b = train_svm(X, y)
margins = y * (X @ w + b)
sv = margins <= 1.0 + 1e-2
print(f"margin width 2/|w| = {2/np.linalg.norm(w):.4f}")
print(f"support vectors: {sv.sum()} / {n}")

keep = np.ones(n, bool); keep[np.argmax(margins)] = False   # drop the safest point
w2, b2 = train_svm(X[keep], y[keep])
print(f"w before deleting it: {w.round(4)}")
print(f"w after:              {w2.round(4)}")
assert np.allclose(w, w2, atol=0.05), "deleting a non-support point must not move the solution"
print("PASS\\n")

# ---------- part 2: the kernel IS a dot product ----------
def poly_kernel(a, z):
    return (a @ z + 1.0) ** 2

def phi(a):
    """The explicit feature map for the degree-2 polynomial kernel in 2-D."""
    x1, x2 = a
    # TODO: return the vector of features whose dot product reproduces poly_kernel
    return np.zeros(6)

for _ in range(5):
    a, z = rng.normal(size=2), rng.normal(size=2)
    direct  = poly_kernel(a, z)
    lifted  = phi(a) @ phi(z)
    print(f"  k(x,z) = {direct:9.5f}    phi(x).phi(z) = {lifted:9.5f}")
    assert abs(direct - lifted) < 1e-9, "the feature map does not match the kernel"
print("PASS -- one 2-D dot product and a square did the work of a 6-D dot product\\n")

# ---------- part 3: RBF, where no finite phi exists ----------
def rbf(A, B, gamma=0.5):
    return np.exp(-gamma * ((A[:, None, :] - B[None, :, :])**2).sum(-1))

pts = rng.normal(size=(40, 2))
K = rbf(pts, pts)
eig = np.linalg.eigvalsh(K)
print(f"RBF similarity table: {K.shape[0]}x{K.shape[1]}")
print(f"  diagonal entries (a point with itself): {K[0,0]:.3f}")
print(f"  smallest eigenvalue: {eig.min():.2e}   (must not be negative)")
print(f"  number of eigenvalues above 1e-8: {(eig > 1e-8).sum()} of {len(eig)}")
print("Every eigenvalue is >= 0, so this table IS a valid table of dot products")
print("in SOME space -- one nobody can write down, and nobody needs to.")`,
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
        w = w - lr * gw; b = b - lr * gb
    return w, b

w, b = train_svm(X, y)
margins = y * (X @ w + b)
sv = margins <= 1.0 + 1e-2
print(f"margin width 2/|w| = {2/np.linalg.norm(w):.4f}")
print(f"support vectors: {sv.sum()} / {n}")

keep = np.ones(n, bool); keep[np.argmax(margins)] = False
w2, b2 = train_svm(X[keep], y[keep])
print(f"w before deleting it: {w.round(4)}")
print(f"w after:              {w2.round(4)}")
assert np.allclose(w, w2, atol=0.05)
print("PASS\\n")

def poly_kernel(a, z):
    return (a @ z + 1.0) ** 2

def phi(a):
    x1, x2 = a
    s = np.sqrt(2.0)
    return np.array([x1**2, x2**2, s*x1*x2, s*x1, s*x2, 1.0])

for _ in range(5):
    a, z = rng.normal(size=2), rng.normal(size=2)
    direct  = poly_kernel(a, z)
    lifted  = phi(a) @ phi(z)
    print(f"  k(x,z) = {direct:9.5f}    phi(x).phi(z) = {lifted:9.5f}")
    assert abs(direct - lifted) < 1e-9
print("PASS -- one 2-D dot product and a square did the work of a 6-D dot product\\n")

def rbf(A, B, gamma=0.5):
    return np.exp(-gamma * ((A[:, None, :] - B[None, :, :])**2).sum(-1))

pts = rng.normal(size=(40, 2))
K = rbf(pts, pts)
eig = np.linalg.eigvalsh(K)
print(f"RBF similarity table: {K.shape[0]}x{K.shape[1]}")
print(f"  diagonal entries (a point with itself): {K[0,0]:.3f}")
print(f"  smallest eigenvalue: {eig.min():.2e}   (must not be negative)")
print(f"  number of eigenvalues above 1e-8: {(eig > 1e-8).sum()} of {len(eig)}")
print("Every eigenvalue is >= 0, so this table IS a valid table of dot products")
print("in SOME space -- one nobody can write down, and nobody needs to.")`,
  explain: `Part 1: removing the point sitting furthest inside its own correct side changes the answer by less than
a rounding error. Its hinge loss was already exactly zero, so it contributed exactly zero gradient, so it was
never influencing anything. Only the support vectors decide the boundary — and that sparsity is what makes
kernel prediction affordable, since you only evaluate the kernel against those.

Part 2 is the kernel trick caught in the act. The correct feature map is
$\\phi(\\mathbf{x}) = (x_1^2,\\ x_2^2,\\ \\sqrt2 x_1x_2,\\ \\sqrt2 x_1,\\ \\sqrt2 x_2,\\ 1)$ — six dimensions of
quadratic features, with the $\\sqrt2$ factors there because the cross terms appear twice when you expand the
square. The two columns agree exactly. So computing one 2-D dot product and squaring it gave you, for free, the
answer to a question about a 6-D space. Scale the same idea to degree 10 in 1000 dimensions and $\\phi$ has
billions of entries, while the kernel is still one dot product and one power.

Part 3 is why the trick generalises. For the RBF kernel there is no finite $\\phi$ to write out at all, and it
does not matter: every eigenvalue of its similarity table is non-negative, which is precisely the condition for
that table to be a table of dot products in *some* space. The algorithm only ever needed the table.`,
},

'ml-unsupervised': {
  title: 'k-means: how much the initialization matters, and the shape it cannot see',
  prompt: `1. **Initialization.** Implement Lloyd's algorithm — assign, then update, repeat. Run it from 20 random
   starts on the same data and look at the spread of final inertias. Then swap in k-means++ seeding and compare
   the spread.
2. **The shape assumption.** The last block hands k-means two long parallel bands. It has the correct $k$, it
   converges cleanly, and it reports a perfectly respectable inertia. Check whether it found the bands.`,
  hint: 'Assignment: `((X[:,None] - C[None])**2).sum(-1).argmin(1)`. Update: move each centre to the mean of the points assigned to it, leaving it alone if it has no points.',
  starter: `import numpy as np
rng = np.random.default_rng(2)

X = np.vstack([rng.normal([0,0], 0.4, (120,2)),
               rng.normal([3,3], 0.4, (120,2)),
               rng.normal([0,3], 0.4, (120,2)),
               rng.normal([3,0], 0.4, (120,2))])

def lloyd(X, C, iters=100):
    C = C.copy()
    for _ in range(iters):
        # TODO: assign every point to its nearest centre,
        #       then move every centre to the mean of its points
        pass
    d = ((X[:,None] - C[None])**2).sum(-1)
    a = d.argmin(1)
    return C, a, ((X - C[a])**2).sum()

def init_random(X, k, r):
    return X[r.choice(len(X), k, replace=False)]

def init_pp(X, k, r):
    """k-means++: each new centre is chosen with probability proportional to
       its squared distance from the nearest centre chosen so far."""
    C = [X[r.integers(len(X))]]
    for _ in range(k-1):
        d2 = ((X[:,None] - np.array(C)[None])**2).sum(-1).min(1)
        C.append(X[r.choice(len(X), p=d2/d2.sum())])
    return np.array(C)

results = {}
for name, init in [("random", init_random), ("k-means++", init_pp)]:
    scores = [lloyd(X, init(X, 4, np.random.default_rng(s)))[2] for s in range(40)]
    results[name] = scores
    best = min(scores)
    hits = sum(1 for v in scores if v < best*1.01)
    print(f"{name:10s} over 40 starts:  best {best:7.2f}   worst {max(scores):7.2f}   "
          f"reached the best answer {hits}/40 times")

assert min(results["random"]) < 200, "Lloyd's algorithm does not look implemented"
assert max(results["random"]) > 1.5 * min(results["random"]), \\
    "some random starts should land in a clearly worse local optimum"
print("PASS\\n")

# ---------- the shape k-means cannot see ----------
band = rng.uniform(-4, 4, 200)
B = np.vstack([np.c_[band, np.full(200,  0.6) + rng.normal(0, .12, 200)],
               np.c_[band, np.full(200, -0.6) + rng.normal(0, .12, 200)]])
true_label = np.r_[np.zeros(200), np.ones(200)]     # which band each point is really in

# do it properly: 20 restarts, keep the lowest inertia, exactly as part 1 advised
best = None
for s in range(20):
    C, a, inertia = lloyd(B, init_pp(B, 2, np.random.default_rng(s)))
    if best is None or inertia < best[2]:
        best = (C, a, inertia)
C, a, inertia = best
agree = max((a == true_label).mean(), (a != true_label).mean())

print("two long horizontal bands, k=2, best of 20 restarts:")
print(f"  centres found:      {np.round(C, 2).tolist()}")
print(f"  inertia:            {inertia:.1f}   (converged, no complaints)")
print(f"  agreement with the real bands: {agree:.1%}")
print("  50% agreement means the split has nothing to do with the true grouping.")`,
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

def init_random(X, k, r):
    return X[r.choice(len(X), k, replace=False)]

def init_pp(X, k, r):
    C = [X[r.integers(len(X))]]
    for _ in range(k-1):
        d2 = ((X[:,None] - np.array(C)[None])**2).sum(-1).min(1)
        C.append(X[r.choice(len(X), p=d2/d2.sum())])
    return np.array(C)

results = {}
for name, init in [("random", init_random), ("k-means++", init_pp)]:
    scores = [lloyd(X, init(X, 4, np.random.default_rng(s)))[2] for s in range(40)]
    results[name] = scores
    best = min(scores)
    hits = sum(1 for v in scores if v < best*1.01)
    print(f"{name:10s} over 40 starts:  best {best:7.2f}   worst {max(scores):7.2f}   "
          f"reached the best answer {hits}/40 times")

assert min(results["random"]) < 200
assert max(results["random"]) > 1.5 * min(results["random"])
print("PASS\\n")

band = rng.uniform(-4, 4, 200)
B = np.vstack([np.c_[band, np.full(200,  0.6) + rng.normal(0, .12, 200)],
               np.c_[band, np.full(200, -0.6) + rng.normal(0, .12, 200)]])
true_label = np.r_[np.zeros(200), np.ones(200)]

# do it properly: 20 restarts, keep the lowest inertia, exactly as part 1 advised
best = None
for s in range(20):
    C, a, inertia = lloyd(B, init_pp(B, 2, np.random.default_rng(s)))
    if best is None or inertia < best[2]:
        best = (C, a, inertia)
C, a, inertia = best
agree = max((a == true_label).mean(), (a != true_label).mean())

print("two long horizontal bands, k=2, best of 20 restarts:")
print(f"  centres found:      {np.round(C, 2).tolist()}")
print(f"  inertia:            {inertia:.1f}   (converged, no complaints)")
print(f"  agreement with the real bands: {agree:.1%}")
print("  50% agreement means the split has nothing to do with the true grouping.")`,
  explain: `Part 1: identical data, identical algorithm, forty different answers. The best is around 156 and the worst is
over 650 — more than four times worse — because Lloyd's algorithm only ever walks downhill from wherever you
dropped it, and a start that puts two centres inside the same blob leaves an entire cluster with nobody to
represent it. k-means++, which spreads the initial centres apart, lands on the best answer noticeably more
often. Neither guarantees it. The practical rule follows directly: run it several times and keep the lowest
inertia, which is what every library does by default.

Part 2 is the more important lesson, and there is a sting in it. The two bands are obviously two groups to any
human eye. k-means was given the correct $k$, seeded well, and restarted twenty times exactly as part 1 said to.

Look at the centres it found: both sit at $y \\approx 0$, one at $x \\approx -2$ and one at $x \\approx +2$. It cut
the picture **left from right** instead of top from bottom, and agreement with the real grouping is 50% — a coin
flip.

It is worth checking that this is not a bug. A left–right split gives inertia around 700; the top–bottom split
you wanted gives around 2300. k-means found the genuinely better answer *to the question it was asked*. And
notice the sting: following part 1's advice made things worse. A single lazy run had a decent chance of landing
in the top–bottom local optimum by accident; twenty careful restarts reliably found the "right" one.

Nothing in the output announces any of this. The inertia is an ordinary number and there is no warning flag. The
assumption lives in the objective — minimising squared distance to a centre makes every cluster implicitly a
round blob, and a long thin band is not one. Plot the result, always.`,
},

'ml-evaluation': {
  title: 'How many test examples you need, and how to make leakage disappear',
  prompt: `1. **Confidence intervals.** Implement the interval from the lesson,
   $\\hat p \\pm 1.96\\sqrt{\\hat p(1-\\hat p)/n}$, and use it to answer a concrete question: with 20 test
   examples, can you tell an 85% model from a 75% one? Then find roughly how many you would need to.
2. **Leakage, and the fix.** The last block builds a dataset of **pure noise** — the features have no
   relationship to the target at all — so the only honest score is 50%. Write \`select_and_score\` twice: once
   choosing features using the whole dataset, and once choosing them using the training rows only. One version
   reports a respectable accuracy on nothing.`,
  hint: 'For part 2, the difference is a single line: the wrong version computes correlations on `X` and `y`, the right one on `X[train]` and `y[train]`. Everything downstream is identical.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

# ---------- part 1: how sure are you, really? ----------
def ci(successes, n, z=1.96):
    """Return (lo, hi): the 95% interval for a proportion."""
    # TODO: p_hat plus or minus z * sqrt(p_hat*(1-p_hat)/n)
    return (0.0, 1.0)

print(f"{'n':>6} {'observed':>10} {'95% interval':>22} {'width':>10}")
for n in [10, 20, 50, 200, 1000, 5000]:
    s = round(0.85 * n)
    lo, hi = ci(s, n)
    print(f"{n:6d} {s/n:10.1%} {f'[{lo:.1%}, {hi:.1%}]':>22} {(hi-lo)*100:8.1f} pts")

lo85, hi85 = ci(17, 20)
lo75, hi75 = ci(15, 20)
print(f"\\n85% on n=20: [{lo85:.3f}, {hi85:.3f}]")
print(f"75% on n=20: [{lo75:.3f}, {hi75:.3f}]")
print("-> ", "cannot distinguish" if hi75 > lo85 else "distinguishable")

for n in [50, 100, 200, 400, 800, 1600]:
    if ci(round(.75*n), n)[1] < ci(round(.85*n), n)[0]:
        print(f"\\nthe intervals first separate at about n = {n}")
        break
assert ci(85, 100)[0] > 0.77 and ci(85, 100)[1] < 0.93, "check the interval formula"
print("PASS\\n")

# ---------- part 2: a score built entirely out of leakage ----------
n, d = 200, 500
X = rng.normal(size=(n, d))            # pure noise
y = rng.integers(0, 2, n)              # unrelated labels. Honest accuracy: 50%.
train, test = np.arange(150), np.arange(150, 200)

def select_and_score(use_all_rows):
    rows = np.arange(n) if use_all_rows else train
    # TODO: correlate each of the d features with y, using only \`rows\`;
    #       keep the 10 strongest, fit least squares on \`train\`, score on \`test\`.
    corr = np.zeros(d)
    top = np.argsort(-np.abs(corr))[:10]
    w = np.linalg.lstsq(X[np.ix_(train, top)], y[train]*2.0-1, rcond=None)[0]
    return ((X[np.ix_(test, top)] @ w > 0) == y[test]).mean()

leaky   = select_and_score(use_all_rows=True)
honest  = select_and_score(use_all_rows=False)
print(f"features chosen using ALL rows   -> test accuracy {leaky:.3f}")
print(f"features chosen using TRAIN only -> test accuracy {honest:.3f}")
print(f"the truth                        -> 0.500, because there is no signal")
lo, hi = ci(round(honest*50), 50)
print(f"\\nand with only 50 test rows, the honest score's own interval is "
      f"[{lo:.2f}, {hi:.2f}] -- wide enough to swallow both numbers.")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def ci(successes, n, z=1.96):
    p = successes / n
    half = z * np.sqrt(p * (1 - p) / n)
    return (max(0.0, p - half), min(1.0, p + half))

print(f"{'n':>6} {'observed':>10} {'95% interval':>22} {'width':>10}")
for n in [10, 20, 50, 200, 1000, 5000]:
    s = round(0.85 * n)
    lo, hi = ci(s, n)
    print(f"{n:6d} {s/n:10.1%} {f'[{lo:.1%}, {hi:.1%}]':>22} {(hi-lo)*100:8.1f} pts")

lo85, hi85 = ci(17, 20)
lo75, hi75 = ci(15, 20)
print(f"\\n85% on n=20: [{lo85:.3f}, {hi85:.3f}]")
print(f"75% on n=20: [{lo75:.3f}, {hi75:.3f}]")
print("-> ", "cannot distinguish" if hi75 > lo85 else "distinguishable")

for n in [50, 100, 200, 400, 800, 1600]:
    if ci(round(.75*n), n)[1] < ci(round(.85*n), n)[0]:
        print(f"\\nthe intervals first separate at about n = {n}")
        break
assert ci(85, 100)[0] > 0.77 and ci(85, 100)[1] < 0.93
print("PASS\\n")

n, d = 200, 500
X = rng.normal(size=(n, d))
y = rng.integers(0, 2, n)
train, test = np.arange(150), np.arange(150, 200)

def select_and_score(use_all_rows):
    rows = np.arange(n) if use_all_rows else train
    corr = np.array([np.corrcoef(X[rows, j], y[rows])[0, 1] for j in range(d)])
    top = np.argsort(-np.abs(corr))[:10]
    w = np.linalg.lstsq(X[np.ix_(train, top)], y[train]*2.0-1, rcond=None)[0]
    return ((X[np.ix_(test, top)] @ w > 0) == y[test]).mean()

leaky   = select_and_score(use_all_rows=True)
honest  = select_and_score(use_all_rows=False)
print(f"features chosen using ALL rows   -> test accuracy {leaky:.3f}")
print(f"features chosen using TRAIN only -> test accuracy {honest:.3f}")
print(f"the truth                        -> 0.500, because there is no signal")
lo, hi = ci(round(honest*50), 50)
print(f"\\nand with only 50 test rows, the honest score's own interval is "
      f"[{lo:.2f}, {hi:.2f}] -- wide enough to swallow both numbers.")`,
  explain: `Part 1: the interval on 20 examples is roughly ±16 points, so 85% and 75% are completely
indistinguishable — their intervals overlap almost entirely. The gap only becomes real somewhere around
$n = 400$. Look at the width column: it shrinks like $1/\\sqrt{n}$, so buying half the uncertainty
costs four times the data. Next time you see two models reported a point or two apart on a small benchmark,
this is the number to have in mind.

Part 2: the features are noise and the labels are unrelated, so 50% is the ceiling — there is nothing to learn.
Yet choosing the ten best-correlated features using all 200 rows produces a test accuracy around **76%** —
on data containing nothing. Doing the identical selection on the training rows alone lands back around chance. No
model ever trained on the test rows. The leak was in the *selection*: with 500 noise features and 200 rows, some
of them correlate with the labels by luck, and asking which ones over the whole dataset means the winners are
partly chosen for matching the test rows' noise.

This is the most common leak in practice, and it applies to anything fitted before the split — scalers,
imputers, PCA, feature selection, target encoding. The rule that covers all of them: **the split comes first,
and everything that learns from data happens after it.**`,
},

'ml-generative-discriminative': {
  title: 'Make the assumption fail, then ask each model to do something it was not built for',
  prompt: `1. **What the assumption costs.** Implement \`naive_bayes\` — fit a mean and standard deviation per
   feature per class, then score a new point by which class's Gaussians find it less surprising. The script runs
   it twice: once on data whose features are correlated, and once on data whose features really are independent.
   **Predict which table will show naive Bayes falling behind, and why.**
2. **What the generative model got for free.** The same fitted naive Bayes is then asked to do two jobs nobody
   trained it for: **invent** new examples of a class, and **fill in a missing measurement** given the rest of
   an input. Write those two functions. Then note what the logistic regression can contribute to either — the
   answer is nothing, and the reason is worth stating out loud.`,
  hint: 'The Gaussian log-density of $x$ under mean $m$ and spread $s$, dropping constants, is $-\\frac{(x-m)^2}{2s^2} - \\log s$. To generate, run that backwards: pick a class, then draw each feature from its own Gaussian. To impute, the best guess for a missing feature is simply that class\'s mean for it.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

d = 60
delta = rng.normal(0, 1, d); delta = 2.2 * delta / np.linalg.norm(delta)
MIX = rng.normal(size=(d, d))/np.sqrt(d) + 0.5*np.eye(d)   # correlates the features,
                                                           # so NB's assumption is false
def make(n, mix=MIX):
    y = rng.integers(0, 2, n)
    return (rng.normal(size=(n, d)) + y[:, None]*delta) @ mix, y

def fit_nb(Xtr, ytr):
    """Return the per-class means and standard deviations."""
    return dict(m0=Xtr[ytr==0].mean(0), s0=Xtr[ytr==0].std(0) + 1e-6,
                m1=Xtr[ytr==1].mean(0), s1=Xtr[ytr==1].std(0) + 1e-6)

def naive_bayes(Xtr, ytr, Xte):
    p = fit_nb(Xtr, ytr)
    # TODO: total Gaussian log-density of each test point under each class,
    #       summed over features; predict whichever class scores higher
    return np.zeros(len(Xte), dtype=int)

def logistic(Xtr, ytr, Xte, steps=3000):
    Xb = np.c_[np.ones(len(Xtr)), Xtr]; w = np.zeros(d+1)
    for _ in range(steps):
        p = 1/(1+np.exp(-np.clip(Xb @ w, -500, 500)))
        w = w - 0.5 * (Xb.T @ (p - ytr)/len(ytr) + 0.01*np.r_[0, w[1:]])
    return (np.c_[np.ones(len(Xte)), Xte] @ w > 0).astype(int)

Xte, yte = make(4000)
print("features correlated, so naive Bayes's independence assumption is FALSE:")
print(f"{'n':>7} {'naive Bayes':>13} {'logistic':>10}")
for n in [30, 120, 1000, 3000, 10000]:
    Xtr, ytr = make(n)
    nb = (naive_bayes(Xtr, ytr, Xte) == yte).mean()
    lr = (logistic(Xtr, ytr, Xte) == yte).mean()
    print(f"{n:7d} {nb:13.3f} {lr:10.3f}")

print("\\nthe control: same code, but with genuinely independent features")
Xte_i, yte_i = make(4000, np.eye(d))
print(f"{'n':>7} {'naive Bayes':>13} {'logistic':>10}")
for n in [30, 120, 1000, 3000, 10000]:
    Xtr, ytr = make(n, np.eye(d))
    nb = (naive_bayes(Xtr, ytr, Xte_i) == yte_i).mean()
    lr = (logistic(Xtr, ytr, Xte_i) == yte_i).mean()
    print(f"{n:7d} {nb:13.3f} {lr:10.3f}")

Xtr, ytr = make(4000)
assert (naive_bayes(Xtr, ytr, Xte) == yte).mean() > 0.7, "naive Bayes should classify well here"
print("PASS\\n")

# ---------- 2. two jobs the classifier cannot even attempt ----------
p = fit_nb(Xtr, ytr)

def generate(cls, k, r):
    """Invent k brand-new examples of class cls."""
    # TODO: draw each feature from that class's own Gaussian
    return np.zeros((k, d))

def impute(x_partial, missing_j, cls):
    """Feature missing_j was not recorded. Best guess, given the class?"""
    # TODO
    return 0.0

fake = generate(1, 200, np.random.default_rng(7))
real = Xte[yte == 1]
print("invented examples of class 1 vs real ones:")
print(f"  mean of first 5 features, invented: {np.round(fake.mean(0)[:5], 2)}")
print(f"  mean of first 5 features, real    : {np.round(real.mean(0)[:5], 2)}")
print(f"  a classifier trained on real data calls {100*(naive_bayes(Xtr, ytr, fake) == 1).mean():.0f}%"
      " of the invented ones class 1")

j = int(np.argmax(np.abs(p["m1"] - p["m0"])))     # a feature the classes disagree about
overall = float(Xtr[:, j].mean())
errs  = [abs(impute(Xte[i], j, yte[i]) - Xte[i, j]) for i in range(500)]
blind = [abs(overall               - Xte[i, j]) for i in range(500)]
print(f"\\nfeature {j} has gone missing. Average error when guessing it:")
print(f"  from the generative model, knowing the class: {np.mean(errs):.3f}")
print(f"  from the overall average, ignoring the class: {np.mean(blind):.3f}")
print("The logistic regression has no opinion about either question. It only knows the boundary.")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

d = 60
delta = rng.normal(0, 1, d); delta = 2.2 * delta / np.linalg.norm(delta)
MIX = rng.normal(size=(d, d))/np.sqrt(d) + 0.5*np.eye(d)   # correlates the features,
                                                           # so NB's assumption is false
def make(n, mix=MIX):
    y = rng.integers(0, 2, n)
    return (rng.normal(size=(n, d)) + y[:, None]*delta) @ mix, y

def fit_nb(Xtr, ytr):
    return dict(m0=Xtr[ytr==0].mean(0), s0=Xtr[ytr==0].std(0) + 1e-6,
                m1=Xtr[ytr==1].mean(0), s1=Xtr[ytr==1].std(0) + 1e-6)

def naive_bayes(Xtr, ytr, Xte):
    p = fit_nb(Xtr, ytr)
    ll0 = -(((Xte-p["m0"])**2)/(2*p["s0"]**2) + np.log(p["s0"])).sum(1)
    ll1 = -(((Xte-p["m1"])**2)/(2*p["s1"]**2) + np.log(p["s1"])).sum(1)
    return (ll1 > ll0).astype(int)

def logistic(Xtr, ytr, Xte, steps=3000):
    Xb = np.c_[np.ones(len(Xtr)), Xtr]; w = np.zeros(d+1)
    for _ in range(steps):
        p = 1/(1+np.exp(-np.clip(Xb @ w, -500, 500)))
        w = w - 0.5 * (Xb.T @ (p - ytr)/len(ytr) + 0.01*np.r_[0, w[1:]])
    return (np.c_[np.ones(len(Xte)), Xte] @ w > 0).astype(int)

Xte, yte = make(4000)
print("features correlated, so naive Bayes's independence assumption is FALSE:")
print(f"{'n':>7} {'naive Bayes':>13} {'logistic':>10}")
for n in [30, 120, 1000, 3000, 10000]:
    Xtr, ytr = make(n)
    nb = (naive_bayes(Xtr, ytr, Xte) == yte).mean()
    lr = (logistic(Xtr, ytr, Xte) == yte).mean()
    print(f"{n:7d} {nb:13.3f} {lr:10.3f}")

print("\\nthe control: same code, but with genuinely independent features")
Xte_i, yte_i = make(4000, np.eye(d))
print(f"{'n':>7} {'naive Bayes':>13} {'logistic':>10}")
for n in [30, 120, 1000, 3000, 10000]:
    Xtr, ytr = make(n, np.eye(d))
    nb = (naive_bayes(Xtr, ytr, Xte_i) == yte_i).mean()
    lr = (logistic(Xtr, ytr, Xte_i) == yte_i).mean()
    print(f"{n:7d} {nb:13.3f} {lr:10.3f}")

Xtr, ytr = make(4000)
assert (naive_bayes(Xtr, ytr, Xte) == yte).mean() > 0.7
print("PASS\\n")

p = fit_nb(Xtr, ytr)

def generate(cls, k, r):
    m, s = (p["m1"], p["s1"]) if cls == 1 else (p["m0"], p["s0"])
    return r.normal(m, s, size=(k, d))

def impute(x_partial, missing_j, cls):
    return (p["m1"] if cls == 1 else p["m0"])[missing_j]

fake = generate(1, 200, np.random.default_rng(7))
real = Xte[yte == 1]
print("invented examples of class 1 vs real ones:")
print(f"  mean of first 5 features, invented: {np.round(fake.mean(0)[:5], 2)}")
print(f"  mean of first 5 features, real    : {np.round(real.mean(0)[:5], 2)}")
print(f"  a classifier trained on real data calls {100*(naive_bayes(Xtr, ytr, fake) == 1).mean():.0f}%"
      " of the invented ones class 1")

j = int(np.argmax(np.abs(p["m1"] - p["m0"])))     # a feature the classes disagree about
overall = float(Xtr[:, j].mean())
errs  = [abs(impute(Xte[i], j, yte[i]) - Xte[i, j]) for i in range(500)]
blind = [abs(overall               - Xte[i, j]) for i in range(500)]
print(f"\\nfeature {j} has gone missing. Average error when guessing it:")
print(f"  from the generative model, knowing the class: {np.mean(errs):.3f}")
print(f"  from the overall average, ignoring the class: {np.mean(blind):.3f}")
print("The logistic regression has no opinion about either question. It only knows the boundary.")`,
  explain: `Part 1: compare the two tables, which differ only in whether the features were mixed together. In
the control, where the features genuinely are independent, naive Bayes keeps pace with logistic regression the
whole way and even leads in the middle — its assumption is *true* there, so believing it costs nothing. In the
correlated table, naive Bayes flattens out around 0.80 by $n=3000$ and never improves again, while logistic
regression carries on past 0.86.

That plateau is what "higher asymptotic error" means. More data cannot help a model whose form is wrong; it
simply estimates the wrong model more precisely. The other half of the classical tradeoff — that the generative
model reaches its ceiling with far less data — is real but needs many more features than examples to show
clearly, which is why naive Bayes made its name on text, where a vocabulary of tens of thousands meets a few
hundred documents.

Part 2 is the modern point, and it is the reason this lesson exists. The same fitted model — no retraining, no
new objective — invents new class-1 examples that a classifier accepts as class 1, and fills in a measurement
that was never provided. It can do these things because it learned *what class 1 looks like*, and that knowledge
was never specific to the question "class 0 or class 1?".

The logistic regression cannot attempt either task, and not because it is weaker. It genuinely does not contain
the information: it learned a boundary, which says where classes differ and nothing about what either one is.
Scale that same distinction up — replace "Gaussian per feature" with a transformer over text — and you get the
reason a language model can be asked to classify, summarise, translate, or rank without ever having been trained
to, while a sentiment classifier can only ever do sentiment.`,
},

};

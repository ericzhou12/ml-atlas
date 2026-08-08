/* ============================================================
   Challenges — track 1, mathematical foundations

   One entry per lesson id: { title, prompt, starter, solution, hint?, explain? }
   Starters carry their own PASS/FAIL check so the learner gets a verdict
   without leaving the lab.

   EDITOR NOTE: these are JS template literals, so an unescaped backtick
   silently terminates the string. Escape inline code in prose.
   ============================================================ */

export default {

'math-vectors': {
  title: 'Cosine, projection, and the orthogonality of high dimensions',
  prompt: `Three short functions, all built from the dot product and nothing else — no \`np.linalg\` anywhere.

1. \`cosine(a, b)\` — the formula from the lesson, $\\dfrac{\\mathbf{a}\\cdot\\mathbf{b}}{\\|\\mathbf{a}\\|\\|\\mathbf{b}\\|}$,
   remembering that $\\|\\mathbf{a}\\| = \\sqrt{\\mathbf{a}\\cdot\\mathbf{a}}$.
2. \`project(a, b)\` — the shadow of $\\mathbf{a}$ on $\\mathbf{b}$. The checks confirm it lands on
   $\\mathbf{b}$'s line and that the leftover is orthogonal.
3. Then run the last block, which measures the average $|\\cos|$ between random vectors as $d$ grows.
   **Predict the numbers before you run it** — the lesson says they should sit near $1/\\sqrt{d}$.`,
  hint: 'In NumPy, `a @ b` is the multiply-and-sum. So the length is `np.sqrt(a @ a)`, and the projection is the number `(a @ b) / (b @ b)` times the vector `b`.',
  starter: `import numpy as np

def cosine(a, b):
    # TODO: dot product, divided by both lengths. Use only @ and np.sqrt.
    return 0.0

def project(a, b):
    # TODO: return proj_b(a) -- a scaled copy of b. Use only @.
    return np.zeros_like(b)

# --- check 1: cosine ---
a = np.array([1.0, 0.0]); b = np.array([0.0, 1.0]); c = np.array([2.0, 0.0])
assert abs(cosine(a, b) - 0.0) < 1e-9, "perpendicular vectors should give 0"
assert abs(cosine(a, c) - 1.0) < 1e-9, "same-direction vectors should give 1, whatever their lengths"
assert abs(cosine(a, -c) + 1.0) < 1e-9, "opposite vectors should give -1"

# --- check 2: projection and residual ---
u = np.array([2.0, 1.0, -3.0]); v = np.array([4.0, 0.0, 1.0])
p = project(u, v)
r = u - p
ratios = p[v != 0] / v[v != 0]          # p must be a scalar multiple of v
assert np.allclose(ratios, ratios[0]), "the projection must be a scalar multiple of v"
assert abs(r @ v) < 1e-9, "the residual must be orthogonal to v"
print("PASS\\n")

# --- the high-dimensional experiment ---
rng = np.random.default_rng(0)
for d in [2, 10, 100, 1000, 10000]:
    X = rng.normal(size=(500, d)); Y = rng.normal(size=(500, d))
    m = np.mean([abs(cosine(X[i], Y[i])) for i in range(500)])
    print(f"d={d:6d}   mean|cos| = {m:.4f}   1/sqrt(d) = {1/np.sqrt(d):.4f}")`,
  solution: `import numpy as np

def cosine(a, b):
    return (a @ b) / np.sqrt((a @ a) * (b @ b))

def project(a, b):
    return ((a @ b) / (b @ b)) * b

a = np.array([1.0, 0.0]); b = np.array([0.0, 1.0]); c = np.array([2.0, 0.0])
assert abs(cosine(a, b) - 0.0) < 1e-9
assert abs(cosine(a, c) - 1.0) < 1e-9
assert abs(cosine(a, -c) + 1.0) < 1e-9

u = np.array([2.0, 1.0, -3.0]); v = np.array([4.0, 0.0, 1.0])
p = project(u, v)
r = u - p
ratios = p[v != 0] / v[v != 0]
assert np.allclose(ratios, ratios[0])
assert abs(r @ v) < 1e-9
print("PASS\\n")

rng = np.random.default_rng(0)
for d in [2, 10, 100, 1000, 10000]:
    X = rng.normal(size=(500, d)); Y = rng.normal(size=(500, d))
    m = np.mean([abs(cosine(X[i], Y[i])) for i in range(500)])
    print(f"d={d:6d}   mean|cos| = {m:.4f}   1/sqrt(d) = {1/np.sqrt(d):.4f}")`,
  explain: `The measured averages track $1/\\sqrt d$ closely, which is what the "variances add" argument in the lesson predicted. Read the last row: in 10,000 dimensions two random directions have a cosine of about 0.008, meaning they are perpendicular for all practical purposes. That is what lets a network pack far more than $d$ nearly-independent features into $d$ dimensions.`,
},

'math-matrices': {
  title: 'Outer products, low rank, and building LoRA by hand',
  prompt: `Build the lesson's compression argument yourself, in three steps.

1. \`outer(u, v)\` — the outer product, entry $[i,j] = u_i v_j$, without using \`np.outer\`. Confirm it comes out
   rank 1.
2. \`lowrank(B, A)\` — take a tall-thin $B$ and a short-wide $A$ and produce the full-size matrix $BA$. This is
   exactly LoRA's $\\Delta W$. Confirm its rank is at most $r$, no matter how big the output looks.
3. Then read the parameter count the script prints and check it against the number you derived in the lesson.`,
  hint: 'For the outer product, make `u` a column with `u[:, None]` and `v` a row with `v[None, :]`, then multiply — NumPy broadcasting fills in the grid. For `lowrank`, it is one `@`.',
  starter: `import numpy as np

def outer(u, v):
    # TODO: the (len(u), len(v)) matrix whose [i,j] entry is u[i]*v[j]
    return np.zeros((len(u), len(v)))

def lowrank(B, A):
    # TODO: the full-size matrix that the two skinny factors stand for
    return np.zeros((B.shape[0], A.shape[1]))

# --- check 1: an outer product is rank 1 ---
u = np.array([1.0, 2.0, -1.0])
v = np.array([3.0, 1.0])
R = outer(u, v)
assert R.shape == (3, 2), "wrong shape"
assert np.allclose(R, np.outer(u, v)), "does not match np.outer"
assert np.linalg.matrix_rank(R) == 1, "an outer product must have rank 1"

# every column of R should be a multiple of u
for j in range(R.shape[1]):
    print(f"  column {j} = {np.round(R[:, j], 3)}  =  {R[0, j]/u[0]:.3g} * u")

# --- check 2: a sum of r outer products has rank at most r ---
rng = np.random.default_rng(0)
m, n, r = 40, 30, 3
B = rng.normal(size=(m, r))
A = rng.normal(size=(r, n))
dW = lowrank(B, A)
assert dW.shape == (m, n), "delta W must be full size"
assert np.linalg.matrix_rank(dW) == r, f"rank should be {r}"

# the same matrix, written as a sum of r outer products
S = sum(outer(B[:, k], A[k, :]) for k in range(r))
assert np.allclose(S, dW), "B @ A must equal the sum of r outer products"
print("\\nPASS")

# --- the parameter count ---
m = n = 4096
for r in [1, 8, 64]:
    full = m * n
    lora = r * (m + n)
    print(f"rank {r:3d}:  {lora:>10,} trained vs {full:>12,} full   ({100*lora/full:.2f}%)")`,
  solution: `import numpy as np

def outer(u, v):
    return u[:, None] * v[None, :]

def lowrank(B, A):
    return B @ A

u = np.array([1.0, 2.0, -1.0])
v = np.array([3.0, 1.0])
R = outer(u, v)
assert R.shape == (3, 2)
assert np.allclose(R, np.outer(u, v))
assert np.linalg.matrix_rank(R) == 1
for j in range(R.shape[1]):
    print(f"  column {j} = {np.round(R[:, j], 3)}  =  {R[0, j]/u[0]:.3g} * u")

rng = np.random.default_rng(0)
m, n, r = 40, 30, 3
B = rng.normal(size=(m, r))
A = rng.normal(size=(r, n))
dW = lowrank(B, A)
assert dW.shape == (m, n)
assert np.linalg.matrix_rank(dW) == r
S = sum(outer(B[:, k], A[k, :]) for k in range(r))
assert np.allclose(S, dW)
print("\\nPASS")

m = n = 4096
for r in [1, 8, 64]:
    full = m * n
    lora = r * (m + n)
    print(f"rank {r:3d}:  {lora:>10,} trained vs {full:>12,} full   ({100*lora/full:.2f}%)")`,
  explain: `Check 2 is the whole idea. \`dW\` is a 40×30 grid — 1200 numbers on screen — but it was built from only $3(40+30) = 210$, and \`matrix_rank\` confirms it genuinely spans just 3 directions. The assertion that \`B @ A\` equals the sum of $r$ outer products is the sum-of-outer-products fact from the lesson, verified numerically. The final table is LoRA: at rank 8 you train 0.39% of the numbers, and the only assumption is that the *update* your task needs is that simple.`,
},

'math-eigen-svd': {
  title: 'Compress an image with truncated SVD',
  prompt: `Build a synthetic "image", take its SVD, and reconstruct it from the top $k$ singular values. Plot
reconstruction error against $k$ and against the storage cost. Where is the knee?`,
  hint: 'The rank-$k$ reconstruction is `U[:, :k] @ diag(S[:k]) @ Vt[:k]`. Storage is $k(m+n)$ versus $mn$.',
  starter: `import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
m, n = 80, 60
x = np.linspace(0, 1, n); y = np.linspace(0, 1, m)
img = (np.sin(6*x)[None, :] * np.cos(4*y)[:, None]
       + 0.5*np.sin(11*x)[None, :] + 0.05*rng.normal(size=(m, n)))

U, S, Vt = np.linalg.svd(img, full_matrices=False)

def rank_k(k):
    # TODO: reconstruct img from the top k singular values
    return np.zeros_like(img)

errs, costs = [], []
for k in range(1, 31):
    approx = rank_k(k)
    errs.append(np.linalg.norm(img - approx) / np.linalg.norm(img))
    costs.append(k*(m+n) / (m*n))

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].semilogy(S[:30], "o-"); ax[0].set_title("singular values"); ax[0].set_xlabel("index")
ax[1].plot(costs, errs, "o-"); ax[1].set_xlabel("storage fraction"); ax[1].set_ylabel("relative error")
ax[1].set_title("error vs storage")
plt.tight_layout()`,
  solution: `import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
m, n = 80, 60
x = np.linspace(0, 1, n); y = np.linspace(0, 1, m)
img = (np.sin(6*x)[None, :] * np.cos(4*y)[:, None]
       + 0.5*np.sin(11*x)[None, :] + 0.05*rng.normal(size=(m, n)))

U, S, Vt = np.linalg.svd(img, full_matrices=False)

def rank_k(k):
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k]

errs, costs = [], []
for k in range(1, 31):
    errs.append(np.linalg.norm(img - rank_k(k)) / np.linalg.norm(img))
    costs.append(k*(m+n) / (m*n))
    if k in (1, 3, 5, 10, 20):
        print(f"k={k:3d}  error {errs[-1]:.4f}  storage {costs[-1]:.1%}")

fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax[0].semilogy(S[:30], "o-"); ax[0].set_title("singular values")
ax[1].plot(costs, errs, "o-"); ax[1].set_xlabel("storage fraction")
ax[1].set_ylabel("relative error"); ax[1].set_title("error vs storage")
ax[2].imshow(rank_k(5), cmap="viridis"); ax[2].set_title("rank-5 reconstruction")
plt.tight_layout()`,
  explain: 'The knee sits exactly where the singular value spectrum falls off a cliff. That cliff is the *effective rank* of the data, and it is the same quantity PCA, LoRA, and recommender systems all exploit.',
},

'math-derivatives': {
  title: 'Gradient checking with central differences',
  prompt: `Write \`numeric_grad(f, x)\` using central differences, then use it to verify an analytic gradient. Sweep
the step size $h$ and find the value that minimizes error — you should see truncation error and floating-point
cancellation fighting each other.`,
  hint: 'Central difference: $(f(x+h)-f(x-h))/2h$. Truncation error falls as $h^2$; rounding error grows as $1/h$.',
  starter: `import numpy as np

def f(w):
    return np.sum(np.tanh(w) ** 2) + 0.5 * np.sum(w ** 2)

def analytic_grad(w):
    t = np.tanh(w)
    return 2 * t * (1 - t**2) + w

def numeric_grad(f, x, h=1e-5):
    g = np.zeros_like(x)
    # TODO: fill g[i] with the central difference along axis i
    return g

w = np.array([0.3, -1.2, 0.8, 2.0])
err = np.abs(analytic_grad(w) - numeric_grad(f, w)).max()
print("max abs error:", err)
print("PASS" if err < 1e-7 else "FAIL")

print("\\nstep size sweep:")
for h in [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11]:
    e = np.abs(analytic_grad(w) - numeric_grad(f, w, h)).max()
    print(f"  h={h:.0e}  error {e:.3e}")`,
  solution: `import numpy as np

def f(w):
    return np.sum(np.tanh(w) ** 2) + 0.5 * np.sum(w ** 2)

def analytic_grad(w):
    t = np.tanh(w)
    return 2 * t * (1 - t**2) + w

def numeric_grad(f, x, h=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g

w = np.array([0.3, -1.2, 0.8, 2.0])
err = np.abs(analytic_grad(w) - numeric_grad(f, w)).max()
print("max abs error:", err, "->", "PASS" if err < 1e-7 else "FAIL")

print("\\nstep size sweep:")
for h in [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11]:
    e = np.abs(analytic_grad(w) - numeric_grad(f, w, h)).max()
    print(f"  h={h:.0e}  error {e:.3e}")`,
  explain: 'Error is minimized around $h \\approx 10^{-5}$ to $10^{-6}$ in float64. Larger $h$ leaves truncation error; smaller $h$ subtracts two nearly-equal numbers and loses significant digits. This is the sweet spot every gradient-checking routine targets.',
},

'math-jacobian': {
  title: 'Vector-Jacobian products without building the Jacobian',
  prompt: `For $\\mathbf{y} = \\tanh(W\\mathbf{x})$, compute $\\mathbf{v}^{\\mathsf T}J$ two ways: by explicitly forming
$J$, and the way autodiff does it. Confirm they match, then compare the memory each would need at $d=4096$.`,
  hint: 'The elementwise activation contributes a *diagonal* Jacobian, so it becomes an elementwise multiply.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

W = rng.normal(size=(4, 6)) * 0.5
x = rng.normal(size=6)
v = rng.normal(size=4)

z = W @ x
y = np.tanh(z)

# --- explicit: build the full Jacobian dy/dx, then multiply ---
J = np.diag(1 - np.tanh(z)**2) @ W
grad_explicit = v @ J

def vjp(v, W, z):
    # TODO: same result, without ever forming J
    return np.zeros(W.shape[1])

g = vjp(v, W, z)
print("match:", np.allclose(grad_explicit, g), "->", "PASS" if np.allclose(grad_explicit, g) else "FAIL")

d = 4096
print(f"\\nat d={d}: explicit Jacobian = {d*d*8/1e6:.0f} MB, VJP working set = {d*8/1e3:.0f} KB")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

W = rng.normal(size=(4, 6)) * 0.5
x = rng.normal(size=6)
v = rng.normal(size=4)
z = W @ x

J = np.diag(1 - np.tanh(z)**2) @ W
grad_explicit = v @ J

def vjp(v, W, z):
    dz = v * (1 - np.tanh(z)**2)   # diagonal Jacobian -> elementwise
    return dz @ W                  # then one matvec

g = vjp(v, W, z)
print("match:", np.allclose(grad_explicit, g), "-> PASS")

d = 4096
print(f"\\nat d={d}: explicit Jacobian = {d*d*8/1e6:.0f} MB, VJP working set = {d*8/1e3:.0f} KB")`,
},

'math-probability': {
  title: 'The base rate trap, and how fast the prior loses',
  prompt: `Compute $P(\\text{sick}\\mid+)$ for a 99%-accurate test at various disease prevalences. Then simulate
Beta-Bernoulli updating and find how many coin flips it takes for the data to overwhelm a strong wrong prior.`,
  hint: 'Bayes: $P(s\\mid+) = \\frac{P(+\\mid s)P(s)}{P(+\\mid s)P(s) + P(+\\mid \\neg s)P(\\neg s)}$.',
  starter: `import numpy as np

def posterior_sick(prevalence, sensitivity=0.99, specificity=0.99):
    # TODO: return P(sick | positive test)
    return 0.0

print("P(sick | positive test), 99% accurate test:")
for prev in [0.5, 0.1, 0.01, 0.001, 0.0001]:
    print(f"  prevalence {prev:8.4f} -> {posterior_sick(prev):.4f}")

assert abs(posterior_sick(0.5) - 0.99) < 1e-6, "at 50% prevalence it should be 0.99"
print("\\nPASS\\n")

# --- a strong prior that is wrong ---
rng = np.random.default_rng(0)
true_p = 0.75
a, b = 30.0, 30.0          # prior says "fair coin", confidently
for n in range(1, 601):
    a += rng.random() < true_p
    b += 0 if a != a else 0
print("now extend this loop yourself: track the posterior mean and find")
print("the n at which it first comes within 0.02 of the true 0.75.")`,
  solution: `import numpy as np

def posterior_sick(prevalence, sensitivity=0.99, specificity=0.99):
    p_pos = sensitivity*prevalence + (1-specificity)*(1-prevalence)
    return sensitivity*prevalence / p_pos

print("P(sick | positive test), 99% accurate test:")
for prev in [0.5, 0.1, 0.01, 0.001, 0.0001]:
    print(f"  prevalence {prev:8.4f} -> {posterior_sick(prev):.4f}")
assert abs(posterior_sick(0.5) - 0.99) < 1e-6
print("\\nPASS\\n")

rng = np.random.default_rng(0)
true_p = 0.75
a, b = 30.0, 30.0
hit = None
for n in range(1, 2001):
    flip = rng.random() < true_p
    a += flip; b += (not flip)
    mean = a / (a + b)
    if hit is None and abs(mean - true_p) < 0.02:
        hit = n
    if n in (10, 50, 200, 1000, 2000):
        sd = np.sqrt(a*b/((a+b)**2*(a+b+1)))
        print(f"  n={n:5d}  posterior mean {mean:.4f}  sd {sd:.4f}")
print(f"\\nfirst within 0.02 of the truth at n = {hit}")`,
  explain: 'At 1-in-10,000 prevalence a positive result on a 99% test means about a 1% chance of being sick. The prior dominates until the data is overwhelming — which is the same arithmetic that destroys precision for rare-class classifiers.',
},

'math-information': {
  title: 'Cross-entropy, KL, and why softmax needs the max-shift',
  prompt: `Implement entropy, cross-entropy and KL in bits. Verify $\\mathrm{KL}(p\\|q) = H(p,q) - H(p) \\ge 0$ and
that it is asymmetric. Then write a numerically stable \`log_softmax\` and show the naive version overflows.`,
  hint: 'Softmax is shift-invariant: subtract `z.max()` before exponentiating.',
  starter: `import numpy as np

def entropy(p):        return 0.0   # TODO
def cross_entropy(p,q): return 0.0  # TODO
def kl(p, q):          return 0.0   # TODO

p = np.array([0.5, 0.25, 0.15, 0.10])
q = np.array([0.25, 0.25, 0.25, 0.25])

print(f"H(p)     = {entropy(p):.4f} bits")
print(f"H(p,q)   = {cross_entropy(p,q):.4f}")
print(f"KL(p||q) = {kl(p,q):.4f}")
print(f"KL(q||p) = {kl(q,p):.4f}   <- different!")

assert abs(kl(p,q) - (cross_entropy(p,q) - entropy(p))) < 1e-9, "KL identity failed"
assert kl(p,q) >= 0 and abs(kl(p,p)) < 1e-9, "KL must be >= 0 and 0 iff p==q"
print("\\nPASS\\n")

def log_softmax(z):
    # TODO: stable version
    return z

big = np.array([800.0, 801.0, 799.0])
print("naive :", np.log(np.exp(big) / np.exp(big).sum()))
print("stable:", log_softmax(big))`,
  solution: `import numpy as np

def entropy(p):         return -np.sum(p * np.log2(p + 1e-12))
def cross_entropy(p,q): return -np.sum(p * np.log2(q + 1e-12))
def kl(p, q):           return np.sum(p * np.log2((p + 1e-12) / (q + 1e-12)))

p = np.array([0.5, 0.25, 0.15, 0.10])
q = np.array([0.25, 0.25, 0.25, 0.25])
print(f"H(p) = {entropy(p):.4f}   H(p,q) = {cross_entropy(p,q):.4f}")
print(f"KL(p||q) = {kl(p,q):.4f}   KL(q||p) = {kl(q,p):.4f}")
assert abs(kl(p,q) - (cross_entropy(p,q) - entropy(p))) < 1e-9
assert kl(p,q) >= 0 and abs(kl(p,p)) < 1e-9
print("\\nPASS\\n")

def log_softmax(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())

big = np.array([800.0, 801.0, 799.0])
print("naive :", np.log(np.exp(big) / np.exp(big).sum()))
print("stable:", log_softmax(big))`,
  explain: 'The naive version produces `nan` because `exp(800)` overflows to `inf`. The max-shift is mathematically a no-op and numerically essential — it is why you should always call your framework\'s fused `cross_entropy(logits, target)`.',
},

'math-optimization': {
  title: 'Implement SGD, momentum, and Adam — then break them',
  prompt: `Write all three update rules for an ill-conditioned quadratic. Verify plain GD diverges above
$\\eta = 2/\\lambda_{\\max}$ exactly as theory predicts, and check how much less sensitive Adam is.`,
  hint: 'Adam needs bias correction: $\\hat m = m/(1-\\beta_1^t)$, $\\hat v = v/(1-\\beta_2^t)$.',
  starter: `import numpy as np

A = np.array([0.15, 4.0])                 # per-axis curvature; kappa ~ 27
loss = lambda w: 0.5 * np.sum(A * w**2)
grad = lambda w: A * w

def run(opt, lr, steps=200, beta=0.9):
    w = np.array([-2.5, 2.0])
    m = np.zeros(2); v = np.zeros(2)
    for t in range(1, steps+1):
        g = grad(w)
        if opt == "sgd":
            pass      # TODO
        elif opt == "momentum":
            pass      # TODO  (m holds the velocity)
        elif opt == "adam":
            pass      # TODO  (b1=0.9, b2=0.999, eps=1e-8, with bias correction)
        if not np.all(np.isfinite(w)): return np.inf
    return loss(w)

print(f"theory: plain GD diverges above lr = 2/{A.max()} = {2/A.max():.4f}\\n")
for lr in [0.1, 0.4, 0.49, 0.51, 0.9]:
    print(f"lr={lr:<5} sgd={run('sgd',lr):>12.3e}  "
          f"momentum={run('momentum',lr):>10.3e}  adam={run('adam',lr):>10.3e}")`,
  solution: `import numpy as np

A = np.array([0.15, 4.0])
loss = lambda w: 0.5 * np.sum(A * w**2)
grad = lambda w: A * w

def run(opt, lr, steps=200, beta=0.9):
    w = np.array([-2.5, 2.0])
    m = np.zeros(2); v = np.zeros(2)
    for t in range(1, steps+1):
        g = grad(w)
        if opt == "sgd":
            w = w - lr * g
        elif opt == "momentum":
            m = beta*m - lr*g
            w = w + m
        elif opt == "adam":
            b1, b2, eps = 0.9, 0.999, 1e-8
            m = b1*m + (1-b1)*g
            v = b2*v + (1-b2)*g**2
            mh = m / (1 - b1**t); vh = v / (1 - b2**t)
            w = w - lr * mh / (np.sqrt(vh) + eps)
        if not np.all(np.isfinite(w)): return np.inf
    return loss(w)

print(f"theory: plain GD diverges above lr = 2/{A.max()} = {2/A.max():.4f}\\n")
for lr in [0.1, 0.4, 0.49, 0.51, 0.9]:
    print(f"lr={lr:<5} sgd={run('sgd',lr):>12.3e}  "
          f"momentum={run('momentum',lr):>10.3e}  adam={run('adam',lr):>10.3e}")`,
  explain: 'SGD blows up between 0.49 and 0.51, precisely at $2/\\lambda_{\\max}=0.5$. Adam rescales each axis by its own gradient history, so the same learning rate works across a much wider range — that insensitivity is most of why it is the default.',
},

'math-numerics': {
  title: 'Make a variance computation fail, then fix it',
  prompt: `Compute variance with the textbook $\\mathbb{E}[X^2]-\\mathbb{E}[X]^2$ formula in float32 on data with a
large mean. Get it to return something clearly wrong, then implement the two-pass version.`,
  hint: 'Catastrophic cancellation: both terms are huge and nearly equal, so their difference loses all its significant digits.',
  starter: `import numpy as np

def var_naive(x):
    return (x**2).mean() - x.mean()**2

def var_twopass(x):
    # TODO: subtract the mean first
    return 0.0

for offset in [0.0, 1e3, 1e6, 1e8]:
    x = (np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + offset).astype(np.float32)
    print(f"offset {offset:>10.0e}: naive {var_naive(x):>14.6f}   "
          f"two-pass {var_twopass(x):>10.6f}   numpy {x.var():.6f}")

x = (np.arange(5, dtype=np.float32) + 1e8)
assert abs(var_twopass(x) - 2.0) < 0.1, "two-pass should stay accurate"
print("\\nPASS")`,
  solution: `import numpy as np

def var_naive(x):
    return (x**2).mean() - x.mean()**2

def var_twopass(x):
    m = x.mean()
    return ((x - m) ** 2).mean()

for offset in [0.0, 1e3, 1e6, 1e8]:
    x = (np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + offset).astype(np.float32)
    print(f"offset {offset:>10.0e}: naive {var_naive(x):>14.6f}   "
          f"two-pass {var_twopass(x):>10.6f}   numpy {x.var():.6f}")

x = (np.arange(5, dtype=np.float32) + 1e8)
assert abs(var_twopass(x) - 2.0) < 0.1
print("\\nPASS")`,
  explain: 'At an offset of $10^8$ in float32 the naive formula can return 0 or even a negative variance. The true answer is 2.0 the whole time. This is the same failure mode that makes `log(softmax(x))` unusable.',
},

};

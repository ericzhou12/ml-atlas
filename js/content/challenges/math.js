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
  title: 'Truncate an SVD, find the cliff, and test Eckart–Young',
  prompt: `You have a matrix that is secretly simple, hidden under noise. Recover that fact from its spectrum.

1. \`rank_k(k)\` — rebuild the matrix from only its first $k$ singular values. This is the truncation from the
   lesson, written out.
2. Read the printed spectrum and the error table. **Before scrolling to the errors, pick the $k$ you would
   keep** from the singular values alone, then see whether the error table agrees.
3. The last block tests Eckart–Young by hand: it builds many random rank-3 matrices and checks that none of
   them beats the truncated SVD. Nothing you write is needed there — just read the result.`,
  hint: 'Keeping the first $k$ terms of $\\sum_i \\sigma_i \\mathbf{u}_i\\mathbf{v}_i^{\\mathsf T}$ is exactly `U[:, :k] @ np.diag(S[:k]) @ Vt[:k]` — take the first $k$ columns of $U$, the first $k$ singular values, and the first $k$ rows of $Vt$.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

# A 60x40 matrix built from only 3 directions, then buried in noise.
m, n, true_rank = 60, 40, 3
A = rng.normal(size=(m, true_rank)) @ rng.normal(size=(true_rank, n))
A = A + 0.4 * rng.normal(size=(m, n))

U, S, Vt = np.linalg.svd(A, full_matrices=False)

def rank_k(k):
    # TODO: rebuild A from its first k singular values only
    return np.zeros_like(A)

# --- checks ---
assert np.allclose(rank_k(len(S)), A), "keeping every term must give back A exactly"
assert np.linalg.matrix_rank(rank_k(2)) == 2, "rank_k(2) must have rank 2"
print("PASS\\n")

# --- the spectrum: where is the cliff? ---
print("singular values (bar length is proportional to size):")
for i, s in enumerate(S[:10]):
    print(f"  sigma_{i+1:<2d} {s:7.3f}  " + "#" * int(40 * s / S[0]))

# --- error and storage as you keep more ---
print("\\n  k   relative error   storage")
for k in [1, 2, 3, 4, 6, 10, 20]:
    err = np.linalg.norm(A - rank_k(k)) / np.linalg.norm(A)
    print(f"  {k:2d}      {err:.4f}        {k*(m+n)/(m*n):6.1%}")

# --- Eckart-Young, tested rather than trusted ---
best = np.linalg.norm(A - rank_k(3))
best_random = min(
    np.linalg.norm(A - Q @ (Q.T @ A))                  # best rank-3 fit inside a random subspace
    for Q in (np.linalg.qr(rng.normal(size=(m, 3)))[0] for _ in range(200))
)
print(f"\\ntruncated SVD, rank 3:        {best:.4f}")
print(f"best of 200 random rank-3:    {best_random:.4f}")
print("The SVD wins. Eckart-Young says nothing can beat it." if best <= best_random
      else "Something is wrong with rank_k.")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

m, n, true_rank = 60, 40, 3
A = rng.normal(size=(m, true_rank)) @ rng.normal(size=(true_rank, n))
A = A + 0.4 * rng.normal(size=(m, n))

U, S, Vt = np.linalg.svd(A, full_matrices=False)

def rank_k(k):
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k]

assert np.allclose(rank_k(len(S)), A)
assert np.linalg.matrix_rank(rank_k(2)) == 2
print("PASS\\n")

print("singular values (bar length is proportional to size):")
for i, s in enumerate(S[:10]):
    print(f"  sigma_{i+1:<2d} {s:7.3f}  " + "#" * int(40 * s / S[0]))

print("\\n  k   relative error   storage")
for k in [1, 2, 3, 4, 6, 10, 20]:
    err = np.linalg.norm(A - rank_k(k)) / np.linalg.norm(A)
    print(f"  {k:2d}      {err:.4f}        {k*(m+n)/(m*n):6.1%}")

best = np.linalg.norm(A - rank_k(3))
best_random = min(
    np.linalg.norm(A - Q @ (Q.T @ A))
    for Q in (np.linalg.qr(rng.normal(size=(m, 3)))[0] for _ in range(200))
)
print(f"\\ntruncated SVD, rank 3:        {best:.4f}")
print(f"best of 200 random rank-3:    {best_random:.4f}")
print("The SVD wins. Eckart-Young says nothing can beat it." if best <= best_random
      else "Something is wrong with rank_k.")`,
  explain: `The bars show the cliff: three tall ones, then a long flat tail of noise. The error table confirms
it — going from $k=3$ to $k=20$ costs seven times the storage and buys very little, because everything past the
third direction was noise to begin with. The last block is Eckart–Young made concrete: 200 honest attempts to
find a better rank-3 approximation, all of them worse than simply truncating the SVD. That is what the
"provably best" in the lesson buys you — permission to stop searching.`,
},
'math-derivatives': {
  title: 'Derive a gradient, check it by nudging, and test the steepest-descent claim',
  prompt: `Two things the lesson claimed, both testable in code.

1. **The chain rule.** Write \`analytic_grad(w)\` for $f(\\mathbf{w}) = \\sum_i \\tanh(w_i)^2$ by peeling the
   composition apart, then write \`numeric_grad\` with central differences and confirm the two agree. Useful
   facts: the derivative of $u^2$ is $2u$, and the derivative of $\\tanh(u)$ is $1 - \\tanh(u)^2$.
2. **Steepest ascent.** The last block takes 2000 random unit directions and measures how fast $f$ rises along
   each one, then compares the best of them against the gradient direction. Predict the outcome before running.

The step-size sweep in between is there to show that a smaller $h$ is not always a better $h$ — watch what the
error does at the bottom.`,
  hint: 'Central difference along axis $i$: nudge only that coordinate, $(f(x + h\\mathbf{e}_i) - f(x - h\\mathbf{e}_i)) / 2h$. For the analytic gradient, $\\partial f/\\partial w_i$ only involves $w_i$, because the sum has no cross terms.',
  starter: `import numpy as np

def f(w):
    return np.sum(np.tanh(w) ** 2)

def analytic_grad(w):
    # TODO: chain rule. Outer function is u^2, inner is tanh(w_i).
    return np.zeros_like(w)

def numeric_grad(f, x, h=1e-5):
    g = np.zeros_like(x)
    # TODO: central difference, one coordinate at a time
    return g

w = np.array([0.3, -1.2, 0.8, 2.0])
err = np.abs(analytic_grad(w) - numeric_grad(f, w)).max()
print("max disagreement:", f"{err:.2e}", "->", "PASS" if err < 1e-7 else "FAIL")

# --- how the step size h affects the numeric estimate ---
print("\\nstep size sweep:")
for h in [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11]:
    e = np.abs(analytic_grad(w) - numeric_grad(f, w, h)).max()
    print(f"  h={h:.0e}   error {e:.3e}")

# --- is the gradient really the steepest direction? ---
g = analytic_grad(w)
eps = 1e-6
rate_along_grad = (f(w + eps * g / np.linalg.norm(g)) - f(w)) / eps

rng = np.random.default_rng(0)
best_random = -np.inf
for _ in range(2000):
    u = rng.normal(size=w.shape)
    u = u / np.linalg.norm(u)                  # a random direction of length 1
    best_random = max(best_random, (f(w + eps * u) - f(w)) / eps)

print(f"\\nrise per unit step along the gradient: {rate_along_grad:.6f}")
print(f"best of 2000 random directions:       {best_random:.6f}")
print(f"predicted maximum, ||grad f||:        {np.linalg.norm(g):.6f}")`,
  solution: `import numpy as np

def f(w):
    return np.sum(np.tanh(w) ** 2)

def analytic_grad(w):
    t = np.tanh(w)
    return 2 * t * (1 - t**2)

def numeric_grad(f, x, h=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g

w = np.array([0.3, -1.2, 0.8, 2.0])
err = np.abs(analytic_grad(w) - numeric_grad(f, w)).max()
print("max disagreement:", f"{err:.2e}", "->", "PASS" if err < 1e-7 else "FAIL")

print("\\nstep size sweep:")
for h in [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11]:
    e = np.abs(analytic_grad(w) - numeric_grad(f, w, h)).max()
    print(f"  h={h:.0e}   error {e:.3e}")

g = analytic_grad(w)
eps = 1e-6
rate_along_grad = (f(w + eps * g / np.linalg.norm(g)) - f(w)) / eps

rng = np.random.default_rng(0)
best_random = -np.inf
for _ in range(2000):
    u = rng.normal(size=w.shape)
    u = u / np.linalg.norm(u)
    best_random = max(best_random, (f(w + eps * u) - f(w)) / eps)

print(f"\\nrise per unit step along the gradient: {rate_along_grad:.6f}")
print(f"best of 2000 random directions:       {best_random:.6f}")
print(f"predicted maximum, ||grad f||:        {np.linalg.norm(g):.6f}")`,
  explain: `Part 1 is gradient checking, the standard way to catch a mistake in a hand-derived backward pass.
The sweep shows why $h$ cannot simply be made tiny: down at $h=10^{-11}$, $f(x+h)$ and $f(x-h)$ agree in almost
every digit, so subtracting them keeps the noise and throws away the signal. Somewhere around $10^{-5}$ to
$10^{-6}$ is the sweet spot in float64 — the reason is [floating point](#/l/math-numerics).

Part 2 is the steepest-ascent property, measured rather than asserted. Two thousand random directions and not
one of them beats the gradient, and the gradient's own rate lands exactly on $\\|\\nabla f\\|$ — which is what
$\\|\\nabla f\\|\\cos\\theta$ predicts at $\\cos\\theta = 1$. Note also how far short the random directions fall:
in higher dimensions that gap grows, for the same $1/\\sqrt{d}$ reason as in the first lesson. That is why
nobody optimizes by guessing directions.`,
},

'math-jacobian': {
  title: 'Write a backward pass using only VJP rules',
  prompt: `Build the backward pass of a two-layer classifier by hand, using nothing but the three rules from the
lesson. No Jacobian is ever constructed.

The forward pass is $\\mathbf{z}_1 = W_1\\mathbf{x}$, $\\mathbf{h} = \\tanh(\\mathbf{z}_1)$,
$\\mathbf{z}_2 = W_2\\mathbf{h}$, then softmax and cross-entropy against a one-hot target.

Fill in \`backward\`, working from the loss end down, using in order:

1. Softmax + cross-entropy fused: $\\partial\\mathcal{L}/\\partial \\mathbf{z}_2 = \\mathbf{p} - \\mathbf{y}$.
2. Linear layer: the gradient for the weights is the outer product
   $\\boldsymbol{\\delta}\\,\\mathbf{h}^{\\mathsf T}$, and the gradient passed down is $W_2^{\\mathsf T}\\boldsymbol{\\delta}$.
3. tanh: its Jacobian is diagonal, so multiply elementwise by $1 - \\tanh(z_1)^2$.

Then repeat rule 2 for the first layer. The check compares every gradient against numeric nudging.`,
  hint: 'Work strictly backwards and carry one vector, $\\boldsymbol{\\delta}$, the whole way. Each step either turns $\\boldsymbol{\\delta}$ into a weight gradient (outer product with that layer\'s input) or hands a new $\\boldsymbol{\\delta}$ to the layer below ($W^{\\mathsf T}\\boldsymbol{\\delta}$, or an elementwise multiply for tanh).',
  starter: `import numpy as np
rng = np.random.default_rng(0)

n_in, n_hid, n_out = 6, 5, 4
W1 = rng.normal(size=(n_hid, n_in)) * 0.5
W2 = rng.normal(size=(n_out, n_hid)) * 0.5
x  = rng.normal(size=n_in)
y  = np.eye(n_out)[2]                       # one-hot: the true class is 2

def softmax(z):
    e = np.exp(z - z.max())                 # subtracting the max changes nothing, avoids overflow
    return e / e.sum()

def forward(W1, W2):
    z1 = W1 @ x
    h  = np.tanh(z1)
    z2 = W2 @ h
    p  = softmax(z2)
    loss = -np.log(p[y.argmax()])
    return loss, (z1, h, z2, p)

def backward(W1, W2):
    """Return (dW1, dW2) using only the three VJP rules."""
    loss, (z1, h, z2, p) = forward(W1, W2)
    # TODO 1: delta at z2 -- the fused softmax + cross-entropy gradient
    d2 = np.zeros(n_out)
    # TODO 2: gradient for W2, and the delta handed down to h
    dW2 = np.zeros_like(W2)
    dh  = np.zeros(n_hid)
    # TODO 3: push that through tanh to get the delta at z1
    d1  = np.zeros(n_hid)
    # TODO 4: gradient for W1
    dW1 = np.zeros_like(W1)
    return dW1, dW2

# --- check against numeric nudging, entry by entry ---
def numeric(W, which, eps=1e-6):
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            up, dn = W.copy(), W.copy()
            up[i, j] += eps; dn[i, j] -= eps
            f_up = forward(up, W2)[0] if which == 1 else forward(W1, up)[0]
            f_dn = forward(dn, W2)[0] if which == 1 else forward(W1, dn)[0]
            g[i, j] = (f_up - f_dn) / (2 * eps)
    return g

dW1, dW2 = backward(W1, W2)
e1 = np.abs(dW1 - numeric(W1, 1)).max()
e2 = np.abs(dW2 - numeric(W2, 2)).max()
print(f"W1 gradient error: {e1:.2e}")
print(f"W2 gradient error: {e2:.2e}")
print("PASS" if max(e1, e2) < 1e-6 else "FAIL")

# --- what you avoided building ---
print(f"\\nJacobians never formed: layer 2 would be {n_out}x{n_hid}, layer 1 {n_hid}x{n_in}.")
print("At 4096 wide, that pair alone is", f"{2*4096*4096*4/1e6:.0f} MB per example.")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

n_in, n_hid, n_out = 6, 5, 4
W1 = rng.normal(size=(n_hid, n_in)) * 0.5
W2 = rng.normal(size=(n_out, n_hid)) * 0.5
x  = rng.normal(size=n_in)
y  = np.eye(n_out)[2]

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def forward(W1, W2):
    z1 = W1 @ x
    h  = np.tanh(z1)
    z2 = W2 @ h
    p  = softmax(z2)
    loss = -np.log(p[y.argmax()])
    return loss, (z1, h, z2, p)

def backward(W1, W2):
    loss, (z1, h, z2, p) = forward(W1, W2)
    d2  = p - y                       # fused softmax + cross-entropy
    dW2 = np.outer(d2, h)             # outer product with this layer's input
    dh  = W2.T @ d2                   # pass down through the linear layer
    d1  = dh * (1 - np.tanh(z1)**2)   # tanh: diagonal Jacobian -> elementwise
    dW1 = np.outer(d1, x)
    return dW1, dW2

def numeric(W, which, eps=1e-6):
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            up, dn = W.copy(), W.copy()
            up[i, j] += eps; dn[i, j] -= eps
            f_up = forward(up, W2)[0] if which == 1 else forward(W1, up)[0]
            f_dn = forward(dn, W2)[0] if which == 1 else forward(W1, dn)[0]
            g[i, j] = (f_up - f_dn) / (2 * eps)
    return g

dW1, dW2 = backward(W1, W2)
e1 = np.abs(dW1 - numeric(W1, 1)).max()
e2 = np.abs(dW2 - numeric(W2, 2)).max()
print(f"W1 gradient error: {e1:.2e}")
print(f"W2 gradient error: {e2:.2e}")
print("PASS" if max(e1, e2) < 1e-6 else "FAIL")

print(f"\\nJacobians never formed: layer 2 would be {n_out}x{n_hid}, layer 1 {n_hid}x{n_in}.")
print("At 4096 wide, that pair alone is", f"{2*4096*4096*4/1e6:.0f} MB per example.")`,
  explain: `Those five lines of \`backward\` are a real backpropagation implementation. Notice the shape of the
computation: one vector $\\boldsymbol{\\delta}$ travels from the loss down to the input, and at each layer it does
exactly two things — spin off a weight gradient by outer product with that layer's stored input, and transform
itself for the layer below. Nothing bigger than a matrix you already had is ever created, which is the point of
the whole lesson.

The numeric check needs one forward pass per weight — 54 of them here, and it would be billions for a real
model. That asymmetry *is* the forward-versus-reverse-mode argument, felt rather than argued.`,
},
'math-probability': {
  title: 'The base rate trap, and proving that least squares is Gaussian MLE',
  prompt: `Two parts, both testing claims the lesson made rather than accepting them.

1. \`posterior_sick(prevalence)\` — Bayes' rule for a 99%-accurate test. Fill it in, then read the table and
   find the prevalence at which a positive result stops being more likely right than wrong.
2. \`neg_log_lik(w)\` — the negative log-likelihood of the data under Gaussian noise, from the derivation in the
   lesson. Then the script minimises it by brute-force search and compares the winner against the closed-form
   least-squares solution. **They should agree**, because they are the same problem.`,
  hint: 'For part 1, a positive test comes from two sources: a sick person testing positive (rate `sensitivity`) and a healthy person testing positive (rate `1 - specificity`). For part 2, drop every term that does not contain `w` — you are ranking values of `w`, so shared constants change nothing.',
  starter: `import numpy as np

# ---------- part 1: the base rate trap ----------
def posterior_sick(prevalence, sensitivity=0.99, specificity=0.99):
    # TODO: return P(sick | positive test)
    return 0.0

assert abs(posterior_sick(0.5) - 0.99) < 1e-9, "at 50% prevalence a 99% test gives 0.99"
assert abs(posterior_sick(0.0) - 0.0) < 1e-9,  "nobody sick means a positive test is always a false alarm"

print("P(sick | positive), for a 99% accurate test:")
for prev in [0.5, 0.1, 0.02, 0.01, 0.001, 0.0001]:
    p = posterior_sick(prev)
    flag = "  <- coin flip" if abs(p - 0.5) < 0.02 else ""
    print(f"  prevalence {prev:8.4f}  ->  {p:.4f}{flag}")

# ---------- part 2: least squares is Gaussian MLE ----------
rng = np.random.default_rng(0)
x = rng.normal(size=80)
y = 2.5*x + 0.7 + rng.normal(0, 0.4, size=80)
X = np.column_stack([np.ones_like(x), x])
SIGMA = 0.4

def neg_log_lik(w):
    """-log p(y | X, w) under y = Xw + Gaussian(0, SIGMA^2) noise."""
    # TODO: from the derivation. Constants that do not involve w may be dropped.
    return 0.0

w_ls = np.linalg.solve(X.T @ X, X.T @ y)        # the closed-form least-squares answer

# search a grid around it and see what maximises the likelihood
grid = np.linspace(-0.6, 0.6, 121)
best_w, best_v = None, np.inf
for da in grid:
    for db in grid:
        w = w_ls + np.array([da, db])
        v = neg_log_lik(w)
        if v < best_v:
            best_v, best_w = v, w

print("\\nleast squares         :", w_ls.round(4))
print("max likelihood (search):", best_w.round(4))
gap = np.abs(best_w - w_ls).max()
print(f"largest disagreement   : {gap:.4f}")
print("PASS -- same answer, two derivations" if gap < 0.011 else "FAIL")`,
  solution: `import numpy as np

def posterior_sick(prevalence, sensitivity=0.99, specificity=0.99):
    p_pos = sensitivity*prevalence + (1 - specificity)*(1 - prevalence)
    return sensitivity*prevalence / p_pos if p_pos > 0 else 0.0

assert abs(posterior_sick(0.5) - 0.99) < 1e-9
assert abs(posterior_sick(0.0) - 0.0) < 1e-9

print("P(sick | positive), for a 99% accurate test:")
for prev in [0.5, 0.1, 0.02, 0.01, 0.001, 0.0001]:
    p = posterior_sick(prev)
    flag = "  <- coin flip" if abs(p - 0.5) < 0.02 else ""
    print(f"  prevalence {prev:8.4f}  ->  {p:.4f}{flag}")

rng = np.random.default_rng(0)
x = rng.normal(size=80)
y = 2.5*x + 0.7 + rng.normal(0, 0.4, size=80)
X = np.column_stack([np.ones_like(x), x])
SIGMA = 0.4

def neg_log_lik(w):
    resid = y - X @ w
    return np.sum(resid**2) / (2 * SIGMA**2)

w_ls = np.linalg.solve(X.T @ X, X.T @ y)

grid = np.linspace(-0.6, 0.6, 121)
best_w, best_v = None, np.inf
for da in grid:
    for db in grid:
        w = w_ls + np.array([da, db])
        v = neg_log_lik(w)
        if v < best_v:
            best_v, best_w = v, w

print("\\nleast squares         :", w_ls.round(4))
print("max likelihood (search):", best_w.round(4))
gap = np.abs(best_w - w_ls).max()
print(f"largest disagreement   : {gap:.4f}")
print("PASS -- same answer, two derivations" if gap < 0.011 else "FAIL")`,
  explain: `Part 1: at 1-in-10,000 prevalence a positive result on a 99%-accurate test means roughly a 1% chance
of being sick, and the table shows the crossover — a positive test only becomes more likely right than wrong
once about 1% of the population has the disease. Nothing about the test changed; only the prior did.

Part 2: the grid search never finds anything the closed-form least-squares solution did not already give,
because minimising $\\sum (y_i - \\mathbf{w}\\cdot\\mathbf{x}_i)^2$ and maximising the Gaussian likelihood are the
same optimisation with different constants in front. Notice also what dropping the constants proved: $\\sigma$
never appears in the answer. Your noise assumption picks the *shape* of the loss; how noisy you think it is does
not move the optimum at all.`,
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

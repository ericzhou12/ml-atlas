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
  title: 'Measure the KL gap, then watch the two directions fit differently',
  prompt: `Three parts. The third is the interesting one.

1. Write \`entropy\`, \`cross_entropy\` and \`kl\` in bits, and let the assertions confirm
   $D_{\\text{KL}}(p\\|q) = H(p,q) - H(p)$, that it is never negative, and that it is not symmetric.
2. Write a numerically stable \`log_softmax\` using the max-shift, and compare it against the naive version.
3. Then run the last block, which fits a single bell curve to a two-humped target — once by minimising
   $D_{\\text{KL}}(p\\|q)$ and once by minimising $D_{\\text{KL}}(q\\|p)$. **Predict where each one lands
   before you run it.**`,
  hint: 'Add a tiny constant inside every log so that a zero probability does not produce `-inf`. For `log_softmax`, subtract `z.max()` first — the softmax is unchanged by it, but `exp` no longer overflows.',
  starter: `import numpy as np

def entropy(p):          return 0.0   # TODO
def cross_entropy(p, q): return 0.0   # TODO
def kl(p, q):            return 0.0   # TODO

p = np.array([0.5, 0.25, 0.125, 0.125])
q = np.array([0.25, 0.25, 0.25, 0.25])

print(f"H(p)     = {entropy(p):.4f} bits")
print(f"H(p,q)   = {cross_entropy(p, q):.4f}")
print(f"KL(p||q) = {kl(p, q):.4f}")
print(f"KL(q||p) = {kl(q, p):.4f}   <- a different number")

assert abs(entropy(p) - 1.75) < 1e-6, "the hand-computed example should give 1.75 bits"
assert abs(kl(p, q) - (cross_entropy(p, q) - entropy(p))) < 1e-9, "KL identity failed"
assert kl(p, q) >= 0 and abs(kl(p, p)) < 1e-9, "KL must be >= 0, and 0 only when p == q"

# nothing beats the truth: no q does better than q = p
worst = min(cross_entropy(p, r / r.sum())
            for r in np.random.default_rng(0).random((3000, 4)) + 1e-3)
assert worst >= entropy(p) - 1e-9, "something scored below the entropy -- impossible"
print(f"\\nbest of 3000 random models: {worst:.4f}  vs  H(p) = {entropy(p):.4f}")
print("PASS\\n")

# ---------- part 2: the max-shift ----------
def log_softmax(z):
    # TODO: stable version
    return z

big = np.array([800.0, 801.0, 799.0])
print("naive :", np.log(np.exp(big) / np.exp(big).sum()))
print("stable:", log_softmax(big))
assert np.isfinite(log_softmax(big)).all(), "still overflowing"

# ---------- part 3: the two directions of KL ----------
grid = np.linspace(-6, 10, 400)
def gauss(mu, sd):
    d = np.exp(-0.5 * ((grid - mu) / sd)**2)
    return d / d.sum()

target = 0.5 * gauss(0.0, 0.8) + 0.5 * gauss(5.0, 0.8)     # two humps

def best_fit(direction):
    best = (np.inf, None)
    for mu in np.linspace(-4, 9, 131):
        for sd in np.linspace(0.3, 6.0, 115):
            q_ = gauss(mu, sd)
            d = kl(target, q_) if direction == "forward" else kl(q_, target)
            if d < best[0]:
                best = (d, (mu, sd))
    return best[1]

print("\\nfitting one bell curve to humps at 0 and 5:")
print("  forward KL(target||q) picks mu, sd =", np.round(best_fit("forward"), 2))
print("  reverse KL(q||target) picks mu, sd =", np.round(best_fit("reverse"), 2))`,
  solution: `import numpy as np

def entropy(p):          return -np.sum(p * np.log2(p + 1e-12))
def cross_entropy(p, q): return -np.sum(p * np.log2(q + 1e-12))
def kl(p, q):            return np.sum(p * np.log2((p + 1e-12) / (q + 1e-12)))

p = np.array([0.5, 0.25, 0.125, 0.125])
q = np.array([0.25, 0.25, 0.25, 0.25])

print(f"H(p)     = {entropy(p):.4f} bits")
print(f"H(p,q)   = {cross_entropy(p, q):.4f}")
print(f"KL(p||q) = {kl(p, q):.4f}")
print(f"KL(q||p) = {kl(q, p):.4f}   <- a different number")

assert abs(entropy(p) - 1.75) < 1e-6
assert abs(kl(p, q) - (cross_entropy(p, q) - entropy(p))) < 1e-9
assert kl(p, q) >= 0 and abs(kl(p, p)) < 1e-9

worst = min(cross_entropy(p, r / r.sum())
            for r in np.random.default_rng(0).random((3000, 4)) + 1e-3)
assert worst >= entropy(p) - 1e-9
print(f"\\nbest of 3000 random models: {worst:.4f}  vs  H(p) = {entropy(p):.4f}")
print("PASS\\n")

def log_softmax(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())

big = np.array([800.0, 801.0, 799.0])
print("naive :", np.log(np.exp(big) / np.exp(big).sum()))
print("stable:", log_softmax(big))
assert np.isfinite(log_softmax(big)).all()

grid = np.linspace(-6, 10, 400)
def gauss(mu, sd):
    d = np.exp(-0.5 * ((grid - mu) / sd)**2)
    return d / d.sum()

target = 0.5 * gauss(0.0, 0.8) + 0.5 * gauss(5.0, 0.8)

def best_fit(direction):
    best = (np.inf, None)
    for mu in np.linspace(-4, 9, 131):
        for sd in np.linspace(0.3, 6.0, 115):
            q_ = gauss(mu, sd)
            d = kl(target, q_) if direction == "forward" else kl(q_, target)
            if d < best[0]:
                best = (d, (mu, sd))
    return best[1]

print("\\nfitting one bell curve to humps at 0 and 5:")
print("  forward KL(target||q) picks mu, sd =", np.round(best_fit("forward"), 2))
print("  reverse KL(q||target) picks mu, sd =", np.round(best_fit("reverse"), 2))`,
  explain: `Part 1 is the KL identity and Gibbs' inequality, checked rather than believed: three thousand random
models and not one scores below $H(p)$, because $H(p)$ is a floor nothing can go under.

Part 3 is the asymmetry made concrete. Forward KL lands the curve between the two humps and stretches it wide,
because it is punished for putting near-zero probability anywhere the target has mass. Reverse KL parks a narrow
curve on one hump and pretends the other does not exist, because a region where $q$ is zero costs it nothing.
Neither answer is wrong; they are answers to different questions. Now you can predict, from the objective alone,
whether a method will produce blurry averages or confident narrow output.`,
},
'math-optimization': {
  title: 'Implement SGD, momentum and Adam, then find the exact point where SGD breaks',
  prompt: `Write the three update rules, then use them to check two numerical predictions the lesson made.

1. **The stability bound.** Plain gradient descent should converge below $\\eta = 2/\\lambda_{\\max}$ and diverge
   above it, with nothing gradual in between. The script narrows in on the boundary by bisection — see how many
   digits it matches.
2. **The condition number tax.** The lesson claimed that a condition number of $\\kappa$ costs you roughly
   $\\kappa$ times as many steps. The last block measures steps-to-converge at several values of $\\kappa$ for
   each optimizer. Predict the shape of each column before running it.`,
  hint: 'Momentum: `m = beta*m - lr*g` then `w = w + m`. Adam: update `m` and `v`, divide each by $1-\\beta^t$ to undo the cold start, then step by `lr * mh / (sqrt(vh) + eps)`.',
  starter: `import numpy as np
np.seterr(over="ignore", invalid="ignore")     # diverging runs overflow on purpose

def make(kappa):
    """A bowl whose sharpest direction is kappa times its flattest."""
    A = np.array([1.0, kappa])
    return A, (lambda w: 0.5*np.sum(A*w**2)), (lambda w: A*w)

def run(opt, lr, A, loss, grad, steps=4000, beta=0.9, tol=1e-8):
    """Return the step count at which the loss first drops below tol, or inf."""
    w = np.array([2.0, 2.0])
    m = np.zeros(2); v = np.zeros(2)
    for t in range(1, steps+1):
        g = grad(w)
        if opt == "sgd":
            pass      # TODO
        elif opt == "momentum":
            pass      # TODO -- m holds the velocity
        elif opt == "adam":
            pass      # TODO -- b1=0.9, b2=0.999, eps=1e-8, with the bias correction
        if not np.all(np.isfinite(w)):
            return np.inf
        if loss(w) < tol:
            return t
    return np.inf

A, loss, grad = make(4.0)
assert run("sgd", 0.1, A, loss, grad) < np.inf, "sgd should converge at lr=0.1"
assert run("momentum", 0.1, A, loss, grad) < np.inf, "momentum should converge at lr=0.1"
assert run("adam", 0.1, A, loss, grad) < np.inf, "adam should converge at lr=0.1"
print("PASS -- all three implemented\\n")

# ---------- part 1: locate the stability boundary by bisection ----------
lo, hi = 0.01, 2.0                       # lo converges, hi does not
for _ in range(50):
    mid = 0.5*(lo + hi)
    if run("sgd", mid, A, loss, grad) < np.inf: lo = mid
    else: hi = mid
print(f"measured breaking point : {lo:.8f}")
print(f"theory, 2/lambda_max    : {2/A.max():.8f}")
assert abs(lo - 2/A.max()) < 1e-3, "the boundary should sit right on 2/lambda_max"

# ---------- part 2: what a bad condition number costs ----------
print("\\n kappa |    sgd |  momentum |   adam     (steps to reach loss < 1e-8)")
for kappa in [1.0, 4.0, 16.0, 64.0, 256.0]:
    A, loss, grad = make(kappa)
    safe = 0.9 * 2/A.max()               # just inside the stability bound
    row = [run(o, safe if o != "adam" else 0.05, A, loss, grad) for o in
           ("sgd", "momentum", "adam")]
    print(f" {kappa:5.0f} | {row[0]:6.0f} | {row[1]:9.0f} | {row[2]:6.0f}")`,
  solution: `import numpy as np
np.seterr(over="ignore", invalid="ignore")

def make(kappa):
    A = np.array([1.0, kappa])
    return A, (lambda w: 0.5*np.sum(A*w**2)), (lambda w: A*w)

def run(opt, lr, A, loss, grad, steps=4000, beta=0.9, tol=1e-8):
    w = np.array([2.0, 2.0])
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
        if not np.all(np.isfinite(w)):
            return np.inf
        if loss(w) < tol:
            return t
    return np.inf

A, loss, grad = make(4.0)
assert run("sgd", 0.1, A, loss, grad) < np.inf
assert run("momentum", 0.1, A, loss, grad) < np.inf
assert run("adam", 0.1, A, loss, grad) < np.inf
print("PASS -- all three implemented\\n")

lo, hi = 0.01, 2.0
for _ in range(50):
    mid = 0.5*(lo + hi)
    if run("sgd", mid, A, loss, grad) < np.inf: lo = mid
    else: hi = mid
print(f"measured breaking point : {lo:.8f}")
print(f"theory, 2/lambda_max    : {2/A.max():.8f}")
assert abs(lo - 2/A.max()) < 1e-3

print("\\n kappa |    sgd |  momentum |   adam     (steps to reach loss < 1e-8)")
for kappa in [1.0, 4.0, 16.0, 64.0, 256.0]:
    A, loss, grad = make(kappa)
    safe = 0.9 * 2/A.max()
    row = [run(o, safe if o != "adam" else 0.05, A, loss, grad) for o in
           ("sgd", "momentum", "adam")]
    print(f" {kappa:5.0f} | {row[0]:6.0f} | {row[1]:9.0f} | {row[2]:6.0f}")`,
  explain: `Part 1 lands on $2/\\lambda_{\\max}$ to several decimal places. The bisection works at all only because
the transition is a genuine cliff: on one side the distance shrinks geometrically, on the other it grows
geometrically, and there is nothing in between.

Part 2 is the condition-number tax, measured. Once $\\kappa$ is large the SGD column grows in proportion to it —
336 steps at $\\kappa=64$ becomes 1355 at $\\kappa=256$, four times the conditioning for four times the steps.
That is what "one sharp direction taxes the whole model" costs in practice. Momentum grows more slowly. Adam barely notices, because dividing by each axis's own gradient history
undoes the difference in scale between the axes, which is exactly the job it was designed for.`,
},

'math-numerics': {
  title: 'Break a variance computation, then break a least-squares solve',
  prompt: `Two failures, both invisible unless you go looking.

1. **Cancellation.** \`var_naive\` uses the textbook $\\mathbb{E}[X^2]-\\mathbb{E}[X]^2$. Write \`var_twopass\`,
   which subtracts the mean *before* squaring, and watch the two diverge in float32 as the mean grows. The
   spread is 2.0 in every row — only the offset changes.
2. **Conditioning.** Write \`solve_normal\`, which fits least squares the textbook way by forming
   $X^{\\mathsf T}X$. Compare it against \`np.linalg.lstsq\` on a matrix with two nearly identical columns, and
   check the answer against the $\\log_{10}\\kappa$ digits-lost rule.

Read the last block carefully — it shows a case where even the good variance formula cannot help you.`,
  hint: 'For the two-pass version: compute the mean, subtract it from every entry, then average the squares. For the normal equations: `np.linalg.solve(X.T @ X, X.T @ y)`.',
  starter: `import numpy as np

# ---------- part 1: cancellation ----------
def var_naive(x):
    return (x**2).mean() - x.mean()**2

def var_twopass(x):
    # TODO: subtract the mean first, then average the squares
    return 0.0

base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])       # variance is exactly 2.0
print("all five rows describe data with a spread of 2.0:\\n")
print("      offset |          naive |   two-pass")
for offset in [0.0, 1e2, 1e4, 1e5, 1e6]:
    x = (base + offset).astype(np.float32)
    print(f"  {offset:10.0e} | {var_naive(x):14.6f} | {var_twopass(x):10.6f}")

x = (base + 1e4).astype(np.float32)
assert abs(var_twopass(x) - 2.0) < 0.01, "the two-pass version should stay accurate here"
assert abs(var_naive(x) - 2.0) > 0.01,   "the naive version should already be wrong here"
print("\\nPASS -- part 1\\n")

# ---------- part 2: conditioning ----------
def solve_normal(X, y):
    # TODO: solve the normal equations X^T X w = X^T y
    return np.zeros(X.shape[1])

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 6))
X[:, 1] = X[:, 0] + 1e-7 * rng.normal(size=200)   # column 1 nearly copies column 0
w_true = rng.normal(size=6)
y = X @ w_true

k = np.linalg.cond(X)
print(f"cond(X)      = {k:.2e}      -> about {np.log10(k):.1f} digits lost")
print(f"cond(X^T X)  = {np.linalg.cond(X.T @ X):.2e}      -> about {2*np.log10(k):.1f} digits lost")
print(f"float64 carries about 16 digits.\\n")

w_norm  = solve_normal(X, y)
w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"prediction error, normal equations: {np.abs(X @ w_norm  - y).max():.3e}")
print(f"prediction error, lstsq           : {np.abs(X @ w_lstsq - y).max():.3e}")
print("PASS -- part 2" if np.abs(X @ w_lstsq - y).max() < np.abs(X @ w_norm - y).max() else "FAIL")

# ---------- where no formula saves you ----------
x = (base + 1e8).astype(np.float32)
print("\\nat offset 1e8 in float32 the five inputs are:", np.unique(x))
print("two-pass variance:", var_twopass(x))
print("Nothing was cancelled here -- the five distinct numbers stopped being distinct")
print("on the way in. No choice of formula can recover data the format cannot hold.")`,
  solution: `import numpy as np

def var_naive(x):
    return (x**2).mean() - x.mean()**2

def var_twopass(x):
    m = x.mean()
    return ((x - m) ** 2).mean()

base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print("all five rows describe data with a spread of 2.0:\\n")
print("      offset |          naive |   two-pass")
for offset in [0.0, 1e2, 1e4, 1e5, 1e6]:
    x = (base + offset).astype(np.float32)
    print(f"  {offset:10.0e} | {var_naive(x):14.6f} | {var_twopass(x):10.6f}")

x = (base + 1e4).astype(np.float32)
assert abs(var_twopass(x) - 2.0) < 0.01
assert abs(var_naive(x) - 2.0) > 0.01
print("\\nPASS -- part 1\\n")

def solve_normal(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 6))
X[:, 1] = X[:, 0] + 1e-7 * rng.normal(size=200)
w_true = rng.normal(size=6)
y = X @ w_true

k = np.linalg.cond(X)
print(f"cond(X)      = {k:.2e}      -> about {np.log10(k):.1f} digits lost")
print(f"cond(X^T X)  = {np.linalg.cond(X.T @ X):.2e}      -> about {2*np.log10(k):.1f} digits lost")
print(f"float64 carries about 16 digits.\\n")

w_norm  = solve_normal(X, y)
w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"prediction error, normal equations: {np.abs(X @ w_norm  - y).max():.3e}")
print(f"prediction error, lstsq           : {np.abs(X @ w_lstsq - y).max():.3e}")
print("PASS -- part 2" if np.abs(X @ w_lstsq - y).max() < np.abs(X @ w_norm - y).max() else "FAIL")

x = (base + 1e8).astype(np.float32)
print("\\nat offset 1e8 in float32 the five inputs are:", np.unique(x))
print("two-pass variance:", var_twopass(x))
print("Nothing was cancelled here -- the five distinct numbers stopped being distinct")
print("on the way in. No choice of formula can recover data the format cannot hold.")`,
  explain: `Part 1: by an offset of $10^4$ the naive formula returns exactly **0.000000** — every significant
digit of the answer has cancelled away — and by $10^6$ it returns **65536**, off by four orders of magnitude in
the other direction. Subtracting the mean first keeps every squared quantity small, and gives 2.0 in every row.

Part 2: two nearly duplicate columns push $\\kappa(X)$ to around $10^7$, which the rule says costs seven of
float64's sixteen digits. Forming $X^{\\mathsf T}X$ squares that to $10^{14}$ and costs fourteen — nearly
everything. \`lstsq\` never builds that product, and its residual comes out orders of magnitude smaller.

The last block is the part worth remembering longest. At an offset of $10^8$, consecutive float32 values are
8 apart, so the five distinct inputs collapse into just **two** stored values before any arithmetic happens at
all — and the two-pass formula dutifully reports 12.8, the correct variance of the data that actually made it
into memory. A better formula cannot help here. Only a wider format, or centring the data before you store it,
can.`,
},

};

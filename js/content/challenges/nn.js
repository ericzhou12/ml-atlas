/* ============================================================
   Challenges — track 3, neural networks

   One entry per lesson id: { title, prompt, starter, solution, hint?, explain? }
   Starters carry their own PASS/FAIL check so the learner gets a verdict
   without leaving the lab.

   EDITOR NOTE: these are JS template literals, so an unescaped backtick
   silently terminates the string. Escape inline code in prose.
   ============================================================ */

export default {

'nn-perceptron-mlp': {
  title: 'Solve XOR, prove a linear model cannot, then look at the features it invented',
  prompt: `1. Complete the backward pass and train a 2-layer MLP on XOR until the loss is near zero. Then rerun it
   with the hidden activation set to the identity and watch it plateau — no width and no amount of training
   rescues it.
2. **Look at what the hidden layer built.** The last block feeds the four inputs through the trained hidden
   layer and then checks whether the four points have become linearly separable *in those new coordinates*.
   The lesson claimed a network is "logistic regression that invents its own features" — this is the check.`,
  hint: 'Backward pass, top to bottom: `d2 = (p - y)/len(X)` from the sigmoid-plus-cross-entropy collapse; then `gW2 = h.T @ d2`; then `d1 = (d2 @ W2.T) * df(z1)`; then `gW1 = X.T @ d1`. Every line is one of the rules from the Jacobian lesson.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

X = np.array([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
y = np.array([[0.], [1.], [1.], [0.]])

def train(act, width=8, steps=4000, lr=0.5):
    r = np.random.default_rng(0)
    W1 = r.normal(0, 1.0, (2, width)); b1 = np.zeros(width)
    W2 = r.normal(0, np.sqrt(2/width), (width, 1)); b2 = np.zeros(1)
    f  = (lambda z: np.maximum(0, z)) if act == "relu" else (lambda z: z)
    df = (lambda z: (z > 0).astype(float)) if act == "relu" else (lambda z: np.ones_like(z))
    for t in range(steps):
        z1 = X @ W1 + b1; h = f(z1)
        p = 1/(1+np.exp(-(h @ W2 + b2)))
        # TODO: backward pass, then subtract lr * gradient from each of W1, b1, W2, b2
        pass
    loss = float(-np.mean(y*np.log(p+1e-12) + (1-y)*np.log(1-p+1e-12)))
    return loss, p.ravel(), (W1, b1, f)

for act in ["relu", "identity"]:
    loss, preds, _ = train(act)
    print(f"{act:9s} loss {loss:.5f}   preds {preds.round(3)}   target {y.ravel()}")

loss_relu, _, (W1, b1, f) = train("relu")
loss_lin,  _, _           = train("identity")
assert loss_relu < 0.05,        "the ReLU network should fit XOR almost exactly"
assert abs(loss_lin - np.log(2)) < 0.02, "the linear one should sit at ln(2) = 0.693, pure chance"
print("PASS\\n")

# ---------- what did the hidden layer build? ----------
H = f(X @ W1 + b1)
print("the four inputs, re-described by the hidden layer:")
for xi, hi, yi in zip(X, H, y.ravel()):
    print(f"  {xi} (label {yi:.0f})  ->  {np.round(hi, 2)}")

# are they linearly separable NOW? fit a plain linear model on H and see.
Hb = np.c_[np.ones(4), H]
w = np.linalg.lstsq(Hb, y.ravel()*2 - 1, rcond=None)[0]
sep_after = ((Hb @ w > 0) == (y.ravel() > 0.5)).all()

Xb = np.c_[np.ones(4), X]
w0 = np.linalg.lstsq(Xb, y.ravel()*2 - 1, rcond=None)[0]
sep_before = ((Xb @ w0 > 0) == (y.ravel() > 0.5)).all()

print(f"\\nlinearly separable in the ORIGINAL coordinates? {sep_before}")
print(f"linearly separable in the HIDDEN coordinates?   {sep_after}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

X = np.array([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
y = np.array([[0.], [1.], [1.], [0.]])

def train(act, width=8, steps=4000, lr=0.5):
    r = np.random.default_rng(0)
    W1 = r.normal(0, 1.0, (2, width)); b1 = np.zeros(width)
    W2 = r.normal(0, np.sqrt(2/width), (width, 1)); b2 = np.zeros(1)
    f  = (lambda z: np.maximum(0, z)) if act == "relu" else (lambda z: z)
    df = (lambda z: (z > 0).astype(float)) if act == "relu" else (lambda z: np.ones_like(z))
    for t in range(steps):
        z1 = X @ W1 + b1; h = f(z1)
        p = 1/(1+np.exp(-(h @ W2 + b2)))
        d2 = (p - y) / len(X)
        gW2, gb2 = h.T @ d2, d2.sum(0)
        d1 = (d2 @ W2.T) * df(z1)
        gW1, gb1 = X.T @ d1, d1.sum(0)
        W2 = W2 - lr*gW2; b2 = b2 - lr*gb2
        W1 = W1 - lr*gW1; b1 = b1 - lr*gb1
    loss = float(-np.mean(y*np.log(p+1e-12) + (1-y)*np.log(1-p+1e-12)))
    return loss, p.ravel(), (W1, b1, f)

for act in ["relu", "identity"]:
    loss, preds, _ = train(act)
    print(f"{act:9s} loss {loss:.5f}   preds {preds.round(3)}   target {y.ravel()}")

loss_relu, _, (W1, b1, f) = train("relu")
loss_lin,  _, _           = train("identity")
assert loss_relu < 0.05
assert abs(loss_lin - np.log(2)) < 0.02
print("PASS\\n")

H = f(X @ W1 + b1)
print("the four inputs, re-described by the hidden layer:")
for xi, hi, yi in zip(X, H, y.ravel()):
    print(f"  {xi} (label {yi:.0f})  ->  {np.round(hi, 2)}")

Hb = np.c_[np.ones(4), H]
w = np.linalg.lstsq(Hb, y.ravel()*2 - 1, rcond=None)[0]
sep_after = ((Hb @ w > 0) == (y.ravel() > 0.5)).all()

Xb = np.c_[np.ones(4), X]
w0 = np.linalg.lstsq(Xb, y.ravel()*2 - 1, rcond=None)[0]
sep_before = ((Xb @ w0 > 0) == (y.ravel() > 0.5)).all()

print(f"\\nlinearly separable in the ORIGINAL coordinates? {sep_before}")
print(f"linearly separable in the HIDDEN coordinates?   {sep_after}")`,
  explain: `Part 1: the identity version settles at a loss of $\\ln 2 \\approx 0.693$ and predicts 0.5 for every
input. That is not slow convergence — 0.693 is exactly the loss of a model that has given up and is guessing,
and it is where a linear model *must* end up on XOR. This is the 1969 Minsky–Papert result, reproduced in a
dozen lines.

Part 2 is the point of having a hidden layer at all. In the original coordinates the four points are not
linearly separable, and the last two lines confirm it. After passing through the trained hidden layer they are.
Nothing was added — no new information about XOR entered the picture — the hidden layer simply re-described each
input as a list of eight numbers, and in that description a plain weighted sum suffices.

That is the whole architecture in miniature. The final layer of any classifier is a linear model; everything
before it exists to hand that linear model a description it can work with. Deep learning is the discovery that
gradient descent can find such descriptions on its own, for problems far harder than XOR.`,
},

'nn-backprop': {
  title: 'Write a scalar autodiff engine, and find the bug that catches everyone',
  prompt: `Complete the three \`_backward\` closures so that \`backward()\` computes correct gradients through any
expression built from these operations. Each closure needs exactly two things, as the lesson said: the local
derivative of its own operation, and the gradient arriving from downstream.

Three checks follow, and the second is the interesting one — it uses the same value in more than one place. A
version that passes check 1 and fails check 2 has the single most common bug in hand-written backward passes.`,
  hint: 'Local derivatives: for `a+b`, both inputs receive the incoming gradient unchanged. For `a*b`, each input receives the incoming gradient times *the other* input. For `tanh`, multiply by $1-\\tanh^2$. And write `+=`, not `=` — think about what should happen to `x` in `x*x`.',
  starter: `import numpy as np

class Value:
    def __init__(self, data, children=()):
        self.data = float(data); self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _b():
            pass    # TODO
        out._backward = _b
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _b():
            pass    # TODO
        out._backward = _b
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,))
        def _b():
            pass    # TODO
        out._backward = _b
        return out

    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo): v._backward()

def numeric(f, at, eps=1e-6):
    return (f(at + eps) - f(at - eps)) / (2*eps)

# --- check 1: one neuron ---
x1, x2 = Value(2.0), Value(-1.0)
w1, w2, b = Value(0.5), Value(1.5), Value(0.3)
out = (x1*w1 + x2*w2 + b).tanh()
out.backward()
e1 = numeric(lambda W: np.tanh(2.0*W - 1.5 + 0.3), 0.5)
print(f"dout/dw1:  yours {w1.grad:.8f}   numeric {e1:.8f}")
assert abs(w1.grad - e1) < 1e-5, "the gradient through the neuron is wrong"

# --- check 2: a value used more than once. This is where += matters. ---
x = Value(1.7)
y = x*x + x
y.backward()
e2 = numeric(lambda v: v*v + v, 1.7)
print(f"\\nd(x*x + x)/dx:  yours {x.grad:.8f}   numeric {e2:.8f}   (2x+1 = {2*1.7+1:.2f})")
assert abs(x.grad - e2) < 1e-5, "a reused value must ACCUMULATE its gradient: use += , not ="

# --- check 3: a deeper chain ---
a, bb = Value(0.8), Value(-0.4)
z = ((a*bb + a).tanh() * a + bb).tanh()
z.backward()
e3 = numeric(lambda v: np.tanh(np.tanh(v*(-0.4) + v)*v + (-0.4)), 0.8)
print(f"\\ndeeper chain, dz/da:  yours {a.grad:.8f}   numeric {e3:.8f}")
assert abs(a.grad - e3) < 1e-5, "the chain rule is not composing correctly"
print("\\nPASS -- all three")`,
  solution: `import numpy as np

class Value:
    def __init__(self, data, children=()):
        self.data = float(data); self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _b():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _b
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _b():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _b
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,))
        def _b():
            self.grad += (1 - t*t) * out.grad
        out._backward = _b
        return out

    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo): v._backward()

def numeric(f, at, eps=1e-6):
    return (f(at + eps) - f(at - eps)) / (2*eps)

x1, x2 = Value(2.0), Value(-1.0)
w1, w2, b = Value(0.5), Value(1.5), Value(0.3)
out = (x1*w1 + x2*w2 + b).tanh()
out.backward()
e1 = numeric(lambda W: np.tanh(2.0*W - 1.5 + 0.3), 0.5)
print(f"dout/dw1:  yours {w1.grad:.8f}   numeric {e1:.8f}")
assert abs(w1.grad - e1) < 1e-5

x = Value(1.7)
y = x*x + x
y.backward()
e2 = numeric(lambda v: v*v + v, 1.7)
print(f"\\nd(x*x + x)/dx:  yours {x.grad:.8f}   numeric {e2:.8f}   (2x+1 = {2*1.7+1:.2f})")
assert abs(x.grad - e2) < 1e-5

a, bb = Value(0.8), Value(-0.4)
z = ((a*bb + a).tanh() * a + bb).tanh()
z.backward()
e3 = numeric(lambda v: np.tanh(np.tanh(v*(-0.4) + v)*v + (-0.4)), 0.8)
print(f"\\ndeeper chain, dz/da:  yours {a.grad:.8f}   numeric {e3:.8f}")
assert abs(a.grad - e3) < 1e-5
print("\\nPASS -- all three")`,
  explain: `Checks 1 and 3 confirm that the chain rule composes correctly through arbitrary expressions — and note
that nothing in your code knew what expression it was part of. Each closure knew one local derivative and
multiplied it by whatever arrived from downstream. Correct gradients for the whole graph fell out of that alone,
which is exactly the locality the lesson emphasised.

Check 2 is the one worth remembering. In \`x*x + x\` the value \`x\` feeds three separate places, and the true
derivative is $2x + 1 = 4.4$. If your closures assign with \`=\`, whichever contribution lands last silently
overwrites the others and you get 1, or 3.4, but never the sum. The rule is: **when a value is used in several
places, its gradient is the sum of what each use sends back.**

This bug is nasty precisely because it rarely crashes and usually does not look wrong — the model still trains,
just toward the wrong thing. Comparing against finite differences is how you catch it, which is why every
framework's test suite is full of the exact comparison you just ran.`,
},

'nn-activations': {
  title: 'Measure saturation, and kill some ReLUs',
  prompt: `Compute what fraction of inputs give each activation a near-zero gradient. Then deliberately induce dead
ReLUs with an aggressive learning rate and count them.`,
  hint: 'A ReLU is dead when its pre-activation is negative for *every* input in the dataset.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def relu(x):    return np.maximum(0, x)
def gelu(x):    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
def sigmoid(x): return 1/(1+np.exp(-x))

def deriv(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

x = np.linspace(-6, 6, 2001)
print(f"{'activation':12s} {'max deriv':>10s} {'% |deriv| < 0.01':>18s}")
for name, f in [("sigmoid", sigmoid), ("tanh", np.tanh), ("relu", relu), ("gelu", gelu)]:
    d = deriv(f, x)
    # TODO: print the max derivative and the saturated fraction
    pass

# --- dead ReLUs ---
W = rng.normal(0, 0.5, (256, 256)); b = np.zeros(256)
h = rng.normal(0, 1, (512, 256))
for step in range(50):
    z = h @ W + b
    g = (z > 0).astype(float) * rng.normal(0, 1, z.shape)
    b -= 0.5 * g.mean(0)                  # deliberately too large
dead = (relu(h @ W + b).max(0) == 0).mean()
print(f"\\nfraction of permanently dead ReLU units: {dead:.1%}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def relu(x):    return np.maximum(0, x)
def gelu(x):    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
def sigmoid(x): return 1/(1+np.exp(-x))
def deriv(f, x, h=1e-5): return (f(x+h) - f(x-h)) / (2*h)

x = np.linspace(-6, 6, 2001)
print(f"{'activation':12s} {'max deriv':>10s} {'% |deriv| < 0.01':>18s}")
for name, f in [("sigmoid", sigmoid), ("tanh", np.tanh), ("relu", relu), ("gelu", gelu)]:
    d = deriv(f, x)
    print(f"{name:12s} {np.abs(d).max():10.4f} {np.mean(np.abs(d)<0.01)*100:17.1f}%")

W = rng.normal(0, 0.5, (256, 256)); b = np.zeros(256)
h = rng.normal(0, 1, (512, 256))
for step in range(50):
    z = h @ W + b
    g = (z > 0).astype(float) * rng.normal(0, 1, z.shape)
    b -= 0.5 * g.mean(0)
dead = (relu(h @ W + b).max(0) == 0).mean()
print(f"\\nfraction of permanently dead ReLU units: {dead:.1%}")`,
},

'nn-initialization': {
  title: 'Derive the He factor empirically',
  prompt: `Propagate a signal through 25 layers at several initialization scales and find, by search, the standard
deviation that keeps activation variance constant. Compare it to $\\sqrt{2/n}$.`,
  hint: 'ReLU zeroes half the distribution, halving the variance — the factor of 2 is compensating for exactly that.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def propagate(std, depth=25, width=256, act="relu"):
    h = rng.normal(0, 1, (128, width))
    stds = []
    for _ in range(depth):
        W = rng.normal(0, std, (width, width))
        z = h @ W
        h = np.maximum(0, z) if act == "relu" else np.tanh(z)
        stds.append(h.std())
    return np.array(stds)

n = 256
print(f"{'scheme':14s} {'std':>10} " + "".join(f"{'L'+str(d):>11}" for d in [1,5,10,25]))
for name, std in [("tiny", 0.01), ("xavier", np.sqrt(1/n)), ("he", np.sqrt(2/n)), ("big", 1.0)]:
    s = propagate(std)
    print(f"{name:14s} {std:10.5f} " + "".join(f"{s[d-1]:11.3e}" for d in [1,5,10,25]))

# TODO: search over std to find the value keeping std(L25) closest to std(L1)
print("\\nsearching for the variance-preserving scale...")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def propagate(std, depth=25, width=256, act="relu"):
    h = rng.normal(0, 1, (128, width))
    stds = []
    for _ in range(depth):
        W = rng.normal(0, std, (width, width))
        z = h @ W
        h = np.maximum(0, z) if act == "relu" else np.tanh(z)
        stds.append(h.std())
    return np.array(stds)

n = 256
print(f"{'scheme':14s} {'std':>10} " + "".join(f"{'L'+str(d):>11}" for d in [1,5,10,25]))
for name, std in [("tiny", 0.01), ("xavier", np.sqrt(1/n)), ("he", np.sqrt(2/n)), ("big", 1.0)]:
    s = propagate(std)
    print(f"{name:14s} {std:10.5f} " + "".join(f"{s[d-1]:11.3e}" for d in [1,5,10,25]))

best, best_score = None, np.inf
for std in np.linspace(0.02, 0.15, 200):
    s = propagate(std)
    if not np.all(np.isfinite(s)) or s[-1] == 0: continue
    score = abs(np.log(s[-1] / s[0]))
    if score < best_score: best_score, best = score, std

print(f"\\nempirical variance-preserving std: {best:.5f}")
print(f"He formula sqrt(2/n):              {np.sqrt(2/n):.5f}")
print(f"Xavier sqrt(1/n):                  {np.sqrt(1/n):.5f}")`,
},

'nn-normalization': {
  title: 'Implement BatchNorm, LayerNorm, and RMSNorm',
  prompt: `Write all three and confirm which axis each one normalizes. Then show that LayerNorm is invariant to
scaling the *input row* while BatchNorm is not.`,
  hint: 'BatchNorm reduces over the batch axis (0); LayerNorm and RMSNorm reduce over features (-1).',
  starter: `import numpy as np
x = np.random.default_rng(0).normal(2.0, 3.0, (8, 16))

def batchnorm(x, eps=1e-5):  return x   # TODO
def layernorm(x, eps=1e-5):  return x   # TODO
def rmsnorm(x, eps=1e-5):    return x   # TODO

for name, f in [("input", lambda z: z), ("batchnorm", batchnorm),
                ("layernorm", layernorm), ("rmsnorm", rmsnorm)]:
    y = f(x)
    print(f"{name:11s} per-feature mean {y.mean(0)[:3].round(3)}  "
          f"per-row std {y.std(-1)[:3].round(3)}")

# scale one example by 10 -- which norms are unaffected?
x2 = x.copy(); x2[0] *= 10
print("\\nrow 0 scaled by 10:")
for name, f in [("batchnorm", batchnorm), ("layernorm", layernorm), ("rmsnorm", rmsnorm)]:
    same = np.allclose(f(x)[1], f(x2)[1], atol=1e-6)
    print(f"  {name:11s} row 1 unchanged: {same}")`,
  solution: `import numpy as np
x = np.random.default_rng(0).normal(2.0, 3.0, (8, 16))

def batchnorm(x, eps=1e-5):
    return (x - x.mean(0, keepdims=True)) / np.sqrt(x.var(0, keepdims=True) + eps)

def layernorm(x, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)

def rmsnorm(x, eps=1e-5):
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps)

for name, f in [("input", lambda z: z), ("batchnorm", batchnorm),
                ("layernorm", layernorm), ("rmsnorm", rmsnorm)]:
    y = f(x)
    print(f"{name:11s} per-feature mean {y.mean(0)[:3].round(3)}  "
          f"per-row std {y.std(-1)[:3].round(3)}")

x2 = x.copy(); x2[0] *= 10
print("\\nrow 0 scaled by 10:")
for name, f in [("batchnorm", batchnorm), ("layernorm", layernorm), ("rmsnorm", rmsnorm)]:
    print(f"  {name:11s} row 1 unchanged: {np.allclose(f(x)[1], f(x2)[1], atol=1e-6)}")`,
  explain: 'Only BatchNorm changes row 1 when row 0 is perturbed — it couples examples through the batch statistics. That coupling is exactly why it misbehaves with small batches and variable-length sequences, and why transformers use LayerNorm.',
},

'nn-regularization': {
  title: 'Inverted dropout, and checking the expectation',
  prompt: `Implement inverted dropout and verify the expected activation is preserved. Then confirm that forgetting
the $1/(1-p)$ scale shifts the mean at inference — a classic bug.`,
  hint: 'Scale the survivors by $1/(1-p)$ during training so nothing has to change at test time.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def dropout(x, p, training=True, scale=True):
    if not training or p == 0: return x
    mask = (rng.random(x.shape) > p).astype(float)
    # TODO: apply the mask, and divide by (1-p) when scale=True
    return x * mask

x = np.ones((2000, 64))
for p in [0.0, 0.2, 0.5, 0.8]:
    correct = dropout(x, p).mean()
    buggy   = dropout(x, p, scale=False).mean()
    print(f"p={p:.1f}  inverted {correct:.4f}   without scaling {buggy:.4f}   "
          f"(train mean should stay 1.0)")

assert abs(dropout(x, 0.5).mean() - 1.0) < 0.02, "expectation not preserved"
print("\\nPASS")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def dropout(x, p, training=True, scale=True):
    if not training or p == 0: return x
    mask = (rng.random(x.shape) > p).astype(float)
    if scale: mask /= (1 - p)
    return x * mask

x = np.ones((2000, 64))
for p in [0.0, 0.2, 0.5, 0.8]:
    print(f"p={p:.1f}  inverted {dropout(x,p).mean():.4f}   "
          f"without scaling {dropout(x,p,scale=False).mean():.4f}")
assert abs(dropout(x, 0.5).mean() - 1.0) < 0.02
print("\\nPASS")`,
},

'nn-losses-training': {
  title: 'Check the initial loss, then overfit one batch',
  prompt: `Two diagnostics that catch most bugs. Confirm a $K$-class model starts at $\\log K$, then drive the loss on
8 examples to near zero. If the second fails, you have a bug rather than a tuning problem.`,
  hint: 'A randomly initialized softmax head should be uniform over classes, giving $-\\log(1/K)$.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

K, D = 10, 20
X = rng.normal(size=(512, D)); y = rng.integers(0, K, 512)
X[np.arange(512), y % D] += 2.0                        # make it learnable

W1 = rng.normal(0, np.sqrt(2/D), (D, 32)); b1 = np.zeros(32)
W2 = rng.normal(0, np.sqrt(2/32), (32, K)); b2 = np.zeros(K)

def forward(X):
    h = np.maximum(0, X @ W1 + b1)
    return h @ W2 + b2, h

def loss_and_grads(X, y):
    logits, h = forward(X)
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    d = p.copy(); d[np.arange(len(y)), y] -= 1; d /= len(y)
    dh = (d @ W2.T) * (h > 0)
    return loss, (X.T @ dh, dh.sum(0), h.T @ d, d.sum(0))

# CHECK 1
l0, _ = loss_and_grads(X, y)
print(f"initial loss {l0:.4f}   expected log({K}) = {np.log(K):.4f}   "
      f"{'PASS' if abs(l0-np.log(K)) < 0.15 else 'FAIL'}")

# CHECK 2: TODO -- train on X[:8], y[:8] for 800 steps at lr=0.3 and print the loss
print("\\noverfit-one-batch: implement the loop")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

K, D = 10, 20
X = rng.normal(size=(512, D)); y = rng.integers(0, K, 512)
X[np.arange(512), y % D] += 2.0

W1 = rng.normal(0, np.sqrt(2/D), (D, 32)); b1 = np.zeros(32)
W2 = rng.normal(0, np.sqrt(2/32), (32, K)); b2 = np.zeros(K)

def forward(X):
    h = np.maximum(0, X @ W1 + b1)
    return h @ W2 + b2, h

def loss_and_grads(X, y):
    logits, h = forward(X)
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    d = p.copy(); d[np.arange(len(y)), y] -= 1; d /= len(y)
    dh = (d @ W2.T) * (h > 0)
    return loss, (X.T @ dh, dh.sum(0), h.T @ d, d.sum(0))

l0, _ = loss_and_grads(X, y)
print(f"CHECK 1  initial loss {l0:.4f}  expected {np.log(K):.4f}  "
      f"{'PASS' if abs(l0-np.log(K)) < 0.15 else 'FAIL'}")

Xs, ys = X[:8], y[:8]
for t in range(800):
    l, (g1, gb1, g2, gb2) = loss_and_grads(Xs, ys)
    W1 -= 0.3*g1; b1 -= 0.3*gb1; W2 -= 0.3*g2; b2 -= 0.3*gb2
print(f"CHECK 2  loss on 8 examples after 800 steps: {l:.6f}  "
      f"{'PASS' if l < 0.01 else 'FAIL - you have a bug'}")`,
},

'nn-cnn': {
  title: 'Convolution by hand, and the parameter-count argument',
  prompt: `Implement 2-D convolution with stride and padding, verify a Sobel filter finds edges, then compute how
many parameters a dense layer would need for the same job.`,
  hint: 'Frameworks actually implement cross-correlation (no kernel flip) — do the same.',
  starter: `import numpy as np

def conv2d(x, k, stride=1, pad=0):
    if pad: x = np.pad(x, pad)
    kh, kw = k.shape
    H = (x.shape[0]-kh)//stride + 1
    W = (x.shape[1]-kw)//stride + 1
    out = np.zeros((H, W))
    # TODO
    return out

img = np.zeros((12, 12)); img[3:9, 3:9] = 1.0
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float)

r = conv2d(img, sobel_x, pad=1)
print("vertical-edge response:")
print(r.astype(int))
assert r.shape == (12, 12), "same-padding should preserve size"
assert abs(r[6, 3]) > 2 and abs(r[6, 6]) < 1e-9, "should fire at edges, not the interior"
print("\\nPASS\\n")

for cin, cout, k, h, w in [(64, 128, 3, 56, 56)]:
    conv = cin*cout*k*k + cout
    dense = h*w*cin * cout + cout
    print(f"{k}x{k} conv {cin}->{cout}: {conv:,} params")
    print(f"dense {h}x{w}x{cin} -> {cout}: {dense:,} params  ({dense/conv:,.0f}x more)")`,
  solution: `import numpy as np

def conv2d(x, k, stride=1, pad=0):
    if pad: x = np.pad(x, pad)
    kh, kw = k.shape
    H = (x.shape[0]-kh)//stride + 1
    W = (x.shape[1]-kw)//stride + 1
    out = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            patch = x[i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[i, j] = (patch * k).sum()
    return out

img = np.zeros((12, 12)); img[3:9, 3:9] = 1.0
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float)
r = conv2d(img, sobel_x, pad=1)
print(r.astype(int))
assert r.shape == (12, 12)
assert abs(r[6, 3]) > 2 and abs(r[6, 6]) < 1e-9
print("\\nPASS\\n")

cin, cout, k, h, w = 64, 128, 3, 56, 56
conv = cin*cout*k*k + cout
dense = h*w*cin*cout + cout
print(f"{k}x{k} conv {cin}->{cout}: {conv:,} params")
print(f"dense equivalent:          {dense:,} params  ({dense/conv:,.0f}x more)")`,
},

'nn-rnn': {
  title: 'Watch a gradient vanish, then fix it with a gate',
  prompt: `Compute $\\partial h_T/\\partial h_0$ for a vanilla RNN across sequence lengths and recurrent weights.
Then show an LSTM-style additive cell state with forget gate 1.0 keeps the gradient at exactly 1.`,
  hint: 'The vanilla gradient is $\\prod_t w\\,\\phi\'(z_t)$; the cell-state gradient is $\\prod_t f_t$.',
  starter: `import numpy as np

def rnn_grad(T, w):
    h, pres = 0.0, []
    for t in range(T):
        z = w*h + (1.0 if t == 0 else 0.0)
        pres.append(z); h = np.tanh(z)
    # TODO: multiply w * tanh'(z_t) backwards over all T steps
    return 1.0

print("vanilla RNN, d h_T / d h_0:")
for w in [0.5, 0.9, 1.0, 1.1]:
    print(f"  w={w}: " + "  ".join(f"T={T}: {rnn_grad(T,w):.3e}" for T in [10, 50, 100]))

print("\\nLSTM cell state (product of forget gates):")
for f in [0.5, 0.9, 0.99, 1.0]:
    print(f"  f={f}: " + "  ".join(f"T={T}: {f**T:.3e}" for T in [10, 50, 100]))

assert abs(1.0**100 - 1.0) < 1e-12
print("\\nWith f=1.0 the gradient is exactly 1 after 100 steps. That is the whole idea.")`,
  solution: `import numpy as np

def rnn_grad(T, w):
    h, pres = 0.0, []
    for t in range(T):
        z = w*h + (1.0 if t == 0 else 0.0)
        pres.append(z); h = np.tanh(z)
    g = 1.0
    for t in reversed(range(T)):
        g *= w * (1 - np.tanh(pres[t])**2)
    return g

print("vanilla RNN, d h_T / d h_0:")
for w in [0.5, 0.9, 1.0, 1.1]:
    print(f"  w={w}: " + "  ".join(f"T={T}: {rnn_grad(T,w):.3e}" for T in [10, 50, 100]))

print("\\nLSTM cell state (product of forget gates):")
for f in [0.5, 0.9, 0.99, 1.0]:
    print(f"  f={f}: " + "  ".join(f"T={T}: {f**T:.3e}" for T in [10, 50, 100]))
print("\\nWith f=1.0 the gradient is exactly 1 after 100 steps.")`,
},

'nn-embeddings': {
  title: 'Train skip-gram embeddings with negative sampling',
  prompt: `Implement the negative-sampling update and check that words appearing in similar contexts end up with
high cosine similarity.`,
  hint: 'For each (center, context) pair, push the true context up and $k$ random words down: gradient is $(\\sigma(u\\cdot v) - \\text{label})$.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

corpus = ("the cat sat on the mat . the dog sat on the rug . "
          "the cat chased the mouse . the dog chased the cat . "
          "a mouse ate the cheese . the cat ate the mouse .").split()
vocab = sorted(set(corpus)); w2i = {w:i for i,w in enumerate(vocab)}
V, D, WIN = len(vocab), 12, 2

pairs = [(w2i[corpus[i]], w2i[corpus[j]])
         for i in range(len(corpus))
         for j in range(max(0,i-WIN), min(len(corpus), i+WIN+1)) if i != j]

Ein  = rng.normal(0, 0.1, (V, D))
Eout = rng.normal(0, 0.1, (V, D))
sigmoid = lambda z: 1/(1+np.exp(-np.clip(z, -30, 30)))

for epoch in range(400):
    for c, o in rng.permutation(pairs):
        negs = rng.integers(0, V, 5)
        targets = np.r_[o, negs]
        labels  = np.r_[1.0, np.zeros(5)]
        v, U = Ein[c], Eout[targets]
        # TODO: p = sigmoid(U @ v); update Ein[c] and Eout[targets] with lr=0.05
        pass

E = Ein / (np.linalg.norm(Ein, axis=1, keepdims=True) + 1e-9)
for w in ["cat", "sat", "the"]:
    sims = E @ E[w2i[w]]
    top = np.argsort(-sims)[1:4]
    print(f"{w:8s} -> {[(vocab[i], round(float(sims[i]),3)) for i in top]}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

corpus = ("the cat sat on the mat . the dog sat on the rug . "
          "the cat chased the mouse . the dog chased the cat . "
          "a mouse ate the cheese . the cat ate the mouse .").split()
vocab = sorted(set(corpus)); w2i = {w:i for i,w in enumerate(vocab)}
V, D, WIN = len(vocab), 12, 2

pairs = [(w2i[corpus[i]], w2i[corpus[j]])
         for i in range(len(corpus))
         for j in range(max(0,i-WIN), min(len(corpus), i+WIN+1)) if i != j]

Ein  = rng.normal(0, 0.1, (V, D))
Eout = rng.normal(0, 0.1, (V, D))
sigmoid = lambda z: 1/(1+np.exp(-np.clip(z, -30, 30)))

for epoch in range(400):
    for c, o in rng.permutation(pairs):
        negs = rng.integers(0, V, 5)
        targets = np.r_[o, negs]
        labels  = np.r_[1.0, np.zeros(5)]
        v, U = Ein[c], Eout[targets]
        p = sigmoid(U @ v)
        g = p - labels
        Ein[c]       -= 0.05 * (g @ U)
        Eout[targets] -= 0.05 * np.outer(g, v)

E = Ein / (np.linalg.norm(Ein, axis=1, keepdims=True) + 1e-9)
for w in ["cat", "sat", "the"]:
    sims = E @ E[w2i[w]]
    top = np.argsort(-sims)[1:4]
    print(f"{w:8s} -> {[(vocab[i], round(float(sims[i]),3)) for i in top]}")`,
  explain: 'Negative sampling turns an expensive $V$-way softmax into $k+1$ binary classifications, which is what made word2vec trainable on billions of words. On this tiny corpus expect noisy results — but "cat"/"dog" and "sat"/"ate" should pull together.',
},

};

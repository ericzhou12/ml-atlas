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
  title: 'Send a gradient back through 30 layers and see which activations survive',
  prompt: `The lesson's claim was that you should judge an activation by its derivative, because backprop
multiplies by that derivative once per layer. Measure it.

1. Fill in \`backward_through\`, which pushes a gradient backwards through a stack of layers using
   [equation (2)](#/l/nn-backprop): at each layer, multiply by $W^{\\mathsf T}$ and then elementwise by
   $\\phi'(\\mathbf{z})$. Report the size of the gradient arriving at layer 0.
2. Run it for sigmoid, tanh, and ReLU at 30 layers deep. **Predict the ordering, and roughly how many orders of
   magnitude separate them, before running.**
3. The last block manufactures dead ReLUs with an over-large step and counts them.`,
  hint: 'Work backwards through the list of layers. The incoming gradient `g` becomes `(g @ W.T) * dphi(z)` for the layer below, where `z` is that layer\'s cached pre-activation.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def sigmoid(x): return 1/(1+np.exp(-np.clip(x, -500, 500)))
ACT = {
    "sigmoid": (sigmoid,          lambda z: sigmoid(z)*(1-sigmoid(z))),
    "tanh":    (np.tanh,          lambda z: 1 - np.tanh(z)**2),
    "relu":    (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
}

def forward(name, L, width=128, gain=1.0):
    """Run a random input through L layers, caching each pre-activation."""
    f, _ = ACT[name]
    r = np.random.default_rng(1)
    Ws = [r.normal(0, gain/np.sqrt(width), (width, width)) for _ in range(L)]
    h, zs = r.normal(size=(64, width)), []
    for W in Ws:
        z = h @ W
        zs.append(z)
        h = f(z)
    return Ws, zs

def backward_through(name, Ws, zs):
    """Push a gradient of ones from the top back to layer 0.
       Return the average magnitude of the gradient arriving at layer 0."""
    _, df = ACT[name]
    g = np.ones_like(zs[-1])
    # TODO: walk the layers from last to first, applying equation (2)
    return np.abs(g).mean()

print("size of the gradient arriving at layer 0:")
print(f"{'depth':>6} {'sigmoid':>12} {'tanh':>12} {'relu':>12}")
for L in [1, 5, 10, 20, 30]:
    row = []
    for name in ("sigmoid", "tanh", "relu"):
        Ws, zs = forward(name, L)
        row.append(backward_through(name, Ws, zs))
    print(f"{L:6d} {row[0]:12.3e} {row[1]:12.3e} {row[2]:12.3e}")

Ws, zs = forward("sigmoid", 30); g_sig = backward_through("sigmoid", Ws, zs)
Ws, zs = forward("relu", 30);    g_rel = backward_through("relu", Ws, zs)
assert g_rel > 1e5 * g_sig, "at 30 layers ReLU should deliver a vastly larger gradient than sigmoid"
print(f"\\nat 30 layers, relu delivers {g_rel/g_sig:.3e} times more gradient than sigmoid")
print("PASS\\n")

# ---------- dead ReLUs: one oversized step ----------
W = rng.normal(0, 0.5, (256, 256))
h = rng.normal(0, 1, (512, 256))
z0 = h @ W
print(f"before any update:  sparsity {(z0 <= 0).mean():.1%}   never-firing units {(z0.max(0) <= 0).mean():.1%}")
print("\\nafter a single bias update of the given size:")
for size in [1, 2, 4, 8, 16, 32]:
    b = -size * rng.normal(1.0, 0.6, 256)
    z = z0 + b
    print(f"  step {size:3d}:  sparsity {(z <= 0).mean():.1%}   permanently dead {(z.max(0) <= 0).mean():.1%}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def sigmoid(x): return 1/(1+np.exp(-np.clip(x, -500, 500)))
ACT = {
    "sigmoid": (sigmoid,          lambda z: sigmoid(z)*(1-sigmoid(z))),
    "tanh":    (np.tanh,          lambda z: 1 - np.tanh(z)**2),
    "relu":    (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
}

def forward(name, L, width=128, gain=1.0):
    f, _ = ACT[name]
    r = np.random.default_rng(1)
    Ws = [r.normal(0, gain/np.sqrt(width), (width, width)) for _ in range(L)]
    h, zs = r.normal(size=(64, width)), []
    for W in Ws:
        z = h @ W
        zs.append(z)
        h = f(z)
    return Ws, zs

def backward_through(name, Ws, zs):
    _, df = ACT[name]
    g = np.ones_like(zs[-1])
    for i in reversed(range(len(Ws))):
        g = g * df(zs[i])
        g = g @ Ws[i].T
    return np.abs(g).mean()

print("size of the gradient arriving at layer 0:")
print(f"{'depth':>6} {'sigmoid':>12} {'tanh':>12} {'relu':>12}")
for L in [1, 5, 10, 20, 30]:
    row = []
    for name in ("sigmoid", "tanh", "relu"):
        Ws, zs = forward(name, L)
        row.append(backward_through(name, Ws, zs))
    print(f"{L:6d} {row[0]:12.3e} {row[1]:12.3e} {row[2]:12.3e}")

Ws, zs = forward("sigmoid", 30); g_sig = backward_through("sigmoid", Ws, zs)
Ws, zs = forward("relu", 30);    g_rel = backward_through("relu", Ws, zs)
assert g_rel > 1e5 * g_sig
print(f"\\nat 30 layers, relu delivers {g_rel/g_sig:.3e} times more gradient than sigmoid")
print("PASS\\n")

# ---------- dead ReLUs: one oversized step ----------
W = rng.normal(0, 0.5, (256, 256))
h = rng.normal(0, 1, (512, 256))
z0 = h @ W
print(f"before any update:  sparsity {(z0 <= 0).mean():.1%}   never-firing units {(z0.max(0) <= 0).mean():.1%}")
print("\\nafter a single bias update of the given size:")
for size in [1, 2, 4, 8, 16, 32]:
    b = -size * rng.normal(1.0, 0.6, 256)
    z = z0 + b
    print(f"  step {size:3d}:  sparsity {(z <= 0).mean():.1%}   permanently dead {(z.max(0) <= 0).mean():.1%}")`,
  explain: `Part 1 is the whole lesson in one table. Every row multiplies by one more copy of $\\phi'$, and the
columns diverge at exactly the rate their derivatives predict. Sigmoid's derivative peaks at 0.25 and is
typically well below that, so thirty layers of it multiply the gradient by something like $0.2^{30}$ — the
number arriving at layer 0 is astronomically small, and that layer is frozen at its initial random values no
matter how long you train. ReLU's derivative is exactly 1 wherever the unit is active, so it contributes nothing
to the decay at all. tanh sits in between, which is why it was the standard for a decade: better than sigmoid,
still not enough.

This is not a subtle effect you would have to squint at a loss curve to notice. It is the difference between a
network that trains and one that does not, and it was the state of the art blocker for roughly twenty years.

Part 2 shows dying ReLU as a threshold effect. At rest the layer is about 50% sparse, which is healthy and is
what ReLU is supposed to do. A modest bias update raises the sparsity without killing anything. But past a
certain step size units start going silent for *every* input in the batch — 25% of the layer at step 16, and
two thirds at step 32 — and those units now receive exactly zero gradient forever. No future step can
revive them.

Note the shape of that curve. It is not gradual damage that you could notice and back away from; almost nothing
happens and then a large fraction of the layer is gone at once. This is why a single learning-rate spike can
quietly cost a network much of its capacity, and why the fix is prevention — warmup, gradient clipping, or an
activation like GELU that keeps a gradient path open on the negative side.`,
},
'nn-initialization': {
  title: 'Find the variance-preserving scale by search, and rediscover the factor of 2',
  prompt: `The lesson derived $\\sigma^2 = 2/n$ for ReLU and $1/n$ without it. Do not take that on trust — find both
numbers by experiment.

1. Write \`best_std\`, which searches a range of initialization scales and returns the one whose activations are
   the same size at layer 25 as at layer 1.
2. Run it once with ReLU and once with no activation at all. Compare each answer against $\\sqrt{2/n}$ and
   $\\sqrt{1/n}$. **The ratio between your two answers is the thing to look at.**
3. The last block shows what being slightly wrong costs, by depth.`,
  hint: 'For each candidate `std`, run `propagate` and score it by how far `stds[-1]/stds[0]` is from 1. A log-spaced grid, `np.logspace(-2.2, -0.7, 60)`, covers the useful range.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def propagate(std, depth=25, width=256, act="relu"):
    """Push a unit-variance input through depth layers; return each layer's std."""
    h = rng.normal(0, 1, (128, width))
    stds = []
    for _ in range(depth):
        W = rng.normal(0, std, (width, width))
        z = h @ W
        h = np.maximum(0, z) if act == "relu" else z      # "linear" = no activation at all
        stds.append(h.std())
    return np.array(stds)

n = 256
print(f"{'scheme':10s} {'std':>9} " + "".join(f"{'L'+str(d):>12}" for d in [1, 5, 10, 25]))
for name, std in [("tiny", 0.01), ("xavier", np.sqrt(1/n)), ("he", np.sqrt(2/n)), ("big", 1.0)]:
    s = propagate(std)
    print(f"{name:10s} {std:9.5f} " + "".join(f"{s[d-1]:12.3e}" for d in [1, 5, 10, 25]))

def best_std(act, width=256):
    """Return the initialization std whose signal is the same size at the top as the bottom."""
    # TODO: score each candidate by |stds[-1]/stds[0] - 1| and keep the best
    return 0.0

for act, formula, name in [("relu",   np.sqrt(2/n), "sqrt(2/n)  (He)"),
                           ("linear", np.sqrt(1/n), "sqrt(1/n)  (Xavier)")]:
    found = best_std(act)
    print(f"\\n{act}:  search found std = {found:.5f}")
    print(f"        the formula says   {formula:.5f}   ({name})")
    assert abs(found - formula) / formula < 0.25, f"the search for {act} is off"

print(f"\\nratio between the two answers: {best_std('relu')/best_std('linear'):.3f}"
      f"   (sqrt(2) = {np.sqrt(2):.3f})")
print("PASS\\n")

# ---------- what does being slightly wrong cost? ----------
print("activation std at the top of the stack, when the scale is off by 10%:")
print(f"{'depth':>7} {'exactly right':>15} {'10% too small':>15} {'10% too big':>14}")
for depth in [5, 10, 25, 50]:
    a = propagate(np.sqrt(2/n),        depth)[-1]
    b = propagate(0.9*np.sqrt(2/n),    depth)[-1]
    c = propagate(1.1*np.sqrt(2/n),    depth)[-1]
    print(f"{depth:7d} {a:15.3e} {b:15.3e} {c:14.3e}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def propagate(std, depth=25, width=256, act="relu"):
    h = rng.normal(0, 1, (128, width))
    stds = []
    for _ in range(depth):
        W = rng.normal(0, std, (width, width))
        z = h @ W
        h = np.maximum(0, z) if act == "relu" else z      # "linear" = no activation at all
        stds.append(h.std())
    return np.array(stds)

n = 256
print(f"{'scheme':10s} {'std':>9} " + "".join(f"{'L'+str(d):>12}" for d in [1, 5, 10, 25]))
for name, std in [("tiny", 0.01), ("xavier", np.sqrt(1/n)), ("he", np.sqrt(2/n)), ("big", 1.0)]:
    s = propagate(std)
    print(f"{name:10s} {std:9.5f} " + "".join(f"{s[d-1]:12.3e}" for d in [1, 5, 10, 25]))

def best_std(act, width=256):
    best = (np.inf, 0.0)
    for std in np.logspace(-2.2, -0.7, 60):
        s = propagate(std, act=act, width=width)
        if not np.all(np.isfinite(s)) or s[0] == 0:
            continue
        score = abs(s[-1]/s[0] - 1.0)
        if score < best[0]:
            best = (score, std)
    return best[1]

for act, formula, name in [("relu",   np.sqrt(2/n), "sqrt(2/n)  (He)"),
                           ("linear", np.sqrt(1/n), "sqrt(1/n)  (Xavier)")]:
    found = best_std(act)
    print(f"\\n{act}:  search found std = {found:.5f}")
    print(f"        the formula says   {formula:.5f}   ({name})")
    assert abs(found - formula) / formula < 0.25, f"the search for {act} is off"

print(f"\\nratio between the two answers: {best_std('relu')/best_std('linear'):.3f}"
      f"   (sqrt(2) = {np.sqrt(2):.3f})")
print("PASS\\n")

print("activation std at the top of the stack, when the scale is off by 10%:")
print(f"{'depth':>7} {'exactly right':>15} {'10% too small':>15} {'10% too big':>14}")
for depth in [5, 10, 25, 50]:
    a = propagate(np.sqrt(2/n),        depth)[-1]
    b = propagate(0.9*np.sqrt(2/n),    depth)[-1]
    c = propagate(1.1*np.sqrt(2/n),    depth)[-1]
    print(f"{depth:7d} {a:15.3e} {b:15.3e} {c:14.3e}")`,
  explain: `Part 1: the search never sees a formula. It tries scales and keeps whichever one leaves the signal the
same size after twenty-five layers — and it lands on $\\sqrt{2/n}$ with ReLU and $\\sqrt{1/n}$ with no activation,
which are He and Xavier, recovered from nothing but a measurement. The ratio between the two comes out at about
$\\sqrt2$, and it is there for exactly the reason the derivation gave: ReLU discards half the signal, so you must
start with twice the variance.

Worth knowing why the control uses no activation rather than tanh, since Xavier is usually stated for tanh. The
derivation assumes the activation is roughly a straight line near zero, which tanh only is while the signal stays
small. Run the same search with \`np.tanh\` and it returns a noticeably larger number, because tanh squashes as
well as passes, and the extra scale is compensating for that squashing. Xavier's $1/n$ is the *linear* answer, and
tanh borrows it on the assumption that it is behaving linearly.

Part 2 is why the exponent matters more than the value. A 10% error in the scale is invisible at depth 5 and has
moved the signal by orders of magnitude by depth 50, because the error is applied once per layer and compounds.
That is the whole reason this has a formula rather than a rule of thumb — "close enough" is not a category that
survives being raised to the fiftieth power.`,
},

'nn-normalization': {
  title: 'Implement all three, then reproduce the bug that makes BatchNorm annoying',
  prompt: `1. Write \`batchnorm\`, \`layernorm\` and \`rmsnorm\`. They are the same two lines each; only the axis
   changes. The printout shows which statistic each one actually zeroes.
2. **Example coupling.** Scale one row of the batch by 10 and check whether the *other* rows' outputs changed.
   The lesson claimed BatchNorm couples examples together and LayerNorm does not — this is that claim, tested.
3. **Train versus eval.** The last block runs the same example through BatchNorm in three settings: alone,
   inside a batch of similar examples, and inside a batch of unusual ones. **Predict whether the three agree
   before you run it.** Then see what running statistics fix, and what they do not.`,
  hint: 'BatchNorm reduces over axis 0 (down the columns, one statistic per feature); LayerNorm and RMSNorm reduce over axis -1 (along each row, one statistic per example). RMSNorm skips the mean subtraction entirely.',
  starter: `import numpy as np
rng = np.random.default_rng(0)
x = rng.normal(2.0, 3.0, (8, 16))          # (batch, features)

def batchnorm(x, eps=1e-5):  return x   # TODO
def layernorm(x, eps=1e-5):  return x   # TODO
def rmsnorm(x, eps=1e-5):    return x   # TODO

print(f"{'':11s} {'per-feature mean':>22s} {'per-example std':>20s}")
for name, f in [("input", lambda z: z), ("batchnorm", batchnorm),
                ("layernorm", layernorm), ("rmsnorm", rmsnorm)]:
    y = f(x)
    print(f"{name:11s} {str(y.mean(0)[:3].round(3)):>22s} {str(y.std(-1)[:3].round(3)):>20s}")

assert np.allclose(batchnorm(x).mean(0), 0, atol=1e-6), "batchnorm should zero the per-feature means"
assert np.allclose(layernorm(x).mean(-1), 0, atol=1e-6), "layernorm should zero each example's own mean"
assert np.allclose(np.sqrt((rmsnorm(x)**2).mean(-1)), 1, atol=1e-5), "rmsnorm should give each row unit RMS"

# ---------- 2. does one example's value affect another's output? ----------
x2 = x.copy(); x2[0] *= 10                 # change ONLY row 0
print("\\nafter scaling row 0 by 10, is row 1's output unchanged?")
for name, f in [("batchnorm", batchnorm), ("layernorm", layernorm), ("rmsnorm", rmsnorm)]:
    print(f"  {name:11s} {np.allclose(f(x)[1], f(x2)[1], atol=1e-6)}")

# ---------- 3. the same example, three different batches ----------
example  = rng.normal(2.0, 3.0, (1, 16))
ordinary = np.vstack([example, rng.normal(2.0, 3.0, (15, 16))])
unusual  = np.vstack([example, rng.normal(9.0, 0.5, (15, 16))])

print("\\nthe SAME example, normalized three ways (first 3 features):")
print(f"  alone in a batch of 1     : {batchnorm(example)[0][:3].round(3)}")
print(f"  in a batch of similar rows: {batchnorm(ordinary)[0][:3].round(3)}")
print(f"  in a batch of odd rows    : {batchnorm(unusual)[0][:3].round(3)}")
print(f"  under LayerNorm, any batch: {layernorm(example)[0][:3].round(3)}")

# what a framework does at inference: freeze statistics from training
mu, var = ordinary.mean(0), ordinary.var(0)
frozen = (example - mu) / np.sqrt(var + 1e-5)
print(f"  BatchNorm with frozen stats: {frozen[0][:3].round(3)}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)
x = rng.normal(2.0, 3.0, (8, 16))

def batchnorm(x, eps=1e-5):
    return (x - x.mean(0, keepdims=True)) / np.sqrt(x.var(0, keepdims=True) + eps)

def layernorm(x, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)

def rmsnorm(x, eps=1e-5):
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps)

print(f"{'':11s} {'per-feature mean':>22s} {'per-example std':>20s}")
for name, f in [("input", lambda z: z), ("batchnorm", batchnorm),
                ("layernorm", layernorm), ("rmsnorm", rmsnorm)]:
    y = f(x)
    print(f"{name:11s} {str(y.mean(0)[:3].round(3)):>22s} {str(y.std(-1)[:3].round(3)):>20s}")

assert np.allclose(batchnorm(x).mean(0), 0, atol=1e-6)
assert np.allclose(layernorm(x).mean(-1), 0, atol=1e-6)
assert np.allclose(np.sqrt((rmsnorm(x)**2).mean(-1)), 1, atol=1e-5)

x2 = x.copy(); x2[0] *= 10
print("\\nafter scaling row 0 by 10, is row 1's output unchanged?")
for name, f in [("batchnorm", batchnorm), ("layernorm", layernorm), ("rmsnorm", rmsnorm)]:
    print(f"  {name:11s} {np.allclose(f(x)[1], f(x2)[1], atol=1e-6)}")

example  = rng.normal(2.0, 3.0, (1, 16))
ordinary = np.vstack([example, rng.normal(2.0, 3.0, (15, 16))])
unusual  = np.vstack([example, rng.normal(9.0, 0.5, (15, 16))])

print("\\nthe SAME example, normalized three ways (first 3 features):")
print(f"  alone in a batch of 1     : {batchnorm(example)[0][:3].round(3)}")
print(f"  in a batch of similar rows: {batchnorm(ordinary)[0][:3].round(3)}")
print(f"  in a batch of odd rows    : {batchnorm(unusual)[0][:3].round(3)}")
print(f"  under LayerNorm, any batch: {layernorm(example)[0][:3].round(3)}")

mu, var = ordinary.mean(0), ordinary.var(0)
frozen = (example - mu) / np.sqrt(var + 1e-5)
print(f"  BatchNorm with frozen stats: {frozen[0][:3].round(3)}")`,
  explain: `Part 2 is example coupling, demonstrated. Nothing about row 1 changed, yet BatchNorm's output for row 1
did — because row 0 moved the column means that row 1 was measured against. LayerNorm and RMSNorm are unaffected,
because each row is normalized entirely by its own numbers. That single difference is why transformers use
LayerNorm: with variable-length sequences and padding, having one example's output depend on its batch-mates is
not a quirk to tolerate, it is a correctness problem.

Part 3 is the bug that catches everyone. The same example produces **three different outputs** depending on what
happened to be batched alongside it — and look at the batch-of-one row: with a single example the per-feature
variance is zero, so BatchNorm divides by $\\sqrt{\\epsilon}$ and returns essentially nothing at all. A model that
worked in training now returns garbage when you serve one request at a time.

The last line is the fix frameworks use: freeze the statistics collected during training and use those at
inference, so the function stops depending on the batch. But notice what that means — the layer computes a
*different function* in training mode and in evaluation mode. Forgetting to switch modes gives you a model that
silently scores worse for no visible reason, and it is one of the most common bugs in practice. LayerNorm has
none of this: one row of numbers goes in, the same row comes out, always.`,
},

'nn-regularization': {
  title: 'Inverted dropout, and testing the "implicit ensemble" claim',
  prompt: `1. Implement inverted dropout and confirm the expected activation is preserved. Then confirm that
   forgetting the $1/(1-p)$ scaling shifts the mean — the classic bug.
2. **Is dropout really an ensemble?** The lesson said running the full network at inference *approximates
   averaging over all the thinned subnetworks*. Test it: push one input through a small network hundreds of
   times with different random masks, average the answers, and compare against the single full-network answer.
3. The last block shows what happens if you leave dropout switched on at inference.`,
  hint: 'The mask is `(rng.random(x.shape) > p)`, which keeps a fraction $1-p$ of the units. Dividing it by $1-p$ raises the survivors so the layer\'s expected output is unchanged.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def dropout(x, p, training=True, scale=True):
    if not training or p == 0: return x
    mask = (rng.random(x.shape) > p).astype(float)
    # TODO: divide the mask by (1-p) when scale=True, then apply it
    return x * mask

x = np.ones((2000, 64))
print("mean activation during training (should stay at 1.0):")
for p in [0.0, 0.2, 0.5, 0.8]:
    print(f"  p={p:.1f}   inverted {dropout(x, p).mean():.4f}      "
          f"without the scaling {dropout(x, p, scale=False).mean():.4f}")
assert abs(dropout(x, 0.5).mean() - 1.0) < 0.02, "the expectation is not preserved"
print("PASS\\n")

# ---------- 2. the ensemble claim ----------
W1 = rng.normal(0, 0.5, (12, 256))
W2 = np.abs(rng.normal(0, 0.3, (256, 1)))    # positive, so the answer is not near zero
inp = rng.normal(size=(1, 12))
P = 0.3

def net(inp, use_dropout):
    h = np.maximum(0, inp @ W1)
    h = dropout(h, P, training=use_dropout)
    return float((h @ W2).ravel()[0])

full = net(inp, use_dropout=False)                 # everything present, no masking
print(f"the full network's answer:        {full:.4f}")
for n in [1, 10, 100, 1000, 10000]:
    avg = np.mean([net(inp, use_dropout=True) for _ in range(n)])
    print(f"  average over {n:6d} thinned nets: {avg:9.4f}    off by {100*abs(avg-full)/full:5.2f}%")

# ---------- 3. leaving dropout on at inference ----------
print("\\nsame input, dropout accidentally left ON, five separate calls:")
print("  ", [round(net(inp, use_dropout=True), 3) for _ in range(5)])
print("  with dropout correctly off:", [round(net(inp, use_dropout=False), 3) for _ in range(5)])`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def dropout(x, p, training=True, scale=True):
    if not training or p == 0: return x
    mask = (rng.random(x.shape) > p).astype(float)
    if scale: mask = mask / (1 - p)
    return x * mask

x = np.ones((2000, 64))
print("mean activation during training (should stay at 1.0):")
for p in [0.0, 0.2, 0.5, 0.8]:
    print(f"  p={p:.1f}   inverted {dropout(x, p).mean():.4f}      "
          f"without the scaling {dropout(x, p, scale=False).mean():.4f}")
assert abs(dropout(x, 0.5).mean() - 1.0) < 0.02
print("PASS\\n")

W1 = rng.normal(0, 0.5, (12, 256))
W2 = np.abs(rng.normal(0, 0.3, (256, 1)))    # positive, so the answer is not near zero
inp = rng.normal(size=(1, 12))
P = 0.3

def net(inp, use_dropout):
    h = np.maximum(0, inp @ W1)
    h = dropout(h, P, training=use_dropout)
    return float((h @ W2).ravel()[0])

full = net(inp, use_dropout=False)
print(f"the full network's answer:        {full:.4f}")
for n in [1, 10, 100, 1000, 10000]:
    avg = np.mean([net(inp, use_dropout=True) for _ in range(n)])
    print(f"  average over {n:6d} thinned nets: {avg:9.4f}    off by {100*abs(avg-full)/full:5.2f}%")

print("\\nsame input, dropout accidentally left ON, five separate calls:")
print("  ", [round(net(inp, use_dropout=True), 3) for _ in range(5)])
print("  with dropout correctly off:", [round(net(inp, use_dropout=False), 3) for _ in range(5)])`,
  explain: `Part 1: without the $1/(1-p)$ factor the training-time mean drops to exactly $1-p$, so at $p=0.8$ the
next layer sees activations five times smaller during training than at inference. The scaling exists purely so
that the two match and inference needs no special case at all.

Part 2 is the ensemble claim, tested. A single thinned subnetwork is off by a few percent; average ten thousand
of them and the answer lands within a fraction of a percent of what the full network computes in one pass. The
error shrinks at the $1/\\sqrt{n}$ rate you would expect for averaging away noise. So running the
full network really is a cheap stand-in for averaging over an exponential number of subnetworks, and you get the
variance reduction of an ensemble for the cost of one forward pass. (This is worth knowing in the other
direction too: deliberately keeping dropout on and averaging a few dozen samples, "Monte Carlo dropout", gives
you a rough uncertainty estimate from a model that was never designed to produce one.)

Part 3 is the failure mode. With dropout left on, the same input gives a different answer every call — the model
is not broken, it is returning one random ensemble member instead of the average. In a framework this is what
forgetting \`model.eval()\` does, and the symptom is a model that scores worse than it did in training and is
non-deterministic for no apparent reason.`,
},

'nn-losses-training': {
  title: 'Watch the loss choose the answer, then run the two diagnostics',
  prompt: `1. **The loss decides what you predict.** On deliberately skewed data — most customers spend nothing, a
   few spend a lot — find the single number that minimises squared error and the single number that minimises
   absolute error, by brute-force search. Compare each against the mean and the median. **Predict which is
   which before running.**
2. **Check the initial loss.** A $K$-class model that has learned nothing should start at $\\log K$. Anything
   else means the head or the labels are wrong.
3. **Overfit one batch.** Train on 8 examples until the loss is essentially zero. If it will not go to zero,
   you have a bug, not a tuning problem — this catches more errors than any other single check.`,
  hint: 'For part 1 you are not fitting a model, just searching over candidate constants $c$ and scoring each by $\\sum_i (y_i-c)^2$ or $\\sum_i |y_i-c|$. For part 3, the gradients are already written for you — just loop.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

# ---------- 1. what does each loss actually report? ----------
spend = np.where(rng.random(4000) < 0.75, 0.0, rng.exponential(160, 4000))
grid = np.linspace(0, 200, 20001)

def best_constant(y, kind):
    """Search the grid for the single number minimising this loss."""
    # TODO: score every candidate in \`grid\` and return the best one
    return 0.0

c_sq  = best_constant(spend, "squared")
c_abs = best_constant(spend, "absolute")
print(f"customers who spend nothing: {(spend == 0).mean():.0%}")
print(f"  minimiser of squared error : {c_sq:8.2f}    the mean is   {spend.mean():8.2f}")
print(f"  minimiser of absolute error: {c_abs:8.2f}    the median is {np.median(spend):8.2f}")
assert abs(c_sq - spend.mean()) < 0.5,      "squared error should land on the mean"
assert abs(c_abs - np.median(spend)) < 0.5, "absolute error should land on the median"
print("PASS\\n")

# ---------- 2 and 3: the two diagnostics ----------
K, D = 10, 20
X = rng.normal(size=(512, D)); y = rng.integers(0, K, 512)
X[np.arange(512), y % D] += 2.0                        # make it learnable

W1 = rng.normal(0, np.sqrt(2/D), (D, 32)); b1 = np.zeros(32)
W2 = rng.normal(0, np.sqrt(2/32), (32, K)); b2 = np.zeros(K)   # a normally-scaled head

def loss_and_grads(X, y):
    h = np.maximum(0, X @ W1 + b1)
    logits = h @ W2 + b2
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    d = p.copy(); d[np.arange(len(y)), y] -= 1; d /= len(y)
    dh = (d @ W2.T) * (h > 0)
    return loss, (X.T @ dh, dh.sum(0), h.T @ d, d.sum(0))

l0, _ = loss_and_grads(X, y)
print(f"CHECK 1  initial loss with a He-scaled head: {l0:.4f}")
print(f"         but log({K}) = {np.log(K):.4f} -- the model starts out opinionated")

W2 = rng.normal(0, 0.01, (32, K))               # shrink the output layer instead
l1, _ = loss_and_grads(X, y)
print(f"         initial loss with a small head:    {l1:.4f}   "
      f"{'PASS' if abs(l1-np.log(K)) < 0.05 else 'FAIL'}")
assert abs(l1 - np.log(K)) < 0.05, "with a small head the initial loss must be log K"

# CHECK 2: TODO -- train on X[:8], y[:8] for 800 steps at lr=0.3, then print the loss
print("\\nCHECK 2  overfit-one-batch: implement the loop")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

spend = np.where(rng.random(4000) < 0.75, 0.0, rng.exponential(160, 4000))
grid = np.linspace(0, 200, 20001)

def best_constant(y, kind):
    if kind == "squared":
        scores = ((y[None, :] - grid[:, None])**2).mean(1)
    else:
        scores = np.abs(y[None, :] - grid[:, None]).mean(1)
    return float(grid[scores.argmin()])

c_sq  = best_constant(spend, "squared")
c_abs = best_constant(spend, "absolute")
print(f"customers who spend nothing: {(spend == 0).mean():.0%}")
print(f"  minimiser of squared error : {c_sq:8.2f}    the mean is   {spend.mean():8.2f}")
print(f"  minimiser of absolute error: {c_abs:8.2f}    the median is {np.median(spend):8.2f}")
assert abs(c_sq - spend.mean()) < 0.5
assert abs(c_abs - np.median(spend)) < 0.5
print("PASS\\n")

K, D = 10, 20
X = rng.normal(size=(512, D)); y = rng.integers(0, K, 512)
X[np.arange(512), y % D] += 2.0

W1 = rng.normal(0, np.sqrt(2/D), (D, 32)); b1 = np.zeros(32)
W2 = rng.normal(0, np.sqrt(2/32), (32, K)); b2 = np.zeros(K)   # a normally-scaled head

def loss_and_grads(X, y):
    h = np.maximum(0, X @ W1 + b1)
    logits = h @ W2 + b2
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    d = p.copy(); d[np.arange(len(y)), y] -= 1; d /= len(y)
    dh = (d @ W2.T) * (h > 0)
    return loss, (X.T @ dh, dh.sum(0), h.T @ d, d.sum(0))

l0, _ = loss_and_grads(X, y)
print(f"CHECK 1  initial loss with a He-scaled head: {l0:.4f}")
print(f"         but log({K}) = {np.log(K):.4f} -- the model starts out opinionated")

W2 = rng.normal(0, 0.01, (32, K))               # shrink the output layer instead
l1, _ = loss_and_grads(X, y)
print(f"         initial loss with a small head:    {l1:.4f}   "
      f"{'PASS' if abs(l1-np.log(K)) < 0.05 else 'FAIL'}")
assert abs(l1 - np.log(K)) < 0.05, "with a small head the initial loss must be log K"

Xs, ys = X[:8], y[:8]
for t in range(800):
    l, (g1, gb1, g2, gb2) = loss_and_grads(Xs, ys)
    W1 -= 0.3*g1; b1 -= 0.3*gb1; W2 -= 0.3*g2; b2 -= 0.3*gb2
print(f"\\nCHECK 2  loss on 8 examples after 800 steps: {l:.6f}   "
      f"{'PASS' if l < 0.01 else 'FAIL - you have a bug'}")`,
  explain: `Part 1: 75% of these customers spend nothing, so the median spend is **0** and the mean is around 40.
The brute-force search confirms that squared error is minimised at the mean and absolute error at the median,
exactly as the derivation said — and note that no model was involved, only a choice of scoring rule.

Now read the two numbers as business answers. A model trained with squared error predicts about £40 for
everyone, a figure almost no individual customer resembles, but one that multiplies up correctly for revenue
forecasting. A model trained with absolute error predicts £0, which describes a typical customer accurately and
forecasts zero total revenue. Neither is broken. You chose which question to answer when you chose the loss, and
on skewed data — most real business data — the two answers are not close.

Part 2 is more interesting than it first looks. With a normally scaled output layer the initial loss is about
**3.20**, not $\\log 10 = 2.30$ — and a loss *above* $\\log K$ means the model starts out confidently wrong. It has
opinions, drawn from random numbers, and the first thing training has to do is beat them out of it. Shrink the
output layer's initialization and the loss lands exactly on $\\log K$: no opinion at all, which is what an
untrained model should have. This is why the last layer of a classifier is often initialized small, and why "is
my initial loss $\\log K$?" is a real diagnostic and not just a sanity formula. A value far *below* $\\log K$ would
be worse news still — it would mean label information is leaking into the input.

Part 3 is the check that catches the most bugs. Driving the loss on eight examples to essentially zero proves
the forward pass, the backward pass, and the update are all connected and pointing the same way. If that fails,
you have a bug, and every hour spent tuning the learning rate is wasted.`,
},

'nn-cnn': {
  title: 'Convolve by hand, then test whether equivariance is real',
  prompt: `1. Implement 2-D convolution with stride and padding, and confirm a Sobel kernel responds at edges and
   nowhere else.
2. **Equivariance, tested.** The lesson claimed that shifting the input shifts the output by exactly the same
   amount, and that a dense layer has no such property. Shift an image by 3 pixels, convolve both versions, and
   check whether one output is a shifted copy of the other. Then do the same for a dense layer.
3. Read the parameter counts, and the receptive-field table that shows what weight sharing costs you.`,
  hint: 'Frameworks implement cross-correlation (no kernel flip) — do the same: for each output position, multiply the window by the kernel and sum. For part 2, compare `conv(shifted)` against `shift(conv(original))` on the region where both are valid.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def conv2d(x, k, stride=1, pad=0):
    if pad: x = np.pad(x, pad)
    kh, kw = k.shape
    H = (x.shape[0]-kh)//stride + 1
    W = (x.shape[1]-kw)//stride + 1
    out = np.zeros((H, W))
    # TODO: for each output position, dot the window with the kernel
    return out

img = np.zeros((12, 12)); img[3:9, 3:9] = 1.0
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float)

r = conv2d(img, sobel_x, pad=1)
print("vertical-edge response:")
print(r.astype(int))
assert r.shape == (12, 12), "same-padding should preserve the size"
assert abs(r[6, 3]) > 2 and abs(r[6, 6]) < 1e-9, "should fire at the edges, not in the interior"
print("PASS\\n")

# ---------- 2. shift the input; does the output shift with it? ----------
S = 3
big = np.zeros((24, 24)); big[5:12, 5:12] = 1.0        # a square
moved = np.roll(big, S, axis=1)                        # the same square, 3 px right

a = conv2d(big,   sobel_x, pad=1)
b = conv2d(moved, sobel_x, pad=1)
same = np.allclose(b[:, S:], a[:, :-S])
print(f"convolution: output of the shifted image equals the shifted output? {same}")

# now a dense layer doing the same job: 576 inputs -> 576 outputs, random weights
W = rng.normal(0, 0.05, (576, 576))
da = (big.ravel()   @ W).reshape(24, 24)
db = (moved.ravel() @ W).reshape(24, 24)
shifted_match = np.abs(db[:, S:] - da[:, :-S]).mean()
print(f"dense layer: average mismatch between the two outputs = {shifted_match:.4f}")
print(f"             (for reference, the outputs themselves average {np.abs(da).mean():.4f})")
assert same, "convolution must be equivariant to shifts"
assert shifted_match > 0.1 * np.abs(da).mean(), "the dense layer should NOT be equivariant"
print("PASS\\n")

# ---------- 3. what weight sharing buys and what it costs ----------
cin, cout, k, h, w = 64, 128, 3, 56, 56
print(f"3x3 conv, {cin}->{cout} channels: {cin*cout*k*k + cout:>12,} parameters")
print(f"dense {h}x{w}x{cin} -> {cout}:    {h*w*cin*cout + cout:>12,} parameters")

print("\\nlayers of 3x3 convolution   how much of the input one neuron sees")
rf = 1
for L in range(1, 9):
    rf += 2
    print(f"{L:15d}   {rf:>12d} x {rf}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def conv2d(x, k, stride=1, pad=0):
    if pad: x = np.pad(x, pad)
    kh, kw = k.shape
    H = (x.shape[0]-kh)//stride + 1
    W = (x.shape[1]-kw)//stride + 1
    out = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            out[i, j] = (x[i*stride:i*stride+kh, j*stride:j*stride+kw] * k).sum()
    return out

img = np.zeros((12, 12)); img[3:9, 3:9] = 1.0
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float)

r = conv2d(img, sobel_x, pad=1)
print("vertical-edge response:")
print(r.astype(int))
assert r.shape == (12, 12)
assert abs(r[6, 3]) > 2 and abs(r[6, 6]) < 1e-9
print("PASS\\n")

S = 3
big = np.zeros((24, 24)); big[5:12, 5:12] = 1.0
moved = np.roll(big, S, axis=1)

a = conv2d(big,   sobel_x, pad=1)
b = conv2d(moved, sobel_x, pad=1)
same = np.allclose(b[:, S:], a[:, :-S])
print(f"convolution: output of the shifted image equals the shifted output? {same}")

W = rng.normal(0, 0.05, (576, 576))
da = (big.ravel()   @ W).reshape(24, 24)
db = (moved.ravel() @ W).reshape(24, 24)
shifted_match = np.abs(db[:, S:] - da[:, :-S]).mean()
print(f"dense layer: average mismatch between the two outputs = {shifted_match:.4f}")
print(f"             (for reference, the outputs themselves average {np.abs(da).mean():.4f})")
assert same
assert shifted_match > 0.1 * np.abs(da).mean()
print("PASS\\n")

cin, cout, k, h, w = 64, 128, 3, 56, 56
print(f"3x3 conv, {cin}->{cout} channels: {cin*cout*k*k + cout:>12,} parameters")
print(f"dense {h}x{w}x{cin} -> {cout}:    {h*w*cin*cout + cout:>12,} parameters")

print("\\nlayers of 3x3 convolution   how much of the input one neuron sees")
rf = 1
for L in range(1, 9):
    rf += 2
    print(f"{L:15d}   {rf:>12d} x {rf}")`,
  explain: `Part 2 is the claim that matters, and the two lines are as different as they could be. The convolution's
output for the shifted image is *exactly* the shifted output — \`allclose\` returns True, not approximately. The
dense layer's two outputs are unrelated: the mismatch is on the same scale as the outputs themselves, meaning the
shift produced a completely different answer.

That is the whole argument for convolution. The dense layer has no idea that pixel $(5,5)$ and pixel $(5,8)$ have
anything to do with each other, so it would have to see the square in every position to learn about it in every
position. The convolution learns "vertical edge" once and gets it everywhere for free — and the nine numbers in
\`sobel_x\` are the entire detector, unchanged as the square moves.

Part 3 is the price and the bill. Weight sharing turns 25 million parameters into 74 thousand, which is why this
is affordable at all. But look at the receptive-field table: after eight layers a neuron still sees only 17
pixels of the input, because each $3\\times3$ layer adds just 2. To recognise an object spanning 200 pixels you
would need about a hundred layers of plain convolution. This is the real reason vision networks are deep, and
why stride and pooling exist — they grow the receptive field by multiplying rather than adding.`,
},

'nn-rnn': {
  title: 'Measure how far back a gradient can reach, in both cell types',
  prompt: `Whether a model can learn a long-range dependency comes down to one number: how much gradient still
arrives at the early timesteps. Measure it.

1. Write \`rnn_reach(T, w)\` — run a vanilla RNN forward for $T$ steps, then walk the chain backwards
   multiplying by $w\\,\\phi'(z_t)$ at each step, and return what is left.
2. Write \`lstm_reach(T, bias)\` — the same for an LSTM cell state, where the backward factor at each step is
   just the forget gate $f = \\sigma(\\text{bias})$.
3. Read the two tables. Then look at the last block, which asks the practical question: **at each setting, how
   many steps back can a gradient travel before it drops below $10^{-4}$ and stops being usable?**`,
  hint: 'For the vanilla cell the backward factor at step $t$ is `w * (1 - tanh(z_t)**2)`, using the pre-activation you cached going forward. For the LSTM, every step contributes the same factor $\\sigma(\\text{bias})$, so the total is that number raised to the power $T$.',
  starter: `import numpy as np

def sigmoid(z): return 1/(1+np.exp(-z))

def rnn_reach(T, w):
    """How much of a gradient at step T survives back to step 0, in a vanilla RNN."""
    h, pres = 0.0, []
    for t in range(T):                       # forward, caching pre-activations
        z = w*h + (1.0 if t == 0 else 0.0)
        pres.append(z); h = np.tanh(z)
    # TODO: walk backwards, multiplying by w * tanh'(z_t) at each step
    return 1.0

def lstm_reach(T, bias):
    """The same, through an LSTM cell state whose forget gate is sigmoid(bias)."""
    # TODO: the backward factor at every step is the forget gate. Apply it T times.
    return 1.0

print("vanilla RNN: fraction of the gradient reaching step 0")
print(f"{'w':>6} " + "".join(f"{'T='+str(T):>12}" for T in [10, 30, 100, 300]))
for w in [0.5, 0.9, 1.0, 1.1]:
    print(f"{w:6.1f} " + "".join(f"{rnn_reach(T, w):12.2e}" for T in [10, 30, 100, 300]))

print("\\nLSTM cell state: the same measurement")
print(f"{'bias':>6} {'f':>7} " + "".join(f"{'T='+str(T):>12}" for T in [10, 30, 100, 300]))
for bias in [0.0, 1.0, 3.0, 6.0]:
    print(f"{bias:6.1f} {sigmoid(bias):7.3f} " +
          "".join(f"{lstm_reach(T, bias):12.2e}" for T in [10, 30, 100, 300]))

assert rnn_reach(100, 0.9) < 1e-5, "a vanilla RNN at w=0.9 should lose the gradient over 100 steps"
assert lstm_reach(100, 6.0) > 0.5, "a forget gate near 1 should preserve it"
print("PASS\\n")

# ---------- how far back can each one actually learn? ----------
def horizon(fn, arg, limit=1e-4, cap=2000):
    T = 1
    while T < cap and fn(T, arg) > limit:
        T += 1
    return T

print("steps a gradient can travel before falling below 1e-4:")
for w in [0.5, 0.9, 0.99]:
    print(f"  vanilla RNN, w={w:<5}          {horizon(rnn_reach, w):>5d} steps")
for bias in [0.0, 1.0, 3.0, 6.0]:
    print(f"  LSTM, forget bias={bias:<4} (f={sigmoid(bias):.3f})  {horizon(lstm_reach, bias):>5d} steps")`,
  solution: `import numpy as np

def sigmoid(z): return 1/(1+np.exp(-z))

def rnn_reach(T, w):
    h, pres = 0.0, []
    for t in range(T):
        z = w*h + (1.0 if t == 0 else 0.0)
        pres.append(z); h = np.tanh(z)
    g = 1.0
    for t in reversed(range(T)):
        g *= w * (1 - np.tanh(pres[t])**2)
    return abs(g)

def lstm_reach(T, bias):
    return sigmoid(bias) ** T

print("vanilla RNN: fraction of the gradient reaching step 0")
print(f"{'w':>6} " + "".join(f"{'T='+str(T):>12}" for T in [10, 30, 100, 300]))
for w in [0.5, 0.9, 1.0, 1.1]:
    print(f"{w:6.1f} " + "".join(f"{rnn_reach(T, w):12.2e}" for T in [10, 30, 100, 300]))

print("\\nLSTM cell state: the same measurement")
print(f"{'bias':>6} {'f':>7} " + "".join(f"{'T='+str(T):>12}" for T in [10, 30, 100, 300]))
for bias in [0.0, 1.0, 3.0, 6.0]:
    print(f"{bias:6.1f} {sigmoid(bias):7.3f} " +
          "".join(f"{lstm_reach(T, bias):12.2e}" for T in [10, 30, 100, 300]))

assert rnn_reach(100, 0.9) < 1e-5
assert lstm_reach(100, 6.0) > 0.5
print("PASS\\n")

def horizon(fn, arg, limit=1e-4, cap=2000):
    T = 1
    while T < cap and fn(T, arg) > limit:
        T += 1
    return T

print("steps a gradient can travel before falling below 1e-4:")
for w in [0.5, 0.9, 0.99]:
    print(f"  vanilla RNN, w={w:<5}          {horizon(rnn_reach, w):>5d} steps")
for bias in [0.0, 1.0, 3.0, 6.0]:
    print(f"  LSTM, forget bias={bias:<4} (f={sigmoid(bias):.3f})  {horizon(lstm_reach, bias):>5d} steps")`,
  explain: `The vanilla table has no usable row, and the reason is more interesting than "the weight was wrong". At
$w = 0.5$ the gradient is gone within a dozen steps, as you would expect. But look at $w = 1.1$, which ought to
*explode* — it decays even faster than $w = 0.9$ does. The culprit is the other factor: a larger $w$ drives the
state further out along $\\tanh$, where $\\tanh' \\approx 0$, and that collapse overwhelms the $1.1$. Even the
best row, $w = 1.0$, is down to $3\\times10^{-4}$ after 300 steps.

So there is no setting of a single recurrent weight that carries a gradient a hundred steps. Turning $w$ up does
not help, because the activation fights back. That is why plain RNNs are unreliable past roughly ten steps, and
why the fix had to be structural rather than a matter of tuning.

The LSTM table is a different situation entirely, and the reason is that the decay rate is no longer a property
of a weight matrix — it is a gate the network sets. A forget bias of 6 gives $f = 0.998$, and $0.998^{300}$ is
still around 0.5. The last block turns this into the number that matters: with the forget gate near 1, a gradient
travels *thousands* of steps.

Two things are worth taking from this. First, the practical note from the lesson — initialize the forget-gate
bias to 1 or more — is not a superstition; the table shows exactly what it buys. Second, notice that this is
still an exponential, $f^T$: the LSTM did not abolish the decay, it made the base something the network chooses
and can push arbitrarily close to 1. That is the same move a residual connection makes, where the base is pinned
at exactly 1 by construction.`,
},

'nn-embeddings': {
  title: 'Train skip-gram embeddings, then find the word that has no good vector',
  prompt: `1. **One-hot has no geometry.** The first block measures every pairwise distance between one-hot vectors.
   Check what it reports before writing any code — that number is the reason embeddings exist.
2. **Implement the update.** For each (centre, context) pair, push the true context up and $k$ random words
   down. Then look at the nearest neighbours: nobody told the model that a cat and a dog are related.
3. **The static-embedding failure.** The corpus deliberately uses "bank" in two unrelated senses. Compare its
   vector against the money words and against the river words. **Predict what you will find before running.**`,
  hint: 'The gradient for a (centre, target) pair is $(\\sigma(\\mathbf{u}\\cdot\\mathbf{v}) - \\text{label})$. That single number scales $\\mathbf{u}$ to update the centre vector, and scales $\\mathbf{v}$ to update each target vector. Use a learning rate of 0.02.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

# ---------- 1. what one-hot vectors look like to a network ----------
onehot = np.eye(6)
d = [np.linalg.norm(onehot[i]-onehot[j]) for i in range(6) for j in range(i+1, 6)]
print(f"distances between one-hot vectors: min {min(d):.4f}  max {max(d):.4f}")
print("every symbol is exactly as far from every other. No word is 'closer' to any word.\\n")

# ---------- 2. skip-gram with negative sampling ----------
corpus = ("the cat sat on the mat . the dog sat on the rug . "
          "the cat chased the mouse . the dog chased the cat . "
          "a mouse ate the cheese . the cat ate the mouse . "
          "i put money in the bank . she withdrew cash from the bank . "
          "the money and the cash are safe . he kept money as cash . "
          "we sat by the river bank . the boat left the river bank . "
          "the river and the boat are calm . the boat crossed the river .").split()
vocab = sorted(set(corpus)); w2i = {w: i for i, w in enumerate(vocab)}
V, D, WIN = len(vocab), 16, 2

pairs = [(w2i[corpus[i]], w2i[corpus[j]])
         for i in range(len(corpus))
         for j in range(max(0, i-WIN), min(len(corpus), i+WIN+1)) if i != j]

Ein  = rng.normal(0, 0.1, (V, D))
Eout = rng.normal(0, 0.1, (V, D))
sigmoid = lambda z: 1/(1+np.exp(-np.clip(z, -30, 30)))

for epoch in range(60):
    for c, o in rng.permutation(pairs):
        targets = np.r_[o, rng.integers(0, V, 10)]
        labels  = np.r_[1.0, np.zeros(10)]
        v, U = Ein[c], Eout[targets]
        # TODO: p = sigmoid(U @ v); g = p - labels;
        #       then update Ein[c] and Eout[targets] with lr = 0.02
        pass

E = Ein / (np.linalg.norm(Ein, axis=1, keepdims=True) + 1e-9)
def sim(a, b): return float(E[w2i[a]] @ E[w2i[b]])

for w in ["cat", "money", "river"]:
    s = E @ E[w2i[w]]
    top = np.argsort(-s)[1:4]
    print(f"{w:8s} nearest: {[(vocab[i], round(float(s[i]), 2)) for i in top]}")

assert sim("cat", "dog") > sim("cat", "cheese"), "cat and dog share contexts; they should be closer"
print("\\nPASS\\n")

# ---------- 3. the word with two meanings ----------
money_words = ["money", "cash"]
river_words = ["river", "boat"]
print(f"within the money group:   {sim('money','cash'):.3f}")
print(f"within the river group:   {sim('river','boat'):.3f}")
print(f"'bank' to the money group: {np.mean([sim('bank', w) for w in money_words]):.3f}")
print(f"'bank' to the river group: {np.mean([sim('bank', w) for w in river_words]):.3f}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

onehot = np.eye(6)
d = [np.linalg.norm(onehot[i]-onehot[j]) for i in range(6) for j in range(i+1, 6)]
print(f"distances between one-hot vectors: min {min(d):.4f}  max {max(d):.4f}")
print("every symbol is exactly as far from every other. No word is 'closer' to any word.\\n")

corpus = ("the cat sat on the mat . the dog sat on the rug . "
          "the cat chased the mouse . the dog chased the cat . "
          "a mouse ate the cheese . the cat ate the mouse . "
          "i put money in the bank . she withdrew cash from the bank . "
          "the money and the cash are safe . he kept money as cash . "
          "we sat by the river bank . the boat left the river bank . "
          "the river and the boat are calm . the boat crossed the river .").split()
vocab = sorted(set(corpus)); w2i = {w: i for i, w in enumerate(vocab)}
V, D, WIN = len(vocab), 16, 2

pairs = [(w2i[corpus[i]], w2i[corpus[j]])
         for i in range(len(corpus))
         for j in range(max(0, i-WIN), min(len(corpus), i+WIN+1)) if i != j]

Ein  = rng.normal(0, 0.1, (V, D))
Eout = rng.normal(0, 0.1, (V, D))
sigmoid = lambda z: 1/(1+np.exp(-np.clip(z, -30, 30)))

for epoch in range(60):
    for c, o in rng.permutation(pairs):
        targets = np.r_[o, rng.integers(0, V, 10)]
        labels  = np.r_[1.0, np.zeros(10)]
        v, U = Ein[c], Eout[targets]
        p = sigmoid(U @ v)
        g = p - labels
        Ein[c]        = Ein[c]        - 0.02 * (g @ U)
        Eout[targets] = Eout[targets] - 0.02 * np.outer(g, v)

E = Ein / (np.linalg.norm(Ein, axis=1, keepdims=True) + 1e-9)
def sim(a, b): return float(E[w2i[a]] @ E[w2i[b]])

for w in ["cat", "money", "river"]:
    s = E @ E[w2i[w]]
    top = np.argsort(-s)[1:4]
    print(f"{w:8s} nearest: {[(vocab[i], round(float(s[i]), 2)) for i in top]}")

assert sim("cat", "dog") > sim("cat", "cheese")
print("\\nPASS\\n")

money_words = ["money", "cash"]
river_words = ["river", "boat"]
print(f"within the money group:   {sim('money','cash'):.3f}")
print(f"within the river group:   {sim('river','boat'):.3f}")
print(f"'bank' to the money group: {np.mean([sim('bank', w) for w in money_words]):.3f}")
print(f"'bank' to the river group: {np.mean([sim('bank', w) for w in river_words]):.3f}")`,
  explain: `Part 1 is the argument for embeddings in one number: every pair of one-hot vectors is exactly
$\\sqrt2$ apart, so before any learning begins the representation asserts that "cat" is precisely as related to
"dog" as it is to "bureaucracy". Whatever structure the language has, one-hot has thrown all of it away.

Part 2: after training, words that shared contexts have pulled together. The corpus contains no definitions and
no supervision — only which words sit near which other words — and the geometry came out of that alone. That is
the distributional hypothesis earning its keep.

Part 3 is the limitation that ended this era. "bank" appears in two unrelated senses, and it gets **one**
vector. Read the four numbers together: the river words are bound tightly to each other (about 0.84), the money
words less so but still clearly (about 0.53) — and "bank" sits at roughly **0.43 to both**, below either group's
internal similarity and almost exactly equidistant between them. It has been placed *between* two meanings and
belongs to neither.

There is no setting of the training that fixes this, because the model has one row per word type and the word
genuinely has two meanings. The only fix is to let the vector depend on the surrounding sentence, which is what
contextual embeddings do and why BERT was a phase change rather than an improvement. (With a corpus this small
the numbers are noisy — the point is the *pattern*, not the third decimal.)`,
},

};

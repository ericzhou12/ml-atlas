/* ============================================================
   Track 3 — Neural networks
   ============================================================ */

import { t, key, intuition, warn, hist, mathnote, viz, deriv, code, quiz, paper, book, course, blog, video, demo, codeRef } from './_helpers.js';

export default [

/* ---------------------------------------------------------- */
{
  id: 'nn-perceptron-mlp',
  title: 'From Perceptron to Multilayer Network',
  sub: 'Why one layer is not enough, and what a hidden layer actually buys you.',
  mins: 24, level: 'core',
  prereq: ['ml-logistic'],
  tags: ['neural networks', 'MLP'],
  sections: [
    t(`## The perceptron

Rosenblatt's 1958 model: weighted sum, then a threshold.

$$\\hat y = \\text{sign}(\\mathbf{w}^{\\mathsf T}\\mathbf{x}+b)$$

with the update rule "if wrong, nudge $\\mathbf{w}$ toward the correct answer": $\\mathbf{w} \\mathrel{+}= \\eta\\, y\\,\\mathbf{x}$
on each mistake. Rosenblatt proved it converges in finitely many steps **if the data is linearly separable**.

That conditional clause did enormous historical damage.`),

    hist(`In 1969 Minsky and Papert published *Perceptrons*, showing that a single-layer perceptron cannot compute XOR —
because no line separates $\\{(0,0),(1,1)\\}$ from $\\{(0,1),(1,0)\\}$. They knew multilayer networks were more powerful;
what they pointed out was that nobody knew how to train them. Funding and interest collapsed for over a decade — the
first "AI winter."

The resolution, backpropagation, was independently discovered several times (Linnainmaa 1970, Werbos 1974) before
Rumelhart, Hinton and Williams made it famous in 1986. It is worth remembering that the field's most important
algorithm sat essentially unnoticed for sixteen years.`),

    t(`## The multilayer perceptron

Stack linear maps with nonlinearities between them:

$$\\mathbf{h}_1 = \\phi(W_1\\mathbf{x}+\\mathbf{b}_1), \\quad
\\mathbf{h}_2 = \\phi(W_2\\mathbf{h}_1+\\mathbf{b}_2), \\quad \\ldots, \\quad
\\hat y = W_L\\mathbf{h}_{L-1}+\\mathbf{b}_L$$

The nonlinearity $\\phi$ is not optional decoration. Without it, the composition of linear maps is just another linear
map: $W_2(W_1\\mathbf{x}) = (W_2W_1)\\mathbf{x}$. A hundred layers without activations is exactly as expressive as one.`),

    key(`Try it yourself below: set the activation to **identity** and try to fit the spiral or XOR datasets. It will
never work, at any width or depth. Then switch to ReLU.`),

    viz('mlp-playground'),

    t(`## What the hidden layer is doing

Two complementary readings, both useful:

**Feature learning.** Each hidden unit computes a feature of the input, and the output layer is a linear model *on those
learned features*. So a neural network is logistic regression that also invents its own basis functions instead of you
hand-engineering them.

**Piecewise linear tiling.** With ReLU, each unit contributes a hinge — flat, then a ramp starting where
$\\mathbf{w}^{\\mathsf T}\\mathbf{x}+b$ crosses zero. Sum enough of them and you can trace any curve.`),

    viz('hidden-units'),

    t(`## Universal approximation, and what it does not say

**Theorem** (Cybenko 1989, Hornik 1991): a network with one hidden layer and a non-polynomial activation can
approximate any continuous function on a compact set to arbitrary accuracy, given enough hidden units.

This is much weaker than it sounds:

- It says nothing about **how many** units. The required width can be exponential in the input dimension.
- It says nothing about whether **gradient descent will find** those weights.
- It says nothing about **generalization** — approximating the function on your training points is not the goal.

So it is a statement about the hypothesis class, not about learning. Its practical value is mostly negative: it rules
out "the architecture is fundamentally incapable" as an explanation for failure.`),

    t(`## Why depth instead of width

If one layer suffices in principle, why is everything deep?

- **Exponential efficiency.** Some functions need exponentially many units in a shallow network but only polynomially
  many when deep. Composition is a form of reuse: layer 3 can build on the features layer 2 already computed, rather
  than reconstructing them.
- **Hierarchy matches the data.** Edges → textures → parts → objects. Characters → words → phrases → meaning. Real
  data is compositional, and depth is the architectural expression of that.
- **It empirically works.** This is not a small point. The theory here lags well behind practice.`),

    code('An MLP with backprop, in NumPy', `import numpy as np
rng = np.random.default_rng(0)

# XOR — the problem that killed the perceptron
X = np.array([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
y = np.array([[0.], [1.], [1.], [0.]])

def init(sizes):
    return [(rng.normal(0, np.sqrt(2/a), (a, b)), np.zeros(b))
            for a, b in zip(sizes[:-1], sizes[1:])]

def relu(z): return np.maximum(0, z)
def sigmoid(z): return 1/(1+np.exp(-np.clip(z, -500, 500)))

params = init([2, 8, 8, 1])

for step in range(4001):
    # ---- forward, caching what backward needs ----
    acts, pres = [X], []
    h = X
    for i, (W, b) in enumerate(params):
        z = h @ W + b
        pres.append(z)
        h = sigmoid(z) if i == len(params)-1 else relu(z)
        acts.append(h)

    loss = -np.mean(y*np.log(h+1e-12) + (1-y)*np.log(1-h+1e-12))

    # ---- backward ----
    delta = (h - y) / len(X)          # sigmoid + BCE collapse to this
    grads = []
    for i in reversed(range(len(params))):
        W, b = params[i]
        grads.append((acts[i].T @ delta, delta.sum(0)))
        if i > 0:
            delta = (delta @ W.T) * (pres[i-1] > 0)      # ReLU derivative
    grads.reverse()

    # ---- SGD step ----
    params = [(W - 0.5*gW, b - 0.5*gb) for (W, b), (gW, gb) in zip(params, grads)]

    if step % 1000 == 0:
        print(f"step {step:5d}  loss {loss:.5f}  preds {h.ravel().round(3)}")

print("\\ntarget:", y.ravel())
print("A linear model cannot do this. Two hidden layers solve it in seconds.")`,
      'This is a complete training loop — forward, backward, update — in about thirty lines. Everything after this lesson is a variation on it.'),

    quiz('You replace every activation in a 10-layer network with the identity. What is the resulting model equivalent to?',
      ['A single linear layer — the composition of linear maps is linear',
       'A 10-layer network with ReLU, but slower to train',
       'A network that can still fit XOR, just less efficiently',
       'A polynomial model of degree 10'],
      0,
      '$W_{10}(W_9(\\cdots W_1\\mathbf{x})) = (W_{10}\\cdots W_1)\\mathbf{x} = W_{\\text{eff}}\\mathbf{x}$. Depth without nonlinearity buys exactly nothing in expressiveness — although it does change the *optimization* dynamics, which is a genuinely interesting research topic (deep linear networks have non-convex loss surfaces and implicit low-rank bias).'),
  ],
  refs: [
    paper('Learning representations by back-propagating errors', 'Rumelhart, Hinton & Williams', 1986, 'https://www.nature.com/articles/323533a0', 'The paper that restarted the field.'),
    paper('Approximation by superpositions of a sigmoidal function', 'George Cybenko', 1989, 'https://link.springer.com/article/10.1007/BF02551274', 'Universal approximation.'),
    book('Deep Learning', 'Goodfellow, Bengio & Courville', 2016, 'https://www.deeplearningbook.org/', 'Free online. Chapter 6 is the canonical MLP treatment.'),
    demo('TensorFlow Playground', 'Smilkov & Carter', 2016, 'https://playground.tensorflow.org/', 'The classic in-browser neural net. Worth playing with alongside the figure above.'),
    course('Neural Networks: Zero to Hero', 'Andrej Karpathy', 2022, 'https://karpathy.ai/zero-to-hero.html', 'Builds backprop, then an MLP, then a transformer, from scratch on video. If you only use one external resource from this atlas, use this one.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'nn-backprop',
  title: 'Backpropagation',
  sub: 'The algorithm. Understand it once, properly, and everything else follows.',
  mins: 28, level: 'core',
  prereq: ['math-jacobian', 'nn-perceptron-mlp'],
  tags: ['backprop', 'autodiff'],
  sections: [
    t(`## The problem

A network has millions of parameters and one scalar loss. You need $\\partial\\mathcal{L}/\\partial\\theta$ for every
$\\theta$. Finite differences would need one forward pass per parameter — utterly hopeless.

Backpropagation computes **all** of them in a single backward pass costing about the same as one forward pass. That is
the whole achievement, and it comes from a single observation: the chain rule lets you *reuse* intermediate results.`),

    t(`## The two-phase structure

**Forward**: compute and cache each layer's output.
**Backward**: propagate $\\delta = \\partial\\mathcal{L}/\\partial z$ from the loss back toward the input, and at each
layer use it to compute the parameter gradients.`),

    viz('backprop-graph'),

    t(`Move the sliders and watch the numbers. Every node needs exactly two things: its **local derivative** (what it
does to its input) and the **incoming gradient** (how much the loss cares about its output). It multiplies them and
passes the result upstream. That is all backprop is.`),

    deriv('The four equations, derived', `Notation: layer $\\ell$ has pre-activation $\\mathbf{z}^\\ell = W^\\ell\\mathbf{a}^{\\ell-1}+\\mathbf{b}^\\ell$ and
activation $\\mathbf{a}^\\ell = \\phi(\\mathbf{z}^\\ell)$. Define $\\boldsymbol\\delta^\\ell = \\partial\\mathcal{L}/\\partial\\mathbf{z}^\\ell$.

**(1) The output layer.** By the chain rule through $\\mathbf{a}^L=\\phi(\\mathbf{z}^L)$:

$$\\boldsymbol\\delta^L = \\nabla_{\\mathbf{a}^L}\\mathcal{L} \\odot \\phi'(\\mathbf{z}^L)$$

(For softmax + cross-entropy this collapses to $\\mathbf{p}-\\mathbf{y}$ — see the [Jacobian lesson](#/l/math-jacobian).)

**(2) Propagate backward.** $\\mathcal{L}$ depends on $\\mathbf{z}^\\ell$ only through $\\mathbf{z}^{\\ell+1}$, and
$\\mathbf{z}^{\\ell+1} = W^{\\ell+1}\\phi(\\mathbf{z}^\\ell)+\\mathbf{b}^{\\ell+1}$, so

$$\\boldsymbol\\delta^\\ell = \\big((W^{\\ell+1})^{\\mathsf T}\\boldsymbol\\delta^{\\ell+1}\\big)\\odot\\phi'(\\mathbf{z}^\\ell)$$

**(3) Weight gradients.** Since $z^\\ell_i = \\sum_j W^\\ell_{ij}a^{\\ell-1}_j + b^\\ell_i$:

$$\\frac{\\partial\\mathcal{L}}{\\partial W^\\ell_{ij}} = \\delta^\\ell_i\\,a^{\\ell-1}_j
\\qquad\\text{i.e.}\\qquad \\nabla_{W^\\ell}\\mathcal{L} = \\boldsymbol\\delta^\\ell (\\mathbf{a}^{\\ell-1})^{\\mathsf T}$$

**(4) Bias gradients.** $\\nabla_{\\mathbf{b}^\\ell}\\mathcal{L} = \\boldsymbol\\delta^\\ell$.

Four equations. Every deep learning framework is an efficient, generalized implementation of exactly these.`),

    key(`Read equation (2) carefully — it explains most training pathologies at once:

$$\\boldsymbol\\delta^\\ell = \\underbrace{(W^{\\ell+1})^{\\mathsf T}}_{\\text{routing}}\\boldsymbol\\delta^{\\ell+1} \\odot \\underbrace{\\phi'(\\mathbf{z}^\\ell)}_{\\text{gating}}$$

The gradient is repeatedly multiplied by weight matrices and activation derivatives. If those factors are on average
below 1, the product **vanishes** exponentially in depth. Above 1 and it **explodes**. There is no third option — you
have to engineer the factors.`),

    viz('vanishing-gradients'),

    t(`Play with that figure. Switch the activation to tanh with a gain of 0.5 and watch the gradient reaching layer 0
fall to $10^{-8}$. Then turn on **residual connections** and watch it flatten immediately: a residual gives the gradient
an identity path, so equation (2) becomes $\\boldsymbol\\delta^\\ell = \\boldsymbol\\delta^{\\ell+1} + (\\ldots)$ — an
addition, which cannot decay.`),

    t(`## Automatic differentiation

Backprop is a special case of **reverse-mode automatic differentiation**. Frameworks build a graph of primitive
operations during the forward pass, each knowing its own vector-Jacobian product, then walk it in reverse.

You never write derivatives. You write the forward computation, and the derivative of any composition comes for free.
This is why the deep learning explosion followed the frameworks: PyTorch and JAX did not make new mathematics
possible, they made experimentation cheap.

The memory cost is real: the forward activations must be kept alive until the backward pass consumes them. For a large
transformer this can exceed the weights. **Gradient checkpointing** stores only a subset and recomputes the rest —
roughly $\\sqrt{L}$ memory for ~30% more compute.`),

    code('Autodiff in 60 lines', `import numpy as np

class Value:
    """A scalar with a gradient. This is micrograd, essentially."""
    def __init__(self, data, children=(), op=''):
        self.data, self.grad = data, 0.0
        self._backward = lambda: None
        self._prev, self._op = set(children), op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad          # d(a+b)/da = 1
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad     # d(ab)/da = b
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        tval = np.tanh(self.data)
        out = Value(tval, (self,), 'tanh')
        def _backward():
            self.grad += (1 - tval**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # topological order, then apply the chain rule in reverse
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

# a tiny neuron, differentiated
x1, x2 = Value(2.0), Value(-1.0)
w1, w2 = Value(0.5), Value(1.5)
b = Value(0.3)
n = x1*w1 + x2*w2 + b
out = n.tanh()
out.backward()

print(f"output      = {out.data:.6f}")
print(f"dout/dw1    = {w1.grad:.6f}   (expect x1 * (1-tanh^2) = {2.0*(1-np.tanh(n.data)**2):.6f})")
print(f"dout/dw2    = {w2.grad:.6f}")
print(f"dout/db     = {b.grad:.6f}")`,
      'This is the core of Karpathy\'s micrograd. Real frameworks operate on tensors rather than scalars and fuse operations for speed, but the structure — build a graph, topologically sort, apply local derivatives in reverse — is identical.'),

    warn(`**Debugging a hand-written backward pass.** Always gradient-check against central differences (see the
[derivatives lesson](#/l/math-derivatives)). Relative error below $10^{-5}$ in float64 means you are fine; above
$10^{-2}$ means a real bug. Common culprits, in order: a missing transpose, forgetting to *accumulate* when a value is
used more than once, and a sum over the wrong axis for the bias.`),

    quiz('Gradient checkpointing trades what for what?',
      ['Extra compute (recomputing activations) for less memory',
       'Extra memory for less compute',
       'Accuracy for speed',
       'Training speed for better generalization'],
      0,
      'Reverse mode must keep forward activations for the backward pass. Checkpointing stores only a subset — typically every $\\sqrt{L}$-th layer — and recomputes the rest on demand, cutting activation memory from $O(L)$ to $O(\\sqrt L)$ at roughly 30% extra compute. It is what lets you train models whose activations would not otherwise fit.'),
  ],
  refs: [
    video('The spelled-out intro to neural networks and backpropagation', 'Andrej Karpathy', 2022, 'https://www.youtube.com/watch?v=VMj-3S1tku0', 'Two hours, builds micrograd from an empty file. The best explanation of backprop that exists.'),
    codeRef('micrograd', 'Andrej Karpathy', 2020, 'https://github.com/karpathy/micrograd', '150 lines. Read all of it.'),
    blog('Calculus on Computational Graphs: Backpropagation', 'Christopher Olah', 2015, 'https://colah.github.io/posts/2015-08-Backprop/', 'The clearest written explanation, with excellent diagrams.'),
    book('Deep Learning, Ch. 6.5', 'Goodfellow, Bengio & Courville', 2016, 'https://www.deeplearningbook.org/contents/mlp.html', 'The formal treatment.'),
    paper('Training Deep Nets with Sublinear Memory Cost', 'Chen et al.', 2016, 'https://arxiv.org/abs/1604.06174', 'Gradient checkpointing.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'nn-activations',
  title: 'Activation Functions',
  sub: 'A one-line choice that decides whether a deep network trains at all.',
  mins: 18, level: 'core',
  prereq: ['nn-backprop'],
  tags: ['activations', 'ReLU'],
  sections: [
    t(`## What to look at

Judge an activation by its **derivative**, because that is the factor backprop multiplies by. Three properties matter:

1. **Does it saturate?** If $\\phi' \\to 0$ for large inputs, gradients die there.
2. **Is it zero-centred?** If outputs are always positive, the gradients for a layer's weights all share a sign, which
   makes optimization zig-zag.
3. **Is it cheap?** It runs on every activation of every layer of every step.`),

    viz('activations'),

    t(`## The lineage

**Sigmoid** $\\sigma(x)=1/(1+e^{-x})$. Historically first, and a bad hidden activation: max derivative 0.25, saturates
at both ends, not zero-centred. Still correct as an *output* for binary probabilities and for gates (LSTM, GLU).

**tanh**. Zero-centred sigmoid, derivative up to 1. Better, still saturates. Standard through the 1990s and 2000s, and
still used in some RNNs.

**ReLU** $\\max(0,x)$. The change that made deep learning work. Derivative is exactly 1 for positive inputs — no decay,
no saturation on that side — and it is a single comparison to compute. It also produces genuine sparsity (~50% of units
output exactly zero).

Its failure mode is **dying ReLU**: a unit pushed to always-negative gets zero gradient forever and never recovers. A
large learning rate can kill a substantial fraction of a layer permanently.

**Leaky ReLU / PReLU** $\\max(\\alpha x, x)$. Small negative slope so nothing fully dies. Helps sometimes; not
universally adopted.

**GELU** $x\\Phi(x)$. Smooth, and non-monotonic near zero. Used by BERT, GPT, and most transformers. The stated intuition
is a stochastic regularizer (multiply by a Bernoulli gate whose probability is $\\Phi(x)$), but honestly it is used
because it consistently works slightly better.

**SiLU / Swish** $x\\sigma(x)$. Nearly identical to GELU in shape and performance, cheaper to compute.

**GLU variants** — **SwiGLU**, used in Llama, PaLM, and most recent LLMs:

$$\\text{SwiGLU}(x) = (\\text{Swish}(xW_1) \\odot xW_2)W_3$$

The elementwise product is a *learned gate*: one branch decides how much of the other passes. It needs three matrices
instead of two, so implementations shrink the hidden dimension to $\\tfrac{2}{3}\\times$ to keep the parameter count
matched. It is worth roughly a 1% loss improvement — small, free, and therefore universal.`),

    t(`## What to actually use

| Situation | Choice |
|---|---|
| Default for CNNs / MLPs | ReLU |
| Transformers | SwiGLU, or GELU |
| Output: binary probability | sigmoid |
| Output: multiclass | softmax |
| Output: unbounded regression | none (linear) |
| Output: non-negative regression | softplus or ReLU |
| Gates inside a cell | sigmoid |

The honest summary: the difference between ReLU, GELU, and SwiGLU is real but small — a fraction of a percent. The
difference between any of them and sigmoid in a deep network is enormous. Get out of the saturating regime and then
stop worrying about it.`),

    warn(`**A dead ReLU is permanent.** Once $\\mathbf{w}^{\\mathsf T}\\mathbf{x}+b < 0$ for every input in the dataset, the
gradient is exactly zero and no update will ever revive it. Diagnose by logging the fraction of zero activations per
layer — above ~90% and something is wrong. Causes: learning rate too high, bad initialization, or a large negative
bias. Leaky ReLU or GELU avoids it structurally.`),

    code('Activation diagnostics', `import numpy as np

def relu(x):    return np.maximum(0, x)
def gelu(x):    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
def silu(x):    return x/(1+np.exp(-x))
def sigmoid(x): return 1/(1+np.exp(-x))

def deriv(f, x, h=1e-5): return (f(x+h)-f(x-h))/(2*h)

x = np.linspace(-6, 6, 2001)
print(f"{'activation':12s} {'max deriv':>10s} {'% |deriv|<0.01':>16s} {'zero-centred':>14s}")
for name, f in [("sigmoid", sigmoid), ("tanh", np.tanh), ("relu", relu),
                ("gelu", gelu), ("silu", silu)]:
    d = deriv(f, x)
    print(f"{name:12s} {np.abs(d).max():10.4f} {np.mean(np.abs(d)<0.01)*100:15.1f}% "
          f"{str(abs(f(x).mean()) < 0.05):>14s}")

# how many units die under an aggressive learning rate?
rng = np.random.default_rng(0)
W = rng.normal(0, 0.5, (256, 256)); b = np.zeros(256)
h = rng.normal(0, 1, (512, 256))
for step in range(50):
    z = h @ W + b
    a = relu(z)
    g = (a > 0).astype(float) * rng.normal(0, 1, a.shape)
    b -= 0.5 * g.mean(0)                     # a deliberately too-large step
dead = (relu(h @ W + b).max(0) == 0).mean()
print(f"\\nfraction of permanently dead ReLU units: {dead:.1%}")`),

    quiz('You log activation statistics and find 97% of a layer\'s ReLU outputs are exactly zero. What is happening?',
      ['Most of the layer is dead — likely too-high a learning rate or bad init; those units will never recover',
       'Healthy sparsity; ReLU is supposed to be sparse',
       'The layer is overfitting',
       'The layer needs more units'],
      0,
      'Healthy ReLU sparsity is around 50%. At 97% the layer has essentially no capacity left and cannot recover, since a zero output means a zero gradient forever. Fix by lowering the learning rate, re-initializing, checking for a runaway negative bias, or switching to GELU/Leaky ReLU which keep a gradient path on the negative side.'),
  ],
  refs: [
    paper('Deep Sparse Rectifier Neural Networks', 'Glorot, Bordes & Bengio', 2011, 'https://proceedings.mlr.press/v15/glorot11a.html', 'The ReLU paper.'),
    paper('Gaussian Error Linear Units (GELUs)', 'Hendrycks & Gimpel', 2016, 'https://arxiv.org/abs/1606.08415', ''),
    paper('GLU Variants Improve Transformer', 'Noam Shazeer', 2020, 'https://arxiv.org/abs/2002.05202', 'Two pages, ends with "we offer no explanation as to why these architectures seem to work." Refreshingly honest, and SwiGLU is now everywhere.'),
    paper('Searching for Activation Functions', 'Ramachandran, Zoph & Le', 2017, 'https://arxiv.org/abs/1710.05941', 'Swish, found by automated search.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'nn-initialization',
  title: 'Initialization and Signal Propagation',
  sub: 'The random numbers you start from determine whether training happens.',
  mins: 18, level: 'core',
  prereq: ['nn-backprop'],
  tags: ['initialization'],
  sections: [
    t(`## Why zero fails, and why "small random" is not enough

**All zeros**: every unit in a layer computes the same thing, gets the same gradient, and stays identical forever.
Symmetry never breaks, and your 512-unit layer has the capacity of one unit.

**All-random-but-badly-scaled**: this is the interesting failure. Consider a linear layer with $n$ inputs and weights
drawn i.i.d. with variance $\\sigma^2$. For unit-variance inputs,

$$\\text{Var}(z_i) = \\sum_{j=1}^{n}\\text{Var}(W_{ij}x_j) = n\\sigma^2$$

So variance is multiplied by $n\\sigma^2$ at every layer. Unless that factor is exactly 1, activations grow or shrink
**geometrically with depth**.`),

    viz('initialization'),

    t(`## The two schemes

**Xavier/Glorot** (2010) — for symmetric activations (tanh, sigmoid). Balance forward and backward variance:

$$\\sigma^2 = \\frac{2}{n_{\\text{in}}+n_{\\text{out}}} \\qquad\\text{or simply}\\qquad \\sigma^2=\\frac{1}{n_{\\text{in}}}$$

**He/Kaiming** (2015) — for ReLU. ReLU zeroes half its inputs, halving the variance, so compensate with a factor of 2:

$$\\sigma^2 = \\frac{2}{n_{\\text{in}}}$$

That factor of 2 is the entire difference between the two schemes, and it is what made 30-layer networks trainable
before residual connections existed.`),

    deriv('Deriving the He factor', `Let $z = \\sum_j w_j x_j$ with $w_j$ i.i.d. zero-mean variance $\\sigma^2$, independent of $x$. Then

$$\\text{Var}(z) = n\\,\\sigma^2\\,\\mathbb{E}[x^2]$$

Now suppose $x = \\text{ReLU}(y)$ where $y$ is symmetric about zero with variance $v$. Half the mass is zeroed, and the
surviving half contributes its full second moment:

$$\\mathbb{E}[x^2] = \\mathbb{E}[\\max(0,y)^2] = \\tfrac12\\mathbb{E}[y^2] = \\tfrac12 v$$

Substituting: $\\text{Var}(z) = \\tfrac12 n\\sigma^2 v$. For variance to be preserved layer to layer we need
$\\text{Var}(z) = v$, hence

$$\\tfrac12 n\\sigma^2 = 1 \\quad\\Longrightarrow\\quad \\sigma^2 = \\frac{2}{n}$$ ∎

The same argument with no ReLU (or with a symmetric activation that is roughly linear near zero) gives
$\\sigma^2 = 1/n$ — Xavier.`),

    t(`## Modern practice

Initialization matters less than it used to, because normalization layers and residual connections rescue you from
moderate mistakes. But the details still show up in every serious training recipe:

- **Biases at zero.** Except forget gates in LSTMs, which are initialized to 1 so the cell remembers by default.
- **Scaled residual init.** GPT-2 scales the weights of layers writing into the residual stream by $1/\\sqrt{2L}$, so
  that the residual stream's variance does not grow with depth.
- **Embeddings** often use $\\mathcal{N}(0, 0.02)$, which is the transformer default and not derived from anything in
  particular.
- **LayerNorm** starts with $\\gamma=1,\\beta=0$ (identity).
- **Zero-init the last layer of a residual branch** (Fixup, ReZero, and the gates in Flamingo's cross-attention). Then
  the network starts as exactly the identity function and learns to deviate — remarkably stable.
- **muP** (maximal update parameterization) scales initialization and learning rates so that hyperparameters found on a
  small model transfer to a large one. Practically valuable when a single large run costs millions.`),

    code('Signal propagation under different inits', `import numpy as np
rng = np.random.default_rng(0)

def propagate(scheme, depth=25, width=256, act="relu"):
    n = width
    sigma = {"zeros": 0.0, "tiny": 0.01, "xavier": np.sqrt(1/n),
             "he": np.sqrt(2/n), "big": 1.0}[scheme]
    h = rng.normal(0, 1, (128, n))
    stds = []
    for _ in range(depth):
        W = rng.normal(0, sigma, (n, n))
        z = h @ W
        h = np.maximum(0, z) if act == "relu" else np.tanh(z)
        stds.append(h.std())
    return stds

print(f"{'scheme':8s} " + "".join(f"L{d:<9d}" for d in [1, 5, 10, 25]))
for s in ["tiny", "xavier", "he", "big"]:
    st = propagate(s)
    print(f"{s:8s} " + "".join(f"{st[d-1]:<10.3e}" for d in [1, 5, 10, 25]))

print("\\n'he' holds steady with ReLU. 'xavier' halves the variance each layer")
print("(that is the missing factor of 2). 'tiny' and 'big' are unusable.")`),

    quiz('Why does He initialization use 2/n while Xavier uses 1/n?',
      ['ReLU zeroes half its inputs, halving the variance; the factor of 2 compensates',
       'He works better for deeper networks in general',
       'Xavier was derived incorrectly',
       'He initialization is for convolutions, Xavier for dense layers'],
      0,
      'A ReLU discards the negative half of a symmetric distribution, so $\\mathbb{E}[\\text{ReLU}(y)^2] = \\tfrac12\\mathbb{E}[y^2]$. Without compensation, activation variance halves at every layer — a factor of $2^{-25}$ over 25 layers. The 2 exactly cancels it.'),
  ],
  refs: [
    paper('Understanding the difficulty of training deep feedforward neural networks', 'Glorot & Bengio', 2010, 'https://proceedings.mlr.press/v9/glorot10a.html', 'Xavier initialization.'),
    paper('Delving Deep into Rectifiers', 'He et al.', 2015, 'https://arxiv.org/abs/1502.01852', 'He initialization, and PReLU.'),
    paper('Fixup Initialization: Residual Learning Without Normalization', 'Zhang, Dauphin & Ma', 2019, 'https://arxiv.org/abs/1901.09321', 'Careful init can replace normalization entirely.'),
    paper('Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer', 'Yang et al.', 2022, 'https://arxiv.org/abs/2203.03466', 'muP. Tune on a small model, transfer to a large one.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'nn-normalization',
  title: 'Normalization Layers',
  sub: 'BatchNorm, LayerNorm, RMSNorm — and why they work is still argued about.',
  mins: 20, level: 'core',
  prereq: ['nn-initialization'],
  tags: ['normalization', 'BatchNorm', 'LayerNorm'],
  sections: [
    t(`## The operation

All of them do the same thing and differ only in **which values get averaged together**:

$$\\hat x = \\frac{x-\\mu}{\\sqrt{\\sigma^2+\\epsilon}}, \\qquad y = \\gamma\\hat x + \\beta$$

The learnable $\\gamma,\\beta$ matter: without them the layer would be strictly less expressive, since it could not
represent an unnormalized function. With them, the network can undo the normalization if it wants to — but it now
controls the scale explicitly rather than having it drift.`),

    viz('normalization'),

    t(`## The variants

**BatchNorm** (2015) normalizes each feature across the batch. Transformative for CNNs, and it comes with real
baggage:

- Behavior depends on batch size; it degrades badly below ~16.
- Train and inference differ — you need running averages, and that discrepancy is a classic source of bugs.
- It couples examples in a batch, which is wrong for sequences and awkward for distributed training.

**LayerNorm** (2016) normalizes across features within each example. Batch-size independent, identical at train and
test, and works for variable-length sequences. **This is why transformers use it.**

**RMSNorm** (2019) drops the mean subtraction entirely:

$$y = \\frac{x}{\\sqrt{\\frac{1}{n}\\sum_i x_i^2 + \\epsilon}}\\cdot g$$

Cheaper, one less pass over the data, and empirically just as good. Used by Llama, T5, Gemma, and most recent models.

**GroupNorm** splits channels into groups — a compromise for vision when batches must be small.`),

    t(`## Pre-norm vs post-norm

Where you put it inside a residual block turns out to matter a great deal.

**Post-norm** (the original transformer): $\\mathbf{x} \\leftarrow \\text{LN}(\\mathbf{x} + \\text{Sublayer}(\\mathbf{x}))$

**Pre-norm** (everything since): $\\mathbf{x} \\leftarrow \\mathbf{x} + \\text{Sublayer}(\\text{LN}(\\mathbf{x}))$

Pre-norm leaves a **clean, unnormalized residual path** from input to output. Gradients flow through it without being
rescaled at every layer, which makes deep models trainable without warmup and far more stable at scale. Post-norm
needs careful warmup and often diverges past a few dozen layers.

The tradeoff is that pre-norm's residual stream grows in magnitude with depth, so a final LayerNorm before the output
head is standard. Some recent models use both (sandwich norm) or add QK-norm inside attention for extra stability at
very large scale.`),

    t(`## Why does it work?

The original BatchNorm paper claimed it reduced "internal covariate shift" — the drift in layer input distributions as
earlier layers update. That explanation has been substantially undermined: Santurkar et al. showed you can *inject*
covariate shift after BatchNorm and it still helps.

The better-supported explanation is that normalization **smooths the loss landscape** — it reduces the Lipschitz
constant of the loss and its gradient, which permits larger stable learning rates. There is also a scale-invariance
effect: after normalization, the loss is invariant to the scale of the preceding weights, which changes the effective
learning rate dynamics in a way that interacts with weight decay in genuinely subtle ways.

This is a good example of something worth internalizing: **a technique can be universally adopted and empirically
essential while its mechanism remains contested.** That is normal in this field.`),

    code('Normalization variants, implemented', `import numpy as np

x = np.random.default_rng(0).normal(2.0, 3.0, (8, 16))     # (batch, features)

def batchnorm(x, eps=1e-5):
    mu, var = x.mean(0, keepdims=True), x.var(0, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

def layernorm(x, eps=1e-5):
    mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

def rmsnorm(x, eps=1e-5):
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps)

for name, f in [("input", lambda z: z), ("batchnorm", batchnorm),
                ("layernorm", layernorm), ("rmsnorm", rmsnorm)]:
    y = f(x)
    print(f"{name:10s}  per-feature mean {y.mean(0)[:3].round(3)}  "
          f"per-example std {y.std(-1)[:3].round(3)}")

# does normalization actually flatten the landscape?
def loss_curvature(normalize):
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.5, (16, 16))
    h = x @ W
    if normalize: h = layernorm(h)
    # crude curvature proxy: how much does the output move for a fixed weight perturbation?
    dW = rng.normal(0, 0.01, (16, 16))
    h2 = x @ (W + dW)
    if normalize: h2 = layernorm(h2)
    return np.abs(h2 - h).mean()

print(f"\\noutput sensitivity to a weight perturbation:")
print(f"  without layernorm: {loss_curvature(False):.5f}")
print(f"  with layernorm:    {loss_curvature(True):.5f}")`),

    quiz('Why do transformers use LayerNorm rather than BatchNorm?',
      ['It is independent of batch size and sequence length, and behaves identically at train and inference',
       'LayerNorm is faster to compute',
       'BatchNorm does not work with attention mathematically',
       'LayerNorm regularizes better'],
      0,
      'BatchNorm computes statistics across the batch, which for sequences means across positions with different lengths and padding — statistically messy and dependent on how you batch. It also needs running averages at inference. LayerNorm normalizes within a single token\'s feature vector: no batch coupling, no train/test discrepancy, no dependence on sequence length.'),
  ],
  refs: [
    paper('Batch Normalization', 'Ioffe & Szegedy', 2015, 'https://arxiv.org/abs/1502.03167', 'The original, including the internal-covariate-shift story.'),
    paper('Layer Normalization', 'Ba, Kiros & Hinton', 2016, 'https://arxiv.org/abs/1607.06450', ''),
    paper('Root Mean Square Layer Normalization', 'Zhang & Sennrich', 2019, 'https://arxiv.org/abs/1910.07467', 'RMSNorm.'),
    paper('How Does Batch Normalization Help Optimization?', 'Santurkar et al.', 2018, 'https://arxiv.org/abs/1805.11604', 'Dismantles the covariate-shift explanation, proposes landscape smoothing.'),
    paper('On Layer Normalization in the Transformer Architecture', 'Xiong et al.', 2020, 'https://arxiv.org/abs/2002.04745', 'Pre-norm vs post-norm, with gradient analysis.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'nn-regularization',
  title: 'Regularizing Deep Networks',
  sub: 'Dropout, augmentation, early stopping — and why the theory here is thin.',
  mins: 20, level: 'core',
  prereq: ['ml-regularization', 'nn-backprop'],
  tags: ['regularization', 'dropout'],
  sections: [
    t(`## Dropout

During training, zero each unit independently with probability $p$ and scale the survivors by $1/(1-p)$. At inference,
do nothing.`),

    viz('dropout'),

    t(`Two readings:

- **Ensemble.** Each step trains a different thinned subnetwork; there are $2^n$ of them sharing weights. Inference
  approximates averaging over all of them.
- **Co-adaptation breaking.** A unit cannot rely on any specific partner being present, so features must be
  individually useful rather than only meaningful in combination.

The $1/(1-p)$ scaling during training ("inverted dropout") keeps the expected activation constant, so inference needs
no adjustment.

Where it stands now: dropout was essential in the pre-BatchNorm era with small datasets. In modern large-scale
transformer pretraining it is often set to **0** — there is so much data that overfitting is not the binding constraint,
and dropout costs you effective capacity. It reappears for fine-tuning on small datasets, where overfitting is real
again.`),

    t(`## Weight decay

L2 on the weights, applied to essentially every model. Two important details:

- **Use AdamW's decoupled form.** Adding $\\lambda\\|\\theta\\|^2$ to the loss routes the penalty through Adam's adaptive
  scaling, so parameters with large gradient history get *less* decay. Decoupled decay applies
  $\\theta \\mathrel{-}= \\eta\\lambda\\theta$ directly.
- **Exclude biases and normalization parameters.** Decaying $\\gamma$ toward zero fights the layer's purpose. Every
  serious training script has a parameter-group split for this.

With normalization layers, weight decay does something less obvious than "keep weights small": since the loss is
scale-invariant with respect to the weights preceding a norm layer, decay effectively controls the *effective learning
rate* rather than the function. This is a genuinely subtle interaction.`),

    t(`## Data augmentation

The most reliable regularizer, because it injects real knowledge: *these transformations do not change the label.*

**Vision** — flips, crops, colour jitter, rotation. Then the stronger ones: **Mixup** (train on convex combinations of
images *and* their labels), **CutMix** (paste a patch from one image into another), **RandAugment** (sample from a pool
of operations with a single magnitude parameter).

**Text** — harder, because most edits change meaning. Back-translation, synonym substitution, and — increasingly —
generating paraphrases with a language model.

**Audio** — time and frequency masking (SpecAugment), speed perturbation, noise injection.

**Self-supervised learning** rests entirely on augmentation: SimCLR's whole training signal is "two augmented views of
the same image should embed to the same point."`),

    t(`## Early stopping

Monitor validation loss, keep the best checkpoint, stop when it stops improving. Nearly free, and for linear models it
is *provably* equivalent to L2 regularization with the number of steps playing the role of $1/\\lambda$.

Practical notes: use patience (do not stop on the first bad epoch), and be aware that with double descent, validation
loss can rise and then fall again — stopping at the first bump can leave real performance on the table.`),

    t(`## Label smoothing

Replace the one-hot target with $(1-\\epsilon)$ on the true class and $\\epsilon/(K-1)$ elsewhere. Typically
$\\epsilon=0.1$.

This stops the model from driving the correct logit to $+\\infty$, which improves calibration and slightly improves
accuracy. The cost: it deliberately blurs the representation, which measurably hurts if you plan to *distill* from the
model — the teacher's inter-class similarity structure is exactly what a student learns from, and smoothing erases it.`),

    warn(`**Regularizers interact, and stacking them blindly does not work.** BatchNorm already regularizes (via batch
noise), so BatchNorm + heavy dropout often underperforms either alone. Strong augmentation reduces the need for
dropout. Tune one at a time against a validation set; the standard failure is adding four regularizers at once and
concluding none of them help.`),

    code('Dropout, mixup, and label smoothing', `import numpy as np
rng = np.random.default_rng(0)

# --- inverted dropout ---
def dropout(x, p, training=True):
    if not training or p == 0: return x
    mask = (rng.random(x.shape) > p) / (1 - p)     # scale survivors
    return x * mask

x = np.ones((4, 8))
d = dropout(x, 0.5)
print("dropout p=0.5:")
print(" mean before:", x.mean(), " mean after:", round(d.mean(), 3),
      " <- preserved in expectation")

# --- mixup ---
def mixup(X, Y, alpha=0.2):
    lam = rng.beta(alpha, alpha)
    idx = rng.permutation(len(X))
    return lam*X + (1-lam)*X[idx], lam*Y + (1-lam)*Y[idx], lam

X = rng.normal(size=(6, 3)); Y = np.eye(3)[rng.integers(0, 3, 6)]
Xm, Ym, lam = mixup(X, Y)
print(f"\\nmixup lambda={lam:.3f}")
print(" a mixed label:", Ym[0].round(3), " <- not one-hot any more")

# --- label smoothing ---
def smooth(Y, eps=0.1):
    K = Y.shape[1]
    return Y*(1-eps) + eps/K

print("\\nlabel smoothing eps=0.1:")
print(" one-hot :", np.eye(3)[0])
print(" smoothed:", smooth(np.eye(3)[None, 0], 0.1)[0].round(4))

# effect on the optimal logit gap
for eps in [0.0, 0.05, 0.1, 0.2]:
    p = 1 - eps + eps/3
    print(f" eps={eps}: target prob {p:.3f} -> "
          f"optimal logit gap {np.log(p/((1-p)/2)) if p<1 else np.inf:.2f}")`),

    quiz('Why is dropout often set to 0 when pretraining a large language model?',
      ['With trillions of tokens the model is data-limited, not capacity-limited; dropout would waste capacity',
       'Dropout is incompatible with attention',
       'It slows training down too much',
       'LayerNorm already implements dropout'],
      0,
      'Dropout fights overfitting, and in a single-epoch pretraining run over trillions of unique tokens there is essentially nothing to overfit to — every batch is new data. Spending capacity on an ensemble effect you do not need costs quality. Dropout returns for fine-tuning, where you may see the same small dataset dozens of times.'),
  ],
  refs: [
    paper('Dropout: A Simple Way to Prevent Neural Networks from Overfitting', 'Srivastava et al.', 2014, 'https://jmlr.org/papers/v15/srivastava14a.html', ''),
    paper('mixup: Beyond Empirical Risk Minimization', 'Zhang et al.', 2017, 'https://arxiv.org/abs/1710.09412', 'Surprisingly effective, surprisingly simple.'),
    paper('When Does Label Smoothing Help?', 'Müller, Kornblith & Hinton', 2019, 'https://arxiv.org/abs/1906.02629', 'Including why it hurts distillation.'),
    paper('RandAugment', 'Cubuk et al.', 2019, 'https://arxiv.org/abs/1909.13719', 'Augmentation search reduced to two hyperparameters.'),
    paper('A Simple Framework for Contrastive Learning of Visual Representations', 'Chen et al.', 2020, 'https://arxiv.org/abs/2002.05709', 'SimCLR — augmentation as the entire training signal.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'nn-losses-training',
  title: 'Loss Functions and the Training Loop',
  sub: 'Choosing what to minimize, and the practical mechanics of making it converge.',
  mins: 22, level: 'core',
  prereq: ['nn-backprop', 'math-optimization'],
  tags: ['losses', 'training'],
  sections: [
    t(`## Every loss is a distributional assumption

From the [probability lesson](#/l/math-probability): minimizing a loss is maximum likelihood under some noise model.

| Loss | Noise model | Optimal prediction |
|---|---|---|
| Squared error | Gaussian | conditional **mean** |
| Absolute error | Laplace | conditional **median** |
| Cross-entropy | Categorical | conditional **probabilities** |
| Poisson NLL | Poisson | conditional **rate** |
| Quantile / pinball | asymmetric Laplace | conditional **quantile** |

That third column is worth remembering. If your target is skewed and you train with MSE, the model predicts the mean —
which may be a value the target never takes.`),

    viz('loss-functions'),

    t(`## Robustness

**Huber loss** is quadratic near zero and linear far away, controlled by $\\delta$:

$$\\mathcal{L}_\\delta(r) = \\begin{cases}\\tfrac12 r^2 & |r|\\le\\delta\\\\ \\delta(|r|-\\tfrac12\\delta) & \\text{otherwise}\\end{cases}$$

Differentiable everywhere (unlike L1) and outlier-tolerant (unlike L2). It is the default in object detection
(as smooth-L1) and worth trying whenever your targets have a heavy tail.

**Focal loss** down-weights easy examples: $-(1-p_t)^\\gamma\\log p_t$. Designed for extreme class imbalance in dense
object detection, where the loss is otherwise dominated by tens of thousands of trivially-correct background boxes.`),

    t(`## Contrastive and ranking losses

When you need a *representation* rather than a prediction:

- **InfoNCE** — treat matching pairs as positives against in-batch negatives, cross-entropy over similarities.
  Powers CLIP and SimCLR.
- **Triplet loss** — $\\max(0, d(a,p)-d(a,n)+m)$. Anchor closer to positive than negative by a margin.
- **Bradley–Terry** — $-\\log\\sigma(r(\\text{chosen})-r(\\text{rejected}))$. This is the reward model objective in RLHF,
  and DPO's loss is a reparameterization of it.

Notice these are all logistic-regression-shaped. Once you see cross-entropy on a difference of scores, you can read
most of them at a glance.`),

    t(`## The training loop

\`\`\`
for epoch in epochs:
    for batch in loader:
        loss = criterion(model(batch.x), batch.y)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
\`\`\`

Points where this goes wrong in practice:

- **Forgetting \`zero_grad()\`** — gradients accumulate, so you silently train on a running sum. Classic.
- **Not clipping** — a single bad batch produces a huge gradient and destroys the run. Clip global norm to ~1.0.
- **Wrong scheduler granularity** — stepping per-epoch when the schedule expects per-step.
- **train/eval mode** — dropout and BatchNorm behave differently. Forgetting \`model.eval()\` corrupts your validation
  numbers.
- **Gradient accumulation** — to simulate a large batch on small hardware, accumulate over $k$ micro-batches before
  stepping, and remember to divide the loss by $k$.`),

    viz('batch-size'),

    t(`## Debugging a model that will not train

In order. Do not skip steps.

1. **Overfit a single batch.** Take 8 examples, turn off regularization and augmentation, and train until the loss is
   ~0. If it cannot, you have a bug — not a tuning problem. This finds more bugs than anything else.
2. **Check the initial loss.** For $K$-way classification it should be $\\log K$ (2.30 for 10 classes). If it is not,
   the head or the labels are wrong.
3. **Sweep the learning rate** over powers of 10. The correct value is usually within a factor of 3 of where the loss
   diverges.
4. **Watch gradient norms per layer.** Vanishing or exploding will be obvious.
5. **Look at the data.** Actually look at it: render the images, print the tokenized text, check the label
   distribution. A shocking share of "model bugs" are data bugs.
6. **Simplify.** Remove augmentation, remove the fancy scheduler, shrink the model. Add complexity back one piece at a
   time.`),

    code('A complete, correct training loop in NumPy', `import numpy as np
rng = np.random.default_rng(0)

# two moons
n = 600
th = rng.uniform(0, np.pi, n)
X = np.where((np.arange(n) % 2)[:, None],
             np.c_[1-np.cos(th)-0.5, -np.sin(th)+0.3],
             np.c_[np.cos(th)-0.5,    np.sin(th)-0.3]) * 1.6
X += rng.normal(0, 0.12, X.shape)
y = (np.arange(n) % 2).astype(float)[:, None]
Xtr, ytr, Xva, yva = X[:450], y[:450], X[450:], y[450:]

sizes = [2, 24, 24, 1]
P = [(rng.normal(0, np.sqrt(2/a), (a, b)), np.zeros(b))
     for a, b in zip(sizes[:-1], sizes[1:])]
M = [(np.zeros_like(W), np.zeros_like(b)) for W, b in P]   # momentum buffers

def forward(P, X):
    acts, pres, h = [X], [], X
    for i, (W, b) in enumerate(P):
        z = h @ W + b
        pres.append(z)
        h = 1/(1+np.exp(-np.clip(z,-500,500))) if i == len(P)-1 else np.maximum(0, z)
        acts.append(h)
    return acts, pres

def evaluate(P, X, y):
    h = forward(P, X)[0][-1]
    loss = -np.mean(y*np.log(h+1e-12) + (1-y)*np.log(1-h+1e-12))
    return loss, np.mean((h > .5) == y)

BS, LR, EPOCHS = 32, 0.15, 200
best, best_P, patience = np.inf, None, 0

for ep in range(EPOCHS):
    perm = rng.permutation(len(Xtr))
    for s in range(0, len(Xtr), BS):
        idx = perm[s:s+BS]
        xb, yb = Xtr[idx], ytr[idx]
        acts, pres = forward(P, xb)
        delta = (acts[-1] - yb) / len(xb)

        grads = []
        for i in reversed(range(len(P))):
            grads.append((acts[i].T @ delta, delta.sum(0)))
            if i > 0:
                delta = (delta @ P[i][0].T) * (pres[i-1] > 0)
        grads.reverse()

        # global gradient clipping
        gn = np.sqrt(sum((gW**2).sum() + (gb**2).sum() for gW, gb in grads))
        scale = min(1.0, 1.0 / (gn + 1e-8))

        # cosine schedule + momentum
        lr = LR * 0.5 * (1 + np.cos(np.pi * ep / EPOCHS))
        newP, newM = [], []
        for (W, b), (gW, gb), (mW, mb) in zip(P, grads, M):
            mW = 0.9*mW + gW*scale; mb = 0.9*mb + gb*scale
            newP.append((W - lr*mW, b - lr*mb)); newM.append((mW, mb))
        P, M = newP, newM

    vl, va = evaluate(P, Xva, yva)
    if vl < best - 1e-4:
        best, best_P, patience = vl, [(W.copy(), b.copy()) for W, b in P], 0
    else:
        patience += 1
    if ep % 40 == 0:
        tl, ta = evaluate(P, Xtr, ytr)
        print(f"epoch {ep:3d}  lr {lr:.4f}  train {tl:.4f}/{ta:.3f}  val {vl:.4f}/{va:.3f}")
    if patience > 30:
        print(f"early stop at epoch {ep}"); break

print(f"\\nbest val loss {best:.4f}, accuracy {evaluate(best_P, Xva, yva)[1]:.3f}")`,
      'Everything discussed above, in one runnable script: minibatching, momentum, gradient clipping, a cosine schedule, early stopping with patience, and best-checkpoint restoration.'),

    quiz('Your 10-class classifier starts training at loss 6.9 instead of 2.3. What is the most likely cause?',
      ['The output layer or the labels are misconfigured — random guessing over 10 classes should give log(10)=2.30',
       'The learning rate is too high',
       'The model is too small',
       'Normal — initial loss varies'],
      0,
      'At initialization the model should be uniform over classes, giving $-\\log(1/10) = 2.303$. A loss of 6.9 is $\\log(1000)$, which strongly suggests the head has 1000 outputs (a copied ImageNet config?) or the labels are misaligned. **Always check the initial loss** — it is a free, instant assertion about your setup.'),
  ],
  refs: [
    blog('A Recipe for Training Neural Networks', 'Andrej Karpathy', 2019, 'http://karpathy.github.io/2019/04/25/recipe/', 'The best practical guide to debugging training that has been written. Read it twice.'),
    paper('Focal Loss for Dense Object Detection', 'Lin et al.', 2017, 'https://arxiv.org/abs/1708.02002', ''),
    paper('Accurate, Large Minibatch SGD', 'Goyal et al.', 2017, 'https://arxiv.org/abs/1706.02677', 'The linear scaling rule and warmup, at ImageNet scale.'),
    paper('An Empirical Model of Large-Batch Training', 'McCandlish et al.', 2018, 'https://arxiv.org/abs/1812.06162', 'The gradient noise scale — how to predict the critical batch size.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'nn-cnn',
  title: 'Convolutional Networks',
  sub: 'Weight sharing, locality, and the architecture that started the deep learning era.',
  mins: 26, level: 'core',
  prereq: ['nn-backprop'],
  tags: ['CNN', 'vision'],
  sections: [
    t(`## The problem with dense layers on images

A $224\\times224\\times3$ image flattened is 150,528 numbers. A dense layer to 1000 units needs 150 million parameters —
for one layer. And it would learn to recognize a cat in the top-left corner completely independently of a cat in the
centre, because it has no notion that those are the same thing shifted.

Two structural facts about images fix this:

1. **Locality** — a pixel's meaning depends mostly on its neighbours.
2. **Translation equivariance** — an edge is an edge wherever it appears.`),

    viz('convolution'),

    t(`## The convolution layer

Slide a small kernel over the input, computing a dot product at each position:

$$(I * K)_{ij} = \\sum_{m}\\sum_{n} I_{i+m,\\,j+n}\\,K_{m,n}$$

The same kernel weights are used at every position — that is **weight sharing**, and it is where the parameter savings
and the equivariance both come from. A $3\\times3$ kernel is 9 parameters regardless of image size.

Key hyperparameters:

- **Kernel size** — 3×3 is the near-universal choice; two stacked 3×3 see 5×5 with fewer parameters and an extra
  nonlinearity.
- **Stride** — step size. Stride 2 halves the spatial dimensions.
- **Padding** — "same" preserves size, "valid" shrinks.
- **Channels** — a layer with $C_{\\text{in}}$ input and $C_{\\text{out}}$ output channels has
  $C_{\\text{in}} \\times C_{\\text{out}} \\times k^2$ weights.`),

    viz('pooling-receptive-field'),

    t(`## Pooling and the receptive field

**Max pooling** takes the max over a window: downsamples, and adds a little translation *invariance* (as opposed to
equivariance). Modern architectures often use strided convolutions instead, letting the network learn its own
downsampling.

The **receptive field** — how much of the input one output neuron can see — grows with depth. Stacking $L$ layers of
$k\\times k$ convolutions gives a receptive field of $1 + L(k-1)$. This is why depth matters for vision: you need enough
of it for the final layer to see the whole object.

Dilated (atrous) convolutions grow the receptive field exponentially by inserting gaps in the kernel — no extra
parameters, no lost resolution.`),

    t(`## The architectural lineage

**LeNet-5** (1998) — Yann LeCun, handwritten digits, deployed on real cheque-reading systems. Conv, pool, conv, pool,
dense. Everything is already here.

**AlexNet** (2012) — the same idea at scale, with ReLU, dropout, and two GPUs. Cut ImageNet top-5 error from 26% to
15%, and started the modern era essentially overnight.

**VGG** (2014) — the observation that stacking small 3×3 kernels beats using large ones. Uniform, simple, and still a
common feature extractor.

**GoogLeNet / Inception** (2014) — parallel branches at multiple kernel sizes; 1×1 convolutions as cheap channel-mixing
and dimensionality reduction.

**ResNet** (2015) — the one that mattered most. Add a skip connection so each block learns a *residual*:

$$\\mathbf{y} = \\mathcal{F}(\\mathbf{x}) + \\mathbf{x}$$

Suddenly 152 layers trained where 30 had failed. The reason is visible in equation (2) of backprop: the gradient
through the skip path is exactly 1, so it cannot vanish. Residual connections are now in essentially every deep
architecture, transformers included.

**DenseNet, EfficientNet, ConvNeXt** — later refinements: dense connectivity, compound scaling of depth/width/
resolution, and finally ConvNeXt, which modernized a ResNet with transformer-era design choices and matched ViTs.
A useful data point: much of the ViT-over-CNN gap was training recipe, not architecture.`),

    t(`## Depthwise separable convolutions

Factor a standard convolution into two cheaper steps:

1. **Depthwise** — one $k\\times k$ kernel per input channel, no mixing.
2. **Pointwise** — a $1\\times1$ convolution mixing channels.

Cost drops from $C_{\\text{in}}C_{\\text{out}}k^2$ to $C_{\\text{in}}k^2 + C_{\\text{in}}C_{\\text{out}}$ — roughly $8$–$9\\times$
less for $3\\times3$ with typical channel counts. This is the core of MobileNet and every efficient vision model since.`),

    code('Convolution, im2col, and receptive fields', `import numpy as np

def conv2d(x, k, stride=1, pad=0):
    """Direct convolution (actually cross-correlation, like every DL framework)."""
    if pad: x = np.pad(x, pad)
    kh, kw = k.shape
    H = (x.shape[0]-kh)//stride + 1
    W = (x.shape[1]-kw)//stride + 1
    out = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            out[i,j] = (x[i*stride:i*stride+kh, j*stride:j*stride+kw] * k).sum()
    return out

img = np.zeros((12, 12)); img[3:9, 3:9] = 1.0
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float)
print("vertical edge response (nonzero only at the box edges):")
print(conv2d(img, sobel_x, pad=1).astype(int))

# --- parameter counting ---
def conv_params(cin, cout, k): return cin*cout*k*k + cout
def dense_params(h, w, c, out): return h*w*c*out + out

print(f"\\n3x3 conv, 64->128 channels: {conv_params(64,128,3):,} params")
print(f"dense 56x56x64 -> 128:      {dense_params(56,56,64,128):,} params")

# --- depthwise separable ---
cin, cout, k = 256, 512, 3
std = cin*cout*k*k
sep = cin*k*k + cin*cout
print(f"\\nstandard conv:  {std:,}")
print(f"separable conv: {sep:,}  ({std/sep:.1f}x cheaper)")

# --- receptive field growth ---
print("\\nlayer  receptive field")
rf, jump = 1, 1
for L in range(1, 9):
    rf += 2 * jump            # 3x3 kernel, stride 1
    print(f"{L:5d}  {rf:3d} x {rf}")`),

    quiz('Why do modern CNNs stack 3×3 convolutions instead of using 7×7 kernels?',
      ['Three stacked 3×3 layers see the same 7×7 region with fewer parameters and two extra nonlinearities',
       '3×3 kernels are faster on GPUs',
       'Large kernels cause overfitting',
       '7×7 kernels cannot detect edges'],
      0,
      'Three 3×3 layers have a 7×7 receptive field using $3\\times 9 = 27$ weights per channel pair versus $49$ — 45% fewer — and insert two nonlinearities where the big kernel has none. This is the central argument of the VGG paper. (Interestingly, ConvNeXt partially reversed this with 7×7 depthwise kernels, imitating attention\'s larger receptive field — the depthwise factorization makes them cheap again.)'),
  ],
  refs: [
    paper('Gradient-Based Learning Applied to Document Recognition', 'LeCun et al.', 1998, 'http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf', 'LeNet. The foundation.'),
    paper('ImageNet Classification with Deep CNNs', 'Krizhevsky, Sutskever & Hinton', 2012, 'https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html', 'AlexNet. The paper that started it.'),
    paper('Deep Residual Learning for Image Recognition', 'He et al.', 2015, 'https://arxiv.org/abs/1512.03385', 'ResNet. Among the most-cited papers in all of science.'),
    paper('A ConvNet for the 2020s', 'Liu et al.', 2022, 'https://arxiv.org/abs/2201.03545', 'Modernizes a ResNet step by step until it matches ViTs. Excellent controlled ablation.'),
    course('CS231n: Convolutional Neural Networks for Visual Recognition', 'Karpathy, Li, Johnson (Stanford)', 2016, 'https://cs231n.github.io/', 'The best CNN course notes ever written. Still worth reading.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'nn-rnn',
  title: 'Recurrent Networks, LSTMs, and Seq2Seq',
  sub: 'How sequences were handled before attention — and why the ideas still matter.',
  mins: 24, level: 'core',
  prereq: ['nn-backprop'],
  tags: ['RNN', 'LSTM', 'sequences'],
  sections: [
    t(`## Recurrence

Process a sequence one element at a time, carrying a hidden state:

$$\\mathbf{h}_t = \\phi(W_{hh}\\mathbf{h}_{t-1} + W_{xh}\\mathbf{x}_t + \\mathbf{b})$$

The same weights at every step, which means the model handles any length and shares statistical strength across
positions. Training uses **backpropagation through time** — unroll the recurrence and apply standard backprop.`),

    viz('rnn-unroll'),

    t(`## Why vanilla RNNs fail

The gradient from step $T$ back to step $t$ carries a product of $T-t$ Jacobians, each roughly $W_{hh}^{\\mathsf T}$
times an activation derivative. So it scales like $\\lambda^{T-t}$ where $\\lambda$ is the largest eigenvalue.

- $\\lambda < 1$: gradients vanish. Information from early tokens cannot reach the loss.
- $\\lambda > 1$: gradients explode. NaN.

There is no good value. Gradient clipping handles the explosion; the vanishing side is structural, and it is why plain
RNNs cannot learn dependencies more than ~10 steps long.`),

    t(`## LSTM: an additive memory path

Hochreiter & Schmidhuber's 1997 fix: add a **cell state** that is updated by addition rather than by matrix
multiplication, with learned gates deciding what to write, keep, and read.

$$\\begin{aligned}
\\mathbf{f}_t &= \\sigma(W_f[\\mathbf{h}_{t-1},\\mathbf{x}_t]) && \\text{forget: what to keep}\\\\
\\mathbf{i}_t &= \\sigma(W_i[\\mathbf{h}_{t-1},\\mathbf{x}_t]) && \\text{input: what to write}\\\\
\\tilde{\\mathbf{c}}_t &= \\tanh(W_c[\\mathbf{h}_{t-1},\\mathbf{x}_t]) && \\text{candidate content}\\\\
\\mathbf{c}_t &= \\mathbf{f}_t\\odot\\mathbf{c}_{t-1} + \\mathbf{i}_t\\odot\\tilde{\\mathbf{c}}_t && \\text{the highway}\\\\
\\mathbf{o}_t &= \\sigma(W_o[\\mathbf{h}_{t-1},\\mathbf{x}_t]) && \\text{output gate}\\\\
\\mathbf{h}_t &= \\mathbf{o}_t\\odot\\tanh(\\mathbf{c}_t)
\\end{aligned}$$`),

    viz('lstm-gates'),

    key(`The line that matters is $\\mathbf{c}_t = \\mathbf{f}_t\\odot\\mathbf{c}_{t-1}+\\ldots$ — **addition, gated by a
value the network controls**. If $\\mathbf{f}_t \\approx 1$, the gradient path is multiplication by 1, repeatedly, and
information survives hundreds of steps.

This is exactly the same idea as a residual connection, invented eighteen years earlier for a different reason.
Set the figure's forget gate to 1.0 and compare against the vanilla RNN.`),

    t(`**GRU** (2014) merges the forget and input gates into one and drops the separate cell state. Fewer parameters,
usually indistinguishable in quality.

Practical note: initialize the forget gate bias to 1 so the cell *remembers by default* and must learn to forget.
Without this, LSTMs train noticeably worse.`),

    t(`## Seq2seq and the bottleneck

For translation, the 2014 design was: an encoder RNN reads the source into a fixed-size vector, a decoder RNN generates
the target from it.

The problem is obvious in hindsight. **Every sentence, however long, must fit through one vector.** Performance
degraded sharply past ~20 words.

Bahdanau et al. (2014) fixed it by letting the decoder look back at *all* encoder states, weighted by learned
relevance:

$$\\mathbf{c}_t = \\sum_i \\alpha_{ti}\\,\\mathbf{h}_i, \\qquad \\alpha_{ti} = \\text{softmax}_i(\\text{score}(\\mathbf{s}_t,\\mathbf{h}_i))$$

**This is attention**, and it was introduced as a patch on the RNN bottleneck. Three years later, "Attention Is All You
Need" removed the recurrence and kept the patch.`),

    t(`## Where RNNs stand now

Transformers won because of **parallelism**, not because attention is inherently better at modeling sequences. An RNN
must process token $t$ before $t+1$; a transformer processes the whole sequence at once during training. On modern
hardware that is decisive.

But the RNN's advantage — $O(1)$ state per token instead of a growing KV cache — has come back into focus for long
context. Modern **state-space models** (S4, Mamba) are linear recurrences engineered to be parallelizable during
training via a scan, giving RNN-like inference cost with transformer-like training throughput.`),

    code('An RNN and an LSTM cell, and what happens to their gradients', `import numpy as np
rng = np.random.default_rng(0)

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

# --- gradient decay through a vanilla RNN ---
def rnn_grad(T, w):
    h, pres = 0.0, []
    for t in range(T):
        z = w*h + (1.0 if t == 0 else 0.0)
        pres.append(z); h = np.tanh(z)
    g = 1.0
    for t in reversed(range(T)):
        g *= w * (1 - np.tanh(pres[t])**2)
    return g

print("gradient from step T back to step 0, vanilla RNN:")
for w in [0.5, 0.9, 1.0, 1.1]:
    print(f"  w={w}: " + "  ".join(f"T={T}: {rnn_grad(T,w):.2e}" for T in [10, 50, 100]))

# --- the same, through an LSTM cell state ---
print("\\ngradient through the LSTM cell state (product of forget gates):")
for f in [0.5, 0.9, 0.99, 1.0]:
    print(f"  f={f}: " + "  ".join(f"T={T}: {f**T:.2e}" for T in [10, 50, 100]))

# --- an LSTM cell, forward ---
H, D = 8, 4
Wf, Wi, Wc, Wo = (rng.normal(0, .3, (H+D, H)) for _ in range(4))
bf = np.ones(H)                       # forget bias = 1: remember by default
h, c = np.zeros(H), np.zeros(H)
for t in range(6):
    x = rng.normal(0, 1, D)
    z = np.r_[h, x]
    f = sigmoid(z @ Wf + bf)
    i = sigmoid(z @ Wi)
    g = np.tanh(z @ Wc)
    o = sigmoid(z @ Wo)
    c = f*c + i*g
    h = o*np.tanh(c)
    print(f"t={t}  mean forget {f.mean():.3f}  |c| {np.linalg.norm(c):.3f}  |h| {np.linalg.norm(h):.3f}")`),

    quiz('What is the essential difference between an LSTM cell state update and a vanilla RNN hidden state update?',
      ['The cell state is updated additively and gated, so gradients flow through multiplication by ~1 rather than by a weight matrix',
       'The LSTM has more parameters',
       'The LSTM uses tanh instead of sigmoid',
       'The LSTM processes the sequence in both directions'],
      0,
      'Vanilla: $h_t = \\phi(Wh_{t-1}+\\ldots)$ — the gradient repeatedly multiplies by $W$ and $\\phi\'$, so it decays or explodes geometrically. LSTM: $c_t = f_t\\odot c_{t-1}+\\ldots$ — with $f_t\\approx1$ the gradient path is multiplication by 1. Additive updates with learned gates. It is the same insight that residual connections rediscovered in 2015.'),
  ],
  refs: [
    paper('Long Short-Term Memory', 'Hochreiter & Schmidhuber', 1997, 'https://www.bioinf.jku.at/publications/older/2604.pdf', 'The original. Well ahead of its time.'),
    blog('Understanding LSTM Networks', 'Christopher Olah', 2015, 'https://colah.github.io/posts/2015-08-Understanding-LSTMs/', 'The diagrams everyone uses. Still the best explanation.'),
    paper('Neural Machine Translation by Jointly Learning to Align and Translate', 'Bahdanau, Cho & Bengio', 2014, 'https://arxiv.org/abs/1409.0473', 'Attention, three years before the transformer.'),
    blog('The Unreasonable Effectiveness of Recurrent Neural Networks', 'Andrej Karpathy', 2015, 'http://karpathy.github.io/2015/05/21/rnn-effectiveness/', 'A time capsule from when char-RNNs were astonishing. Worth reading for perspective on how fast this moves.'),
    paper('On the difficulty of training Recurrent Neural Networks', 'Pascanu, Mikolov & Bengio', 2013, 'https://arxiv.org/abs/1211.5063', 'The rigorous analysis of vanishing/exploding gradients, and gradient clipping.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'nn-embeddings',
  title: 'Embeddings and Representation Learning',
  sub: 'Turning discrete symbols into geometry.',
  mins: 20, level: 'core',
  prereq: ['nn-perceptron-mlp'],
  tags: ['embeddings', 'word2vec'],
  sections: [
    t(`## The problem with one-hot

Represent a 50,000-word vocabulary as one-hot vectors and every pair of words is equidistant and orthogonal. "cat" is
exactly as similar to "dog" as it is to "bureaucracy." All semantic structure is thrown away before learning starts.

An **embedding** replaces this with a learned dense vector $\\mathbf{e}_w \\in \\mathbb{R}^d$ with $d\\approx 100$–$4096$.
Implementation is a lookup table: a matrix of shape (vocab, $d$), indexed by token id. Mathematically it is a matrix
multiply with a one-hot vector, which is why gradients flow to exactly one row.`),

    viz('embedding-space'),

    t(`## word2vec and the distributional hypothesis

> "You shall know a word by the company it keeps." — J. R. Firth, 1957

**Skip-gram**: given a word, predict its context words. **CBOW**: given the context, predict the word. Either way, the
embedding is a byproduct — the useful thing is the hidden layer, not the prediction.

The practical trick that made it work at scale is **negative sampling**: instead of a softmax over 50,000 words per
step, treat it as binary classification against a handful of randomly sampled "negative" words. Turns an $O(V)$ update
into $O(k)$.

The famous result is that relationships become **directions**: $\\text{king}-\\text{man}+\\text{woman}\\approx\\text{queen}$.
Worth being honest about it — analogy accuracy in real embeddings is far shakier than the celebrated examples suggest,
and the evaluation protocol excludes the input words from the answer, which is doing meaningful work.`),

    warn(`Embeddings absorb whatever is in the training text, including its biases. Bolukbasi et al. found
"man:computer programmer :: woman:homemaker" in embeddings trained on Google News. This is not a quirk to be patched
away — it is the model faithfully learning statistical regularities of the corpus. Debiasing methods exist and are
partial at best; the more durable response is to be careful about what you train on and where you deploy.`),

    t(`## Static vs contextual

word2vec and GloVe give **one vector per word type**. "bank" gets a single vector averaging river banks and financial
institutions.

**ELMo** (2018) and then **BERT** (2018) produce **contextual** embeddings: the vector for "bank" depends on the
sentence it appears in. This was a genuine phase change in NLP, and it is what modern transformer embeddings are.

Present practice: static embeddings only as an input layer (transformers still have one), while what people call "an
embedding" for search or clustering is a *contextual* sentence-level vector from a model like E5, BGE, or
text-embedding-3.`),

    t(`## Embeddings in practice

- **Tied weights.** Many language models share the input embedding matrix with the output projection. Saves
  $V\\times d$ parameters (substantial — for a 128k vocab at $d$=4096 that is 500M) and usually helps.
- **Retrieval.** Embed documents and queries into one space, then do approximate nearest-neighbour search (HNSW,
  IVF-PQ). This is the "R" in RAG.
- **Categorical features.** Embeddings replaced one-hot encoding for high-cardinality categoricals in tabular deep
  learning, and are standard in recommender systems.
- **Multimodal.** CLIP puts images and text in a *shared* embedding space, which is what enables zero-shot
  classification and text-to-image retrieval.
- **Normalization.** For cosine similarity, L2-normalize first. Most retrieval systems do this by default.`),

    code('Training embeddings with skip-gram + negative sampling', `import numpy as np
rng = np.random.default_rng(0)

corpus = ("the cat sat on the mat . the dog sat on the rug . "
          "the cat chased the mouse . the dog chased the cat . "
          "a mouse ate the cheese . the cat ate the mouse .").split()
vocab = sorted(set(corpus))
w2i = {w: i for i, w in enumerate(vocab)}
V, D, WIN = len(vocab), 12, 2

# skip-gram pairs
pairs = [(w2i[corpus[i]], w2i[corpus[j]])
         for i in range(len(corpus))
         for j in range(max(0, i-WIN), min(len(corpus), i+WIN+1)) if i != j]

Ein = rng.normal(0, 0.1, (V, D))     # center-word embeddings
Eout = rng.normal(0, 0.1, (V, D))    # context embeddings

def sigmoid(z): return 1/(1+np.exp(-np.clip(z, -30, 30)))

for epoch in range(400):
    for c, o in rng.permutation(pairs):
        negs = rng.integers(0, V, 5)
        targets = np.r_[o, negs]
        labels = np.r_[1.0, np.zeros(5)]
        v, U = Ein[c], Eout[targets]
        p = sigmoid(U @ v)
        g = (p - labels)
        Ein[c] -= 0.05 * (g @ U)
        Eout[targets] -= 0.05 * np.outer(g, v)

E = Ein / (np.linalg.norm(Ein, axis=1, keepdims=True) + 1e-9)
def near(w, k=4):
    sims = E @ E[w2i[w]]
    idx = np.argsort(-sims)[1:k+1]
    return [(vocab[i], round(float(sims[i]), 3)) for i in idx]

for w in ["cat", "sat", "the"]:
    print(f"{w:8s} -> {near(w)}")
print("\\nA tiny corpus gives noisy results, but 'cat'/'dog' and 'sat'/'ate'")
print("should already be pulling together — they appear in similar contexts.")`),

    quiz('Why did contextual embeddings (BERT) represent such a large improvement over word2vec?',
      ['A word gets a different vector in each context, resolving polysemy that a single static vector cannot',
       'They are trained on more data',
       'They have higher dimensionality',
       'They are faster to compute'],
      0,
      'word2vec assigns one vector per word *type*, so "bank" must simultaneously mean riverbank and financial institution — the vector lands somewhere unhelpful in between. A transformer produces a vector per *token occurrence*, conditioned on the sentence, so the two senses separate cleanly. Scale and data helped, but this structural change is the core of it.'),
  ],
  refs: [
    paper('Efficient Estimation of Word Representations in Vector Space', 'Mikolov et al.', 2013, 'https://arxiv.org/abs/1301.3781', 'word2vec.'),
    paper('Distributed Representations of Words and Phrases', 'Mikolov et al.', 2013, 'https://arxiv.org/abs/1310.4546', 'Negative sampling — the trick that made it scale.'),
    paper('GloVe: Global Vectors for Word Representation', 'Pennington, Socher & Manning', 2014, 'https://nlp.stanford.edu/projects/glove/', 'Matrix-factorization view of the same idea.'),
    paper('Man is to Computer Programmer as Woman is to Homemaker?', 'Bolukbasi et al.', 2016, 'https://arxiv.org/abs/1607.06520', 'Bias in embeddings, and partial mitigations.'),
    paper('Deep contextualized word representations', 'Peters et al.', 2018, 'https://arxiv.org/abs/1802.05365', 'ELMo. The move to contextual.'),
  ],
},

];

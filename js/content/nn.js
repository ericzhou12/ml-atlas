/* ============================================================
   Track 3 — Neural networks
   ============================================================ */

import { t, key, intuition, warn, hist, mathnote, viz, deriv, code, quiz, paper, book, course, blog, video, demo, codeRef,
         tldr, recap, jargon, steps, diagram } from './_helpers.js';

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
    tldr(`A neural network is linear regression, stacked — with one crucial ingredient between the layers.

That ingredient is a **nonlinearity**, and without it the whole edifice collapses: a hundred stacked linear
layers can be multiplied out into a single linear layer, so the depth buys you exactly nothing. Add a
nonlinearity between them and the same stack can approximate any function at all.

The other thing worth taking from this lesson is a reframing. A neural network is logistic regression that
**invents its own features** instead of making you hand-engineer them. The last layer is a plain linear
classifier; everything before it exists to build the features that classifier gets to use.`),

    jargon([
      ['perceptron', 'The 1958 ancestor: weighted sum, then a hard threshold. One layer, no hidden units.'],
      ['MLP', 'Multi-Layer Perceptron. The standard feedforward neural network: alternating linear layers and nonlinearities. Also called a "fully connected network" or "dense network".'],
      ['layer', 'One linear transformation (a matrix multiply plus a bias) usually followed by a nonlinearity.'],
      ['hidden layer', 'Any layer that is not the input or the output. "Hidden" only because you never directly observe its values.'],
      ['unit / neuron', 'One output of a layer — one number. The biological analogy is very loose; think of it as a learned feature detector.'],
      ['activation function $\\phi$', 'The nonlinearity applied elementwise after each linear layer. ReLU, sigmoid, tanh, GELU.'],
      ['ReLU', 'Rectified Linear Unit: $\\max(0, x)$. Passes positives through unchanged, zeroes negatives. The default for two decades.'],
      ['width / depth', 'How many units per layer / how many layers. The two ways to make a network bigger.'],
      ['XOR', 'Exclusive-or: true when exactly one input is true. The simplest function a single-layer network provably cannot compute.'],
      ['linearly separable', 'Splittable by a single straight line. XOR is not.'],
      ['universal approximation', 'The theorem that a wide enough single hidden layer can approximate any continuous function. True, and less useful than it sounds.'],
    ]),

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

The fix is to stack. Apply a linear map, apply a nonlinearity, apply another linear map, and so on:

$$\\mathbf{h}_1 = \\phi(W_1\\mathbf{x}+\\mathbf{b}_1), \\quad
\\mathbf{h}_2 = \\phi(W_2\\mathbf{h}_1+\\mathbf{b}_2), \\quad \\ldots, \\quad
\\hat y = W_L\\mathbf{h}_{L-1}+\\mathbf{b}_L$$

Read one of those as a sentence: *take the previous layer's output, multiply by a weight matrix, add a bias,
then squash each element through $\\phi$.* Every layer is the same two operations. That is genuinely the entire
architecture — a modern transformer is this pattern with attention interleaved.

Now, the nonlinearity $\\phi$. It is easy to read as optional decoration, and it is the load-bearing part.
Without it, layers collapse:

$$W_2(W_1\\mathbf{x}) = (W_2W_1)\\mathbf{x} = W'\\mathbf{x}$$

Two matrices multiplied together are just... another matrix. Which is the [composition](#/l/math-matrices)
result from the maths track, and here it is fatal: **a hundred layers without activations is exactly as
expressive as one.** All that depth, and you have reinvented linear regression.

The nonlinearity breaks that collapse. $\\phi(W_1\\mathbf{x})$ cannot be absorbed into the next matrix, so each
layer genuinely adds something.`),

    diagram('Why the nonlinearity is not optional',
`<svg viewBox="0 0 620 210" role="img" aria-label="Stacked linear layers collapse; layers with nonlinearities do not">
  <text class="dtitle" x="20" y="24" style="fill: var(--s6)">WITHOUT a nonlinearity — the layers collapse</text>
  <g>
    <rect x="20" y="36" width="58" height="34" rx="4" style="fill: color-mix(in srgb, var(--s6) 12%, transparent); stroke: var(--s6)"/>
    <text class="dmono" x="49" y="57" text-anchor="middle" style="fill: var(--s6)">W1</text>
    <path d="M80,53 L96,53" style="stroke: var(--border); stroke-width: 1.4"/>
    <rect x="98" y="36" width="58" height="34" rx="4" style="fill: color-mix(in srgb, var(--s6) 12%, transparent); stroke: var(--s6)"/>
    <text class="dmono" x="127" y="57" text-anchor="middle" style="fill: var(--s6)">W2</text>
    <path d="M158,53 L174,53" style="stroke: var(--border); stroke-width: 1.4"/>
    <rect x="176" y="36" width="58" height="34" rx="4" style="fill: color-mix(in srgb, var(--s6) 12%, transparent); stroke: var(--s6)"/>
    <text class="dmono" x="205" y="57" text-anchor="middle" style="fill: var(--s6)">W3</text>
    <text x="258" y="57" style="font-size:15px; fill: var(--text-dim)">=</text>
    <rect x="282" y="36" width="90" height="34" rx="4" style="fill: color-mix(in srgb, var(--s6) 22%, transparent); stroke: var(--s6); stroke-width: 1.8"/>
    <text class="dmono" x="327" y="57" text-anchor="middle" style="fill: var(--s6)">one matrix</text>
    <text class="dlabel" x="392" y="57">all that depth bought nothing</text>
  </g>
  <text class="dtitle" x="20" y="124" style="fill: var(--s3)">WITH a nonlinearity — no collapse is possible</text>
  <g>
    <rect x="20" y="136" width="50" height="34" rx="4" style="fill: color-mix(in srgb, var(--s1) 12%, transparent); stroke: var(--s1)"/>
    <text class="dmono" x="45" y="157" text-anchor="middle" style="fill: var(--s1)">W1</text>
    <circle cx="88" cy="153" r="13" style="fill: color-mix(in srgb, var(--s3) 20%, transparent); stroke: var(--s3)"/>
    <text class="dmono" x="88" y="157" text-anchor="middle" style="fill: var(--s3)">φ</text>
    <rect x="110" y="136" width="50" height="34" rx="4" style="fill: color-mix(in srgb, var(--s1) 12%, transparent); stroke: var(--s1)"/>
    <text class="dmono" x="135" y="157" text-anchor="middle" style="fill: var(--s1)">W2</text>
    <circle cx="178" cy="153" r="13" style="fill: color-mix(in srgb, var(--s3) 20%, transparent); stroke: var(--s3)"/>
    <text class="dmono" x="178" y="157" text-anchor="middle" style="fill: var(--s3)">φ</text>
    <rect x="200" y="136" width="50" height="34" rx="4" style="fill: color-mix(in srgb, var(--s1) 12%, transparent); stroke: var(--s1)"/>
    <text class="dmono" x="225" y="157" text-anchor="middle" style="fill: var(--s1)">W3</text>
    <text x="272" y="157" style="font-size:15px; fill: var(--text-dim)">≠</text>
    <text class="dlabel" x="300" y="150" style="fill: var(--s3)">the phi blocks the merge,</text>
    <text class="dlabel" x="300" y="168" style="fill: var(--s3)">so every layer earns its keep</text>
  </g>
</svg>`,
      `The circles are the only thing separating a deep network from a linear model. Everything difficult about
training — vanishing gradients, initialization, normalization — is a consequence of those circles being there,
and worth it every time.`),

    key(`Try it yourself below: set the activation to **identity** and try to fit the spiral or XOR datasets. It will
never work, at any width or depth. Then switch to ReLU.`),

    viz('mlp-playground'),

    steps('Two hidden units solving XOR, by hand', [
      { h: 'The problem', md: `XOR is 1 when exactly one input is 1: the points $(0,1)$ and $(1,0)$ are positive, while $(0,0)$ and $(1,1)$ are negative. Plot those four points and the two positives sit on opposite corners of the square, so no straight line can put them on one side and the negatives on the other.` },
      { h: 'Build a unit that detects "at least one input is on"', md: `Set $h_1 = \\text{ReLU}(x_1 + x_2 - 0.5)$. On the four inputs this gives $0,\\ 0.5,\\ 0.5,\\ 1.5$ — zero only for $(0,0)$.` },
      { h: 'Build a unit that detects "both inputs are on"', md: `Set $h_2 = \\text{ReLU}(x_1 + x_2 - 1.5)$. This gives $0,\\ 0,\\ 0,\\ 0.5$ — nonzero only for $(1,1)$.` },
      { h: 'Subtract one from the other', md: `Take $\\hat y = 2h_1 - 4h_2$, giving $0,\\ 1,\\ 1,\\ 1$ for the first three... and $2(1.5) - 4(0.5) = 1$ for the last. Not right yet — so scale the second unit harder: $\\hat y = 2h_1 - 6h_2$ gives $0,\\ 1,\\ 1,\\ 0$. That is XOR, exactly.` },
      { h: 'What just happened', md: `The two hidden units re-described each input as a pair of numbers $(h_1, h_2)$, and in *those* coordinates the four points became linearly separable — the output layer is a plain weighted sum, and it worked. Nothing was added but a change of description. That is what "the hidden layer invents features" means in the smallest possible case.` },
    ]),


    t(`## What the hidden layer is doing

Two complementary readings. Neither is more correct; having both available is what lets you reason about
architectures.

**Reading 1 — feature learning.** Look at the last layer of any classifier: it is a plain linear model. So
everything before it exists to *transform the input into features on which a linear model works*.

That is the reframe worth carrying: **a neural network is logistic regression that invents its own features.**
Before deep learning, that invention was a human job — you sat down and engineered SIFT descriptors for images
or TF-IDF vectors for text, then fed them to a linear classifier. The network does the same job by gradient
descent, and does it better.

**Reading 2 — piecewise linear tiling.** With ReLU, each unit contributes a *hinge*: flat zero until
$\\mathbf{w}^{\\mathsf T}\\mathbf{x}+b$ crosses zero, then a straight ramp. One hinge is a crude thing. But sum
a few hundred of them, each with its own kink location and slope, and you can trace any curve to any accuracy —
the same way enough short straight segments approximate a circle.

This reading explains something you can see in the figure below: a ReLU network's output is *literally*
piecewise linear, made of flat facets. It never produces a smooth curve, only a fine enough approximation of
one.`),

    viz('hidden-units'),

    t(`## Universal approximation, and what it does not say

**Theorem** (Cybenko 1989, Hornik 1991): a network with a single hidden layer can approximate any continuous
function, to any accuracy you name, given enough hidden units — provided the activation is not a polynomial, and
provided you only care about a bounded region of input space rather than all the way out to infinity.

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
      '$W_{10}(W_9(\\cdots W_1\\mathbf{x})) = (W_{10}\\cdots W_1)\\mathbf{x} = W_{\\text{eff}}\\mathbf{x}$ — one matrix, so one linear layer. Depth without a nonlinearity buys exactly nothing in what the model can represent. It does change how *training* behaves, which is a real research topic in its own right, but no amount of training can make a linear model fit XOR.'),

    recap(`- Explain what XOR proved about single-layer models, and what it did *not* prove about multilayer ones.
- Show algebraically why stacked linear layers collapse, and say what the nonlinearity prevents.
- Describe a network as "logistic regression that learns its own features", and identify which layer is the
  logistic regression.
- Explain a ReLU network's output as a sum of hinges, and say why it is always piecewise linear.
- State the universal approximation theorem *and* the three things it does not promise.
- Give two reasons depth beats width in practice.`),
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
    tldr(`Training needs the gradient of the loss with respect to *every* parameter — a billion numbers. The
obvious approach (nudge each parameter, see what happens to the loss) needs a billion forward passes and is
hopeless.

**Backpropagation gets all of them in one backward pass**, costing roughly the same as a single forward pass.
The trick is not clever calculus; it is bookkeeping. The chain rule says gradients multiply along a chain, and
if you walk the chain from the output backwards you can reuse every partial product instead of recomputing it.

If you understand one algorithm in this atlas properly, make it this one. Everything from vanishing gradients
to residual connections to why your GPU runs out of memory follows from it.`),

    jargon([
      ['backpropagation', 'The algorithm for computing all parameter gradients efficiently. Short for "backward propagation of errors".'],
      ['forward pass', 'Running the network input → output, computing and caching each layer\'s values.'],
      ['backward pass', 'Walking back output → input, computing gradients using the cached values.'],
      ['$\\delta$ (delta)', 'The gradient of the loss with respect to a layer\'s pre-activation. The quantity being propagated backwards. Sometimes called the "error signal".'],
      ['local derivative', 'How one operation\'s output responds to its own input. Each node only needs to know this about itself.'],
      ['pre-activation $\\mathbf{z}$', 'The value after the matrix multiply but *before* the activation function.'],
      ['activation $\\mathbf{a}$', 'The value after the activation function. What gets passed to the next layer.'],
      ['$\\odot$', 'Elementwise multiplication (Hadamard product). Multiply matching entries; do **not** do a matrix multiply.'],
      ['computation graph', 'The record of which operations produced which values during the forward pass. What autodiff walks backwards.'],
      ['autodiff', 'Automatic differentiation — the framework machinery that generalises backprop to any program you write.'],
      ['residual connection', 'A shortcut that adds a layer\'s input to its output: $\\mathbf{y} = \\mathbf{x} + f(\\mathbf{x})$. Gives gradients a direct path backwards.'],
    ]),

    t(`## The problem

A network has a billion parameters and produces one number: the loss. You need
$\\partial\\mathcal{L}/\\partial\\theta$ for every single parameter.

The naive approach is to nudge each parameter a little and see how the loss moves — a finite difference. That
needs one forward pass *per parameter*. At a billion parameters and, say, 50 ms per forward pass, one gradient
step would take about 18 months.

Backpropagation computes **all** of them in a single backward pass costing roughly what one forward pass costs.
Not a constant-factor improvement — a factor of a billion.

Where does that come from? A single observation. The chain rule says the gradient at an early layer is a product
of local derivatives along the whole path back from the loss. Nearby parameters share almost all of that path. So
if you compute the products *once*, walking backwards from the loss and accumulating, every parameter gets its
gradient from work that was already done for the parameters after it. **Backprop is dynamic programming applied
to the chain rule.**`),

    t(`## The two-phase structure

**Forward** — run the network normally, but cache each layer's output. You will need those values on the way
back, which is why memory usage is what it is.

**Backward** — start from the loss with $\\delta = 1$ (the derivative of the loss with respect to itself), and
propagate it backwards. At each layer, use the incoming $\\delta$ plus the cached forward values to (a) compute
that layer's parameter gradients and (b) work out the $\\delta$ to hand to the layer before it.`),

    viz('backprop-graph'),

    key(`Move the sliders in that figure and watch the numbers propagate. The thing worth noticing is how
**local** every step is.

Each node needs exactly two pieces of information:
1. Its **local derivative** — what this one operation does to its own input. A multiplication node knows the
   derivative of multiplication. It does not know or care what the rest of the network is.
2. The **incoming gradient** — how much the loss cares about this node's output. Handed to it from downstream.

It multiplies them together, keeps what it needs for its own parameters, and passes the rest upstream. **That is
the entire algorithm.** No node ever has a global view, which is exactly why it scales to a billion parameters
and why frameworks can differentiate arbitrary code you write.`),

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

To get from layer $\\ell+1$ back to layer $\\ell$, the gradient is multiplied by a weight matrix and then by an
activation derivative. Do that thirty times and the gradient reaching layer 0 has been multiplied by sixty
factors.

If those factors average below 1, the product **vanishes** exponentially — early layers receive nothing and stop
learning. If they average above 1, it **explodes** into \\\`NaN\\\`. There is no third option, and "roughly 1" is not
somewhere a randomly initialized network lands by accident. **You have to engineer the factors**, and that is
exactly what initialization schemes, normalization layers, and residual connections are for.`),

    viz('vanishing-gradients'),

    t(`Play with that figure to make it concrete. Switch the activation to tanh with a gain of 0.5 and watch the
gradient reaching layer 0 collapse to around $10^{-8}$ — that layer is receiving a signal eight orders of
magnitude weaker than the last layer, so for all practical purposes it is frozen at its initialization.

Now turn on **residual connections** and watch the curve flatten immediately.

Here is why that works, and it is a one-line argument. A residual block computes $\\mathbf{y} = \\mathbf{x} +
f(\\mathbf{x})$ instead of $\\mathbf{y} = f(\\mathbf{x})$. Differentiate: the derivative of the sum is the
derivative of $f$ **plus 1**. So equation (2) picks up an additive identity term,

$$\\boldsymbol\\delta^\\ell = \\boldsymbol\\delta^{\\ell+1} + (\\text{the usual multiplied term})$$

and an addition cannot decay the way a product can. However badly the multiplicative path behaves, the gradient
still has a clean route home. That single change is what made networks with hundreds of layers trainable, and it
is why residual connections appear in essentially every architecture built since 2015.`),

    t(`## Automatic differentiation

Everything above was derived for a specific architecture — dense layers with elementwise activations. Nobody
does that by hand any more.

Backprop is a special case of **reverse-mode automatic differentiation** (from [the Jacobian
lesson](#/l/math-jacobian)). Frameworks record a graph of primitive operations as you run the forward pass, with
each primitive knowing only its own local derivative rule. Then they walk that graph in reverse, applying the
same two-step logic at every node.

The consequence is that **you never write a derivative**. You write the forward computation — in ordinary Python,
with loops and branches and whatever else — and the derivative of the whole composition comes for free. This is
why the deep learning explosion followed the frameworks rather than preceding them: PyTorch and JAX did not make
new mathematics possible, they made trying a new architecture a ten-minute job instead of a two-week
derivation.`),

    warn(`The memory cost is real and it is usually what stops you. Every forward activation must stay alive
until the backward pass consumes it, so peak memory scales with depth × batch size × sequence length. For a
large transformer, activations routinely exceed the weights themselves.

Symptom: your model fits in memory and the forward pass runs fine, then you get an out-of-memory error during
\\\`.backward()\\\`. That is not a leak — it is the design.

**Gradient checkpointing** is the standard escape: store only every $k$-th layer's activations and recompute the
rest on the way back. Roughly $\\sqrt{L}$ memory instead of $L$, for about 30% more compute. It is one flag in
most frameworks and worth reaching for before you reduce the batch size.`),

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

    recap(`- Explain why finite differences cannot work at scale, with an order-of-magnitude estimate.
- Describe backprop as dynamic programming over the chain rule — reusing shared partial products.
- Say what information a single node needs to do its part, and why that locality is the reason it scales.
- Read equation (2) and predict vanishing versus exploding gradients from the size of the per-layer factors.
- Explain in one line why a residual connection fixes vanishing gradients — an addition, not a product.
- Diagnose "forward pass fine, OOM on backward" and name the fix.`),
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
    tldr(`The activation function is a one-line choice — \\\`nn.ReLU()\\\` — that decides whether a deep network
trains at all.

The rule for reasoning about it is short: **judge an activation by its derivative, not by its shape.** Backprop
multiplies by that derivative once per layer, so a function whose derivative is often near zero will strangle
the gradient before it reaches the early layers. Sigmoid's derivative maxes out at 0.25, which is why deep
sigmoid networks were untrainable for two decades. ReLU's is exactly 1, which is why swapping it in was one of
the changes that made deep learning work.`),

    jargon([
      ['activation function', 'The nonlinearity applied elementwise after each linear layer. Written $\\phi$ or $\\sigma$.'],
      ['saturate', 'Flatten out. A saturating function has near-zero derivative for large inputs, so gradients passing through it die.'],
      ['zero-centred', 'Outputs are roughly balanced around 0 rather than all positive. Matters for the *direction* gradient updates can take.'],
      ['dying ReLU', 'A unit whose input has become permanently negative. Its output is always 0, so its gradient is always 0, so it can never recover. Dead for the rest of training.'],
      ['sparsity (of activations)', 'Many outputs being exactly zero. ReLU produces this naturally — roughly half of its outputs are 0.'],
      ['gate', 'A value between 0 and 1 multiplied into a signal, controlling how much passes through. The core mechanism of LSTMs and GLU-family activations.'],
      ['monotonic', 'Always increasing. Most activations are; GELU and Swish dip slightly below zero near the origin, which turns out to help.'],
      ['GELU / SiLU / Swish', 'Smooth, near-identical modern activations. GELU is standard in BERT and GPT; SiLU is the cheaper twin.'],
      ['SwiGLU', 'A gated activation used in Llama, PaLM, and most recent LLMs. Two branches, one gating the other.'],
      ['softplus', 'A smooth version of ReLU, $\\log(1+e^x)$. Always positive, never exactly zero.'],
    ]),

    t(`## What to look at

The instinct is to judge an activation by how its curve looks. That is the wrong thing to look at.

Look at its **derivative**. Backprop multiplies the gradient by $\\phi'$ once at every layer
([equation 2](#/l/nn-backprop)), so over 30 layers you are multiplying by 30 copies of it. The function's shape
determines what the network can express; its *derivative's* shape determines whether you can ever train it.

Three properties, in order of importance:

1. **Does it saturate?** If $\\phi' \\to 0$ for large inputs, then any unit that drifts into that region stops
   receiving gradient — and stops learning, possibly permanently. This is the property that killed sigmoid.
2. **Is it zero-centred?** If a layer's outputs are always positive, then by equation (3) every weight gradient
   in the next layer shares the same sign. The update can only move all weights up together or all down
   together, so it zig-zags toward the optimum instead of going straight there.
3. **Is it cheap?** This sounds trivial and is not. The activation runs on every unit of every layer of every
   example of every step. \\\`max(0, x)\\\` is one comparison; an exponential is roughly twenty times more
   expensive.`),

    viz('activations'),

    t(`## The lineage

**Sigmoid** $\\sigma(x)=1/(1+e^{-x})$. Historically first, and a bad hidden activation: max derivative 0.25, saturates
at both ends, not zero-centred. Still correct as an *output* for binary probabilities and for gates (LSTM, GLU).

**tanh**. Zero-centred sigmoid, derivative up to 1. Better, still saturates. Standard through the 1990s and 2000s, and
still used in some RNNs.

**ReLU** $\\max(0,x)$. One of the handful of changes that made deep learning work, and almost embarrassingly
simple: pass positives through untouched, zero out negatives.

Check it against the three criteria. Its derivative is exactly **1** for every positive input — the gradient
passes through completely undamaged, no matter how many layers it crosses. It is zero for negatives, which
sounds bad but means roughly half the units are inactive on any given input, giving genuine sparsity. And it is
a single comparison, which is as cheap as an operation gets.

Its failure mode is **dying ReLU**, and it is worth understanding because it is permanent. If a unit's input
becomes negative for every example in the dataset, its output is always 0, so its gradient is always 0, so its
weights never change, so its input stays negative. Forever. A learning rate spike can push a substantial
fraction of a layer into this state in a single step, and no amount of further training brings them back. If
you find a layer where a third of the units never fire, this is what happened.

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

    recap(`- Say why you judge an activation by its **derivative**, and connect that to backprop's equation (2).
- Explain why sigmoid makes deep networks untrainable, using the number 0.25.
- Describe the dying-ReLU failure mode and why it is irreversible.
- Say what "zero-centred" buys you, in terms of the signs of a layer's weight gradients.
- Pick a sensible activation for a hidden layer, a binary output, and a multiclass output, and justify each.
- Explain what the gate in SwiGLU is doing, and why the hidden dimension gets shrunk to compensate.`),
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
    tldr(`Before training starts, every weight is a random number. Which random numbers turns out to matter a
great deal — the wrong scale and a deep network will not train at all, no matter how long you wait.

The reason is compounding. Each layer multiplies the *variance* of its signal by some factor. If that factor is
1.2, then after 40 layers the signal is $1.2^{40} \\approx 1500\\times$ bigger. If it is 0.8, the signal has
been reduced to $10^{-4}$ of its original size. You need the factor to be almost exactly 1, and there is a
formula that arranges it.`),

    jargon([
      ['initialization', 'The random values weights start at, before any training. Not arbitrary — the scale is derived.'],
      ['symmetry breaking', 'Making units differ from each other so they can learn different things. Identical units stay identical forever.'],
      ['variance', 'The spread of a set of numbers. Here, the spread of activations within a layer — the thing being kept constant across depth.'],
      ['fan-in / fan-out', 'How many inputs a unit receives / how many outputs it feeds. $n_\\text{in}$ and $n_\\text{out}$.'],
      ['Xavier / Glorot init', 'Scale weights by $1/n_\\text{in}$. Designed for symmetric activations like tanh.'],
      ['He / Kaiming init', 'Scale weights by $2/n_\\text{in}$. The factor of 2 compensates for ReLU discarding half the signal. The default for ReLU networks.'],
      ['signal propagation', 'Tracking how activation and gradient magnitudes change as they pass through layers. The lens this whole lesson uses.'],
    ]),

    t(`## Why zero fails, and why "small random" is not enough

Two failure modes, and the second is the interesting one.

**All zeros.** The obvious choice, and it destroys the network completely. If every weight is identical, then
every unit in a layer computes exactly the same function, receives exactly the same gradient, and updates by
exactly the same amount. They stay identical forever. Your 512-unit layer has the expressive capacity of one
unit, permanently. This is why initialization must be **random** at all: randomness is what breaks the symmetry
between units.

**Random but badly scaled.** This one is subtler and it is what the rest of the lesson is about.

Take a linear layer with $n$ inputs and weights drawn independently with variance $\\sigma^2$. If the inputs have
unit variance, then each output is a sum of $n$ independent terms, and variances of independent things add:

$$\\text{Var}(z_i) = \\sum_{j=1}^{n}\\text{Var}(W_{ij}x_j) = n\\sigma^2$$

So each layer multiplies the signal's variance by $n\\sigma^2$. That is a *constant factor applied once per
layer*, which means over $L$ layers the signal is scaled by $(n\\sigma^2)^L$ — exponential in depth.

Unless $n\\sigma^2$ is essentially exactly 1, activations explode or collapse geometrically. Off by 10% in
either direction and a 50-layer network is broken.`),

    viz('initialization'),

    t(`## The two schemes

Set $n\\sigma^2 = 1$ and solve for $\\sigma^2$. That is the whole idea; the two named schemes differ only in a
correction for what the activation function does.

**Xavier/Glorot** (2010) — for symmetric activations like tanh, which are roughly linear near zero and do not
throw signal away:

$$\\sigma^2 = \\frac{2}{n_{\\text{in}}+n_{\\text{out}}} \\qquad\\text{or, in the simpler form,}\\qquad \\sigma^2=\\frac{1}{n_{\\text{in}}}$$

(The first version averages fan-in and fan-out, because you want the *forward* signal and the *backward* gradient
both preserved, and those pull in slightly different directions.)

**He/Kaiming** (2015) — for ReLU. ReLU sets half its inputs to exactly zero, which halves the variance passing
through. So you need to start with twice as much:

$$\\sigma^2 = \\frac{2}{n_{\\text{in}}}$$

**That factor of 2 is the entire difference between the two schemes.** It sounds like a rounding error. It is
not: $2^{L/2}$ compounds, and getting it wrong is the difference between a 30-layer network training and a
30-layer network producing garbage. This one correction is what made very deep networks trainable in 2015,
before residual connections existed to paper over the problem.`),

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

    recap(`- Say why all-zero initialization destroys a layer, in terms of symmetry.
- Derive the variance factor $n\\sigma^2$ per layer, and explain why it compounds exponentially with depth.
- State both initialization formulas and explain where ReLU's factor of 2 comes from.
- Explain why initialization matters less now than it did in 2015, and what took over the job.`),
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
    tldr(`Initialization gets your activations to a sensible scale at step 0. Normalization layers keep them
there for the rest of training, by re-centring and re-scaling the values as they flow through.

All the variants — BatchNorm, LayerNorm, RMSNorm, GroupNorm — do the *same arithmetic* and differ only in
**which set of numbers gets averaged together**. Once you see that, the alphabet soup collapses into one idea
with four groupings.

Worth knowing: normalization unambiguously works, and *why* it works is still argued about. The explanation in
the original paper has been substantially undermined and the field kept using the technique anyway.`),

    jargon([
      ['normalization layer', 'A layer that rescales its inputs to have roughly zero mean and unit variance, then applies a learned scale and shift.'],
      ['$\\mu$, $\\sigma^2$', 'The mean and variance computed over some set of values. Which set is what distinguishes the variants.'],
      ['$\\gamma$, $\\beta$ (gamma, beta)', 'Learned scale and shift applied after normalizing. Let the network choose its own output scale rather than being forced to unit variance.'],
      ['$\\epsilon$ (epsilon)', 'A tiny constant added inside the square root so you never divide by zero. Typically $10^{-5}$.'],
      ['batch', 'The group of examples processed together in one step.'],
      ['feature / channel', 'One dimension of a layer\'s output vector. A layer with 512 units has 512 features.'],
      ['BatchNorm', 'Normalizes each feature *across the batch*. Standard in CNNs.'],
      ['LayerNorm', 'Normalizes across the features *within one example*. Standard in transformers.'],
      ['RMSNorm', 'LayerNorm without the mean subtraction. Cheaper, works just as well, used by most recent LLMs.'],
      ['residual stream', 'The running sum carried through a network by residual connections — the "main highway" that each block reads from and writes to.'],
      ['pre-norm / post-norm', 'Whether normalization is applied *before* a block\'s computation or *after* adding its output back. Sounds trivial, decides whether very deep models train.'],
      ['warmup', 'Starting with a tiny learning rate and ramping it up over the first few thousand steps.'],
    ]),

    t(`## The operation

Every normalization layer runs the same two steps:

$$\\underbrace{\\hat x = \\frac{x-\\mu}{\\sqrt{\\sigma^2+\\epsilon}}}_{\\text{1. force zero mean, unit variance}}, \\qquad \\underbrace{y = \\gamma\\hat x + \\beta}_{\\text{2. let the network pick a new scale}}$$

Step 1 is a standard statistical z-score: subtract the mean, divide by the standard deviation. Whatever came in,
what comes out has mean 0 and variance 1.

Step 2 looks like it undoes step 1, and asking why it is there is the right question. Two answers:

**Expressiveness.** Forcing every layer's output to unit variance is a real restriction — some functions genuinely
need a large-magnitude intermediate value. With learnable $\\gamma$ and $\\beta$, the network *can* recover any
scale it wants, so nothing is lost.

**Control.** The difference is that the scale is now an explicit parameter, learned by gradient descent, rather
than an emergent property of forty layers of weight matrices compounding. It cannot drift. That is the whole
benefit: not that the values are normalized, but that their scale is under direct control.

Everything else in this lesson is a question about **which numbers go into computing $\\mu$ and $\\sigma^2$.**`),

    viz('normalization'),

    diagram('The only thing that differs: which numbers get averaged',
`<svg viewBox="0 0 620 210" role="img" aria-label="BatchNorm averages down a column, LayerNorm across a row">
  <text class="dlabel" x="24" y="20">rows = examples in the batch &nbsp;·&nbsp; columns = features</text>
  <g transform="translate(24,34)">
    <text class="dtitle" x="0" y="0" style="fill: var(--s1)">BatchNorm — down each column</text>
    <g>
      <rect x="0" y="12" width="34" height="26" style="fill: color-mix(in srgb, var(--s1) 30%, transparent); stroke: var(--border)"/>
      <rect x="36" y="12" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
      <rect x="72" y="12" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
      <rect x="108" y="12" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
      <rect x="0" y="40" width="34" height="26" style="fill: color-mix(in srgb, var(--s1) 30%, transparent); stroke: var(--border)"/>
      <rect x="36" y="40" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
      <rect x="72" y="40" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
      <rect x="108" y="40" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
      <rect x="0" y="68" width="34" height="26" style="fill: color-mix(in srgb, var(--s1) 30%, transparent); stroke: var(--border)"/>
      <rect x="36" y="68" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
      <rect x="72" y="68" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
      <rect x="108" y="68" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
    </g>
    <text class="dlabel" x="0" y="118" style="fill: var(--s1)">one mean per feature,</text>
    <text class="dlabel" x="0" y="134" style="fill: var(--s1)">pooled over the whole batch</text>
    <text class="dlabel" x="0" y="156">→ examples become coupled</text>
    <text class="dlabel" x="0" y="172">→ breaks with small batches</text>
  </g>
  <g transform="translate(330,34)">
    <text class="dtitle" x="0" y="0" style="fill: var(--s3)">LayerNorm — across each row</text>
    <g>
      <rect x="0" y="12" width="34" height="26" style="fill: color-mix(in srgb, var(--s3) 30%, transparent); stroke: var(--border)"/>
      <rect x="36" y="12" width="34" height="26" style="fill: color-mix(in srgb, var(--s3) 30%, transparent); stroke: var(--border)"/>
      <rect x="72" y="12" width="34" height="26" style="fill: color-mix(in srgb, var(--s3) 30%, transparent); stroke: var(--border)"/>
      <rect x="108" y="12" width="34" height="26" style="fill: color-mix(in srgb, var(--s3) 30%, transparent); stroke: var(--border)"/>
      <rect x="0" y="40" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
      <rect x="36" y="40" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
      <rect x="72" y="40" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
      <rect x="108" y="40" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
      <rect x="0" y="68" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
      <rect x="36" y="68" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
      <rect x="72" y="68" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
      <rect x="108" y="68" width="34" height="26" style="fill: var(--bg-elev); stroke: var(--border)"/>
    </g>
    <text class="dlabel" x="0" y="118" style="fill: var(--s3)">one mean per example,</text>
    <text class="dlabel" x="0" y="134" style="fill: var(--s3)">over its own features</text>
    <text class="dlabel" x="0" y="156">→ examples stay independent</text>
    <text class="dlabel" x="0" y="172">→ batch size irrelevant</text>
  </g>
</svg>`,
      `Same formula, different axis. Everything people find confusing about normalization variants — why
transformers use LayerNorm, why BatchNorm needs running averages at inference, why BatchNorm behaves oddly with
small batches — falls out of which direction the shading runs.`),

    t(`## The variants

**BatchNorm** (2015) normalizes each feature across the batch. Transformative for CNNs, and it comes with real
baggage — all three problems below trace directly to the fact that it mixes information *between examples*:`),

    t(`- **Behaviour depends on batch size**, degrading badly below about 16, because the mean of 4 numbers is a
  noisy estimate of anything.
- **Train and inference differ.** At inference you often have one example, so there is no batch to average
  over. BatchNorm keeps running averages from training and uses those instead — which means the layer computes
  a *different function* in \\\`train()\\\` and \\\`eval()\\\` mode. Forgetting to switch modes is a classic and
  maddening bug.
- **It couples examples together.** One example's prediction depends on which others happened to share its
  batch. That is wrong for sequences, awkward for distributed training (the batch is split across GPUs), and
  makes exact reproducibility harder.

**LayerNorm** (2016) normalizes across features *within* each example. None of the three problems above can
occur: batch size is irrelevant, train and test are identical, and examples stay independent. **This is why
transformers use it** — variable-length sequences and per-example independence are non-negotiable there.

**RMSNorm** (2019) drops the mean subtraction entirely, keeping only the rescaling:

$$y = \\frac{x}{\\sqrt{\\frac{1}{n}\\sum_i x_i^2 + \\epsilon}}\\cdot g$$

One less pass over the data and one less thing to store. It turns out the centring was doing almost none of the
work — the rescaling is the part that matters. Used by Llama, T5, Gemma, and most recent models. A good example
of the field discovering that a component everyone assumed was load-bearing was not.

**GroupNorm** splits channels into groups and normalizes within each — a middle ground for vision tasks like
segmentation and detection, where memory forces batches of 2 or 4 and BatchNorm falls apart.`),

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

This section is here because the honest answer is instructive about how the field operates.

The original BatchNorm paper proposed **internal covariate shift**: as earlier layers update, the distribution
of inputs each later layer sees keeps drifting, and normalizing stabilises it. Clean story, plausible mechanism,
and it is the explanation most tutorials still give.

It has been substantially undermined. Santurkar et al. ran the decisive experiment: *deliberately inject*
covariate shift immediately after the BatchNorm layer — adding random noise that re-destabilises the
distributions — and BatchNorm still helps just as much. If the stated mechanism were the real one, that should
have destroyed the benefit.

The better-supported explanation is that normalization **smooths the loss landscape**: it reduces how fast the
loss and its gradient can change, which means larger learning rates stay stable. Connect that to the
[stability bound](#/l/math-optimization) $\\eta < 2/\\lambda_{\\max}$ — normalization lowers $\\lambda_{\\max}$,
so it directly raises the learning rate ceiling.

There is also a scale-invariance effect worth flagging because it causes real confusion. After normalization,
the loss does not care about the *scale* of the weights feeding into it — double them and the normalization
divides the doubling right back out. This makes weight decay interact with normalized layers in genuinely
non-obvious ways, since shrinking weights no longer shrinks the function.

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

    recap(`- Write the normalization formula and explain what the learnable $\\gamma$ and $\\beta$ are for.
- Say which axis each variant averages over, and derive BatchNorm's three problems from that one fact.
- Explain why RMSNorm dropping the mean subtraction costs nothing.
- State the difference between pre-norm and post-norm and say which one lets you train very deep models
  without warmup.
- Describe why the original "internal covariate shift" explanation is not well supported, and what experiment
  undermined it.`),
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
    tldr(`Deep networks have far more capacity than their training sets can constrain, so something has to stop
them memorising. The classical penalty terms still apply, but the techniques that matter most in deep learning
do not look like penalties at all.

**Dropout** randomly deletes units during training. **Data augmentation** manufactures new training examples
from old ones. **Early stopping** just quits before the overfitting starts. And the honest summary of the
theory tying these together is that it is thin — most of what is known here is empirical, and the advice
changes as models and datasets grow.`),

    jargon([
      ['dropout', 'Randomly setting a fraction of units to zero during training. Off at inference.'],
      ['$p$ (dropout rate)', 'The probability a given unit is zeroed. 0.1–0.5 typically; 0 in most modern LLM pretraining.'],
      ['inverted dropout', 'The standard implementation: scale up the survivors during training so inference needs no adjustment at all.'],
      ['co-adaptation', 'Units learning to only work in specific combinations, so no unit is individually meaningful. What dropout is meant to prevent.'],
      ['weight decay', 'L2 regularization, in its deep learning name. Shrinks weights slightly on every step.'],
      ['data augmentation', 'Generating new training examples by transforming existing ones in ways that do not change the label.'],
      ['label smoothing', 'Training toward 0.9 instead of 1.0 for the correct class, so the model never becomes infinitely confident.'],
      ['early stopping', 'Halting training when validation loss stops improving, rather than at a fixed step count.'],
      ['Mixup / CutMix', 'Augmentations that blend two training images *and* their labels together.'],
      ['effective capacity', 'How complex a function the model actually ends up using, as opposed to how complex a function it could represent.'],
    ]),

    t(`## Dropout

During training, zero out each unit independently with probability $p$, and scale the survivors up by
$1/(1-p)$. At inference, do nothing at all — the full network runs.

Deliberately breaking your own network sounds like sabotage. Two readings explain why it helps.`),

    viz('dropout'),

    t(`**Reading 1 — an ensemble in disguise.** Each training step effectively trains a *different* thinned
network, since a different random subset of units is present. With $n$ units there are $2^n$ possible
subnetworks, all sharing the same weights. At inference, running the full network with everything present
approximates averaging over that entire ensemble.

And ensembling reliably reduces variance ([as with random forests](#/l/ml-trees-ensembles)) — you are getting
that benefit for the price of one model.

**Reading 2 — breaking co-adaptation.** Without dropout, units can settle into arrangements where unit 7 is only
meaningful when unit 12 is also active, each correcting the other's quirks. That is a brittle way to store
information: it fits the training set and shatters on anything new.

With dropout, no unit can rely on any specific partner being present, because that partner may be gone on any
given step. Features are pushed toward being individually useful.

**Why the $1/(1-p)$ scaling?** If you zero 30% of units, the layer's output sums are about 30% smaller — so the
next layer sees systematically different magnitudes during training than at inference. Scaling survivors up by
$1/(1-p)$ restores the expected value, so the two match. This is called *inverted* dropout because the older
convention scaled *down* at inference instead; inverted is now universal since it keeps inference code free of
dropout entirely.`),

    warn(`**Dropout's status has changed, and much online advice is out of date.**

It was close to essential in the 2012–2016 era: small datasets, large networks, no normalization layers,
overfitting everywhere. Modern large-scale transformer pretraining frequently sets it to **exactly 0**.

The reason is that dropout addresses overfitting, and overfitting is not the binding constraint when you train
on trillions of tokens for a single epoch — the model never sees an example twice, so there is nothing to
memorise. Dropout in that regime just throws away effective capacity you paid for.

It comes back for **fine-tuning on small datasets**, where you make many passes over a few thousand examples and
overfitting is entirely real again. So: pretraining, usually 0; fine-tuning, usually 0.1 or more.`),

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

The most reliable regularizer of the lot, and it is worth understanding *why* it is more reliable than the
others.

Dropout and weight decay are generic — they constrain the model without knowing anything about your problem.
Augmentation is different: it injects **real knowledge about the world** in the form of a specific claim, *these
transformations do not change the label*. A horizontally flipped cat is still a cat. That is a fact about cats,
supplied by you, that the model would otherwise have to discover from data.

Which means augmentation is only as good as the claim is true. Horizontally flipping a photo of a cat is fine;
horizontally flipping a photo of text or a road sign destroys the label. Rotating a digit by 180° turns a 6 into
a 9. **Every augmentation is an assertion about invariance, and a wrong one actively teaches your model
something false.**

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

    recap(`- Give both readings of dropout — implicit ensemble, and breaking co-adaptation — and explain the
  $1/(1-p)$ scaling.
- Say when dropout is and is not appropriate, and why the answer changed between 2015 and now.
- Name the two things to get right about weight decay in deep learning (decoupled form, excluded parameters).
- Explain why augmentation is more reliable than generic regularizers, and what makes a *bad* augmentation
  actively harmful.
- Describe how self-supervised methods turn augmentation from a regularizer into the entire training signal.`),
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
    tldr(`Two things that are usually taught separately and belong together.

First, **choosing a loss is choosing what your model will predict.** Not "how accurate it will be" — literally
*what quantity comes out*. Train with squared error and you get the conditional mean; train with absolute error
and you get the median. On skewed data those are different numbers, and one of them may be an answer that never
actually occurs.

Second, the training loop itself: a short, fixed sequence of operations whose ordering matters, plus the
handful of practical safeguards (gradient clipping, warmup, checkpointing) that separate a run that finishes
from a run that dies at 3am.`),

    jargon([
      ['loss / criterion / objective', 'Three words for the same thing: the number being minimised. PyTorch says `criterion`, papers say `objective`.'],
      ['conditional mean / median', 'The average / middle value of $y$ among examples with this particular $\\mathbf{x}$. Which one your model outputs is decided by the loss.'],
      ['MSE / MAE', 'Mean Squared Error / Mean Absolute Error. The squared and absolute versions of "how far off were you".'],
      ['Huber loss', 'Squared error near zero, absolute error far away. Smooth *and* outlier-tolerant.'],
      ['focal loss', 'A loss that automatically down-weights examples the model already gets right, so rare hard cases are not drowned out.'],
      ['contrastive loss', 'A loss on *pairs*: pull matching things together in embedding space, push non-matching apart.'],
      ['InfoNCE', 'The standard contrastive loss. Cross-entropy over similarities, with other items in the batch as negatives. Behind CLIP and SimCLR.'],
      ['negatives', 'The non-matching items a contrastive loss pushes away from. "In-batch negatives" means reusing the other examples in the same batch, for free.'],
      ['gradient clipping', 'Capping the gradient\'s magnitude before stepping, so one anomalous batch cannot destroy the run.'],
      ['warmup', 'Ramping the learning rate up from near-zero over the first few thousand steps.'],
      ['scheduler', 'The rule that changes the learning rate over time — warmup then decay, typically.'],
      ['checkpoint', 'A saved snapshot of weights and optimizer state, so a crashed run can resume.'],
    ]),

    t(`## Every loss is a distributional assumption

This is the single most useful idea about loss functions, and it comes straight from
[the probability lesson](#/l/math-probability): **minimising a loss is maximum likelihood under some noise
model.** Choosing a loss and choosing what you believe about the noise are the same act.

The practical consequence is in the third column below — what the trained model actually outputs:

| Loss | Noise model | Optimal prediction |
|---|---|---|
| Squared error | Gaussian | conditional **mean** |
| Absolute error | Laplace | conditional **median** |
| Cross-entropy | Categorical | conditional **probabilities** |
| Poisson NLL | Poisson | conditional **rate** |
| Quantile / pinball | asymmetric Laplace | conditional **quantile** |`),

    warn(`Take that third column seriously. Suppose you are predicting customer spend, where most people spend
nothing and a few spend thousands. Train with MSE and the model outputs the conditional **mean** — perhaps £40,
a figure essentially no individual customer's spend resembles. Train with MAE and it outputs the **median** —
perhaps £0, which describes a typical customer accurately and is useless for revenue forecasting.

Neither model is broken. They are answering different questions, and the loss function is where you chose which
question. On symmetric data the distinction is invisible; on skewed data — which is most real business data — it
determines whether your predictions mean anything.

If you want the whole distribution rather than one summary, use **quantile loss** at several quantiles and get a
prediction interval instead of a point.`),

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

Notice that all three have the same shape: **cross-entropy applied to a difference of similarity scores.** That
is logistic regression, again, with "which of these pairs go together" as the classification problem.

Once you see that pattern you can read most representation-learning papers at a glance. The interesting content
is never the loss — it is what counts as a positive pair, and where the negatives come from.`),

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

    recap(`- Explain "every loss is a distributional assumption", and predict what a model trained with MSE versus
  MAE will output on skewed data.
- Choose a loss from a description of the target's distribution and what the prediction will be used for.
- Say what Huber loss buys over both L1 and L2.
- Recognise contrastive losses as logistic regression on score differences, and identify what actually varies
  between papers.
- Write the training loop in the right order, and say what each safeguard (clipping, warmup, scheduling) is
  protecting against.
- Compute the expected initial loss for a $K$-class classifier and use it as a setup check.`),
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
    tldr(`A dense layer on an image is both ruinously expensive and structurally stupid — it would need 150
million parameters and would still have to learn "cat in the corner" and "cat in the middle" as two unrelated
facts.

A convolution fixes both problems with one idea: **use the same small set of weights everywhere.** A 3×3 edge
detector is 9 numbers, and sliding it across the whole image means the network learns "edge" once rather than
once per location.

That is the entire concept. Everything else — pooling, stride, channels, receptive fields — is bookkeeping
around it.`),

    jargon([
      ['convolution', 'Sliding a small window of weights over the input and computing a dot product at each position.'],
      ['kernel / filter', 'The small window of weights. A 3×3 kernel is 9 numbers. Both words mean the same thing.'],
      ['weight sharing', 'Using the *same* kernel weights at every position. The source of both the parameter savings and the translation behaviour.'],
      ['feature map', 'The output of applying one kernel across the whole input — a 2-D grid showing where that feature was detected.'],
      ['channel', 'One feature map. An RGB image has 3 input channels; a conv layer might output 64.'],
      ['stride', 'How far the window jumps between positions. Stride 2 halves the output size.'],
      ['padding', 'Adding a border of zeros so the output stays the same size as the input. "same" keeps the size, "valid" shrinks it.'],
      ['equivariance', 'Shift the input, and the output shifts the same way. What convolution gives you.'],
      ['invariance', 'Shift the input, and the output does not change at all. What pooling adds a little of. Not the same as equivariance.'],
      ['pooling', 'Shrinking a feature map by summarising each window — usually taking the maximum.'],
      ['receptive field', 'How much of the original input a single deep neuron can "see". Grows with depth.'],
      ['dilated / atrous convolution', 'A kernel with gaps in it, covering more ground for the same number of weights.'],
      ['residual / skip connection', 'Adding a block\'s input to its output, so gradients have a direct path backwards.'],
    ]),

    t(`## The problem with dense layers on images

Start by counting. A $224\\times224\\times3$ image flattened into a vector is 150,528 numbers. A single dense
layer mapping that to 1000 units needs $150{,}528 \\times 1000 \\approx$ **150 million parameters** — for one
layer, of what would be a many-layer network. That alone rules it out.

But the deeper problem is not cost, it is *structure*. In a dense layer, the weight connecting pixel (5,5) to
some unit has no relationship whatsoever to the weight connecting pixel (5,6). The layer has no idea those
pixels are adjacent — you could shuffle every pixel in your dataset with a fixed permutation and it would train
just as well.

Which means it must learn "cat in the top-left" and "cat in the centre" as two entirely unrelated facts,
needing separate examples of each. That is an absurd way to spend data.

Two structural facts about images fix both problems:

1. **Locality** — a pixel's meaning depends mostly on its immediate neighbours, not on a pixel 200 across.
2. **Translation equivariance** — an edge is an edge wherever it appears. The detector should not care about
   position.`),

    viz('convolution'),

    t(`## The convolution layer

Slide a small kernel over the input, computing a dot product at each position:

$$(I * K)_{ij} = \\sum_{m}\\sum_{n} I_{i+m,\\,j+n}\\,K_{m,n}$$

The same kernel weights are used at **every** position. That is **weight sharing**, and both benefits fall out
of it at once:

- **Parameters.** A $3\\times3$ kernel is 9 numbers whether the image is 32×32 or 4000×3000. Compare with 150
  million.
- **Equivariance.** Because the same detector runs everywhere, shifting the input by 5 pixels shifts the output
  by exactly 5 pixels. The network learns "edge" once and gets it at every location for free.

That second point is the one worth dwelling on. Weight sharing is not a compression hack that happens to work —
it is an encoding of a true fact about images, and it is the reason CNNs need so much less data than a dense
network would.

Key hyperparameters:

- **Kernel size** — 3×3 is the near-universal choice; two stacked 3×3 see 5×5 with fewer parameters and an extra
  nonlinearity.
- **Stride** — step size. Stride 2 halves the spatial dimensions.
- **Padding** — "same" preserves size, "valid" shrinks.
- **Channels** — a layer with $C_{\\text{in}}$ input and $C_{\\text{out}}$ output channels has
  $C_{\\text{in}} \\times C_{\\text{out}} \\times k^2$ weights.`),

    viz('pooling-receptive-field'),

    t(`## Pooling and the receptive field

**Max pooling** takes the maximum over a small window — typically 2×2, halving both spatial dimensions. It does
two things: shrinks the data (cheaper downstream), and adds a little translation **invariance**.

That word is worth separating from the one before it. Convolution is *equivariant*: shift the input, the output
shifts too. Pooling is (partially) *invariant*: shift the input by one pixel, and the maximum over a 2×2 window
often does not change at all. Equivariance preserves where things are; invariance forgets. A classifier wants
invariance at the end ("is there a cat?") and equivariance in the middle ("where are the edges?"), which is
roughly how the two are distributed through a CNN.

Modern architectures frequently drop pooling in favour of **strided convolutions**, on the reasoning that a
learned downsampling beats a hardcoded one.

The **receptive field** is how much of the original input a single deep neuron can see. It starts tiny and grows
with depth: stack $L$ layers of $k\\times k$ convolutions and you get $1 + L(k-1)$.

Plug in numbers and the practical consequence appears. With $3\\times3$ kernels, each layer adds 2 — so seeing a
224-pixel-wide object requires about 110 layers of plain convolution. **This is why vision networks are deep**:
not for abstraction alone, but because the final layer physically cannot see the whole object otherwise. Stride
and pooling are how real architectures cheat, growing the receptive field multiplicatively instead of
additively.

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

    recap(`- Give both reasons a dense layer is wrong for images — the parameter count, and the missing structure.
- Explain weight sharing and say which two benefits come out of it simultaneously.
- Distinguish equivariance from invariance, and say which part of a CNN provides each.
- Compute a receptive field from depth and kernel size, and explain why vision networks must be deep.
- Justify stacking 3×3 kernels over one large kernel, on both parameters and nonlinearities.
- Say what residual connections solved, and why they made 100+ layer networks possible.`),
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
    tldr(`Before transformers, sequences were handled by walking through them one element at a time, carrying a
running summary — the **hidden state** — from step to step. It is the obvious design, and it works for short
sequences.

It fails for long ones, and the reason is the same multiplicative decay from
[backprop](#/l/nn-backprop): getting a gradient from step 500 back to step 1 means multiplying by 500
Jacobians, and that product either vanishes or explodes.

The **LSTM** fixed this with an idea worth stealing: give information an *additive* path through time instead of
a multiplicative one. That is the same insight as a residual connection, invented eighteen years earlier for a
completely different reason.`),

    jargon([
      ['recurrent network (RNN)', 'A network that processes a sequence step by step, feeding its own output from the previous step back in as input.'],
      ['hidden state $\\mathbf{h}_t$', 'The running summary carried from one step to the next. Everything the model remembers about what it has seen so far, compressed into one vector.'],
      ['timestep', 'One position in the sequence — one word, one audio frame, one day of data.'],
      ['unrolling', 'Writing out the recurrence as a deep feedforward network, one layer per timestep, so ordinary backprop applies.'],
      ['BPTT', 'Backpropagation Through Time — backprop applied to the unrolled network.'],
      ['LSTM', 'Long Short-Term Memory. An RNN with a separate memory cell updated by addition and controlled by learned gates.'],
      ['cell state $\\mathbf{c}_t$', 'The LSTM\'s memory highway — updated by adding, not by matrix multiplication. This is the whole trick.'],
      ['gate', 'A vector of values between 0 and 1, multiplied elementwise into a signal to control how much passes. Learned, and input-dependent.'],
      ['GRU', 'Gated Recurrent Unit. A simpler LSTM with two gates instead of three and no separate cell state.'],
      ['seq2seq', 'Encoder–decoder: read the whole input sequence into a representation, then generate the output sequence from it. Machine translation\'s original architecture.'],
      ['teacher forcing', 'During training, feeding the *correct* previous token rather than the model\'s own prediction. Fast to train, and creates a train/inference mismatch.'],
    ]),

    t(`## Recurrence

Process the sequence one element at a time, carrying a hidden state forward:

$$\\mathbf{h}_t = \\phi(W_{hh}\\mathbf{h}_{t-1} + W_{xh}\\mathbf{x}_t + \\mathbf{b})$$

Read it as: *the new state is a function of the old state and the new input.* $W_{hh}$ decides what to carry
forward from memory, $W_{xh}$ decides how to incorporate what just arrived.

The important structural point is that **the same weights are used at every step**. This is weight sharing again,
now across time rather than across space — and it buys the same two things it bought a CNN. The model handles
sequences of any length (nothing in the parameters depends on $T$), and a pattern learned at position 3 applies
at position 300.

Training works by **unrolling**: write out the recurrence as a deep feedforward network, one layer per timestep,
then run ordinary backprop. The name for this is backpropagation through time, and notice what it implies — a
50-token sequence is a 50-layer-deep network, which is where all the trouble comes from.`),

    viz('rnn-unroll'),

    t(`## Why vanilla RNNs fail

The gradient from step $T$ back to step $t$ carries a product of $T-t$ Jacobians, each roughly $W_{hh}^{\\mathsf T}$
times an activation derivative. So it scales like $\\lambda^{T-t}$ where $\\lambda$ is the largest eigenvalue.

- $\\lambda < 1$: gradients **vanish**. Information from early tokens cannot influence the loss, so the model
  cannot learn long-range dependencies at all.
- $\\lambda > 1$: gradients **explode**. \\\`NaN\\\`.

There is no good value, and this is worse than it is in a feedforward network. There, each layer has its own
weight matrix, so the factors are at least independent and can partially cancel. Here it is the *same* matrix
$W_{hh}$ raised to a power, so whatever it does, it does relentlessly in the same direction.

The two failures are also not symmetric in difficulty. Explosion is loud and easily fixed — clip the gradient
norm and move on. Vanishing is **silent**: nothing crashes, the loss goes down, and you simply never learn that
the pronoun in sentence 12 refers to the noun in sentence 1. It is why plain RNNs cannot reliably learn
dependencies beyond about 10 steps, and why every practical sequence model since 1997 has been an attempt to
route around it.`),

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

    key(`Six equations, and only one of them matters:

$$\\mathbf{c}_t = \\mathbf{f}_t\\odot\\mathbf{c}_{t-1} + \\mathbf{i}_t\\odot\\tilde{\\mathbf{c}}_t$$

The cell state is updated by **addition**, gated by a value the network controls. Compare with the vanilla RNN,
where the state is *replaced* by a matrix multiply every step.

Work the gradient through it. If the forget gate $\\mathbf{f}_t \\approx 1$, then $\\partial\\mathbf{c}_t /
\\partial\\mathbf{c}_{t-1} \\approx 1$ — the gradient path is multiplication by one, repeatedly, which does
nothing. Information and gradient both survive hundreds of steps intact.

And crucially the gate is **learned and input-dependent**: the network decides, per timestep and per dimension,
what to preserve and what to overwrite. Memory becomes something it controls rather than something that decays
at a rate fixed by $W_{hh}$'s eigenvalues.

This is the identical insight to a residual connection — replace a product with a sum so gradients have a clean
path — arrived at eighteen years earlier for a completely different reason. Set the figure's forget gate to 1.0
and compare against the vanilla RNN to watch it happen.`),

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

    recap(`- Write the RNN recurrence and say what weight sharing across time buys you.
- Explain why a $T$-step sequence is effectively a $T$-layer network, and why that is the source of the problem.
- Say why vanishing gradients are more dangerous than exploding ones, in terms of how each announces itself.
- Point at the one LSTM equation that does the work, and explain why addition beats multiplication here.
- Connect the LSTM cell state to residual connections in one sentence.
- Describe the seq2seq bottleneck, and say what attention replaced it with.`),
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
    tldr(`Neural networks operate on vectors, but words, user IDs, and product categories are discrete symbols.
An **embedding** is the bridge: a learned lookup table mapping each symbol to a vector.

The reason this is more than a technicality is what happens to the geometry. Once every word is a point in
space, *distance means something* — similar words end up near each other, because they were used in similar
contexts and the training signal pushed them together. Discrete symbols with no structure become a continuous
space where you can measure similarity, interpolate, and search.

That single move underpins semantic search, recommendation, RAG, and the input layer of every language model.`),

    jargon([
      ['one-hot', 'Representing symbol $k$ out of $V$ as a vector of $V$ zeros with a single 1 in slot $k$. Simple, and geometrically useless.'],
      ['embedding', 'A learned dense vector representing a discrete item. Also the *table* of all of them.'],
      ['embedding dimension $d$', 'How long each vector is. 100 for small models, 4096+ for large ones.'],
      ['lookup table', 'How embeddings are implemented: a matrix of shape (vocabulary, $d$), indexed by token id.'],
      ['distributional hypothesis', 'The claim that words appearing in similar contexts have similar meanings. The assumption everything here rests on.'],
      ['word2vec / GloVe', 'The 2013–14 methods that made word embeddings mainstream. Now mostly of historical interest, but the ideas persist.'],
      ['skip-gram / CBOW', 'Predict the context from a word / predict the word from its context. Two ways to generate a training signal from raw text.'],
      ['negative sampling', 'Replacing an expensive softmax over 50,000 words with a cheap binary classification against a few random ones.'],
      ['static vs contextual', 'One fixed vector per word, versus a vector that depends on the surrounding sentence. BERT and later models are contextual.'],
      ['tied weights', 'Sharing the input embedding matrix with the output projection layer. Saves hundreds of millions of parameters.'],
      ['cold start', 'Having no embedding for a new item because it has no history yet. The standard failure of learned embeddings in recommenders.'],
    ]),

    t(`## The problem with one-hot

The obvious way to feed a word into a network is a **one-hot** vector: 50,000 slots, all zero except a single 1
marking which word it is.

Now look at what that representation says geometrically. Every pair of one-hot vectors is orthogonal, and every
pair is exactly $\\sqrt{2}$ apart. So "cat" is precisely as similar to "dog" as it is to "bureaucracy". All
semantic structure has been destroyed *before learning even starts*, and the network must rediscover from scratch
that some of these 50,000 arbitrary symbols relate to each other.

An **embedding** replaces this with a learned dense vector $\\mathbf{e}_w \\in \\mathbb{R}^d$, where $d$ is
typically 100 to 4096 — vastly smaller than 50,000, and with every dimension carrying information rather than
50,000 dimensions carrying one bit between them.

The implementation is deliberately boring: a matrix of shape (vocabulary, $d$), and you index into it by token
id. That is genuinely all \\\`nn.Embedding\\\` does.

Mathematically, though, indexing a row *is* multiplying the matrix by a one-hot vector — which is worth knowing
because it explains the backward pass. A one-hot input means every gradient lands on exactly one row, so only the
embeddings of tokens that actually appeared in the batch get updated. The other 49,999 sit untouched.`),

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

Here is the limitation that ended the word2vec era. It gives **one vector per word type** — so "bank" gets a
single vector that has to average together river banks and financial institutions, plus "banking on it" and
"bank shot". The resulting point is a blurry compromise sitting between meanings, resembling none of them.

Any word with multiple senses has this problem, and in English that is most common words.

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

    recap(`- Say what is geometrically wrong with one-hot vectors, in terms of distances and angles.
- Describe what \\\`nn.Embedding\\\` actually is, and explain why only a few rows receive gradient per batch.
- State the distributional hypothesis and say how word2vec turns it into a training signal.
- Explain why negative sampling was necessary to make word2vec practical.
- Give the difference between static and contextual embeddings, with a word whose senses require the
  distinction.
- Be appropriately sceptical about "king − man + woman = queen", and say what the evaluation quietly does.`),
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

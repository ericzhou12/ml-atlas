/* ============================================================
   Track 1 — Mathematical foundations
   ============================================================ */

import { t, key, intuition, warn, hist, mathnote, viz, deriv, code, quiz, paper, book, course, blog, video, demo, codeRef,
         tldr, recap, jargon, steps, diagram } from './_helpers.js';

export default [

/* ---------------------------------------------------------- */
{
  id: 'math-vectors',
  title: 'Vectors, Dot Products, and Geometry',
  sub: 'Everything in ML is a list of numbers. Every question about similarity is a dot product.',
  mins: 20, level: 'foundations',
  tags: ['linear algebra', 'geometry'],
  sections: [
    tldr(`A **vector** is a list of numbers. That is the whole definition. \`[2, 1, -0.5]\` is a vector.

The useful move is to treat that list as a *position* in space: the list \`[2, 1]\` becomes the point you reach
by going 2 right and 1 up. Once data is a position, you can ask geometric questions about it — how far apart
are these two things, are they pointing the same way — even when the data is words or photos and has nothing
to do with geometry.

There is one operation you need: the **dot product**. Multiply two lists entry by entry, add up the results,
get a single number. That number tells you how much the two vectors agree in direction. Search engines,
recommendations, and every layer of every neural network run on it.`),

    jargon([
      ['scalar', 'A single number, like `3.7`. The word only exists so we can say "one number" as opposed to "a list of numbers".'],
      ['vector', 'An ordered list of numbers. In code, a 1-D array. Written in **bold**: $\\mathbf{x}$.'],
      ['component', 'One entry of a vector. The vector $\\mathbf{x} = (2, 1)$ has components $x_1 = 2$ and $x_2 = 1$.'],
      ['dimension $d$', 'How many numbers are in the list. A list of 768 numbers is a "768-dimensional vector". It means nothing deeper than `len(x)`.'],
      ['$\\mathbf{x} \\in \\mathbb{R}^d$', 'Read aloud: "x is a list of $d$ real numbers." $\\mathbb{R}$ is the set of all real numbers, and the superscript says how many of them. It is a type signature, like `x: float[d]`. You will see this line in the first paragraph of almost every paper.'],
      ['embedding', 'A vector that a model outputs to stand in for something that is not a number — a word, an image, a user. The whole point is that similar things get similar vectors.'],
      ['norm', 'The length of a vector, written $\\|\\mathbf{x}\\|$.'],
      ['orthogonal', 'A synonym for perpendicular, i.e. at 90°. It gets its own word because in ML it means something specific: "these two carry unrelated information."'],
      ['$\\sum$', 'Sigma, the sum symbol. $\\sum_{i=1}^{d} a_i$ means "add up $a_1$ through $a_d$." It is a for-loop with an accumulator.'],
    ]),

    t(`## Why everything becomes a list of numbers

A computer cannot multiply a photograph. It can multiply numbers. So the first thing anyone does with data is
turn it into a list of numbers, and after that every algorithm in this atlas works on lists of numbers only.

| The thing | The list |
|---|---|
| A 28×28 greyscale image | 784 numbers, one brightness per pixel |
| A word, after running it through an embedding model | 768 numbers, none of which means anything on its own |
| A user of a streaming site | 64 numbers summarising what they watch |
| One layer's weights inside a network | a few million numbers |

The payoff is uniformity. A method that works on lists of 784 numbers also works on lists of 768 numbers,
so a technique invented for images transfers to text without being reinvented. Everything in this track is
about what you can do with a list of numbers once you have one.`),

    t(`## A vector is a point, and also an arrow

Take the list $(3, 2)$. There are two pictures of it, and you need both.

The **point** picture: go 3 right and 2 up, and put a dot there. The vector *is* that location. This is the
right picture for data — one photo is one point, and photos of similar things sit near each other.

The **arrow** picture: draw an arrow from the origin to that same spot. Now the vector describes a *movement* —
a direction to go and a distance to go in it. This is the right picture for changes: "adjust these weights by
this much in this direction."

Same three numbers either way. Which picture is in play changes what the numbers are for.`),

    diagram('The same two numbers, read two ways',
`<svg viewBox="0 0 640 220" role="img" aria-label="A vector drawn as a point and as an arrow">
  <g style="stroke: var(--border); stroke-width: 1">
    <line x1="40" y1="180" x2="270" y2="180"/><line x1="40" y1="180" x2="40" y2="30"/>
    <line x1="370" y1="180" x2="600" y2="180"/><line x1="370" y1="180" x2="370" y2="30"/>
  </g>
  <circle cx="190" cy="80" r="6" style="fill: var(--s1)"/>
  <line x1="190" y1="80" x2="190" y2="180" style="stroke: var(--text-faint); stroke-width: 1; stroke-dasharray: 3 3"/>
  <line x1="40" y1="80" x2="190" y2="80" style="stroke: var(--text-faint); stroke-width: 1; stroke-dasharray: 3 3"/>
  <text class="dmono" x="202" y="76" style="fill: var(--s1)">(3, 2)</text>
  <text class="dtitle" x="40" y="205">as a POINT — "the thing is here"</text>
  <text class="dlabel" x="40" y="20">use for: data, samples, embeddings</text>

  <defs><marker id="ar1" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
    <path d="M0,0 L9,4.5 L0,9 z" style="fill: var(--s2)"/></marker></defs>
  <line x1="370" y1="180" x2="514" y2="82" style="stroke: var(--s2); stroke-width: 2.5" marker-end="url(#ar1)"/>
  <text class="dmono" x="524" y="78" style="fill: var(--s2)">(3, 2)</text>
  <text class="dtitle" x="370" y="205">as an ARROW — "move this way, this far"</text>
  <text class="dlabel" x="370" y="20">use for: gradients, weight updates, directions</text>
</svg>`,
      `When you read "the embedding of *cat*", picture the dot. When you read "the update step", picture the
arrow. Writers switch between the two without announcing it, so it is worth asking yourself which one a
sentence means.`),

    t(`## The two things you can do to vectors

**Add them.** Line the lists up and add entry by entry:

$$\\mathbf{a} + \\mathbf{b} = (a_1+b_1,\\ \\ldots,\\ a_d+b_d)$$

In the arrow picture: walk along the first arrow, then, from wherever you stopped, walk along the second.

**Scale them.** Multiply every entry by one number $c$:

$$c\\,\\mathbf{a} = (c\\,a_1, \\ldots, c\\,a_d)$$

The arrow keeps its direction and changes its length: $c > 1$ stretches, $0 < c < 1$ shrinks, and a negative
$c$ flips it to point the opposite way.

That is the complete list. Adding and scaling, plus the obvious rules they obey (order does not matter when
adding; scaling distributes over addition), are the *definition* of a vector space. So anything you can add
and scale gets to reuse this machinery — including functions and images, which is why you will see the word
"vector" applied to things that are not obviously lists.`),

    t(`## The dot product

Multiply two vectors entry by entry, then add the products up:

$$\\mathbf{a}\\cdot\\mathbf{b} = a_1b_1 + a_2b_2 + \\cdots + a_db_d = \\sum_{i=1}^{d} a_i b_i$$

Both vectors must have the same length, or there is nothing to pair up. The result is a single number, not a
vector. In NumPy it is \`a @ b\`.`),

    steps('One dot product, by hand', [
      { h: 'Line the lists up', md: `$\\mathbf{a} = (2,\\ 1,\\ -3)$ and $\\mathbf{b} = (4,\\ 0,\\ 1)$. Both have three entries, so this is allowed.` },
      { h: 'Multiply matching positions', md: `$2\\times4 = 8$, then $1\\times0 = 0$, then $-3\\times1 = -3$.` },
      { h: 'Add the three products', md: `$8 + 0 - 3 = 5$. So $\\mathbf{a}\\cdot\\mathbf{b} = 5$.` },
      { h: 'Notice it is one number', md: `Three-entry inputs, one-number output. The dot product always collapses two lists into a single score.` },
    ]),

    t(`## Why that number measures direction

Nothing above mentions angles, so it is not obvious that this has anything to do with geometry. Here is why
it does.

Start with the simplest possible second vector: $\\mathbf{b} = (1, 0)$, an arrow of length 1 pointing straight
along the horizontal axis. Then

$$\\mathbf{a}\\cdot\\mathbf{b} = a_1 \\cdot 1 + a_2 \\cdot 0 = a_1,$$

which is just the horizontal coordinate of $\\mathbf{a}$. Now draw the right triangle under the arrow
$\\mathbf{a}$, with $\\theta$ the angle between $\\mathbf{a}$ and the horizontal axis. The hypotenuse is
$\\|\\mathbf{a}\\|$ and the horizontal side is $a_1$, so by the definition of cosine,

$$a_1 = \\|\\mathbf{a}\\| \\cos\\theta.$$

Putting those together: when $\\mathbf{b}$ has length 1, the dot product equals $\\|\\mathbf{a}\\|\\cos\\theta$.
And if you then stretch $\\mathbf{b}$ to length $\\|\\mathbf{b}\\|$, every product $a_ib_i$ in the sum scales by
the same factor, so the dot product scales by it too:

$$\\mathbf{a}\\cdot\\mathbf{b} = \\|\\mathbf{a}\\|\\,\\|\\mathbf{b}\\|\\cos\\theta$$

Nothing about the argument used the fact that we were in two dimensions except the picture; the identity holds
in any number of dimensions, and the derivation below proves it without pictures.

These two formulas being the same number is the most useful fact in this track. The first one is easy to
compute. The second one tells you what the answer *means*.`),

    deriv('The same identity without pictures', `The picture argument above assumed we could see the triangle. Here is the algebra, which works in any dimension.

Take the triangle whose sides are the arrows $\\mathbf{a}$, $\\mathbf{b}$, and the arrow from the tip of $\\mathbf{b}$ to the tip of $\\mathbf{a}$, which is $\\mathbf{a}-\\mathbf{b}$. The law of cosines from trigonometry says

$$\\|\\mathbf{a}-\\mathbf{b}\\|^2 = \\|\\mathbf{a}\\|^2 + \\|\\mathbf{b}\\|^2 - 2\\|\\mathbf{a}\\|\\|\\mathbf{b}\\|\\cos\\theta$$

Separately, expand the left-hand side using only the definition of the dot product (and the fact, from the next section, that $\\|\\mathbf{v}\\|^2 = \\mathbf{v}\\cdot\\mathbf{v}$):

$$\\|\\mathbf{a}-\\mathbf{b}\\|^2 = (\\mathbf{a}-\\mathbf{b})\\cdot(\\mathbf{a}-\\mathbf{b}) = \\mathbf{a}\\cdot\\mathbf{a} - 2\\,\\mathbf{a}\\cdot\\mathbf{b} + \\mathbf{b}\\cdot\\mathbf{b} = \\|\\mathbf{a}\\|^2 - 2\\,\\mathbf{a}\\cdot\\mathbf{b} + \\|\\mathbf{b}\\|^2$$

Set the two right-hand sides equal. The $\\|\\mathbf{a}\\|^2$ and $\\|\\mathbf{b}\\|^2$ terms cancel from both sides, leaving

$$-2\\,\\mathbf{a}\\cdot\\mathbf{b} = -2\\|\\mathbf{a}\\|\\|\\mathbf{b}\\|\\cos\\theta$$

Divide by $-2$. ∎`),

    t(`## Reading the sign

Lengths are never negative, so in $\\|\\mathbf{a}\\|\\|\\mathbf{b}\\|\\cos\\theta$ the only part that can be
negative is $\\cos\\theta$. And $\\cos\\theta$ is positive for angles under 90°, exactly zero at 90°, and
negative past 90°. So the sign of a dot product — one cheap number — tells you which side of perpendicular the
two vectors are on.`),

    diagram('What the sign of a dot product tells you',
`<svg viewBox="0 0 660 180" role="img" aria-label="Three cases: positive, zero and negative dot product">
  <defs>
    <marker id="ap" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" style="fill: var(--s1)"/></marker>
    <marker id="aq" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" style="fill: var(--s2)"/></marker>
  </defs>
  <g>
    <line x1="60" y1="120" x2="170" y2="120" style="stroke: var(--s1); stroke-width: 2.5" marker-end="url(#ap)"/>
    <line x1="60" y1="120" x2="150" y2="55"  style="stroke: var(--s2); stroke-width: 2.5" marker-end="url(#aq)"/>
    <path d="M100,120 A40,40 0 0,0 88,97" style="fill:none; stroke: var(--text-faint); stroke-width: 1"/>
    <text class="dtitle" x="60" y="155" style="fill: var(--s3)">a · b &gt; 0</text>
    <text class="dlabel" x="60" y="172">angle under 90° — they agree</text>
  </g>
  <g transform="translate(215,0)">
    <line x1="60" y1="120" x2="170" y2="120" style="stroke: var(--s1); stroke-width: 2.5" marker-end="url(#ap)"/>
    <line x1="60" y1="120" x2="60"  y2="40"  style="stroke: var(--s2); stroke-width: 2.5" marker-end="url(#aq)"/>
    <path d="M60,100 L80,100 L80,120" style="fill:none; stroke: var(--text-faint); stroke-width: 1"/>
    <text class="dtitle" x="60" y="155" style="fill: var(--s5)">a · b = 0</text>
    <text class="dlabel" x="60" y="172">exactly 90° — orthogonal</text>
  </g>
  <g transform="translate(430,0)">
    <line x1="60" y1="120" x2="170" y2="120" style="stroke: var(--s1); stroke-width: 2.5" marker-end="url(#ap)"/>
    <line x1="60" y1="120" x2="0"   y2="62"  style="stroke: var(--s2); stroke-width: 2.5" marker-end="url(#aq)"/>
    <text class="dtitle" x="30" y="155" style="fill: var(--s6)">a · b &lt; 0</text>
    <text class="dlabel" x="30" y="172">over 90° — they oppose</text>
  </g>
</svg>`,
      `The middle case is the one to remember. When two feature vectors are orthogonal, knowing one tells you
nothing about the other — they describe genuinely separate things. That is why "orthogonal" is used in ML as a
plain synonym for "unrelated".`),

    viz('vector-playground'),

    t(`## Length, and comparing direction only

The dot product of a vector with **itself** is $\\mathbf{a}\\cdot\\mathbf{a} = a_1^2 + \\cdots + a_d^2$. In two
dimensions that is $a_1^2 + a_2^2$, which the Pythagorean theorem says is the squared length of the arrow. So
length comes free from the dot product:

$$\\|\\mathbf{a}\\| = \\sqrt{\\mathbf{a}\\cdot\\mathbf{a}} = \\sqrt{a_1^2 + \\cdots + a_d^2}$$

and the same formula is taken as the definition of length in higher dimensions, where you cannot draw the
triangle.

Now suppose you only care about *direction* and not size. Rearranging
$\\mathbf{a}\\cdot\\mathbf{b} = \\|\\mathbf{a}\\|\\|\\mathbf{b}\\|\\cos\\theta$ to isolate the angle gives
**cosine similarity**:

$$\\cos\\theta = \\frac{\\mathbf{a}\\cdot\\mathbf{b}}{\\|\\mathbf{a}\\|\\,\\|\\mathbf{b}\\|}$$

Dividing by both lengths cancels them out, so what is left depends only on the angle. It runs from $1$ (same
direction) through $0$ (orthogonal) to $-1$ (opposite), and it never leaves that range.`),

    key(`Three numbers, all built from the dot product:

- $\\mathbf{a}\\cdot\\mathbf{b}$ — raw agreement. Grows if either vector gets longer.
- $\\|\\mathbf{a}\\| = \\sqrt{\\mathbf{a}\\cdot\\mathbf{a}}$ — length.
- $\\cos\\theta = \\dfrac{\\mathbf{a}\\cdot\\mathbf{b}}{\\|\\mathbf{a}\\|\\|\\mathbf{b}\\|} \\in [-1, 1]$ — agreement in direction only, with length divided out.`),

    intuition(`Why retrieval systems compare with cosine rather than the raw dot product: a 50-page report and a
one-line note about the same topic should count as similar, but embedding models give longer inputs longer
vectors. Raw dot products would then rank the report above the note for essentially every query, because it is
*bigger*, not because it is a better match. Dividing the lengths out removes that. Nearly every vector database
defaults to cosine for this reason.`),

    t(`## Projection: how much of one vector lies along another

Here is a question that keeps coming back: *how much of $\\mathbf{a}$ points in the direction of
$\\mathbf{b}$?*

Shine a light straight down onto the line through $\\mathbf{b}$ and see where $\\mathbf{a}$'s shadow lands.
That shadow is the **projection** of $\\mathbf{a}$ onto $\\mathbf{b}$:

$$\\text{proj}_{\\mathbf{b}}(\\mathbf{a}) = \\underbrace{\\frac{\\mathbf{a}\\cdot\\mathbf{b}}{\\mathbf{b}\\cdot\\mathbf{b}}}_{\\text{one number}}\\ \\mathbf{b}$$

Read it right to left: the answer is a scaled copy of $\\mathbf{b}$, so it is guaranteed to lie on
$\\mathbf{b}$'s line. The scale factor is "how much they agree" divided by "$\\mathbf{b}$'s length squared" —
the second division is there so that making $\\mathbf{b}$ twice as long does not change where the shadow falls.

Whatever is left over,

$$\\mathbf{r} = \\mathbf{a} - \\text{proj}_{\\mathbf{b}}(\\mathbf{a}),$$

is called the **residual**, and it is always orthogonal to $\\mathbf{b}$. You can check that in one line:
$\\mathbf{r}\\cdot\\mathbf{b} = \\mathbf{a}\\cdot\\mathbf{b} - \\frac{\\mathbf{a}\\cdot\\mathbf{b}}{\\mathbf{b}\\cdot\\mathbf{b}}(\\mathbf{b}\\cdot\\mathbf{b}) = 0$.
Turn on "residual" in the figure above and drag: the angle stays 90° no matter what.`),

    key(`Every vector splits into two pieces — the part $\\mathbf{b}$ explains, and the part it cannot:

$$\\mathbf{a} = \\underbrace{\\text{proj}_{\\mathbf{b}}(\\mathbf{a})}_{\\text{explained by } \\mathbf{b}} + \\underbrace{\\mathbf{r}}_{\\text{left over, orthogonal to } \\mathbf{b}}$$

Fitting a line to data is this split (prediction = the explained part, error = the residual). So is PCA (keep
the directions that explain the most, drop the rest). When those show up later they are this picture with more
indices.`),

    t(`## Other ways to measure length

$\\|\\mathbf{x}\\| = \\sqrt{\\sum_i x_i^2}$ is straight-line distance, and it is the default. But it is not the
only sensible way to turn a list of numbers into one number called "size", and the choice changes what a model
does. The alternatives are written $\\|\\mathbf{x}\\|_p$, with the subscript naming which one.

| Name | Formula | What "distance" means | Where you meet it |
|---|---|---|---|
| $L_1$ | $\\sum_i |x_i|$ | blocks walked on a city grid | Lasso, sparsity |
| $L_2$ | $\\sqrt{\\sum_i x_i^2}$ | straight line, as the crow flies | ridge, weight decay, distances |
| $L_\\infty$ | $\\max_i |x_i|$ | the single largest entry | adversarial robustness budgets |
| $L_0$ | how many entries are nonzero | how many things are switched on | measuring sparsity (not really a length) |

To see how they differ, draw the set of all vectors whose size is exactly 1 — the **unit ball** for that
measure. Under $L_2$ it is the circle you would expect. Under $L_1$ it is a diamond, because to have
$|x_1| + |x_2| = 1$ you trade one coordinate off against the other in a straight line.`),

    diagram('"Length exactly 1" under three different measures',
`<svg viewBox="0 0 660 200" role="img" aria-label="Unit balls of the L1, L2 and Linfinity norms">
  <g transform="translate(20,10)">
    <line x1="90" y1="20" x2="90" y2="160" style="stroke: var(--border)"/><line x1="20" y1="90" x2="160" y2="90" style="stroke: var(--border)"/>
    <polygon points="90,30 150,90 90,150 30,90" style="fill: color-mix(in srgb, var(--s2) 16%, transparent); stroke: var(--s2); stroke-width: 2"/>
    <circle cx="90" cy="30" r="4" style="fill: var(--s2)"/><circle cx="150" cy="90" r="4" style="fill: var(--s2)"/>
    <text class="dtitle" x="55" y="185" style="fill: var(--s2)">L1 — a diamond</text>
  </g>
  <g transform="translate(240,10)">
    <line x1="90" y1="20" x2="90" y2="160" style="stroke: var(--border)"/><line x1="20" y1="90" x2="160" y2="90" style="stroke: var(--border)"/>
    <circle cx="90" cy="90" r="60" style="fill: color-mix(in srgb, var(--s1) 16%, transparent); stroke: var(--s1); stroke-width: 2"/>
    <text class="dtitle" x="60" y="185" style="fill: var(--s1)">L2 — a circle</text>
  </g>
  <g transform="translate(460,10)">
    <line x1="90" y1="20" x2="90" y2="160" style="stroke: var(--border)"/><line x1="20" y1="90" x2="160" y2="90" style="stroke: var(--border)"/>
    <rect x="30" y="30" width="120" height="120" style="fill: color-mix(in srgb, var(--s4) 16%, transparent); stroke: var(--s4); stroke-width: 2"/>
    <text class="dtitle" x="50" y="185" style="fill: var(--s4)">L-infinity — a square</text>
  </g>
</svg>`,
      `Look at the dots where the diamond meets the axes. Those are sharp **corners**, and a corner sits at a
place where one coordinate is exactly zero. The circle has no corners — it slides past the axes smoothly. That
single difference in shape is the whole reason $L_1$ penalties set coefficients to exactly 0 while $L_2$
penalties only shrink them toward 0. The picture comes back, with loss contours drawn on top, in
[regularization](#/l/ml-regularization).`),

    t(`## High dimensions do not behave like the pictures

Every drawing so far has been in two dimensions. Real vectors have 768 or 4096 entries, and three things that
are false in 2-D become true out there. Each one has a short reason, and the reasons all use the same tool:
when you add up many independent random numbers, the sum grows faster than its own wobble.`),

    mathnote(`The tool, stated once. Suppose $z_1, \\ldots, z_d$ are independent random numbers, each with mean
0 and variance 1. Variances of independent things add, so $\\sum_i z_i$ has variance $d$ and therefore standard
deviation $\\sqrt{d}$ — not $d$. This is the same fact behind the $\\sigma/\\sqrt{n}$ standard error you met in
statistics. Everything below is that one line applied three times.`),

    t(`**1. Two random directions are almost always perpendicular.**

Take $\\mathbf{x}$ and $\\mathbf{y}$ with independent random entries, each of mean 0 and variance 1. Their dot
product $\\sum_i x_i y_i$ is a sum of $d$ independent terms with mean 0, so it has mean 0 and typical size
$\\sqrt{d}$. Meanwhile $\\|\\mathbf{x}\\|^2 = \\sum_i x_i^2$ is a sum of $d$ terms each averaging 1, so
$\\|\\mathbf{x}\\| \\approx \\sqrt{d}$, and the same for $\\mathbf{y}$. Divide:

$$\\cos\\theta = \\frac{\\mathbf{x}\\cdot\\mathbf{y}}{\\|\\mathbf{x}\\|\\|\\mathbf{y}\\|} \\approx \\frac{\\sqrt{d}}{\\sqrt{d}\\cdot\\sqrt{d}} = \\frac{1}{\\sqrt{d}}$$

In 2-D that is 0.71, so random vectors are often quite aligned. In 1000-D it is 0.03. Random directions in high
dimensions are, in practice, always perpendicular. You will measure this yourself in the challenge.

**2. Almost all the volume is near the surface.**

Scale a $d$-dimensional ball down by 10% and its volume drops by a factor of $0.9^d$. At $d = 100$ that is
$0.9^{100} \\approx 0.00003$. So the inner 90% of the radius holds essentially none of the volume — sample
points uniformly from a high-dimensional ball and none of them land near the middle. The "typical" point is not
near the average point.

**3. Everything is about equally far away.**

The squared distance between two random points is a sum of $d$ independent per-coordinate contributions. By the
same argument as fact 1, that sum is around $d$ with a wobble of around $\\sqrt{d}$, so the *relative* spread in
distances shrinks like $\\sqrt{d}/d = 1/\\sqrt{d}$. As $d$ grows, your nearest neighbour and your farthest
neighbour end up nearly the same distance away. This is what quietly breaks k-nearest-neighbours and anything
else that relies on "close" being meaningfully different from "far".`),

    intuition(`Fact 1 sounds like bad news and is secretly the best news in the subject. If almost every pair of
random directions is perpendicular, then a space with only $d$ dimensions can hold *far more than $d$*
directions that are nearly perpendicular to each other — vastly more. So a layer with 4096 neurons is not
limited to storing 4096 separate features. It can pack in many more, as long as it tolerates a little
interference between them.

That packing has a name, **superposition**, and it is a large part of why interpretability is hard: one neuron
does not correspond to one concept. There is a [whole lesson on it](#/l/fr-interpretability) later.`),

    code('Vectors in NumPy', `import numpy as np

a = np.array([2.0, 1.0, -0.5])
b = np.array([1.0, 3.0,  2.0])

print("a + b        =", a + b)
print("3a           =", 3 * a)
print("a . b        =", a @ b)              # @ does the multiply-and-sum
print("|a|          =", np.sqrt(a @ a))     # length, straight from the dot product
print("cos(a, b)    =", (a @ b) / np.sqrt((a @ a) * (b @ b)))

# projection of a onto b, and the leftover
proj  = (a @ b) / (b @ b) * b
resid = a - proj
print("proj_b(a)    =", proj.round(4))
print("residual . b =", (resid @ b).round(12), " <- zero, as promised")`,
      'Two things to notice: the length needs no special function (it is just the dot product with itself, square-rooted), and the residual really is orthogonal to b — the last line prints 0 up to floating-point dust.'),

    quiz('Two embedding vectors have cosine similarity 0.95 but very different lengths. What does that tell you?',
      ['They point in nearly the same direction, but one vector is longer — often because that input was longer or more common',
       'They are nearly identical vectors',
       'They are almost orthogonal',
       'Nothing — cosine similarity does not apply when the lengths differ'],
      0,
      'Cosine has both lengths divided out, so it reports direction and nothing else. High cosine with very different lengths is normal for embeddings: input length and word frequency inflate the norm without changing the meaning. That is the whole reason retrieval systems compare cosine instead of raw dot products.'),

    recap(`- Read $\\mathbf{x} \\in \\mathbb{R}^d$ out loud as "a list of $d$ numbers", and say whether a given
  sentence wants the point picture or the arrow picture.
- Compute a dot product by hand, and read its sign: positive means under 90°, zero means orthogonal, negative
  means over 90°.
- Explain why $\\mathbf{a}\\cdot\\mathbf{b} = \\|\\mathbf{a}\\|\\|\\mathbf{b}\\|\\cos\\theta$, starting from the
  case where $\\mathbf{b}$ has length 1.
- Get length and cosine similarity out of the dot product, and say why retrieval uses cosine.
- Split a vector into a projection plus an orthogonal residual, and verify the residual is orthogonal.
- Say why $L_1$ produces exact zeros and $L_2$ does not, pointing at the corners of the diamond.
- Explain, using "variances add", why two random directions in 1000-D are almost certainly perpendicular.`),
  ],
  refs: [
    book('Mathematics for Machine Learning', 'Deisenroth, Faisal & Ong', 2020, 'https://mml-book.github.io/', 'Free PDF. Chapters 2–3 cover this material properly, with ML motivation throughout. The best single reference for this track.'),
    video('Essence of Linear Algebra', '3Blue1Brown', 2016, 'https://www.3blue1brown.com/topics/linear-algebra', 'If the geometric picture has not clicked, watch this before reading anything else. Fifteen short episodes.'),
    book('Introduction to Linear Algebra', 'Gilbert Strang', 2016, 'https://math.mit.edu/~gs/linearalgebra/', 'The classic. Pair it with his MIT 18.06 lectures, which are free.'),
    course('MIT 18.06 Linear Algebra', 'Gilbert Strang', 2011, 'https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/', 'Full video lectures, problem sets, exams.'),
    blog('A Few Useful Things to Know About Machine Learning', 'Pedro Domingos', 2012, 'https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf', 'The section on the curse of dimensionality is the clearest short treatment anywhere.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'math-matrices',
  title: 'Matrices as Transformations',
  sub: 'A matrix is not a grid of numbers. It is a function that moves space.',
  mins: 25, level: 'foundations',
  prereq: ['math-vectors'],
  tags: ['linear algebra', 'matrices'],
  sections: [
    tldr(`A matrix is a grid of numbers, but that is the least useful way to think about it. A matrix is a
**function that takes a vector in and gives a vector back**, and it is restricted so that straight lines stay
straight and the origin never moves.

Two numbers describe what a given matrix does to space. Its **determinant** says how much it scales area. Its
**rank** says how many independent directions come out the other side. When the rank is low, the matrix carries
much less information than its size suggests — which means you can store it with far fewer numbers.

That last sentence has a big payoff: it is the entire reason you can fine-tune a 16-million-number weight
matrix by training only 65 thousand numbers. This lesson builds to that.`),

    jargon([
      ['matrix', 'A rectangular grid of numbers. $A \\in \\mathbb{R}^{m\\times n}$ means $m$ rows and $n$ columns. In code, a 2-D array.'],
      ['$A\\mathbf{x}$', 'The matrix $A$ applied to the vector $\\mathbf{x}$. Read it as function application, like `f(x)`.'],
      ['linear function', 'A function that survives addition and scaling unchanged — defined precisely below. Matrices are exactly the functions that do.'],
      ['basis vector $\\mathbf{e}_i$', 'The vector with a 1 in position $i$ and 0 everywhere else. $\\mathbf{e}_1 = (1,0)$ and $\\mathbf{e}_2 = (0,1)$ are the one-unit steps along each axis.'],
      ['linearly independent', 'A set of vectors where none is a combination of the others. Two vectors are independent exactly when they do not lie on the same line.'],
      ['span', 'Everything you can reach by adding scaled copies of some vectors. The span of one vector is a line through the origin; of two independent vectors, a plane.'],
      ['rank', 'How many independent directions come out of a matrix. If every output of a 2×2 matrix lands on one line, its rank is 1.'],
      ['determinant $\\det A$', 'The factor by which the matrix scales area (in 2-D) or volume (in higher dimensions).'],
      ['invertible', 'There is a second matrix that undoes this one. $A^{-1}A\\mathbf{x} = \\mathbf{x}$ for every $\\mathbf{x}$.'],
      ['transpose $A^{\\mathsf T}$', 'The grid flipped across its diagonal, so rows become columns. `A.T` in NumPy.'],
      ['outer product $\\mathbf{u}\\mathbf{v}^{\\mathsf T}$', 'A column vector times a row vector. Unlike the dot product, this produces a whole *matrix*. It is always rank 1, and it is the atom that low-rank structure is built from.'],
      ['weight matrix', 'Inside a neural network, the numbers one layer multiplies its input by. Training means adjusting them; a large language model is mostly a pile of these.'],
      ['fine-tuning', 'Taking an already-trained model and nudging its weights to specialise it for a new task, instead of training from scratch.'],
    ]),

    t(`## The shift in perspective

You were probably taught matrix multiplication as a procedure: take a row, take a column, multiply and add,
repeat. That tells you *how to compute* the answer and nothing about *what the answer is*.

Here is what it is. A matrix $A \\in \\mathbb{R}^{m\\times n}$ is a **function**. Hand it a list of $n$ numbers,
it hands back a list of $m$ numbers. That is the same job as any \`f(x)\` in code, but restricted to a small,
well-behaved family of functions.

The restriction is called **linearity**, and it is these two rules:

$$A(\\mathbf{x}+\\mathbf{y}) = A\\mathbf{x} + A\\mathbf{y}, \\qquad A(c\\,\\mathbf{x}) = c\\,A\\mathbf{x}$$

In words: processing a sum gives the same answer as summing the processed parts, and doubling the input doubles
the output. Notice what those rules forbid — no squaring an input, no thresholds, no if-statements. Linear
functions are deliberately boring, and the reward for that is that a finite grid of numbers can describe one
completely.

Geometrically, linearity means gridlines stay straight, stay parallel, and stay evenly spaced, and the origin
stays where it is. A matrix can rotate, stretch, shear, reflect, and flatten space. It cannot bend it.`),

    key(`**The columns of $A$ are where the basis vectors land.**

Check it on a 2×2. Applying $A$ to $\\mathbf{e}_1 = (1,0)$ picks out the first column and multiplies the second
by zero, so $A\\mathbf{e}_1$ *is* column 1. Likewise $A\\mathbf{e}_2$ is column 2.

Once you know where $\\mathbf{e}_1$ and $\\mathbf{e}_2$ go, linearity fixes everything else, because every vector
is built from them: $\\mathbf{x} = x_1\\mathbf{e}_1 + x_2\\mathbf{e}_2$. Apply $A$ and use both linearity rules:

$$A\\mathbf{x} = A(x_1\\mathbf{e}_1 + x_2\\mathbf{e}_2) = x_1 A\\mathbf{e}_1 + x_2 A\\mathbf{e}_2 = x_1(\\text{col}_1) + x_2(\\text{col}_2)$$

**A matrix-vector product is a weighted sum of the matrix's columns**, with the input's entries as the weights.
This is the most useful way to read a matrix, and the row-times-column recipe hides it completely.`),

    steps('The same product, computed both ways', [
      { h: 'Set it up', md: `$A = \\begin{pmatrix} 2 & 1 \\\\ 0 & 3\\end{pmatrix}$ and $\\mathbf{x} = (4, 5)$.` },
      { h: 'The way you were taught: rows against the input', md: `Row 1 is $(2, 1)$, so the first output entry is $2(4) + 1(5) = 13$. Row 2 is $(0, 3)$, so the second is $0(4) + 3(5) = 15$. Answer: $(13, 15)$.` },
      { h: 'The way to think about it: weighted columns', md: `Column 1 is $(2,0)$ and column 2 is $(1,3)$. Take 4 copies of the first plus 5 copies of the second: $4(2,0) + 5(1,3) = (8,0) + (5,15) = (13,15)$.` },
      { h: 'Same answer, different story', md: `The second version says the output is *built out of the columns*. That immediately tells you something the first version does not: whatever comes out of $A$ must lie in the span of $A$'s columns. Nothing outside that span is reachable, no matter what you feed in.` },
    ]),

    viz('matrix-transform'),

    t(`## Multiplying matrices is chaining functions

If a matrix is a function, then multiplying two of them should mean running one after the other — and it does.
$AB$ means "apply $B$ first, then apply $A$", right to left, the same order as $f(g(x))$.

Once you read it that way, two rules that otherwise look arbitrary stop being arbitrary:

- **$AB$ is usually not $BA$.** Rotating a book 90° and then flipping it does not land in the same place as
  flipping it and then rotating. Doing things in a different order gives a different result — that was already
  true of functions, and matrices inherit it.
- **The shapes have to line up.** $B$'s output is fed straight into $A$, so the number of entries $B$ produces
  must equal the number $A$ expects.`),

    diagram('The shape rule, and why it is the only one you need to remember',
`<svg viewBox="0 0 620 210" role="img" aria-label="Shape matching in a matrix product">
  <rect x="40" y="50" width="120" height="90" rx="3" style="fill: color-mix(in srgb, var(--s1) 12%, transparent); stroke: var(--s1); stroke-width: 1.6"/>
  <text class="dmono" x="100" y="100" text-anchor="middle" style="fill: var(--s1)">A</text>
  <text class="dlabel" x="100" y="164" text-anchor="middle">m x k</text>
  <text x="185" y="100" text-anchor="middle" style="font-size:19px; fill: var(--text-dim)">·</text>
  <rect x="210" y="50" width="150" height="90" rx="3" style="fill: color-mix(in srgb, var(--s2) 12%, transparent); stroke: var(--s2); stroke-width: 1.6"/>
  <text class="dmono" x="285" y="100" text-anchor="middle" style="fill: var(--s2)">B</text>
  <text class="dlabel" x="285" y="164" text-anchor="middle">k x n</text>
  <text x="388" y="100" text-anchor="middle" style="font-size:19px; fill: var(--text-dim)">=</text>
  <rect x="415" y="50" width="150" height="90" rx="3" style="fill: color-mix(in srgb, var(--s3) 12%, transparent); stroke: var(--s3); stroke-width: 1.6"/>
  <text class="dmono" x="490" y="100" text-anchor="middle" style="fill: var(--s3)">AB</text>
  <text class="dlabel" x="490" y="164" text-anchor="middle">m x n</text>
  <path d="M148,32 C165,14 250,14 268,32" style="fill:none; stroke: var(--s5); stroke-width: 1.6; stroke-dasharray: 4 3"/>
  <text class="dtitle" x="208" y="16" text-anchor="middle" style="fill: var(--s5)">these must match — they cancel</text>
  <text class="dlabel" x="310" y="196" text-anchor="middle">the two k's vanish; the outer dimensions m and n survive</text>
</svg>`,
      `Almost every shape error you will ever hit in PyTorch is this diagram being violated. Write the two shapes
down, check the inner pair matches, and read the answer's shape off the outer pair.`),

    t(`Only now is the arithmetic formula worth writing down:

$$(AB)_{ij} = \\sum_k A_{ik}B_{kj}$$

which says: to get the entry in row $i$, column $j$ of the answer, walk along row $i$ of $A$ and column $j$ of
$B$ together, multiplying pairs and adding. That is a dot product of a row with a column — the operation from
the previous lesson, done $mn$ times. It is a *consequence* of chaining two linear functions, not a definition
handed down from nowhere.`),

    viz('matmul-walkthrough'),

    t(`## Determinant: how much does area change?

Take the unit square — the square with corners at the origin, $\\mathbf{e}_1$, $\\mathbf{e}_2$, and
$\\mathbf{e}_1 + \\mathbf{e}_2$. Its area is 1. Push it through $A$ and, because straight lines stay straight,
it becomes a parallelogram. **The determinant is defined as that parallelogram's area.**

- $\\det A = 2$: every region in the plane comes out twice as big.
- $\\det A = 0.5$: everything shrinks by half.
- $\\det A < 0$: space got flipped over, like turning a page. The size still gives the area factor; the minus
  sign records the flip.
- $\\det A = 0$: the square was squashed flat onto a line segment, which has no area.

The last case is the one that matters. If the square is squashed onto a line, then many different input points
land on the same output point, and there is no way to look at an output and tell which input produced it. So
there can be no undo function:

$$\\det A = 0 \\iff A \\text{ has no inverse} \\iff A \\text{ flattened at least one direction}$$

($\\iff$ means "these say the same thing".)`),

    t(`## Rank: how many directions survive?

The determinant answers a yes/no question: did anything get flattened? **Rank** answers the detailed version:
*how many independent directions are left?*

The definition: the rank of $A$ is the number of linearly independent columns it has, which is the same as the
number of independent directions its outputs can span. For a 2×2 matrix there are three cases:

- **rank 2** — the two columns point in different directions, so weighted sums of them reach the whole plane.
  Nothing was lost.
- **rank 1** — the two columns lie on the same line, so every weighted sum of them lies on that line too. The
  entire plane gets mapped onto a single line.
- **rank 0** — both columns are zero, so everything maps to the origin. Only the zero matrix does this.

Drag $\\sigma_2$ to zero in the figure below and watch the rank-1 case happen. That collapse *is* what "rank
deficient" means; there is nothing more to it.`),

    viz('rank-collapse'),

    warn(`Rank counts *directions*, not size. A matrix full of enormous numbers can have rank 1, and a matrix of
tiny numbers can have full rank. The strict definition is also brittle: change one entry by $10^{-9}$ and a
rank-1 matrix is technically rank 2, even though it behaves identically.

That is why you will constantly read "approximately low rank" or "effective rank" instead of plain "rank". What
people mean is that after the first few directions, the rest carry so little that they might as well be zero.
The [SVD lesson](#/l/math-eigen-svd) makes "carry so little" into an exact measurement.`),

    t(`## Why low rank is a resource, not a defect

The first time you meet rank deficiency it looks like damage — a matrix that lost information. Turn it around.
If a matrix's outputs only span $r$ directions, then it is doing a much simpler job than its $m \\times n$
numbers suggest, and it should be describable with fewer of them. **Low rank means compressible.**

To see how, we need one new operation. The **outer product** of a column vector $\\mathbf{u}$ (length $m$) and a
row vector $\\mathbf{v}^{\\mathsf T}$ (length $n$) is the $m \\times n$ matrix whose entry in row $i$, column $j$
is $u_i v_j$:

$$\\mathbf{u}\\mathbf{v}^{\\mathsf T} = \\begin{pmatrix} u_1 \\\\ u_2 \\end{pmatrix}\\begin{pmatrix} v_1 & v_2 & v_3\\end{pmatrix} = \\begin{pmatrix} u_1v_1 & u_1v_2 & u_1v_3 \\\\ u_2v_1 & u_2v_2 & u_2v_3 \\end{pmatrix}$$

Every column of that matrix is a copy of $\\mathbf{u}$, scaled by one entry of $\\mathbf{v}$. All the columns lie
on the same line, so **an outer product always has rank 1** — and it took only $m + n$ numbers to write down a
grid of $mn$ entries.

The general fact, which the [SVD lesson](#/l/math-eigen-svd) proves, is that this goes the other way too: any
rank-$r$ matrix can be written as a sum of exactly $r$ outer products.

$$A = \\mathbf{u}_1\\mathbf{v}_1^{\\mathsf T} + \\mathbf{u}_2\\mathbf{v}_2^{\\mathsf T} + \\cdots + \\mathbf{u}_r\\mathbf{v}_r^{\\mathsf T}$$

Count the storage: $r(m+n)$ numbers instead of $mn$. When $r$ is small and $m, n$ are large, that gap is
enormous. A $4096\\times4096$ matrix holds 16.8 million numbers; at rank 8 it needs $8(4096+4096) = 65{,}536$.`),

    viz('low-rank-approx'),

    t(`### Three places this pays off

**PCA.** Data that arrives as 1000 measurements per sample often really varies along only about 10 underlying
factors. The data matrix is then approximately rank 10, and PCA is the procedure that finds those 10 directions.
[Covered in the next lesson](#/l/math-eigen-svd).

**Recommender systems.** A user-by-film rating matrix might be a million users tall and a hundred thousand films
wide, yet be well approximated at rank 50 — there really are only about 50 independent "tastes" driving
everyone's ratings. Netflix's recommender was essentially that observation, cashed in.

**LoRA.** Fine-tuning a giant model cheaply. This one is worth walking through slowly, because it is the most
widely used practical consequence of everything above.`),

    steps('LoRA, derived from what you now know', [
      { h: 'The problem', md: `A pretrained language model contains weight matrices $W$ of size around $4096 \\times 4096$ — $16{,}777{,}216$ numbers in **one** matrix, and a model has hundreds of them. Fine-tuning normally updates every one of those numbers, which means the GPU has to hold the weights, a gradient for each weight, and the optimizer's bookkeeping for each weight. That is the expense.` },
      { h: 'You only need the change, not the result', md: `Fine-tuning does not replace $W$, it nudges it. Write the nudge on its own: $W_{\\text{new}} = W_{\\text{pretrained}} + \\Delta W$. Freeze the pretrained part and never touch it. The only thing left to learn is $\\Delta W$.` },
      { h: 'The bet', md: `Here is the hypothesis, and it is an empirical one, not a theorem: **$\\Delta W$ is approximately low rank.** Teaching a general model to handle legal documents, or to answer in a particular style, is a focused change — it moves the weights along a handful of directions rather than all 4096. Note carefully what is *not* being claimed: the pretrained weights themselves are full rank. Only the update is assumed to be simple.` },
      { h: 'Store the change as two skinny matrices', md: `If $\\Delta W$ has rank $r$, then by the sum-of-outer-products fact it factors as $\\Delta W = BA$ with $B \\in \\mathbb{R}^{4096 \\times r}$ (tall and thin) and $A \\in \\mathbb{R}^{r \\times 4096}$ (short and wide). Train $A$ and $B$; leave $W$ frozen.` },
      { h: 'Count the parameters', md: `Take $r = 8$. Then $B$ holds $4096 \\times 8 = 32{,}768$ numbers, $A$ holds another $32{,}768$, and the total trained is **65,536** against 16.8 million. That is 0.4%, a 256× reduction, and it came entirely from "a rank-8 matrix does not need $4096^2$ numbers to write down".` },
      { h: 'Two things you get for free', md: `Because $W$ is untouched, one copy of the base model can serve many tasks: keep a 65k-number $(A, B)$ pair per task and swap them at inference time. And since $\\Delta W = BA$ is just a matrix, you can add it into $W$ once you are finished, so the deployed model runs at exactly its original speed.` },
    ]),

    diagram('Full fine-tuning vs. LoRA, drawn to scale',
`<svg viewBox="0 0 620 250" role="img" aria-label="A full weight matrix compared with two skinny LoRA factors">
  <rect x="35" y="45" width="150" height="150" rx="3" style="fill: color-mix(in srgb, var(--s6) 16%, transparent); stroke: var(--s6); stroke-width: 1.8"/>
  <text class="dmono" x="110" y="115" text-anchor="middle" style="fill: var(--s6)">delta W</text>
  <text class="dlabel" x="110" y="136" text-anchor="middle">4096 x 4096</text>
  <text class="dtitle" x="110" y="30" text-anchor="middle">full fine-tuning</text>
  <text class="dlabel" x="110" y="220" text-anchor="middle" style="fill: var(--s6)">16,777,216 trained</text>

  <text x="235" y="120" text-anchor="middle" style="font-size:17px; fill: var(--text-dim)">≈</text>

  <rect x="285" y="45" width="22" height="150" rx="2" style="fill: color-mix(in srgb, var(--s1) 22%, transparent); stroke: var(--s1); stroke-width: 1.8"/>
  <text class="dmono" x="296" y="120" text-anchor="middle" style="fill: var(--s1)">B</text>
  <text class="dlabel" x="296" y="212" text-anchor="middle">4096 x 8</text>
  <text x="325" y="120" text-anchor="middle" style="font-size:17px; fill: var(--text-dim)">·</text>
  <rect x="345" y="109" width="150" height="22" rx="2" style="fill: color-mix(in srgb, var(--s3) 22%, transparent); stroke: var(--s3); stroke-width: 1.8"/>
  <text class="dmono" x="420" y="121" text-anchor="middle" style="fill: var(--s3)">A</text>
  <text class="dlabel" x="420" y="150" text-anchor="middle">8 x 4096</text>
  <text class="dtitle" x="390" y="30" text-anchor="middle">LoRA, rank 8</text>
  <text class="dlabel" x="390" y="220" text-anchor="middle" style="fill: var(--s3)">65,536 trained — 0.4%</text>
</svg>`,
      `The picture is the argument. The thin blue strip times the thin green strip produces something the same
*shape* as the big square, but described by 256× fewer numbers. Whether that is *enough* description for your
task is an empirical question, and for fine-tuning the answer has turned out to be usually yes.`),

    t(`## The transpose, and why backprop is full of them

The transpose $A^{\\mathsf T}$ flips the grid across its diagonal, so a $3 \\times 5$ matrix becomes $5 \\times 3$.
On its own that is bookkeeping. The reason it appears constantly is one identity:

$$\\mathbf{x} \\cdot (A\\mathbf{y}) = (A^{\\mathsf T}\\mathbf{x}) \\cdot \\mathbf{y}$$

You can move a matrix from one side of a dot product to the other, as long as you transpose it. Here is the
one-line proof, using only the definition of the dot product and of matrix-vector multiplication:

$$\\mathbf{x}\\cdot(A\\mathbf{y}) = \\sum_i x_i \\sum_j A_{ij} y_j = \\sum_j \\Big(\\sum_i A_{ij} x_i\\Big) y_j = (A^{\\mathsf T}\\mathbf{x})\\cdot\\mathbf{y}$$

All that happened was swapping the order of the two sums, and noticing that $\\sum_i A_{ij}x_i$ is row $j$ of
$A^{\\mathsf T}$ dotted with $\\mathbf{x}$.`),

    intuition(`When you meet $W^{\\mathsf T}\\delta$ in a backpropagation derivation, do not read it as "transpose,
for algebraic reasons". Read it as routing. The forward pass sent information from one layer to the next through
$W$. The backward pass has to send blame back along exactly those same wires, and $W^{\\mathsf T}$ is the matrix
that runs the wiring in reverse. You will derive this properly in [backpropagation](#/l/nn-backprop).`),

    t(`## Special matrices worth recognizing on sight

| Type | Condition | Why it matters |
|---|---|---|
| Diagonal | zeros everywhere off the diagonal | Scales each axis independently. Trivial to invert: flip each diagonal entry. |
| Identity $I$ | ones on the diagonal, zeros elsewhere | The do-nothing function, $I\\mathbf{x} = \\mathbf{x}$. |
| Symmetric | $A = A^{\\mathsf T}$ | Covariance matrices and second-derivative matrices are always symmetric, and symmetry buys them very clean structure (next lesson). |
| Orthogonal | $A^{\\mathsf T}A = I$ | A rotation or reflection. Preserves every length and angle, and its inverse is free: $A^{-1} = A^{\\mathsf T}$. |
| Low rank | rank much smaller than $\\min(m,n)$ | Compressible, as above. |

One more that is worth naming now because optimization leans on it. A symmetric matrix is **positive definite**
when the number $\\mathbf{x}\\cdot(A\\mathbf{x})$ is strictly positive for every nonzero $\\mathbf{x}$. That
expression is a quadratic in the entries of $\\mathbf{x}$, and "always positive" makes its graph a bowl with a
single lowest point. That is exactly the shape you want a loss surface to have, which is why the condition keeps
reappearing in [optimization](#/l/math-optimization).`),

    code('Matrices in NumPy', `import numpy as np

A = np.array([[2.0, 1.0],
              [0.0, 3.0]])
x = np.array([4.0, 5.0])

print("A @ x                 =", A @ x)
print("as a sum of columns   =", x[0]*A[:,0] + x[1]*A[:,1], " <- identical")
print("det A                 =", np.linalg.det(A))
print("rank A                =", np.linalg.matrix_rank(A))

B = np.array([[0.0, -1.0], [1.0, 0.0]])       # a 90 degree rotation
print("AB is not BA          :", not np.allclose(A@B, B@A))

# an outer product: one column times one row
u = np.array([1.0, 2.0])
v = np.array([3.0, 1.0])
R = u[:, None] * v[None, :]                   # the 2x2 outer product u v^T
print("\\nu v^T =\\n", R)
print("rank                  =", np.linalg.matrix_rank(R), "   det =", round(np.linalg.det(R), 12))

# every output of R lands on the line through u
pts  = np.random.default_rng(0).normal(size=(4, 2))
outs = pts @ R.T
for o in outs:
    print(f"  output {np.round(o,3)}  ->  ratio to u: {np.round(o/u, 6)}")`,
      'The last loop is the point. Each output divided by u gives the same number in both slots, meaning every output is a scalar multiple of u — the whole plane has been mapped onto one line. That is what rank 1 and a zero determinant mean concretely.'),

    quiz('You fine-tune a 4096×4096 weight matrix with LoRA at rank 8. How many numbers do you train, versus full fine-tuning?',
      ['65,536 vs 16,777,216 — about 0.4%',
       '8 vs 16,777,216',
       '4096 vs 16,777,216',
       'The same number; LoRA only changes the optimizer'],
      0,
      'LoRA writes the update as $\\Delta W = BA$ with $B$ of shape $4096\\times 8$ and $A$ of shape $8 \\times 4096$, so $2 \\times 4096 \\times 8 = 65{,}536$ trainable numbers against $4096^2 = 16.8$ million. The bet — empirically a good one — is that the *update* the task needs is low rank, even though the pretrained weights are not.'),

    quiz('A 5×5 matrix has rank 2. Which of these is guaranteed?',
      ['It can be written as a sum of exactly 2 outer products, and its determinant is 0',
       'All of its entries are small',
       'It is invertible, but the inverse is hard to compute numerically',
       'It has exactly 2 nonzero entries'],
      0,
      'Rank counts surviving directions, not magnitudes and not nonzero entries. Rank 2 inside a 5-dimensional space means three directions were flattened, so volume is destroyed, $\\det = 0$, and no inverse exists. And "rank $r$" is exactly the statement "writable as $r$ outer products", which is what makes it a compression claim: $2(5+5) = 20$ numbers instead of 25. Scale that same gap up to 4096×4096 and you have LoRA.'),

    recap(`- Describe a matrix as a function that moves space, and say what linearity forbids: bending, thresholds,
  and moving the origin.
- Read a matrix's columns as "where the basis vectors land", and compute $A\\mathbf{x}$ as a weighted sum of
  columns rather than a row-by-row recipe.
- Explain both $AB \\neq BA$ and the inner-dimension shape rule as facts about doing one function after another.
- Say what $\\det A = 0$ means physically: space was flattened, inputs collided, no inverse exists.
- Define rank as the number of surviving directions, and explain why an outer product always has rank 1.
- Derive LoRA's 65,536 from scratch, and state the empirical bet it rests on.
- Prove $\\mathbf{x}\\cdot(A\\mathbf{y}) = (A^{\\mathsf T}\\mathbf{x})\\cdot\\mathbf{y}$ by swapping two sums.`),
  ],
  refs: [
    video('Linear transformations and matrices', '3Blue1Brown', 2016, 'https://www.youtube.com/watch?v=kYB8IZa5AuE', 'The single best five minutes on why the columns are the images of the basis vectors.'),
    book('Mathematics for Machine Learning, Ch. 2', 'Deisenroth, Faisal & Ong', 2020, 'https://mml-book.github.io/', 'Rigorous treatment of vector spaces, rank, and linear maps.'),
    paper('LoRA: Low-Rank Adaptation of Large Language Models', 'Hu et al.', 2021, 'https://arxiv.org/abs/2106.09685', 'The low-rank-update idea, and the empirical evidence that fine-tuning updates really do have low intrinsic rank.'),
    book('Matrix Computations', 'Golub & Van Loan', 2013, 'https://epubs.siam.org/doi/book/10.1137/1.9781421407944', 'The reference for when you need numerical detail rather than intuition.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'math-eigen-svd',
  title: 'Eigenvectors, SVD, and Low-Rank Structure',
  sub: 'The directions a matrix does not rotate, and the decomposition that works on every matrix.',
  mins: 30, level: 'foundations',
  prereq: ['math-matrices'],
  tags: ['linear algebra', 'SVD', 'PCA'],
  sections: [
    tldr(`Most vectors come out of a matrix pointing somewhere new. A few special directions come out pointing
the same way, only longer or shorter. Those are the **eigenvectors**, and the amount of stretch is the
**eigenvalue**. Finding them turns one complicated matrix into a list of independent one-number problems.

Eigenvectors are great when they exist, and they often don't — a non-square matrix has none at all. The
**SVD** is the repair. It says that *every* matrix, with no exceptions, does exactly three things in order:
rotate, stretch along axes, rotate again.

Because the stretches come out sorted from largest to smallest, you can drop the small ones and keep a
simplified version of the matrix. That single move — keep the big stretches, discard the rest — is what PCA,
image compression, recommender systems, and LoRA all are underneath.`),

    jargon([
      ['eigenvector', 'A direction a matrix does not turn; it only lengthens or shortens it. "Eigen" is German for "own" — these are the matrix\'s own natural axes.'],
      ['eigenvalue $\\lambda$', 'The stretch factor along an eigenvector. $\\lambda = 2$ doubles the vector, $\\lambda = -1$ flips it, and $|\\lambda| < 1$ shrinks it.'],
      ['symmetric matrix', '$A = A^{\\mathsf T}$: the grid is its own mirror image across the diagonal. Covariance matrices and second-derivative matrices are always symmetric, which is lucky, because symmetric matrices behave much better than general ones.'],
      ['singular value $\\sigma_i$', 'Like an eigenvalue, but always real and never negative, and defined for *any* matrix including non-square ones. It measures how much the matrix stretches along its $i$-th special direction.'],
      ['SVD', 'Singular Value Decomposition — splitting any matrix $A$ into $U\\Sigma V^{\\mathsf T}$, which reads "rotate, stretch, rotate". In code it is one call, `np.linalg.svd`.'],
      ['orthogonal matrix', 'A matrix that only rotates or reflects and never stretches, so it preserves every length and angle. From the last lesson: $A^{\\mathsf T}A = I$, so its inverse is free.'],
      ['spectrum', 'The list of a matrix\'s eigenvalues or singular values. Whenever you see the adjective "spectral", it means "about that list".'],
      ['covariance matrix', 'For a dataset, the grid whose $(i,j)$ entry says how much features $i$ and $j$ vary together. Its eigenvectors are the directions the data spreads along.'],
      ['Hessian', 'The matrix of second derivatives of a loss function. It describes curvature — how bowl-shaped the loss is, and in which directions. Built properly in [the Jacobian lesson](#/l/math-jacobian).'],
      ['Frobenius norm', 'The size of a whole matrix: flatten it into one long list and take the ordinary length. $\\|A\\|_F = \\sqrt{\\sum_{ij}A_{ij}^2}$.'],
      ['condition number $\\kappa$', 'The biggest stretch divided by the smallest, $\\sigma_1/\\sigma_r$. A large $\\kappa$ means the problem is numerically delicate and gradient descent will crawl.'],
    ]),

    t(`## Eigenvectors: the directions a matrix leaves alone

Push a random vector through a matrix and two things usually happen at once: it gets rotated to point somewhere
new, and it gets stretched. A few special vectors only get the stretch — they come out lying on the same line
they went in on:

$$A\\mathbf{v} = \\lambda \\mathbf{v}$$

Read both sides. The left side applies a whole matrix, doing $n^2$ multiplications. The right side multiplies by
one number. For these particular directions, the entire matrix collapses to a single scalar.

$\\mathbf{v}$ is an **eigenvector** and $\\lambda$ is its **eigenvalue**. If $\\mathbf{v}$ satisfies the equation
then so does $3\\mathbf{v}$ (scale both sides), so what an eigenvector really names is a *direction*; by
convention we pick the one of length 1.`),

    steps('Finding the eigenvectors of a small matrix, once', [
      { h: 'Rewrite the equation so one side is zero', md: `$A\\mathbf{v} = \\lambda\\mathbf{v}$ becomes $A\\mathbf{v} - \\lambda\\mathbf{v} = \\mathbf{0}$, and since $\\lambda\\mathbf{v} = \\lambda I \\mathbf{v}$ we can factor: $(A - \\lambda I)\\mathbf{v} = \\mathbf{0}$.` },
      { h: 'Ask what that requires', md: `We want a **nonzero** $\\mathbf{v}$ that the matrix $A - \\lambda I$ sends to the origin. A matrix that squashes a nonzero vector to zero has flattened a direction — so by the last lesson, $\\det(A - \\lambda I) = 0$.` },
      { h: 'Solve that for $\\lambda$', md: `Take $A = \\begin{pmatrix} 3 & 1 \\\\ 0 & 2\\end{pmatrix}$. Then $A - \\lambda I = \\begin{pmatrix} 3-\\lambda & 1 \\\\ 0 & 2-\\lambda\\end{pmatrix}$, whose determinant is $(3-\\lambda)(2-\\lambda)$. Setting that to zero gives $\\lambda = 3$ and $\\lambda = 2$.` },
      { h: 'Get each eigenvector back', md: `For $\\lambda = 3$, solve $(A - 3I)\\mathbf{v} = 0$, i.e. $\\begin{pmatrix} 0 & 1 \\\\ 0 & -1\\end{pmatrix}\\mathbf{v} = 0$, which forces $v_2 = 0$ and leaves $v_1$ free: $\\mathbf{v} = (1, 0)$. For $\\lambda = 2$ the same steps give $\\mathbf{v} = (1, -1)$ up to scaling.` },
      { h: 'Sanity check', md: `$A(1,0) = (3, 0) = 3(1,0)$. ✓ You will never do this by hand past 2×2 — \`np.linalg.eig\` exists — but doing it once shows where the determinant condition comes from instead of leaving it as a rule.` },
    ]),

    diagram('Eigenvector vs. ordinary vector, under the same matrix',
`<svg viewBox="0 0 620 210" role="img" aria-label="An eigenvector stays on its line while an ordinary vector rotates">
  <defs>
    <marker id="ei1" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" style="fill: var(--text-faint)"/></marker>
    <marker id="ei2" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" style="fill: var(--s1)"/></marker>
    <marker id="ei3" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" style="fill: var(--s3)"/></marker>
  </defs>
  <g transform="translate(30,0)">
    <line x1="30" y1="160" x2="250" y2="160" style="stroke: var(--border)"/><line x1="30" y1="160" x2="30" y2="30" style="stroke: var(--border)"/>
    <line x1="30" y1="160" x2="120" y2="100" style="stroke: var(--text-faint); stroke-width: 2; stroke-dasharray: 4 3" marker-end="url(#ei1)"/>
    <line x1="30" y1="160" x2="200" y2="60" style="stroke: var(--s1); stroke-width: 2.6" marker-end="url(#ei2)"/>
    <text class="dlabel" x="126" y="96" style="fill: var(--text-faint)">x</text>
    <text class="dlabel" x="206" y="56" style="fill: var(--s1)">Ax</text>
    <text class="dtitle" x="30" y="190" style="fill: var(--s1)">ordinary vector: turned AND stretched</text>
  </g>
  <g transform="translate(340,0)">
    <line x1="30" y1="160" x2="250" y2="160" style="stroke: var(--border)"/><line x1="30" y1="160" x2="30" y2="30" style="stroke: var(--border)"/>
    <line x1="30" y1="160" x2="240" y2="55" style="stroke: var(--s3); stroke-width: 1; stroke-dasharray: 2 4"/>
    <line x1="30" y1="160" x2="110" y2="120" style="stroke: var(--text-faint); stroke-width: 2; stroke-dasharray: 4 3" marker-end="url(#ei1)"/>
    <line x1="30" y1="160" x2="230" y2="60" style="stroke: var(--s3); stroke-width: 2.6" marker-end="url(#ei3)"/>
    <text class="dlabel" x="114" y="116" style="fill: var(--text-faint)">v</text>
    <text class="dlabel" x="236" y="56" style="fill: var(--s3)">Av = 2.5v</text>
    <text class="dtitle" x="30" y="190" style="fill: var(--s3)">eigenvector: same line, just longer</text>
  </g>
</svg>`,
      `The dotted line on the right is the whole point: the eigenvector never leaves it. If a matrix has enough
such lines, then everything it does can be described as "stretch by $\\lambda_i$ along line $i$" — and that
"if" is exactly the gap the SVD closes.`),

    warn(`Some matrices have no eigenvectors at all in the ordinary sense. Take a 90° rotation: it moves
*every* direction off its own line, so there is nothing for the equation to describe. Solving
$\\det(A - \\lambda I) = 0$ for that matrix gives $\\lambda^2 + 1 = 0$, whose solutions are imaginary. That is
not a technicality to route around; it is the algebra correctly reporting that no real direction is preserved.`),

    t(`### Why eigenvectors matter in practice

The payoff is what happens when you apply the same matrix over and over. Applying $A$ a hundred times to a
general vector is a hundred matrix multiplications. Applying it to an eigenvector is one power:

$$A^k\\mathbf{v} = \\lambda^k\\mathbf{v}$$

Each application just multiplies by $\\lambda$ again. And since any vector can be written as a combination of
eigenvectors (when there are enough of them), every "what happens after many steps" question turns into "what
does $\\lambda^k$ do", which you can answer by looking at whether $|\\lambda|$ is above or below 1.

That single observation explains three things you will meet later:

- **Recurrent networks that blow up or forget.** An RNN's hidden state evolves roughly as $h_t \\approx W^t h_0$.
  If the largest $|\\lambda| > 1$, then $\\lambda^t$ grows without bound and the state explodes. If
  $|\\lambda| < 1$, it decays to nothing. There is no comfortable middle, which is the
  [vanishing and exploding gradient problem](#/l/nn-rnn) in one line.
- **How curved a loss surface is.** The eigenvalues of the Hessian say how sharply the loss bends along each
  direction: a big $\\lambda$ is a steep narrow valley, a small one is a flat plain.
- **Which directions data varies along.** The eigenvectors of a covariance matrix are the directions the data
  actually spreads out in, which is precisely what PCA computes.`),

    viz('quadratic-form'),

    mathnote(`**Symmetric matrices are the well-behaved case**, and most of the matrices you meet in ML —
covariance, Hessians — are symmetric. For them a theorem guarantees two things: every eigenvalue is a real
number (no imaginary surprises), and eigenvectors belonging to different eigenvalues are orthogonal to each
other. So a symmetric matrix has a full set of perpendicular natural axes, and can be written
$A = Q\\Lambda Q^{\\mathsf T}$ where $Q$ holds those axes as columns and $\\Lambda$ is diagonal with the
eigenvalues on it. Read right to left, that says: rotate into the natural axes, scale each one, rotate back.

The positive-definite condition from the last lesson has a clean reading here too: a symmetric matrix is
positive definite exactly when all of its eigenvalues are positive. Every direction curves upward, so the bowl
has a single bottom.`),

    t(`## SVD: the decomposition that always exists

Eigendecomposition has real limits. It needs a square matrix, so a 50×3 table of data is disqualified
immediately. Even for square matrices the eigenvalues can be imaginary, or there can be too few independent
eigenvectors to describe the whole space.

The **singular value decomposition** has none of those restrictions. Every matrix — square or not, full rank or
not — can be written as

$$A = U\\Sigma V^{\\mathsf T}$$

where $U$ and $V$ are orthogonal (pure rotations and reflections, no stretching) and $\\Sigma$ is diagonal with
non-negative entries $\\sigma_1 \\ge \\sigma_2 \\ge \\cdots \\ge 0$ sorted largest first.

Apply it to a vector and read right to left, since that is the order the operations happen in:
$A\\mathbf{x} = U(\\Sigma(V^{\\mathsf T}\\mathbf{x}))$. **Every linear map is a rotation, then a stretch along the
axes, then another rotation.** That is the complete list of what a matrix can do, with no exceptions — a
genuinely surprising structural fact given how varied matrices look.`),

    diagram('SVD as three moves: rotate, stretch, rotate',
`<svg viewBox="0 0 660 190" role="img" aria-label="A circle rotated, stretched into an ellipse, then rotated again">
  <g transform="translate(20,20)">
    <circle cx="60" cy="60" r="42" style="fill: color-mix(in srgb, var(--s1) 14%, transparent); stroke: var(--s1); stroke-width: 2"/>
    <line x1="60" y1="60" x2="102" y2="60" style="stroke: var(--s2); stroke-width: 2"/>
    <line x1="60" y1="60" x2="60" y2="18" style="stroke: var(--s3); stroke-width: 2"/>
    <text class="dtitle" x="60" y="140" text-anchor="middle">the input</text>
    <text class="dlabel" x="60" y="157" text-anchor="middle">a unit circle</text>
  </g>
  <text x="148" y="80" text-anchor="middle" class="dmono" style="fill: var(--text-dim)">V-transpose</text>
  <text x="148" y="98" text-anchor="middle" class="dlabel">rotate</text>
  <g transform="translate(190,20)">
    <circle cx="60" cy="60" r="42" style="fill: color-mix(in srgb, var(--s1) 14%, transparent); stroke: var(--s1); stroke-width: 2"/>
    <line x1="60" y1="60" x2="90" y2="30" style="stroke: var(--s2); stroke-width: 2"/>
    <line x1="60" y1="60" x2="30" y2="30" style="stroke: var(--s3); stroke-width: 2"/>
    <text class="dtitle" x="60" y="140" text-anchor="middle">axes aligned</text>
    <text class="dlabel" x="60" y="157" text-anchor="middle">shape unchanged</text>
  </g>
  <text x="318" y="80" text-anchor="middle" class="dmono" style="fill: var(--text-dim)">Sigma</text>
  <text x="318" y="98" text-anchor="middle" class="dlabel">stretch</text>
  <g transform="translate(355,20)">
    <ellipse cx="70" cy="60" rx="62" ry="26" style="fill: color-mix(in srgb, var(--s4) 14%, transparent); stroke: var(--s4); stroke-width: 2"/>
    <line x1="70" y1="60" x2="132" y2="60" style="stroke: var(--s2); stroke-width: 2"/>
    <line x1="70" y1="60" x2="70" y2="34" style="stroke: var(--s3); stroke-width: 2"/>
    <text class="dtitle" x="70" y="140" text-anchor="middle">stretched</text>
    <text class="dlabel" x="70" y="157" text-anchor="middle">by sigma-1 and sigma-2</text>
  </g>
  <text x="516" y="80" text-anchor="middle" class="dmono" style="fill: var(--text-dim)">U</text>
  <text x="516" y="98" text-anchor="middle" class="dlabel">rotate</text>
  <g transform="translate(548,20)">
    <ellipse cx="55" cy="60" rx="55" ry="24" transform="rotate(-28 55 60)" style="fill: color-mix(in srgb, var(--s4) 14%, transparent); stroke: var(--s4); stroke-width: 2"/>
    <text class="dtitle" x="55" y="140" text-anchor="middle">the output</text>
    <text class="dlabel" x="55" y="157" text-anchor="middle">A times the circle</text>
  </g>
</svg>`,
      `Start with a circle of all the length-1 inputs and follow it across. The two rotations change orientation
but never shape, so a circle stays a circle under them. Only the middle step changes the shape, which means
**only $\\Sigma$ can flatten a dimension**. A singular value of zero is exactly one lost dimension — which is
how this connects back to rank: the rank of a matrix is the number of nonzero singular values.`),

    viz('svd-demo'),

    t(`## Low-rank approximation, and why it is everywhere

Multiply the SVD out and it becomes a sum of the outer products from
[the last lesson](#/l/math-matrices), each with a weight attached:

$$A = \\sigma_1 \\mathbf{u}_1 \\mathbf{v}_1^{\\mathsf T} + \\sigma_2 \\mathbf{u}_2 \\mathbf{v}_2^{\\mathsf T} + \\cdots + \\sigma_r \\mathbf{u}_r \\mathbf{v}_r^{\\mathsf T}$$

This is the promise made earlier — that any rank-$r$ matrix is a sum of $r$ outer products — now delivered, with
a bonus: because the $\\sigma_i$ are sorted from big to small, the pieces come **ordered by importance**. The
first term carries the most structure; the last carries the least.

So truncate. Keep the first $k$ terms and throw the rest away, giving a rank-$k$ matrix $A_k$. The
**Eckart–Young theorem** says $A_k$ is not merely a decent approximation of $A$ — it is the *best possible one*.
No rank-$k$ matrix anywhere is closer to $A$, measured in Frobenius norm. That guarantee is worth a lot in
practice: it means you can stop looking for a cleverer factorization, because there isn't one.

Every one of these is the same move — look at the singular values, notice they fall off quickly, keep the first
few:

| Application | What gets truncated |
|---|---|
| PCA | The covariance spectrum — keep the top $k$ directions of variance |
| Image compression | The singular values of the pixel grid |
| Search over documents | The word-by-document count matrix |
| Recommender systems | The user-by-item rating matrix |
| LoRA | The *update* to a weight matrix, forced to rank $r$ |

Nothing in the mathematics knows whether the matrix holds a photo, a ratings table, or neural network weights.`),

    key(`**The singular value spectrum is a diagnostic you can run on any matrix before committing to anything.**
Compute the $\\sigma_i$ and look at how they fall:

- A sharp **cliff** after a few values means there is genuine low-dimensional structure. Compress hard — the
  data really only varies in $k$ directions.
- A **slow, smooth decay** means there is no free lunch. The data fills its space, and truncating will throw
  away information you needed.

Run this before assuming PCA, a low-rank factorization, or a small LoRA rank will work on your problem. The code
block below prints exactly this.`),

    t(`### One more thing the spectrum tells you

The **condition number** is the largest singular value divided by the smallest:

$$\\kappa = \\frac{\\sigma_1}{\\sigma_r}$$

It says how much a small wobble in the input can be amplified into a large wobble in the output. If
$\\kappa = 10^6$, six digits of accuracy in your input can be entirely eaten by the matrix, so an answer you
compute from it deserves suspicion.

For optimization the picture is concrete: the loss surface is a bowl stretched $\\kappa$ times longer in one
direction than another — a long narrow valley. Gradient descent then has to use a step size small enough not to
overshoot the steep walls, and that same small step barely moves it along the flat floor. Momentum, Adam, and
normalization layers all exist largely to fight this.`),

    deriv('Where the SVD comes from, if you already believe in eigenvectors', `The SVD is not pulled from nowhere. It falls out of applying the symmetric-matrix result to $A^{\\mathsf T}A$.

For any matrix $A$, the product $A^{\\mathsf T}A$ is square and symmetric — check the second claim: $(A^{\\mathsf T}A)^{\\mathsf T} = A^{\\mathsf T}(A^{\\mathsf T})^{\\mathsf T} = A^{\\mathsf T}A$. So it has real eigenvalues and orthogonal eigenvectors, no matter what shape $A$ had. Now suppose the SVD exists and substitute it in:

$$A^{\\mathsf T}A = (U\\Sigma V^{\\mathsf T})^{\\mathsf T}(U\\Sigma V^{\\mathsf T}) = V\\Sigma^{\\mathsf T}U^{\\mathsf T}U\\Sigma V^{\\mathsf T} = V\\Sigma^2 V^{\\mathsf T}$$

using $U^{\\mathsf T}U = I$ because $U$ is orthogonal. Compare that with the symmetric form $Q\\Lambda Q^{\\mathsf T}$: the **right singular vectors $V$ are the eigenvectors of $A^{\\mathsf T}A$**, and the singular values are the square roots of its eigenvalues. Running the same argument on $AA^{\\mathsf T} = U\\Sigma^2U^{\\mathsf T}$ produces $U$.

This also explains why PCA is described sometimes as "eigendecomposition of the covariance matrix" and sometimes as "SVD of the centred data matrix". For centred data $X$ the covariance is $\\frac{1}{n}X^{\\mathsf T}X$, so the two are the same computation. In practice you run the SVD directly on $X$: forming $X^{\\mathsf T}X$ first squares the condition number and throws away precision for no reason.`),

    code('Reading a singular value spectrum', `import numpy as np

rng = np.random.default_rng(0)

# a 50x40 matrix that is secretly rank 3, plus a little noise
A = rng.normal(size=(50, 3)) @ rng.normal(size=(3, 40)) + 0.05 * rng.normal(size=(50, 40))

U, S, Vt = np.linalg.svd(A, full_matrices=False)
print("singular values:", np.round(S[:8], 3))
print("  ^ the cliff after the 3rd is the low-rank structure showing through\\n")

def rank_k(k):
    """Keep only the first k terms of the sum of outer products."""
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k]

for k in [1, 2, 3, 5, 10]:
    err = np.linalg.norm(A - rank_k(k)) / np.linalg.norm(A)
    store = k * (50 + 40) / (50 * 40)
    print(f"rank {k:2d}:  relative error {err:.4f}    storage {store:.0%} of full")

print("\\ncondition number:", round(S[0] / S[-1], 1))`,
      'Two things to take away. The error barely improves past k=3, because there was never any structure past the third direction — only noise. And rank 3 costs 13% of the storage. The spectrum told you where to cut before you tried a single value of k.'),

    quiz('You plot the singular values of your 1000×1000 data matrix and they decay slowly and smoothly, with no cliff. What should you conclude?',
      ['Low-rank compression will cost you real information here — the data genuinely varies in many directions',
       'The matrix is rank 1',
       'PCA will work especially well on this data',
       'The matrix is not invertible'],
      0,
      'A cliff in the spectrum means the matrix is nearly a sum of a few outer products, so truncating is cheap. A smooth decay means every direction is carrying a comparable share, and each one you drop costs you something. Eckart–Young still guarantees the truncated SVD is the *best* rank-$k$ approximation — but "best available" is not the same as "good enough", and the spectrum is what tells you which situation you are in.'),

    quiz('A loss surface has Hessian eigenvalues 100 and 0.1. What should you expect from plain gradient descent?',
      ['Slow, zig-zagging progress: the surface is a long narrow valley with a condition number of 1000',
       'Fast convergence, since one direction has strong curvature',
       'Divergence no matter what step size you pick',
       'Nothing in particular — curvature does not affect gradient descent'],
      0,
      'The condition number here is $100/0.1 = 1000$, so the bowl is a thousand times longer in one direction than the other. The step size has to stay small enough not to overshoot the steep direction, and that same small step then makes almost no progress along the flat one, so descent bounces across the valley instead of running down it. This is exactly the problem momentum and Adam were designed for — you can watch it happen in the [optimization lesson](#/l/math-optimization).'),

    recap(`- State the eigenvector equation $A\\mathbf{v} = \\lambda\\mathbf{v}$ and explain what is remarkable about
  it: a whole matrix collapsing to one number along one direction.
- Find the eigenvalues of a 2×2 matrix by solving $\\det(A - \\lambda I) = 0$, and say why that condition is the
  right one.
- Explain vanishing and exploding gradients as a statement about $\\lambda^t$.
- Recite the SVD as "rotate, stretch, rotate", and say why it exists for every matrix when eigendecomposition
  does not.
- Explain why only $\\Sigma$ can flatten a dimension, and connect that to rank.
- Read a singular value spectrum and decide whether low-rank compression will work on your data.
- Say what Eckart–Young guarantees and what it does *not* guarantee.
- Explain what a large condition number does to gradient descent.`),
  ],
  refs: [
    paper('The approximation of one matrix by another of lower rank', 'Eckart & Young', 1936, 'https://doi.org/10.1007/BF02288367', 'The original optimality theorem for truncated SVD.'),
    video('Eigenvectors and eigenvalues', '3Blue1Brown', 2016, 'https://www.youtube.com/watch?v=PFDu9oVAE-g', 'The geometric picture, in ten minutes, if the algebra above has not landed yet.'),
    blog('MIT 18.065 — Matrix Methods in Data Analysis', 'Gilbert Strang', 2018, 'https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/', 'An entire course built around the SVD and its uses in ML. Free lectures.'),
    paper('Finding Structure with Randomness', 'Halko, Martinsson & Tropp', 2011, 'https://arxiv.org/abs/0909.4061', 'Randomized SVD — how you actually compute this on matrices too big to fit in memory.'),
    book('Numerical Linear Algebra', 'Trefethen & Bau', 1997, 'https://people.maths.ox.ac.uk/trefethen/text.html', 'Beautifully written. Lectures 4–5 on the SVD are worth the price alone.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'math-derivatives',
  title: 'Derivatives and Gradients',
  sub: 'The local linear approximation — and the only thing an optimizer ever sees.',
  mins: 25, level: 'foundations',
  tags: ['calculus', 'gradients'],
  sections: [
    tldr(`Training a model means asking one question over and over: *if I nudge this number a little, does the
loss go up or down, and by how much?* The **derivative** answers it for one number. The **gradient** answers it
for all of them at once, packaged as a vector with one entry per parameter.

The gradient has one property that makes the whole enterprise work: it points in the direction the loss
increases fastest. So to decrease the loss, step the opposite way. That is all of gradient descent. Everything
else in optimization is an argument about how big the step should be.

The other half of the lesson is the chain rule, because a neural network is functions inside functions, and the
chain rule says their derivatives get **multiplied**. Multiplying many numbers together is unstable, and that
one fact predicts most of the things that go wrong during training.`),

    jargon([
      ['derivative $f\'(x)$', 'The slope of $f$ at the point $x$: how fast the output moves when you wiggle the input.'],
      ['$\\lim_{h\\to 0}$', 'Read as "what value does this approach as $h$ shrinks toward zero?" It is how you say "an infinitely small nudge" without literally dividing by zero.'],
      ['partial derivative $\\partial f/\\partial x_i$', 'The slope with respect to *one* input while every other input is held frozen. The curly $\\partial$ instead of $d$ is the only thing announcing "there are other variables in this function".'],
      ['gradient $\\nabla f$', 'All the partial derivatives collected into one vector, in the same order as the inputs. Say "grad f"; the $\\nabla$ symbol is called nabla. It has the same shape as the parameter vector.'],
      ['contour / level set', 'The set of input points where $f$ takes the same value — the rings on a topographic map.'],
      ['chain rule', 'The rule for differentiating a function of a function. The derivatives of the pieces get multiplied together.'],
      ['Hessian', 'The matrix of all second derivatives. It describes curvature: how fast the slope itself is changing, and in which directions.'],
      ['critical point', 'A point where the gradient is exactly zero. Could be a minimum, a maximum, or neither.'],
      ['saddle point', 'A critical point that is neither a peak nor a valley — uphill in some directions and downhill in others, like a mountain pass.'],
      ['Taylor series', 'A way of approximating a curvy function near a point using a line, then a parabola, then a cubic, each one closer than the last.'],
    ]),

    t(`## What a derivative is

$$f'(x) = \\lim_{h\\to 0}\\frac{f(x+h)-f(x)}{h}$$

Look at the fraction before worrying about the limit. The numerator $f(x+h)-f(x)$ is how much the output
changed; the denominator $h$ is how much the input changed. Their ratio is rise over run, which is a slope. The
$\\lim_{h\\to 0}$ then shrinks the run toward nothing, so the answer describes the function *at the point*
instead of averaged over an interval.

There are three ways to read the number that comes out. All three are correct, and you want all three
available:

1. **The slope** of the tangent line at $x$. This is the picture.
2. **A rate of change**: nudge $x$ by a small amount $\\epsilon$ and $f$ changes by about $f'(x)\\,\\epsilon$.
   This is the sentence you say out loud.
3. **The best straight-line stand-in** near $x$: $f(x+h) \\approx f(x) + f'(x)\\,h$. This is the one that
   generalizes.

Reading 3 is what optimization actually uses. It says: near this point, pretend the function is a straight
line, and here is the right straight line. Every gradient step you will ever take is a bet that the pretence
still holds over the distance you moved.`),

    viz('derivative-tangent'),

    t(`## Gradients: one derivative per parameter

A network's loss does not depend on one number. It depends on millions. So we need one derivative per
parameter, and the way to get them is to differentiate with respect to one input at a time while treating the
others as constants. That is a **partial derivative**, and there is nothing new in it — it is the single-variable
derivative you already know, applied while the rest of the inputs sit still.`),

    steps('A partial derivative, worked out', [
      { h: 'Take a function of two inputs', md: `$f(x, y) = x^2 y + 3y$. There is no single "the derivative" here, because there are two directions you could move.` },
      { h: 'Freeze $y$ and differentiate in $x$', md: `Treat $y$ as if it were a fixed number like 7. Then $x^2y$ is (constant)$\\cdot x^2$, whose derivative is $2xy$, and $3y$ is a constant, whose derivative is 0. So $\\partial f/\\partial x = 2xy$.` },
      { h: 'Now freeze $x$ and differentiate in $y$', md: `Now $x^2y$ is (constant)$\\cdot y$, with derivative $x^2$, and $3y$ has derivative 3. So $\\partial f/\\partial y = x^2 + 3$.` },
      { h: 'Stack them into the gradient', md: `$\\nabla f(x,y) = (2xy,\\ x^2+3)$. At the point $(1, 2)$ that is $(4, 4)$: moving in $x$ and moving in $y$ each raise $f$ at the same rate there.` },
    ]),

    t(`In general, for $f: \\mathbb{R}^d \\to \\mathbb{R}$ — read that as "takes $d$ numbers, returns one number",
which is exactly the type signature of a loss function — the gradient is

$$\\nabla f(\\mathbf{x}) = \\left(\\frac{\\partial f}{\\partial x_1},\\ \\ldots,\\ \\frac{\\partial f}{\\partial x_d}\\right)$$

Each entry on its own is unremarkable. The interesting part is what the vector as a whole turns out to mean.`),

    key(`Two properties of the gradient, neither obvious from the definition, both load-bearing:

1. **$\\nabla f$ points in the direction of steepest increase**, and its length $\\|\\nabla f\\|$ is how steep
   that is.
2. **$\\nabla f$ is perpendicular to the contour** through that point.

Property 1 is why gradient descent steps along $-\\nabla f$. Property 2 is why, in every picture of gradient
descent you have seen, the arrows cross the contour rings at right angles.`),

    t(`Both properties come from one calculation, and it only uses the dot product from
[the first lesson](#/l/math-vectors).

Pick a direction to walk in: a unit vector $\\mathbf{u}$. How fast does $f$ change as you walk that way? Take the
same rise-over-run limit, but move along $\\mathbf{u}$ instead of along an axis:

$$D_{\\mathbf{u}}f = \\lim_{h\\to 0}\\frac{f(\\mathbf{x}+h\\mathbf{u})-f(\\mathbf{x})}{h} = \\nabla f \\cdot \\mathbf{u}$$

The right-hand equality is the chain rule doing its job, worked out in the derivation below. Take it for now and
look at what it says: **the rate of change in any direction is the gradient dotted with that direction.** So now
use the geometric form of the dot product, with $\\|\\mathbf{u}\\| = 1$:

$$D_{\\mathbf{u}}f = \\|\\nabla f\\|\\,\\|\\mathbf{u}\\|\\cos\\theta = \\|\\nabla f\\|\\cos\\theta$$

The only thing you control is $\\theta$, the angle between your direction and the gradient. That expression is
biggest when $\\cos\\theta = 1$, meaning you walk *along* the gradient — property 1. It is zero when
$\\cos\\theta = 0$, meaning you walk perpendicular to the gradient — and walking without changing $f$ is exactly
what staying on a contour means, which is property 2. And it is most negative at $\\cos\\theta = -1$: the fastest
way down is straight against the gradient, which is what gradient descent does.`),

    viz('gradient-field'),

    deriv('Why the directional derivative equals $\\nabla f \\cdot \\mathbf{u}$', `Define $g(h) = f(\\mathbf{x} + h\\mathbf{u})$, a plain single-variable function of $h$. Then $D_{\\mathbf{u}}f$ is exactly $g'(0)$.

Each coordinate of the input is $x_i + h u_i$, so as $h$ changes, coordinate $i$ changes at rate $u_i$. The multivariable chain rule (the next section covers the single-variable version; this is the same statement with a sum over inputs) gives

$$g'(h) = \\sum_{i=1}^{d} \\frac{\\partial f}{\\partial x_i}\\Big|_{\\mathbf{x}+h\\mathbf{u}} \\cdot u_i$$

Read that sum: each input contributes its own sensitivity $\\partial f/\\partial x_i$ times how fast that input is moving, $u_i$. Setting $h = 0$ and recognising the sum as a dot product:

$$g'(0) = \\sum_i \\frac{\\partial f}{\\partial x_i}u_i = \\nabla f(\\mathbf{x}) \\cdot \\mathbf{u} \\qquad \\blacksquare$$

This is worth noticing as a pattern: the gradient is defined by $d$ separate axis-aligned questions, but it silently answers the question for *every* direction at once.`),

    t(`## The chain rule is the whole of backpropagation

Real models are functions inside functions inside functions. The chain rule says how to differentiate those, and
the punchline is that the derivatives get **multiplied**:

$$\\frac{d}{dx}g(h(x)) = g'(h(x))\\cdot h'(x)$$

In words: how sensitive the final output is to $x$ equals how sensitive $g$ is to its input, times how sensitive
$h$ is to $x$. Sensitivities compound, the way percentage changes compound — a 10% increase followed by a 10%
increase multiplies to 1.21×, it does not add to 1.2×.

A neural network is a deep composition: the last layer applied to the layer before it, applied to the one before
that, all the way down to the input. So the derivative of the loss with respect to an *early* parameter is a
product of one local derivative per layer that the signal has to travel back through. That product structure is
not a technicality. It explains most of the ways training fails.`),

    viz('chain-rule'),

    warn(`Multiplying many numbers together is unstable in a way that adding them is not.

Suppose every factor in the chain is a little under 1 — say 0.8. After 30 layers the product is
$0.8^{30} \\approx 0.001$. The gradient arriving at the earliest layers is a thousandth of what the last layers
see, so those early layers barely move. These are **vanishing gradients**.

Now suppose every factor is a little over 1 — say 1.2. Then $1.2^{30} \\approx 237$, and after a hundred layers
the number overflows to \`NaN\`. These are **exploding gradients**.

Neither is a bug in backpropagation; it is simply what exponentials do. Notice how narrow the safe zone is: the
factors have to sit near 1, and 0.8 is not near enough. Every fix you will meet later — careful initialization,
normalization layers, residual connections, ReLU instead of sigmoid — is ultimately a scheme for keeping those
per-layer factors close to 1.`),

    t(`## Second derivatives: curvature

The first derivative gives the slope. The **second** derivative gives how fast the slope is changing — whether
you are in a tight bowl or on a gentle plain. With many inputs there is a second derivative for each *pair* of
directions, and they form a matrix called the **Hessian**:

$$H_{ij} = \\frac{\\partial^2 f}{\\partial x_i \\partial x_j}$$

Read entry $(i,j)$ as: take the slope along axis $i$, then ask how fast *that slope* changes as you move along
axis $j$. The diagonal entries ($i = j$) are the ordinary "how curved is it along this axis" numbers; the
off-diagonal ones say whether the directions interact. The Hessian is always symmetric, because doing the two
differentiations in the other order gives the same answer.

The Hessian classifies any critical point. Look at its [eigenvalues](#/l/math-eigen-svd) — the curvatures along
its own natural axes:

- All eigenvalues **positive**: curves upward in every direction, so it is a local **minimum** (a bowl).
- All **negative**: curves downward everywhere, so it is a local **maximum** (a dome).
- **Mixed signs**: up in some directions, down in others — a **saddle point**.`),

    intuition(`Here is a counting argument worth keeping. For a critical point to be a local minimum, *all* $d$
Hessian eigenvalues have to come out positive. In a loss landscape with $d$ in the millions, that is like asking
a million coin flips to all land heads. So a point where the gradient vanishes is overwhelmingly likely to be a
**saddle**, not a minimum.

This reframed the field around 2014. The old worry was "deep networks get stuck in bad local minima". The better
description is that bad local minima are rare, while saddles and long flat plateaus are everywhere — and the
randomness in stochastic gradient descent is usually enough to slide off a saddle, since a saddle only needs one
downhill direction to escape through. That is a much more optimistic picture, and it matches what practitioners
observe.`),

    t(`## Taylor series: where "small learning rate" comes from

A Taylor series approximates a curvy function near a point by adding polynomial corrections one at a time:

$$f(x+h) = \\underbrace{f(x)}_{\\text{a flat guess}} + \\underbrace{f'(x)h}_{\\text{a line}} + \\underbrace{\\tfrac{1}{2}f''(x)h^2}_{\\text{a parabola}} + \\tfrac{1}{6}f'''(x)h^3 + \\cdots$$

Each term uses a higher derivative and a higher power of $h$, the distance you moved. That second factor is what
makes the series useful: if $h = 0.01$ then $h^2 = 0.0001$, so the later terms fade out fast and cutting the
series short is safe *provided $h$ is small*.

Every optimizer is a choice of where to cut:

| Cut after | You are assuming the loss nearby is | Methods that assume it |
|---|---|---|
| the constant term | flat | random search |
| the linear term | a straight line or plane | gradient descent, SGD, Adam |
| the quadratic term | a parabola or bowl | Newton's method, L-BFGS |

And here is the payoff. Both approximations are only valid *near* $x$, and how near depends on how curved the
function is. Take a step so large that the straight line you extrapolated along stopped describing the function
partway through, and the step can easily make the loss worse. That is the honest reason learning rates have to
be small: not tradition, but the distance over which your linear approximation is still telling the truth.`),

    viz('taylor-approx'),

    code('The chain rule by hand, checked numerically', `import numpy as np

x = np.array([1.0, 2.0, -1.0])     # a fixed input
y = 0.7                            # the target

def f(w):
    """A tiny loss: squared error after a tanh."""
    return (np.tanh(w @ x) - y) ** 2

def grad_analytic(w):
    z    = w @ x                   # step 1: the linear part
    pred = np.tanh(z)              # step 2: the squashing
    # loss = (pred - y)^2, so peel the composition apart one layer at a time:
    dL_dpred = 2 * (pred - y)      # derivative of the square
    dpred_dz = 1 - pred**2         # derivative of tanh
    dz_dw    = x                   # derivative of w @ x with respect to w
    return dL_dpred * dpred_dz * dz_dw     # <- the chain rule: multiply them

def grad_numeric(w, h=1e-6):
    """Nudge each weight in turn and watch the loss. No calculus involved."""
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w); e[i] = h
        g[i] = (f(w + e) - f(w - e)) / (2 * h)
    return g

w = np.array([0.3, -0.5, 0.8])
print("by chain rule:", grad_analytic(w).round(8))
print("by nudging   :", grad_numeric(w).round(8))
print("they agree to", f"{np.abs(grad_analytic(w) - grad_numeric(w)).max():.1e}")`,
      'Two ways to get the same numbers. The chain-rule version is what a framework computes and it is fast; the nudging version needs two extra evaluations per parameter and would be hopeless for a real model, but it needs no derivation and so makes an excellent cross-check. Comparing them is called gradient checking, and it is the standard way to catch a mistake in a hand-written backward pass.'),

    quiz('A 20-layer network uses sigmoid activations. The derivative of a sigmoid is never larger than 0.25. What happens to the gradient reaching the first layer?',
      ['It is multiplied by at most 0.25 twenty times over, giving roughly $10^{-12}$ — the first layer barely learns',
       'It is 20 times larger than the gradient at the last layer',
       'It is unaffected; each layer gets its own independent gradient',
       'It explodes toward NaN'],
      0,
      'Backpropagation multiplies one factor per layer, so a ceiling of 0.25 per layer means a ceiling of $0.25^{20} \\approx 10^{-12}$ over twenty of them — and in practice it is even smaller, because a sigmoid only reaches 0.25 at its steepest point. Historically this is exactly what blocked deep networks until ReLU (whose derivative is 1 on the positive side), better initialization, and residual connections arrived. The fix in every case is the same: keep the per-layer factor near 1.'),

    recap(`- Give all three readings of a derivative, and say which one optimization depends on.
- Compute a partial derivative by freezing the other variables, and assemble a gradient.
- Explain why $-\\nabla f$ is the direction to step, using $\\nabla f \\cdot \\mathbf{u} = \\|\\nabla f\\|\\cos\\theta$.
- Say why the gradient is perpendicular to the contours, without hand-waving.
- Explain vanishing and exploding gradients as a fact about multiplying many numbers, and predict which one you
  will get from the size of a typical factor.
- Read the signs of the Hessian's eigenvalues to classify a critical point, and say why saddles vastly outnumber
  minima in high dimensions.
- Justify a small learning rate by pointing at the truncated Taylor series rather than at tradition.`),
  ],
  refs: [
    video('Essence of Calculus', '3Blue1Brown', 2017, 'https://www.3blue1brown.com/topics/calculus', 'Builds the geometric picture of derivatives from scratch.'),
    book('Mathematics for Machine Learning, Ch. 5', 'Deisenroth, Faisal & Ong', 2020, 'https://mml-book.github.io/', 'Vector calculus, Jacobians, and backprop, done carefully.'),
    paper('Identifying and attacking the saddle point problem', 'Dauphin et al.', 2014, 'https://arxiv.org/abs/1406.2572', 'The paper that reframed deep learning optimization around saddle points rather than local minima.'),
    blog('The Matrix Calculus You Need For Deep Learning', 'Parr & Howard', 2018, 'https://explained.ai/matrix-calculus/', 'Exactly what the title says, and no more. Excellent.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'math-jacobian',
  title: 'Jacobians, Vector Calculus, and Matrix Derivatives',
  sub: 'What backprop actually multiplies, and how to differentiate expressions with matrices in them.',
  mins: 25, level: 'foundations',
  prereq: ['math-derivatives'],
  tags: ['calculus', 'backprop'],
  sections: [
    tldr(`A gradient is what you get when a function has many inputs and *one* output. A **Jacobian** is what you
get when it has many inputs and many outputs — a whole matrix of partial derivatives.

Every layer of a neural network is a many-in, many-out function, so backprop is conceptually a long product of
Jacobian matrices. The practical insight is that you never build them: for the shapes involved, forming one
Jacobian would cost more memory than the entire model. Autodiff multiplies by them without ever writing them
down, and that trick is what makes training large networks possible at all.`),

    jargon([
      ['Jacobian $J$', 'The derivative of a function with many inputs *and* many outputs. An $m \\times n$ grid: row $i$, column $j$ says how output $i$ responds to input $j$.'],
      ['$f: \\mathbb{R}^n \\to \\mathbb{R}^m$', 'A type signature: "takes $n$ numbers, returns $m$ numbers". A gradient is the special case $m = 1$.'],
      ['VJP (vector–Jacobian product)', 'Computing $\\mathbf{v}^{\\mathsf T}J$ *without ever constructing* $J$. The core operation of backprop.'],
      ['JVP (Jacobian–vector product)', 'The mirror image, $J\\mathbf{v}$. What forward-mode autodiff computes.'],
      ['autodiff / automatic differentiation', 'A framework computing exact derivatives by tracking operations as you run the forward pass. Not symbolic algebra, and not finite differences — a third thing.'],
      ['reverse mode', 'Autodiff run backwards from the output. Cheap when there are few outputs and many inputs — i.e. always, in deep learning. "Backpropagation" is this.'],
      ['activations', 'The intermediate values a network computes on its way from input to output. Reverse mode has to remember them.'],
      ['gradient checkpointing', 'Deliberately throwing away stored activations and recomputing them during the backward pass. Trades compute for memory.'],
      ['logits', 'The raw scores a classifier outputs before softmax turns them into probabilities.'],
      ['one-hot', 'Encoding a category as a vector of zeros with a single 1 in the correct slot. Class 2 of 5 becomes $(0,0,1,0,0)$.'],
    ]),

    t(`## The Jacobian

So far the functions have been many-in, one-out: a loss takes all your parameters and returns a single number.
Its derivative is a gradient — one vector.

But a *layer* is not like that. It takes $n$ numbers and returns $m$ numbers. Now there is no single slope,
because there are $m \\times n$ separate questions to ask: how does output 1 respond to input 1, how does
output 1 respond to input 2, and so on. Collect all those answers into a grid and you have the **Jacobian**:

$$J_{ij} = \\frac{\\partial f_i}{\\partial x_j} \\qquad \\text{(row } i \\text{ = which output, column } j \\text{ = which input)}$$

Despite being a matrix, it means what derivatives always mean — the best local linear approximation:

$$f(\\mathbf{x}+\\mathbf{h}) \\approx f(\\mathbf{x}) + J\\mathbf{h}$$

Compare that with $f(x+h) \\approx f(x) + f'(x)h$ from the previous lesson. Identical sentence; the slope has
just become a matrix, because "multiply by a matrix" is what a linear map looks like in many dimensions.`),

    viz('jacobian'),

    t(`## The chain rule with Jacobians

If a derivative in one dimension is a number, and the chain rule multiplies numbers, then a derivative in many
dimensions is a matrix and the chain rule multiplies matrices:

$$J_{g\\circ f}(\\mathbf{x}) = J_g(f(\\mathbf{x}))\\; J_f(\\mathbf{x})$$

(That $\\circ$ means composition — "$g$ applied to the result of $f$".) And since matrix multiplication *is*
function composition, this is the same statement as before rather than a new rule.

Stack that up over a network $L$ layers deep and the gradient of the loss with respect to the first layer's
activations is a chain of $L-1$ Jacobians:

$$\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{h}_1} = \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{h}_L}\\, J_L\\, J_{L-1}\\cdots J_2$$

Which raises an immediate practical problem, and the answer to it is the whole design of modern autodiff.`),

    key(`**Backprop never builds these matrices.** For a layer with 4096 inputs and outputs, the Jacobian is
$4096^2 = 16.8$M entries — per example, per layer. Instead, autodiff computes **vector–Jacobian products**
$\\mathbf{v}^{\\mathsf T}J$ directly. For a linear layer $\\mathbf{y}=W\\mathbf{x}$ the VJP is just
$\\mathbf{v}^{\\mathsf T}W$, i.e. one matrix-vector multiply. For an elementwise activation the Jacobian is diagonal, so
the VJP is an elementwise multiply.

This is the entire computational trick behind reverse-mode automatic differentiation.`),

    t(`## Forward mode vs reverse mode

A product like $J_L J_{L-1} \\cdots J_2$ can be evaluated in either order — matrix multiplication is
associative, so the answer is the same. But the *cost* is wildly different, and choosing correctly is the
single most consequential decision in the design of an autodiff system.

- **Forward mode** goes left to right through the network, computing $J\\mathbf{v}$ (Jacobian–vector products).
  Each pass gives you the derivative with respect to **one input**. Cost scales with the number of *inputs*.
- **Reverse mode** goes right to left, computing $\\mathbf{v}^{\\mathsf T}J$ (vector–Jacobian products). Each
  pass gives you the derivative of **one output** with respect to everything. Cost scales with the number of
  *outputs*.

Now count. Deep learning has $10^9$ inputs (the parameters) and exactly **one** output (the scalar loss). Forward
mode would need a billion passes. Reverse mode needs one. That factor of a billion is the entire reason
"backpropagation" is the algorithm — and backpropagation is nothing more exotic than reverse-mode autodiff
applied to a neural network.`),

    diagram('Why the multiplication order matters so much',
`<svg viewBox="0 0 620 200" role="img" aria-label="Forward mode versus reverse mode cost">
  <text class="dtitle" x="20" y="24">FORWARD — one pass per input</text>
  <g>
    <rect x="20" y="38" width="60" height="46" rx="3" style="fill: color-mix(in srgb, var(--s6) 14%, transparent); stroke: var(--s6)"/>
    <text class="dmono" x="50" y="63" text-anchor="middle" style="fill: var(--s6)">J2</text>
    <text x="93" y="63" text-anchor="middle" style="font-size:14px; fill: var(--text-faint)">·</text>
    <rect x="106" y="38" width="60" height="46" rx="3" style="fill: color-mix(in srgb, var(--s6) 14%, transparent); stroke: var(--s6)"/>
    <text class="dmono" x="136" y="63" text-anchor="middle" style="fill: var(--s6)">J3</text>
    <text x="179" y="63" text-anchor="middle" style="font-size:14px; fill: var(--text-faint)">·</text>
    <rect x="192" y="38" width="60" height="46" rx="3" style="fill: color-mix(in srgb, var(--s6) 14%, transparent); stroke: var(--s6)"/>
    <text class="dmono" x="222" y="63" text-anchor="middle" style="fill: var(--s6)">J4</text>
    <text class="dlabel" x="300" y="63" style="fill: var(--s6)">start left: matrix x matrix, over and over</text>
    <text class="dlabel" x="300" y="80" style="fill: var(--text-faint)">10^9 passes for 10^9 parameters</text>
  </g>
  <text class="dtitle" x="20" y="124">REVERSE — one pass, total</text>
  <g>
    <rect x="20" y="138" width="30" height="46" rx="3" style="fill: color-mix(in srgb, var(--s3) 22%, transparent); stroke: var(--s3)"/>
    <text class="dmono" x="35" y="163" text-anchor="middle" style="fill: var(--s3)">v</text>
    <text x="62" y="163" text-anchor="middle" style="font-size:14px; fill: var(--text-faint)">·</text>
    <rect x="76" y="138" width="60" height="46" rx="3" style="fill: color-mix(in srgb, var(--s3) 14%, transparent); stroke: var(--s3)"/>
    <text class="dmono" x="106" y="163" text-anchor="middle" style="fill: var(--s3)">J4</text>
    <text x="149" y="163" text-anchor="middle" style="font-size:14px; fill: var(--text-faint)">·</text>
    <rect x="162" y="138" width="60" height="46" rx="3" style="fill: color-mix(in srgb, var(--s3) 14%, transparent); stroke: var(--s3)"/>
    <text class="dmono" x="192" y="163" text-anchor="middle" style="fill: var(--s3)">J3</text>
    <text class="dlabel" x="300" y="163" style="fill: var(--s3)">start right: vector x matrix, always</text>
    <text class="dlabel" x="300" y="180" style="fill: var(--text-faint)">1 pass, because the loss is 1 number</text>
  </g>
</svg>`,
      `The trick is starting from the side where the vector is. A vector times a matrix stays a vector, so every
subsequent step is a cheap matrix–vector multiply. Start from the other end and you are doing full matrix–matrix
products the whole way. Same answer, wildly different bill.`),

    warn(`Reverse mode's advantage is not free — it is bought with **memory**. To compute a backward pass you
need the activations from the forward pass, so all of them must be kept alive until the backward pass reaches
them. For a large model on a long sequence, activations, not weights, are what fills your GPU.

**Gradient checkpointing** is the standard escape hatch: store only every $k$-th layer's activations and
recompute the rest on the way back. Roughly $\\sqrt{L}$ memory instead of $L$, for about 30% more compute. When
you see "OOM during backward but forward was fine", this is why, and this is the fix.`),

    t(`## Matrix calculus you will actually need

You do not need to memorize a table, but these five come up constantly. Assume $\\mathbf{a}$, $\\mathbf{x}$ are vectors,
$A$ a matrix, and $\\mathcal{L}$ a scalar.

| Expression | Derivative |
|---|---|
| $\\frac{\\partial}{\\partial \\mathbf{x}}(\\mathbf{a}^{\\mathsf T}\\mathbf{x})$ | $\\mathbf{a}$ |
| $\\frac{\\partial}{\\partial \\mathbf{x}}(\\mathbf{x}^{\\mathsf T}A\\mathbf{x})$ | $(A + A^{\\mathsf T})\\mathbf{x}$, and $2A\\mathbf{x}$ if $A$ symmetric |
| $\\frac{\\partial}{\\partial \\mathbf{x}}\\|\\mathbf{x}\\|^2$ | $2\\mathbf{x}$ |
| $\\frac{\\partial \\mathcal{L}}{\\partial W}$ where $\\mathbf{y}=W\\mathbf{x}$ | $\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{y}}\\mathbf{x}^{\\mathsf T}$ (outer product) |
| $\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{x}}$ where $\\mathbf{y}=W\\mathbf{x}$ | $W^{\\mathsf T}\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{y}}$ |

The last two are the forward and backward passes of a linear layer, and they are the two lines at the heart of every
deep learning framework.`),

    intuition(`**The shape rule.** If you forget a formula, reconstruct it from shapes. $\\partial\\mathcal{L}/\\partial W$
must have the same shape as $W$ (that is $m\\times n$). You have $\\partial\\mathcal{L}/\\partial\\mathbf{y}$ ($m$-vector)
and $\\mathbf{x}$ ($n$-vector). The only way to combine them into $m\\times n$ is the outer product
$\\delta\\mathbf{x}^{\\mathsf T}$. This trick resolves most sign-and-transpose confusion in practice.`),

    deriv('The softmax + cross-entropy gradient', `This is the most important derivative in classification, and it collapses beautifully.

Let $z$ be the logits, $p = \\text{softmax}(z)$, and the loss be cross-entropy against a one-hot target $y$:
$\\mathcal{L} = -\\sum_k y_k \\log p_k$.

First, the softmax Jacobian. With $p_i = e^{z_i}/\\sum_j e^{z_j}$:

$$\\frac{\\partial p_i}{\\partial z_j} = p_i(\\delta_{ij} - p_j)$$

Now the chain rule, using $\\partial\\mathcal{L}/\\partial p_i = -y_i/p_i$:

$$\\frac{\\partial \\mathcal{L}}{\\partial z_j} = \\sum_i \\frac{\\partial \\mathcal{L}}{\\partial p_i}\\frac{\\partial p_i}{\\partial z_j}
= -\\sum_i \\frac{y_i}{p_i}\\,p_i(\\delta_{ij}-p_j) = -y_j + p_j\\sum_i y_i$$

Since $y$ is one-hot, $\\sum_i y_i = 1$, giving

$$\\boxed{\\frac{\\partial\\mathcal{L}}{\\partial \\mathbf{z}} = \\mathbf{p} - \\mathbf{y}}$$

Just "predicted minus actual." Every framework fuses softmax and cross-entropy into one op for exactly this reason:
the fused gradient is numerically stable and costs nothing, whereas computing them separately involves dividing by
$p_i$, which can underflow.`),

    code('Vector-Jacobian products, by hand', `import numpy as np

rng = np.random.default_rng(0)
W = rng.normal(size=(4, 6)) * 0.5
x = rng.normal(size=6)

# forward
y = np.tanh(W @ x)

# suppose the loss gradient w.r.t. y is v
v = rng.normal(size=4)

# --- the expensive way: build the Jacobian explicitly ---
z = W @ x
J = np.diag(1 - np.tanh(z)**2) @ W        # (4,6) Jacobian of y w.r.t. x
grad_x_explicit = v @ J

# --- the way autodiff does it: never form J ---
dz = v * (1 - np.tanh(z)**2)              # elementwise: diagonal Jacobian
grad_x_vjp = dz @ W                       # one matvec
grad_W_vjp = np.outer(dz, x)              # outer product

print("match:", np.allclose(grad_x_explicit, grad_x_vjp))
print("explicit Jacobian entries:", J.size, " vs VJP work:", dz.size + W.size)

# scale it up to see why this matters
d = 4096
print(f"\\nFor a {d}x{d} layer:")
print(f"  explicit Jacobian: {d*d:,} floats = {d*d*4/1e6:.0f} MB per example")
print(f"  VJP:               {d:,} floats  = {d*4/1e3:.0f} KB")`),

    quiz('Why does reverse-mode autodiff, not forward mode, power deep learning?',
      ['Cost scales with the number of outputs; a loss has exactly one, while parameters number in the billions',
       'Reverse mode is numerically more accurate',
       'Forward mode cannot handle nonlinearities',
       'Reverse mode uses less memory'],
      0,
      'Forward mode costs one pass per *input* dimension; reverse mode costs one pass per *output* dimension. With $10^9$ parameters and a single scalar loss, reverse mode is a billion times cheaper. It actually uses **more** memory (it must cache activations) — that is the trade you accept, and gradient checkpointing is how you claw some of it back.'),

    recap(`- Say what a Jacobian is and how it relates to a gradient (a gradient is the one-output case).
- Explain why backprop is a product of Jacobians, and why nobody ever builds one.
- Choose reverse mode over forward mode from a parameter count and an output count, and justify it in one
  sentence.
- Name the price reverse mode pays, and what gradient checkpointing buys back.
- Reconstruct $\\partial\\mathcal{L}/\\partial W = \\delta\\mathbf{x}^{\\mathsf T}$ from shapes alone when you
  cannot remember the formula.
- State the softmax + cross-entropy gradient from memory — "predicted minus actual" — and say why frameworks
  fuse the two operations.`),
  ],
  refs: [
    blog('The Matrix Calculus You Need For Deep Learning', 'Parr & Howard', 2018, 'https://explained.ai/matrix-calculus/', ''),
    book('The Matrix Cookbook', 'Petersen & Pedersen', 2012, 'https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf', 'The lookup table. Print it.'),
    paper('Automatic differentiation in machine learning: a survey', 'Baydin et al.', 2018, 'https://arxiv.org/abs/1502.05767', 'Forward vs reverse mode, and what frameworks actually implement.'),
    blog('Autodidax: JAX core from scratch', 'JAX team', 2021, 'https://jax.readthedocs.io/en/latest/autodidax.html', 'Build an autodiff system in a few hundred lines. The best way to truly understand it.'),
    paper('Training Deep Nets with Sublinear Memory Cost', 'Chen et al.', 2016, 'https://arxiv.org/abs/1604.06174', 'Gradient checkpointing: trade compute for the memory reverse mode demands.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'math-probability',
  title: 'Probability, Distributions, and Bayes',
  sub: 'The language for reasoning about uncertainty — which is all of machine learning.',
  mins: 30, level: 'foundations',
  tags: ['probability', 'statistics', 'Bayes'],
  sections: [
    tldr(`Every model in this atlas outputs a **distribution**, not an answer. A classifier does not say "cat",
it says "83% cat". A language model does not pick the next word, it scores every possible word. Probability is
the language for that, and once you see it you cannot unsee it: loss functions, regularization, and evaluation
metrics all turn out to be probabilistic statements wearing different clothes.

Three objects carry the whole lesson — **distributions** (what could happen and how often), **expectations**
(the average of something random), and **conditioning** (updating what you believe when new information
arrives). Bayes' rule is just conditioning written down carefully.`),

    jargon([
      ['random variable', 'A quantity whose value depends on chance. Written with a capital letter, $X$, while a specific value it took is lowercase, $x$.'],
      ['distribution', 'The full description of what values a random variable can take and how likely each is. Not a single number.'],
      ['$p(x)$', 'The probability (or density) of the value $x$. Shorthand that gets overloaded constantly — context tells you which variable it is about.'],
      ['$p(y \\mid x)$', 'Read: "the probability of $y$ **given** $x$". The vertical bar means "conditional on", i.e. assuming $x$ already happened. This is what a classifier outputs.'],
      ['joint $p(x, y)$', 'The probability of $x$ **and** $y$ both happening.'],
      ['marginal', 'A joint distribution with some variables summed away. "Marginalising out $y$" means "I no longer care about $y$, average over it".'],
      ['expectation $\\mathbb{E}[X]$', 'The long-run average value of $X$, weighted by how likely each outcome is. Not necessarily a value $X$ can actually take.'],
      ['variance', 'How spread out a distribution is. Big variance = unpredictable.'],
      ['i.i.d.', '"Independent and identically distributed" — each data point drawn from the same distribution, none influencing the others. The standard (often false) assumption behind nearly every method.'],
      ['likelihood', '$p(\\text{data} \\mid \\text{parameters})$, read as a function of the *parameters*. "How well does this setting explain what I saw?"'],
      ['prior / posterior', 'What you believed before seeing the data / after seeing it. Bayes\' rule converts one into the other.'],
      ['MLE / MAP', 'Maximum Likelihood Estimate (pick the parameters that best explain the data) / Maximum A Posteriori (same, but also weighted by your prior).'],
      ['conjugate prior', 'A prior chosen so the posterior comes out in the same family, making the update pure arithmetic instead of an integral.'],
    ]),

    t(`## Why probability

Every ML model is a claim about a distribution, even when it does not look like one:

- A classifier outputs $p(y \\mid \\mathbf{x})$ — a probability for each class given the input.
- A language model outputs $p(\\text{next token} \\mid \\text{context})$ — a score over the full vocabulary.
- A diffusion model *is* a sampler for $p(\\text{image})$ — it draws from the distribution of plausible images.
- Even plain linear regression is a distributional claim in disguise. Saying $y = f(\\mathbf{x}) + \\epsilon$
  with Gaussian noise $\\epsilon$ *is* a statement about a distribution, and it is the reason least squares is
  the right loss rather than an arbitrary one.

That last point generalises into one of the most clarifying facts in the subject, and we will derive it below:
**every loss function is a distributional assumption**. If you know what noise you believe in, the loss is
determined; you do not get to pick it separately.`),

    t(`## Random variables and distributions

A **random variable** is a quantity whose value depends on chance — a die roll, tomorrow's temperature, the
next token. The convention is capital $X$ for the variable and lowercase $x$ for a value it took.

How you describe its distribution depends on whether the values are countable:

- **Discrete** variables (die rolls, classes, tokens) have a **probability mass function** $p(x) = P(X = x)$.
  Each value gets an actual probability, and they sum to 1.
- **Continuous** variables (heights, weights, pixel intensities) have a **density** $p(x)$, where probability
  comes from *area*: $P(a\\le X\\le b)=\\int_a^b p(x)\\,dx$. The $\\int$ is a sum over a continuum.`),

    warn(`A density is not a probability. For a continuous variable, $p(x)$ can be greater than 1 — a narrow
Gaussian with $\\sigma = 0.01$ has a peak density around 40. Nothing is broken. The probability of landing
*exactly* on any single real number is zero; only the **area** under the curve is a probability, and only that
area must integrate to 1.

This trips people up when reading log-density values from a model and finding them positive. Positive
log-density is fine. Positive log-*probability* would not be.`),

    viz('distributions'),

    t(`### The ones you must recognize

- **Bernoulli / Binomial** — binary outcomes and their counts. Binary classification lives here.
- **Categorical** — one of $K$ outcomes. Every softmax output is a categorical distribution.
- **Gaussian** — the default for continuous noise. Justified by the CLT, and it is the maximum-entropy distribution
  given a fixed mean and variance (i.e. the least-committed choice when that's all you know).
- **Laplace** — heavier tails and a sharp peak. Its log-density is $-|x|$, which is why a Laplace prior gives L1
  regularization and sparse solutions.
- **Exponential / Poisson** — waiting times and counts of rare events.
- **Beta / Dirichlet** — distributions *over* probabilities. The conjugate priors for Bernoulli/categorical.`),

    t(`## Expectation and variance

The **expectation** $\\mathbb{E}[X]$ is the average value of $X$ you would see over infinitely many draws,
weighting each outcome by how likely it is:

$$\\mathbb{E}[X] = \\sum_x x\\,p(x) \\quad\\text{or, for continuous } X,\\quad \\int x\\,p(x)\\,dx$$

The **variance** measures spread — the average squared distance from the mean:

$$\\text{Var}(X) = \\mathbb{E}[(X-\\mathbb{E}[X])^2] = \\mathbb{E}[X^2]-\\mathbb{E}[X]^2$$

(The second form is the one you compute with; it follows from expanding the square.)

Two properties do almost all the work in ML, and the difference between them matters enormously:`),

    key(`**Linearity of expectation** holds *always*:
$\\mathbb{E}[aX+bY] = a\\mathbb{E}[X]+b\\mathbb{E}[Y]$ — even if $X$ and $Y$ are wildly dependent. No conditions,
no fine print.

*Why you care:* a minibatch gradient is an average of per-example gradients, so its expectation equals the full
gradient. Minibatch SGD is therefore **unbiased** — it takes the right step on average, no matter how small the
batch. That is the licence to train on batches of 32 instead of 50 million.

**Variance of a sum** does *not* hold always:
$\\text{Var}(X+Y) = \\text{Var}(X)+\\text{Var}(Y)$ only when $X$ and $Y$ are **uncorrelated**.

*Why you care:* when it does hold, averaging $B$ independent samples divides variance by $B$, so gradient noise
shrinks like $1/\\sqrt{B}$. Quadrupling the batch size only halves the noise — which is why batch size has
sharply diminishing returns, and why correlated samples (a batch of near-duplicate examples) buy you far less
than the count suggests.`),

    viz('clt'),

    t(`## Conditioning and Bayes' rule

**Conditioning** is what happens to a distribution when you learn something. Before you know anything, the
chance of rain is whatever it is; once you know the sky is dark, it changes. Written $p(\\text{rain} \\mid
\\text{dark sky})$.

Joint, marginal, and conditional distributions are tied together by one relation and one operation:

$$\\underbrace{p(x,y) = p(x\\mid y)\\,p(y)}_{\\text{the chain rule of probability}}, \\qquad \\underbrace{p(x) = \\sum_y p(x,y)}_{\\text{marginalising } y \\text{ away}}$$

The first says a joint probability factors into "the chance of $y$" times "the chance of $x$ once you know
$y$". Since $p(x,y) = p(y,x)$, you can factor it the other way too — and setting those two factorizations equal
and rearranging gives **Bayes' rule**:

$$\\underbrace{p(\\theta\\mid \\mathcal{D})}_{\\text{posterior}} = \\frac{\\overbrace{p(\\mathcal{D}\\mid\\theta)}^{\\text{likelihood}}\\ \\overbrace{p(\\theta)}^{\\text{prior}}}{\\underbrace{p(\\mathcal{D})}_{\\text{evidence}}}$$

That is the entire derivation — Bayes' rule is algebra, not a deep principle. What is deep is how you read it.`),

    steps("Reading Bayes' rule as a procedure", [
      { h: 'Start with a belief — the prior $p(\\theta)$', md: `Before seeing any data, what do you think the parameters are? "This coin is probably fair-ish." Choosing a prior is a modelling decision, and pretending you have not made one is itself a choice (a flat prior).` },
      { h: 'Ask how well each hypothesis explains the data — the likelihood $p(\\mathcal{D} \\mid \\theta)$', md: `For each candidate $\\theta$: if the world really worked that way, how likely was the data I actually observed? A $\\theta$ that makes your observations look like a miracle scores badly.` },
      { h: 'Multiply', md: `$p(\\mathcal{D}\\mid\\theta)\\,p(\\theta)$. A hypothesis needs *both* to be plausible beforehand *and* to explain the data. Failing either kills it.` },
      { h: 'Normalise — divide by the evidence $p(\\mathcal{D})$', md: `This denominator does not depend on $\\theta$ at all; it just makes everything sum to 1. Which is why you constantly see the rule written with a $\\propto$ ("proportional to") and the denominator dropped: $p(\\theta\\mid\\mathcal D) \\propto p(\\mathcal D\\mid\\theta)p(\\theta)$.` },
      { h: 'The result is your new belief — the posterior', md: `And it becomes the prior for the next batch of data. Belief updating is a loop, not a one-shot calculation.` },
    ]),

    viz('bayes-coin'),

    warn(`**The base rate trap.** A test is 99% accurate for a disease affecting 1 in 10,000. You test positive. The
probability you are sick is about **1%**, not 99%.

$$P(\\text{sick}\\mid+) = \\frac{0.99 \\times 0.0001}{0.99\\times 0.0001 + 0.01\\times 0.9999} \\approx 0.0098$$

Among 10,000 people, 1 is sick (and tests positive), while ~100 healthy people also test positive. The prior dominates.
This is the same arithmetic that makes precision collapse for rare-class classifiers — see the
[ROC and prevalence figure](#/l/ml-evaluation).`),

    t(`## Maximum likelihood — where loss functions come from

Suppose you have data $\\mathcal{D}=\\{x_1,\\ldots,x_n\\}$ and a family of models indexed by parameters $\\theta$.
Which $\\theta$ should you pick? Maximum likelihood says: **the one that makes the data you actually saw as
unsurprising as possible.**

If the data points are i.i.d., the probability of the whole dataset is the product of the individual
probabilities, $\\prod_i p(x_i\\mid\\theta)$. In practice we maximise its logarithm instead:

$$\\hat\\theta_{\\text{MLE}} = \\arg\\max_\\theta \\sum_{i=1}^n \\log p(x_i\\mid\\theta)$$

Two reasons for the log, both mundane and both important. First, $\\log$ turns products into sums, and sums are
far easier to differentiate. Second, a product of ten thousand numbers each below 1 underflows to exactly \`0.0\`
in float32 — the log keeps everything in a sane range. Since $\\log$ is increasing, whatever maximises the sum
also maximises the product, so nothing is lost.

**MAP** (maximum a posteriori) is the same thing with the prior kept:

$$\\hat\\theta_{\\text{MAP}} = \\arg\\max_\\theta \\Big[\\underbrace{\\textstyle\\sum_i \\log p(x_i\\mid\\theta)}_{\\text{fit the data}} + \\underbrace{\\log p(\\theta)}_{\\text{stay plausible}}\\Big]$$

And that second term is *exactly* regularization. Put a zero-mean Gaussian prior on the weights and $\\log
p(\\theta)$ works out to $-\\lambda\\|\\theta\\|^2$ — L2 / weight decay. Use a Laplace prior instead and you get
$-\\lambda\\|\\theta\\|_1$ — L1 and sparsity. Regularization was never a hack bolted onto the loss; it is what a
prior looks like after you take logs.`),

    viz('mle-fit'),

    deriv('Gaussian MLE ⇒ least squares', `Assume $y_i = f(x_i;\\theta) + \\epsilon_i$ with $\\epsilon_i \\sim \\mathcal{N}(0,\\sigma^2)$. Then

$$p(y_i \\mid x_i,\\theta) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}}\\exp\\!\\left(-\\frac{(y_i-f(x_i;\\theta))^2}{2\\sigma^2}\\right)$$

Take logs and sum:

$$\\log p(\\mathcal{D}\\mid\\theta) = -\\frac{1}{2\\sigma^2}\\sum_i (y_i - f(x_i;\\theta))^2 - \\frac{n}{2}\\log(2\\pi\\sigma^2)$$

The second term does not involve $\\theta$. Maximizing the first is identical to **minimizing the sum of squared
errors**. ∎

So least squares is not an arbitrary choice of loss — it is maximum likelihood under Gaussian noise. Change the noise
model and the loss changes with it: Laplace noise gives absolute error (L1), and a categorical model gives
cross-entropy. **Every loss function is a distributional assumption.** That is one of the most clarifying facts in ML.`),

    code('Bayes, MLE, and conjugacy in NumPy', `import numpy as np

# --- the base rate trap, computed ---
prior, sens, spec = 1/10_000, 0.99, 0.99
p_pos = sens*prior + (1-spec)*(1-prior)
print(f"P(sick | positive) = {sens*prior/p_pos:.4f}   <- not 0.99\\n")

# --- MLE for a Gaussian, closed form vs numerical ---
rng = np.random.default_rng(3)
data = rng.normal(loc=2.0, scale=1.5, size=200)
print("MLE mu    =", data.mean().round(4))
print("MLE sigma =", data.std(ddof=0).round(4), " (biased: divides by n)")
print("unbiased  =", data.std(ddof=1).round(4), " (divides by n-1)\\n")

# --- Beta-Bernoulli conjugacy: updating is just counting ---
a, b = 2.0, 2.0                      # prior Beta(2,2)
flips = rng.binomial(1, 0.7, size=50)
for i, f in enumerate(flips):
    a, b = a + f, b + (1 - f)
    if i in (0, 4, 19, 49):
        mean = a/(a+b)
        sd = np.sqrt(a*b/((a+b)**2*(a+b+1)))
        print(f"after {i+1:3d} flips: Beta({a:.0f},{b:.0f})  "
              f"mean={mean:.4f}  sd={sd:.4f}  MLE={flips[:i+1].mean():.4f}")`),

    quiz('You add L2 weight decay to your loss. In Bayesian terms, what have you done?',
      ['Placed a zero-mean Gaussian prior on the weights and switched from MLE to MAP estimation',
       'Placed a Laplace prior on the weights',
       'Changed the likelihood from Gaussian to Laplace',
       'Nothing Bayesian — weight decay is purely an optimization trick'],
      0,
      'MAP maximizes $\\log p(\\mathcal D\\mid\\theta)+\\log p(\\theta)$. With $\\theta\\sim\\mathcal N(0,\\tau^2 I)$, the log-prior is $-\\|\\theta\\|^2/(2\\tau^2)+\\text{const}$ — precisely an L2 penalty with $\\lambda = 1/(2\\tau^2)$. A Laplace prior would give L1 instead. (Careful: with Adam, decoupled weight decay as in AdamW is *not* the same as adding L2 to the loss — see the optimizers lesson.)'),

    recap(`- Say what $p(y \\mid \\mathbf{x})$ means out loud, and name what a classifier, a language model, and a
  diffusion model each put on the left of that bar.
- Explain why a density can exceed 1 while a probability cannot.
- Use linearity of expectation to justify minibatch SGD, and the variance rule to predict what doubling the
  batch size buys you.
- Walk through Bayes' rule as prior → likelihood → multiply → normalise, and explain why the denominator is
  usually dropped.
- Work the base rate trap and say why a 99%-accurate test for a rare disease is mostly wrong when positive.
- Derive least squares from Gaussian noise, and state the general principle: **every loss is a distributional
  assumption**.
- Translate weight decay into Bayesian language and back.`),
  ],
  refs: [
    book('Pattern Recognition and Machine Learning, Ch. 1–2', 'Christopher Bishop', 2006, 'https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/', 'The canonical Bayesian treatment. Free PDF from Microsoft Research.'),
    book('Probabilistic Machine Learning: An Introduction', 'Kevin Murphy', 2022, 'https://probml.github.io/pml-book/book1.html', 'Free, modern, comprehensive. The best single ML textbook currently available.'),
    book('Information Theory, Inference, and Learning Algorithms', 'David MacKay', 2003, 'https://www.inference.org.uk/mackay/itila/', 'Free, idiosyncratic, brilliant. The Bayesian perspective delivered with real force.'),
    video('Seeing Theory', 'Kunin et al.', 2018, 'https://seeing-theory.brown.edu/', 'Interactive visual probability. Excellent companion to this lesson.'),
    course('Statistics 110', 'Joe Blitzstein (Harvard)', 2013, 'https://projects.iq.harvard.edu/stat110', 'Free lectures. The best probability course on the internet.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'math-information',
  title: 'Entropy, Cross-Entropy, and KL Divergence',
  sub: 'Why your classification loss has that particular form.',
  mins: 22, level: 'foundations',
  prereq: ['math-probability'],
  tags: ['information theory', 'loss functions'],
  sections: [
    tldr(`If you have ever typed \`nn.CrossEntropyLoss()\` and wondered where that formula came from, this is
the lesson.

The chain is short. **Surprise** is $-\\log p$. **Entropy** is average surprise — the true cost of describing
data from a distribution. **Cross-entropy** is what it costs when your model is wrong about that distribution.
**KL divergence** is the excess: exactly how much extra you pay for being wrong. Minimising cross-entropy,
minimising KL, and maximising likelihood are all literally the same computation with three different names.`),

    jargon([
      ['bit / nat', 'A unit of information. Bits use $\\log_2$, nats use $\\log_e$. Code uses nats (natural log is faster and cleaner to differentiate); textbooks often use bits because they are easier to interpret. The only difference is a constant factor.'],
      ['surprise / surprisal', '$-\\log p(x)$. How startling it is that outcome $x$ happened. Certain events carry zero surprise; impossible ones carry infinite surprise.'],
      ['entropy $H(p)$', 'Average surprise under $p$. Also: the minimum average bits needed to encode data drawn from $p$. High entropy = unpredictable.'],
      ['cross-entropy $H(p, q)$', 'Average surprise when the truth is $p$ but your model says $q$. Always at least $H(p)$. **This is your classification loss.**'],
      ['KL divergence $D_{\\text{KL}}(p\\|q)$', 'The gap $H(p,q) - H(p)$ — the bits wasted by using the wrong model. Never negative, zero only when $p = q$. Not a distance: it is asymmetric.'],
      ['perplexity', '$e^{\\text{cross-entropy}}$. "How many options is the model effectively choosing between?" The standard language-model metric.'],
      ['mutual information $I(X;Y)$', 'How many bits knowing $Y$ saves you when describing $X$. Zero exactly when they are independent.'],
      ['logits', 'Raw model outputs before softmax. Can be any real number, positive or negative.'],
    ]),

    t(`## Surprise, quantified

Start with a question that sounds unanswerable: *how surprising is an event?*

It turns out you can pin the answer down by listing what any sensible measure of surprise must do:

1. **Decreasing in $p$.** Rarer events are more surprising.
2. **Zero when $p = 1$.** A certain event surprises no one.
3. **Additive for independent events.** Learning two unrelated facts should be as surprising as the sum of
   learning each separately.

Exactly one function satisfies all three:

$$I(x) = -\\log p(x)$$

Property 3 is what forces the logarithm specifically — logs are the only functions turning multiplication (of
independent probabilities) into addition. The minus sign is there because probabilities are below 1, so their
logs are negative, and we want surprise to be a positive quantity.

Sanity-check it: an event with $p = 1/2$ carries $-\\log_2(1/2) = 1$ bit. One coin flip, one bit. An event with
$p = 1/1024$ carries 10 bits — ten coin flips' worth of information. That matches intuition exactly.

**Entropy** is then just the average surprise you expect from a distribution:

$$H(p) = -\\sum_x p(x)\\log p(x) = \\mathbb{E}_{x\\sim p}[-\\log p(x)]$$

It is largest for the uniform distribution (nothing is predictable, every outcome equally startling) and zero
for a point mass (you always know what is coming). And Shannon's source coding theorem gives it a hard
operational meaning, not merely a suggestive one: **$H(p)$ is the minimum average number of bits needed to
encode samples from $p$.** No compression scheme can do better. That is a theorem, not a heuristic.`),

    t(`## Cross-entropy: the cost of coding with the wrong model

Now the case that matters for training. The truth is $p$, but your model believes $q$, and you built your code
around $q$. What do you pay?

You are still drawing events from $p$ — that is reality, and it does not care what you believe. But your cost
per event, $-\\log q(x)$, is set by your model. So your average bill is:

$$H(p,q) = -\\sum_x p(x)\\log q(x) \\qquad \\text{(weight by reality } p \\text{, pay by belief } q\\text{)}$$

That is **cross-entropy**, and it is precisely the loss function every classifier and language model minimises.

You are clearly paying more than you had to. The excess is the **KL divergence**:

$$D_{\\text{KL}}(p\\,\\|\\,q) = \\underbrace{H(p,q)}_{\\text{what you paid}} - \\underbrace{H(p)}_{\\text{the best possible}} = \\sum_x p(x)\\log\\frac{p(x)}{q(x)} \\;\\ge\\; 0$$

It is never negative — you can never beat the true entropy — and it is zero only when $q = p$ exactly. That
result is Gibbs' inequality.`),

    diagram('The three quantities, and how they stack',
`<svg viewBox="0 0 600 190" role="img" aria-label="Cross-entropy equals entropy plus KL divergence">
  <rect x="60" y="50" width="230" height="46" rx="4" style="fill: color-mix(in srgb, var(--s1) 20%, transparent); stroke: var(--s1); stroke-width: 1.6"/>
  <text class="dmono" x="175" y="74" text-anchor="middle" style="fill: var(--s1)">H(p)</text>
  <rect x="290" y="50" width="120" height="46" rx="4" style="fill: color-mix(in srgb, var(--s6) 20%, transparent); stroke: var(--s6); stroke-width: 1.6"/>
  <text class="dmono" x="350" y="74" text-anchor="middle" style="fill: var(--s6)">KL(p||q)</text>
  <line x1="60" y1="118" x2="410" y2="118" style="stroke: var(--s3); stroke-width: 1.6"/>
  <line x1="60" y1="112" x2="60" y2="124" style="stroke: var(--s3); stroke-width: 1.6"/>
  <line x1="410" y1="112" x2="410" y2="124" style="stroke: var(--s3); stroke-width: 1.6"/>
  <text class="dmono" x="235" y="138" text-anchor="middle" style="fill: var(--s3)">H(p, q) — what your loss reports</text>
  <text class="dlabel" x="175" y="36" text-anchor="middle">irreducible: the data's own randomness</text>
  <text class="dlabel" x="350" y="36" text-anchor="middle">your model's error</text>
  <text class="dlabel" x="60" y="168">Training can only shrink the right-hand block. The left one is a property of the world.</text>
</svg>`,
      `This picture explains why training loss plateaus above zero and *should*. $H(p)$ is the intrinsic
unpredictability of the data — the irreducible noise. A perfect model still pays it. If your language-model loss
bottoms out around 2.0 nats, that is not a failure to converge; a good chunk of it is English simply not being
deterministic.`),

    viz('entropy-kl'),

    key(`Training a classifier or a language model minimises cross-entropy $H(p, q_\\theta)$, where $p$ is the
empirical data distribution and $q_\\theta$ is your model.

Now look at the decomposition: $H(p, q_\\theta) = H(p) + D_{\\text{KL}}(p\\|q_\\theta)$. The first term has no
$\\theta$ in it — it is fixed by the data. So **minimising cross-entropy is identical to minimising
$D_{\\text{KL}}(p\\|q_\\theta)$**, which (expand the definitions) is identical to maximising likelihood.

Three names, one objective. When one paper says "we minimise the KL to the data distribution" and another says
"we maximise likelihood" and your code says \`cross_entropy\`, all three are running the same arithmetic. This is
worth knowing purely so that the vocabulary stops being intimidating.`),

    viz('entropy-kl'),

    key(`Training a classifier or a language model minimizes cross-entropy $H(p, q_\\theta)$ where $p$ is the empirical
data distribution. Since $H(p)$ does not depend on $\\theta$, **minimizing cross-entropy is identical to minimizing
$D_{\\text{KL}}(p\\|q_\\theta)$**, which is identical to maximum likelihood.

Three names, one objective. When a paper says "we minimize the KL to the data distribution" and another says "we
maximize likelihood," they are doing the same arithmetic.`),

    t(`## KL is not symmetric, and the asymmetry has consequences

$D_{\\text{KL}}(p\\|q) \\ne D_{\\text{KL}}(q\\|p)$. This is not a technicality to file away — swapping the
arguments produces qualitatively different models, and knowing which one an algorithm minimises tells you in
advance how it will fail.

Look at where each version blows up. In $\\sum p \\log(p/q)$, the terms are weighted by $p$, so only places
where $p$ has mass can contribute. If $q$ is near zero somewhere $p$ is not, that term goes to infinity.

**Forward KL, $D(p\\|q)$ — "mass covering."** Infinite penalty for putting near-zero probability where the data
has mass. So $q$ is forced to stretch over everything $p$ covers, even if that means smearing probability across
empty regions in between. **Maximum likelihood minimises this**, which is why fitting a single Gaussian to a
two-humped distribution lands it in the valley between the humps, covering both badly and the middle wrongly.

**Reverse KL, $D(q\\|p)$ — "mode seeking."** Now the weights are $q$'s, so the penalty is for $q$ putting mass
where $p$ has none. $q$ is free to ignore entire modes; it must only avoid hallucinating. **Variational
inference minimises this**, which is exactly why VI is notorious for underestimating posterior variance — and
why an RL policy trained with a KL penalty against a reference model collapses toward a narrow, safe subset of
its behaviour rather than covering it.`),

    diagram('The same bimodal target, fitted two ways',
`<svg viewBox="0 0 620 200" role="img" aria-label="Forward KL covers both modes badly, reverse KL picks one mode">
  <g transform="translate(15,0)">
    <path d="M20,150 C60,150 60,60 100,60 C140,60 130,150 155,150 C180,150 175,70 210,70 C245,70 250,150 280,150"
          style="fill:none; stroke: var(--text-faint); stroke-width: 2; stroke-dasharray: 4 3"/>
    <path d="M20,150 C80,150 90,88 150,88 C210,88 220,150 280,150" style="fill: color-mix(in srgb, var(--s1) 14%, transparent); stroke: var(--s1); stroke-width: 2.4"/>
    <text class="dtitle" x="20" y="178" style="fill: var(--s1)">forward KL — covers everything, fits nothing</text>
    <text class="dlabel" x="20" y="30">maximum likelihood does this</text>
  </g>
  <g transform="translate(330,0)">
    <path d="M20,150 C60,150 60,60 100,60 C140,60 130,150 155,150 C180,150 175,70 210,70 C245,70 250,150 280,150"
          style="fill:none; stroke: var(--text-faint); stroke-width: 2; stroke-dasharray: 4 3"/>
    <path d="M45,150 C75,150 75,62 100,62 C125,62 125,150 155,150" style="fill: color-mix(in srgb, var(--s4) 14%, transparent); stroke: var(--s4); stroke-width: 2.4"/>
    <text class="dtitle" x="20" y="178" style="fill: var(--s4)">reverse KL — picks one mode, ignores the rest</text>
    <text class="dlabel" x="20" y="30">variational inference, RL with a KL penalty</text>
  </g>
</svg>`,
      `Dashed grey is the truth in both panels. Neither fit is "wrong" — they optimise different things. When a
generative model produces bland averaged-out samples, suspect forward KL; when it produces confident but
low-diversity output, suspect reverse KL.`),

    t(`## Perplexity

Language modelling reports **perplexity** rather than raw loss. It is nothing new — just cross-entropy run back
through an exponential:

$$\\text{PPL} = \\exp\\!\\left(\\frac{1}{N}\\sum_{i=1}^{N} -\\log q(x_i\\mid x_{<i})\\right) = e^{\\text{cross-entropy}}$$

The exponential is there to make the number interpretable. Cross-entropy is measured in bits or nats, which
few people have intuition for. Perplexity converts it to **the effective number of equally likely options the
model is choosing between at each step**. Perplexity 10 means "about as uncertain as picking uniformly from 10
tokens". Perplexity 1 would be perfect prediction; perplexity equal to your vocabulary size means the model has
learned nothing.

One caution that bites people comparing models: **perplexity depends on the tokenizer.** A model with a bigger
vocabulary packs more text into each token, so its per-token perplexity is smaller without the model being any
better. Comparing perplexities across different tokenizers is meaningless. **Bits-per-byte** normalises by raw
text length instead and is the number to use for cross-model comparison.`),

    t(`## Mutual information

The last piece of vocabulary. **Mutual information** asks: how much does knowing one variable tell you about
another?

$$I(X;Y) = \\underbrace{H(X)}_{\\text{uncertainty about } X} - \\underbrace{H(X\\mid Y)}_{\\text{uncertainty left after seeing } Y}$$

Literally "bits of uncertainty removed". Equivalently it is a KL divergence — between the true joint
distribution and the fake one you would get if $X$ and $Y$ were independent:

$$I(X;Y) = D_{\\text{KL}}\\big(p(x,y)\\,\\|\\,p(x)p(y)\\big)$$

Read that as measuring *how far from independent they are*. It is zero exactly when knowing $Y$ tells you
nothing about $X$.

Where it shows up: feature selection (keep features with high MI to the label), the information-bottleneck
account of what representation learning does, and — most importantly for this atlas — as the quantity that
contrastive objectives like InfoNCE secretly lower-bound. When CLIP or SimCLR pull matching pairs together and
push non-matching pairs apart, the thing being maximised is a bound on mutual information.`),

    code('Entropy, cross-entropy, KL — and the stable way to compute them', `import numpy as np

def entropy(p):        return -np.sum(p * np.log2(p + 1e-12))
def cross_entropy(p,q): return -np.sum(p * np.log2(q + 1e-12))
def kl(p, q):          return np.sum(p * np.log2((p + 1e-12) / (q + 1e-12)))

p = np.array([0.5, 0.25, 0.15, 0.10])
q = np.array([0.25, 0.25, 0.25, 0.25])

print(f"H(p)      = {entropy(p):.4f} bits")
print(f"H(q)      = {entropy(q):.4f} bits   <- uniform is maximal")
print(f"H(p,q)    = {cross_entropy(p,q):.4f}")
print(f"KL(p||q)  = {kl(p,q):.4f}")
print(f"KL(q||p)  = {kl(q,p):.4f}   <- asymmetric!\\n")

# Why frameworks fuse softmax with cross-entropy
logits = np.array([50.0, 51.0, 49.0])          # large logits
naive_p = np.exp(logits) / np.exp(logits).sum()  # overflows in float32
print("naive softmax:", naive_p)

def log_softmax(z):
    z = z - z.max()                             # the stabilizing shift
    return z - np.log(np.exp(z).sum())

target = 1
print("stable NLL   :", -log_softmax(logits)[target])
print("perplexity   :", np.exp(-log_softmax(logits)[target]))`,
      'The `z - z.max()` shift is mathematically a no-op (softmax is shift-invariant) and numerically essential. This is why you should always use your framework\'s fused `cross_entropy(logits, target)` rather than `log(softmax(logits))`.'),

    quiz('A language model reports perplexity 8 on a test set. What does that mean?',
      ['On average it is as uncertain about the next token as if choosing uniformly among 8 options',
       'It gets 1 in 8 tokens correct',
       'Its cross-entropy is 8 bits per token',
       'It needs 8 tokens of context to make a prediction'],
      0,
      'Perplexity is $e^{H(p,q)}$ — the *effective branching factor*. Cross-entropy here is $\\log_2 8 = 3$ bits per token, not 8. And it is not an accuracy: a model can have low perplexity while rarely making the single most likely choice, because it spreads probability sensibly.'),

    recap(`- Derive $-\\log p$ as the only sensible measure of surprise, from three requirements.
- Explain entropy as both "average surprise" and "the minimum bits to encode this data", and say why the second
  is a theorem rather than an analogy.
- Write cross-entropy as $H(p) + D_{\\text{KL}}(p\\|q)$ and use it to explain why training loss plateaus above zero.
- State why minimising cross-entropy, minimising KL, and maximising likelihood are the same computation.
- Predict whether a method will smear across modes or collapse onto one, from which direction of KL it minimises.
- Interpret a perplexity number, and say why perplexities across different tokenizers cannot be compared.`),
  ],
  refs: [
    paper('A Mathematical Theory of Communication', 'Claude Shannon', 1948, 'https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf', 'The founding paper. Genuinely readable, and worth reading in the original.'),
    book('Information Theory, Inference, and Learning Algorithms', 'David MacKay', 2003, 'https://www.inference.org.uk/mackay/itila/', 'Chapters 1–6. Free PDF.'),
    blog('KL Divergence for Machine Learning', 'Will Kurt', 2017, 'https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained', 'Clear, concrete, worked example.'),
    paper('Representation Learning with Contrastive Predictive Coding', 'van den Oord et al.', 2018, 'https://arxiv.org/abs/1807.03748', 'InfoNCE — the mutual-information bound behind CLIP and SimCLR.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'math-optimization',
  title: 'Optimization: Gradient Descent and Its Descendants',
  sub: 'How every model in this atlas is actually trained.',
  mins: 32, level: 'foundations',
  prereq: ['math-derivatives'],
  tags: ['optimization', 'SGD', 'Adam'],
  sections: [
    tldr(`Training a model is a loop with three lines: compute the loss, compute the gradient, step downhill.
Everything difficult about it comes down to **how big should the step be?**

Too big and you overshoot and diverge. Too small and training takes a week. And there is no single right
answer, because the loss surface is steep in some directions and flat in others *at the same time*. Momentum,
Adam, learning-rate schedules, and normalization layers are all attacks on that one problem, and knowing which
problem they solve is more useful than memorising their update rules.`),

    jargon([
      ['$\\theta$ (theta)', 'The parameters — every trainable number in the model, stacked into one long vector. Training means searching over $\\theta$.'],
      ['loss $\\mathcal{L}(\\theta)$', 'A single number saying how badly the model is doing right now. Lower is better. Always a function of $\\theta$.'],
      ['$\\arg\\min_\\theta$', '"The value of $\\theta$ that makes what follows smallest." Note: the *argument*, not the minimum value itself.'],
      ['$\\eta$ (eta)', 'The learning rate — how far to move per step. Often written `lr` in code. The single most consequential hyperparameter.'],
      ['step / iteration', 'One update of the parameters, using one minibatch.'],
      ['epoch', 'One full pass over the training set. Many steps.'],
      ['minibatch $B$', 'The handful of examples (32, 256, …) used to estimate the gradient for one step, instead of the whole dataset.'],
      ['SGD', 'Stochastic Gradient Descent — gradient descent using minibatch estimates. "Stochastic" just means "with randomness in it".'],
      ['unbiased estimate', 'A noisy measurement that is correct *on average*. Minibatch gradients are noisy but unbiased, which is what makes SGD sound.'],
      ['momentum', 'Averaging the recent gradients instead of using only the latest one, so the update carries velocity like a rolling ball.'],
      ['convex', 'Bowl-shaped: any local minimum is *the* global minimum. Convex problems are solved; non-convex ones are gambled on.'],
      ['condition number', 'Ratio of the sharpest curvature to the flattest. Large = a long narrow valley = slow, awkward optimization.'],
    ]),

    t(`## The setup

Training is minimisation, and nothing more exotic. You have parameters $\\theta$, a loss $\\mathcal{L}(\\theta)$
averaged over your data, and you want the parameters that make it smallest:

$$\\theta^* = \\arg\\min_\\theta \\mathcal{L}(\\theta)$$

For a small handful of models — linear regression, some SVMs — you can solve that with algebra and be done. For
everything else in this atlas there is no closed form, so you iterate:

$$\\theta_{t+1} = \\theta_t - \\eta\\,\\nabla\\mathcal{L}(\\theta_t)$$

Read it left to right: *new parameters = old parameters, minus a small step in the direction of the gradient*.
The minus sign is the whole idea — the gradient points **uphill**, and we want to go down. $\\eta$ controls how
far. That is it. Every optimizer in this lesson is a variation on this one line.`),

    viz('gradient-descent'),

    t(`## Learning rate: the one hyperparameter that will ruin your day

How big can $\\eta$ be before things break? For once this has an exact answer, and the simplest possible case
gives it away.

Take a one-dimensional parabola, $\\mathcal{L}=\\tfrac12 a\\theta^2$, where $a$ is the curvature. Its gradient is
$a\\theta$, so a single step does:

$$\\theta_{t+1} = \\theta_t - \\eta\\, a\\theta_t = (1-\\eta a)\\,\\theta_t$$

Each step multiplies your distance-from-the-minimum by the fixed factor $(1-\\eta a)$. So the behaviour is
completely determined by that one number:

| $(1-\\eta a)$ | What happens |
|---|---|
| between 0 and 1 | shrinks smoothly toward the minimum |
| between −1 and 0 | overshoots past the minimum each step, but by less each time — converges while oscillating |
| exactly −1 | bounces between two points forever |
| below −1 | overshoots by *more* each time — **diverges**, exponentially |

Converging requires $|1-\\eta a| < 1$, which rearranges to the stability bound:

$$\\eta < \\frac{2}{a} = \\frac{2}{\\lambda_{\\max}(H)}$$

In words: **the maximum safe learning rate is set by the sharpest curvature in the landscape.** Exceed it and no
amount of patience will save the run — the divergence is geometric.`),

    viz('lr-stability'),

    key(`Here is why that bound is a genuine problem rather than a formula to plug into.

A real loss surface is not one parabola — it is steep along some directions and nearly flat along others, at the
same point. But you only get **one** $\\eta$, shared by every parameter. So the learning rate is capped by the
*sharpest* direction anywhere in the landscape, while progress along the *flattest* direction crawls at
$\\eta\\lambda_{\\min}$.

The damage is exactly the condition number: if $\\lambda_{\\max}/\\lambda_{\\min} = 1000$, you need roughly 1000×
more steps than you would in a nicely-rounded bowl. **One badly-behaved direction taxes every parameter in the
model.**

Every technique in the rest of this lesson attacks that:

- **Momentum / Adam** — give different directions different effective step sizes.
- **Normalization layers** — reshape the landscape so curvature is more uniform to begin with.
- **Warmup** — start small, because early curvature estimates are unreliable and the sharpest region is often
  encountered in the first few hundred steps.
- **Gradient clipping** — survive the occasional sharp cliff without a single step destroying the run.`),

    t(`## Stochastic gradient descent

The gradient in that update rule is defined as an average over your *entire* dataset. For a million examples,
one step would mean a million forward and backward passes. Absurd.

So estimate it from a small random sample instead — a **minibatch**:

$$\\nabla\\mathcal{L}(\\theta) \\approx \\frac{1}{B}\\sum_{i\\in\\text{batch}} \\nabla\\ell_i(\\theta)$$

This is legitimate because of linearity of expectation from the [probability lesson](#/l/math-probability): the
minibatch gradient is an **unbiased** estimate of the true one. It is wrong on any given step, but it is wrong
in a way that averages out, and its noise shrinks like $1/\\sqrt{B}$.

The trade is overwhelmingly favourable. With $B = 256$ out of $N = 10^6$ you get roughly 4000× more steps for
the same compute. Taking 4000 slightly-wrong steps beats taking one perfect one, essentially always.`),

    intuition(`The noise in SGD is not purely a cost you tolerate — it does useful work.

A perfectly computed gradient is exactly zero at a saddle point, so full-batch gradient descent parks there
indefinitely. Minibatch noise jostles you off. Likewise, noise makes it hard to settle into a *narrow* minimum
— you get shaken out — while a *wide* flat one holds you even with jitter. So SGD is quietly biased toward flat
minima, and flat minima empirically generalise better.

This is part of why "just use a huge batch size" underperforms: you compute a cleaner gradient and lose the
regularizing noise along with it. Very large batch training needs learning-rate scaling and warmup to
compensate.`),

    t(`## Momentum

Picture the narrow valley again: gradient descent bounces off the steep side walls while barely advancing along
the floor. Look at the sequence of gradients — the across-the-valley component keeps **flipping sign**, while
the along-the-floor component points the **same way** every time.

That suggests an obvious fix: average the recent gradients instead of using only the latest. Flip-flopping
components cancel; consistent ones survive and add up. That is momentum:

$$v_{t+1} = \\beta v_t - \\eta\\nabla\\mathcal{L}(\\theta_t), \\qquad \\theta_{t+1} = \\theta_t + v_{t+1}$$

$v$ is a velocity that decays by a factor $\\beta$ each step while accumulating new gradient. The physical
analogy is exact: a ball with inertia rolling down the surface, rather than a hiker who re-reads the map and
starts fresh at every step.

With the standard $\\beta=0.9$, a consistently-pointing direction gets amplified by $\\frac{1}{1-\\beta} = 10\\times$
compared to plain SGD, while the oscillating directions are damped. You get a bigger effective step exactly where
a bigger step is safe.

**Nesterov momentum** is a small refinement: evaluate the gradient at the *look-ahead* position $\\theta_t +
\\beta v_t$ — where momentum is about to carry you — rather than where you currently are. It notices an upcoming
overshoot one step earlier. Marginally but consistently better.`),

    t(`## Adaptive methods: RMSProp and Adam

Momentum helps, but it still applies one global learning rate. The next idea is to give **every parameter its
own step size**, inferred from its own gradient history.

The motivation is concrete: a weight attached to a feature that appears in 1% of examples receives a gradient
only 1% of the time. It should take bigger steps when it finally gets one. A weight in a constantly-active path
gets a gradient every step and needs smaller ones. A single $\\eta$ cannot serve both.

The heuristic that works: **divide each parameter's step by the typical magnitude of its recent gradients.**
Large historical gradients ⇒ damp it down. Small ones ⇒ let it move. That is RMSProp.

**Adam** is RMSProp plus momentum — it tracks two running averages, one of the gradient (first moment, the
direction) and one of the squared gradient (second moment, the scale):

$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1)g_t \\qquad v_t = \\beta_2 v_{t-1}+(1-\\beta_2)g_t^2$$
$$\\hat m_t = \\frac{m_t}{1-\\beta_1^t}, \\quad \\hat v_t = \\frac{v_t}{1-\\beta_2^t}, \\qquad
\\theta_{t+1} = \\theta_t - \\eta\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon}$$

Line by line: $m_t$ is a running average of the gradient (momentum). $v_t$ is a running average of the gradient
*squared*, which measures magnitude regardless of sign. The hatted versions correct a startup bias — $m$ and $v$
begin at zero, so early on they under-report, and dividing by $1-\\beta^t$ compensates until $t$ grows. The final
line is the update: step in the momentum direction, scaled down by the recent gradient magnitude.

The $\\epsilon$ in the denominator only exists to stop division by zero for parameters that have received no
gradient yet.

Defaults $\\beta_1=0.9,\\ \\beta_2=0.999,\\ \\epsilon=10^{-8}$ work remarkably often, which is most of why Adam won:
it is far less sensitive to getting $\\eta$ exactly right than plain SGD, and that saves an enormous amount of
tuning time.`),

    warn(`**Adam has a memory cost.** It stores two extra float32 values per parameter, so optimizer state is
$8$ bytes/param on top of weights and gradients — often the largest single term in your training memory budget. This is
what 8-bit optimizers, Adafactor, and Lion are trying to reduce.

**AdamW ≠ Adam + L2.** Adding $\\lambda\\|\\theta\\|^2$ to the loss puts the penalty through Adam's adaptive scaling, so
parameters with large historical gradients get *less* decay — usually not what you want. AdamW applies
$\\theta \\mathrel{-}= \\eta\\lambda\\theta$ separately, decoupled from the adaptive term. **Use AdamW.** It is the default
for essentially all transformer training.`),

    viz('lr-schedules'),

    t(`## Convexity, and how much we should care

A **convex** function is bowl-shaped: pick any two points on it, draw a straight line between them, and the
function stays below that line. The consequence that matters is a guarantee — *any local minimum is the global
minimum*. There is nowhere to get stuck. Run any reasonable solver, get the right answer, go home.

Linear regression, logistic regression, SVMs, and Lasso are all convex. This is why those methods felt solid
and theoretically respectable, and why 1990s ML theory was largely built around them.`),

    viz('convexity'),

    t(`Neural networks are not convex. Not slightly non-convex — wildly so, with an astronomical number of
critical points and no guarantee whatsoever that gradient descent finds anything good. On the theory of the day,
they should not work.

They work anyway, and the reasons are partly understood:

- **Most critical points are saddles, not minima.** As the [derivatives lesson](#/l/math-derivatives) argued,
  needing all million eigenvalues to be positive is an absurdly strong coincidence. Gradient noise slides off
  saddles.
- **Overparameterised networks have vast connected basins of near-optimal loss.** You do not need to find *the*
  minimum; a huge region is good enough, and many of them turn out to be connected to each other.
- **SGD is biased toward flat minima**, and flatness correlates with generalisation.`),

    warn(`Take that list as *current best understanding*, not settled fact. Why non-convex optimization works so
reliably for deep networks — and why the solutions it finds generalise rather than memorise — is genuinely
open. Statements you will see asserted confidently in blog posts ("SGD finds flat minima which generalise
better") have real empirical support and known counterexamples both.

Being aware of the gap is useful in practice: it is why deep learning results are established by running the
experiment rather than by deriving them, and why a technique that works for one architecture and dataset may
simply not transfer.`),

    code('Optimizers from scratch', `import numpy as np

# ill-conditioned quadratic: the classic pathological case
A = np.array([0.15, 4.0])          # curvature per axis; condition number ~27
def loss(w): return 0.5 * np.sum(A * w**2)
def grad(w): return A * w

def run(opt, lr, steps=120, **kw):
    w = np.array([-2.5, 2.0])
    state = {"m": np.zeros(2), "v": np.zeros(2), "t": 0}
    for _ in range(steps):
        g = grad(w)
        if opt == "sgd":
            w -= lr * g
        elif opt == "momentum":
            state["m"] = kw["beta"]*state["m"] - lr*g
            w += state["m"]
        elif opt == "adam":
            state["t"] += 1
            b1, b2, eps = 0.9, 0.999, 1e-8
            state["m"] = b1*state["m"] + (1-b1)*g
            state["v"] = b2*state["v"] + (1-b2)*g**2
            mh = state["m"]/(1-b1**state["t"])
            vh = state["v"]/(1-b2**state["t"])
            w -= lr * mh/(np.sqrt(vh)+eps)
        if not np.all(np.isfinite(w)): return np.inf
    return loss(w)

print(f"stable lr limit for plain GD: {2/A.max():.4f}\\n")
for name, lr, kw in [("sgd", 0.4, {}), ("sgd", 0.49, {}), ("sgd", 0.51, {}),
                     ("momentum", 0.1, {"beta": 0.9}), ("adam", 0.1, {})]:
    print(f"{name:9s} lr={lr:<5} final loss = {run(name, lr, **kw):.3e}")`,
      'Note how plain GD is unstable above $2/\\lambda_{\\max}=0.5$, while Adam is nearly insensitive to the conditioning — it rescales each axis by its own gradient history.'),

    quiz('Your loss suddenly spikes to NaN at step 4000 of a transformer run. What is the most productive first check?',
      ['Gradient norm history — look for a spike, then add gradient clipping and possibly more warmup',
       'Reduce the batch size',
       'Switch from AdamW to SGD',
       'Add more layers'],
      0,
      'NaN almost always follows a gradient explosion or an overflow (especially in fp16). Log the gradient norm every step: you will usually see it climb for several steps before the blowup. Standard fixes, in order: clip the global grad norm to ~1.0, lengthen warmup, lower the peak LR, and check for fp16 overflow in attention logits (bf16 avoids most of this).'),

    recap(`- Write the gradient descent update from memory and explain every symbol in it, including the minus sign.
- Derive the stability bound $\\eta < 2/\\lambda_{\\max}$ from a one-dimensional parabola.
- Explain why one sharp direction throttles the learning rate for the entire model.
- Justify minibatching with "unbiased estimate", and name a benefit of the noise beyond saving compute.
- Say what momentum does to oscillating vs. consistent gradient directions.
- Read Adam's update and identify which term is momentum, which is per-parameter scaling, and what the bias
  correction is for.
- State the difference between AdamW and "Adam plus L2", and which to use.
- Say honestly what is and is not understood about why non-convex training works.`),
  ],
  refs: [
    paper('Adam: A Method for Stochastic Optimization', 'Kingma & Ba', 2014, 'https://arxiv.org/abs/1412.6980', 'The most-cited optimizer paper in ML.'),
    paper('Decoupled Weight Decay Regularization', 'Loshchilov & Hutter', 2017, 'https://arxiv.org/abs/1711.05101', 'AdamW. Explains precisely why L2-in-the-loss and weight decay differ under adaptive methods.'),
    blog('Why Momentum Really Works', 'Gabriel Goh', 2017, 'https://distill.pub/2017/momentum/', 'Interactive and genuinely illuminating. Read this one.'),
    paper('On the Convergence of Adam and Beyond', 'Reddi et al.', 2018, 'https://arxiv.org/abs/1904.09237', 'AMSGrad — where Adam\'s convergence proof breaks and how to patch it.'),
    book('Convex Optimization', 'Boyd & Vandenberghe', 2004, 'https://web.stanford.edu/~boyd/cvxbook/', 'Free PDF. The reference for the convex case.'),
    paper('An overview of gradient descent optimization algorithms', 'Sebastian Ruder', 2016, 'https://arxiv.org/abs/1609.04747', 'A clean survey of everything in this lesson.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'math-numerics',
  title: 'Numerical Precision and Conditioning',
  sub: 'Where the abstraction leaks: floats, overflow, and why bf16 won.',
  mins: 20, level: 'foundations',
  tags: ['numerics', 'systems'],
  sections: [
    tldr(`Every number in a model is stored in a fixed number of bits, and that storage format leaks into your
results. Most of the time you can ignore it. When you cannot, the symptoms are dramatic: \`NaN\` losses,
negative variances, training that diverges only on some hardware.

Two ideas cover almost all of it. **Precision** — how many digits a format keeps — and **range** — the biggest
and smallest magnitudes it can express. bf16 beat fp16 for training because it traded away precision to keep
range, and range is what attention logits need.`),

    jargon([
      ['floating point', 'How computers store non-integer numbers: a few bits for the exponent (the scale) and the rest for the mantissa (the digits). Scientific notation in binary.'],
      ['mantissa', 'The significant digits. More mantissa bits = finer distinctions between nearby numbers = more **precision**.'],
      ['exponent', 'The scale. More exponent bits = a wider span between the largest and smallest representable magnitudes = more **range**.'],
      ['fp32 / fp16 / bf16', '32-bit float, 16-bit float, and "brain float" 16. The number is the total bit count; the difference between fp16 and bf16 is how those 16 bits are split.'],
      ['overflow / underflow', 'A value too large for the format (becomes `inf`) or too small (becomes `0`). Both silently destroy your computation.'],
      ['mixed precision', 'Doing most arithmetic in 16 bits for speed and memory, but keeping a 32-bit copy of the weights so tiny updates are not lost.'],
      ['loss scaling', 'Multiplying the loss by a large constant before the backward pass so small gradients do not underflow to zero in fp16. bf16 does not need this.'],
      ['catastrophic cancellation', 'Subtracting two nearly-equal large numbers. The leading digits cancel and you are left with mostly rounding error.'],
      ['conditioning', 'How much a small change to a problem\'s input can change its answer. Badly conditioned problems amplify floating-point error.'],
      ['quantization', 'Deliberately storing weights in very few bits (8 or 4) to shrink a model for inference.'],
    ]),

    t(`## Floats are not real numbers

The mathematics in this track assumed real numbers: infinitely many, infinitely precise. Your hardware has 32
bits. Something has to give.

A floating-point number is stored as $\\pm m \\times 2^{e}$ — a sign bit, some **exponent** bits, and some
**mantissa** bits. It is scientific notation in binary, and the split between those two groups is the entire
design space:

- More **exponent** bits ⇒ wider **range**. You can represent $10^{38}$ and $10^{-38}$ without overflowing.
- More **mantissa** bits ⇒ finer **precision**. You can tell $1.0000001$ from $1.0000002$.

With a fixed total bit budget, you buy one with the other. Every format in the table below is a different answer
to that trade, and the reason bf16 won for training is that it made the right call about which one matters.`),

    viz('float-precision'),

    t(`| Format | Bits | Exponent | Mantissa | Max value | Where used |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | $3.4\\times10^{38}$ | master weights, optimizer state |
| tf32 | 19* | 8 | 10 | $3.4\\times10^{38}$ | NVIDIA tensor cores, transparent |
| bf16 | 16 | 8 | 7 | $3.4\\times10^{38}$ | **the default for training** |
| fp16 | 16 | 5 | 10 | $65{,}504$ | older hardware, needs loss scaling |
| fp8 | 8 | 4 or 5 | 3 or 2 | $448$ | H100+ training, inference |
| int8 / int4 | 8 / 4 | — | — | — | quantized inference |

Read the exponent column and the story tells itself. **bf16 is simply fp32 with the mantissa chopped off** — it
keeps all 8 exponent bits, so exactly the same *range* of magnitudes is representable. You lose precision, not
reach. Casting fp32 → bf16 can never overflow.

fp16 made the opposite bet: 5 exponent bits, 10 mantissa bits. More precision, far less range — its largest
value is 65,504. That sounds like plenty until you remember attention logits are dot products over thousands of
dimensions, which routinely exceed it. Training in fp16 therefore requires **loss scaling** (multiply the loss
up before the backward pass, divide out after) plus overflow detection and step-skipping, all of which is
machinery you simply do not need with bf16.

That is the whole reason the industry switched. Not that bf16 is more accurate — it is *less* accurate — but
that neural network training turns out to be robust to low precision and extremely fragile to overflow.`),

    t(`## Mixed precision, in practice

You do not pick one format — you use several at once, each where it is strongest. The standard recipe:`),

    steps('The mixed-precision training loop', [
      { h: 'Keep master weights in fp32', md: `A single optimizer step changes a weight by something like $10^{-7}$ of its magnitude. In bf16, with 7 mantissa bits, that update rounds to *exactly nothing* and the weight never moves. The fp32 master copy is what makes tiny accumulated updates possible at all.` },
      { h: 'Cast to bf16 for the forward and backward passes', md: `Half the bytes means half the memory traffic, and matrix-multiply units run several times faster on 16-bit inputs. This is where nearly all the speedup comes from.` },
      { h: 'Accumulate matmul results in fp32', md: `Summing thousands of products in 16 bits would lose precision badly. The hardware multiplies in bf16 but accumulates in fp32 internally — you get the speed without the error. This happens automatically inside tensor cores.` },
      { h: 'Apply the optimizer update in fp32', md: `Back on the master copy, at full precision. Then cast down again for the next forward pass.` },
      { h: 'Keep the precision-sensitive operations in fp32 regardless', md: `Softmax denominators, layer-norm statistics, and the loss itself all involve sums or divisions where small errors compound. Frameworks maintain an allowlist of ops that stay in fp32 no matter what the surrounding cast says.` },
    ]),

    warn(`**Catastrophic cancellation.** Subtracting nearly-equal numbers destroys significant digits. The classic
example is computing variance as $\\mathbb{E}[X^2]-\\mathbb{E}[X]^2$: if the mean is large relative to the spread, both
terms are huge and nearly equal, and you can get a *negative* variance. Use a stable two-pass or Welford algorithm.

The same issue is why softmax subtracts the max before exponentiating, and why you should never write
\`log(softmax(x))\` instead of \`log_softmax(x)\`.`),

    t(`## Conditioning: when the problem itself amplifies error

Floating-point error is small. Whether it *stays* small depends on the problem, and the condition number is
what tells you.

The **condition number** measures how much the answer can move when the input is nudged. For solving
$A\\mathbf{x}=\\mathbf{b}$ it is the same ratio from the [SVD lesson](#/l/math-eigen-svd),
$\\kappa(A)=\\sigma_{\\max}/\\sigma_{\\min}$, and it comes with a blunt rule of thumb:

> **You lose about $\\log_{10}\\kappa$ digits of accuracy.**

float32 carries roughly 7 decimal digits. So a problem with $\\kappa=10^8$ consumes all of them and your answer
is pure noise — not "slightly off", but meaningless. And you get no warning: the solver returns
confident-looking numbers either way.

This has one very concrete consequence worth memorising. You should never solve the normal
equations $X^{\\mathsf T}X\\mathbf{w}=X^{\\mathsf T}\\mathbf{y}$ by forming $X^{\\mathsf T}X$ — that **squares** the condition
number. Use a QR or SVD-based least-squares solver (\`np.linalg.lstsq\`) instead.`),

    code('Where floating point bites', `import numpy as np

# 1. The classic
print("0.1 + 0.2 == 0.3 ?", 0.1 + 0.2 == 0.3, " diff =", 0.1+0.2-0.3)

# 2. Catastrophic cancellation in a variance formula
x = np.array([1e8, 1e8 + 1, 1e8 + 2], dtype=np.float32)
naive = (x**2).mean() - x.mean()**2
print("\\nnaive variance :", naive, " <- can even go negative")
print("two-pass       :", ((x - x.mean())**2).mean())
print("numpy          :", x.var())

# 3. fp16 overflow in attention logits
logits = np.array([100.0, 200.0, 300.0], dtype=np.float32)
print("\\nfp16 max is 65504")
print("exp(300) in fp32:", np.exp(logits[2]))          # inf even in fp32
def softmax_stable(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
print("stable softmax  :", softmax_stable(logits))

# 4. Conditioning: never form X^T X
rng = np.random.default_rng(0)
n, d = 100, 8
X = rng.normal(size=(n, d))
X[:, 1] = X[:, 0] + 1e-6 * rng.normal(size=n)          # near-duplicate column
y = rng.normal(size=n)
print("\\ncond(X)    =", f"{np.linalg.cond(X):.2e}")
print("cond(X^T X) =", f"{np.linalg.cond(X.T @ X):.2e}", " <- squared!")
w_normal = np.linalg.solve(X.T @ X, X.T @ y)
w_lstsq  = np.linalg.lstsq(X, y, rcond=None)[0]
print("difference between the two solutions:",
      np.abs(w_normal - w_lstsq).max())`),

    quiz('Why did bf16 replace fp16 as the default for large model training?',
      ['It keeps all 8 exponent bits of fp32, so it has the same dynamic range and needs no loss scaling',
       'It has more mantissa bits, so it is more precise',
       'It is faster on GPUs',
       'It uses less memory than fp16'],
      0,
      'bf16 and fp16 are both 16 bits and use identical memory. bf16 trades mantissa bits for exponent bits: **less precise, but the same range as fp32**. Since training blows up from overflow far more often than from rounding, that is the right trade — and it removes the loss-scaling machinery fp16 requires.'),

    recap(`- Explain the exponent/mantissa trade in one sentence, and map it onto range vs. precision.
- Say why bf16 replaced fp16 for training, and why "it is more accurate" is the wrong answer.
- Describe the mixed-precision recipe and, in particular, why fp32 master weights are not optional.
- Recognise catastrophic cancellation, and name two places it is deliberately engineered around
  (\\\`log_softmax\\\`, the softmax max-subtraction).
- Use $\\log_{10}\\kappa$ to predict how many digits a computation will cost you.
- Explain why forming $X^{\\mathsf T}X$ is a bad idea, and what to call instead.`),
  ],
  refs: [
    paper('What Every Computer Scientist Should Know About Floating-Point Arithmetic', 'David Goldberg', 1991, 'https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html', 'The definitive reference. Long, but the first few sections are essential.'),
    paper('Mixed Precision Training', 'Micikevicius et al.', 2017, 'https://arxiv.org/abs/1710.03740', 'The original recipe: master weights, loss scaling, and which ops must stay fp32.'),
    paper('FP8 Formats for Deep Learning', 'Micikevicius et al.', 2022, 'https://arxiv.org/abs/2209.05433', 'E4M3 and E5M2, and where each is appropriate.'),
    book('Accuracy and Stability of Numerical Algorithms', 'Nicholas Higham', 2002, 'https://epubs.siam.org/doi/book/10.1137/1.9780898718027', 'The serious treatment of conditioning and stability.'),
  ],
},

];

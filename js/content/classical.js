/* ============================================================
   Track 2 — Classical machine learning
   ============================================================ */

import { t, key, intuition, warn, hist, mathnote, viz, deriv, code, quiz, paper, book, course, blog, video, demo, codeRef,
         tldr, recap, jargon, steps, diagram } from './_helpers.js';

export default [

/* ---------------------------------------------------------- */
{
  id: 'ml-framing',
  title: 'What Learning Means',
  sub: 'Risk, empirical risk, and the assumption that makes any of this possible.',
  mins: 20, level: 'core',
  tags: ['theory', 'generalization'],
  sections: [
    tldr(`"Learning" has a precise meaning here, and it is not "getting good scores on your data". It is doing
well on data you have **not seen yet**.

That gap is the whole subject. You can only measure performance on the examples you hold, but you only care
about performance on the ones you don't. Every piece of machinery in classical ML — validation splits,
regularization, cross-validation, the strict rules about the test set — exists to stop you from fooling yourself
about the difference.

The lesson also answers a question you may already have: given that many different models could fit the same
data, what makes one of them the right choice? The answer is that every model comes with built-in assumptions
about what the world looks like, and the good ones are the ones whose assumptions happen to be true.`),

    jargon([
      ['distribution $\\mathcal{D}$', 'The imaginary infinite pool of all possible examples your data was drawn from. You never see it; you only ever hold a finite sample.'],
      ['$(\\mathbf{x}, y)$', 'One labelled example: the input features $\\mathbf{x}$, and the correct answer $y$.'],
      ['loss $\\ell$', 'A number saying how wrong a single prediction was — squared error, cross-entropy, and so on.'],
      ['risk', 'Confusingly, this just means "average loss". **True risk** is the average over the whole distribution, which is what you want. **Empirical risk** is the average over your dataset, which is what you can measure.'],
      ['ERM', 'Empirical Risk Minimization: the strategy of minimising the average loss on the data you have and hoping it transfers. Almost all supervised learning is this plus a safeguard.'],
      ['generalization', 'How well a model does on data it has not seen. Measured as the gap between empirical and true risk.'],
      ['i.i.d.', 'Independent and identically distributed — the assumption that your test data looks like your training data and that examples do not influence one another.'],
      ['capacity', 'How complicated a function a model is able to represent. High capacity can fit anything, including pure noise.'],
      ['inductive bias', 'The assumptions a model makes before it sees any data at all. A model that assumes nearby pixels belong together has an inductive bias, and that bias is why it needs less data than one that assumes nothing.'],
      ['hyperparameter', 'A setting you choose rather than learn — learning rate, tree depth, number of layers. Contrast with parameters, which are fit from the data.'],
    ]),

    t(`## The setup, stated carefully

The whole problem is four objects.

There is a **distribution** $\\mathcal{D}$ over input–output pairs $(\\mathbf{x}, y)$. Think of it as an infinite
pool containing every photo that could ever be taken, each paired with its correct label. You never see
$\\mathcal{D}$; you see $n$ examples drawn from it. What you want is a function $f$ that predicts $y$ from
$\\mathbf{x}$ well on **future** draws from that same pool.

Pick a **loss** $\\ell(f(\\mathbf{x}), y)$ that scores how wrong one prediction is. What you actually care about
is the average loss over the entire pool — the **true risk**:

$$R(f) = \\mathbb{E}_{(\\mathbf{x},y)\\sim\\mathcal{D}}\\big[\\ell(f(\\mathbf{x}), y)\\big]$$

And here is the problem in one line: **you cannot compute that.** It is an average over a distribution you have
no access to. The best available substitute is the average over the examples you happen to hold — the
**empirical risk**:

$$\\hat R(f) = \\frac{1}{n}\\sum_{i=1}^{n} \\ell(f(\\mathbf{x}_i), y_i)$$

Minimising that is called **empirical risk minimization**, and essentially all supervised learning is ERM plus
some scheme for stopping the model from taking the word "minimise" too literally.`),

    diagram('The one gap that everything else is about',
`<svg viewBox="0 0 620 210" role="img" aria-label="Empirical risk measured on a sample versus true risk on the full distribution">
  <ellipse cx="150" cy="105" rx="128" ry="80" style="fill: color-mix(in srgb, var(--s1) 8%, transparent); stroke: var(--s1); stroke-width: 1.6; stroke-dasharray: 5 4"/>
  <text class="dtitle" x="150" y="42" text-anchor="middle" style="fill: var(--s1)">the distribution D</text>
  <text class="dlabel" x="150" y="60" text-anchor="middle">everything that could happen</text>
  <g style="fill: var(--s1); opacity: .35">
    <circle cx="90" cy="95" r="3"/><circle cx="200" cy="80" r="3"/><circle cx="118" cy="145" r="3"/>
    <circle cx="215" cy="140" r="3"/><circle cx="72" cy="130" r="3"/><circle cx="175" cy="120" r="3"/>
    <circle cx="140" cy="72" r="3"/><circle cx="235" cy="105" r="3"/><circle cx="60" cy="100" r="3"/>
  </g>
  <rect x="100" y="88" width="88" height="56" rx="4" style="fill: color-mix(in srgb, var(--s2) 16%, transparent); stroke: var(--s2); stroke-width: 1.8"/>
  <g style="fill: var(--s2)"><circle cx="118" cy="102" r="3.4"/><circle cx="150" cy="118" r="3.4"/><circle cx="172" cy="100" r="3.4"/><circle cx="132" cy="132" r="3.4"/></g>
  <text class="dtitle" x="144" y="170" text-anchor="middle" style="fill: var(--s2)">your dataset</text>
  <line x1="285" y1="105" x2="330" y2="105" style="stroke: var(--border); stroke-width: 1.5"/>
  <text class="dmono" x="350" y="82" style="fill: var(--s2)">R-hat  = what you measure</text>
  <text class="dmono" x="350" y="106" style="fill: var(--s1)">R      = what you care about</text>
  <text class="dtitle" x="350" y="140" style="fill: var(--s6)">the difference = generalization gap</text>
  <text class="dlabel" x="350" y="160">shrinks with more data,</text>
  <text class="dlabel" x="350" y="177">grows with more model capacity</text>
</svg>`,
      `Keep this picture in mind whenever you read a benchmark number. Every reported score is the small orange
box; the claim being made is about the whole blue ellipse. The two agree only when the sample is representative
and nobody tuned their way into fitting it.`),

    t(`Why should the empirical risk resemble the true risk at all? Because an average over a random sample
estimates an average over the population — the same fact behind every opinion poll. With $n$ independent
examples, the estimate wobbles around the truth with a spread of about $1/\\sqrt{n}$.

But that argument only holds for a function $f$ chosen **before** you looked at the data. The moment you pick
$f$ *because* it scores well on those particular points, the estimate stops being fair — you have selected for
whichever function happened to match this sample's noise. That is the entire reason training error is
optimistic, and it is worth carrying as a sentence: **you cannot honestly evaluate on data you used to
choose.**`),

    key(`The whole discipline hinges on one question: **when does a small $\\hat R$ imply a small $R$?**

Two assumptions do all the work, and both fail in the real world more often than people admit:

1. **i.i.d.** — training and test data come from the same distribution, drawn independently. Broken by
   distribution shift, by data that drifts over time, by selection bias in how the data was collected, and by
   test examples that leaked into training.
2. **Limited capacity** — the model cannot fit arbitrary noise, so fitting the training set is evidence that it
   found real structure. Broken spectacularly by modern large networks, which *can* memorise entirely random
   labels and yet still generalise on real ones. Why that happens is an open problem.`),

    t(`## The taxonomy, briefly

- **Supervised** — you have labelled pairs. Classification when $y$ is a category, regression when it is a
  number.
- **Unsupervised** — no labels. Clustering, density estimation, dimensionality reduction.
- **Self-supervised** — the labels are manufactured from the data itself: hide the next word and predict it,
  mask part of an image and reconstruct it. This is how essentially every frontier model is pretrained, and it
  is why scale became possible — the internet is an enormous free labelled dataset if the labels come from the
  data.
- **Reinforcement** — learning from rewards rather than from correct answers.`),

    t(`## No free lunch, and why it does not paralyse us

There is a theorem that sounds like it should end the field. The **No Free Lunch theorem** says that averaged
over *all possible* target functions, every learning algorithm performs identically — random guessing included.

The escape is hiding in the premise. "All possible functions" is dominated by functions that map inputs to
outputs with no structure at all: no smoothness, no pattern, nothing to learn from having seen a nearby example.
Almost none of them is anything you will ever be asked to predict. Real data is smooth, local, compositional,
and repetitive.

So an algorithm wins not by being universally good, but by **encoding assumptions that happen to match the
structure of your data**. Those assumptions are its **inductive bias**, and choosing them well is the actual
craft of the field:

| Model | What it assumes before seeing any data |
|---|---|
| Linear regression | The output changes at a constant rate with each input |
| k-nearest-neighbours | Points that are close together have similar labels |
| Decision tree | The answer is a series of threshold questions on individual features |
| CNN | Useful features are local, and a pattern means the same thing anywhere in the image |
| RNN | The data is a sequence, and the recent past matters most |
| Transformer | Any position may depend on any other, and which ones should be learned |`),

    intuition(`A useful way to hold this: **more inductive bias means less data needed, but a lower ceiling if the
bias is wrong.**

A CNN's assumption that a cat is a cat wherever it appears in the frame is free, correct, and worth an enormous
amount of training data. A transformer assumes almost nothing of the kind, so on a small image dataset it loses
badly. Give it enough data and it wins, because it can *learn* a better assumption than the one we hand-designed.

Hand-coded structure versus learned structure is the recurring argument of the last decade, and scale has
consistently favoured the second.`),

    t(`## Train, validation, test

Three splits, three distinct jobs. Conflating any two of them will quietly ruin your results.

- **Train** — fit the parameters. The model sees these directly.
- **Validation** — choose hyperparameters, pick the architecture, decide when to stop. The model does not train
  on these, but *you* do: every decision you make by looking at validation performance is a form of fitting.
- **Test** — estimate the true risk. **Look once, at the very end.**

The third rule is stricter than it feels like it needs to be, and "I never trained on the test set" is not
enough.`),

    warn(`**Selecting is training.** Every time you look at a test score and change something — a hyperparameter,
an architecture, a preprocessing step — you have used the test set to make a decision. That is optimization,
performed by you, by hand.

Here is why it biases the number. Each model's test score is its true accuracy plus some noise from the finite
test set. Pick the best of twenty models by test score and you have systematically favoured the ones whose noise
happened to be positive. With a 1000-example test set the noise on each score is around 1.5 percentage points,
and the maximum of twenty such draws sits roughly 3 points above the average — so a model that is genuinely no
better than its rivals reports a 3-point lead. That lead evaporates on any fresh data.

This is the most common methodological failure in applied ML, it is entirely self-inflicted, and it is why
published results so often fail to reproduce. The discipline is: hold out a test set, forget it exists, iterate
on validation, and look exactly once.`),

    viz('cross-validation'),

    code('Empirical risk, and how optimistic it is', `import numpy as np
rng = np.random.default_rng(0)

def true_f(x): return 1.5 * x - 0.4          # the truth, which we never get to see
NOISE = 0.5

def draw(n):
    x = rng.uniform(-3, 3, size=n)
    return x, true_f(x) + rng.normal(0, NOISE, size=n)

def risk(w, b, x, y):
    return np.mean((w*x + b - y) ** 2)

# one dataset, one ERM fit
x, y = draw(40)
w_hat, b_hat = np.polyfit(x, y, 1)           # the line minimising empirical risk
print(f"ERM picked: w={w_hat:.3f}  b={b_hat:.3f}   (truth: 1.500, -0.400)\\n")

# the same experiment repeated, so we can see the systematic part
for n in [8, 20, 100, 1000]:
    gaps = []
    for _ in range(400):
        x, y = draw(n)
        w_, b_ = np.polyfit(x, y, 1)
        xte, yte = draw(20_000)              # a fresh sample stands in for the true risk
        gaps.append(risk(w_, b_, xte, yte) - risk(w_, b_, x, y))
    print(f"n={n:5d}   average (true - empirical) risk = {np.mean(gaps):+.4f}")
print(f"\\nirreducible noise floor: {NOISE**2:.4f}")`,
      'On any single dataset the gap is noisy and can land either way, which is why the loop repeats it 400 times. The average is what matters: it is positive at every $n$, so the training score is systematically optimistic, and it shrinks as $n$ grows. For a least-squares fit the size of that optimism is known exactly — it is $2\\sigma^2 p/n$, with $p$ the number of fitted parameters — which here is $2(0.25)(2)/n = 1/n$. Check the printed numbers against that. Note what the formula says: the gap grows with the number of parameters and shrinks with the amount of data, which is the whole of the next lesson in one fraction.'),

    quiz('You try 200 hyperparameter configurations, pick the one with the best test accuracy, and report that number. What is wrong?',
      ['Selecting on the test set makes the number optimistically biased — it has become a validation score',
       'Nothing, as long as you never trained on the test set',
       '200 configurations is too few to be meaningful',
       'You should have used a larger test set'],
      0,
      'You reported the maximum of 200 noisy estimates, and the maximum of noisy estimates sits above the average of what they estimate. So the number is biased upward even though no gradient ever touched the test set. The fix is to select on a validation set and then evaluate the single chosen model on the test set once — and to expect that number to be meaningfully lower than the one you were admiring.'),

    recap(`- Define true risk and empirical risk, and say precisely why one is computable and the other is not.
- Explain why an average over a sample estimates an average over a population, and why that argument fails for a
  model you chose using that sample.
- Name the two assumptions that make ERM work, and give a real situation that breaks each.
- State No Free Lunch, then explain why it does not mean all models are equally good in practice.
- Read off the inductive bias of a model you are considering, and predict whether it will win on small or large
  data.
- Say why selecting a model on the test set turns the test score into a lie, in terms of the maximum of noisy
  estimates.`),
  ],
  refs: [
    book('Understanding Machine Learning: From Theory to Algorithms', 'Shalev-Shwartz & Ben-David', 2014, 'https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/', 'Free PDF. The rigorous foundation for everything in this lesson.'),
    book('The Elements of Statistical Learning', 'Hastie, Tibshirani & Friedman', 2009, 'https://hastie.su.domains/ElemStatLearn/', 'Free PDF. The standard reference for classical ML.'),
    paper('The Lack of A Priori Distinctions Between Learning Algorithms', 'David Wolpert', 1996, 'https://doi.org/10.1162/neco.1996.8.7.1341', 'The No Free Lunch theorem, in the original.'),
    paper('Understanding deep learning requires rethinking generalization', 'Zhang et al.', 2016, 'https://arxiv.org/abs/1611.03530', 'Deep networks fit random labels perfectly and still generalise on real ones. The paper that broke classical capacity theory.'),
  ],
},
{
  id: 'ml-linear-regression',
  title: 'Linear Regression',
  sub: 'The model you should always try first, and the geometry behind least squares.',
  mins: 26, level: 'core',
  prereq: ['math-vectors', 'math-probability'],
  tags: ['regression', 'least squares'],
  sections: [
    tldr(`Fit a straight line (or a flat plane, in more dimensions) through your data by making the squared
errors as small as possible. That is the entire model, and it has a closed-form solution — no iteration, no
learning rate, no training loop.

It is worth taking seriously rather than treating as a warm-up. Linear regression is the baseline that tells
you whether a fancier model is earning its complexity, its coefficients are interpretable in a way no neural
network's are, and the geometry behind it — **projection onto what your features can explain** — reappears in
PCA, in attention, and in the least-squares core of half the methods in this atlas.`),

    jargon([
      ['$\\hat y$ ("y-hat")', 'The model\'s *prediction*. The hat always means "estimated" as opposed to the true value $y$.'],
      ['$\\mathbf{w}$ (weights) and $b$ (bias)', 'The numbers being learned. Each $w_j$ scales one feature; $b$ shifts the whole line up or down (the intercept).'],
      ['feature', 'One input column. "Square footage" is a feature; "square footage, bedrooms, zip code" are three.'],
      ['residual', 'The error on one example: actual minus predicted, $y_i - \\hat y_i$. What the model failed to explain.'],
      ['design matrix $X$', 'Your whole dataset as a matrix: one row per example, one column per feature.'],
      ['least squares', 'Choosing $\\mathbf{w}$ to minimise the *sum of squared* residuals. The squaring is a real modelling choice, not an aesthetic one.'],
      ['closed form', 'A solution you can write down as a formula and compute directly, with no iterative search.'],
      ['normal equations', 'The specific formula that solves least squares. "Normal" here means *perpendicular* — it comes from the geometry, not from the normal distribution.'],
      ['column space of $X$', 'Every prediction your features are capable of producing — the [span](#/l/math-matrices) of the columns of $X$.'],
      ['$R^2$ ("R-squared")', 'The fraction of the target\'s variance the model explains. 1 is perfect, 0 is no better than predicting the average.'],
      ['covariate', 'Another word for a feature, more common in statistics.'],
    ]),

    t(`## The model

A prediction is a weighted sum of the features, plus an offset:

$$\\hat y = \\mathbf{w}^{\\mathsf T}\\mathbf{x} + b = w_1x_1 + \\cdots + w_dx_d + b$$

Read $\\mathbf{w}^{\\mathsf T}\\mathbf{x}$ as the dot product from [the vectors lesson](#/l/math-vectors) — it is
literally "multiply each feature by its weight and add them up". Each $w_j$ says how much feature $j$ pushes
the prediction; $b$ is where the line crosses zero.

A standard piece of bookkeeping: append a constant 1 to every input, and $b$ becomes just another weight. Then
the model is simply $\\hat y = \\mathbf{w}^{\\mathsf T}\\mathbf{x}$, with no special case for the intercept. Every
library does this internally.

Now fit it. Stack all $n$ examples into a matrix $X$ (one row each) so $X\\mathbf{w}$ gives all $n$ predictions
at once, and minimise the average squared error:

$$\\mathcal{L}(\\mathbf{w}) = \\frac{1}{n}\\|X\\mathbf{w}-\\mathbf{y}\\|^2$$

Inside the norm, $X\\mathbf{w}-\\mathbf{y}$ is the vector of residuals. The $\\|\\cdot\\|^2$ squares each one and
adds them up. So the objective reads: *make the total squared miss as small as possible*.`),

    viz('linear-regression'),

    t(`## Why squared error?

Squaring the errors is a choice. Why not absolute error, or the fourth power? Three completely independent
lines of reasoning land on the square, which is usually a sign that a definition is the right one:

1. **Probabilistic.** It is exactly maximum likelihood if you assume the noise is Gaussian — derived in the
   [probability lesson](#/l/math-probability). So "use squared error" and "I believe the errors are normally
   distributed" are the same statement.
2. **Geometric.** It finds the orthogonal projection of $\\mathbf{y}$ onto the column space of $X$: the closest
   point to your target that your features are capable of producing. This is the projection picture from
   [the vectors lesson](#/l/math-vectors), scaled up.
3. **Computational.** Squared error is convex and quadratic, so there is exactly one minimum and you can solve
   for it directly. No local minima, no learning rate, no waiting.

The price for all that convenience is fragility. A residual of 10 contributes 100 to the loss while a residual
of 1 contributes 1, so a single wild outlier can outvote a hundred well-behaved points and drag the whole line
toward itself. Press **+ outlier** in the figure and watch it happen. When your data has outliers you want
absolute error (L1) or a Huber loss, which are more robust precisely because they punish large residuals less.`),

    diagram('Least squares as projection: the picture behind the algebra',
`<svg viewBox="0 0 560 230" role="img" aria-label="y projected onto the plane spanned by the feature columns">
  <polygon points="60,175 330,175 430,110 160,110" style="fill: color-mix(in srgb, var(--s1) 12%, transparent); stroke: var(--s1); stroke-width: 1.5"/>
  <text class="dlabel" x="72" y="196" style="fill: var(--s1)">column space of X — every prediction your features can make</text>
  <defs>
    <marker id="ls1" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" style="fill: var(--s6)"/></marker>
    <marker id="ls2" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" style="fill: var(--s3)"/></marker>
  </defs>
  <line x1="180" y1="150" x2="300" y2="35" style="stroke: var(--s6); stroke-width: 2.4" marker-end="url(#ls1)"/>
  <text class="dmono" x="308" y="32" style="fill: var(--s6)">y  (the target)</text>
  <line x1="180" y1="150" x2="292" y2="140" style="stroke: var(--s3); stroke-width: 2.4" marker-end="url(#ls2)"/>
  <text class="dmono" x="300" y="146" style="fill: var(--s3)">y-hat = Xw  (best fit)</text>
  <line x1="296" y1="139" x2="300" y2="35" style="stroke: var(--s2); stroke-width: 2; stroke-dasharray: 4 3"/>
  <text class="dmono" x="310" y="90" style="fill: var(--s2)">residual</text>
  <path d="M287,131 L279,130 L280,138" style="fill:none; stroke: var(--text-faint); stroke-width: 1.4"/>
  <text class="dlabel" x="60" y="222">The residual meets the plane at 90 degrees. That right angle IS the normal equations.</text>
</svg>`,
      `The target $\\mathbf{y}$ almost never lies in the plane your features can reach — real data has noise your
model cannot express. Least squares drops a perpendicular onto that plane. The right angle is not decoration:
"the residual is orthogonal to every feature column" is algebraically identical to $X^{\\mathsf
T}(X\\mathbf{w}-\\mathbf{y}) = 0$, which *is* the normal equations. That is also why they are called "normal" —
the word means perpendicular here, and has nothing to do with the normal distribution.`),

    deriv('The normal equations, one partial derivative at a time', `We want the $\\mathbf{w}$ minimising $\\mathcal{L}(\\mathbf{w}) = \\sum_{i=1}^{n} (\\mathbf{x}_i \\cdot \\mathbf{w} - y_i)^2$, where $\\mathbf{x}_i$ is row $i$ of $X$. Nothing here needs matrix calculus — a partial derivative and the chain rule are enough.

Differentiate with respect to one weight $w_j$, holding the others fixed. The outer function is a square, whose derivative is twice the inside; the inside is $\\mathbf{x}_i\\cdot\\mathbf{w} - y_i$, and the only term of that dot product containing $w_j$ is $x_{ij}w_j$, so its derivative is $x_{ij}$. Multiply them and sum over the data:

$$\\frac{\\partial \\mathcal{L}}{\\partial w_j} = \\sum_{i=1}^{n} 2\\,(\\mathbf{x}_i\\cdot\\mathbf{w} - y_i)\\,x_{ij}$$

Now read that sum. The quantity $r_i = \\mathbf{x}_i\\cdot\\mathbf{w} - y_i$ is residual $i$, and $x_{ij}$ is entry $i$ of column $j$ of $X$. So the sum is twice the dot product of column $j$ with the residual vector. Setting every one of these to zero at once — one equation per column — gives, in matrix form:

$$X^{\\mathsf T}(X\\mathbf{w}-\\mathbf{y}) = \\mathbf{0} \\qquad\\Longrightarrow\\qquad \\hat{\\mathbf{w}} = (X^{\\mathsf T}X)^{-1}X^{\\mathsf T}\\mathbf{y}$$

**Now read the condition geometrically.** "Column $j$ of $X$, dotted with the residual, is zero" is exactly "the residual is orthogonal to feature $j$" — and it holds for every feature at once. That is the defining property of dropping a perpendicular, which is why the picture above and the algebra here are the same statement. It also explains the name: *normal* is the old word for perpendicular.

There is a sanity check hiding in it, too. If the residual still had any component along a feature column, you could reduce your error by moving a little along that feature. Being orthogonal to all of them means there is nothing left to gain — which is what "minimum" should mean.

**Do not implement the boxed formula literally.** Forming $X^{\\mathsf T}X$ squares the condition number, as [the numerics lesson](#/l/math-numerics) showed. Call \`np.linalg.lstsq\`, which works on $X$ directly.`),

    viz('regression-loss-surface'),

    t(`## Reading the coefficients

Interpretability is linear regression's main selling point over a neural network, so it is worth being precise
about what a coefficient actually claims.

$w_j$ is the expected change in $y$ per unit change in $x_j$, **holding every other feature fixed**. That last
clause is small, easy to skip, and where nearly all misinterpretation happens.

**Correlated features make "holding others fixed" fictional.** If square footage and bedroom count move together
in your data, then "one more bedroom with the square footage unchanged" describes a house that essentially never
appears. The model still reports a number for it, extrapolated from nothing. Symptom: coefficients that swing
wildly — even flipping sign — when you refit on a slightly different sample.

**Magnitudes are not comparable across features unless the scales are.** A coefficient of 0.001 on "house price
in dollars" and 5.0 on "number of bathrooms" tells you nothing about relative importance. Standardise the
features first (subtract the mean, divide by the standard deviation) if you want to compare.`),

    warn(`**A regression coefficient is a correlation, and correlation is not causation.** This is repeated so
often that it has lost its force, so here is the concrete version:

Adding or removing a single covariate can *reverse* a coefficient's sign. This is Simpson's paradox, and it is
not a rare edge case — it shows up in medical trials, hiring data, and A/B tests routinely. A model showing
"treatment X is associated with worse outcomes" can flip to "better outcomes" once severity is included, because
sicker patients got the treatment.

Regression answers causal questions only under assumptions about *which* variables to include, and the
regression itself has no way to check those assumptions. Nothing in the fit will warn you. If you need a causal
claim, you need a causal design — a randomised experiment, or an explicit causal model — not a bigger $R^2$.`),

    t(`## $R^2$ and its traps

$$R^2 = 1 - \\frac{\\sum_i (y_i-\\hat y_i)^2}{\\sum_i (y_i-\\bar y)^2} = 1 - \\frac{\\text{your error}}{\\text{the error of just guessing the mean}}$$

That second form is the useful reading. The denominator is how badly you would do with the dumbest possible
model — always predict $\\bar y$, the average. So $R^2$ measures how much of that baseline error you eliminated.
$R^2 = 0.8$ means "I removed 80% of the error a constant predictor would have made".

Three traps, in increasing order of how often they catch people:

- **$R^2$ never decreases when you add a feature** — not even a column of pure random noise, which will always
  find some accidental correlation to exploit. So $R^2$ on the training set cannot be used to choose between
  models of different sizes. Use adjusted $R^2$, or better, just look at held-out error.
- **High $R^2$ does not mean the model is right**, and low $R^2$ does not mean it is useless. A model that
  explains 3% of the variance in stock returns is extraordinarily valuable. A model that explains 95% of the
  variance using a feature that leaked from the future is worthless.
- **On the test set, $R^2$ can be negative.** It just means you did worse than predicting the mean. Surprising
  the first time you see it, entirely possible.`),

    t(`## Beyond straight lines

One point that trips up almost everyone: linear regression is linear **in the parameters**, not in the
features. The requirement is that the prediction is a weighted sum of *something* — the somethings can be as
curvy as you like.

So all of this is still linear regression, solvable with the exact same closed form:

$$\\hat y = w_0 + w_1x + w_2x^2 + w_3\\sin(x) + w_4 x_1x_2$$

Just compute $x^2$, $\\sin(x)$, and $x_1x_2$ as new columns and hand them to the same solver. The model has no
idea they are related to each other. Under this trick, linear regression can fit any curve at all, given enough
engineered features.

Which raises the obvious next question, and it is not a happy one: if I can always add more features and always
reduce my training error, when should I stop? That is [the next lesson](#/l/ml-overfitting).`),

    t(`## How much would these coefficients move on different data?

Your coefficients came from one particular sample. Had you collected a different sample from the same source,
you would have got different numbers. A **standard error** estimates how much different — it is the standard
deviation of a coefficient across the samples you might have drawn.

You do not need to actually collect more data to estimate it. The residuals tell you how noisy $y$ is, and
$X^{\\mathsf T}X$ tells you how much the features overlap:

$$\\text{Var}(\\hat{\\mathbf{w}}) = \\sigma^2 (X^{\\mathsf T}X)^{-1}, \\qquad \\sigma^2 \\approx \\frac{\\sum_i r_i^2}{n - d - 1}$$

The standard error of $w_j$ is the square root of the $j$-th diagonal entry. Two things fall out of that formula,
and both match intuition. More noise in $y$ means bigger standard errors. And if two features are nearly
identical, $X^{\\mathsf T}X$ is nearly singular, its inverse is enormous, and the standard errors explode — which
is precisely the multicollinearity problem from earlier, now with a number attached.

Divide a coefficient by its own standard error and you get a **t-statistic**: how many standard errors the
estimate sits away from zero. A magnitude below about 2 means this data cannot distinguish the coefficient from
zero, so do not tell a story about its sign.`),

    code('Least squares, three ways', `import numpy as np

rng = np.random.default_rng(0)
n, d = 200, 5
X = rng.normal(size=(n, d))
w_true = np.array([2.0, -1.0, 0.5, 0.0, 3.0])
y = X @ w_true + rng.normal(0, 0.3, n)

Xb = np.column_stack([np.ones(n), X])              # add intercept

# 1. normal equations (educational only — squares the condition number)
w1 = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)

# 2. lstsq: what you should actually call
w2, *_ = np.linalg.lstsq(Xb, y, rcond=None)

# 3. gradient descent, to connect with everything later
w3 = np.zeros(d + 1)
for step in range(3000):
    grad = 2 * Xb.T @ (Xb @ w3 - y) / n
    w3 -= 0.05 * grad

print("truth      :", np.r_[0.0, w_true])
print("normal eqns:", w1.round(4))
print("lstsq      :", w2.round(4))
print("gradient   :", w3.round(4))

# diagnostics
resid = y - Xb @ w2
ss_res, ss_tot = (resid**2).sum(), ((y - y.mean())**2).sum()
print(f"\\nR^2 = {1 - ss_res/ss_tot:.4f}")
print(f"residual std = {resid.std():.4f}   (true noise = 0.30)")

# standard errors: how much would these coefficients move on new data?
sigma2 = ss_res / (n - d - 1)
cov = sigma2 * np.linalg.inv(Xb.T @ Xb)
se = np.sqrt(np.diag(cov))
print("\\ncoef     estimate    std.err     t-stat")
for i, (c, s) in enumerate(zip(w2, se)):
    print(f"w{i}   {c:10.4f} {s:10.4f} {c/s:10.2f}")
print("\\nw4 (true value 0) should have |t| < 2 -> not distinguishable from zero")`,
      'All three fitting methods land on the same weights, which is the point: the closed form and gradient descent are solving the same problem, and the closed form only exists because squared error is quadratic. The diagnostics at the end are where the practical value is. The residual standard deviation recovers the true noise level of 0.30, meaning the model has explained everything except the noise — and the coefficient with a true value of zero is the one whose t-statistic comes out small, exactly as it should.'),

    quiz('Two features in your regression are correlated at 0.99. What should you expect?',
      ['Individually unstable coefficients with huge standard errors, though predictions may still be fine',
       'The model will fail to fit the training data',
       'One coefficient will automatically become zero',
       'Predictions will be badly biased'],
      0,
      'This is **multicollinearity**. $X^{\\mathsf T}X$ becomes near-singular, so the coefficient split between the two features is nearly arbitrary — the fit can trade a large positive weight on one against a large negative weight on the other. Predictions stay accurate (their *sum* is well determined) but interpretation is worthless. Ridge regularization is the standard remedy.'),

    recap(`- Write the linear model and say what each symbol does, including why the intercept is usually folded
  into $\\mathbf{w}$.
- Give the three independent justifications for squared error, and name the failure mode you accept in return.
- Draw the projection picture and explain why "the residual is perpendicular" *is* the normal equations.
- State exactly what a coefficient claims, and two reasons that claim is often meaningless in practice.
- Explain why $R^2$ always improves on training data and what to look at instead.
- Explain why $\\hat y = w_0 + w_1x + w_2x^2$ is still linear regression.`),
  ],
  refs: [
    book('The Elements of Statistical Learning, Ch. 3', 'Hastie, Tibshirani & Friedman', 2009, 'https://hastie.su.domains/ElemStatLearn/', 'The definitive treatment of linear methods for regression.'),
    book('An Introduction to Statistical Learning', 'James, Witten, Hastie & Tibshirani', 2021, 'https://www.statlearning.com/', 'Free PDF. Gentler than ESL, with R and Python labs.'),
    book('Regression and Other Stories', 'Gelman, Hill & Vehtari', 2020, 'https://avehtari.github.io/ROS-Examples/', 'The best book on what regression coefficients actually mean, and how interpretation goes wrong.'),
    blog('Ordinary Least Squares Regression explained visually', 'Setosa', 2014, 'https://setosa.io/ev/ordinary-least-squares-regression/', 'Interactive.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'ml-overfitting',
  title: 'Overfitting, Bias, and Variance',
  sub: 'The central tension — and why the textbook U-curve is only half the story.',
  mins: 28, level: 'core',
  prereq: ['ml-linear-regression'],
  tags: ['generalization', 'bias-variance'],
  sections: [
    tldr(`Give a model enough flexibility and it will fit your training data perfectly — including the parts
that were pure noise. Noise does not repeat, so the model then fails on anything new. That is **overfitting**,
and managing it is most of what applied ML practice consists of.

The classical framing splits your error into **bias** (the model is too rigid to capture the truth) and
**variance** (the model is so flexible it chases the randomness in your particular sample), and says you must
trade one against the other.

That framing is correct and incomplete. Modern deep networks sit in a regime where it visibly fails, and the
lesson ends with two phenomena — double descent and grokking — that the textbook story cannot account for.`),

    jargon([
      ['overfitting', 'Fitting the noise in your training data, not just the signal. Training error keeps dropping while test error rises.'],
      ['underfitting', 'The opposite: the model is too simple to capture the real pattern. Both errors stay high.'],
      ['capacity', 'How complicated a function a model *can* represent. More capacity = more ability to overfit.'],
      ['bias (in this lesson)', 'How far off the model is *on average across different training sets*. Nothing to do with the $b$ term in linear regression, or with social bias. Unfortunate overloading of the word.'],
      ['variance (in this lesson)', 'How much the fitted model changes when you retrain it on a different sample of the same size.'],
      ['irreducible noise $\\sigma^2$', 'Randomness in the data itself that no model can predict. The floor on your error.'],
      ['interpolate', 'Fit the training data *exactly* — zero training error. In classical theory this was a warning sign; in deep learning it is normal.'],
      ['data leakage', 'Information about the answer sneaking into your features. Produces suspiciously good scores and models that fail in production.'],
      ['early stopping', 'Halting training before the model has had time to overfit. A regularizer disguised as a scheduling decision.'],
      ['double descent', 'The observation that test error goes up, then *down again* as models grow past the interpolation point.'],
      ['grokking', 'A model reaching perfect training accuracy while still at chance on test, then abruptly generalizing many thousands of steps later.'],
    ]),

    t(`## The phenomenon

A model with enough flexibility will fit your training data perfectly. That sounds like success and is often
the opposite: real data is signal plus noise, and a sufficiently flexible model fits both. The signal
generalises. The noise does not — it was random, and the randomness in your next batch of data will be
different.

So the model has memorised accidents.`),

    viz('polynomial-overfit'),

    t(`Move the degree slider from 0 to 14 with only 12 training points and watch the two error numbers separate.
At low degree the curve is too stiff to follow the data at all. Around degree 3 it tracks the underlying shape.
By degree 11 it passes through every single point exactly — training error zero — while writhing violently in
between, and the test error has gone through the roof.

**That gap between training and test error is the thing all of ML methodology exists to manage.** Everything
that follows — the bias-variance decomposition, regularization, cross-validation — is a tool for reasoning
about it.`),

    t(`## The decomposition

Where does test error actually come from? For squared loss it splits into exactly three parts, and the split is
an identity, not an approximation:

$$\\underbrace{\\mathbb{E}\\big[(y-\\hat f(\\mathbf{x}))^2\\big]}_{\\text{expected test error}} =
\\underbrace{\\big(\\mathbb{E}[\\hat f(\\mathbf{x})]-f(\\mathbf{x})\\big)^2}_{\\text{bias}^2} +
\\underbrace{\\mathbb{E}\\big[(\\hat f(\\mathbf{x})-\\mathbb{E}[\\hat f(\\mathbf{x})])^2\\big]}_{\\text{variance}} +
\\underbrace{\\sigma^2}_{\\text{noise}}$$

The crucial and easily-missed detail: **the expectations are over random draws of the training set.** Imagine
collecting 100 different training sets of the same size, fitting your model on each, and getting 100 slightly
different models. The decomposition is a statement about that collection.

- **Bias** — how far the *average* of those 100 models is from the truth. High bias means the model class is
  fundamentally too rigid; collecting more data will not help. This is **underfitting**.
- **Variance** — how much the 100 models differ *from each other*. High variance means the model is chasing
  the accidents of whichever sample it happened to get. This is **overfitting**.
- **Noise** — the randomness in $y$ itself. Irreducible. No model, however good, beats this floor, and a
  model that appears to beat it is leaking.`),

    deriv('Where the three terms come from', `The whole derivation is one trick — add and subtract the same thing — plus the expectation rules from [the probability lesson](#/l/math-probability). Fix an input $\\mathbf{x}$ and write $\\bar f = \\mathbb{E}[\\hat f(\\mathbf{x})]$ for the average prediction across training sets. The truth is $y = f(\\mathbf{x}) + \\epsilon$ with $\\mathbb{E}[\\epsilon] = 0$ and $\\text{Var}(\\epsilon) = \\sigma^2$.

**Step 1: split off the noise.** The noise $\\epsilon$ is independent of anything the model does, so

$$\\mathbb{E}\\big[(y - \\hat f)^2\\big] = \\mathbb{E}\\big[(f + \\epsilon - \\hat f)^2\\big] = \\mathbb{E}\\big[(f - \\hat f)^2\\big] + \\underbrace{\\mathbb{E}[\\epsilon^2]}_{\\sigma^2}$$

The cross term $2\\mathbb{E}[\\epsilon(f - \\hat f)]$ vanishes because $\\epsilon$ is independent with mean zero.

**Step 2: add and subtract the average prediction.** Inside the remaining term, write $f - \\hat f = (f - \\bar f) + (\\bar f - \\hat f)$ and expand the square:

$$\\mathbb{E}\\big[(f - \\hat f)^2\\big] = \\underbrace{(f - \\bar f)^2}_{\\text{bias}^2} + \\mathbb{E}\\big[(\\bar f - \\hat f)^2\\big] + 2(f - \\bar f)\\,\\mathbb{E}\\big[\\bar f - \\hat f\\big]$$

The first term has no randomness left in it — both $f$ and $\\bar f$ are fixed numbers — so the expectation does nothing to it. The last term dies because $\\mathbb{E}[\\bar f - \\hat f] = \\bar f - \\bar f = 0$: the average prediction is, by definition, the average of the predictions.

$$\\mathbb{E}\\big[(y-\\hat f)^2\\big] = \\underbrace{(f - \\bar f)^2}_{\\text{bias}^2} + \\underbrace{\\mathbb{E}[(\\hat f - \\bar f)^2]}_{\\text{variance}} + \\underbrace{\\sigma^2}_{\\text{noise}}$$

Every step was exact, which is why this is an identity rather than an approximation — and why the three numbers really do add up when you measure them, as the challenge does.`),

    diagram('Bias and variance, as a dartboard',
`<svg viewBox="0 0 620 220" role="img" aria-label="Four dartboards showing high and low bias crossed with high and low variance">
  <g transform="translate(30,20)">
    <circle cx="55" cy="55" r="48" style="fill:none; stroke: var(--border)"/><circle cx="55" cy="55" r="30" style="fill:none; stroke: var(--border)"/><circle cx="55" cy="55" r="12" style="fill: color-mix(in srgb, var(--s3) 25%, transparent); stroke: var(--s3)"/>
    <g style="fill: var(--s1)"><circle cx="52" cy="53" r="3"/><circle cx="58" cy="57" r="3"/><circle cx="55" cy="60" r="3"/><circle cx="59" cy="51" r="3"/></g>
    <text class="dtitle" x="55" y="126" text-anchor="middle" style="fill: var(--s3)">just right</text>
    <text class="dlabel" x="55" y="143" text-anchor="middle">low bias, low variance</text>
  </g>
  <g transform="translate(180,20)">
    <circle cx="55" cy="55" r="48" style="fill:none; stroke: var(--border)"/><circle cx="55" cy="55" r="30" style="fill:none; stroke: var(--border)"/><circle cx="55" cy="55" r="12" style="fill:none; stroke: var(--s3)"/>
    <g style="fill: var(--s1)"><circle cx="30" cy="35" r="3"/><circle cx="36" cy="30" r="3"/><circle cx="33" cy="41" r="3"/><circle cx="27" cy="33" r="3"/></g>
    <text class="dtitle" x="55" y="126" text-anchor="middle" style="fill: var(--s2)">underfitting</text>
    <text class="dlabel" x="55" y="143" text-anchor="middle">HIGH bias, low variance</text>
  </g>
  <g transform="translate(330,20)">
    <circle cx="55" cy="55" r="48" style="fill:none; stroke: var(--border)"/><circle cx="55" cy="55" r="30" style="fill:none; stroke: var(--border)"/><circle cx="55" cy="55" r="12" style="fill:none; stroke: var(--s3)"/>
    <g style="fill: var(--s1)"><circle cx="25" cy="60" r="3"/><circle cx="80" cy="35" r="3"/><circle cx="50" cy="88" r="3"/><circle cx="70" cy="70" r="3"/><circle cx="40" cy="25" r="3"/></g>
    <text class="dtitle" x="55" y="126" text-anchor="middle" style="fill: var(--s6)">overfitting</text>
    <text class="dlabel" x="55" y="143" text-anchor="middle">low bias, HIGH variance</text>
  </g>
  <g transform="translate(480,20)">
    <circle cx="55" cy="55" r="48" style="fill:none; stroke: var(--border)"/><circle cx="55" cy="55" r="30" style="fill:none; stroke: var(--border)"/><circle cx="55" cy="55" r="12" style="fill:none; stroke: var(--s3)"/>
    <g style="fill: var(--s1)"><circle cx="18" cy="28" r="3"/><circle cx="42" cy="20" r="3"/><circle cx="22" cy="48" r="3"/><circle cx="38" cy="38" r="3"/></g>
    <text class="dtitle" x="55" y="126" text-anchor="middle" style="fill: var(--text-faint)">worst case</text>
    <text class="dlabel" x="55" y="143" text-anchor="middle">HIGH bias, HIGH variance</text>
  </g>
  <text class="dlabel" x="30" y="200">Each dot is the model you would get from one training set. Bias = how far the CLUSTER is from the bullseye.</text>
  <text class="dlabel" x="30" y="216">Variance = how SPREAD OUT the dots are. You only ever get to throw one dart — that is the difficulty.</text>
</svg>`,
      `The last line is the practical trap. You train once, so you see one dot. A model with high variance can
land near the centre by luck, and you will believe it is a good model. Cross-validation is essentially a way of
throwing several darts so you can see the spread.`),

    viz('bias-variance'),

    key(`Each faint curve above is the model fit to a *different* random dataset of the same size. Degree 1: all the
curves lie on top of each other (low variance) but far from the truth (high bias). Degree 10: they scatter wildly
(high variance) around a mean that tracks the truth well (low bias).

Now raise **points per dataset** and watch variance collapse. That is the whole reason more data lets you use bigger
models.`),

    t(`## What actually controls capacity

Parameter count is a poor proxy. What matters is the **effective** capacity, which you can restrict many ways:

| Lever | Mechanism |
|---|---|
| Fewer parameters | Direct |
| L1 / L2 penalties | Shrinks the reachable region of parameter space |
| Early stopping | Limits how far the optimizer travels from the initialization |
| Data augmentation | Enlarges the effective dataset with known invariances |
| Dropout | Trains an implicit ensemble of subnetworks |
| More data | Does not reduce capacity — makes it affordable |

Early stopping is worth a note: for linear models it is *provably* equivalent to a form of L2 regularization, with the
number of steps playing the role of $1/\\lambda$. It is regularization by not-yet-having-overfit.`),

    t(`## Double descent: the picture is incomplete

Everything above is the classical story, and it predicts a U-shaped curve: error falls as the model gets big
enough to capture the signal, then rises as it starts capturing noise. Pick the bottom of the U.

That is true right up until it isn't. Keep growing the model past the point where it can fit the training data
*exactly*, and test error **falls again** — frequently below the classical sweet spot.`),

    viz('double-descent'),

    t(`Look at where the peak sits: precisely at $d = n$, the point where the model has just barely enough
parameters to interpolate. That location is the clue.

At $d = n$ there is essentially **one** way to fit the data exactly, and the model is forced into it whatever
the cost — typically enormous, precariously balanced weights that cancel each other out. It is the worst of both
worlds: no flexibility to choose a sensible fit, and no slack to ignore the noise.

Push past $d = n$ and suddenly there are *many* exact fits available. Now the question becomes which one the
optimizer picks — and gradient descent, started from small weights, tends to settle on the one with the smallest
weights that still fits, which is also the smoothest one. More parameters did not give the model more ways to go
wrong; they gave the optimizer room to choose well among fits that were all equally correct on the training
data.

Notice that this is a claim about the *optimizer*, not about the model class. The extra capacity is there
either way; what changed is that gradient descent has a preference, and the preference happens to be a good one.
That preference has a name — **implicit regularization** — and pinning down exactly what it prefers, and why it
generalises, is one of the more active open questions in the field.`),

    warn(`This is not a curiosity from a toy experiment. **It is the regime every modern deep network lives
in**: far more parameters than training examples, training loss driven to essentially zero, and good
generalization anyway.

Classical capacity theory says that should not work, and it does not merely fail to predict it — it predicts the
opposite. The candidate explanations (implicit regularization by SGD, the norm rather than the count of
parameters, effective versus nominal capacity) are all plausible and none is settled.

Practical consequence: **be suspicious of "the model is too big for this dataset" as an argument.** It was
excellent advice for 2005 and is often wrong now. Measure it.`),

    t(`## Grokking: "training loss stopped improving" is not "learning stopped"

Here is a stranger one. Train a small transformer on a clean algorithmic task — modular arithmetic, say. Watch
what happens:

- Within a few thousand steps it reaches **100% training accuracy**. Memorised the table.
- Test accuracy sits at **chance**. It has learned nothing generalizable.
- Training loss is flat. By every normal signal, the run is finished.
- Then, tens of thousands of steps later, test accuracy **abruptly jumps to 100%**.

It did not learn gradually. It sat there apparently doing nothing and then suddenly understood modular
arithmetic.`),

    viz('grokking'),

    t(`Set weight decay to zero in that figure and it never happens — the model memorises and stays memorised
forever.

The current explanation goes: two different internal circuits can fit the training data. One memorises the
lookup table; the other implements the actual algorithm. Both achieve zero training loss, so the loss provides
no pressure to prefer either. But the generalizing circuit has **smaller weight norm**, and weight decay applies
a slow, constant pull toward small norm. So regularization gradually dismantles the memorising solution and
grows the algorithmic one, long after the loss curve went flat.

The practical lesson is uncomfortable, and it is why this sits in a lesson about overfitting: **a flat training
loss is not reliable evidence that learning has stopped.** Something can still be reorganising internally.`),

    warn(`**Overfitting is not the only failure.** The others are often worse because they are invisible in your metrics:

- **Data leakage** — a feature that encodes the target (a patient ID that correlates with diagnosis; normalizing
  before splitting). Symptom: implausibly good validation scores.
- **Distribution shift** — test data from a different regime than training. Your held-out set will not catch it.
- **Shortcut learning** — the model finds a spurious cue that works in your data and nowhere else (the classic:
  detecting pneumonia from the hospital's scanner watermark rather than the lungs).`),

    code('Watching the two errors diverge', `import numpy as np

rng = np.random.default_rng(1)
def truth(x): return np.sin(2.2 * x) * 1.3 + 0.35 * x

n_train = 15
Xtr = rng.uniform(-2.5, 2.5, n_train)
ytr = truth(Xtr) + rng.normal(0, 0.35, n_train)
Xte = rng.uniform(-2.5, 2.5, 2000)
yte = truth(Xte) + rng.normal(0, 0.35, 2000)

def fit_poly(X, y, deg, lam=0.0):
    A = np.vander(X, deg + 1, increasing=True)
    return np.linalg.solve(A.T @ A + lam * np.eye(deg + 1), A.T @ y)

def mse(w, X, y):
    return np.mean((np.vander(X, len(w), increasing=True) @ w - y) ** 2)

print(" deg   train      test      |w|")
for deg in range(0, 14):
    w = fit_poly(Xtr, ytr, deg, lam=1e-9)
    print(f"{deg:4d}  {mse(w,Xtr,ytr):8.4f}  {mse(w,Xte,yte):9.4f}  {np.linalg.norm(w):9.1f}")

print("\\nnow with ridge (lam=1e-2), degree 13:")
w = fit_poly(Xtr, ytr, 13, lam=1e-2)
print(f"      {mse(w,Xtr,ytr):8.4f}  {mse(w,Xte,yte):9.4f}  {np.linalg.norm(w):9.1f}")
print("\\nSame 14 parameters. The penalty, not the count, controls capacity.")`,
      'Watch the $\\|w\\|$ column: overfitting shows up as enormous coefficients fighting each other. That is exactly the quantity ridge regularization penalizes.'),

    quiz('Your model gets 99% train accuracy and 71% test accuracy. You add more training data. What happens to each?',
      ['Train accuracy falls, test accuracy rises — the gap narrows from both sides',
       'Both rise',
       'Train stays at 99%, test is unchanged',
       'Both fall'],
      0,
      'More data is harder to memorize, so train accuracy drops toward the true achievable rate. It also reduces variance, so test accuracy rises. Both effects shrink the gap. If test accuracy does *not* improve with substantially more data, your problem is bias or distribution shift, not overfitting — and more data will not fix it.'),

    recap(`- Explain overfitting as "fitting the noise", and say why noise is the part that cannot transfer.
- Name the three terms of the bias–variance decomposition and say what the expectation is taken over.
- Diagnose a model from its train/test gap: high both = bias, big gap = variance, and what to do about each.
- List four ways to reduce effective capacity that are not "use fewer parameters".
- Describe double descent and say why the peak sits at $d = n$.
- Explain why "your model is too big for this dataset" is weaker advice than it used to be.
- Name three failure modes that overfitting language does not cover — leakage, distribution shift, shortcut
  learning — and how each announces itself.`),
  ],
  refs: [
    paper('Reconciling modern machine learning practice and the bias-variance trade-off', 'Belkin et al.', 2019, 'https://arxiv.org/abs/1812.11118', 'The double descent paper. Short and important.'),
    paper('Deep Double Descent', 'Nakkiran et al.', 2019, 'https://arxiv.org/abs/1912.02292', 'Shows the same phenomenon across model size, data size, and training time.'),
    book('The Elements of Statistical Learning, Ch. 7', 'Hastie, Tibshirani & Friedman', 2009, 'https://hastie.su.domains/ElemStatLearn/', 'The classical decomposition, done rigorously.'),
    paper('Shortcut Learning in Deep Neural Networks', 'Geirhos et al.', 2020, 'https://arxiv.org/abs/2004.07780', 'The failure mode your validation set will not catch.'),
    paper('Leakage in data mining', 'Kaufman et al.', 2011, 'https://dl.acm.org/doi/10.1145/2020408.2020496', 'A taxonomy of leakage, with real competition examples.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'ml-regularization',
  title: 'Regularization: Ridge, Lasso, and Elastic Net',
  sub: 'Constraining a model so it generalizes — and why L1 zeroes coefficients.',
  mins: 24, level: 'core',
  prereq: ['ml-overfitting'],
  tags: ['regularization', 'sparsity'],
  sections: [
    tldr(`If overfitting is caused by a model having too much freedom, the fix is to take some freedom away.
**Regularization** does that by adding a penalty for large weights, so the model has to *earn* every bit of
complexity with a matching reduction in error.

Two flavours dominate, and the difference between them is genuinely interesting rather than a detail.
**Ridge (L2)** shrinks every coefficient smoothly toward zero. **Lasso (L1)** shrinks them *and drives some to
exactly zero*, doing feature selection as a side effect. Why one produces exact zeros and the other does not
comes down to the shape of a diamond versus a circle.`),

    jargon([
      ['regularization', 'Anything that constrains a model to reduce overfitting. Usually a penalty term, but early stopping and dropout count too.'],
      ['penalty term', 'An extra quantity added to the loss that grows as the weights grow, so the optimizer is pushed toward smaller weights.'],
      ['$\\lambda$ (lambda)', 'The regularization strength — the knob controlling how much you care about small weights versus fitting the data. Chosen by cross-validation.'],
      ['ridge / L2', 'Penalising the sum of *squared* weights, $\\|\\mathbf{w}\\|_2^2$. Also called weight decay in deep learning, and Tikhonov regularization in numerical analysis. Three names, one thing.'],
      ['lasso / L1', 'Penalising the sum of *absolute* weights, $\\|\\mathbf{w}\\|_1$. Produces sparse solutions.'],
      ['sparse', 'Most entries are exactly zero. A sparse model uses only a few of the available features.'],
      ['shrinkage', 'Pulling coefficient estimates toward zero. Adds a little bias in exchange for a large reduction in variance — usually a good trade.'],
      ['support', 'The set of features with nonzero coefficients. "Recovering the support" = correctly identifying which features actually matter.'],
      ['elastic net', 'Using L1 and L2 penalties together, to get sparsity without L1\'s instability on correlated features.'],
      ['standardize', 'Rescale each feature to mean 0 and standard deviation 1. Mandatory before regularizing — see the warning below.'],
    ]),

    t(`## Add a penalty

The idea in one line: stop asking only "does this fit the data?" and start asking "does this fit the data
*without needing extreme weights*?"

$$\\mathcal{L}(\\mathbf{w}) = \\underbrace{\\|X\\mathbf{w}-\\mathbf{y}\\|^2}_{\\text{fit the data}} + \\lambda\\, \\underbrace{\\Omega(\\mathbf{w})}_{\\text{stay simple}}$$

The optimizer now has to balance two demands. Reducing the fit term by adding an elaborate wiggle is only worth
it if the wiggle does not cost too much in the penalty term. Complexity has a price.

Which penalty you use determines the character of the solution:

- **Ridge (L2)**: $\\Omega = \\|\\mathbf{w}\\|_2^2 = \\sum_j w_j^2$. Shrinks all coefficients smoothly toward zero,
  never quite reaching it.
- **Lasso (L1)**: $\\Omega = \\|\\mathbf{w}\\|_1 = \\sum_j |w_j|$. Shrinks *and* sets some to exactly zero.
- **Elastic net**: $\\alpha\\|\\mathbf{w}\\|_1 + (1-\\alpha)\\|\\mathbf{w}\\|_2^2$. Both at once.

The knob $\\lambda$ sweeps between two extremes: at $\\lambda \\to 0$ you recover the ordinary unpenalized fit,
and at $\\lambda \\to \\infty$ every coefficient is crushed to zero and the model predicts a constant. The useful
setting is somewhere in between, and it is found by cross-validation essentially every time — there is no
formula for it.`),

    t(`## Why L1 gives you exact zeros

This is the question everyone asks and few explanations answer properly. The trick is to stop thinking about
penalties and start thinking about **constraints**.

Minimising $\\text{loss} + \\lambda\\Omega(\\mathbf{w})$ turns out to be equivalent to minimising the loss *subject
to* a budget, $\\Omega(\\mathbf{w}) \\le t$. The link is intuitive even without the proof: charging a price per
unit of $\\Omega$ and imposing a hard cap on $\\Omega$ both push the solution the same way, and for every price
$\\lambda$ there is some cap $t$ that lands you on exactly the same answer — a bigger $\\lambda$ meaning a tighter
$t$.

The constraint version is the one you can draw. The loss forms elliptical contours around the unpenalized
optimum, growing outward from it. The constraint carves out a region around the origin. Your answer must lie
inside the region, and you want the smallest loss possible, so it sits where the smallest loss contour first
**touches** the region's boundary.

Now recall the [unit ball shapes](#/l/math-vectors): the L1 region is a **diamond with sharp corners sitting on
the axes**, while the L2 region is a **smooth circle**. That difference is the whole answer.`),

    viz('regularization-geometry'),

    key(`**A corner is a coordinate that equals exactly zero.**

The L1 diamond has corners on the axes, and a corner sticks out. A loss contour drifting in from a generic
direction hits the pointy bit first — and at that point of contact, one coordinate is precisely 0. That is a
coefficient set to exactly zero, and it happens for a *range* of directions, not by fluke.

The L2 circle has no corners. Contact happens at whichever boundary point the contour reaches first, which is
generically a place where every coordinate is small but nonzero. Coefficients approach zero and never arrive.

That is the entire explanation. Sparsity from L1 is not a numerical accident or a property of the solver — it is
a consequence of the constraint region having corners aligned with the axes.`),

    viz('regularization-path'),

    deriv('Ridge in closed form, and what it does to the spectrum', `The ridge objective $\\|X\\mathbf{w}-\\mathbf{y}\\|^2 + \\lambda\\|\\mathbf{w}\\|^2$ has gradient
$2X^{\\mathsf T}(X\\mathbf{w}-\\mathbf{y}) + 2\\lambda\\mathbf{w}$, so

$$\\hat{\\mathbf{w}}_{\\text{ridge}} = (X^{\\mathsf T}X + \\lambda I)^{-1}X^{\\mathsf T}\\mathbf{y}$$

Substituting the SVD $X = U\\Sigma V^{\\mathsf T}$:

$$\\hat{\\mathbf{w}}_{\\text{ridge}} = V\\,\\text{diag}\\!\\left(\\frac{\\sigma_i}{\\sigma_i^2+\\lambda}\\right)U^{\\mathsf T}\\mathbf{y}$$

Compare against OLS, which has $1/\\sigma_i$. Ridge multiplies each direction by the shrinkage factor
$\\frac{\\sigma_i^2}{\\sigma_i^2+\\lambda}$:

- Large $\\sigma_i$ (well-determined directions): factor ≈ 1, barely touched.
- Small $\\sigma_i$ (poorly-determined directions): factor ≈ 0, heavily suppressed.

**Ridge shrinks selectively, along the directions your data does not constrain.** It also makes $X^{\\mathsf T}X+\\lambda I$
invertible even when $X^{\\mathsf T}X$ is not, which is why ridge works fine with more features than samples.

Lasso has no closed form, because $|w|$ has a sharp corner at zero and cannot be differentiated there. It is
solved one coordinate at a time instead — see the next derivation.`),

    deriv('The lasso update, and where the zeros come from algebraically', `The corner picture explains sparsity geometrically. Here is the same fact as arithmetic, and it is also exactly the update the challenge asks you to write.

Solve for one coefficient $w_j$ at a time, holding the rest fixed. With everything else frozen, the objective as a function of $w_j$ alone is a parabola plus an absolute value:

$$g(w_j) = \\tfrac{1}{2}a\\,w_j^2 - \\rho\\, w_j + \\lambda|w_j| + \\text{const}$$

where $a = \\sum_i x_{ij}^2$ is how much that feature varies, and $\\rho = \\sum_i x_{ij}(y_i - \\text{other terms})$ is how strongly the feature correlates with what the other coefficients have not yet explained. Without the $\\lambda$ term, the minimum would be at $w_j = \\rho/a$ — ordinary least squares for one coordinate.

Now handle the absolute value by splitting into cases, because $|w_j|$ has a different derivative on each side of zero.

**If $w_j > 0$:** $|w_j| = w_j$, so $g'(w_j) = a w_j - \\rho + \\lambda = 0$, giving $w_j = (\\rho - \\lambda)/a$. This is only consistent with our assumption $w_j > 0$ when $\\rho > \\lambda$.

**If $w_j < 0$:** $|w_j| = -w_j$, so $w_j = (\\rho + \\lambda)/a$, consistent only when $\\rho < -\\lambda$.

**If $-\\lambda \\le \\rho \\le \\lambda$:** neither case is consistent, so the minimum is not on either side. It is at the corner itself: $w_j = 0$ exactly.

Putting the three cases together gives the **soft-threshold** rule:

$$w_j = \\frac{\\text{sign}(\\rho)\\,\\max(|\\rho| - \\lambda,\\ 0)}{a}$$

Read the $\\max(\\cdot, 0)$: any feature whose correlation with the unexplained part is smaller than $\\lambda$ gets a coefficient of *precisely* zero, not a small one. And notice that "smaller than $\\lambda$" is a whole interval, not a knife edge — which is the algebraic version of "a corner sticks out, so a whole range of directions hits it".

Ridge, for comparison, has no absolute value and no cases: its one-coordinate solution is $\\rho/(a + \\lambda)$, which shrinks toward zero as $\\lambda$ grows but reaches it only in the limit.`),

    t(`## When to use which

| | Ridge | Lasso |
|---|---|---|
| Many small effects | ✓ best | ✗ arbitrarily discards |
| Few large effects, rest noise | fine | ✓ best — recovers the support |
| Correlated features | ✓ splits weight among them | ✗ picks one arbitrarily, unstably |
| $d > n$ | ✓ | ✓ (but selects at most $n$ features) |
| Interpretability | worse | ✓ sparse model |

Elastic net exists precisely for the correlated-features case: the L2 term encourages correlated predictors to enter
or leave together (the "grouping effect") while L1 still produces sparsity.`),

    warn(`**Standardize your features before regularizing.** This is the single most common regularization bug,
and it fails silently.

The penalty treats all coefficients identically — it just sums them up. But a coefficient's *size* depends on
the units of its feature. Measure a length in metres and you get some coefficient $w$; measure the same length
in millimetres and the coefficient becomes $w/1000$ to compensate. Identical model, identical predictions — but
the millimetre version is penalized a thousand times less.

So without standardisation, your regularization strength is silently set by your unit choices. Subtract the
mean and divide by the standard deviation for every feature first.

**And do not penalize the intercept.** Shrinking $b$ toward zero is a statement that the target's mean should be
near zero, which is almost never something you believe. Every library excludes it by default; make sure yours
does if you write the penalty by hand.`),

    t(`## The Bayesian reading

From the [probability lesson](#/l/math-probability): MAP estimation adds $\\log p(\\mathbf{w})$ to the log-likelihood.

- Gaussian prior $\\mathbf{w}\\sim\\mathcal{N}(0,\\tau^2I)$ → log-prior $\\propto -\\|\\mathbf{w}\\|^2$ → **ridge**.
- Laplace prior $p(w_j)\\propto e^{-|w_j|/b}$ → log-prior $\\propto -\\|\\mathbf{w}\\|_1$ → **lasso**.

The Laplace density has a sharp spike at zero, which is exactly the prior belief "most coefficients are probably
exactly zero." The geometry and the probability tell the same story.`),

    t(`## In deep learning

- **Weight decay** is L2, and it is used universally. With AdamW it is applied decoupled from the adaptive scaling
  (see the [optimization lesson](#/l/math-optimization)).
- **L1 is rare** for weights — sparsity is more usefully obtained through structured pruning. But L1 is central to
  sparse autoencoders in interpretability, where the *activations* are what you want sparse.
- **Dropout, early stopping, data augmentation, and label smoothing** are all regularizers that do not look like
  penalty terms.`),

    code('Ridge, lasso, and the coefficient paths', `import numpy as np

rng = np.random.default_rng(4)
n, d = 60, 20
X = rng.normal(size=(n, d))
X = (X - X.mean(0)) / X.std(0)                 # standardize! see the warning above
w_true = np.zeros(d); w_true[[0, 3, 7]] = [2.5, -1.8, 1.2]   # only 3 real signals
y = X @ w_true + rng.normal(0, 0.5, n)

def ridge(lam):
    return np.linalg.solve(X.T @ X + lam*np.eye(d), X.T @ y)

def lasso(lam, iters=500):
    w = np.zeros(d)
    XtX, Xty = X.T @ X, X.T @ y
    for _ in range(iters):
        for j in range(d):
            rho = Xty[j] - XtX[j] @ w + XtX[j, j] * w[j]
            w[j] = np.sign(rho) * max(abs(rho) - lam, 0) / XtX[j, j]   # soft threshold
    return w

for lam in [0.1, 5, 30]:
    wr, wl = ridge(lam), lasso(lam)
    print(f"lambda = {lam}")
    print(f"  ridge: {np.sum(np.abs(wr) < 1e-8):2d} exact zeros, "
          f"max|w| = {np.abs(wr).max():.3f}")
    print(f"  lasso: {np.sum(np.abs(wl) < 1e-8):2d} exact zeros, "
          f"max|w| = {np.abs(wl).max():.3f}")
    found = set(np.argsort(-np.abs(wl))[:3])
    print(f"  lasso top-3 features {sorted(found)}  (truth: [0, 3, 7])\\n")`,
      'The soft-threshold update `sign(rho)·max(|rho|−λ, 0)` is the closed-form solution to the one-coordinate lasso problem. That `max(·, 0)` is literally where the zeros come from.'),

    quiz('You have 10,000 features, 200 samples, and believe ~20 features matter. Which regularizer?',
      ['Lasso or elastic net — you want variable selection, and lasso produces exact zeros',
       'Ridge, because it handles d > n',
       'No regularization; use all features',
       'Dropout'],
      0,
      'Your prior is *sparsity*, so match it: L1 selects. Ridge would also make the problem well-posed but would give you 10,000 small nonzero coefficients — no selection. If features are correlated in groups, prefer elastic net, since pure lasso picks one member of each correlated group arbitrarily and unstably.'),

    recap(`- Write a regularized objective and explain what the two terms are competing over.
- Explain L1 sparsity from the geometry — corners on the axes — without saying "it just does".
- Choose between ridge, lasso, and elastic net from a description of the data, and justify the choice.
- Say what ridge does to well-determined versus poorly-determined directions, and connect it to singular values.
- Explain why failing to standardise features silently corrupts regularization.
- Translate ridge and lasso into priors, and say which prior believes "most coefficients are exactly zero".`),
  ],
  refs: [
    paper('Regression Shrinkage and Selection via the Lasso', 'Robert Tibshirani', 1996, 'https://www.jstor.org/stable/2346178', 'The original lasso paper.'),
    paper('Regularization and variable selection via the elastic net', 'Zou & Hastie', 2005, 'https://doi.org/10.1111/j.1467-9868.2005.00503.x', 'Why pure lasso struggles with correlated features.'),
    book('Statistical Learning with Sparsity', 'Hastie, Tibshirani & Wainwright', 2015, 'https://hastie.su.domains/StatLearnSparsity/', 'Free PDF. Everything about L1 methods.'),
    paper('Ridge Regression: Biased Estimation for Nonorthogonal Problems', 'Hoerl & Kennard', 1970, 'https://www.jstor.org/stable/1267351', 'The original ridge paper.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'ml-logistic',
  title: 'Logistic Regression and Classification',
  sub: 'The linear model for probabilities, and the loss every classifier inherits.',
  mins: 24, level: 'core',
  prereq: ['ml-linear-regression', 'math-information'],
  tags: ['classification', 'cross-entropy'],
  sections: [
    tldr(`Take linear regression, and squash its output through a function that maps any real number into
$[0,1]$. Now it outputs a probability instead of an unbounded score, and you have a classifier.

Despite the name, logistic regression is a **classification** method. It is worth more attention than its
simplicity suggests, because its final layer *is* the final layer of essentially every neural classifier and
every language model. When GPT picks a next token, the last thing it does is softmax regression over the
vocabulary.`),

    jargon([
      ['classification', 'Predicting a category (cat/dog, spam/not) rather than a number. Regression predicts numbers.'],
      ['sigmoid $\\sigma(z)$', 'The S-shaped squashing function $1/(1+e^{-z})$. Maps any real number into $(0,1)$, so its output can be read as a probability.'],
      ['logit', 'The raw pre-squash score $\\mathbf{w}^{\\mathsf T}\\mathbf{x}+b$. Any real number. Also the name of the inverse of the sigmoid.'],
      ['odds', 'Probability expressed as a ratio: $p/(1-p)$. A probability of 0.75 is odds of 3 (three to one).'],
      ['log-odds', 'The logarithm of the odds. Ranges over all reals, which is exactly why a linear model can predict it directly.'],
      ['odds ratio', '$e^{w_j}$ — how much the odds multiply per unit increase in feature $j$. The number reported in every medical paper.'],
      ['cross-entropy loss', 'The standard classification loss, from [the information lesson](#/l/math-information). Also called log loss or negative log-likelihood.'],
      ['softmax', 'The multi-class version of the sigmoid: turns $K$ scores into $K$ probabilities that sum to 1.'],
      ['calibration', 'Whether the reported probabilities are honest. A calibrated model that says "70%" is right about 70% of the time.'],
      ['threshold', 'The cutoff for turning a probability into a decision. Defaults to 0.5, which is usually wrong.'],
      ['separable data', 'Data where some straight line perfectly divides the classes with no mistakes.'],
    ]),

    t(`## From scores to probabilities

We want to output $p(y=1\\mid\\mathbf{x})$ — a probability, so a number between 0 and 1. But a linear function
$\\mathbf{w}^{\\mathsf T}\\mathbf{x}+b$ produces any real number at all, including $-47$ and $1000$.

The fix is to squash it:

$$p(y=1\\mid\\mathbf{x}) = \\sigma(\\mathbf{w}^{\\mathsf T}\\mathbf{x}+b), \\qquad \\sigma(z)=\\frac{1}{1+e^{-z}}$$

Check the ends: as $z \\to +\\infty$, $e^{-z} \\to 0$ and $\\sigma \\to 1$. As $z \\to -\\infty$, $e^{-z}$ blows up
and $\\sigma \\to 0$. At $z = 0$ it gives exactly $0.5$. Smooth, monotonic, bounded — everything we asked for.

But there is a cleaner way to read the model, by turning the sigmoid inside out. Rearranging gives:

$$\\log\\frac{p}{1-p} = \\mathbf{w}^{\\mathsf T}\\mathbf{x}+b$$

The left side is the **log-odds**. So logistic regression is not really "linear regression with a squash bolted
on" — it is a linear model *of the log-odds*, and the sigmoid is just how you get back to a probability.

This is why the model is interpretable at all. $w_j$ is the change in log-odds per unit of $x_j$, and $e^{w_j}$
is the **odds ratio** — "each additional year of age multiplies the odds of the diagnosis by 1.07". That is the
form every clinical paper reports, and now you know where it comes from.`),

    diagram('Probability, odds, and log-odds are the same information in three coordinate systems',
`<svg viewBox="0 0 620 170" role="img" aria-label="Mapping between probability, odds and log-odds">
  <text class="dtitle" x="20" y="26">probability p</text>
  <line x1="150" y1="20" x2="580" y2="20" style="stroke: var(--s1); stroke-width: 2"/>
  <g style="fill: var(--s1)"><circle cx="150" cy="20" r="4"/><circle cx="365" cy="20" r="4"/><circle cx="580" cy="20" r="4"/></g>
  <text class="dmono" x="150" y="42" text-anchor="middle" style="fill: var(--s1)">0</text>
  <text class="dmono" x="365" y="42" text-anchor="middle" style="fill: var(--s1)">0.5</text>
  <text class="dmono" x="580" y="42" text-anchor="middle" style="fill: var(--s1)">1</text>
  <text class="dlabel" x="150" y="58">bounded — a linear model cannot safely predict this</text>

  <text class="dtitle" x="20" y="94">odds p/(1-p)</text>
  <line x1="150" y1="88" x2="580" y2="88" style="stroke: var(--s2); stroke-width: 2"/>
  <g style="fill: var(--s2)"><circle cx="150" cy="88" r="4"/><circle cx="365" cy="88" r="4"/></g>
  <text class="dmono" x="150" y="110" text-anchor="middle" style="fill: var(--s2)">0</text>
  <text class="dmono" x="365" y="110" text-anchor="middle" style="fill: var(--s2)">1</text>
  <text class="dmono" x="590" y="92" style="fill: var(--s2)">→ ∞</text>
  <text class="dlabel" x="150" y="126">half-bounded — better, still asymmetric</text>

  <text class="dtitle" x="20" y="152">log-odds</text>
  <line x1="150" y1="146" x2="580" y2="146" style="stroke: var(--s3); stroke-width: 2"/>
  <circle cx="365" cy="146" r="4" style="fill: var(--s3)"/>
  <text class="dmono" x="130" y="150" text-anchor="end" style="fill: var(--s3)">-∞ ←</text>
  <text class="dmono" x="365" y="168" text-anchor="middle" style="fill: var(--s3)">0</text>
  <text class="dmono" x="590" y="150" style="fill: var(--s3)">→ ∞</text>
</svg>`,
      `The bottom line is the whole design. Log-odds run over *all* real numbers, symmetrically around zero, which
is exactly the range a linear model naturally produces. Model the log-odds linearly, then map back down. The
sigmoid is just that map.`),

    viz('sigmoid-softmax'),

    t(`## The loss

Now, how do we fit it? The obvious move — reuse squared error — is wrong, and instructively so.

Two problems. First, squared error composed with a sigmoid is **non-convex** in $\\mathbf{w}$, so you lose the
guarantee of a single optimum. Second, and worse: it produces near-zero gradients exactly where the model is
*most confidently wrong*. A model that says $p = 0.001$ when the answer is 1 sits in the sigmoid's flat tail,
and squared error's gradient there is essentially nothing. The model is maximally wrong and receives almost no
signal to change.

The right loss is the negative log-likelihood, known here as **binary cross-entropy**:

$$\\mathcal{L} = -\\frac{1}{n}\\sum_i \\big[y_i\\log p_i + (1-y_i)\\log(1-p_i)\\big]$$

Read it as a switch. When $y_i = 1$, the second term vanishes and you are left with $-\\log p_i$ — the
[surprise](#/l/math-information) of the true answer. When $y_i = 0$, the first term vanishes and you pay
$-\\log(1-p_i)$ instead. Either way: *how surprised was the model by what actually happened?*

This is convex in $\\mathbf{w}$, its gradient is remarkably clean, and — as the derivation below shows — it
solves the vanishing-gradient problem exactly.`),

    deriv('The gradient, and why it looks like linear regression', `**First, the derivative of the sigmoid**, which you need for anything that follows. Write $\\sigma(z) = (1+e^{-z})^{-1}$ and use the chain rule:

$$\\sigma'(z) = -(1+e^{-z})^{-2}\\cdot(-e^{-z}) = \\frac{e^{-z}}{(1+e^{-z})^{2}} = \\underbrace{\\frac{1}{1+e^{-z}}}_{\\sigma}\\cdot\\underbrace{\\frac{e^{-z}}{1+e^{-z}}}_{1-\\sigma}$$

so $\\sigma'(z) = \\sigma(z)\\big(1-\\sigma(z)\\big) = p(1-p)$. Worth remembering: it is largest at $p = 0.5$ (where it equals $0.25$) and collapses toward zero at both ends. Those flat tails are where the trouble lives.

**Now the loss gradient.** With $p = \\sigma(z)$ and $z = \\mathbf{w}\\cdot\\mathbf{x}$, the per-example loss is $\\mathcal{L}_i = -[y\\log p + (1-y)\\log(1-p)]$. Differentiate with respect to $p$, then chain through $\\sigma$:

$$\\frac{\\partial \\mathcal{L}_i}{\\partial z} = \\underbrace{-\\left[\\frac{y}{p} - \\frac{1-y}{1-p}\\right]}_{\\partial\\mathcal{L}/\\partial p}\\cdot\\underbrace{p(1-p)}_{\\partial p/\\partial z} = -\\big[y(1-p) - (1-y)p\\big] = p - y$$

Multiply by $\\partial z/\\partial w_j = x_j$ and average over the data:

$$\\nabla_{\\mathbf{w}}\\mathcal{L} = \\frac{1}{n}\\sum_i (p_i - y_i)\\,\\mathbf{x}_i$$

That is the *same shape* as linear regression's gradient, with the predicted probability in place of the predicted number: (prediction − target) times input. You have now seen this form three times — least squares, softmax with cross-entropy, and here — which is a good sign it is worth remembering.

**And notice what happened to the $p(1-p)$.** It appeared from the sigmoid and was cancelled by the $1/p$ and $1/(1-p)$ coming out of the logarithms. That cancellation is the whole reason cross-entropy is the right loss here. With squared error there are no logarithms, nothing cancels the $p(1-p)$, and a confidently wrong prediction — $p \\approx 0$ when $y = 1$ — sits in the flat tail where $p(1-p) \\approx 0$ and receives almost no gradient. Cross-entropy gives that same example a gradient of $p - y \\approx -1$: the largest it can be.`),

    viz('logistic-regression'),

    t(`## Multiclass: softmax regression

For $K$ classes, one weight vector each, normalized by softmax:

$$p(y=k\\mid\\mathbf{x}) = \\frac{\\exp(\\mathbf{w}_k^{\\mathsf T}\\mathbf{x})}{\\sum_{j=1}^K \\exp(\\mathbf{w}_j^{\\mathsf T}\\mathbf{x})}$$

The loss is cross-entropy, and the gradient is again $\\mathbf{p}-\\mathbf{y}$ (derived in the
[Jacobians lesson](#/l/math-jacobian)).

**This is the last layer of essentially every neural classifier and every language model.** A language model's output
layer is softmax regression over a 100,000-way vocabulary, applied to the features the transformer computed.`),

    warn(`**On separable data, unregularized logistic regression does not converge.** If a hyperplane perfectly separates
the classes, scaling $\\mathbf{w}$ up always improves the loss (predictions get more confident, log-loss keeps falling),
so $\\|\\mathbf{w}\\|\\to\\infty$. Always regularize. Interestingly, the *direction* it diverges along converges to the
max-margin separator — an early and clean example of implicit bias in gradient descent.`),

    t(`## Calibration

A model outputs 0.7. Does that *mean* anything, or is it just a number that happens to be large?

A model is **calibrated** if it does: among all the examples where it said 0.7, about 70% really are positive.
Uncalibrated models can still rank correctly — putting positives above negatives — while their actual numbers
are nonsense.

Logistic regression is usually well calibrated, and this follows directly from the loss. Cross-entropy is a
**proper scoring rule**, meaning it is minimised precisely by reporting your true beliefs. Any systematic
over- or under-confidence increases the loss, so training pushes toward honesty.`),

    warn(`Deep networks trained with *the same loss* are frequently badly **overconfident** — a network reporting
0.99 might be right only 80% of the time. The proper-scoring-rule argument does not save you, because it assumes
you actually reached the minimum; a high-capacity network trained to near-zero training loss has learned to be
confident on training data and carries that habit to test data.

Standard remedies: **temperature scaling** (divide the logits by a single constant tuned on a validation set —
cheap, effective, does not change the ranking) or **label smoothing** during training.

RLHF makes this markedly worse. Human raters prefer confident-sounding answers regardless of correctness, so
optimizing for their approval directly trains overconfidence in. This is a large part of why chat models
state wrong things with total assurance.`),

    t(`## Thresholds are a separate decision

The model gives you $p$. Turning $p$ into an action — flag this transaction, order this biopsy — requires a
**threshold**, and that is a different question with a different answer.

The default of 0.5 is optimal only when a false positive and a false negative cost the same. Consider how rarely
that is true: a missed cancer diagnosis and an unnecessary follow-up scan are not comparable, and neither are a
blocked legitimate payment and an approved fraudulent one.

The threshold should come from the cost structure — if a false negative is 10× worse than a false positive,
threshold near 0.1, not 0.5 — or from a target precision or recall your application actually requires. It is a
business decision that happens to be implemented in code, and separating it from the model is what lets you
change it without retraining.`),

    viz('roc-curve'),

    code('Logistic regression from scratch', `import numpy as np

rng = np.random.default_rng(0)
n = 400
X = np.vstack([rng.normal([-1, -0.5], 1.0, (n//2, 2)),
               rng.normal([ 1.5, 1.0], 1.0, (n//2, 2))])
y = np.r_[np.zeros(n//2), np.ones(n//2)]
Xb = np.column_stack([np.ones(n), X])

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

w = np.zeros(3)
lam = 0.01
for step in range(2000):
    p = sigmoid(Xb @ w)
    grad = Xb.T @ (p - y) / n + lam * np.r_[0, w[1:]]     # do not penalize intercept
    w -= 1.0 * grad
    if step % 500 == 0:
        loss = -np.mean(y*np.log(p+1e-12) + (1-y)*np.log(1-p+1e-12))
        print(f"step {step:5d}  loss {loss:.4f}  |w| {np.linalg.norm(w[1:]):.3f}")

p = sigmoid(Xb @ w)
print(f"\\naccuracy: {np.mean((p > 0.5) == y):.3f}")
print(f"weights:  {w.round(3)}")
print(f"odds ratio for x1: {np.exp(w[1]):.3f}")

# calibration check: bin by predicted probability, compare to actual rate
print("\\npredicted   actual    count")
for lo in np.arange(0, 1, 0.2):
    m = (p >= lo) & (p < lo + 0.2)
    if m.sum() > 5:
        print(f"  {lo:.1f}-{lo+0.2:.1f}    {y[m].mean():.3f}   {m.sum():5d}")`,
      'The calibration table at the end is the check worth running on any classifier. Group the predictions into bins and, for each bin, compare what the model claimed against what actually happened. The two columns track each other closely here — when this model says 0.7, roughly 70% of those cases really are positive. Run the same table on a deep network trained to near-zero training loss and the right column will fall well below the left.'),

    quiz('Why is squared error a bad loss for classification with a sigmoid output?',
      ["It is non-convex, and its gradient vanishes exactly when the model is confidently wrong",
       'It is not differentiable',
       'It cannot handle more than two classes',
       'It always overfits'],
      0,
      "With squared error the gradient carries a factor of $\\sigma'(z) = p(1-p)$, which goes to zero as $p \\to 0$ or $1$. A model that predicts 0.001 for a positive example gets essentially no gradient — it is maximally wrong and maximally stuck. Cross-entropy's $p-y$ gradient is largest precisely there. The $p(1-p)$ factors cancel; that cancellation is the whole point."),

    recap(`- Explain why the model predicts **log-odds** linearly rather than probability directly.
- Convert a coefficient into an odds ratio and state it in a sentence a doctor would accept.
- Give two reasons squared error is the wrong loss here, one about convexity and one about gradients.
- Recall the gradient $(p - y)\\mathbf{x}$ and say why it looks identical to linear regression's.
- Explain what calibration means, why logistic regression tends to have it, and why deep networks tend not to.
- Argue that choosing a decision threshold is a separate question from fitting the model, with an example where
  0.5 would be badly wrong.`),
  ],
  refs: [
    book('The Elements of Statistical Learning, Ch. 4', 'Hastie, Tibshirani & Friedman', 2009, 'https://hastie.su.domains/ElemStatLearn/', ''),
    paper('On Calibration of Modern Neural Networks', 'Guo et al.', 2017, 'https://arxiv.org/abs/1706.04599', 'Why deep nets are overconfident, and why temperature scaling fixes most of it.'),
    paper('The Implicit Bias of Gradient Descent on Separable Data', 'Soudry et al.', 2017, 'https://arxiv.org/abs/1710.10345', 'Where the weights go when the loss has no minimum.'),
    book('Applied Logistic Regression', 'Hosmer, Lemeshow & Sturdivant', 2013, '', 'The standard applied-statistics treatment, if you need odds ratios and inference rather than prediction.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'ml-trees-ensembles',
  title: 'Trees, Forests, and Gradient Boosting',
  sub: 'Still the best thing to run on tabular data, and it is not close.',
  mins: 26, level: 'core',
  tags: ['trees', 'ensembles', 'boosting'],
  sections: [
    tldr(`A decision tree is a flowchart of yes/no questions learned from data — "is age > 35? if so, is income
> 60k?" — with a prediction at each leaf. One tree is weak and unstable. Combining many is not.

There are two ways to combine them, and the distinction is the real content of this lesson. **Bagging** grows
many trees in parallel on different samples and averages them, which cancels out their instability. **Boosting**
grows trees one at a time, each fixing the errors of the ones before it, which builds accuracy incrementally.

On tabular data — spreadsheets, databases, anything with named columns — boosted trees are still the thing to
beat, and neural networks have repeatedly failed to beat them. That is worth understanding rather than
resenting.`),

    jargon([
      ['decision tree', 'A model that makes predictions by asking a sequence of threshold questions about features. Reads like nested if-statements, because that is what it is.'],
      ['split', 'One yes/no question: "feature 3 ≤ 0.7?" Each split divides the data into two groups.'],
      ['leaf', 'A terminal node — no more questions. Holds the prediction for everything that lands there.'],
      ['axis-aligned', 'Each split uses one feature at a time, so boundaries are always horizontal or vertical (in 2-D). Trees cannot draw a diagonal line except as a staircase.'],
      ['impurity', 'How mixed the labels are in a group. Zero when all examples share a label. Splits are chosen to reduce it.'],
      ['Gini / entropy', 'Two common impurity measures. They almost always pick the same splits; the choice rarely matters.'],
      ['ensemble', 'Many models combined into one prediction. Nearly always better than any single member.'],
      ['bagging', 'Bootstrap AGGregatING: train many models on random resamples of the data and average them. Reduces variance.'],
      ['bootstrap resample', 'A new dataset of the same size, drawn from the original *with replacement* — so some rows appear twice and about 37% not at all.'],
      ['random forest', 'Bagging applied to trees, plus a random subset of features considered at each split.'],
      ['boosting', 'Training models sequentially, each one focused on the mistakes of the previous ones. Reduces bias.'],
      ['residual', 'What the current model got wrong — actual minus predicted. Boosting fits new trees to these.'],
      ['stump', 'A tree with a single split. Deliberately weak; boosting prefers weak learners.'],
      ['tabular data', 'Data in rows and columns with heterogeneous, named features. As opposed to images, audio, or text.'],
    ]),

    t(`## Decision trees

The algorithm is almost embarrassingly simple. Look at all your data. Try every possible question of the form
"is feature $j$ above threshold $\\tau$?", and pick the one that best separates the labels. Split the data in
two, then repeat on each half. Stop when a group is pure enough or too small to bother with.

"Best separates the labels" is made precise by an **impurity** measure — a number that is zero when a group is
all one class and large when it is evenly mixed:

- **Gini**: $1-\\sum_k p_k^2$ — the probability two randomly drawn members of the group disagree.
- **Entropy**: $-\\sum_k p_k\\log p_k$ — [average surprise](#/l/math-information), same as before.
- **Variance**, for regression — the spread of the target values in the group.

At each step you choose the split that reduces impurity most. To predict, walk a new example down the questions
until it reaches a leaf, and return that leaf's majority class or mean value.`),

    viz('decision-tree'),

    t(`Trees have a genuinely attractive set of properties, and it is worth being specific because they explain
the final section of this lesson:

- **No feature scaling needed.** A split at "income > 60000" is unaffected if you switch to thousands. Trees are
  invariant to any monotone transformation of a feature — logging it changes nothing.
- **Mixed types and missing values handled natively.** Categorical, ordinal, numeric all coexist; missing can be
  its own branch.
- **Interactions come free.** Splitting on age and then on income *within* that branch is an interaction effect,
  discovered automatically without anyone specifying it.
- **A shallow tree is genuinely interpretable** — not "interpretable" in the aspirational sense, but readable as
  a set of rules by a domain expert.

And then the fatal flaw: **trees are high variance.** Move a handful of data points near the top split and a
different feature may win, changing every subtree below it. A fully-grown tree drives training error to zero by
carving out a leaf per example — pure memorisation.

That is exactly the failure mode from [the bias-variance lesson](#/l/ml-overfitting), and it is why nobody
deploys a single tree. It is also why the two great ensembling ideas work so unusually well *on* trees: they are
low-bias, high-variance base learners, which is precisely what averaging fixes.`),

    t(`## Bagging and random forests

**Bagging** = **B**ootstrap **AGG**regat**ING**. Draw $M$ resamples of your data — each the same size as the
original, sampled *with replacement*, so some rows appear twice and others not at all — train a tree on each,
and average their predictions.

(How many rows get left out? A particular row survives one draw with probability $1 - 1/n$, and there are $n$
draws, so it is absent from a resample with probability $(1-1/n)^n$. For any decent $n$ that is about $1/e
\\approx 0.37$. Hence the recurring 37%.)

Why should averaging help? Because it cancels variance — but only under a condition worth knowing exactly.
Average $M$ estimators, each with variance $\\sigma^2$ and correlation $\\rho$ between any pair, and the average
has variance

$$\\underbrace{\\rho\\sigma^2}_{\\text{does not shrink}} + \\underbrace{\\frac{1-\\rho}{M}\\sigma^2}_{\\text{shrinks to nothing}}$$

Look at the two terms. The second vanishes as $M$ grows — throw enough trees at it and it disappears. The first
does **not**. It is a floor, set entirely by how much the trees agree with each other.

So past a certain point the win does not come from *more* trees. **It comes from making the trees disagree.**
Two hundred nearly-identical trees are barely better than one.`),

    deriv('Where the two-term variance formula comes from', `This uses only the variance rules from [the probability lesson](#/l/math-probability), extended one step: when two variables are *not* independent, the variance of their sum picks up a covariance term.

$$\\text{Var}(A + B) = \\text{Var}(A) + \\text{Var}(B) + 2\\,\\text{Cov}(A, B)$$

Take $M$ tree predictions $T_1, \\ldots, T_M$, each with variance $\\sigma^2$, and each pair having correlation $\\rho$ — which means $\\text{Cov}(T_i, T_j) = \\rho\\sigma^2$ for $i \\neq j$. Sum them all:

$$\\text{Var}\\Big(\\sum_i T_i\\Big) = \\underbrace{M\\sigma^2}_{M \\text{ variance terms}} + \\underbrace{M(M-1)\\,\\rho\\sigma^2}_{\\text{one covariance per ordered pair}}$$

The average is that sum divided by $M$, and dividing a random quantity by $M$ divides its variance by $M^2$:

$$\\text{Var}\\Big(\\frac{1}{M}\\sum_i T_i\\Big) = \\frac{M\\sigma^2 + M(M-1)\\rho\\sigma^2}{M^2} = \\frac{\\sigma^2}{M} + \\frac{(M-1)\\rho\\sigma^2}{M}$$

Rearrange into the two-term form by writing $\\frac{M-1}{M} = 1 - \\frac{1}{M}$:

$$= \\rho\\sigma^2 + \\frac{1-\\rho}{M}\\sigma^2$$

Now read the two special cases. At $\\rho = 0$ — perfectly independent trees — this is $\\sigma^2/M$, the familiar "averaging $M$ independent things divides the variance by $M$". At $\\rho = 1$ — identical trees — it is $\\sigma^2$, and averaging achieves exactly nothing, which is obviously right since all $M$ copies are the same model.

Everything real sits in between, and the formula says precisely how much of the benefit survives.`),

    key(`**Random forests** are bagging plus one addition aimed squarely at that $\\rho\\sigma^2$ floor: at each
split, only a random subset of features is even considered (typically $\\sqrt{d}$ of them for classification).

That sounds like sabotage — you are hiding the best feature from most splits. It is, and that is the point. If
one feature is strongly predictive, every bagged tree would split on it first and they would all look alike.
Forcing trees to work with different features makes them genuinely different models, driving $\\rho$ down and
lowering the floor.

Two practical consequences worth knowing:
- **Forests do not overfit as you add trees.** More is monotonically better, up to diminishing returns. $M$ is
  a compute budget, not a hyperparameter to tune carefully.
- **Out-of-bag error is free cross-validation.** Each tree never saw ~37% of the data, so evaluate it on those
  rows. No separate validation split required.`),

    t(`## Gradient boosting

Boosting attacks the problem from the opposite end. Instead of many strong trees averaged, it builds many
**weak** trees in sequence, each one correcting what the ensemble so far still gets wrong:

$$F_m(\\mathbf{x}) = F_{m-1}(\\mathbf{x}) + \\nu\\, h_m(\\mathbf{x})$$

$F_{m-1}$ is the ensemble so far, $h_m$ is the new tree, and $\\nu$ is a small learning rate (0.1 or less) that
keeps any single tree from dominating.

What is $h_m$ fit to? The **negative gradient of the loss** with respect to the current predictions — which is
where the name comes from. For squared loss that gradient works out to be exactly the residual, $y - F_{m-1}$,
which is why boosting is usually first explained as "each tree fits the errors of the previous ones".

That explanation is correct but undersells it. Framing it as a gradient is what lets you boost *any*
differentiable loss — log loss, ranking losses, Poisson, quantile regression — by fitting trees to whatever the
negative gradient happens to be. It is gradient descent, where each "step" is a whole decision tree rather than
a parameter update.`),

    viz('boosting'),

    key(`**Bagging reduces variance. Boosting reduces bias.** Read that against the
[decomposition](#/l/ml-overfitting) and everything else follows.

Random forests start from trees that are already accurate on average but wildly unstable — low bias, high
variance — and average the instability away. Boosting starts from stumps that are barely better than guessing —
high bias, low variance — and stacks them until they collectively fit.

The consequence that matters in practice: **boosting can overfit and will.** Every added tree fits the residuals
harder, so past some number of rounds you are fitting noise. Boosting needs early stopping on a validation set,
always. Random forests essentially do not, which makes them the more forgiving choice when you have no time to
tune.`),

    diagram('Two ways to combine trees',
`<svg viewBox="0 0 620 230" role="img" aria-label="Bagging trains trees in parallel, boosting trains them in sequence">
  <text class="dtitle" x="20" y="24" style="fill: var(--s1)">BAGGING — in parallel, on different samples, then averaged</text>
  <g>
    <rect x="20" y="38" width="66" height="36" rx="4" style="fill: color-mix(in srgb, var(--s1) 14%, transparent); stroke: var(--s1)"/>
    <text class="dmono" x="53" y="60" text-anchor="middle" style="fill: var(--s1)">tree 1</text>
    <rect x="98" y="38" width="66" height="36" rx="4" style="fill: color-mix(in srgb, var(--s1) 14%, transparent); stroke: var(--s1)"/>
    <text class="dmono" x="131" y="60" text-anchor="middle" style="fill: var(--s1)">tree 2</text>
    <rect x="176" y="38" width="66" height="36" rx="4" style="fill: color-mix(in srgb, var(--s1) 14%, transparent); stroke: var(--s1)"/>
    <text class="dmono" x="209" y="60" text-anchor="middle" style="fill: var(--s1)">tree 3</text>
    <text class="dmono" x="262" y="60" style="fill: var(--text-faint)">...</text>
    <path d="M53,78 L160,100 M131,78 L160,100 M209,78 L160,100" style="fill:none; stroke: var(--border)"/>
    <rect x="112" y="100" width="96" height="30" rx="4" style="fill: color-mix(in srgb, var(--s3) 16%, transparent); stroke: var(--s3)"/>
    <text class="dmono" x="160" y="118" text-anchor="middle" style="fill: var(--s3)">average</text>
    <text class="dlabel" x="330" y="60">each tree sees a different resample,</text>
    <text class="dlabel" x="330" y="78">so their errors are uncorrelated</text>
    <text class="dlabel" x="330" y="102" style="fill: var(--s3)">→ cancels VARIANCE</text>
  </g>
  <text class="dtitle" x="20" y="168" style="fill: var(--s2)">BOOSTING — in sequence, each fixing the last one's mistakes</text>
  <g>
    <rect x="20" y="182" width="66" height="34" rx="4" style="fill: color-mix(in srgb, var(--s2) 14%, transparent); stroke: var(--s2)"/>
    <text class="dmono" x="53" y="203" text-anchor="middle" style="fill: var(--s2)">stump 1</text>
    <path d="M88,199 L96,199" style="stroke: var(--s2); stroke-width: 1.5"/>
    <rect x="98" y="182" width="66" height="34" rx="4" style="fill: color-mix(in srgb, var(--s2) 14%, transparent); stroke: var(--s2)"/>
    <text class="dmono" x="131" y="203" text-anchor="middle" style="fill: var(--s2)">stump 2</text>
    <path d="M166,199 L174,199" style="stroke: var(--s2); stroke-width: 1.5"/>
    <rect x="176" y="182" width="66" height="34" rx="4" style="fill: color-mix(in srgb, var(--s2) 14%, transparent); stroke: var(--s2)"/>
    <text class="dmono" x="209" y="203" text-anchor="middle" style="fill: var(--s2)">stump 3</text>
    <text class="dmono" x="262" y="203" style="fill: var(--text-faint)">...</text>
    <text class="dlabel" x="330" y="192">each tree is fit to what is still wrong,</text>
    <text class="dlabel" x="330" y="210">so it cannot be parallelised</text>
    <text class="dlabel" x="330" y="226" style="fill: var(--s2)">→ grinds down BIAS</text>
  </g>
</svg>`,
      `The structural difference has a practical consequence beyond accuracy: bagging parallelises perfectly
across cores, while boosting is inherently sequential. LightGBM and XGBoost recover speed by parallelising
*within* each tree's split search instead.`),

    t(`## Why gradient boosting still wins on tabular data

XGBoost, LightGBM, and CatBoost remain the default for tabular problems, and deep learning has repeatedly tried
and failed to displace them. This is not inertia or a lack of effort — the reasons are structural, and each one
is a mismatch between what MLPs assume and what tabular data is:

- Tabular features are **heterogeneous** — different scales, types, meanings. Trees are invariant to monotone feature
  transformations; MLPs are not.
- Real tabular relationships are often **non-smooth** (thresholds, step functions). Trees model those natively;
  MLPs have a bias toward smooth functions.
- Tabular datasets are **small** (thousands to millions of rows), where the strong inductive bias pays off.
- Irrelevant features are common, and trees ignore them almost for free.

The practical implementations add substantially to the basic algorithm: second-order (Newton) split finding,
histogram-based binning for speed, L1/L2 leaf regularization, and principled missing-value handling.`),

    t(`## Feature importance, and how it misleads

Three common measures, in increasing order of trustworthiness:

1. **Gain / split importance** — total impurity reduction from splits on a feature. Fast, and **biased toward
   high-cardinality features**, which have more possible split points. Do not trust it.
2. **Permutation importance** — shuffle a feature, measure the performance drop. Honest about what the model uses, but
   splits credit unpredictably among correlated features.
3. **SHAP values** — game-theoretic attribution with consistency guarantees, and per-prediction rather than global.
   Slower, and still not causal.

All three tell you what the *model* uses, never what *matters* in the world. A feature can be causally essential and
show zero importance because a correlated proxy absorbed it.`),

    code('A tree and a forest from scratch', `import numpy as np
rng = np.random.default_rng(0)

# XOR-ish data: impossible for a linear model, trivial for a tree
n = 400
X = rng.uniform(-2, 2, (n, 2))
y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
y = np.abs(y - (rng.random(n) < 0.05))            # 5% label noise

def gini(y):
    if len(y) == 0: return 0.0
    p = y.mean()
    return 1 - p**2 - (1-p)**2

def build(X, y, depth, max_depth, min_leaf=5):
    node = {"pred": y.mean() if len(y) else 0.5, "n": len(y)}
    if depth >= max_depth or len(y) < 2*min_leaf or y.std() == 0:
        return node
    best = None
    for f in range(X.shape[1]):
        for thr in np.quantile(X[:, f], np.linspace(0.05, 0.95, 20)):
            m = X[:, f] <= thr
            if m.sum() < min_leaf or (~m).sum() < min_leaf: continue
            score = (m.sum()*gini(y[m]) + (~m).sum()*gini(y[~m])) / len(y)
            if best is None or score < best[0]:
                best = (score, f, thr, m)
    if best is None: return node
    _, f, thr, m = best
    node.update(f=f, thr=thr,
                L=build(X[m], y[m], depth+1, max_depth),
                R=build(X[~m], y[~m], depth+1, max_depth))
    return node

def predict(node, X):
    if "f" not in node: return np.full(len(X), node["pred"])
    m = X[:, node["f"]] <= node["thr"]
    out = np.empty(len(X))
    out[m]  = predict(node["L"], X[m])
    out[~m] = predict(node["R"], X[~m])
    return out

split = 300
Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

for d in [1, 2, 3, 6, 12]:
    tree = build(Xtr, ytr, 0, d)
    tr = ((predict(tree, Xtr) > .5) == ytr).mean()
    te = ((predict(tree, Xte) > .5) == yte).mean()
    print(f"depth {d:2d}:  train {tr:.3f}   test {te:.3f}")

# a bagged forest of deep trees
forest = [build(*(lambda i: (Xtr[i], ytr[i]))(rng.integers(0, split, split)), 0, 12)
          for _ in range(25)]
pred = np.mean([predict(t, Xte) for t in forest], axis=0)
print(f"\\nforest of 25 deep trees: test {((pred > .5) == yte).mean():.3f}")
print("(deep single trees overfit; averaging them does not)")`,
      'Two things to read here. The depth sweep shows a single tree doing exactly what the lesson predicted: training accuracy climbs toward 1.0 while test accuracy peaks early and then falls, because a depth-12 tree on 300 points is carving out leaves around individual noisy labels. Then the forest, built from twenty-five of those same overfitting depth-12 trees, beats every single tree in the table. Nothing made the individual trees better — the resampling made their mistakes different, and different mistakes cancel when you average.'),

    quiz('A random forest scores 0.85 out of the box. Gradient boosting on the same features reaches 0.88, but only after you tune the learning rate, depth, and number of rounds together. What does this suggest?',
      ['Normal behavior — boosting usually edges out forests but is far more sensitive to hyperparameters',
       'The forest is broken',
       'The boosting model is overfitting',
       'The features are unsuitable for trees'],
      0,
      'This is the standard pattern. Random forests are nearly hyperparameter-free and give you a strong baseline immediately; boosting reaches a higher ceiling but needs the learning rate, tree depth, and number of rounds tuned together, with early stopping. A sensible workflow is: forest first for the baseline, boosting second for the final model.'),

    recap(`- Describe how a tree is grown, and what "impurity reduction" is choosing between.
- Say why a single tree is high variance, and connect that to why ensembling works so well on trees
  specifically.
- Read the bagging variance formula and explain why decorrelating trees matters more than adding them.
- Explain what random feature subsetting at each split is *for*.
- State the one-line difference: bagging kills variance, boosting kills bias — and derive from it which one
  needs early stopping.
- Give three structural reasons boosted trees still beat neural networks on tabular data.
- Name the failure mode of split-gain feature importance, and what to use instead.`),
  ],
  refs: [
    paper('Random Forests', 'Leo Breiman', 2001, 'https://link.springer.com/article/10.1023/A:1010933404324', 'The original. Still worth reading.'),
    paper('Greedy Function Approximation: A Gradient Boosting Machine', 'Jerome Friedman', 2001, 'https://projecteuclid.org/euclid.aos/1013203451', 'The gradient boosting framework.'),
    paper('XGBoost: A Scalable Tree Boosting System', 'Chen & Guestrin', 2016, 'https://arxiv.org/abs/1603.02754', 'The engineering that made boosting dominate Kaggle.'),
    paper('Why do tree-based models still outperform deep learning on tabular data?', 'Grinsztajn et al.', 2022, 'https://arxiv.org/abs/2207.08815', 'A careful, honest benchmark. Answers the question in the title.'),
    paper('A Unified Approach to Interpreting Model Predictions', 'Lundberg & Lee', 2017, 'https://arxiv.org/abs/1705.07874', 'SHAP.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'ml-svm-knn',
  title: 'SVMs, Kernels, and Nearest Neighbours',
  sub: 'Margins, the kernel trick, and the simplest algorithm that works.',
  mins: 24, level: 'core',
  prereq: ['ml-logistic'],
  tags: ['SVM', 'kernels', 'kNN'],
  sections: [
    tldr(`Two ideas that were the state of the art for fifteen years, and are still worth knowing.

**k-nearest neighbours** is the simplest algorithm that works: remember everything, and predict by looking up
what happened to similar examples. There is no training step at all. It is also, structurally, what a vector
database and a RAG pipeline do.

**Support vector machines** ask for the separating boundary with the widest safety margin, then pull off a
genuinely clever trick — the **kernel trick** — that lets a linear method draw curved boundaries by computing
similarities in a space it never actually visits.`),

    jargon([
      ['kNN', 'k-Nearest Neighbours. Predict by majority vote among the $k$ most similar training examples.'],
      ['non-parametric', 'A model with no fixed set of parameters — its complexity grows with the data. kNN stores the whole dataset.'],
      ['hyperplane', 'The flat dividing surface a linear classifier draws. A line in 2-D, a plane in 3-D, a "hyperplane" in general.'],
      ['margin', 'The width of the empty corridor between the decision boundary and the nearest data points on either side. Wider is safer.'],
      ['support vector', 'A training point sitting on or inside the margin. These are the only points that affect the solution — delete any other and nothing changes.'],
      ['hinge loss', '$\\max(0, 1 - y f(x))$ — zero once you are correct *and* outside the margin, growing linearly as you fall inside or misclassify.'],
      ['slack', 'Allowance for points to violate the margin, so the method works on data that is not perfectly separable.'],
      ['$C$', 'The SVM regularization knob. Large $C$ = punish violations hard = narrow margin, tight fit. Small $C$ = tolerant, wide margin.'],
      ['feature map $\\phi$', 'A transformation to a higher-dimensional space where data that was tangled becomes linearly separable.'],
      ['kernel $k(\\mathbf{x},\\mathbf{x}\')', 'A function computing the *inner product in the transformed space*, without ever performing the transformation.'],
      ['RBF kernel', 'Radial Basis Function, $\\exp(-\\gamma\\|\\mathbf{x}-\\mathbf{x}\'\\|^2)$. Corresponds to an infinite-dimensional feature space. The usual default.'],
      ['Gram matrix', 'The $n \\times n$ table of kernel values between every pair of training points.'],
    ]),

    t(`## k-nearest neighbours

Store the training data. To predict, find the $k$ closest stored points and let them vote. That is the whole
algorithm.

There is no training phase, no loss function, and no parameters — **the data *is* the model**. This makes it
the fastest baseline you can possibly stand up, and a useful sanity check: if an elaborate model cannot beat
kNN, something is wrong.`),

    viz('knn-boundary'),

    t(`It also demonstrates several ideas unusually cleanly:

- **$k$ is a direct bias-variance dial.** $k=1$ means every point gets its own territory — jagged boundary,
  maximum variance. Large $k$ smooths everything into a blur — high bias. There is no other knob.
- **Training accuracy at $k=1$ is exactly 100% and completely meaningless**, since each point is its own nearest
  neighbour. A memorable demonstration that training accuracy tells you nothing on its own.
- **Leave-one-out cross-validation is nearly free**, since there is no refitting to do.

Its problems are equally instructive, and both come back later:

- **Prediction cost scales with the dataset**, not the model. Every query compares against everything you
  stored. That is backwards from every other method here.
- **Distance has to mean something.** In high dimensions, [as we saw](#/l/math-vectors), every point is roughly
  equidistant from every other, so "nearest" stops carrying information. kNN degrades badly as $d$ grows.

Both problems have modern answers. Vector databases solve the first with approximate nearest-neighbour search
(HNSW, IVF-PQ), trading exactness for speed. And the second is why you run kNN in a *learned embedding space*
rather than on raw features — which is exactly what RAG is. **Retrieval-augmented generation is kNN over
embeddings, with a language model reading the results.**`),

    t(`## Support vector machines

Suppose your data *is* linearly separable. Then there are infinitely many lines that separate it, and most
classifiers will happily return any of them. Which should you prefer?

The SVM's answer: the one with the **widest margin** — the boundary that stays as far as possible from the
nearest point of either class. The intuition is robustness. A boundary skimming past your data will misclassify
the next slightly-shifted example; one running down the middle of the empty corridor has room to spare.

Formally, with labels written as $y_i\\in\\{-1,+1\\}$ (a convention that makes the algebra clean):

$$\\min_{\\mathbf{w},b}\\ \\tfrac12\\|\\mathbf{w}\\|^2 \\quad\\text{subject to}\\quad y_i(\\mathbf{w}^{\\mathsf T}\\mathbf{x}_i+b)\\ge 1 \\ \\text{ for every } i$$

The constraint says every point must be correctly classified *and* kept a unit of *score* away from the
boundary. Note that this is a unit of score, not of distance — and converting between the two is where the
objective comes from.

The distance from a point to the hyperplane $\\mathbf{w}\\cdot\\mathbf{x}+b = 0$ is
$|\\mathbf{w}\\cdot\\mathbf{x}+b| \\,/\\, \\|\\mathbf{w}\\|$: the score, divided by the length of $\\mathbf{w}$. (That
division is just the projection formula from [the vectors lesson](#/l/math-vectors) — $\\mathbf{w}$ is the
direction perpendicular to the boundary, and dividing by its length converts a dot product into a distance.)

So a point sitting at the constraint boundary with a score of exactly 1 is at distance $1/\\|\\mathbf{w}\\|$, and
the empty corridor spanning both sides is $2/\\|\\mathbf{w}\\|$ wide. **Making $\\|\\mathbf{w}\\|$ small makes the
margin wide.** That is why the objective looks like a regularization penalty: here it *is* the goal, not a
concession to overfitting.`),

    viz('svm-margin'),

    t(`Real data is never perfectly separable, so the hard constraint has to soften. Allow points to violate the
margin at a price, and the problem becomes an unconstrained one:

$$\\min_{\\mathbf{w}}\\ \\underbrace{\\tfrac{1}{2}\\|\\mathbf{w}\\|^2}_{\\text{keep the margin wide}} + C\\sum_i \\underbrace{\\max(0, 1-y_i f(\\mathbf{x}_i))}_{\\text{hinge loss: penalty for violations}}$$

Which is worth recognising for what it is: **regularized ERM with a particular loss**, exactly the shape from
[the regularization lesson](#/l/ml-regularization). The hinge loss is zero once a point is correct and beyond
the margin — being *more* correct earns nothing — and grows linearly once it is not.

$C$ sets the exchange rate. Small $C$ means violations are cheap, so the optimizer buys a wide forgiving margin
and accepts some errors: more regularization. Large $C$ means violations are expensive, forcing a narrow margin
that contorts to classify everything: less regularization.`),

    key(`Look at the hinge loss again: it is exactly zero for any point that is correctly classified and outside
the margin. Zero loss means zero gradient means **no influence on the solution**.

So only the points on or inside the margin matter. Those are the **support vectors**, and you could delete every
other training point and refit and get an identical answer. On a typical problem that might be a few percent of
your data.

That sparsity is not a curiosity — it is what makes the next section affordable. Prediction requires kernel
evaluations only against the support vectors, not against the whole training set.`),

    t(`## The kernel trick

Here is the setup. Data that is hopelessly tangled in its original space often becomes trivially separable if
you map it somewhere higher-dimensional. Two concentric rings cannot be split by any line in 2-D — but add a
third coordinate $z = x^2 + y^2$ and the inner ring drops below the outer one, separable by a flat plane.

So: define a feature map $\\phi$ that lifts your data somewhere useful, and run a linear SVM there. The obvious
objection is cost. A good $\\phi$ might map to thousands of dimensions, or infinitely many, and you cannot store
an infinite vector.

The trick starts with an observation about where the solution can possibly live. It turns out that the optimal
$\\mathbf{w}$ is always a weighted combination of the training points themselves:

$$\\mathbf{w} = \\sum_i \\alpha_i y_i \\mathbf{x}_i$$

That is not an assumption but a consequence — any component of $\\mathbf{w}$ pointing away from all the data
would increase $\\|\\mathbf{w}\\|$ without changing a single one of the scores $\\mathbf{w}\\cdot\\mathbf{x}_i$, so
the objective would simply delete it.

Now substitute that into the prediction and watch what happens:

$$f(\\mathbf{x}) = \\mathbf{w}\\cdot\\mathbf{x} + b = \\sum_i \\alpha_i y_i\\,(\\mathbf{x}_i \\cdot \\mathbf{x}) + b$$

**The data appears only inside dot products.** Individual vectors have vanished from the formula; the only thing
the algorithm ever asks about the data is "how similar are these two points?" The same is true of the training
procedure, which needs only the pairwise dot products $\\mathbf{x}_i \\cdot \\mathbf{x}_j$.

So run the whole thing in the lifted space by replacing $\\mathbf{x}_i \\cdot \\mathbf{x}_j$ with
$\\phi(\\mathbf{x}_i) \\cdot \\phi(\\mathbf{x}_j)$ — and if there is a shortcut for computing *that number*
directly, you never need $\\phi(\\mathbf{x})$ at all. The shortcut is called a **kernel**:

$$k(\\mathbf{x}_i,\\mathbf{x}_j) = \\phi(\\mathbf{x}_i)\\cdot\\phi(\\mathbf{x}_j)$$`),

    viz('kernel-trick'),

    t(`| Kernel | $k(\\mathbf{x},\\mathbf{x}')$ | Implied feature space |
|---|---|---|
| Linear | $\\mathbf{x}^{\\mathsf T}\\mathbf{x}'$ | the original |
| Polynomial | $(\\gamma\\,\\mathbf{x}^{\\mathsf T}\\mathbf{x}'+r)^d$ | all monomials up to degree $d$ |
| RBF / Gaussian | $\\exp(-\\gamma\\|\\mathbf{x}-\\mathbf{x}'\\|^2)$ | **infinite-dimensional** |

Read the last row slowly. **The RBF kernel computes a dot product in an infinite-dimensional space using one
exponential of one distance.** You get the separating power of an infinite feature map for the cost of a
subtraction, a norm, and an exponential. That is the trick in its purest form, and it is genuinely startling the
first time.

There is a sanity check hiding in the RBF formula. It equals 1 when the two points coincide and decays toward 0
as they move apart, so it behaves exactly like a similarity score — which is what a dot product between
unit-length vectors is. The parameter $\\gamma$ sets how quickly similarity falls off with distance, and that
single number is the model's whole notion of "nearby".

You are not restricted to this table. Any function whose table of pairwise values is always a valid
"similarity matrix" — technically, positive semi-definite, meaning it never assigns a negative squared length to
any combination of points — corresponds to *some* feature map, even if nobody can write it down. This is
Mercer's condition, and it is what lets people design kernels for strings, graphs, and molecules, where there is
no natural way to write the object as a vector but there is a natural way to say how similar two of them are.`),

    hist(`SVMs with RBF kernels were the state of the art from roughly 1995 to 2012, and there was a real intellectual
argument that they were the *right* answer: convex, theoretically grounded, strong generalization bounds. Then AlexNet
happened.

The structural reason kernels lost: computing the kernel matrix is $O(n^2)$ and solving is worse, so they do not scale
past ~$10^5$ samples. And a fixed kernel is a fixed similarity measure chosen in advance, while a deep network *learns*
its representation. When data got big, learned features beat designed ones. This is the same lesson as CNN-vs-ViT, one
level up.

Kernels are not gone: Gaussian processes are kernel methods and remain the tool of choice for small-data problems with
calibrated uncertainty, and the Neural Tangent Kernel connects wide networks back to kernel regression.`),

    code('kNN and a kernel SVM from scratch', `import numpy as np
rng = np.random.default_rng(1)

# concentric circles: linearly inseparable
n = 200
theta = rng.uniform(0, 2*np.pi, n)
r = np.where(np.arange(n) % 2, 1.9, 0.8) + rng.normal(0, 0.15, n)
X = np.c_[r*np.cos(theta), r*np.sin(theta)]
y = np.where(np.arange(n) % 2, 1.0, -1.0)

# --- kNN ---
def knn_predict(Xtr, ytr, Xq, k=5):
    d = ((Xq[:, None, :] - Xtr[None, :, :])**2).sum(-1)
    idx = np.argsort(d, axis=1)[:, :k]
    return np.sign(ytr[idx].sum(1))

acc = (knn_predict(X[:150], y[:150], X[150:], k=5) == y[150:]).mean()
print(f"kNN (k=5) test accuracy: {acc:.3f}")

# --- kernel SVM via projected gradient on the dual ---
def rbf(A, B, gamma=0.5):
    return np.exp(-gamma * ((A[:, None, :] - B[None, :, :])**2).sum(-1))

K = rbf(X, X)
alpha = np.zeros(n)
C = 1.0
for _ in range(3000):
    grad = 1 - y * (K @ (alpha * y))
    alpha = np.clip(alpha + 0.01 * grad, 0, C)

sv = alpha > 1e-4
print(f"\\nsupport vectors: {sv.sum()} / {n}  ({sv.mean():.0%})")

f = K @ (alpha * y)
b = np.mean(y[sv] - f[sv])
print(f"training accuracy: {(np.sign(f + b) == y).mean():.3f}")
print("\\nA linear model scores ~0.5 here. The kernel does all the work.")`,
      'Notice what the SVM code never does: it never builds a feature vector. `rbf` produces the matrix K of pairwise similarities, and from there everything — training and prediction alike — is arithmetic on that table. The two rings are not separable by any line in these two dimensions, yet the model classifies them almost perfectly, because the separating surface is flat in a space the code never visits. Also check the support-vector count: only a fraction of the points ended up mattering, and the rest could be deleted with no effect.'),

    quiz('You train an RBF-SVM with γ very large. What happens?',
      ['Each support vector influences only its immediate neighbourhood — the boundary fragments into islands and overfits',
       'The decision boundary becomes linear',
       'Training fails to converge',
       'The model underfits'],
      0,
      '$\\gamma$ sets the width of the Gaussian bump around each support vector. Large $\\gamma$ = narrow bumps = the model can only "see" points essentially on top of each other, so it memorizes. Small $\\gamma$ = wide bumps = an almost-linear boundary. $\\gamma$ and $C$ must be tuned jointly, which is why grid search over both is the standard recipe.'),

    recap(`- Explain why kNN has no training step, and what it costs you at prediction time instead.
- Say what $k$ trades off, and why 100% training accuracy at $k=1$ is meaningless.
- Describe RAG as kNN, and say which of kNN's two weaknesses embeddings fix.
- Explain why maximising the margin is the same as minimising $\\|\\mathbf{w}\\|$.
- Say what a support vector is and why deleting non-support points changes nothing.
- State the kernel trick in one sentence: the algorithm only needs inner products, so replace them.
- Give the structural reason kernel methods lost to deep learning, and name where they are still preferred.`),
  ],
  refs: [
    paper('Support-Vector Networks', 'Cortes & Vapnik', 1995, 'https://link.springer.com/article/10.1007/BF00994018', 'The paper.'),
    book('Learning with Kernels', 'Schölkopf & Smola', 2002, 'https://mitpress.mit.edu/9780262536578/learning-with-kernels/', 'The comprehensive treatment of kernel methods.'),
    blog('Support Vector Machines', 'Andrew Ng (CS229 notes)', 2020, 'https://cs229.stanford.edu/notes2022fall/main_notes.pdf', 'The clearest derivation of the dual and the kernel trick.'),
    paper('Neural Tangent Kernel', 'Jacot et al.', 2018, 'https://arxiv.org/abs/1806.07572', 'Infinitely wide networks behave like kernel regression. The bridge back.'),
    book('Gaussian Processes for Machine Learning', 'Rasmussen & Williams', 2006, 'https://gaussianprocess.org/gpml/', 'Free PDF. Kernels with calibrated uncertainty.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'ml-unsupervised',
  title: 'Clustering and Dimensionality Reduction',
  sub: 'Finding structure with no labels: k-means, GMMs, PCA, and the embedding-plot trap.',
  mins: 26, level: 'core',
  prereq: ['math-eigen-svd', 'math-probability'],
  tags: ['unsupervised', 'clustering', 'PCA'],
  sections: [
    tldr(`Everything so far needed labels. This lesson is about what you can do without them — which is a lot,
because labels are expensive and raw data is not.

Two families. **Clustering** (k-means, Gaussian mixtures) groups similar examples together. **Dimensionality
reduction** (PCA, t-SNE, UMAP) squeezes high-dimensional data down to two or three dimensions so you can look
at it.

The lesson ends with a warning that is worth the price of admission on its own: those beautiful t-SNE scatter
plots you have seen in papers are far less informative than they appear, and there is a specific list of what
you may and may not conclude from one.`),

    jargon([
      ['unsupervised learning', 'Learning structure from data with no labels. Nobody tells you the right answer, so "right" has to be defined by the method itself.'],
      ['cluster', 'A group of examples the algorithm judges similar. Note that the algorithm decides what "similar" means — that is a modelling choice, not a fact about the data.'],
      ['centroid', 'The mean position of a cluster. The "centre" k-means moves around.'],
      ['inertia', 'Total squared distance from every point to its cluster centre. What k-means minimises.'],
      ['Voronoi diagram', 'Space carved into regions by "which centre is nearest". Always produces straight-line boundaries — which caps what k-means can express.'],
      ['GMM', 'Gaussian Mixture Model. Assumes the data came from several overlapping Gaussian blobs and tries to recover them.'],
      ['EM (expectation-maximization)', 'An alternating algorithm: guess which cluster each point belongs to, refit the clusters, repeat.'],
      ['responsibility', 'In a GMM, the probability that point $i$ came from cluster $k$. A *soft* assignment — a point can be 70% one cluster and 30% another.'],
      ['PCA', 'Principal Component Analysis. Finds the directions your data varies most along, and lets you keep only the top few.'],
      ['principal component', 'One of those directions. PC1 is the direction of greatest variance, PC2 the greatest remaining, and so on.'],
      ['explained variance', 'What fraction of the data\'s total spread a component accounts for. How you decide how many to keep.'],
      ['manifold', 'A curved surface that the data lies on or near — like a sheet of paper crumpled up inside 3-D space.'],
      ['t-SNE / UMAP', 'Nonlinear methods that squash data to 2-D for visualisation, preserving *who is near whom* rather than actual distances.'],
    ]),

    t(`## k-means

The most-used clustering algorithm, and one you could reinvent from scratch. You want to split the data into
$k$ groups so that points within a group are close to each other. Formally: minimise the total squared distance
from each point to its group's centre.

Solving that exactly is NP-hard — meaning no known method beats checking essentially every way of splitting
the points, which is hopeless past a few dozen of them. So k-means alternates two easy steps instead:

1. **Assign** — put each point with whichever centre is nearest.
2. **Update** — move each centre to the mean of the points assigned to it.

Repeat until nothing changes.`),

    viz('kmeans'),

    t(`Each step provably decreases the objective, so the algorithm always converges. But converges to *a* local
optimum, not *the* optimum, and which one depends entirely on where the centres started. Press "new init" a few
times on the same data and watch the answer change — that is not a bug, that is the algorithm.

The standard mitigations: **k-means++** initialization, which places the initial centres spread far apart (and
comes with a provable approximation guarantee), plus simply running the whole thing several times and keeping
the best. Both are on by default in scikit-learn.`),

    warn(`**k-means cannot express most cluster shapes, and it will not tell you when it has failed.**

The limitation is baked into the objective. "Minimise squared distance to a centre" means each cluster is
implicitly a sphere, all the same size, with straight-line boundaries between them (a Voronoi diagram). Hand it
two elongated parallel bands, or a ring around a blob, or one cluster ten times bigger than another, and it
will return $k$ confident-looking spherical groups that carve straight through the real structure.

There is no diagnostic in the output that says "the shape assumption was wrong". Always plot the result. If your
clusters are non-spherical, look at DBSCAN (density-based, finds arbitrary shapes) or a Gaussian mixture (below,
which at least allows ellipses).`),

    t(`### Choosing $k$

The uncomfortable truth is that there is often no right answer, because "how many clusters are in this data" is
frequently not a well-posed question.

- **The elbow method** — plot inertia against $k$, look for where the curve bends. Subjective, and real data
  regularly produces a smooth curve with no elbow at all.
- **Silhouette score** — measures how much closer each point is to its own cluster than to the next nearest.
  Better behaved, gives an actual number to maximise.
- **BIC / AIC** — scores that add up how well the model explains the data and then subtract a penalty for each
  parameter it used, so a more complicated model has to earn its complexity. They are only available if you fit
  a proper probabilistic model, which is one good reason to prefer a GMM over k-means.
- **An external reason.** By far the best option. "We have shelf space for 6 customer segments" is a stronger
  justification for $k=6$ than any curve.`),

    t(`## Gaussian mixtures and EM

k-means makes a *hard* choice: this point belongs to cluster 3, full stop, even if it sits exactly between
clusters 3 and 4. A Gaussian mixture makes a *soft* one, and gets a real probabilistic model in the bargain.

The story it tells is generative: each data point was produced by first picking a cluster (with probability
$\\pi_k$), then drawing from that cluster's Gaussian. Written out:

$$p(\\mathbf{x}) = \\sum_{k=1}^K \\underbrace{\\pi_k}_{\\text{how often cluster } k} \\underbrace{\\mathcal{N}(\\mathbf{x}\\mid\\boldsymbol\\mu_k,\\Sigma_k)}_{\\text{what cluster } k \\text{ looks like}}$$

Each $\\Sigma_k$ is a full covariance matrix, so a cluster can be an ellipse at any orientation rather than
only a sphere.

There is a chicken-and-egg problem in fitting this: if you knew which cluster each point came from you could
fit the Gaussians easily, and if you knew the Gaussians you could assign the points easily. **Expectation-
maximization** cuts the knot by alternating:

- **E-step** — with the current Gaussians, compute each point's *responsibility* $\\gamma_{ik} =
  p(z_i=k\\mid\\mathbf{x}_i)$: the probability it came from cluster $k$. A soft assignment; a boundary point
  might be 0.6 and 0.4.
- **M-step** — refit each Gaussian's mean and covariance, weighting every point by its responsibility. A point
  that is 60% yours contributes 60% to your mean.

Repeat. EM's guarantee is that the log-likelihood increases every iteration and never decreases.`),

    viz('gmm-em'),

    key(`**k-means is GMM with hard assignments and shared spherical covariance.** Going soft buys you: clusters with
shape and orientation, honest uncertainty at boundaries, and a proper likelihood — which means you can use AIC/BIC to
choose $K$ rather than squinting at an elbow.

EM's guarantee is that the log-likelihood increases on every iteration and never decreases — which is why it always
converges, though again to a local optimum rather than the best one. Keep the shape of the algorithm in mind: guess the
hidden assignments, refit the model, repeat. It comes back in [variational autoencoders](#/l/gen-autoencoders), where
the guessing step is replaced by a neural network trained to do it in one shot.`),

    t(`## PCA

Now the other half of the lesson: not grouping the examples, but shrinking the *features*.

PCA can be defined two ways, and the fact that they turn out to be the same thing is the heart of it:

- **Maximise variance.** Find the direction along which the data spreads out most. Then the next such
  direction, perpendicular to the first. And so on.
- **Minimise reconstruction error.** Find the $k$-dimensional flat subspace that the data sits closest to, so
  that projecting onto it loses as little as possible.

These sound like different goals, and they are provably the same goal. The reason is the
[projection split](#/l/math-vectors) from the first lesson: every centred data point breaks into the part its
subspace explains plus an orthogonal residual, and because those two pieces meet at a right angle, Pythagoras
applies:

$$\\underbrace{\\|\\mathbf{x}_i\\|^2}_{\\text{fixed by the data}} = \\underbrace{\\|\\text{projection}\\|^2}_{\\text{captured}} + \\underbrace{\\|\\text{residual}\\|^2}_{\\text{lost}}$$

Add that up over all the points. The left side does not depend on which subspace you chose, so it is a constant.
Therefore making the captured term as large as possible is *exactly* making the lost term as small as possible —
they are two ways of describing one split of a fixed total. Maximising variance and minimising reconstruction
error are not two goals that happen to agree; they are the same equation read from opposite ends.`),

    viz('pca'),

    t(`The recipe is three lines: centre the data, take the [SVD](#/l/math-eigen-svd) of $X$, keep the top $k$
right singular vectors. Those are your components. The squared singular values tell you how much variance each
one captured, which is how you decide where to cut.

This should look familiar — it is Eckart–Young truncation applied to the data matrix. PCA is low-rank
approximation wearing a statistician's hat.

Four things to keep straight, in decreasing order of how often they go wrong:

- **Centring is mandatory.** Subtract the mean of every feature first. Skip it and PC1 will simply point from
  the origin toward the centre of your data, capturing the mean rather than any variation. Libraries do this
  for you; hand-rolled implementations often do not.
- **Scaling matters, and silently.** PCA maximises *numerical* variance, so a feature measured in millimetres
  swamps one measured in metres purely through units. Standardize every feature unless the units are genuinely
  comparable (all pixel intensities, say).
- **PCA is linear.** Data lying on a curved manifold — a spiral, a Swiss roll — will not unroll. PCA can only
  ever rotate and drop axes; it cannot bend. That limitation is precisely what the next section's methods exist
  to work around.
- **Components are orthogonal by construction, and therefore often physically meaningless.** Nothing forces
  reality's underlying factors to be perpendicular to each other, so "PC2 represents X" is usually a story you
  are telling rather than something the method found. If you need interpretable factors, use a method that
  targets them (ICA, NMF, sparse coding).`),

    t(`## t-SNE and UMAP: read the fine print

PCA is linear, so it cannot unroll a curved manifold. t-SNE and UMAP can, by giving up on preserving distances
and preserving **neighbourhoods** instead: they try to keep each point's nearest neighbours nearby in the 2-D
picture, and are indifferent to everything else.

That is a reasonable trade for visualisation, and it produces genuinely beautiful plots. It also means most of
what people read off those plots is not supported by the method. The table below is the one to internalise —
and it applies to every embedding plot you have seen in a paper.

| Feature of the plot | Trustworthy? |
|---|---|
| Points that are close together | Usually yes — local structure is what is preserved |
| Distances **between** clusters | **No.** Meaningless in t-SNE, only weakly meaningful in UMAP |
| Relative cluster **sizes** | **No.** t-SNE expands dense clusters and contracts sparse ones |
| The number of clusters | **Be careful.** Perplexity/n_neighbors strongly affects apparent cluster count |
| Clusters appearing at all | **Not on its own.** t-SNE will produce clusters from pure noise |

Always run these at several perplexity settings before believing anything. And if you need a plot where distances mean
something, use PCA.`),

    code('k-means, GMM, and PCA', `import numpy as np
rng = np.random.default_rng(2)

# three genuine clusters, one of them elongated
X = np.vstack([
    rng.normal([0, 0],  [0.4, 0.4], (120, 2)),
    rng.normal([3, 3],  [0.4, 0.4], (120, 2)),
    rng.normal([0, 3],  [1.6, 0.25], (120, 2)),      # <- elongated
])

def kmeans(X, k, seed, iters=60):
    rs = np.random.default_rng(seed)
    C = X[rs.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None] - C[None])**2).sum(-1)
        a = d.argmin(1)
        C = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
    inertia = ((X - C[a])**2).sum()
    return C, a, inertia

print("k-means from 5 different initializations (same data, k=3):")
for seed in range(5):
    _, _, inertia = kmeans(X, 3, seed)
    print(f"  seed {seed}: inertia = {inertia:.1f}")
print("  ^ different local optima. Always use multiple restarts.\\n")

print("elbow plot data:")
for k in range(1, 7):
    best = min(kmeans(X, k, s)[2] for s in range(5))
    print(f"  k={k}: inertia = {best:8.1f}")

# --- PCA ---
Xc = X - X.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
var = S**2 / len(X)
print(f"\\nexplained variance: {(var/var.sum()).round(3)}")
print(f"PC1 direction: {Vt[0].round(3)}")

# reconstruction error from keeping only PC1
recon = (Xc @ Vt[0][:, None]) @ Vt[0][None, :]
print(f"1-component reconstruction MSE: {((Xc - recon)**2).mean():.4f}")`,
      'The five seeds give five different inertias on identical data, which is the local-optimum problem made concrete — and note that you would never have known if you had only run it once. The elbow numbers fall steeply to k=3 and then flatten, which is what an elbow looks like when the data actually has an answer; on real data it usually does not look this cooperative. The PCA block closes the loop with the SVD lesson: the explained-variance figures are the squared singular values, normalised.'),

    quiz('Your t-SNE plot shows two clusters far apart and one nearby pair. What can you conclude about the underlying data?',
      ['Only that there is local neighbourhood structure — inter-cluster distances in t-SNE carry no reliable meaning',
       'The two distant clusters are more different from each other than the nearby pair',
       'There are exactly three groups in the data',
       'The data is three-dimensional'],
      0,
      't-SNE optimizes a KL objective over pairwise neighbour probabilities, which pins down local structure and leaves global layout largely arbitrary. Distances between clusters are an artifact of the optimization, not a property of your data. Run it at several perplexities: the global arrangement will move while local groupings stay put, which tells you exactly which parts to trust.'),

    recap(`- Describe k-means as two alternating steps, and say why it converges but not to the best answer.
- Name the cluster shapes k-means structurally cannot find, and say what to reach for instead.
- Explain the difference between hard and soft assignment, and one concrete thing soft assignment buys you.
- Give both definitions of PCA — maximise variance, minimise reconstruction error — and say why they coincide.
- List the two preprocessing steps PCA requires and what goes wrong without each.
- State, from the table, exactly which features of a t-SNE plot you are allowed to interpret.`),
  ],
  refs: [
    blog('How to Use t-SNE Effectively', 'Wattenberg, Viégas & Johnson', 2016, 'https://distill.pub/2016/misread-tsne/', 'Interactive, and required reading before you show anyone a t-SNE plot.'),
    paper('Visualizing Data using t-SNE', 'van der Maaten & Hinton', 2008, 'https://www.jmlr.org/papers/v9/vandermaaten08a.html', ''),
    paper('UMAP: Uniform Manifold Approximation and Projection', 'McInnes et al.', 2018, 'https://arxiv.org/abs/1802.03426', 'Faster than t-SNE, and preserves somewhat more global structure.'),
    paper('k-means++: The Advantages of Careful Seeding', 'Arthur & Vassilvitskii', 2007, 'https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf', ''),
    book('Pattern Recognition and Machine Learning, Ch. 9', 'Christopher Bishop', 2006, 'https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/', 'The definitive EM and GMM chapter.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'ml-evaluation',
  title: 'Evaluation: Metrics, Validation, and Self-Deception',
  sub: 'The part everyone rushes, and the reason most reported results do not hold up.',
  mins: 24, level: 'core',
  tags: ['evaluation', 'metrics'],
  sections: [
    tldr(`This is the lesson people skim, and it is the reason most reported results do not survive contact
with reality.

Two themes. First, **accuracy is usually the wrong number** — on a problem with 1% positives, a model that
always says "no" scores 99%. You need precision, recall, and an explicit decision about which errors you can
afford. Second, **the ways you can accidentally lie to yourself are systematic and enumerable**, and they mostly
come down to information from your test set leaking into a decision you made.

Neither theme is technically difficult. Both are routinely got wrong in published work.`),

    jargon([
      ['confusion matrix', 'The 2×2 table of predicted-versus-actual counts. Every classification metric is a ratio computed from these four numbers.'],
      ['TP / FP / TN / FN', 'True Positive (correctly flagged), False Positive (false alarm), True Negative, False Negative (a miss).'],
      ['precision', 'Of everything you flagged, what fraction was right. Answers "can I trust an alert?"'],
      ['recall (a.k.a. sensitivity, TPR)', 'Of everything that was really positive, what fraction you caught. Answers "how much am I missing?"'],
      ['F1', 'The harmonic mean of precision and recall — a single number that summarises both, at the cost of hiding which one you sacrificed.'],
      ['prevalence / base rate', 'How common the positive class actually is. Changes what precision means without changing the model at all.'],
      ['ROC curve', 'Plots recall against false-alarm rate as you sweep the threshold. Shows every operating point at once.'],
      ['AUC', 'Area under the ROC curve. Equals the probability a random positive scores higher than a random negative.'],
      ['operating point', 'One particular threshold choice — one point on the curve. A model has one curve but infinitely many operating points.'],
      ['cross-validation', 'Splitting the data into $k$ folds and training $k$ times, each time holding out a different fold. Gives you a spread, not just a number.'],
      ['leakage', 'Information from outside the training fold sneaking in. Produces excellent validation scores and a model that fails in production.'],
      ['baseline', 'The simplest thing that could work — majority class, a linear model. What your fancy model must beat to have earned anything.'],
    ]),

    t(`## Accuracy is usually the wrong metric

Start with the failure. You are building a fraud detector, and 1% of transactions are fraudulent. Here is a
model:

    def predict(transaction):
        return "not fraud"

It is 99% accurate. It is also worthless, and no amount of accuracy reporting will reveal that. Accuracy sums
over both classes, so a rare class simply cannot move the number.

Everything better starts from the **confusion matrix** — the four ways a binary prediction can turn out:

| | Predicted + | Predicted − |
|---|---|---|
| **Actual +** | TP (caught it) | FN (missed it) |
| **Actual −** | FP (false alarm) | TN (correctly ignored) |

From those four numbers:

- **Precision** $= \\frac{TP}{TP+FP}$ — of the ones you flagged, how many were real? *This is the cost of false
  alarms.* Low precision means your alerts are noise and people stop reading them.
- **Recall** $= \\frac{TP}{TP+FN}$ — of the real ones, how many did you catch? *This is the cost of misses.*
  Low recall means things slip through.
- **F1** $= \\frac{2\\,\\text{precision}\\cdot\\text{recall}}{\\text{precision}+\\text{recall}}$ — the harmonic mean of
  the two, which is a kind of average that sits much closer to the smaller number than the ordinary average
  does. Precision 0.9 and recall 0.1 average to 0.5 but give an F1 of 0.18. That is deliberate: it stops a model
  from scoring well by being excellent at one and useless at the other. It is convenient for leaderboards, and
  it papers over exactly the tradeoff you should be making on purpose.
- **Specificity** $= \\frac{TN}{TN+FP}$ — the mirror image of recall, for the negative class.

Notice that precision and recall trade off directly: lower your threshold and you catch more (recall up) at the
cost of more false alarms (precision down). There is no setting that maximises both, so **which one matters is
a question about your application, not your model**.

Cancer screening wants recall — a missed tumour is catastrophic, a false alarm costs one follow-up scan. Spam
filtering wants precision — a spam message reaching the inbox is an annoyance, a real email silently deleted is
a disaster. Same mathematics, opposite answers.`),

    viz('roc-curve'),

    t(`## ROC vs precision-recall

Both are ways of showing a model's behaviour at *every* threshold at once, rather than committing to one. The
difference between them is subtle and matters enormously on rare-event problems.

**ROC** plots recall against false-alarm rate as the threshold sweeps from 0 to 1. **AUC** — the area under it
— has a clean interpretation: the probability that a randomly chosen positive scores higher than a randomly
chosen negative. It measures pure ranking ability.

Crucially, ROC-AUC is **prevalence-independent**. Both its axes are normalised within a class, so changing how
rare the positive class is does not move the curve. That is a feature when comparing models across datasets and
a trap when you want to know whether your alerts will be useful.

**Precision-recall** curves *do* depend on prevalence, because precision has false positives from the (huge)
negative class in its denominator. That dependence is exactly why they are the honest choice for rare events.`),

    key(`Drag prevalence down to 2% in the figure and watch what happens: **AUC barely moves while precision
collapses.**

The arithmetic behind that: with 1% positives and a model at 90% recall and a 5% false-alarm rate, out of 10,000
transactions you catch 90 of the 100 frauds — and also flag 495 legitimate ones. Precision is 90/585 = **15%**.
Five out of six alerts are wrong, and your ROC-AUC still looks excellent.

If your positive class is rare, **report PR-AUC (average precision)**, and report the precision at the recall
you actually plan to operate at. Reporting ROC-AUC alone on an imbalanced problem is technically true and
practically misleading.

And a related discipline: a model has one curve but infinitely many operating points. **Reporting a single F1
without saying which threshold produced it is close to meaningless** — someone else cannot reproduce it and you
cannot tell whether it was tuned.`),

    t(`## Regression metrics

| Metric | Behavior |
|---|---|
| MSE | Punishes large errors quadratically. Outlier-sensitive. |
| RMSE | Same, in the units of $y$. |
| MAE | Linear penalty. Robust. Optimal predictor is the **median**, not the mean. |
| MAPE | Percentage error. Explodes near $y=0$, and asymmetric (under-prediction is capped at 100%). |
| $R^2$ | Fraction of variance explained. Can be negative on held-out data. |

Choose by asking what a doubled error costs you. If it costs exactly twice as much, use MAE.`),

    t(`## Validation done right

- **k-fold CV** for i.i.d. data. $k=5$ or $10$.
- **Stratified** folds for imbalanced classes, so each fold has the same class ratio.
- **Grouped** folds when samples cluster (multiple readings per patient, multiple photos per user). Otherwise the same
  entity appears in train and validation, and you are measuring memorization.
- **Time-series split** for temporal data: always train on the past, validate on the future. Random splits leak the
  future and produce beautiful, worthless numbers.
- **Nested CV** when you tune hyperparameters *and* want an unbiased performance estimate. The outer loop measures,
  the inner loop tunes.`),

    viz('cross-validation'),

    warn(`**Ways to fool yourself, roughly in order of frequency:**

1. **Preprocessing before splitting.** Fitting a scaler, imputer, or feature selector on the full dataset leaks test
   statistics into training. Fit inside the fold.
2. **Threshold tuned on test.** Pick it on validation.
3. **Repeated peeking.** Fifty experiments evaluated on the test set means your best result is the max of fifty noisy
   draws.
4. **Group leakage.** See above.
5. **No confidence interval.** With $n=200$ test samples, 87% and 84% accuracy are not distinguishable. A rough 95% CI
   for a proportion is $\\pm 1.96\\sqrt{p(1-p)/n}$ — for $p=0.85, n=200$ that is $\\pm 5\\%$.
6. **No baseline.** Always report the majority-class rate, and a linear model or gradient boosting baseline. A
   surprising fraction of published deep-learning results do not beat a well-tuned baseline.`),

    t(`## Statistical significance, briefly

Compare two models on the *same* test set with a **paired** test — one that looks at the examples where the two
models disagreed, rather than at their two overall scores. McNemar's test does this for classification: it
ignores every example both models got right and both got wrong, and asks only whether the disagreements lean
one way more than chance would explain. A paired bootstrap works for any metric.

Why bother with pairing? Because most of the variation in a test score comes from which examples happened to be
in the test set, and *both* models faced the same examples. Comparing two independent accuracy numbers throws
that shared information away and makes it much harder to detect a real difference.

And report **variance across seeds**. A single training run's number is a sample from a distribution, and for
reinforcement learning and small-data problems that distribution is alarmingly wide.`),

    code('Metrics, bootstrap CIs, and a leakage demonstration', `import numpy as np
rng = np.random.default_rng(0)

# imbalanced problem: 2% positives
n = 5000
y = (rng.random(n) < 0.02).astype(int)
score = rng.normal(y * 1.8, 1.0)              # a mediocre model

def metrics(y, score, thr):
    pred = (score > thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(acc=(tp+tn)/n, precision=prec, recall=rec,
                f1=2*prec*rec/max(prec+rec, 1e-9))

print("threshold sweep (2% positive rate):")
for thr in [0.0, 1.0, 2.0, 3.0]:
    m = metrics(y, score, thr)
    print(f"  thr={thr}: acc={m['acc']:.3f} prec={m['precision']:.3f} "
          f"rec={m['recall']:.3f} f1={m['f1']:.3f}")
print(f"  always-negative baseline accuracy: {1-y.mean():.3f}\\n")

# bootstrap confidence interval on accuracy
acc_point = metrics(y, score, 1.0)['acc']
boots = [metrics(y[i], score[i], 1.0)['acc']
         for i in (rng.integers(0, n, (400, n)))]
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"accuracy {acc_point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]\\n")

# --- leakage: standardizing before the split ---
X = rng.normal(size=(200, 500))               # pure noise, no signal at all
target = rng.integers(0, 2, 200)
# WRONG: select features using all the data
corr = np.abs([np.corrcoef(X[:, j], target)[0, 1] for j in range(500)])
top = np.argsort(-corr)[:10]
Xs = X[:, top]
tr, te = slice(0, 150), slice(150, 200)
w = np.linalg.lstsq(Xs[tr], target[tr]*2-1, rcond=None)[0]
acc_leaky = ((Xs[te] @ w > 0) == target[te]).mean()
print(f"'accuracy' after selecting features on ALL data: {acc_leaky:.3f}")
print("There is no signal. Any number above 0.5 here is pure leakage.")`,
      'Three things worth pausing on. In the sweep, raising the threshold pushes accuracy up and recall down — and the always-say-no baseline beats several of the thresholds outright, which is the point. The bootstrap interval shows how wide the uncertainty on a single accuracy number really is. And the last block is leakage in its purest form: `X` is random noise with no relationship to the target whatsoever, but picking the ten features that correlate best *using all 200 rows* smuggles information about the test rows into the choice, and the reported score comes out above chance. Nothing in the code looks wrong, which is why this one catches people.'),

    quiz('A fraud model has 99.5% accuracy on a dataset with 0.5% fraud. Your next question should be:',
      ['What are precision and recall? It may be predicting "not fraud" for everything',
       'How can we push accuracy to 99.9%?',
       'What is the training time?',
       'Nothing — 99.5% is excellent'],
      0,
      'Predicting the majority class always gives 99.5% here. The model may have learned literally nothing. Ask for precision and recall at the operating threshold, PR-AUC across thresholds, and the confusion matrix in raw counts. Accuracy on an imbalanced problem is close to uninformative.'),

    recap(`- Explain why accuracy fails on imbalanced data, with the one-line model that proves it.
- Define precision and recall in words a non-specialist would follow, and say which one a given application
  needs.
- Say why ROC-AUC can look excellent while precision is terrible, and what to report instead.
- Pick the right cross-validation scheme for grouped data, time-series data, and imbalanced data.
- Name the six standard ways to fool yourself, and identify which one is present in a described workflow.
- Put a rough confidence interval on an accuracy number, and use it to say whether two models actually differ.`),
  ],
  refs: [
    paper('The Relationship Between Precision-Recall and ROC Curves', 'Davis & Goadrich', 2006, 'https://www.biostat.wisc.edu/~page/rocpr.pdf', 'Why PR curves are the right choice for rare classes.'),
    blog('Machine Learning Yearning', 'Andrew Ng', 2018, 'https://info.deeplearning.ai/machine-learning-yearning-book', 'Free. About how to set up evaluation and iterate, not about algorithms.'),
    paper('Approximate is Better than "Exact" for Interval Estimation', 'Agresti & Coull', 1998, 'https://doi.org/10.1080/00031305.1998.10480550', 'How to actually put a CI on an accuracy.'),
    paper('Deep Reinforcement Learning that Matters', 'Henderson et al.', 2017, 'https://arxiv.org/abs/1709.06560', 'A devastating look at seed variance and evaluation practice.'),
    paper('Leakage and the reproducibility crisis in ML-based science', 'Kapoor & Narayanan', 2023, 'https://arxiv.org/abs/2207.07048', 'Surveys leakage across 17 scientific fields. Sobering.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'ml-generative-discriminative',
  title: 'Generative vs Discriminative Models',
  sub: 'Model p(x,y) or model p(y|x)? The choice shapes everything downstream.',
  mins: 18, level: 'core',
  prereq: ['ml-logistic'],
  tags: ['naive bayes', 'theory'],
  sections: [
    tldr(`Two philosophies for building a classifier.

**Discriminative**: learn only what you need to tell the classes apart. Where is the boundary? Nothing else
matters.

**Generative**: learn what each class *looks like*, then classify by asking which class more plausibly produced
this example.

The classical version of this debate is a small technical point about sample efficiency. The modern version is
the entire reason language models can do tasks nobody trained them on — a model that learned to generate text
can be *asked* to classify, while a classifier can only ever classify.`),

    jargon([
      ['discriminative model', 'Models $p(y \\mid \\mathbf{x})$ — the answer given the input. Or sometimes just a boundary, with no probabilities at all.'],
      ['generative model', 'Models $p(\\mathbf{x}, y)$ or $p(\\mathbf{x})$ — how the data itself is distributed. Can therefore *produce* new examples.'],
      ['joint distribution $p(\\mathbf{x}, y)$', 'The probability of an input and its label occurring together. Contains strictly more information than $p(y\\mid\\mathbf{x})$.'],
      ['class-conditional $p(\\mathbf{x} \\mid y)$', '"What do examples of class $y$ look like?" The thing a generative classifier learns.'],
      ['naive Bayes', 'A generative classifier that assumes all features are independent given the class. The assumption is false and it works anyway.'],
      ['asymptotic error', 'The error a method converges to given unlimited data. Its ceiling.'],
      ['sample efficiency', 'How much data a method needs to reach its ceiling. A method can have a worse ceiling but reach it far sooner.'],
      ['zero-shot / few-shot', 'Performing a task with no training examples, or a handful shown in the prompt. Only generative models can really do this.'],
    ]),

    t(`## Two strategies

Suppose you want to tell cats from dogs. There are two fundamentally different ways to approach it.

**The discriminative approach**: learn the *difference*. Find whatever feature separates them — ear shape, say
— and draw a boundary. You need not know anything else about cats. Logistic regression, SVMs, and standard
neural classifiers all do this, modelling $p(y\\mid\\mathbf{x})$ directly (or, for an SVM, not even that — just
the boundary).

**The generative approach**: learn what cats look like and what dogs look like, separately. Then for a new
photo, ask which model finds it less surprising. Formally, learn the class-conditionals $p(\\mathbf{x}\\mid y)$
and the class frequencies $p(y)$, then apply Bayes' rule:

$$p(y\\mid\\mathbf{x}) \\propto p(\\mathbf{x}\\mid y)\\,p(y)$$

Naive Bayes, linear discriminant analysis (which fits one Gaussian per class and shares a single covariance
between them), and Gaussian mixture classifiers are the classical examples. VAEs, diffusion models, and
language models are the modern ones.

The generative approach is doing strictly more work — it learns enough to *generate* new cats, which
classification never required. Whether that extra work is wasted effort or the whole point is what the rest of
this lesson is about.`),

    viz('generative-discriminative'),

    t(`## The tradeoff

Ng & Jordan's 2001 analysis made the comparison precise for the naive Bayes / logistic regression pair:

- **The generative model has a worse ceiling.** Its assumptions about how the data was produced are usually
  wrong, so even with unlimited data it converges to a higher error rate than the discriminative model does.
- **But it reaches that ceiling far sooner.** Roughly, naive Bayes needs a number of examples growing like
  $\\log d$ in the number of features, while logistic regression needs a number growing like $d$ itself. With
  1000 features that is the difference between dozens of examples and thousands.

So the two accuracy curves cross: generative wins on small data, discriminative wins once data is plentiful.

The reason is the same [inductive bias](#/l/ml-framing) tradeoff as before, in a new costume. Modelling
$p(\\mathbf{x}\\mid y)$ means committing to strong assumptions about what each class looks like. Those assumptions
are wrong, which caps your accuracy — and they are also *information you did not have to learn from data*, which
is exactly what you need when there is barely any data to learn from.`),

    t(`## Naive Bayes

Assume features are conditionally independent given the class:

$$p(\\mathbf{x}\\mid y) = \\prod_{j=1}^d p(x_j\\mid y)$$

This assumption is essentially always false — and naive Bayes works anyway, especially for text. The reason is that
for *classification* you only need the argmax to be right, not the probabilities. Correlated features cause
systematically overconfident probability estimates while often leaving the ranking intact.

It remains a genuinely good baseline for text: it trains in one pass, needs almost no data, and handles high
dimensions gracefully.`),

    t(`## The modern reading

For decades this was a fairly dry technical debate about sample efficiency, and discriminative methods mostly
won because data got cheap. Then the distinction became the most important idea in the field:

- A **language model is a generative model** of text, $p(x_1,\\ldots,x_T)$. Its ability to do classification is a side
  effect ("Q: Is this positive or negative? A:") of having modeled the joint.
- That is exactly why LLMs do zero-shot and few-shot tasks: a generative model over everything can be conditioned into
  any discriminative task. A discriminative sentiment classifier can only do sentiment.
- **Diffusion models** are generative models of images; classifier-free guidance is a way to steer them with
  conditional information.
- Generative pretraining followed by discriminative fine-tuning is the dominant paradigm across every modality. Learn
  the joint at scale, then specialize.`),

    intuition(`Feynman's line — "What I cannot create, I do not understand" — is the case for generative modeling, and
it has largely been borne out. A model that can *generate* plausible text must have learned syntax, semantics, world
knowledge, and reasoning patterns as a byproduct. A model trained only to discriminate learns just enough to separate
the classes in front of it, and nothing more.`),

    code('What naive Bayes\'s assumption actually costs', `import numpy as np
rng = np.random.default_rng(0)

d = 60
delta = rng.normal(0, 1, d); delta = 2.2 * delta / np.linalg.norm(delta)

# two worlds: one where the independence assumption is TRUE, one where it is not
independent = np.eye(d)
correlated  = rng.normal(size=(d, d))/np.sqrt(d) + 0.5*np.eye(d)   # mixes the features

def make(n, mix):
    y = rng.integers(0, 2, n)
    return (rng.normal(size=(n, d)) + y[:, None]*delta) @ mix, y

def naive_bayes(Xtr, ytr, Xte):
    m0, s0 = Xtr[ytr==0].mean(0), Xtr[ytr==0].std(0) + 1e-6
    m1, s1 = Xtr[ytr==1].mean(0), Xtr[ytr==1].std(0) + 1e-6
    ll0 = -(((Xte-m0)**2)/(2*s0**2) + np.log(s0)).sum(1)     # treats features as independent
    ll1 = -(((Xte-m1)**2)/(2*s1**2) + np.log(s1)).sum(1)
    return (ll1 > ll0).astype(int)

def logistic(Xtr, ytr, Xte, steps=4000):
    Xb = np.c_[np.ones(len(Xtr)), Xtr]; w = np.zeros(d + 1)
    for _ in range(steps):
        p = 1/(1+np.exp(-np.clip(Xb @ w, -500, 500)))
        w = w - 0.5 * (Xb.T @ (p - ytr)/len(ytr) + 0.01*np.r_[0, w[1:]])
    return (np.c_[np.ones(len(Xte)), Xte] @ w > 0).astype(int)

for label, mix in [("features really ARE independent", independent),
                   ("features are correlated (assumption false)", correlated)]:
    Xte, yte = make(4000, mix)
    print(label)
    print(f"{'n':>7} {'naive Bayes':>13} {'logistic':>10}")
    for n in [30, 120, 1000, 3000, 10000]:
        Xtr, ytr = make(n, mix)
        nb = (naive_bayes(Xtr, ytr, Xte) == yte).mean()
        lr = (logistic(Xtr, ytr, Xte) == yte).mean()
        print(f"{n:7d} {nb:13.3f} {lr:10.3f}")
    print()`,
      'The two tables are the lesson. In the first, where the independence assumption happens to be true, naive Bayes matches logistic regression all the way out and even leads in the middle — the generative model is the *correct* model there, so it loses nothing. In the second, the only change is that the features have been mixed together, and naive Bayes flattens out around 0.80 while logistic regression keeps climbing past 0.86. Nothing about the method changed; only whether its assumption held. That is what "higher asymptotic error" means concretely: not a weaker algorithm, but a commitment made in advance that the data did not honour.'),

    quiz('Why can a language model perform sentiment classification it was never trained on?',
      ['It models the joint distribution over text, so it can be conditioned into any task expressible in text',
       'It was secretly trained on sentiment labels',
       'Because it is a discriminative model',
       'It cannot — that requires fine-tuning'],
      0,
      'A generative model of $p(\\text{text})$ implicitly contains $p(\\text{label}\\mid\\text{review})$ for any labeling scheme that appears in natural language, because "This movie was great. Sentiment: positive" is itself text. Prompting is conditioning. This is the practical payoff of generative modeling and the reason the field moved decisively toward it.'),

    recap(`- State the difference between modelling $p(y\\mid\\mathbf{x})$ and $p(\\mathbf{x}, y)$, and give an example
  of each.
- Explain why generative models win on small data and lose on large data, in terms of assumptions acting as a
  prior.
- Say why naive Bayes works despite its central assumption being false.
- Explain, in one sentence, why a language model can do a task it was never trained on — and why a
  discriminative sentiment classifier cannot.
- Describe the "generative pretraining, discriminative fine-tuning" recipe and why it dominates every modality.`),
  ],
  refs: [
    paper('On Discriminative vs. Generative Classifiers', 'Ng & Jordan', 2001, 'https://papers.nips.cc/paper/2020-on-discriminative-vs-generative-classifiers-a-comparison-of-logistic-regression-and-naive-bayes', 'The paper that made the tradeoff precise.'),
    paper('Language Models are Few-Shot Learners', 'Brown et al.', 2020, 'https://arxiv.org/abs/2005.14165', 'GPT-3. The clearest demonstration of generative-model-as-universal-task-solver.'),
    book('Machine Learning: A Probabilistic Perspective, Ch. 3–4', 'Kevin Murphy', 2012, 'https://probml.github.io/pml-book/book0.html', ''),
  ],
},

];

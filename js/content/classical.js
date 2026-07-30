/* ============================================================
   Track 2 — Classical machine learning
   ============================================================ */

import { t, key, intuition, warn, hist, mathnote, viz, deriv, code, quiz, paper, book, course, blog, video, demo, codeRef } from './_helpers.js';

export default [

/* ---------------------------------------------------------- */
{
  id: 'ml-framing',
  title: 'What Learning Means',
  sub: 'Risk, empirical risk, and the assumption that makes any of this possible.',
  mins: 20, level: 'core',
  tags: ['theory', 'generalization'],
  sections: [
    t(`## The setup, formally

There is a distribution $\\mathcal{D}$ over pairs $(\\mathbf{x}, y)$. You never see it. You see $n$ samples drawn from
it. You want a function $f$ that predicts $y$ from $\\mathbf{x}$ on **future** samples from the same $\\mathcal{D}$.

Define a loss $\\ell(f(\\mathbf{x}), y)$. The thing you actually care about is the **true risk**:

$$R(f) = \\mathbb{E}_{(\\mathbf{x},y)\\sim\\mathcal{D}}\\big[\\ell(f(\\mathbf{x}), y)\\big]$$

You cannot compute it. What you can compute is the **empirical risk** on your sample:

$$\\hat R(f) = \\frac{1}{n}\\sum_{i=1}^{n} \\ell(f(\\mathbf{x}_i), y_i)$$

and minimizing that is called **empirical risk minimization** (ERM). Essentially all supervised learning is ERM plus
some form of capacity control.`),

    key(`The entire discipline hinges on one question: **when does small $\\hat R$ imply small $R$?**

Two assumptions are doing all the work, and both fail in the real world more often than people admit:

1. **i.i.d.** — training and test data come from the same distribution, independently. Broken by distribution shift,
   temporal drift, selection bias, and by test sets that leaked into training.
2. **Limited capacity** — your model cannot fit arbitrary noise. Broken, spectacularly, by modern overparameterized
   networks, which *can* fit random labels and yet still generalize. Explaining that is an open problem.`),

    t(`## The taxonomy, briefly

- **Supervised** — labeled pairs. Classification (discrete $y$) or regression (continuous $y$).
- **Unsupervised** — no labels. Clustering, density estimation, dimensionality reduction.
- **Self-supervised** — labels manufactured from the data itself ("predict the next token", "predict the masked
  patch"). This is how essentially every frontier model is pretrained, and it is the reason scale became possible:
  the internet is a free, enormous labeled dataset if your labels come from the data.
- **Reinforcement** — learning from reward rather than answers.`),

    t(`## No free lunch, and why it does not paralyze us

The No Free Lunch theorem says that averaged over *all possible* target functions, every learning algorithm performs
identically. Which sounds devastating until you notice the premise: we do not care about all possible functions. Real
data has structure — smoothness, locality, compositionality, hierarchy — and an algorithm wins by encoding assumptions
that match that structure.

Those assumptions are called **inductive bias**, and choosing them is the actual craft:

| Model | Inductive bias |
|---|---|
| Linear regression | The relationship is linear |
| kNN | Nearby points have similar labels |
| CNN | Features are local and translation-equivariant |
| RNN | The data has sequential, Markov-ish structure |
| Transformer | Any position may depend on any other; learn which |
| Diffusion model | Data lies near a low-dimensional manifold reachable by denoising |`),

    intuition(`A useful reframing: **more inductive bias = less data needed, but a lower ceiling if the bias is wrong.**
CNNs beat ViTs on small image datasets because their bias is correct and free. ViTs beat CNNs on huge datasets because
they can *learn* a better bias than we hand-designed. This tradeoff — hand-coded structure versus learned structure —
is the recurring theme of the last decade, and scale has consistently favored the latter.`),

    t(`## Train, validation, test

Three splits, three distinct jobs:

- **Train** — fit parameters.
- **Validation** — choose hyperparameters, architecture, when to stop.
- **Test** — estimate true risk. **Touch once.**

Every time you look at the test set and change something, you leak a bit of it into your model. Do it enough and your
test score becomes a training score wearing a disguise. This is the most common methodological failure in applied ML,
and it is entirely self-inflicted.`),

    viz('cross-validation'),

    code('ERM in twelve lines', `import numpy as np

rng = np.random.default_rng(0)

# ground truth we are trying to recover (we never get to see this)
def true_f(x): return 1.5 * x - 0.4

n = 40
X = rng.uniform(-3, 3, size=n)
y = true_f(X) + rng.normal(0, 0.5, size=n)          # noisy observations

def empirical_risk(w, b, X, y):
    return np.mean((w * X + b - y) ** 2)

# minimize it by brute force over a grid, just to make the idea concrete
ws = np.linspace(0, 3, 300); bs = np.linspace(-2, 2, 300)
W, B = np.meshgrid(ws, bs)
R = np.array([[empirical_risk(w, b, X, y) for w in ws] for b in bs])
i, j = np.unravel_index(R.argmin(), R.shape)
print(f"ERM solution:   w={ws[j]:.3f}  b={bs[i]:.3f}")
print(f"truth:          w=1.500  b=-0.400")

# true risk, estimable only because we cheated and know the generator
Xte = rng.uniform(-3, 3, 100_000)
yte = true_f(Xte) + rng.normal(0, 0.5, 100_000)
print(f"\\nempirical risk (train, n={n}): {empirical_risk(ws[j], bs[i], X, y):.4f}")
print(f"true risk      (n=100000):    {empirical_risk(ws[j], bs[i], Xte, yte):.4f}")
print(f"irreducible noise floor:      {0.5**2:.4f}")`,
      'Notice the empirical risk on 40 points is *optimistic* — lower than the true risk. That gap is exactly what generalization theory tries to bound, and it grows with model capacity.'),

    quiz('You tune 200 hyperparameter configurations, pick the one with the best test accuracy, and report it. What is wrong?',
      ['Selecting on the test set makes the reported number optimistically biased — it is now a validation score',
       'Nothing, as long as you never trained on the test set',
       '200 configurations is too few to be meaningful',
       'You should have used a larger test set'],
      0,
      'With 200 draws you are selecting the maximum of 200 noisy estimates, and the max of noisy estimates is biased upward. This is the multiple-comparisons problem. The fix: select on a validation set, then evaluate the single chosen model on the test set once. Expect the test number to be meaningfully lower.'),
  ],
  refs: [
    book('Understanding Machine Learning: From Theory to Algorithms', 'Shalev-Shwartz & Ben-David', 2014, 'https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/', 'Free PDF. The rigorous foundation for everything in this lesson.'),
    book('The Elements of Statistical Learning', 'Hastie, Tibshirani & Friedman', 2009, 'https://hastie.su.domains/ElemStatLearn/', 'Free PDF. The standard reference for classical ML.'),
    paper('The Lack of A Priori Distinctions Between Learning Algorithms', 'David Wolpert', 1996, 'https://doi.org/10.1162/neco.1996.8.7.1341', 'The No Free Lunch theorem, in the original.'),
    paper('Understanding deep learning requires rethinking generalization', 'Zhang et al.', 2016, 'https://arxiv.org/abs/1611.03530', 'Deep nets fit random labels perfectly and still generalize on real ones. The paper that broke classical capacity theory.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'ml-linear-regression',
  title: 'Linear Regression',
  sub: 'The model you should always try first, and the geometry behind least squares.',
  mins: 26, level: 'core',
  prereq: ['math-vectors', 'math-probability'],
  tags: ['regression', 'least squares'],
  sections: [
    t(`## The model

$$\\hat y = \\mathbf{w}^{\\mathsf T}\\mathbf{x} + b = w_1x_1 + \\cdots + w_dx_d + b$$

Fold $b$ into $\\mathbf{w}$ by appending a constant 1 feature, and it becomes $\\hat y = \\mathbf{w}^{\\mathsf T}\\mathbf{x}$.
Fit by minimizing squared error:

$$\\mathcal{L}(\\mathbf{w}) = \\frac{1}{n}\\|X\\mathbf{w}-\\mathbf{y}\\|^2$$`),

    viz('linear-regression'),

    t(`## Why squared error?

Three independent justifications converge on it, which is a good sign:

1. **Probabilistic.** It is maximum likelihood under Gaussian noise (derived in the
   [probability lesson](#/l/math-probability)).
2. **Geometric.** It finds the orthogonal projection of $\\mathbf{y}$ onto the column space of $X$ — the closest
   reachable point.
3. **Computational.** It is convex and quadratic, so the minimum is unique and available in closed form.

The cost of this convenience is sensitivity: squared error punishes large residuals quadratically, so one outlier can
dominate the fit. Try the **+ outlier** button above.`),

    deriv('The normal equations', `Minimize $\\mathcal{L}(\\mathbf{w}) = \\|X\\mathbf{w}-\\mathbf{y}\\|^2 = (X\\mathbf{w}-\\mathbf{y})^{\\mathsf T}(X\\mathbf{w}-\\mathbf{y})$.

Expand:
$$\\mathcal{L} = \\mathbf{w}^{\\mathsf T}X^{\\mathsf T}X\\mathbf{w} - 2\\mathbf{y}^{\\mathsf T}X\\mathbf{w} + \\mathbf{y}^{\\mathsf T}\\mathbf{y}$$

Differentiate (using $\\partial(\\mathbf{w}^{\\mathsf T}A\\mathbf{w})/\\partial\\mathbf{w} = 2A\\mathbf{w}$ for symmetric $A$):

$$\\nabla_{\\mathbf w}\\mathcal{L} = 2X^{\\mathsf T}X\\mathbf{w} - 2X^{\\mathsf T}\\mathbf{y} = 0$$

$$\\boxed{\\ \\hat{\\mathbf{w}} = (X^{\\mathsf T}X)^{-1}X^{\\mathsf T}\\mathbf{y}\\ }$$

**The geometry.** The condition $X^{\\mathsf T}(X\\mathbf{w}-\\mathbf{y}) = 0$ says the residual is orthogonal to every
column of $X$. That is the defining property of an orthogonal projection: $\\hat{\\mathbf y} = X\\hat{\\mathbf w}$ is the
point in the column space of $X$ closest to $\\mathbf{y}$, and $H = X(X^{\\mathsf T}X)^{-1}X^{\\mathsf T}$ is the projection
matrix (statisticians call it the "hat matrix").

**Do not implement this literally.** Forming $X^{\\mathsf T}X$ squares the condition number. Use \`np.linalg.lstsq\`,
which goes through QR or SVD.`),

    viz('regression-loss-surface'),

    t(`## Reading the coefficients

$w_j$ is the expected change in $y$ per unit change in $x_j$, **holding all other features fixed**. That last clause is
where interpretation usually goes wrong:

- If features are correlated, "holding others fixed" may describe a situation that never occurs in your data. The
  coefficients become unstable — small data changes can flip their signs.
- Coefficient magnitude means nothing unless features are on comparable scales. Standardize first if you want to
  compare.
- **Correlation is not causation, and a regression coefficient is a correlation.** Adding or removing a covariate can
  reverse a coefficient's sign entirely (Simpson's paradox). Regression coefficients answer causal questions only
  under assumptions the regression itself cannot check.`),

    t(`## $R^2$ and its traps

$$R^2 = 1 - \\frac{\\sum_i (y_i-\\hat y_i)^2}{\\sum_i (y_i-\\bar y)^2}$$

"Fraction of variance explained." Useful, with caveats:

- $R^2$ **never decreases** when you add a feature, even a random one. Use adjusted $R^2$, or just look at held-out
  error.
- High $R^2$ does not mean the model is right; low $R^2$ does not mean it is useless (a weak signal on a noisy target
  can still be valuable).
- On the *test* set, $R^2$ can be negative — meaning you would have done better predicting the mean.`),

    t(`## Beyond straight lines

Linear regression is linear **in the parameters**, not in the features. So this is still linear regression:

$$\\hat y = w_0 + w_1x + w_2x^2 + w_3\\sin(x) + w_4 x_1x_2$$

You can fit any curve by engineering features. Which immediately raises the question of how many features to add —
and that is the next lesson.`),

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
print("\\nw4 (true value 0) should have |t| < 2 -> not distinguishable from zero")`),

    quiz('Two features in your regression are correlated at 0.99. What should you expect?',
      ['Individually unstable coefficients with huge standard errors, though predictions may still be fine',
       'The model will fail to fit the training data',
       'One coefficient will automatically become zero',
       'Predictions will be badly biased'],
      0,
      'This is **multicollinearity**. $X^{\\mathsf T}X$ becomes near-singular, so the coefficient split between the two features is nearly arbitrary — the fit can trade a large positive weight on one against a large negative weight on the other. Predictions stay accurate (their *sum* is well determined) but interpretation is worthless. Ridge regularization is the standard remedy.'),
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
    t(`## The phenomenon

A model complex enough will fit your training data perfectly, including the noise. It will then fail on new data,
because the noise does not repeat.`),

    viz('polynomial-overfit'),

    t(`Move the degree slider from 0 to 14 with 12 training points and watch the two error numbers diverge. That gap
between train and test error is the thing all of ML methodology exists to manage.`),

    t(`## The decomposition

For squared loss, expected test error at a point decomposes exactly:

$$\\underbrace{\\mathbb{E}\\big[(y-\\hat f(\\mathbf{x}))^2\\big]}_{\\text{expected test error}} =
\\underbrace{\\big(\\mathbb{E}[\\hat f(\\mathbf{x})]-f(\\mathbf{x})\\big)^2}_{\\text{bias}^2} +
\\underbrace{\\mathbb{E}\\big[(\\hat f(\\mathbf{x})-\\mathbb{E}[\\hat f(\\mathbf{x})])^2\\big]}_{\\text{variance}} +
\\underbrace{\\sigma^2}_{\\text{noise}}$$

where the expectations are over *random draws of the training set*.

- **Bias** — how far the average model is from the truth. High bias = too rigid = underfitting.
- **Variance** — how much the model changes if you resample the data. High variance = too flexible = overfitting.
- **Noise** — irreducible. No model beats this floor.`),

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

The classical U-curve says error rises past the sweet spot. It does — and then, past the point where the model can
exactly interpolate the training data, it **falls again**, often below the classical optimum.`),

    viz('double-descent'),

    t(`The peak sits exactly at $d = n$, where the model can *just barely* fit the data and must do so with enormous,
precariously balanced weights. Push past it and many interpolating solutions exist; gradient descent finds the
minimum-norm one, which is smooth.

This is not a curiosity. It is the regime every modern deep network lives in — vastly more parameters than data
points, zero training loss, and good generalization anyway. The classical theory does not explain it. Candidate
explanations involve implicit regularization by SGD, the norm of the interpolating solution, and the effective rather
than nominal parameter count.`),

    t(`## Grokking: training loss stopping is not learning stopping

A related and even stranger phenomenon. On small algorithmic tasks — modular arithmetic, say — a transformer reaches
**100% training accuracy while test accuracy sits at chance**, and then, tens of thousands of steps later, test
accuracy suddenly jumps to 100%.`),

    viz('grokking'),

    t(`Set weight decay to zero in that figure and it never happens. The current explanation: both a memorizing circuit
and a generalizing circuit fit the training data, but the generalizing one has smaller norm, so regularization slowly
replaces one with the other long after the loss curve went flat.

The practical lesson is uncomfortable — "the training loss stopped improving" is not reliable evidence that learning
has stopped.`),

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
    t(`## Add a penalty

$$\\mathcal{L}(\\mathbf{w}) = \\underbrace{\\|X\\mathbf{w}-\\mathbf{y}\\|^2}_{\\text{fit}} + \\lambda\\, \\underbrace{\\Omega(\\mathbf{w})}_{\\text{penalty}}$$

- **Ridge (L2)**: $\\Omega = \\|\\mathbf{w}\\|_2^2$. Shrinks all coefficients smoothly toward zero.
- **Lasso (L1)**: $\\Omega = \\|\\mathbf{w}\\|_1$. Shrinks *and* sets some exactly to zero.
- **Elastic net**: $\\alpha\\|\\mathbf{w}\\|_1 + (1-\\alpha)\\|\\mathbf{w}\\|_2^2$. Both.

$\\lambda$ interpolates between the unpenalized fit ($\\lambda\\to0$) and the all-zero model ($\\lambda\\to\\infty$). It is
chosen by cross-validation, essentially always.`),

    t(`## Why L1 gives you exact zeros

This is the question everyone asks, and the constrained-optimization picture answers it completely.`),

    viz('regularization-geometry'),

    key(`Minimizing $\\text{loss} + \\lambda\\Omega(\\mathbf{w})$ is equivalent to minimizing the loss subject to
$\\Omega(\\mathbf{w}) \\le t$ for some $t$. So: find the smallest loss contour that **touches** the constraint region.

The L1 ball is a **diamond with corners on the axes**. Contours coming in from a generic direction hit a corner first —
and a corner has a coordinate exactly equal to zero.

The L2 ball is a **smooth circle**. Contact happens at a generic boundary point, which has no zero coordinates.
Coefficients shrink toward zero but never arrive.`),

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

Lasso has no closed form (the L1 term is not differentiable at 0) and is solved by coordinate descent or LARS.`),

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

    warn(`**Standardize your features before regularizing.** The penalty treats all coefficients equally, so a feature
measured in millimetres gets a coefficient 1000× smaller than the same feature in metres — and is therefore penalized
1000× less. This is a genuinely common bug. Also: do not penalize the intercept.`),

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
    t(`## From scores to probabilities

We want $p(y=1\\mid\\mathbf{x}) \\in [0,1]$, but a linear function produces any real number. Squash it:

$$p(y=1\\mid\\mathbf{x}) = \\sigma(\\mathbf{w}^{\\mathsf T}\\mathbf{x}+b), \\qquad \\sigma(z)=\\frac{1}{1+e^{-z}}$$

The inverse view is cleaner: logistic regression models the **log-odds** as linear.

$$\\log\\frac{p}{1-p} = \\mathbf{w}^{\\mathsf T}\\mathbf{x}+b$$

So $w_j$ is "the change in log-odds per unit of $x_j$", and $e^{w_j}$ is the odds ratio — which is how the coefficient
is reported in every medical paper you will ever read.`),

    viz('sigmoid-softmax'),

    t(`## The loss

Squared error is wrong here. It is non-convex under the sigmoid, and it produces tiny gradients exactly where the
model is most confidently wrong. Use the negative log-likelihood, a.k.a. **binary cross-entropy**:

$$\\mathcal{L} = -\\frac{1}{n}\\sum_i \\big[y_i\\log p_i + (1-y_i)\\log(1-p_i)\\big]$$

This is convex in $\\mathbf{w}$, and its gradient is remarkably clean.`),

    deriv('The gradient, and why it looks like linear regression', `With $p = \\sigma(z)$ and $z = \\mathbf{w}^{\\mathsf T}\\mathbf{x}$, use $\\sigma'(z) = \\sigma(z)(1-\\sigma(z))$:

$$\\frac{\\partial \\mathcal{L}_i}{\\partial z} = -\\left[\\frac{y}{p} - \\frac{1-y}{1-p}\\right]p(1-p) = -\\big[y(1-p) - (1-y)p\\big] = p - y$$

and therefore

$$\\nabla_{\\mathbf{w}}\\mathcal{L} = \\frac{1}{n}\\sum_i (p_i - y_i)\\,\\mathbf{x}_i$$

Identical in form to linear regression's gradient, with $p_i$ in place of $\\hat y_i$. **This is not a coincidence** —
both are generalized linear models, and every GLM with its canonical link has gradient (prediction − target) × input.

Note the $p(1-p)$ factors *cancelled*. That cancellation is exactly why cross-entropy is the right loss: with squared
error they would survive, and a confidently-wrong prediction ($p\\approx0$ when $y=1$) would produce a vanishing
gradient — the model would be stuck precisely where it most needs to move.`),

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

Logistic regression is usually well calibrated: among examples where it says 0.7, about 70% are positive. That is a
direct consequence of the loss — cross-entropy is a **proper scoring rule**, minimized by reporting true probabilities.

Deep networks trained with the same loss are often badly *over*confident anyway, for reasons involving capacity and
training to zero loss. Standard remedies: temperature scaling on a validation set, or label smoothing during training. RLHF makes this
markedly worse — human raters reward confident-sounding answers whether or not they are correct.`),

    t(`## Thresholds are a separate decision

The model gives $p$. Turning $p$ into a decision requires a threshold, and 0.5 is only optimal when false positives and
false negatives cost the same. They rarely do. Choose the threshold from the cost structure, or from a target
precision/recall — not by default.`),

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
        print(f"  {lo:.1f}-{lo+0.2:.1f}    {y[m].mean():.3f}   {m.sum():5d}")`),

    quiz('Why is squared error a bad loss for classification with a sigmoid output?',
      ["It is non-convex, and its gradient vanishes exactly when the model is confidently wrong",
       'It is not differentiable',
       'It cannot handle more than two classes',
       'It always overfits'],
      0,
      "With squared error the gradient carries a factor of $\\sigma'(z) = p(1-p)$, which goes to zero as $p \\to 0$ or $1$. A model that predicts 0.001 for a positive example gets essentially no gradient — it is maximally wrong and maximally stuck. Cross-entropy's $p-y$ gradient is largest precisely there. The $p(1-p)$ factors cancel; that cancellation is the whole point."),
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
    t(`## Decision trees

Recursively split the feature space with axis-aligned cuts, choosing at each step the (feature, threshold) that most
reduces impurity:

- **Gini**: $1-\\sum_k p_k^2$
- **Entropy**: $-\\sum_k p_k\\log p_k$
- **Variance** (for regression)

Predict the majority class or mean value in each leaf.`),

    viz('decision-tree'),

    t(`Trees are attractive: no scaling needed, they handle mixed types and missing values natively, they capture
interactions automatically, and a shallow one is genuinely interpretable.

They are also **high variance**. Change a few data points near a split and the entire subtree below it can change.
A fully grown tree memorizes. Which is why nobody uses a single tree — but it is why the two great ensembling ideas
work so well on them.`),

    t(`## Bagging and random forests

**Bagging**: train $M$ trees on bootstrap resamples, average their predictions. Averaging $M$ estimators with
individual variance $\\sigma^2$ and pairwise correlation $\\rho$ gives variance

$$\\rho\\sigma^2 + \\frac{1-\\rho}{M}\\sigma^2$$

The second term vanishes with $M$; the first does not. **So the win comes from decorrelating the trees**, not from
adding more of them.

**Random forests** add exactly that: at each split, consider only a random subset of features (typically
$\\sqrt{d}$ for classification). This forces different trees to rely on different features, driving $\\rho$ down.

Practical notes: forests essentially do not overfit as you add trees — more is monotonically better up to diminishing
returns. Out-of-bag error (each tree evaluated on the ~37% of samples it did not see) gives you free cross-validation.`),

    t(`## Gradient boosting

Boosting builds trees **sequentially**, each fit to the errors of the current ensemble:

$$F_m(\\mathbf{x}) = F_{m-1}(\\mathbf{x}) + \\nu\\, h_m(\\mathbf{x})$$

where $h_m$ is fit to the negative gradient of the loss with respect to the current predictions, and $\\nu$ is the
learning rate (shrinkage). For squared loss the negative gradient *is* the residual, which is where the "fit the
residuals" description comes from — but the gradient framing is what lets you boost any differentiable loss.`),

    viz('boosting'),

    key(`**Bagging reduces variance; boosting reduces bias.** Random forests average many low-bias, high-variance trees.
Boosting adds many high-bias, low-variance stumps until they collectively fit.

The consequence is that boosting *can* overfit, and does — it needs early stopping on a validation set. Random forests
mostly cannot.`),

    t(`## Why gradient boosting still wins on tabular data

XGBoost, LightGBM, and CatBoost remain the default for tabular problems, and neural networks have repeatedly failed to
beat them convincingly. The reasons are structural:

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
print("(deep single trees overfit; averaging them does not)")`),

    quiz('Random forest gets 0.85 AUC; gradient boosting with the same features gets 0.88 but only after careful tuning. What does this suggest?',
      ['Normal behavior — boosting usually edges out forests but is far more sensitive to hyperparameters',
       'The forest is broken',
       'The boosting model is overfitting',
       'The features are unsuitable for trees'],
      0,
      'This is the standard pattern. Random forests are nearly hyperparameter-free and give you a strong baseline immediately; boosting reaches a higher ceiling but needs the learning rate, tree depth, and number of rounds tuned together, with early stopping. A sensible workflow is: forest first for the baseline, boosting second for the final model.'),
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
    t(`## k-nearest neighbours

Store the training data. To predict, find the $k$ closest points and vote. There is no training and no model —
the data *is* the model.`),

    viz('knn-boundary'),

    t(`It is a genuinely useful baseline and it exposes several ideas cleanly: $k$ directly controls the bias-variance
tradeoff, training accuracy at $k=1$ is a meaningless 100%, and leave-one-out cross-validation is nearly free.

Its problems are also instructive. Prediction cost scales with the dataset. Distances require a metric, and in high
dimensions every point is roughly equidistant from every other, so "nearest" stops carrying information. Modern vector
databases are kNN with approximate search (HNSW, IVF-PQ) to make the first problem tractable — and RAG is, structurally,
kNN in embedding space.`),

    t(`## Support vector machines

A separating hyperplane is not enough; the SVM asks for the one with the **widest margin**. With labels $y_i\\in\\{-1,+1\\}$:

$$\\min_{\\mathbf{w},b}\\ \\tfrac12\\|\\mathbf{w}\\|^2 \\quad\\text{s.t.}\\quad y_i(\\mathbf{w}^{\\mathsf T}\\mathbf{x}_i+b)\\ge 1$$

The margin width is $2/\\|\\mathbf{w}\\|$, so minimizing $\\|\\mathbf{w}\\|$ maximizes it.`),

    viz('svm-margin'),

    t(`Real data is not separable, so introduce slack, which turns the problem into the equivalent unconstrained form:

$$\\min_{\\mathbf{w}}\\ \\underbrace{\\tfrac{1}{2}\\|\\mathbf{w}\\|^2}_{\\text{margin}} + C\\sum_i \\underbrace{\\max(0, 1-y_i f(\\mathbf{x}_i))}_{\\text{hinge loss}}$$

which is just regularized ERM with hinge loss. Small $C$ = more regularization = wider, more forgiving margin.`),

    key(`Only points **on or inside the margin** have nonzero weight in the solution — the *support vectors*. Delete any
other training point and the answer is unchanged. That sparsity is what makes the kernel trick affordable: prediction
only requires kernel evaluations against the support vectors, not the whole dataset.`),

    t(`## The kernel trick

The dual formulation of the SVM involves the data only through inner products $\\mathbf{x}_i^{\\mathsf T}\\mathbf{x}_j$.
So replace them with $k(\\mathbf{x}_i,\\mathbf{x}_j) = \\phi(\\mathbf{x}_i)^{\\mathsf T}\\phi(\\mathbf{x}_j)$ for some feature
map $\\phi$ — **and never compute $\\phi$ at all.**`),

    viz('kernel-trick'),

    t(`| Kernel | $k(\\mathbf{x},\\mathbf{x}')$ | Implied feature space |
|---|---|---|
| Linear | $\\mathbf{x}^{\\mathsf T}\\mathbf{x}'$ | the original |
| Polynomial | $(\\gamma\\,\\mathbf{x}^{\\mathsf T}\\mathbf{x}'+r)^d$ | all monomials up to degree $d$ |
| RBF / Gaussian | $\\exp(-\\gamma\\|\\mathbf{x}-\\mathbf{x}'\\|^2)$ | **infinite-dimensional** |

The RBF kernel computes an inner product in an infinite-dimensional space with one exponential. That is the trick in
its purest form. Any function producing a positive semi-definite Gram matrix is a valid kernel (Mercer's condition).`),

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
print("\\nA linear model scores ~0.5 here. The kernel does all the work.")`),

    quiz('You train an RBF-SVM with γ very large. What happens?',
      ['Each support vector influences only its immediate neighbourhood — the boundary fragments into islands and overfits',
       'The decision boundary becomes linear',
       'Training fails to converge',
       'The model underfits'],
      0,
      '$\\gamma$ sets the width of the Gaussian bump around each support vector. Large $\\gamma$ = narrow bumps = the model can only "see" points essentially on top of each other, so it memorizes. Small $\\gamma$ = wide bumps = an almost-linear boundary. $\\gamma$ and $C$ must be tuned jointly, which is why grid search over both is the standard recipe.'),
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
    t(`## k-means

Minimize within-cluster squared distance by alternating two steps: assign each point to its nearest centre, then move
each centre to the mean of its points.`),

    viz('kmeans'),

    t(`Both steps decrease the objective, so it always converges — but to a *local* optimum that depends on
initialization. Press "new init" a few times on the same data and watch the answer change. Standard mitigations:
**k-means++** initialization (choose seeds spread apart, with a provable approximation guarantee) and multiple restarts.

The deeper limitation is baked into the objective. Minimizing squared distance to a centre means the implied clusters
are **spherical and equally sized**, with straight boundaries between them (a Voronoi diagram). Elongated, nested, or
differently-sized clusters are simply outside what k-means can express.

Choosing $k$: the elbow method (plot inertia vs $k$, look for the bend) is subjective and often has no elbow. The
silhouette score is better. Best of all is having an external reason to prefer a particular $k$.`),

    t(`## Gaussian mixtures and EM

Model the data as generated by a mixture of Gaussians:

$$p(\\mathbf{x}) = \\sum_{k=1}^K \\pi_k\\,\\mathcal{N}(\\mathbf{x}\\mid\\boldsymbol\\mu_k,\\Sigma_k)$$

Fit by **expectation-maximization**:

- **E-step**: compute each point's *responsibility* $\\gamma_{ik} = p(z_i=k\\mid\\mathbf{x}_i)$ — a soft assignment.
- **M-step**: refit each Gaussian's mean and covariance, weighted by those responsibilities.`),

    viz('gmm-em'),

    key(`**k-means is GMM with hard assignments and shared spherical covariance.** Going soft buys you: clusters with
shape and orientation, honest uncertainty at boundaries, and a proper likelihood — which means you can use AIC/BIC to
choose $K$ rather than squinting at an elbow.

EM's guarantee is that log-likelihood increases monotonically. It is the same alternating-optimization pattern you will
see again in the VAE (where the E-step is replaced by an amortized encoder network).`),

    t(`## PCA

Find the orthogonal directions of greatest variance. Equivalently — and this equivalence is the heart of it — find the
$k$-dimensional subspace minimizing reconstruction error.`),

    viz('pca'),

    t(`The recipe: centre the data, take the SVD of $X$, keep the top $k$ right singular vectors. The eigenvalues of the
covariance give the explained variance per component.

Things to keep straight:

- **Centring is mandatory.** Without it, PC1 just points at the mean.
- **Scaling matters.** If features have different units, PCA finds whichever has the largest numerical variance.
  Standardize unless the units are genuinely comparable.
- PCA is **linear**. Data on a curved manifold (a spiral, a Swiss roll) will not unroll.
- Components are orthogonal by construction, which makes them mathematically clean and often physically meaningless.`),

    t(`## t-SNE and UMAP: read the fine print

Nonlinear methods that preserve *neighbourhoods* rather than distances. They make beautiful pictures and they are
routinely over-interpreted. What is and is not trustworthy in one of these plots:

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
print(f"1-component reconstruction MSE: {((Xc - recon)**2).mean():.4f}")`),

    quiz('Your t-SNE plot shows two clusters far apart and one nearby pair. What can you conclude about the underlying data?',
      ['Only that there is local neighbourhood structure — inter-cluster distances in t-SNE carry no reliable meaning',
       'The two distant clusters are more different from each other than the nearby pair',
       'There are exactly three groups in the data',
       'The data is three-dimensional'],
      0,
      't-SNE optimizes a KL objective over pairwise neighbour probabilities, which pins down local structure and leaves global layout largely arbitrary. Distances between clusters are an artifact of the optimization, not a property of your data. Run it at several perplexities: the global arrangement will move while local groupings stay put, which tells you exactly which parts to trust.'),
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
    t(`## Accuracy is usually the wrong metric

With 1% positives, predicting "negative" always gives 99% accuracy and zero value. Start from the confusion matrix:

| | Predicted + | Predicted − |
|---|---|---|
| **Actual +** | TP | FN |
| **Actual −** | FP | TN |

- **Precision** $= \\frac{TP}{TP+FP}$ — of the ones you flagged, how many were right? *Cost of false alarms.*
- **Recall / TPR** $= \\frac{TP}{TP+FN}$ — of the real ones, how many did you catch? *Cost of misses.*
- **F1** — harmonic mean of the two. Convenient, and it hides the tradeoff you should be making explicitly.
- **Specificity / TNR** $= \\frac{TN}{TN+FP}$.

Which matters depends entirely on relative costs. Cancer screening wants recall. Spam filtering wants precision
(a lost real email is worse than a spam that got through).`),

    viz('roc-curve'),

    t(`## ROC vs precision-recall

**ROC** plots TPR against FPR across all thresholds; AUC is the probability a random positive scores above a random
negative. It is **prevalence-independent**, which is either a feature or a trap depending on what you are doing.

**Precision-recall** curves *do* depend on prevalence, which makes them the honest choice for rare-event problems.
Drag prevalence down to 2% in the figure: AUC barely moves while precision collapses. If your positive class is rare,
report PR-AUC (average precision).

A model has one curve but infinitely many operating points. Reporting a single F1 without saying which threshold
produced it is close to meaningless.`),

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

Compare two models on the *same* test set with a paired test — McNemar's test for classification, a paired bootstrap
for anything. Comparing independent accuracy numbers with a two-sample test throws away the pairing and is much less
powerful.

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
print("There is no signal. Any number above 0.5 here is pure leakage.")`),

    quiz('A fraud model has 99.5% accuracy on a dataset with 0.5% fraud. Your next question should be:',
      ['What are precision and recall? It may be predicting "not fraud" for everything',
       'How can we push accuracy to 99.9%?',
       'What is the training time?',
       'Nothing — 99.5% is excellent'],
      0,
      'Predicting the majority class always gives 99.5% here. The model may have learned literally nothing. Ask for precision and recall at the operating threshold, PR-AUC across thresholds, and the confusion matrix in raw counts. Accuracy on an imbalanced problem is close to uninformative.'),
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
    t(`## Two strategies

**Discriminative** models learn $p(y\\mid\\mathbf{x})$ — or just a decision boundary — directly. Logistic regression,
SVMs, and standard neural classifiers are discriminative.

**Generative** models learn the joint $p(\\mathbf{x},y) = p(\\mathbf{x}\\mid y)p(y)$, then apply Bayes' rule:

$$p(y\\mid\\mathbf{x}) \\propto p(\\mathbf{x}\\mid y)\\,p(y)$$

Naive Bayes, LDA, and Gaussian mixture classifiers are generative — as are, in the modern sense, VAEs, diffusion
models, and language models.`),

    viz('generative-discriminative'),

    t(`## The tradeoff

Ng & Jordan's 2001 analysis made the comparison precise for the naive Bayes / logistic regression pair:

- **Generative models have higher asymptotic error** (their modeling assumptions are usually wrong) but **converge
  faster** — they reach their ceiling with $O(\\log d)$ samples versus $O(d)$ for the discriminative model.
- So generative wins on small data, discriminative wins as $n$ grows, and the curves cross.

The intuition: modeling $p(\\mathbf{x}\\mid y)$ is a harder problem than you need to solve, and solving a harder problem
costs you accuracy — but the strong assumptions act as a powerful prior when data is scarce.`),

    t(`## Naive Bayes

Assume features are conditionally independent given the class:

$$p(\\mathbf{x}\\mid y) = \\prod_{j=1}^d p(x_j\\mid y)$$

This assumption is essentially always false — and naive Bayes works anyway, especially for text. The reason is that
for *classification* you only need the argmax to be right, not the probabilities. Correlated features cause
systematically overconfident probability estimates while often leaving the ranking intact.

It remains a genuinely good baseline for text: it trains in one pass, needs almost no data, and handles high
dimensions gracefully.`),

    t(`## The modern reading

The distinction has become more interesting, not less:

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

    code('Naive Bayes vs logistic regression as n grows', `import numpy as np
rng = np.random.default_rng(0)

d = 30
mu0, mu1 = rng.normal(0, 1, d), rng.normal(0.8, 1, d)

def make(n):
    y = rng.integers(0, 2, n)
    X = rng.normal(np.where(y[:, None], mu1, mu0), 1.0)
    return X, y

def naive_bayes(Xtr, ytr, Xte):
    m0, m1 = Xtr[ytr == 0].mean(0), Xtr[ytr == 1].mean(0)
    s0 = Xtr[ytr == 0].std(0) + 1e-6
    s1 = Xtr[ytr == 1].std(0) + 1e-6
    ll0 = -((Xte - m0)**2 / (2*s0**2) + np.log(s0)).sum(1)
    ll1 = -((Xte - m1)**2 / (2*s1**2) + np.log(s1)).sum(1)
    return (ll1 > ll0).astype(int)

def logistic(Xtr, ytr, Xte, steps=3000):
    Xb = np.c_[np.ones(len(Xtr)), Xtr]
    w = np.zeros(d + 1)
    for _ in range(steps):
        p = 1/(1+np.exp(-np.clip(Xb @ w, -500, 500)))
        w -= 0.5 * (Xb.T @ (p - ytr) / len(ytr) + 0.01*np.r_[0, w[1:]])
    return (np.c_[np.ones(len(Xte)), Xte] @ w > 0).astype(int)

Xte, yte = make(4000)
print("   n     naive Bayes   logistic")
for n in [20, 50, 100, 300, 1000, 5000]:
    Xtr, ytr = make(n)
    nb = (naive_bayes(Xtr, ytr, Xte) == yte).mean()
    lr = (logistic(Xtr, ytr, Xte) == yte).mean()
    print(f"{n:5d}      {nb:.3f}       {lr:.3f}  {'<- NB ahead' if nb > lr else ''}")`,
      'Naive Bayes leads at small $n$ and logistic regression overtakes it. The crossover point is exactly what Ng & Jordan characterized.'),

    quiz('Why can a language model perform sentiment classification it was never trained on?',
      ['It models the joint distribution over text, so it can be conditioned into any task expressible in text',
       'It was secretly trained on sentiment labels',
       'Because it is a discriminative model',
       'It cannot — that requires fine-tuning'],
      0,
      'A generative model of $p(\\text{text})$ implicitly contains $p(\\text{label}\\mid\\text{review})$ for any labeling scheme that appears in natural language, because "This movie was great. Sentiment: positive" is itself text. Prompting is conditioning. This is the practical payoff of generative modeling and the reason the field moved decisively toward it.'),
  ],
  refs: [
    paper('On Discriminative vs. Generative Classifiers', 'Ng & Jordan', 2001, 'https://papers.nips.cc/paper/2020-on-discriminative-vs-generative-classifiers-a-comparison-of-logistic-regression-and-naive-bayes', 'The paper that made the tradeoff precise.'),
    paper('Language Models are Few-Shot Learners', 'Brown et al.', 2020, 'https://arxiv.org/abs/2005.14165', 'GPT-3. The clearest demonstration of generative-model-as-universal-task-solver.'),
    book('Machine Learning: A Probabilistic Perspective, Ch. 3–4', 'Kevin Murphy', 2012, 'https://probml.github.io/pml-book/book0.html', ''),
  ],
},

];

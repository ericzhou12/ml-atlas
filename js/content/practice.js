/* ============================================================
   Track 10 — Practice
   ============================================================ */

import { t, key, intuition, warn, hist, mathnote, viz, deriv, code, quiz, paper, book, course, blog, video, demo, codeRef } from './_helpers.js';

export default [

/* ---------------------------------------------------------- */
{
  id: 'pr-debugging',
  title: 'Debugging Models That Do Not Work',
  sub: 'A procedure, in order, for the most common situation in applied ML.',
  mins: 22, level: 'core',
  prereq: ['nn-losses-training'],
  tags: ['debugging', 'practice'],
  sections: [
    t(`## The procedure

Do these in order. Each step is cheap and eliminates a large class of causes. Skipping ahead is how people lose days.

**1. Overfit a single batch.**
Take 8 examples, disable augmentation, dropout, and weight decay, and train until the loss is ~0. If you cannot, you
have a **bug**, not a tuning problem. This single test catches more real defects than anything else — wrong loss
reduction, detached gradients, labels misaligned with inputs, a frozen parameter you meant to train.

**2. Check the initial loss.**
For $K$-way classification it should be $\\log K$: 2.303 for 10 classes, 6.908 for 1000. For regression with
standardized targets it should be ~1.0. A wrong value means the head, the labels, or the loss is misconfigured — before
a single step of training.

**3. Sweep the learning rate over powers of 10.**
Run 100 steps each at $10^{-1}$ through $10^{-6}$. Find where it diverges; the usable value is typically 3–10× below
that. Do this before tuning anything else, because everything else depends on it.

**4. Watch gradient norms per layer.**
Log them. Vanishing or exploding will be obvious, and it localizes the problem to a specific depth.

**5. Look at the data. Actually look at it.**
Render the images after augmentation. Print the tokenized text. Check the label distribution, the value ranges, the
NaN count. A large fraction of "model bugs" are data bugs, and they are invisible from the loss curve.

**6. Simplify until it works, then add back.**
Remove the scheduler, the augmentation, the fancy architecture. Get something training. Add one piece at a time.`),

    warn(`**The specific bugs that recur:**

- **Forgetting \`optimizer.zero_grad()\`** — gradients accumulate silently and you train on a running sum.
- **Forgetting \`model.eval()\`** — dropout and BatchNorm stay in training mode, corrupting validation numbers.
- **Preprocessing fit on the full dataset** before splitting. Leakage.
- **Shuffling inputs and labels separately.** The model learns noise; loss plateaus at chance.
- **The wrong loss reduction** — \`sum\` instead of \`mean\` makes the effective learning rate scale with batch size.
- **Softmax applied twice** — once in the model, once in the loss. Frameworks' \`cross_entropy\` expects raw logits.
- **Ignoring the padding mask**, so the model attends to or is trained on padding tokens.
- **Wrong tensor layout** — silent broadcasting that "works" and computes nonsense. Assert shapes.`),

    t(`## Reading a loss curve

| Pattern | Likely cause |
|---|---|
| Flat at initial value | LR far too low, gradients not flowing, or frozen parameters |
| Diverges to NaN | LR too high, no gradient clipping, or fp16 overflow |
| Falls then plateaus high | Underfitting — more capacity, longer training, or better features |
| Train falls, val rises | Overfitting — regularize, augment, or get more data |
| Both fall, val far above train | Distribution shift between splits, or leakage |
| Spiky | Batch too small, LR too high, or outliers in the data |
| Sudden jump mid-training | Bad data batch, or a scheduler transition |

For NaN specifically: log the gradient norm every step. You will usually see it climb for several steps before the
blowup, which tells you it is an optimization problem rather than a single poisoned example.`),

    t(`## Diagnosing capacity vs data

A useful decomposition. Compare four numbers: train error, validation error, and human (or best-known) error.

- **Train error ≫ human error** → **underfitting / high bias**. Bigger model, train longer, better features, less
  regularization.
- **Train error ≈ human, val ≫ train** → **overfitting / high variance**. More data, augmentation, regularization.
- **Both near human** → you are done, or your metric is wrong.
- **Val ≫ train AND train is fine, but test ≫ val** → your validation set is not representative of your test set.

Plotting a **learning curve** — error vs training set size — tells you directly whether more data will help. If the
validation curve has flattened, it will not.`),

    code('A diagnostic harness', `import numpy as np
rng = np.random.default_rng(0)

def make_model(n_in=20, n_hidden=32, n_out=3, seed=0):
    r = np.random.default_rng(seed)
    return [(r.normal(0, np.sqrt(2/a), (a,b)), np.zeros(b))
            for a,b in [(n_in,n_hidden), (n_hidden,n_out)]]

def forward(P, X):
    h = np.maximum(0, X @ P[0][0] + P[0][1])
    return h @ P[1][0] + P[1][1], h

def loss_and_grads(P, X, y):
    logits, h = forward(P, X)
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    d = p.copy(); d[np.arange(len(y)), y] -= 1; d /= len(y)
    g1 = (h.T @ d, d.sum(0))
    dh = (d @ P[1][0].T) * (h > 0)
    g0 = (X.T @ dh, dh.sum(0))
    return loss, [g0, g1]

X = rng.normal(size=(512, 20))
y = rng.integers(0, 3, 512)
X[y == 1] += 0.9; X[y == 2] -= 0.9              # make it learnable

# --- CHECK 1: initial loss should be log(K) ---
P = make_model()
l0, _ = loss_and_grads(P, X, y)
print(f"CHECK 1  initial loss {l0:.4f}   expected log(3) = {np.log(3):.4f}   "
      f"{'PASS' if abs(l0-np.log(3)) < 0.1 else 'FAIL'}")

# --- CHECK 2: can we overfit 8 examples? ---
P = make_model()
Xs, ys = X[:8], y[:8]
for _ in range(600):
    l, g = loss_and_grads(P, Xs, ys)
    P = [(W-0.1*gW, b-0.1*gb) for (W,b),(gW,gb) in zip(P,g)]
print(f"CHECK 2  loss on 8 examples after 600 steps: {l:.6f}   "
      f"{'PASS' if l < 0.01 else 'FAIL - you have a bug, not a tuning problem'}")

# --- CHECK 3: learning rate sweep ---
print("\\nCHECK 3  learning rate sweep (100 steps each):")
for lr in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    P = make_model()
    for _ in range(100):
        l, g = loss_and_grads(P, X, y)
        P = [(W-lr*gW, b-lr*gb) for (W,b),(gW,gb) in zip(P,g)]
        if not np.isfinite(l): break
    status = "diverged" if not np.isfinite(l) else f"loss {l:.4f}"
    print(f"  lr={lr:<7} {status}")

# --- CHECK 4: gradient norms per layer ---
P = make_model()
_, g = loss_and_grads(P, X, y)
print("\\nCHECK 4  gradient norms per layer:")
for i, (gW, gb) in enumerate(g):
    print(f"  layer {i}: |dW| {np.linalg.norm(gW):.5f}  |db| {np.linalg.norm(gb):.5f}")

# --- CHECK 5: gradient check against finite differences ---
P = make_model()
_, g = loss_and_grads(P, X[:16], y[:16])
eps, i, j = 1e-5, 3, 7
P[0][0][i,j] += eps; lp, _ = loss_and_grads(P, X[:16], y[:16])
P[0][0][i,j] -= 2*eps; lm, _ = loss_and_grads(P, X[:16], y[:16])
P[0][0][i,j] += eps
num = (lp - lm) / (2*eps)
rel = abs(num - g[0][0][i,j]) / (abs(num) + 1e-12)
print(f"\\nCHECK 5  analytic {g[0][0][i,j]:.8f}  numeric {num:.8f}  "
      f"rel err {rel:.2e}  {'PASS' if rel < 1e-4 else 'FAIL'}")`,
      'Run this before debugging anything else. Checks 1, 2, and 5 in particular will catch the majority of real bugs in a hand-written model.'),

    quiz('Your model cannot reach zero loss on a single batch of 8 examples, even after 5,000 steps. What does this tell you?',
      ['There is a bug — a model with enough capacity should memorize 8 examples trivially',
       'You need more training data',
       'The learning rate is slightly too low',
       'The model needs regularization'],
      0,
      'Eight examples is trivially memorizable by almost any network. Failure means something is structurally broken: gradients not reaching some parameters, labels misaligned with inputs, the loss computed on the wrong tensor, a detached graph, or a frozen layer. This is the single most valuable test in ML debugging precisely because it separates "bug" from "tuning" in about a minute.'),
  ],
  refs: [
    blog('A Recipe for Training Neural Networks', 'Andrej Karpathy', 2019, 'http://karpathy.github.io/2019/04/25/recipe/', 'The best practical guide written. This lesson is largely a structured version of it.'),
    blog('Machine Learning Yearning', 'Andrew Ng', 2018, 'https://info.deeplearning.ai/machine-learning-yearning-book', 'Free. About diagnosing where error comes from and what to do about it.'),
    blog('37 Reasons why your Neural Network is not working', 'Slav Ivanov', 2017, 'https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-ac4a3bcb2b6', 'A checklist, and most items on it are real.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'pr-experiments',
  title: 'Running Experiments You Can Trust',
  sub: 'Reproducibility, baselines, ablations, and not fooling yourself.',
  mins: 20, level: 'core',
  prereq: ['ml-evaluation'],
  tags: ['methodology', 'reproducibility'],
  sections: [
    t(`## Baselines first

Before anything clever, establish:

1. **The trivial baseline** — majority class, predict the mean, copy the input. Surprisingly often competitive, and
   it tells you what your metric means.
2. **The simple model** — logistic regression, gradient boosting, a small MLP. Tuned properly, with the same effort
   you will spend on the fancy one.
3. **The published baseline**, reproduced by you. Not the number from the paper — the number you get.

A substantial fraction of published improvements evaporate when baselines are tuned with equal care. Ensure yours is
not one of them.`),

    key(`**The tuning-effort asymmetry** is the most common source of self-deception in ML research and applied work
alike. You spend three weeks on your method and an afternoon on the baseline, then report that yours is better.

The fix is procedural: give the baseline the same hyperparameter search budget, and report what that budget was.`),

    t(`## Reproducibility

- **Seed everything** — Python, NumPy, the framework, and the dataloader workers. Note that full determinism on GPU
  usually requires disabling some optimizations and costs speed; it is worth it for debugging, not for production runs.
- **Log the environment** — library versions, hardware, CUDA version, and the git commit. A results file without a
  commit hash is not reproducible.
- **Version the data**, not just the code. Data changes silently; a hash of the dataset belongs in the run record.
- **Save configs, not scripts.** Configuration in a file that gets logged with the run; no hardcoded constants edited
  between experiments.
- **Report variance across seeds.** A single run's number is one draw from a distribution. For RL and small datasets
  that distribution is alarmingly wide — Henderson et al. showed RL results can invert with a different seed.`),

    t(`## Ablations

An ablation answers "which part of my method is doing the work?" Remove one component at a time and measure.

Done well, this is the most informative section of any paper. Done badly — or omitted — the reader has no way to know
whether your gain comes from your idea, from a better learning rate, or from three extra epochs.

Look for, and provide: a component-by-component table, a hyperparameter sensitivity analysis, and at least one
negative result. Methods that only work in one configuration usually do not work.`),

    t(`## Hyperparameter search

- **Random search beats grid search** for the same budget. Grid search wastes evaluations on dimensions that do not
  matter; random search covers the important dimensions more finely. Bergstra & Bengio showed this cleanly.
- **Search on a log scale** for learning rates, regularization strengths, and anything spanning orders of magnitude.
- **Bayesian optimization / Hyperband** helps when each run is expensive, by pruning bad configurations early.
- **Fix a budget in advance** and report it. "We searched 200 configurations" is information; silently searching until
  the number looks good is not.
- **Never select on the test set.** With enough draws you will find a good test number for a bad model.`),

    t(`## Reading a paper critically

A working checklist:

- What is the **baseline**, and was it tuned as carefully as the method?
- Is there **variance** across seeds, or a single number?
- Is the **evaluation set** genuinely held out? Could it be in pretraining?
- Do the **ablations** support the stated mechanism, or just the headline result?
- Are there **negative results** or failure cases?
- Would the claimed effect survive if the comparison were compute-matched rather than epoch-matched?
- Has anyone **independent** reproduced it?

None of this requires expertise in the subfield, and it filters a great deal.`),

    code('Random vs grid search, and seed variance', `import numpy as np
rng = np.random.default_rng(0)

# A response surface where only 2 of 5 hyperparameters actually matter.
def objective(h):
    return (np.exp(-((h[0]-0.3)**2)/0.02)
            + 0.7*np.exp(-((h[1]-0.7)**2)/0.05)
            + 0.02*rng.normal())

BUDGET = 81

# grid: 3 points per dimension over 5 dims = 243, so use 3^4 on 4 dims + 1
best_grid = -np.inf
grid_pts = np.linspace(0.1, 0.9, 3)
for a in grid_pts:
    for b in grid_pts:
        for c in grid_pts:
            for d in grid_pts:
                best_grid = max(best_grid, objective([a,b,c,d,0.5]))

best_random = max(objective(rng.random(5)) for _ in range(BUDGET))

print(f"budget = {BUDGET} evaluations")
print(f"  grid search  : best {best_grid:.4f}  (only 3 distinct values per dim)")
print(f"  random search: best {best_random:.4f}  ({BUDGET} distinct values per dim)")
print("\\nGrid search spends its budget on dimensions that do not matter.")
print("Random search gets fine resolution on the ones that do.\\n")

# --- seed variance ---
def train_run(seed, true_effect=0.0):
    r = np.random.default_rng(seed)
    return 0.75 + true_effect + r.normal(0, 0.025)

print("comparing two methods where the true difference is 1%:")
for n_seeds in [1, 3, 10, 30]:
    a = [train_run(s, 0.00) for s in range(n_seeds)]
    b = [train_run(s+1000, 0.01) for s in range(n_seeds)]
    diff = np.mean(b) - np.mean(a)
    se = np.sqrt(np.var(a)/n_seeds + np.var(b)/n_seeds) if n_seeds > 1 else float('nan')
    verdict = "?" if n_seeds == 1 else ("significant" if abs(diff) > 2*se else "not significant")
    print(f"  {n_seeds:2d} seed(s): observed diff {diff:+.4f}  {verdict}")
print("\\nWith one seed you cannot distinguish a 1% effect from noise.")`),

    quiz('You try 200 hyperparameter configurations and report the best validation score as your result. What is wrong?',
      ['The best of 200 noisy estimates is biased upward — you need a separate held-out set for the final number',
       'Nothing, as long as you did not touch the test set',
       '200 configurations is too many',
       'You should have used grid search'],
      0,
      'Selecting the maximum over 200 noisy evaluations gives an optimistically biased estimate — this is the multiple-comparisons problem. The validation score of your chosen configuration is a *selection* statistic, not an unbiased performance estimate. Report the test-set number for the single chosen configuration, and expect it to be meaningfully lower.'),
  ],
  refs: [
    paper('Random Search for Hyper-Parameter Optimization', 'Bergstra & Bengio', 2012, 'https://www.jmlr.org/papers/v13/bergstra12a.html', 'Why random beats grid. Short and convincing.'),
    paper('Deep Reinforcement Learning that Matters', 'Henderson et al.', 2017, 'https://arxiv.org/abs/1709.06560', 'Seed variance can invert published conclusions.'),
    paper('Show Your Work: Improved Reporting of Experimental Results', 'Dodge et al.', 2019, 'https://arxiv.org/abs/1909.03004', 'Report performance as a function of search budget, not as a single number.'),
    paper('A Metric Learning Reality Check', 'Musgrave, Belongie & Lim', 2020, 'https://arxiv.org/abs/2003.08505', 'A decade of claimed progress largely disappears under fair comparison. Worth reading as a cautionary tale.'),
    blog('The Bitter Lesson', 'Rich Sutton', 2019, 'http://www.incompleteideas.net/IncIdeas/BitterLesson.html', 'Short, and it has aged extremely well.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'pr-reading',
  title: 'Reading Papers and Staying Current',
  sub: 'How to get through a paper efficiently, and how to filter a firehose.',
  mins: 16, level: 'core',
  tags: ['practice', 'papers'],
  sections: [
    t(`## The three-pass method

From Keshav's widely-circulated note, and it works.

**Pass 1 (5–10 minutes).** Title, abstract, introduction, section headings, conclusions, and every figure. Then answer:
what problem, what approach, what is claimed. Decide whether to continue. Most papers stop here, correctly.

**Pass 2 (an hour).** Read the body, skip the proofs, examine figures and tables carefully. You should be able to
summarize the method and its evidence to someone else. Mark what you do not understand and which references you need.

**Pass 3 (several hours).** Reconstruct the work. Attempt the derivations. Ask what you would have done differently and
what the paper does not say. Reserve this for papers you actually need to build on.`),

    t(`## Read the figures first

For most ML papers, the figures and tables contain the entire result. Read them before the prose and form your own
view of what happened. Then read the text and see whether the authors' interpretation matches yours.

Specific things to check in a results table: is the baseline reasonable? Are the numbers compute-matched? Is there a
standard deviation? Is the bold number's advantage larger than the variation across nearby rows?`),

    t(`## Filtering

The volume is unmanageable — thousands of ML papers per month. Some heuristics:

- **Follow people, not keywords.** A handful of researchers whose taste you trust will surface more of value than any
  alert.
- **Wait two weeks.** Genuinely important work gets discussed, reproduced, and criticized. The signal improves enormously
  with a short delay, and you read far less.
- **Prefer papers with code**, and prefer code that has been run by someone other than the authors.
- **Read the classics.** They are dense with reusable ideas and are cited constantly. Attention Is All You Need,
  ResNet, Adam, GANs, DDPM, Chinchilla — a few dozen papers carry most of the field's conceptual weight.
- **Reproduce something.** The fastest way to actually understand a method is to implement it badly, then fix it.`),

    t(`## Where things are published

- **arXiv** — everything appears here first, unreviewed. cs.LG, cs.CL, cs.CV, stat.ML.
- **NeurIPS, ICML, ICLR** — the main conferences. ICLR reviews are public on OpenReview, which is often more
  illuminating than the paper.
- **ACL/EMNLP** (NLP), **CVPR/ICCV/ECCV** (vision).
- **Lab blogs and technical reports** — increasingly where frontier work appears, sometimes without peer review and
  sometimes without enough detail to reproduce.
- **Distill** (archived) and **transformer-circuits.pub** — exceptional interactive explanations.`),

    intuition(`A note on pace. It is easy to feel that the field moves so fast that learning fundamentals is wasted
effort. The opposite is true: the fundamentals are what let you read a new paper in twenty minutes instead of a week.

Backpropagation is 40 years old. Attention is 10. Cross-entropy, maximum likelihood, the bias-variance tradeoff, and
gradient descent are older still and are not going anywhere. The half-life of an architecture is a few years; the
half-life of the mathematics is indefinite.

Learn the parts that compound.`),

    t(`## A minimal reading list

If you read nothing else from the [reference library](#/library), read these, in this order:

1. **Attention Is All You Need** (2017) — the architecture everything runs on.
2. **Deep Residual Learning** (2015) — why deep networks train.
3. **Adam** (2014) — the optimizer you will use.
4. **Batch Normalization** (2015) + **Layer Normalization** (2016) — and the paper arguing the original explanation
   was wrong.
5. **Language Models are Few-Shot Learners** (2020) — in-context learning.
6. **Training Compute-Optimal LLMs** (2022) — how to allocate a budget.
7. **DDPM** (2020) — the other generative paradigm.
8. **LoRA** (2021) — how fine-tuning is actually done.
9. **A Mathematical Framework for Transformer Circuits** (2021) — how to think about what is inside.
10. **The Bitter Lesson** (2019) — one page, and it will change how you read the other nine.`),

    quiz('You have 30 minutes and a paper claiming a 2% improvement on a benchmark. What is the highest-value thing to check?',
      ['Whether the baseline was tuned as carefully as the proposed method, and whether variance across seeds is reported',
       'Whether the mathematics in the appendix is correct',
       'Whether the authors are from a well-known lab',
       'Whether the related work section is complete'],
      0,
      'A 2% gain is well within the range that a better-tuned baseline, a different seed, or extra training can produce. If the baseline got an afternoon and the method got three weeks, the comparison is uninformative regardless of how correct the mathematics is. Tuning effort and seed variance are the two things that most often explain away a small reported improvement.'),
  ],
  refs: [
    blog('How to Read a Paper', 'S. Keshav', 2007, 'https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf', 'Two pages. The three-pass method above.'),
    blog('The Bitter Lesson', 'Rich Sutton', 2019, 'http://www.incompleteideas.net/IncIdeas/BitterLesson.html', 'One page, and the most-cited blog post in the field.'),
    blog('Papers with Code', 'Meta AI', 2018, 'https://paperswithcode.com/', 'Papers linked to implementations and leaderboards.'),
    blog('Transformer Circuits Thread', 'Anthropic', 2021, 'https://transformer-circuits.pub/', 'The interpretability work, written with unusual care.'),
    blog('Lil\'Log', 'Lilian Weng', 2018, 'https://lilianweng.github.io/', 'Long-form technical surveys that are usually better than the papers they summarize.'),
  ],
},

];

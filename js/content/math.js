/* ============================================================
   Track 1 — Mathematical foundations
   ============================================================ */

import { t, key, intuition, warn, hist, mathnote, viz, deriv, code, quiz, paper, book, course, blog, video, demo, codeRef } from './_helpers.js';

export default [

/* ---------------------------------------------------------- */
{
  id: 'math-vectors',
  title: 'Vectors, Dot Products, and Geometry',
  sub: 'Everything in ML is a vector. Every question about similarity is a dot product.',
  mins: 20, level: 'foundations',
  tags: ['linear algebra', 'geometry'],
  sections: [
    t(`## Why start here

Open any ML paper and you will find $\\mathbf{x} \\in \\mathbb{R}^d$ in the first paragraph. An image is a vector of pixel
intensities. A sentence, after embedding, is a sequence of vectors. A neural network's weights are vectors. Whatever
you are doing — classifying, generating, retrieving — you are doing geometry in a high-dimensional space.

A **vector** is an ordered list of numbers, $\\mathbf{x} = (x_1, x_2, \\ldots, x_d)$. You can read it two ways, and you
need both:

- **As a point**: a location in $d$-dimensional space. "This image is *here*."
- **As an arrow**: a direction and a magnitude from the origin. "Move *this way* by *this much*."

The point reading is right for data. The arrow reading is right for gradients and updates.`),

    t(`## The two operations that matter

**Addition** is componentwise, and geometrically it is "walk along the first arrow, then the second":

$$\\mathbf{a} + \\mathbf{b} = (a_1+b_1,\\ \\ldots,\\ a_d+b_d)$$

**Scalar multiplication** stretches: $c\\mathbf{a} = (ca_1, \\ldots, ca_d)$. Negative $c$ flips the direction.

That's it. Those two operations, with their obvious algebraic properties, *define* a vector space. Anything obeying
them — polynomials, functions, images — can be treated with the same machinery.`),

    t(`## The dot product is a similarity meter

$$\\mathbf{a}\\cdot\\mathbf{b} = \\sum_{i=1}^{d} a_i b_i = \\|\\mathbf{a}\\|\\,\\|\\mathbf{b}\\|\\cos\\theta$$

Those two expressions being equal is the single most useful fact in this track. The left side is trivially computable;
the right side tells you what it *means*.`),

    viz('vector-playground'),

    key(`- $\\mathbf{a}\\cdot\\mathbf{b} > 0$: the vectors point in broadly the same direction.
- $\\mathbf{a}\\cdot\\mathbf{b} = 0$: they are **orthogonal** — perpendicular, uncorrelated, carrying independent information.
- $\\mathbf{a}\\cdot\\mathbf{b} < 0$: they oppose.

The **length** (or $L_2$ norm) is $\\|\\mathbf{a}\\| = \\sqrt{\\mathbf{a}\\cdot\\mathbf{a}}$, and **cosine similarity**
is the dot product with lengths divided out:
$\\cos\\theta = \\frac{\\mathbf{a}\\cdot\\mathbf{b}}{\\|\\mathbf{a}\\|\\|\\mathbf{b}\\|} \\in [-1, 1]$.`),

    intuition(`Cosine similarity ignores magnitude and keeps only direction. That is exactly what you want for text
embeddings — a long document and a short one about the same topic should count as similar, and their raw magnitudes
mostly reflect length, not meaning. It is why every vector database defaults to cosine.`),

    deriv('Where $\\|a\\|\\|b\\|\\cos\\theta$ comes from', `Start with the law of cosines applied to the triangle with sides $\\mathbf{a}$, $\\mathbf{b}$, and $\\mathbf{a}-\\mathbf{b}$:

$$\\|\\mathbf{a}-\\mathbf{b}\\|^2 = \\|\\mathbf{a}\\|^2 + \\|\\mathbf{b}\\|^2 - 2\\|\\mathbf{a}\\|\\|\\mathbf{b}\\|\\cos\\theta$$

Now expand the left side algebraically:

$$\\|\\mathbf{a}-\\mathbf{b}\\|^2 = (\\mathbf{a}-\\mathbf{b})\\cdot(\\mathbf{a}-\\mathbf{b}) = \\|\\mathbf{a}\\|^2 - 2\\,\\mathbf{a}\\cdot\\mathbf{b} + \\|\\mathbf{b}\\|^2$$

Setting the two right-hand sides equal, the $\\|\\mathbf{a}\\|^2$ and $\\|\\mathbf{b}\\|^2$ cancel, leaving
$-2\\,\\mathbf{a}\\cdot\\mathbf{b} = -2\\|\\mathbf{a}\\|\\|\\mathbf{b}\\|\\cos\\theta$. Divide by $-2$. ∎`),

    t(`## Projection

How much of $\\mathbf{a}$ points along $\\mathbf{b}$? Project it:

$$\\text{proj}_{\\mathbf{b}}(\\mathbf{a}) = \\frac{\\mathbf{a}\\cdot\\mathbf{b}}{\\mathbf{b}\\cdot\\mathbf{b}}\\,\\mathbf{b}$$

The leftover, $\\mathbf{a} - \\text{proj}_{\\mathbf{b}}(\\mathbf{a})$, is orthogonal to $\\mathbf{b}$ — turn on
"residual" in the figure above and check. **This decomposition into "explained by $\\mathbf b$" plus "orthogonal
residual" is the skeleton of least squares, PCA, and Gram–Schmidt.** You will meet it repeatedly.`),

    t(`## Other norms

The $L_2$ norm is not the only ruler:

| Norm | Formula | Unit ball | Where it shows up |
|---|---|---|---|
| $L_1$ | $\\sum_i \\|x_i\\|$ | diamond | Lasso, sparsity, robust losses |
| $L_2$ | $\\sqrt{\\sum_i x_i^2}$ | circle | ridge, weight decay, distances |
| $L_\\infty$ | $\\max_i \\|x_i\\|$ | square | adversarial robustness budgets |
| $L_0$ | count of nonzeros | — | "how sparse" (not a real norm) |

The shape of that unit ball is *not* trivia. When you constrain $\\|w\\|_1 \\le t$ you are constraining $w$ to a diamond,
and diamonds have corners on the axes — which is precisely why L1 produces exactly-zero coefficients and L2 does not.
You will see that picture in the regularization lesson.`),

    t(`## High dimensions are strange

Your 2-D and 3-D intuitions will mislead you. Some facts worth internalizing early:

- **Almost everything is orthogonal.** Two random unit vectors in $\\mathbb{R}^d$ have expected cosine similarity $0$ with
  standard deviation $\\approx 1/\\sqrt{d}$. In 1000 dimensions, random directions are essentially always perpendicular.
- **Volume flees to the shell.** Nearly all the volume of a high-dimensional ball sits in a thin skin near the surface.
- **Distances concentrate.** The ratio of farthest to nearest neighbor distance approaches 1, which is what breaks
  k-nearest-neighbours in high dimensions.

The first fact is oddly good news: it means a $d$-dimensional space can hold *far more than $d$* nearly-orthogonal
directions, which is why models can pack many more features than they have neurons. That is the phenomenon called
**superposition**, and there is a whole lesson on it later.`),

    code('Vectors in NumPy', `import numpy as np

a = np.array([2.0, 1.0, -0.5])
b = np.array([1.0, 3.0,  2.0])

print("a + b        =", a + b)
print("3a           =", 3 * a)
print("a . b        =", a @ b)              # @ is matmul/dot
print("|a|          =", np.linalg.norm(a))
print("cos(a, b)    =", (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# projection of a onto b, and the orthogonal residual
proj = (a @ b) / (b @ b) * b
resid = a - proj
print("proj_b(a)    =", proj.round(4))
print("residual . b =", (resid @ b).round(12), " <- zero, as promised")

# the concentration of measure, empirically
rng = np.random.default_rng(0)
for d in [2, 10, 100, 1000]:
    X = rng.normal(size=(2000, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    cos = (X[:1000] * X[1000:]).sum(1)
    print(f"d={d:5d}  mean|cos|={np.abs(cos).mean():.4f}  (1/sqrt(d)={1/np.sqrt(d):.4f})")`,
      'Run the last block and watch random vectors become orthogonal as dimension grows. This is not an accident of the sampler — it is the geometry.'),

    quiz('Two embedding vectors have cosine similarity 0.95 but very different norms. What does that tell you?',
      ['They encode nearly the same direction of meaning, but one is "stronger" — often just a longer or more frequent input',
       'They are nearly identical vectors',
       'They are almost orthogonal',
       'Nothing — cosine similarity is undefined for different norms'],
      0,
      'Cosine measures **direction only**. High cosine with different magnitudes is extremely common in embeddings: token frequency and sequence length inflate norms without changing semantics. This is exactly why retrieval systems normalize before comparing.'),
  ],
  refs: [
    book('Mathematics for Machine Learning', 'Deisenroth, Faisal & Ong', 2020, 'https://mml-book.github.io/', 'Free PDF. Chapters 2–3 cover this material properly, with ML motivation throughout. The best single reference for this track.'),
    video('Essence of Linear Algebra', '3Blue1Brown', 2016, 'https://www.3blue1brown.com/topics/linear-algebra', 'If the geometric picture has not clicked, watch this before reading anything else. Fifteen short episodes.'),
    book('Introduction to Linear Algebra', 'Gilbert Strang', 2016, 'https://math.mit.edu/~gs/linearalgebra/', 'The classic. Pair it with his MIT 18.06 lectures, which are free.'),
    course('MIT 18.06 Linear Algebra', 'Gilbert Strang', 2011, 'https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/', 'Full video lectures, problem sets, exams.'),
    blog('A Few Useful Things to Know About Machine Learning', 'Pedro Domingos', 2012, 'https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf', 'Section on the curse of dimensionality is the clearest short treatment anywhere.'),
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
    t(`## The shift in perspective

You were probably taught matrix multiplication as a procedure: row times column, sum, repeat. That is *how* to compute
it, and it tells you nothing about *what it is*.

Here is what it is. A matrix $A \\in \\mathbb{R}^{m\\times n}$ is a **linear function** from $\\mathbb{R}^n$ to $\\mathbb{R}^m$.
"Linear" means exactly two things:

$$A(\\mathbf{x}+\\mathbf{y}) = A\\mathbf{x} + A\\mathbf{y}, \\qquad A(c\\mathbf{x}) = cA\\mathbf{x}$$

Geometrically: gridlines stay straight, parallel, and evenly spaced, and the origin stays put.`),

    key(`**The columns of $A$ are the images of the basis vectors.** $A\\mathbf{e}_1$ is the first column, $A\\mathbf{e}_2$
the second, and so on. Once you know where the basis vectors land, linearity determines everything else:
$A\\mathbf{x} = x_1 (\\text{col}_1) + x_2(\\text{col}_2) + \\cdots$

This is the single most useful way to read a matrix. Matrix-vector multiplication is a **weighted sum of the columns**.`),

    viz('matrix-transform'),

    t(`## Matrix multiplication is function composition

$AB$ means "do $B$, then do $A$." That immediately explains two things students find arbitrary:

- **Why it is not commutative**: rotating then stretching ≠ stretching then rotating. Try it in the figure by setting
  a rotation preset and then editing the entries.
- **Why the dimensions must match**: the output space of $B$ must be the input space of $A$.

$$(AB)_{ij} = \\sum_k A_{ik}B_{kj}$$

The formula falls out of composing the two linear maps; it is a consequence, not a definition.`),

    t(`## Determinant, rank, and invertibility

The **determinant** $\\det A$ is the signed factor by which the transformation scales area (in 2-D) or volume
(in $n$-D). Watch it in the figure:

- $\\det A = 2$: areas double.
- $\\det A < 0$: space was flipped over.
- $\\det A = 0$: space was **squashed onto a lower-dimensional subspace**. Information was destroyed, and there is no
  way back — the matrix is not invertible.

The **rank** of $A$ is the dimension of its output space (the span of its columns). Full rank means nothing collapsed.
Rank deficiency is not a pathology to avoid; it is a resource. Low-rank structure is why:

- **PCA** works — data that looks $d$-dimensional often lives near a $k$-dimensional subspace.
- **LoRA** works — you can fine-tune a huge weight matrix by adding a rank-8 update, because the *change* needed is
  low-rank even when the weights are not.
- **Recommender systems** work — a user–item matrix is approximately rank-50.`),

    t(`## Special matrices worth recognizing on sight

| Type | Condition | Why it matters |
|---|---|---|
| Symmetric | $A = A^{\\mathsf T}$ | Real eigenvalues, orthogonal eigenvectors. Covariance and Hessians are symmetric. |
| Orthogonal | $A^{\\mathsf T}A = I$ | Rotation/reflection. Preserves lengths and angles. $A^{-1}=A^{\\mathsf T}$ — free inverse. |
| Positive definite | $\\mathbf{x}^{\\mathsf T}A\\mathbf{x} > 0\\ \\forall \\mathbf{x}\\neq 0$ | Bowl-shaped quadratic → unique minimum. Covariance matrices, well-behaved Hessians. |
| Diagonal | zero off-diagonal | Independent scaling per axis. Trivial to invert. |
| Low-rank | $\\text{rank} \\ll \\min(m,n)$ | Compressible: $A \\approx UV^{\\mathsf T}$ with skinny $U, V$. |

**The transpose** $A^{\\mathsf T}$ swaps rows and columns, and the identity $(\\mathbf{x}, A\\mathbf{y}) = (A^{\\mathsf T}\\mathbf{x}, \\mathbf{y})$
is what makes backpropagation work: the backward pass through a linear layer multiplies by $W^{\\mathsf T}$.`),

    intuition(`When you see $W^{\\mathsf T}\\delta$ in a backprop derivation, do not read it as "the transpose for algebraic
reasons." Read it as: *the forward pass sent information from layer $\\ell$ to layer $\\ell+1$ through $W$; the backward
pass sends blame back along the same wires, and $W^{\\mathsf T}$ is the matrix that reverses the routing.*`),

    code('Matrices in NumPy', `import numpy as np

A = np.array([[2.0, 1.0],
              [0.0, 1.5]])
x = np.array([1.0, 1.0])

print("A @ x            =", A @ x)
print("as column sum    =", x[0]*A[:,0] + x[1]*A[:,1], " <- identical")
print("det A            =", np.linalg.det(A))
print("rank A           =", np.linalg.matrix_rank(A))

B = np.array([[0.0, -1.0], [1.0, 0.0]])       # 90 degree rotation
print("AB != BA         :", not np.allclose(A@B, B@A))

# rank deficiency: a rank-1 matrix, and what it does to space
u = np.array([[1.0], [2.0]])
v = np.array([[3.0, 1.0]])
R = u @ v
print("rank(u v^T)      =", np.linalg.matrix_rank(R), " det =", round(np.linalg.det(R), 12))

# every output lands on a single line — the span of u
pts = np.random.default_rng(0).normal(size=(5, 2))
out = pts @ R.T
print("all outputs parallel to u?",
      np.allclose(np.cross(out, u.ravel()), 0))`,
      'The last check is worth pausing on: a rank-1 matrix maps the *entire plane* onto one line. That is what a zero determinant means concretely.'),

    quiz('You fine-tune a 4096×4096 weight matrix with LoRA at rank 8. How many parameters do you train, versus full fine-tuning?',
      ['65,536 vs 16,777,216 — about 0.4%',
       '8 vs 16,777,216',
       '4096 vs 16,777,216',
       'The same number; LoRA only changes the optimizer'],
      0,
      'LoRA writes $\\Delta W = BA$ with $B \\in \\mathbb{R}^{4096\\times 8}$ and $A \\in \\mathbb{R}^{8 \\times 4096}$, so $2 \\times 4096 \\times 8 = 65{,}536$ trainable parameters against $4096^2 = 16.8$M. The bet — empirically a good one — is that the *update* a task needs is low-rank even though the pretrained weights are not.'),
  ],
  refs: [
    video('Linear transformations and matrices', '3Blue1Brown', 2016, 'https://www.youtube.com/watch?v=kYB8IZa5AuE', 'The single best five minutes on why columns are the images of basis vectors.'),
    book('Mathematics for Machine Learning, Ch. 2', 'Deisenroth, Faisal & Ong', 2020, 'https://mml-book.github.io/', 'Rigorous treatment of vector spaces, rank, and linear maps.'),
    paper('LoRA: Low-Rank Adaptation of Large Language Models', 'Hu et al.', 2021, 'https://arxiv.org/abs/2106.09685', 'The low-rank-update idea, and the empirical evidence that fine-tuning updates really do have low intrinsic rank.'),
    book('Matrix Computations', 'Golub & Van Loan', 2013, 'https://epubs.siam.org/doi/book/10.1137/1.9781421407944', 'The reference when you need numerical detail rather than intuition.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'math-eigen-svd',
  title: 'Eigenvectors, SVD, and Low-Rank Structure',
  sub: 'The directions a matrix does not rotate, and the universal decomposition.',
  mins: 30, level: 'foundations',
  prereq: ['math-matrices'],
  tags: ['linear algebra', 'SVD', 'PCA'],
  sections: [
    t(`## Eigenvectors: the axes of a transformation

Most vectors get rotated *and* stretched by a matrix. A few special ones only get stretched:

$$A\\mathbf{v} = \\lambda \\mathbf{v}$$

$\\mathbf{v}$ is an **eigenvector**, $\\lambda$ its **eigenvalue**. These are the natural axes of the transformation —
along them, the matrix acts like plain multiplication by a number.

Go back to the [matrix transformation figure](#/l/math-matrices) and watch the thick arrows: they are the eigenvectors,
and they stay on their own line no matter what you do to the other entries (until the eigenvalues become complex,
which is what happens for rotations — nothing stays put under a rotation, and that is exactly why).`),

    t(`### Why they matter in practice

Repeatedly applying $A$ is trivial in the eigenbasis: $A^k\\mathbf{v} = \\lambda^k\\mathbf{v}$. So:

- **Stability**: a recurrent network's hidden state is roughly $h_t \\approx W^t h_0$. If the largest $|\\lambda| > 1$
  the state explodes; if $< 1$ it decays. That single fact is the vanishing/exploding gradient problem.
- **Curvature**: the eigenvalues of the Hessian are the curvatures of the loss surface along its principal axes. Their
  ratio, the **condition number**, determines how badly gradient descent zig-zags.
- **Variance**: the eigenvectors of the covariance matrix are the principal components.`),

    viz('quadratic-form'),

    mathnote(`For **symmetric** matrices (covariance, Hessian, Gram matrices — most of the ones you meet) the spectral
theorem guarantees: all eigenvalues are real, and eigenvectors for distinct eigenvalues are orthogonal. So
$A = Q\\Lambda Q^{\\mathsf T}$ with $Q$ orthogonal. That is a *very* strong structure and it is why symmetric matrices
are so pleasant.

A symmetric matrix is **positive definite** iff all $\\lambda_i > 0$, which is exactly the condition for
$\\mathbf{x}^{\\mathsf T}A\\mathbf{x}$ to be a bowl with a unique minimum.`),

    t(`## SVD: the decomposition that always exists

Eigendecomposition needs a square matrix, and even then may not exist over the reals. The **singular value
decomposition** has no such caveats. *Every* matrix — square or not, any rank — factors as:

$$A = U\\Sigma V^{\\mathsf T}$$

with $U$ and $V$ orthogonal and $\\Sigma$ diagonal with non-negative entries $\\sigma_1 \\ge \\sigma_2 \\ge \\cdots \\ge 0$.

In words: **every linear map is a rotation, then an axis-aligned stretch, then another rotation.** Nothing else is
possible. That is a remarkable structural fact about linear algebra.`),

    viz('svd-demo'),

    t(`## Low-rank approximation, and why it is everywhere

Write the SVD as a sum of rank-1 pieces:

$$A = \\sum_{i=1}^{r} \\sigma_i\\, \\mathbf{u}_i \\mathbf{v}_i^{\\mathsf T}$$

Keep only the largest $k$ terms and you get $A_k$, the **best rank-$k$ approximation of $A$ in both Frobenius and
spectral norm** (the Eckart–Young theorem). Not a good one — the provably optimal one.

This single theorem underwrites:

| Application | What gets truncated |
|---|---|
| PCA | Covariance eigenvalues → keep top-$k$ directions of variance |
| Image compression | Singular values of the pixel matrix |
| Latent semantic analysis | Term–document matrix |
| Recommender systems | User–item matrix |
| LoRA | The *update* to a weight matrix, constrained to rank $r$ |
| Model compression | Weight matrices factored into two skinny ones |

The **condition number** $\\kappa = \\sigma_1/\\sigma_r$ measures numerical sensitivity: how much a small perturbation of
the input can change the output. Large $\\kappa$ means an ill-conditioned problem, slow optimization, and unreliable
solutions.`),

    deriv('Connecting SVD to eigendecomposition', `Consider $A^{\\mathsf T}A$, which is symmetric and positive semi-definite:

$$A^{\\mathsf T}A = (U\\Sigma V^{\\mathsf T})^{\\mathsf T}(U\\Sigma V^{\\mathsf T}) = V\\Sigma^{\\mathsf T}U^{\\mathsf T}U\\Sigma V^{\\mathsf T} = V\\Sigma^2 V^{\\mathsf T}$$

using $U^{\\mathsf T}U = I$. So the **right singular vectors $V$ are the eigenvectors of $A^{\\mathsf T}A$**, and the
singular values are the square roots of its eigenvalues. Symmetrically, $AA^{\\mathsf T} = U\\Sigma^2 U^{\\mathsf T}$ gives
the left singular vectors.

This also explains why PCA is usually described *both* as "eigendecomposition of the covariance matrix" and as
"SVD of the centered data matrix" — for centered data $X$, the covariance is $\\frac{1}{n}X^{\\mathsf T}X$, so they are
the same computation. In practice you do the SVD directly: forming $X^{\\mathsf T}X$ squares the condition number and
loses precision.`),

    code('SVD and rank truncation', `import numpy as np

rng = np.random.default_rng(0)

# a matrix that is secretly rank 3, plus noise
U_true = rng.normal(size=(50, 3))
V_true = rng.normal(size=(3, 40))
A = U_true @ V_true + 0.05 * rng.normal(size=(50, 40))

U, S, Vt = np.linalg.svd(A, full_matrices=False)
print("singular values:", np.round(S[:8], 3))
print("  ^ notice the cliff after the 3rd\\n")

def rank_k(k):
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k]

for k in [1, 2, 3, 5, 10]:
    err = np.linalg.norm(A - rank_k(k)) / np.linalg.norm(A)
    print(f"rank {k:2d}: relative error {err:.4f}   "
          f"storage {(50*k + 40*k)/ (50*40):.1%} of full")

print("\\ncondition number:", round(S[0]/S[-1], 1))

# Eckart-Young: no rank-3 matrix does better than the truncated SVD
best = np.linalg.norm(A - rank_k(3))
for _ in range(5):
    Ur = rng.normal(size=(50,3)); Vr = rng.normal(size=(3,40))
    # least-squares fit of a random-subspace rank-3 approximation
    P = Ur @ np.linalg.lstsq(Ur, A, rcond=None)[0]
    print("random rank-3 subspace error:", round(np.linalg.norm(A-P), 3),
          " vs SVD:", round(best, 3))`,
      'The singular value spectrum tells you the *effective* rank of your data. A sharp cliff means genuine low-dimensional structure; a slow decay means the data really does fill the space.'),

    quiz('Your loss surface has Hessian eigenvalues 100 and 0.1. What should you expect from plain gradient descent?',
      ['Severe zig-zagging: the stable learning rate is capped by λ_max=100 while progress along the λ=0.1 direction is 1000× slower',
       'Fast convergence, since one direction has strong curvature',
       'Divergence regardless of learning rate',
       'Nothing in particular — eigenvalues do not affect gradient descent'],
      0,
      'The condition number is $\\kappa = 1000$. Stability requires $\\eta < 2/\\lambda_{\\max} = 0.02$, but with that $\\eta$ the flat direction moves at rate $\\eta\\lambda_{\\min} = 0.002$ per step. Convergence takes $O(\\kappa)$ iterations. This is precisely what momentum, Adam, and normalization layers exist to mitigate — try it yourself in the [gradient descent figure](#/l/math-optimization).'),
  ],
  refs: [
    paper('The approximation of one matrix by another of lower rank', 'Eckart & Young', 1936, 'https://doi.org/10.1007/BF02288367', 'The original optimality theorem for truncated SVD.'),
    video('Eigenvectors and eigenvalues', '3Blue1Brown', 2016, 'https://www.youtube.com/watch?v=PFDu9oVAE-g', ''),
    blog('The Extraordinary Power of the SVD', 'Gilbert Strang / MIT', 2019, 'https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/', 'MIT 18.065 is built around exactly this material for ML.'),
    paper('Finding Structure with Randomness', 'Halko, Martinsson & Tropp', 2011, 'https://arxiv.org/abs/0909.4061', 'Randomized SVD — how you actually compute this on matrices too big to fit in memory.'),
    book('Numerical Linear Algebra', 'Trefethen & Bau', 1997, 'https://people.maths.ox.ac.uk/trefethen/text.html', 'Beautifully written. Lectures 4–5 on SVD are worth the price alone.'),
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
    t(`## What a derivative is

$$f'(x) = \\lim_{h\\to 0}\\frac{f(x+h)-f(x)}{h}$$

Three readings, all correct, all useful:

1. **Slope** of the tangent line at $x$.
2. **Rate of change**: "if I nudge $x$ by $\\epsilon$, $f$ changes by about $f'(x)\\epsilon$."
3. **Best local linear approximation**: $f(x+h) \\approx f(x) + f'(x)h$.

Reading 3 is the one that generalizes and the one that matters for optimization.`),

    viz('derivative-tangent'),

    t(`## Gradients: derivatives in many variables

For $f: \\mathbb{R}^d \\to \\mathbb{R}$ (a loss function, say), the **gradient** collects all the partial derivatives:

$$\\nabla f(\\mathbf{x}) = \\left(\\frac{\\partial f}{\\partial x_1},\\ \\ldots,\\ \\frac{\\partial f}{\\partial x_d}\\right)$$

Each partial answers "how does $f$ change if I move along axis $i$ and hold everything else fixed?"

Assembled into a vector, the gradient acquires two properties that are not obvious from the definition:`),

    key(`1. **$\\nabla f$ points in the direction of steepest ascent**, and $\\|\\nabla f\\|$ is how steep.
2. **$\\nabla f$ is perpendicular to the level set** (contour line) through that point.

Property 1 is why gradient descent moves along $-\\nabla f$. Property 2 is why the descent direction and the contours
meet at right angles in every picture you have seen.`),

    viz('gradient-field'),

    deriv('Why the gradient is the steepest direction', `The **directional derivative** of $f$ along a unit vector $\\mathbf{u}$ is

$$D_{\\mathbf{u}}f = \\lim_{h\\to 0}\\frac{f(\\mathbf{x}+h\\mathbf{u})-f(\\mathbf{x})}{h} = \\nabla f \\cdot \\mathbf{u}$$

We want the $\\mathbf{u}$ maximizing this, subject to $\\|\\mathbf{u}\\|=1$. By the geometric form of the dot product,

$$\\nabla f\\cdot\\mathbf{u} = \\|\\nabla f\\|\\,\\|\\mathbf{u}\\|\\cos\\theta = \\|\\nabla f\\|\\cos\\theta$$

which is maximized when $\\cos\\theta = 1$, i.e. $\\mathbf{u}$ points along $\\nabla f$. ∎

The same computation gives property 2: moving along a level set means $f$ does not change, so $D_{\\mathbf u}f = 0$,
so $\\nabla f\\cdot\\mathbf{u}=0$ — the gradient is orthogonal to every direction that stays on the contour.`),

    t(`## The chain rule is the whole of backpropagation

For composed functions, derivatives multiply:

$$\\frac{d}{dx}g(h(x)) = g'(h(x))\\cdot h'(x)$$

A neural network is a deep composition $f_L \\circ f_{L-1}\\circ\\cdots\\circ f_1$, so the gradient with respect to an
early parameter is a **product of many local derivatives**. That product structure explains almost every training
pathology you will encounter:`),

    viz('chain-rule'),

    warn(`If each factor is slightly less than 1, the product decays exponentially with depth — **vanishing gradients**,
and early layers stop learning. If each is slightly more than 1, it explodes. Neither is a bug in the algorithm; it is
arithmetic. The fixes (careful initialization, normalization layers, residual connections) all work by controlling
the magnitude of those factors.`),

    t(`## Second derivatives: curvature

The **Hessian** $H_{ij} = \\frac{\\partial^2 f}{\\partial x_i \\partial x_j}$ collects the second derivatives. It tells
you how the gradient itself changes:

- All eigenvalues positive → local **minimum** (bowl).
- All negative → local **maximum** (dome).
- Mixed signs → **saddle point**.

In high dimensions, saddle points vastly outnumber local minima — for a random landscape, having *all* $d$ eigenvalues
come out negative is exponentially unlikely. This was an important realization around 2014: deep learning's optimization
problem is not "escaping bad local minima" so much as "escaping saddles and plateaus," and gradient noise handles that
reasonably well.`),

    t(`## Taylor series: the approximation hierarchy

$$f(x+h) = f(x) + f'(x)h + \\tfrac{1}{2}f''(x)h^2 + \\tfrac{1}{6}f'''(x)h^3 + \\cdots$$

Truncate at order 1 and you have what gradient descent believes. Truncate at order 2 and you have what Newton's method
believes. Both are only valid **near** $x$ — which is the real reason for small learning rates.`),

    viz('taylor-approx'),

    code('Gradients by hand, by finite difference, and by autograd', `import numpy as np

def f(w):
    """A small loss: least squares with a nonlinearity."""
    x = np.array([1.0, 2.0, -1.0])
    y = 0.7
    pred = np.tanh(w @ x)
    return (pred - y) ** 2

def grad_analytic(w):
    x = np.array([1.0, 2.0, -1.0])
    y = 0.7
    z = w @ x
    pred = np.tanh(z)
    # dL/dw = dL/dpred * dpred/dz * dz/dw   <- the chain rule, factor by factor
    return 2 * (pred - y) * (1 - pred**2) * x

def grad_numeric(w, h=1e-6):
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w); e[i] = h
        g[i] = (f(w + e) - f(w - e)) / (2 * h)     # central difference
    return g

w = np.array([0.3, -0.5, 0.8])
print("analytic:", grad_analytic(w).round(8))
print("numeric :", grad_numeric(w).round(8))
print("max abs difference:", np.abs(grad_analytic(w) - grad_numeric(w)).max())

# gradient checking: the standard sanity test for a hand-written backward pass
rel = np.abs(grad_analytic(w) - grad_numeric(w)).max() / (np.abs(grad_analytic(w)).max() + 1e-12)
print("relative error:", f"{rel:.2e}", "->", "PASS" if rel < 1e-5 else "FAIL")`,
      'Gradient checking with central differences is how you verify a hand-written backward pass. The central difference has $O(h^2)$ error versus $O(h)$ for the forward difference, so it is worth the extra function evaluation. Too small an $h$ and floating-point cancellation ruins it — $10^{-5}$ to $10^{-7}$ is the usable window in float64.'),

    quiz('A deep sigmoid network trains fine for 3 layers but stops learning at 20. The most likely cause is:',
      ["Gradients vanish: σ' ≤ 0.25, so 20 layers multiply to at most 0.25²⁰ ≈ 10⁻¹²",
       'The learning rate is too high',
       'There is not enough training data',
       'The loss function is wrong'],
      0,
      "Each sigmoid contributes a factor of at most 0.25 to the backward product, and typically much less since it saturates. Over 20 layers this annihilates the gradient reaching the early weights. Historically this is exactly what blocked deep networks until ReLU (gradient of 1 on the positive side), better initialization, and residual connections arrived. See the [vanishing gradients figure](#/l/nn-backprop)."),
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
    t(`## The Jacobian

For a function $f:\\mathbb{R}^n\\to\\mathbb{R}^m$, the derivative is not a number or a vector — it is an $m\\times n$
**matrix** of all the partials:

$$J_{ij} = \\frac{\\partial f_i}{\\partial x_j}$$

And it means the same thing derivatives always mean: $f(\\mathbf{x}+\\mathbf{h}) \\approx f(\\mathbf{x}) + J\\mathbf{h}$.
The Jacobian *is* the best local linear map.`),

    viz('jacobian'),

    t(`## The chain rule with Jacobians

Composition means matrix multiplication:

$$J_{g\\circ f}(\\mathbf{x}) = J_g(f(\\mathbf{x}))\\; J_f(\\mathbf{x})$$

For a network $L$ layers deep, the gradient of the loss with respect to layer-1 activations is

$$\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{h}_1} = \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{h}_L}\\, J_L\\, J_{L-1}\\cdots J_2$$

a product of $L-1$ Jacobians.`),

    key(`**Backprop never builds these matrices.** For a layer with 4096 inputs and outputs, the Jacobian is
$4096^2 = 16.8$M entries — per example, per layer. Instead, autodiff computes **vector–Jacobian products**
$\\mathbf{v}^{\\mathsf T}J$ directly. For a linear layer $\\mathbf{y}=W\\mathbf{x}$ the VJP is just
$\\mathbf{v}^{\\mathsf T}W$, i.e. one matrix-vector multiply. For an elementwise activation the Jacobian is diagonal, so
the VJP is an elementwise multiply.

This is the entire computational trick behind reverse-mode automatic differentiation.`),

    t(`## Forward mode vs reverse mode

There are two ways to accumulate that product of Jacobians:

- **Forward mode** computes $J\\mathbf{v}$ (Jacobian–vector products), going left to right through the network. Cost
  scales with the number of *inputs*.
- **Reverse mode** computes $\\mathbf{v}^{\\mathsf T}J$ (vector–Jacobian products), going right to left. Cost scales with
  the number of *outputs*.

Deep learning has millions of inputs (parameters) and exactly one output (the scalar loss). So reverse mode wins by a
factor of millions, and "backpropagation" is just reverse-mode autodiff applied to a neural network.

The price is memory: reverse mode must store the forward activations to use in the backward pass. That is what
**gradient checkpointing** trades away — recompute some activations instead of storing them.`),

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
    t(`## Why probability

Every ML model is a claim about a distribution. A classifier outputs $p(y \\mid \\mathbf{x})$. A language model outputs
$p(\\text{next token}\\mid \\text{context})$. A diffusion model *is* a sampler for $p(\\text{image})$. Even a plain
regression is a distributional claim in disguise — it says $y = f(\\mathbf{x}) + \\epsilon$ with $\\epsilon$ Gaussian,
which is why least squares is the right loss.

Get comfortable with three objects and the rest follows: distributions, expectations, and conditioning.`),

    t(`## Random variables and distributions

A **random variable** is a quantity whose value depends on chance. Discrete ones have a probability mass function
$p(x) = P(X=x)$; continuous ones have a density $p(x)$ where $P(a\\le X\\le b)=\\int_a^b p(x)\\,dx$.

For a density, $p(x)$ can exceed 1 — it is a density, not a probability. Only its integral must equal 1.`),

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

$$\\mathbb{E}[X] = \\sum_x x\\,p(x) \\quad\\text{or}\\quad \\int x\\,p(x)\\,dx, \\qquad
\\text{Var}(X) = \\mathbb{E}[(X-\\mathbb{E}[X])^2] = \\mathbb{E}[X^2]-\\mathbb{E}[X]^2$$

Two properties do most of the work:

- **Linearity**: $\\mathbb{E}[aX+bY] = a\\mathbb{E}[X]+b\\mathbb{E}[Y]$, *always*, even for dependent variables. This is
  why minibatch gradients are unbiased estimates of the full gradient.
- **Variance of a sum**: $\\text{Var}(X+Y) = \\text{Var}(X)+\\text{Var}(Y)$ only when they are **uncorrelated**. This is
  why averaging $B$ independent samples reduces variance by $B$ — and why a batch of $B$ gradients has noise
  $\\propto 1/\\sqrt{B}$.`),

    viz('clt'),

    t(`## Conditioning and Bayes' rule

The joint, marginal, and conditional are related by

$$p(x,y) = p(x\\mid y)\\,p(y), \\qquad p(x) = \\sum_y p(x,y)$$

Rearranging gives **Bayes' rule**:

$$\\underbrace{p(\\theta\\mid \\mathcal{D})}_{\\text{posterior}} = \\frac{\\overbrace{p(\\mathcal{D}\\mid\\theta)}^{\\text{likelihood}}\\ \\overbrace{p(\\theta)}^{\\text{prior}}}{\\underbrace{p(\\mathcal{D})}_{\\text{evidence}}}$$

Read it as a procedure: *start with a belief, see data, update.* The denominator is just a normalizing constant, which
is why you will constantly see $p(\\theta\\mid\\mathcal D) \\propto p(\\mathcal D\\mid\\theta)p(\\theta)$.`),

    viz('bayes-coin'),

    warn(`**The base rate trap.** A test is 99% accurate for a disease affecting 1 in 10,000. You test positive. The
probability you are sick is about **1%**, not 99%.

$$P(\\text{sick}\\mid+) = \\frac{0.99 \\times 0.0001}{0.99\\times 0.0001 + 0.01\\times 0.9999} \\approx 0.0098$$

Among 10,000 people, 1 is sick (and tests positive), while ~100 healthy people also test positive. The prior dominates.
This is the same arithmetic that makes precision collapse for rare-class classifiers — see the
[ROC and prevalence figure](#/l/ml-evaluation).`),

    t(`## Maximum likelihood

Given data $\\mathcal{D}=\\{x_1,\\ldots,x_n\\}$ assumed i.i.d. from $p(x\\mid\\theta)$, the likelihood is
$\\prod_i p(x_i\\mid\\theta)$. We maximize its logarithm, because sums are easier than products and floats do not
underflow:

$$\\hat\\theta_{\\text{MLE}} = \\arg\\max_\\theta \\sum_{i=1}^n \\log p(x_i\\mid\\theta)$$

**MAP** adds a prior: $\\arg\\max_\\theta [\\sum_i \\log p(x_i\\mid\\theta) + \\log p(\\theta)]$. And that extra term is
*exactly* regularization: a Gaussian prior on the weights gives you L2 / weight decay; a Laplace prior gives L1.`),

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
    t(`## Surprise, quantified

How surprising is an outcome with probability $p$? Whatever function we choose should be (a) decreasing in $p$,
(b) zero when $p=1$, and (c) additive for independent events. Only one function satisfies all three:

$$I(x) = -\\log p(x)$$

In bits (log base 2) or nats (base $e$). An event with $p=1/2$ carries 1 bit; $p=1/1024$ carries 10 bits.

**Entropy** is average surprise:

$$H(p) = -\\sum_x p(x)\\log p(x) = \\mathbb{E}_{x\\sim p}[-\\log p(x)]$$

It is maximal for the uniform distribution (maximum uncertainty) and zero for a point mass (no uncertainty). Shannon's
source coding theorem gives it a hard operational meaning: **$H(p)$ is the minimum average number of bits needed to
encode samples from $p$.**`),

    t(`## Cross-entropy: coding with the wrong model

Suppose the truth is $p$ but you built your code assuming $q$. Your average code length is

$$H(p,q) = -\\sum_x p(x)\\log q(x)$$

You pay a penalty for being wrong, and that penalty is the **KL divergence**:

$$D_{\\text{KL}}(p\\,\\|\\,q) = H(p,q) - H(p) = \\sum_x p(x)\\log\\frac{p(x)}{q(x)} \\ \\ge 0$$

with equality iff $p=q$ (Gibbs' inequality).`),

    viz('entropy-kl'),

    key(`Training a classifier or a language model minimizes cross-entropy $H(p, q_\\theta)$ where $p$ is the empirical
data distribution. Since $H(p)$ does not depend on $\\theta$, **minimizing cross-entropy is identical to minimizing
$D_{\\text{KL}}(p\\|q_\\theta)$**, which is identical to maximum likelihood.

Three names, one objective. When a paper says "we minimize the KL to the data distribution" and another says "we
maximize likelihood," they are doing the same arithmetic.`),

    t(`## KL is not symmetric, and the asymmetry matters

$D_{\\text{KL}}(p\\|q) \\ne D_{\\text{KL}}(q\\|p)$, and choosing which one to minimize changes the answer qualitatively:

- **Forward KL, $D(p\\|q)$ — "mass covering."** The term $p(x)\\log\\frac{p(x)}{q(x)}$ blows up wherever $p$ has mass
  and $q$ does not. So $q$ is forced to cover everything $p$ does, even if that means smearing over regions $p$
  ignores. Maximum likelihood minimizes this. It is why a Gaussian fit to a bimodal distribution lands in the middle,
  covering both modes badly.
- **Reverse KL, $D(q\\|p)$ — "mode seeking."** Now the penalty is for $q$ putting mass where $p$ has none. $q$ can
  safely ignore modes; it just must not hallucinate. Variational inference minimizes this, which is why VI is famous
  for underestimating posterior variance, and why RL-with-KL-penalty to a reference policy behaves the way it does.`),

    t(`## Perplexity

Language modeling reports **perplexity** rather than loss:

$$\\text{PPL} = \\exp\\!\\left(\\frac{1}{N}\\sum_{i=1}^{N} -\\log q(x_i\\mid x_{<i})\\right) = e^{\\text{cross-entropy}}$$

It has a useful interpretation: *the effective number of equally likely choices the model is deciding between at each
step.* Perplexity 10 means the model is about as uncertain as if it were picking uniformly among 10 tokens.

A caution: perplexity depends on the tokenizer. A model with a larger vocabulary compresses more text per token, so
its per-token perplexity is not comparable to a model with a smaller one. Bits-per-byte is the tokenizer-independent
alternative.`),

    t(`## Mutual information

$$I(X;Y) = D_{\\text{KL}}\\big(p(x,y)\\,\\|\\,p(x)p(y)\\big) = H(X) - H(X\\mid Y)$$

"How many bits does knowing $Y$ save me when describing $X$?" Zero iff independent. It appears in feature selection,
in the information bottleneck view of representation learning, and — importantly — as the quantity that contrastive
objectives like InfoNCE (used by CLIP and SimCLR) lower-bound.`),

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
    t(`## The setup

Training is minimization. You have parameters $\\theta$, a loss $\\mathcal{L}(\\theta)$ averaged over data, and you want

$$\\theta^* = \\arg\\min_\\theta \\mathcal{L}(\\theta)$$

For a handful of models (linear regression, some SVMs) there is a closed form. For everything else you iterate:

$$\\theta_{t+1} = \\theta_t - \\eta\\,\\nabla\\mathcal{L}(\\theta_t)$$

Take the steepest downhill direction, step a little, repeat.`),

    viz('gradient-descent'),

    t(`## Learning rate: the one hyperparameter that will ruin your day

On a quadratic $\\mathcal{L}=\\tfrac12 a\\theta^2$, one step maps $\\theta \\mapsto (1-\\eta a)\\theta$. Convergence requires
$|1-\\eta a|<1$, that is:

$$\\eta < \\frac{2}{a} = \\frac{2}{\\lambda_{\\max}(H)}$$

Above that, you diverge geometrically. Below $1/\\lambda_{\\max}$ you descend monotonically; between the two you
oscillate while converging.`),

    viz('lr-stability'),

    key(`In a deep network, $\\lambda_{\\max}$ is the sharpest curvature *anywhere in the whole parameter space*. So a single
badly-conditioned direction caps the learning rate for **every** parameter. That is the core problem, and everything
below is a response to it:

- **Momentum / Adam** — adapt the effective step per direction.
- **Normalization layers** — reshape the landscape so curvature is more uniform.
- **Warmup** — avoid the early phase where curvature estimates are unreliable.
- **Gradient clipping** — survive the occasional sharp region without blowing up.`),

    t(`## Stochastic gradient descent

Computing $\\nabla\\mathcal{L}$ over millions of examples per step is wasteful. Use a minibatch:

$$\\nabla\\mathcal{L}(\\theta) \\approx \\frac{1}{B}\\sum_{i\\in\\text{batch}} \\nabla\\ell_i(\\theta)$$

This is an **unbiased** estimate (by linearity of expectation) with variance $\\propto 1/B$. Two consequences:

1. You get roughly $N/B$ times more steps for the same compute. Almost always worth it.
2. The noise is not purely a cost. It helps escape saddle points and shallow minima, and it biases SGD toward flatter
   regions of the loss surface, which correlates with better generalization.`),

    t(`## Momentum

Plain GD in a narrow valley oscillates across the walls while creeping along the floor. Momentum accumulates a velocity:

$$v_{t+1} = \\beta v_t - \\eta\\nabla\\mathcal{L}(\\theta_t), \\qquad \\theta_{t+1} = \\theta_t + v_{t+1}$$

Oscillating components cancel across steps; consistent components accumulate. With $\\beta=0.9$ the effective step in a
consistent direction is amplified by $\\frac{1}{1-\\beta}=10\\times$.

**Nesterov momentum** evaluates the gradient at the *look-ahead* point $\\theta_t + \\beta v_t$, which corrects the
overshoot a step earlier. Small change, consistently slightly better.`),

    t(`## Adaptive methods: RMSProp and Adam

Different parameters need different step sizes — a rarely-active feature's weight should move further per gradient
than a constantly-active one. Adaptive methods estimate a per-parameter scale from the gradient history.

**Adam** combines momentum (first moment) with per-parameter scaling (second moment):

$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1)g_t \\qquad v_t = \\beta_2 v_{t-1}+(1-\\beta_2)g_t^2$$
$$\\hat m_t = \\frac{m_t}{1-\\beta_1^t}, \\quad \\hat v_t = \\frac{v_t}{1-\\beta_2^t}, \\qquad
\\theta_{t+1} = \\theta_t - \\eta\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon}$$

Defaults $\\beta_1=0.9,\\beta_2=0.999,\\epsilon=10^{-8}$ work remarkably often. The bias-correction terms matter early,
when $m$ and $v$ are still initialized near zero.`),

    warn(`**Adam has a memory cost.** It stores two extra float32 values per parameter, so optimizer state is
$8$ bytes/param on top of weights and gradients — often the largest single term in your training memory budget. This is
what 8-bit optimizers, Adafactor, and Lion are trying to reduce.

**AdamW ≠ Adam + L2.** Adding $\\lambda\\|\\theta\\|^2$ to the loss puts the penalty through Adam's adaptive scaling, so
parameters with large historical gradients get *less* decay — usually not what you want. AdamW applies
$\\theta \\mathrel{-}= \\eta\\lambda\\theta$ separately, decoupled from the adaptive term. **Use AdamW.** It is the default
for essentially all transformer training.`),

    viz('lr-schedules'),

    t(`## Convexity, and how much we should care

A convex function has one minimum, and any local minimum is global. Linear and logistic regression, SVMs, and Lasso
are convex — solvers just work.`),

    viz('convexity'),

    t(`Neural networks are wildly non-convex, with exponentially many critical points. Empirically this matters far less
than 1990s theory predicted:

- Most critical points in high dimensions are **saddles**, not local minima, and gradient noise escapes them.
- Overparameterized networks have vast connected regions of near-optimal loss; you do not need *the* minimum.
- The minima SGD finds tend to be flat, and flatness correlates with generalization.

None of this is fully explained. It is one of the genuinely open theoretical questions in the field.`),

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
    t(`## Floats are not real numbers

A floating-point number is $\\pm m \\times 2^{e}$ — a sign bit, some exponent bits (setting the *range*), and some
mantissa bits (setting the *precision*). The split between exponent and mantissa is the entire design space.`),

    viz('float-precision'),

    t(`| Format | Bits | Exponent | Mantissa | Max value | Where used |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | $3.4\\times10^{38}$ | master weights, optimizer state |
| tf32 | 19* | 8 | 10 | $3.4\\times10^{38}$ | NVIDIA tensor cores, transparent |
| bf16 | 16 | 8 | 7 | $3.4\\times10^{38}$ | **the default for training** |
| fp16 | 16 | 5 | 10 | $65{,}504$ | older hardware, needs loss scaling |
| fp8 | 8 | 4 or 5 | 3 or 2 | $448$ | H100+ training, inference |
| int8 / int4 | 8 / 4 | — | — | — | quantized inference |

**bf16 is fp32 with the mantissa chopped off.** Same exponent range, so the same values are representable — you just
lose precision. That is why it needs no loss scaling and why it displaced fp16 for large-scale training. fp16, with
only 5 exponent bits, overflows above 65,504 — and attention logits routinely exceed that.`),

    t(`## Mixed precision, in practice

The standard recipe:

1. Keep **master weights in fp32**.
2. Cast to bf16 for the forward and backward passes (2× less memory traffic, and tensor cores are much faster).
3. Accumulate matmuls in fp32 inside the hardware.
4. Apply the optimizer update in fp32.

Some operations stay in fp32 regardless because they are precision-sensitive: softmax denominators, layer norm
statistics, and loss computation. Frameworks maintain an allowlist for this.`),

    warn(`**Catastrophic cancellation.** Subtracting nearly-equal numbers destroys significant digits. The classic
example is computing variance as $\\mathbb{E}[X^2]-\\mathbb{E}[X]^2$: if the mean is large relative to the spread, both
terms are huge and nearly equal, and you can get a *negative* variance. Use a stable two-pass or Welford algorithm.

The same issue is why softmax subtracts the max before exponentiating, and why you should never write
\`log(softmax(x))\` instead of \`log_softmax(x)\`.`),

    t(`## Conditioning

The **condition number** of a problem measures how much the output can move when the input is perturbed slightly.
For solving $A\\mathbf{x}=\\mathbf{b}$, it is $\\kappa(A)=\\sigma_{\\max}/\\sigma_{\\min}$, and the rule of thumb is:

> You lose about $\\log_{10}\\kappa$ digits of accuracy.

With $\\kappa=10^8$ in float32 (≈7 digits), your answer is noise. This is exactly why you should never solve the normal
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
  ],
  refs: [
    paper('What Every Computer Scientist Should Know About Floating-Point Arithmetic', 'David Goldberg', 1991, 'https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html', 'The definitive reference. Long, but the first few sections are essential.'),
    paper('Mixed Precision Training', 'Micikevicius et al.', 2017, 'https://arxiv.org/abs/1710.03740', 'The original recipe: master weights, loss scaling, and which ops must stay fp32.'),
    paper('FP8 Formats for Deep Learning', 'Micikevicius et al.', 2022, 'https://arxiv.org/abs/2209.05433', 'E4M3 and E5M2, and where each is appropriate.'),
    book('Accuracy and Stability of Numerical Algorithms', 'Nicholas Higham', 2002, 'https://epubs.siam.org/doi/book/10.1137/1.9780898718027', 'The serious treatment of conditioning and stability.'),
  ],
},

];

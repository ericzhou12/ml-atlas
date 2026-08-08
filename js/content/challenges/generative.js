/* ============================================================
   Challenges — track 6, generative models

   One entry per lesson id: { title, prompt, starter, solution, hint?, explain? }
   Starters carry their own PASS/FAIL check so the learner gets a verdict
   without leaving the lab.

   EDITOR NOTE: these are JS template literals, so an unescaped backtick
   silently terminates the string. Escape inline code in prose.
   ============================================================ */

export default {

'gen-autoencoders': {
  title: 'Train a VAE, then turn the KL term up until the encoder gives up',
  prompt: `1. Implement the reparameterization trick and the closed-form Gaussian KL, and fill in the backward pass.
2. Sweep $\\beta$, the weight on the KL term, and read three numbers at each setting: how well it reconstructs,
   how far the posteriors sit from the prior, and how large the encoder's outputs $\\mu$ are.
3. **Predict what happens to $\\mu$ as $\\beta$ grows** before you run it. The failure at the top of the sweep has
   a name, and it is visible in that third column rather than in the loss.`,
  hint: 'The sample is $\\mathbf{z} = \\mu + \\sigma \\odot \\epsilon$ with $\\epsilon$ drawn once per step — that is what lets a gradient pass through a random draw. The KL against $\\mathcal{N}(0,I)$ is $-\\tfrac12\\sum_j(1 + \\log\\sigma_j^2 - \\mu_j^2 - \\sigma_j^2)$, so its gradient pulls $\\mu$ toward 0 and $\\sigma$ toward 1.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

n = 800
tt = rng.uniform(-2.2, 2.2, n)
X = np.c_[tt, 0.45*tt + 0.35*tt**2 - 0.5] + rng.normal(0, 0.06, (n,2))
X = (X - X.mean(0)) / X.std(0)

def run(beta, steps=3000, lr=0.005):
    r = np.random.default_rng(0)
    D, H, Z = 2, 24, 1
    init = lambda a,b: (r.normal(0, np.sqrt(2/a), (a,b)), np.zeros(b))
    We1,be1 = init(D,H); Wmu,bmu = init(H,Z); Wlv,blv = init(H,Z)
    Wd1,bd1 = init(Z,H); Wd2,bd2 = init(H,D)
    relu = lambda x: np.maximum(0, x)
    for _ in range(steps):
        x = X[r.integers(0, n, 128)]
        h  = relu(x @ We1 + be1)
        mu = h @ Wmu + bmu
        lv = np.clip(h @ Wlv + blv, -6, 2)      # log variance
        sd = np.exp(0.5*lv)
        eps = r.normal(size=mu.shape)

        # TODO 1: z = the reparameterized sample
        z = mu
        hd = relu(z @ Wd1 + bd1)
        xh = hd @ Wd2 + bd2

        rec = ((xh-x)**2).sum(1).mean()
        # TODO 2: the closed-form KL against N(0, I)
        kl = 0.0

        dxh = 2*(xh-x)/len(x)
        dWd2, dbd2 = hd.T @ dxh, dxh.sum(0)
        dhd = (dxh @ Wd2.T) * (hd > 0)
        dWd1, dbd1 = z.T @ dhd, dhd.sum(0)
        dz = dhd @ Wd1.T
        # TODO 3: mu and lv each receive a decoder path AND a pull from the KL term
        dmu = dz
        dlv = dz*(0.5*sd*eps)
        dh = (dmu @ Wmu.T + dlv @ Wlv.T) * (h > 0)
        for W,b,gW,gb in [(Wd2,bd2,dWd2,dbd2),(Wd1,bd1,dWd1,dbd1),
                          (Wmu,bmu,h.T@dmu,dmu.sum(0)),(Wlv,blv,h.T@dlv,dlv.sum(0)),
                          (We1,be1,x.T@dh,dh.sum(0))]:
            W -= lr*gW; b -= lr*gb
    return rec, kl, float(np.abs(mu).mean())

print(f"{'beta':>6} {'reconstruction':>16} {'KL':>10} {'mean |mu|':>11}")
for beta in [0.0, 0.5, 1.0, 4.0, 10.0]:
    rec, kl, mu_mag = run(beta)
    tag = "   <- posterior collapse" if kl < 0.05 else ""
    print(f"{beta:6.1f} {rec:16.4f} {kl:10.4f} {mu_mag:11.4f}{tag}")

r0, r10 = run(0.0), run(10.0)
assert r0[0] < r10[0],  "beta=0 should reconstruct far better than beta=10"
assert r0[1] > 2.0,     "with no KL term the posteriors should drift far from the prior"
assert r10[1] < 0.05,   "beta=10 should drive the KL to nearly zero"
assert r10[2] < 0.1,    "under collapse the encoder outputs mu near 0 whatever the input"
print("\\nPASS")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

n = 800
tt = rng.uniform(-2.2, 2.2, n)
X = np.c_[tt, 0.45*tt + 0.35*tt**2 - 0.5] + rng.normal(0, 0.06, (n,2))
X = (X - X.mean(0)) / X.std(0)

def run(beta, steps=3000, lr=0.005):
    r = np.random.default_rng(0)
    D, H, Z = 2, 24, 1
    init = lambda a,b: (r.normal(0, np.sqrt(2/a), (a,b)), np.zeros(b))
    We1,be1 = init(D,H); Wmu,bmu = init(H,Z); Wlv,blv = init(H,Z)
    Wd1,bd1 = init(Z,H); Wd2,bd2 = init(H,D)
    relu = lambda x: np.maximum(0, x)
    for _ in range(steps):
        x = X[r.integers(0, n, 128)]
        h  = relu(x @ We1 + be1)
        mu = h @ Wmu + bmu
        lv = np.clip(h @ Wlv + blv, -6, 2)
        sd = np.exp(0.5*lv)
        eps = r.normal(size=mu.shape)

        z = mu + sd*eps                                   # reparameterization
        hd = relu(z @ Wd1 + bd1)
        xh = hd @ Wd2 + bd2

        rec = ((xh-x)**2).sum(1).mean()
        kl  = (-0.5*(1 + lv - mu**2 - np.exp(lv))).sum(1).mean()

        dxh = 2*(xh-x)/len(x)
        dWd2, dbd2 = hd.T @ dxh, dxh.sum(0)
        dhd = (dxh @ Wd2.T) * (hd > 0)
        dWd1, dbd1 = z.T @ dhd, dhd.sum(0)
        dz = dhd @ Wd1.T
        dmu = dz + beta * mu / len(x)
        dlv = dz*(0.5*sd*eps) + beta*0.5*(np.exp(lv)-1)/len(x)
        dh = (dmu @ Wmu.T + dlv @ Wlv.T) * (h > 0)
        for W,b,gW,gb in [(Wd2,bd2,dWd2,dbd2),(Wd1,bd1,dWd1,dbd1),
                          (Wmu,bmu,h.T@dmu,dmu.sum(0)),(Wlv,blv,h.T@dlv,dlv.sum(0)),
                          (We1,be1,x.T@dh,dh.sum(0))]:
            W -= lr*gW; b -= lr*gb
    return rec, kl, float(np.abs(mu).mean())

print(f"{'beta':>6} {'reconstruction':>16} {'KL':>10} {'mean |mu|':>11}")
for beta in [0.0, 0.5, 1.0, 4.0, 10.0]:
    rec, kl, mu_mag = run(beta)
    tag = "   <- posterior collapse" if kl < 0.05 else ""
    print(f"{beta:6.1f} {rec:16.4f} {kl:10.4f} {mu_mag:11.4f}{tag}")

r0, r10 = run(0.0), run(10.0)
assert r0[0] < r10[0]
assert r0[1] > 2.0
assert r10[1] < 0.05
assert r10[2] < 0.1
print("\\nPASS")`,
  explain: `Read the three columns together as $\\beta$ rises.

At $\\beta = 0$ there is no KL term, so this is a plain autoencoder. Reconstruction is nearly perfect and the KL
is enormous — the encoder scattered each example's code wherever it found convenient, treating the latent space
as a filing cabinet with no obligation to fill it. Draw a random $\\mathbf{z}$ from the prior and you will land in
a gap between codes and decode nonsense. An excellent compressor, and not a generator.

At $\\beta = 1$ the two terms balance. Reconstruction is worse, the KL is under 1, and the codes are now packed
tightly enough around the prior that sampling from it works. That trade *is* the ELBO, and it is not free — the
blur people complain about in VAE samples is this row.

At $\\beta \\ge 4$ the KL wins outright, and the tell is the last column collapsing to nearly zero. The encoder has
given up and outputs the prior for *every* input, ignoring $\\mathbf{x}$ completely. The KL is then essentially 0,
which the objective is delighted by, while reconstruction is as bad as predicting the dataset mean. That is
**posterior collapse** — and note how it presents: one term of the loss looks excellent while the latent variable
has quietly stopped carrying any information at all. Watching only the total loss, you might not notice.`,
},

'gen-gans': {
  title: 'Starve the generator by over-training the discriminator',
  prompt: `Train a 1-D GAN. Show that with too many discriminator steps the generator gradient vanishes, and that the
non-saturating loss fixes early training.`,
  hint: 'Non-saturating: maximize log D(G(z)) instead of minimizing log(1 - D(G(z))).',
  starter: `import numpy as np
rng = np.random.default_rng(0)

sigmoid = lambda z: 1/(1+np.exp(-np.clip(z,-30,30)))
real = lambda n: rng.normal(2.0, 0.5, n)

def train(d_steps=1, d_lr=0.05, g_lr=0.005, non_saturating=True, steps=1500):
    g_mu, g_sd, d_w, d_b = -2.0, 0.5, 0.5, 0.0
    for t in range(steps):
        for _ in range(d_steps):
            xr, xf = real(64), g_mu + g_sd*rng.normal(size=64)
            dr, df = sigmoid(d_w*xr + d_b), sigmoid(d_w*xf + d_b)
            # TODO: ascend the discriminator objective
            pass
        z = rng.normal(size=64)
        xf = g_mu + g_sd*z
        df = sigmoid(d_w*xf + d_b)
        # TODO: generator gradient, saturating vs non-saturating
        grad = 0.0
        g_mu += g_lr * grad
    return g_mu, d_w

print("target mean = 2.00\\n")
for label, kw in [("balanced, non-saturating", dict()),
                  ("D too strong (8 steps)", dict(d_steps=8, d_lr=0.3)),
                  ("saturating loss", dict(non_saturating=False))]:
    mu, dw = train(**kw)
    print(f"{label:28s} generator mean {mu:6.3f}   |D weight| {abs(dw):6.2f}")

good  = train()[0]
satur = train(non_saturating=False)[0]
assert abs(good - 2.0) < 0.4,  "the balanced run should find the target near 2.0"
assert abs(satur - 2.0) > 1.0, "the saturating loss should fail to get there"
print("\\nPASS")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

sigmoid = lambda z: 1/(1+np.exp(-np.clip(z,-30,30)))
real = lambda n: rng.normal(2.0, 0.5, n)

def train(d_steps=1, d_lr=0.05, g_lr=0.005, non_saturating=True, steps=1500):
    g_mu, g_sd, d_w, d_b = -2.0, 0.5, 0.5, 0.0
    for t in range(steps):
        for _ in range(d_steps):
            xr, xf = real(64), g_mu + g_sd*rng.normal(size=64)
            dr, df = sigmoid(d_w*xr + d_b), sigmoid(d_w*xf + d_b)
            d_w += d_lr * (((1-dr)*xr).mean() - (df*xf).mean())
            d_b += d_lr * ((1-dr).mean() - df.mean())
            d_w = np.clip(d_w, -20, 20)
        z = rng.normal(size=64)
        xf = g_mu + g_sd*z
        df = sigmoid(d_w*xf + d_b)
        if non_saturating:
            grad = ((1-df)*d_w).mean()
        else:
            grad = (-(df*d_w)/(1-df+1e-9)).mean()
        g_mu += g_lr * grad
    return g_mu, d_w

print("target mean = 2.00\\n")
for label, kw in [("balanced, non-saturating", dict()),
                  ("D too strong (8 steps)", dict(d_steps=8, d_lr=0.3)),
                  ("saturating loss", dict(non_saturating=False))]:
    mu, dw = train(**kw)
    print(f"{label:28s} generator mean {mu:6.3f}   |D weight| {abs(dw):6.2f}")

good  = train()[0]
satur = train(non_saturating=False)[0]
assert abs(good - 2.0) < 0.4,  "the balanced run should find the target near 2.0"
assert abs(satur - 2.0) > 1.0, "the saturating loss should fail to get there"
print("\\nPASS")`,
  explain: `The balanced run finds the target. The other two rows are the two classic GAN failures, and they fail for
opposite reasons.

**Over-training the discriminator** looks like it ought to help — a sharper critic gives better feedback. It does
the reverse. Once the discriminator separates real from fake almost perfectly its output saturates: it says
"fake, certainly" everywhere the generator currently sits, and a saturated sigmoid has
[essentially no gradient](#/l/nn-activations). The generator is told it is wrong and nothing about which way to
move. The discriminator has to stay *beatable* for training to proceed, which is why GAN training is a balancing
act rather than an optimization.

**The saturating loss** is the same problem written into the objective. Minimising $\\log(1-D(G(z)))$ has almost no
gradient exactly where the generator starts — far from the data, where $D$ is confident — so early training barely
moves, and the run here never reaches the target at all. The non-saturating form flips it to maximising
$\\log D(G(z))$, which has its *largest* gradient precisely there. Same fixed point, completely different early
dynamics, and it is why every practical GAN uses it.`,
},
'gen-diffusion': {
  title: 'Train a diffusion model, then see how few sampling steps you can get away with',
  prompt: `1. **Forward noising.** Build $\\mathbf{x}_t$ from a clean sample and a noise draw using
   $\\mathbf{x}_t = \\sqrt{\\bar\\alpha_t}\\,\\mathbf{x}_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\boldsymbol\\epsilon$ — the
   closed form that lets you jump to any timestep in one step instead of simulating $t$ of them.
2. **The objective.** Train the network to predict the noise that was added, and nothing else.
3. **Sampling.** Then run DDIM at 5, 20 and 100 steps and measure how often each of the three modes appears.
   **Predict two things first:** whether any mode gets dropped, and how many steps you actually need.`,
  hint: 'The training target is $\\boldsymbol\\epsilon$ itself, so the loss is a plain mean squared error between the prediction and the noise you drew. For sampling, at each step estimate the clean point $\\hat{\\mathbf{x}}_0 = (\\mathbf{x}_t - \\sqrt{1-\\bar\\alpha_t}\\,\\hat{\\boldsymbol\\epsilon})/\\sqrt{\\bar\\alpha_t}$, then re-noise it to the *next* timestep on the schedule.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

MU = np.array([[-1.3,-1.0], [1.4,-0.8], [0.0,1.35]])
W, S = np.array([0.3,0.3,0.4]), 0.22
def sample_data(n):
    k = rng.choice(3, n, p=W)
    return MU[k] + rng.normal(0, S, (n,2))

T = 200
betas = np.linspace(1e-4, 0.02, T)
abar = np.cumprod(1 - betas)          # alpha-bar: how much signal survives to step t

H = 96
def init(a,b): return rng.normal(0, np.sqrt(2/a), (a,b)), np.zeros(b)
W1,b1 = init(4,H); W2,b2 = init(H,H); W3,b3 = init(H,2)

def feats(x, t):
    tt = (t/T)[:,None]
    return np.c_[x, tt, np.sin(6*tt)]          # the model is told which timestep it is at

def fwd(x, t):
    h1 = np.tanh(feats(x,t) @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return h2 @ W3 + b3, (h1,h2)

lr = 0.004
for step in range(6001):
    x0 = sample_data(256)
    t = rng.integers(0, T, 256)
    eps = rng.normal(size=x0.shape)
    # TODO 1: build xt from x0 and eps using abar[t]
    xt = x0

    pred, (h1,h2) = fwd(xt, t)
    # TODO 2: the loss is the mean squared error between pred and the noise that was added
    loss = 0.0

    d = 2*(pred-eps)/len(x0)
    dW3, db3 = h2.T@d, d.sum(0)
    d2 = (d@W3.T)*(1-h2**2); dW2, db2 = h1.T@d2, d2.sum(0)
    d1 = (d2@W2.T)*(1-h1**2); dW1, db1 = feats(xt,t).T@d1, d1.sum(0)
    for Wm,bm,gW,gb in [(W3,b3,dW3,db3),(W2,b2,dW2,db2),(W1,b1,dW1,db1)]:
        Wm -= lr*gW; bm -= lr*gb
    if step % 1500 == 0: print(f"step {step:5d}  loss {loss:.4f}")

def sample(n, steps):
    x = rng.normal(size=(n,2))                 # start from pure noise
    ts = np.linspace(T-1, 0, steps).astype(int)
    for i, t in enumerate(ts):
        e, _ = fwd(x, np.full(n, t))
        a = abar[t]
        # TODO 3: estimate the clean point from x and the predicted noise,
        #         then re-noise it to the next timestep on the schedule
        a_next = abar[ts[i+1]] if i+1 < len(ts) else 1.0
        x = x
    return x

print("\\nmode coverage (the data is 30/30/40%):")
for steps in [5, 20, 100]:
    s = sample(3000, steps)
    a = ((s[:,None,:]-MU[None])**2).sum(-1).argmin(1)
    print(f"  {steps:3d} DDIM steps -> {np.round(np.bincount(a, minlength=3)/len(s)*100,1)}%")

cov = np.bincount(((sample(3000, 50)[:,None,:]-MU[None])**2).sum(-1).argmin(1), minlength=3)/3000
assert (cov > 0.15).all(), "every mode should be covered -- diffusion does not drop modes"
assert abs(cov - np.array([0.3, 0.3, 0.4])).max() < 0.08, "the proportions should roughly match the data"
print("\\nPASS")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

MU = np.array([[-1.3,-1.0], [1.4,-0.8], [0.0,1.35]])
W, S = np.array([0.3,0.3,0.4]), 0.22
def sample_data(n):
    k = rng.choice(3, n, p=W)
    return MU[k] + rng.normal(0, S, (n,2))

T = 200
betas = np.linspace(1e-4, 0.02, T)
abar = np.cumprod(1 - betas)

H = 96
def init(a,b): return rng.normal(0, np.sqrt(2/a), (a,b)), np.zeros(b)
W1,b1 = init(4,H); W2,b2 = init(H,H); W3,b3 = init(H,2)

def feats(x, t):
    tt = (t/T)[:,None]
    return np.c_[x, tt, np.sin(6*tt)]

def fwd(x, t):
    h1 = np.tanh(feats(x,t) @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return h2 @ W3 + b3, (h1,h2)

lr = 0.004
for step in range(6001):
    x0 = sample_data(256)
    t = rng.integers(0, T, 256)
    eps = rng.normal(size=x0.shape)
    xt = np.sqrt(abar[t])[:,None]*x0 + np.sqrt(1-abar[t])[:,None]*eps

    pred, (h1,h2) = fwd(xt, t)
    loss = ((pred-eps)**2).mean()
    d = 2*(pred-eps)/len(x0)
    dW3, db3 = h2.T@d, d.sum(0)
    d2 = (d@W3.T)*(1-h2**2); dW2, db2 = h1.T@d2, d2.sum(0)
    d1 = (d2@W2.T)*(1-h1**2); dW1, db1 = feats(xt,t).T@d1, d1.sum(0)
    for Wm,bm,gW,gb in [(W3,b3,dW3,db3),(W2,b2,dW2,db2),(W1,b1,dW1,db1)]:
        Wm -= lr*gW; bm -= lr*gb
    if step % 1500 == 0: print(f"step {step:5d}  loss {loss:.4f}")

def sample(n, steps):
    x = rng.normal(size=(n,2))
    ts = np.linspace(T-1, 0, steps).astype(int)
    for i, t in enumerate(ts):
        e, _ = fwd(x, np.full(n, t))
        a = abar[t]
        x0h = (x - np.sqrt(1-a)*e) / np.sqrt(a)
        a_next = abar[ts[i+1]] if i+1 < len(ts) else 1.0
        x = np.sqrt(a_next)*x0h + np.sqrt(1-a_next)*e
    return x

print("\\nmode coverage (the data is 30/30/40%):")
for steps in [5, 20, 100]:
    s = sample(3000, steps)
    a = ((s[:,None,:]-MU[None])**2).sum(-1).argmin(1)
    print(f"  {steps:3d} DDIM steps -> {np.round(np.bincount(a, minlength=3)/len(s)*100,1)}%")

cov = np.bincount(((sample(3000, 50)[:,None,:]-MU[None])**2).sum(-1).argmin(1), minlength=3)/3000
assert (cov > 0.15).all()
assert abs(cov - np.array([0.3, 0.3, 0.4])).max() < 0.08
print("\\nPASS")`,
  explain: `Two things are worth taking from the coverage table.

First, **every mode is represented, in roughly the right proportion**. Compare that with the GAN in the previous
lesson: a GAN is rewarded for fooling the discriminator, and covering one mode perfectly can do that, which is why
mode collapse is its signature failure. Diffusion is trained by predicting noise under a squared-error loss, which
is maximum likelihood in disguise, and [forward KL](#/l/math-information) is punished savagely for assigning
almost no probability anywhere the data actually is. It is structurally obliged to cover everything.

Second, look at how *few* steps are needed. Coverage is already close at 20 steps and barely improves by 100 —
even though training defined a 200-step noising schedule. This is what DDIM buys: sampling does not have to
retrace the schedule step by step, it can skip along the same trajectory. That single observation is most of what
turned diffusion from a slow curiosity into something servable, and it is why the sampler step count is a knob
you tune rather than a property of the model.

Note also what the network was never asked to do. It never sees a clean image, never learns a likelihood, and
never learns to generate. It only ever answers "how much noise is in this?" — and running that one answer
backwards produces samples.`,
},

};

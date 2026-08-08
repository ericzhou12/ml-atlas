/* ============================================================
   Track 6 — Generative Models
   ============================================================ */

import { t, key, intuition, warn, hist, mathnote, viz, deriv, code, quiz, paper, book, course, blog, video, demo, codeRef,
         tldr, recap, jargon, steps, diagram } from './_helpers.js';

export default [

/* ---------------------------------------------------------- */
{
  id: 'gen-autoencoders',
  title: 'Autoencoders and VAEs',
  sub: 'Compression, and the trick that turns a compressor into a generator.',
  mins: 24, level: 'core',
  prereq: ['nn-backprop', 'math-information'],
  tags: ['VAE', 'latent variables'],
  sections: [
    tldr(`An **autoencoder** squeezes data through a narrow bottleneck and back out again. To reconstruct the
input from a 32-number summary, the network has to learn what actually matters — so the bottleneck becomes a
compressed representation.

That gives you a compressor, not a generator. Nothing organises the space of codes, so most points in it decode
to garbage and you cannot sample.

The **VAE** fixes this with one change: make the encoder output a *distribution* instead of a point, and push
those distributions toward a standard Gaussian. Now the code space is filled rather than pocked with holes, so
you can draw a random point, decode it, and get something plausible. The price is blurry samples, for a reason
this lesson makes precise.`),

    jargon([
      ['autoencoder', 'A network trained to reproduce its own input through a narrow bottleneck.'],
      ['encoder / decoder', 'The two halves: input → compressed code, and code → reconstruction.'],
      ['latent / code $\\mathbf{z}$', 'The compressed representation in the middle. "Latent" means hidden — it is not observed, only inferred.'],
      ['bottleneck', 'The deliberately small middle layer. Forcing information through it is what makes the network learn rather than copy.'],
      ['reconstruction loss', 'How far the output is from the input. Usually squared error.'],
      ['prior $p(\\mathbf{z})$', 'The distribution you *want* the codes to follow — a standard Gaussian, so you can sample from it.'],
      ['posterior $q(\\mathbf{z}\\mid\\mathbf{x})$', 'The distribution the encoder outputs for a particular input.'],
      ['ELBO', 'Evidence Lower BOund. The VAE objective: reconstruction quality minus a KL penalty for straying from the prior.'],
      ['reparameterization trick', 'Writing $\\mathbf{z} = \\mu + \\sigma\\odot\\epsilon$ so randomness enters as an input, letting gradients flow through a sampling step.'],
      ['posterior collapse', 'The failure where the encoder ignores the input entirely and outputs the prior, because the KL term won.'],
      ['VQ-VAE', 'A VAE with a *discrete* latent space, snapping codes to a learned codebook. What most modern image generators use as their compressor.'],
    ]),

    t(`## The autoencoder

An encoder maps $\\mathbf{x}$ to a low-dimensional code $\\mathbf{z}$; a decoder maps it back. Train to minimize
reconstruction error. The bottleneck forces the network to discard everything except what is needed to rebuild the
input.`),

    viz('autoencoder'),

    t(`A **linear** autoencoder with a $k$-dimensional bottleneck learns exactly the PCA subspace — nothing more. Bend
the data manifold in the figure and the straight line can no longer follow it. That gap is what nonlinear encoders buy.

But a plain autoencoder is **not a generative model**. Nothing constrains the latent space, so most points in it decode
to garbage. You have a compressor and no way to sample.`),

    t(`## The VAE

Make the encoder output a *distribution* rather than a point, and force that distribution toward a known prior. Now
you can sample: draw $\\mathbf{z}\\sim\\mathcal{N}(0,I)$, decode, and get something plausible.

The objective is the **evidence lower bound**:

$$\\log p(\\mathbf{x}) \\ge \\underbrace{\\mathbb{E}_{q(\\mathbf{z}\\mid\\mathbf{x})}[\\log p(\\mathbf{x}\\mid\\mathbf{z})]}_{\\text{reconstruction}} - \\underbrace{D_{\\text{KL}}\\big(q(\\mathbf{z}\\mid\\mathbf{x})\\,\\|\\,p(\\mathbf{z})\\big)}_{\\text{regularizer}}$$`),

    viz('vae-latent'),

    key(`Two terms pulling against each other:

- **Reconstruction** wants each example's posterior to be a tight, distinctive blob — great for copying, useless for
  sampling, because most of the prior lands in empty space.
- **KL** pulls every posterior toward $\\mathcal{N}(0,I)$ so the latent space is filled and samplable.

Set $\\beta = 0$ in the figure and you get an autoencoder with holes. Set $\\beta = 4$ and the posteriors merge
entirely — **posterior collapse**, where the decoder learns to ignore $\\mathbf{z}$ and the model degenerates into an
unconditional generator.`),

    deriv('Deriving the ELBO, and the reparameterization trick', `We want $\\log p(\\mathbf{x}) = \\log\\int p(\\mathbf{x}\\mid\\mathbf{z})p(\\mathbf{z})\\,d\\mathbf{z}$, which is intractable.
Introduce any distribution $q(\\mathbf{z}\\mid\\mathbf{x})$ and apply Jensen's inequality:

$$\\log p(\\mathbf{x}) = \\log \\mathbb{E}_{q}\\!\\left[\\frac{p(\\mathbf{x},\\mathbf{z})}{q(\\mathbf{z}\\mid\\mathbf{x})}\\right]
\\ \\ge\\ \\mathbb{E}_{q}\\!\\left[\\log\\frac{p(\\mathbf{x},\\mathbf{z})}{q(\\mathbf{z}\\mid\\mathbf{x})}\\right]
= \\mathbb{E}_q[\\log p(\\mathbf{x}\\mid\\mathbf{z})] - D_{\\text{KL}}(q\\|p)$$

The gap between $\\log p(\\mathbf{x})$ and the bound is exactly $D_{\\text{KL}}(q(\\mathbf{z}\\mid\\mathbf{x})\\,\\|\\,p(\\mathbf{z}\\mid\\mathbf{x}))$,
so maximizing the ELBO simultaneously fits the data and improves the posterior approximation.

**The reparameterization trick.** We need $\\nabla_\\phi \\mathbb{E}_{q_\\phi}[\\cdot]$, but the expectation is over a
distribution that depends on $\\phi$ — you cannot differentiate through a sampling operation. Rewrite the sample:

$$\\mathbf{z} = \\boldsymbol\\mu_\\phi(\\mathbf{x}) + \\boldsymbol\\sigma_\\phi(\\mathbf{x})\\odot\\boldsymbol\\epsilon,
\\qquad \\boldsymbol\\epsilon\\sim\\mathcal{N}(0,I)$$

Now the randomness is in $\\boldsymbol\\epsilon$, which does not depend on $\\phi$, and gradients flow through
$\\boldsymbol\\mu$ and $\\boldsymbol\\sigma$ normally. This one substitution is what made VAEs trainable, and the same
idea reappears throughout ML wherever you need to differentiate through sampling.

For Gaussian $q$ and standard normal prior the KL has a closed form:
$$D_{\\text{KL}} = -\\tfrac12\\sum_j\\left(1+\\log\\sigma_j^2-\\mu_j^2-\\sigma_j^2\\right)$$`),

    t(`## Where VAEs actually get used

VAE samples are famously blurry, because the Gaussian likelihood is equivalent to an L2 reconstruction loss, and L2's
optimal prediction is the *mean* — averaging over all plausible completions. GANs and diffusion produce far sharper
images.

But the VAE did not lose; it became a **component**. Stable Diffusion runs its diffusion process in the latent space of
a pretrained VAE, not in pixel space. The VAE compresses 512×512×3 to 64×64×4 — a 48× reduction — and diffusion
operates there. The blurriness that made VAEs poor standalone generators is irrelevant when a diffusion model supplies
the detail.

**VQ-VAE** replaces the continuous latent with a discrete codebook, which lets you model images with an autoregressive
*transformer* over discrete codes. That lineage runs to VQGAN and into modern any-to-any multimodal models.`),

    code('A VAE in NumPy', `import numpy as np
rng = np.random.default_rng(0)

# data: a 1-D manifold curled into 2-D
n = 800
tt = rng.uniform(-2.2, 2.2, n)
X = np.c_[tt, 0.45*tt + 0.35*tt**2 - 0.5] + rng.normal(0, 0.06, (n, 2))
X = (X - X.mean(0)) / X.std(0)

D, H, Z = 2, 24, 1
def init(a, b): return rng.normal(0, np.sqrt(2/a), (a, b)), np.zeros(b)
We1, be1 = init(D, H); Wmu, bmu = init(H, Z); Wlv, blv = init(H, Z)
Wd1, bd1 = init(Z, H); Wd2, bd2 = init(H, D)

def relu(x): return np.maximum(0, x)

beta, lr = 1.0, 0.02
for step in range(4001):
    idx = rng.integers(0, n, 128)
    x = X[idx]

    # --- encode ---
    h  = relu(x @ We1 + be1)
    mu = h @ Wmu + bmu
    lv = np.clip(h @ Wlv + blv, -6, 6)             # log variance
    sd = np.exp(0.5*lv)

    # --- reparameterize: the trick that makes this differentiable ---
    eps = rng.normal(size=mu.shape)
    z = mu + sd*eps

    # --- decode ---
    hd = relu(z @ Wd1 + bd1)
    xh = hd @ Wd2 + bd2

    rec = ((xh - x)**2).sum(1).mean()
    kl  = (-0.5*(1 + lv - mu**2 - np.exp(lv))).sum(1).mean()

    # --- backward ---
    dxh = 2*(xh - x) / len(x)
    dWd2, dbd2 = hd.T @ dxh, dxh.sum(0)
    dhd = (dxh @ Wd2.T) * (hd > 0)
    dWd1, dbd1 = z.T @ dhd, dhd.sum(0)
    dz = dhd @ Wd1.T

    dmu = dz + beta * mu / len(x)
    dlv = dz * (0.5*sd*eps) + beta * 0.5*(np.exp(lv) - 1) / len(x)
    dh = (dmu @ Wmu.T + dlv @ Wlv.T) * (h > 0)

    for W, b, gW, gb in [(Wd2,bd2,dWd2,dbd2), (Wd1,bd1,dWd1,dbd1),
                         (Wmu,bmu,h.T@dmu,dmu.sum(0)), (Wlv,blv,h.T@dlv,dlv.sum(0)),
                         (We1,be1,x.T@dh,dh.sum(0))]:
        W -= lr*gW; b -= lr*gb

    if step % 1000 == 0:
        print(f"step {step:4d}  recon {rec:.4f}  KL {kl:.4f}  ELBO {-(rec+beta*kl):.4f}")

# sample from the prior and decode
zs = rng.normal(size=(6, Z))
gen = relu(zs @ Wd1 + bd1) @ Wd2 + bd2
print("\\nsamples decoded from N(0,1):")
for z, g in zip(zs.ravel(), gen):
    print(f"  z={z:+.2f} -> ({g[0]:+.3f}, {g[1]:+.3f})")
print("\\nThey land on the data manifold — that is what the KL term bought.")`,
      'Two lines carry the whole idea. `z = mu + sd*eps` is the reparameterization trick: the randomness enters through `eps`, which does not depend on any weight, so a gradient can pass straight through a sampling step that would otherwise be a dead end. And the KL line is what makes the latent space samplable at all — without it the encoder would scatter codes wherever it liked and most of the prior would decode to nothing. Watch the two loss terms move in opposite directions as training proceeds; that tension is the model.'),

    quiz('You set the KL weight β to 0 in a VAE. What have you built?',
      ['A plain autoencoder — good reconstructions, but sampling from the prior produces garbage',
       'A GAN',
       'A model with better samples',
       'A model that will not train'],
      0,
      'Without the KL term nothing organizes the latent space. Each example gets a tight posterior somewhere arbitrary, so reconstruction is excellent while the prior $\\mathcal N(0,I)$ mostly covers regions no encoder ever mapped to — decode those and you get nothing meaningful. The KL term is precisely what makes the model *generative* rather than merely compressive.'),

    recap(`- Explain what a bottleneck forces a network to do, and why a linear autoencoder is just PCA.
- Say precisely why a plain autoencoder cannot generate, in terms of the geometry of its latent space.
- Read the ELBO as two competing terms and say what each one is buying.
- Explain the reparameterization trick and the problem it solves (gradients through randomness).
- Say why VAE samples are blurry, and connect it to the loss.
- Explain what VQ-VAE changed and why almost every modern image generator contains one.`),
  ],
  refs: [
    paper('Auto-Encoding Variational Bayes', 'Kingma & Welling', 2013, 'https://arxiv.org/abs/1312.6114', 'The VAE, and the reparameterization trick.'),
    paper('beta-VAE', 'Higgins et al.', 2017, 'https://openreview.net/forum?id=Sy2fzU9gl', 'Turning up the KL weight to encourage disentangled factors.'),
    paper('Neural Discrete Representation Learning', 'van den Oord, Vinyals & Kavukcuoglu', 2017, 'https://arxiv.org/abs/1711.00937', 'VQ-VAE — discrete latents, and the road to transformer-based image generation.'),
    blog('Tutorial on Variational Autoencoders', 'Carl Doersch', 2016, 'https://arxiv.org/abs/1606.05908', 'The clearest walkthrough of the derivation.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'gen-gans',
  title: 'Generative Adversarial Networks',
  sub: 'Two networks in a duel. Brilliant, unstable, and largely superseded.',
  mins: 20, level: 'core',
  prereq: ['nn-backprop'],
  tags: ['GAN'],
  sections: [
    tldr(`A GAN pits two networks against each other. A **generator** makes fake images from noise. A
**discriminator** tries to spot the fakes. Each gets better because the other does.

The genuinely elegant part: **the loss function is learned.** Nobody has to specify what makes an image look
real — an impossible thing to write down — because the discriminator works it out and its gradient tells the
generator where to go.

The genuinely painful part: this is a two-player game, not a minimisation, and games do not converge the way
minimisation does. GANs produced the first photorealistic faces and were then largely displaced by diffusion,
almost entirely because diffusion is *stable to train* rather than because it is better in principle.`),

    jargon([
      ['generator $G$', 'Maps a random noise vector to a fake sample.'],
      ['discriminator $D$', 'A classifier trying to distinguish real data from the generator\'s output.'],
      ['adversarial', 'The two networks have directly opposed objectives — one\'s gain is the other\'s loss.'],
      ['minimax', 'An optimization where one player minimises what the other maximises. Its solution is an *equilibrium*, not a minimum.'],
      ['equilibrium', 'The point where neither player can improve unilaterally. For a GAN: the discriminator is reduced to guessing.'],
      ['mode collapse', 'The generator producing only a few kinds of output, having found something that reliably fools the discriminator.'],
      ['non-saturating loss', 'A reformulation of the generator loss that gives useful gradients when the discriminator is winning.'],
      ['Wasserstein / WGAN', 'A GAN variant using a different distance between distributions, giving gradients that do not vanish.'],
      ['FID', 'Fréchet Inception Distance. The standard image-generation metric. Measures distribution similarity, and is insensitive to mode collapse in ways worth knowing.'],
    ]),

    t(`## The setup

A **generator** maps noise to samples. A **discriminator** tries to tell real from fake. They optimize opposite
objectives:

$$\\min_G\\max_D\\ \\mathbb{E}_{\\mathbf{x}\\sim p_{\\text{data}}}[\\log D(\\mathbf{x})] + \\mathbb{E}_{\\mathbf{z}}[\\log(1-D(G(\\mathbf{z})))]$$

The elegance is that **the loss function is learned**. No one has to specify what makes an image realistic — the
discriminator discovers it, and its gradient tells the generator where to move.`),

    viz('gan-training'),

    t(`At the optimum $D^*(\\mathbf{x}) = \\frac{p_{\\text{data}}}{p_{\\text{data}}+p_g}$, and plugging that back in shows
the generator is minimizing the Jensen–Shannon divergence between real and generated distributions. Equilibrium is
$D \\equiv \\tfrac12$ — the discriminator cannot do better than guessing.`),

    t(`## Why they are hard

**Vanishing generator gradient.** If $D$ becomes too good, $\\log(1-D(G(z)))$ saturates and $G$ receives nothing. Set
D-steps to 8 with a high D learning rate in the figure and watch it happen. The standard fix is the **non-saturating
loss**: instead of minimizing $\\log(1-D(G(z)))$, maximize $\\log D(G(z))$ — same fixed point, much better gradients
early.

**Mode collapse.** $G$ discovers one output that fools $D$ and produces only that. The objective does not require
covering the data distribution, only fooling the critic. Invisible in the 1-D figure; catastrophic in practice.

**No meaningful loss curve.** Both losses oscillate around equilibrium and tell you almost nothing about sample
quality. You evaluate by looking, or with proxy metrics like FID.

**Balance.** If either network outpaces the other, training fails. Practitioners developed an extensive folklore of
tricks — and folklore is the right word.

**WGAN** replaced JS divergence with the Wasserstein distance, which gives usable gradients even when the
distributions do not overlap, and produces a loss that actually correlates with quality. **WGAN-GP** enforces the
required Lipschitz constraint with a gradient penalty. **Spectral normalization** is the cheaper modern standard.`),

    hist(`GANs dominated image generation from 2014 to roughly 2021. StyleGAN produced photorealistic faces that
genuinely surprised people, and the latent-space editing work built on it was beautiful.

Diffusion displaced them for one main reason: **stable training with a simple loss**. Diffusion optimizes a plain
regression objective, covers modes by construction, scales predictably, and conditions on text easily. GANs required
expert tuning and could collapse without warning.

GANs are not gone. They remain competitive where **single-step generation** matters — super-resolution, real-time
audio vocoders, and increasingly as the distillation target that turns a 50-step diffusion model into a 1-step one.
The adversarial *objective* survived even as the architecture receded.`),

    code('A GAN on 1-D data, with the failure modes', `import numpy as np
rng = np.random.default_rng(0)

def sigmoid(z): return 1/(1+np.exp(-np.clip(z, -30, 30)))
real = lambda n: rng.normal(2.0, 0.5, n)

def train(d_steps=1, d_lr=0.05, g_lr=0.02, non_saturating=True, steps=1500):
    g_mu, g_sd = -2.0, 0.5
    d_w, d_b = 0.5, 0.0
    hist = []
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
        grad = ((1-df)*d_w).mean() if non_saturating else (-(df*d_w)/(1-df+1e-9)).mean()
        g_mu += g_lr * grad
        if t % 300 == 0: hist.append((t, g_mu, d_w))
    return g_mu, d_w, hist

print("target mean = 2.00\\n")
for label, kw in [
    ("balanced, non-saturating", dict()),
    ("D too strong (8 steps, high lr)", dict(d_steps=8, d_lr=0.3)),
    ("saturating loss", dict(non_saturating=False)),
]:
    mu, dw, hist = train(**kw)
    print(f"{label:34s} final generator mean {mu:6.3f}   |D weight| {abs(dw):6.2f}")
    print("    trajectory: " + "  ".join(f"t={t}:{m:+.2f}" for t, m, _ in hist))

print("\\nAn over-trained discriminator saturates and starves the generator of gradient.")
print("The non-saturating loss is what makes early training work at all.")`,
      'The three runs are one working configuration and two failures. Note that neither failure is a bug in the code — the balanced run and the over-trained-discriminator run share every line except the number of discriminator steps. That is the thing to take away about GANs: the same implementation succeeds or collapses depending on a balance between two networks that nothing in the objective maintains for you.'),

    quiz('Why did diffusion models displace GANs for image generation?',
      ['Diffusion optimizes a stable regression objective and covers modes by construction; GANs need a delicate adversarial balance and can collapse',
       'Diffusion models are faster at inference',
       'GANs cannot produce high-resolution images',
       'Diffusion models need less training data'],
      0,
      'Diffusion is *slower* at inference — that was its main drawback. It won on trainability: a simple MSE-style loss, a monotone objective, no min-max game, no mode collapse, predictable scaling, and easy conditioning. In deep learning, "trains reliably at scale" beats "elegant" nearly every time.'),

    recap(`- Explain what it means for the loss function to be *learned*, and why that is the appealing part.
- Say why a minimax game does not converge the way ordinary minimisation does.
- Describe mode collapse and why the objective does not penalise it.
- Explain when the generator's gradient vanishes and what the non-saturating loss changes.
- Give the honest reason diffusion displaced GANs, and say what GANs are still preferred for.`),
  ],
  refs: [
    paper('Generative Adversarial Networks', 'Goodfellow et al.', 2014, 'https://arxiv.org/abs/1406.2661', 'The original.'),
    paper('Wasserstein GAN', 'Arjovsky, Chintala & Bottou', 2017, 'https://arxiv.org/abs/1701.07875', 'Diagnoses why the original objective gives bad gradients, and fixes it.'),
    paper('A Style-Based Generator Architecture for GANs', 'Karras, Laine & Aila', 2018, 'https://arxiv.org/abs/1812.04948', 'StyleGAN. The high-water mark.'),
    paper('Diffusion Models Beat GANs on Image Synthesis', 'Dhariwal & Nichol', 2021, 'https://arxiv.org/abs/2105.05233', 'The changing of the guard.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'gen-diffusion',
  title: 'Diffusion Models',
  sub: 'Destroy structure with noise on a fixed schedule, then learn to reverse it.',
  mins: 32, level: 'advanced',
  prereq: ['math-probability', 'nn-backprop'],
  tags: ['diffusion', 'score matching'],
  sections: [
    tldr(`Generating an image from nothing is hard. Removing a *small* amount of noise from a slightly noisy
image is easy. Diffusion's entire idea is to replace the first problem with a thousand instances of the second.

Training could not be simpler: take a real image, add a known amount of noise, and train a network to **predict
the noise you added**. That is a plain mean-squared-error regression — no adversary, no equilibrium, no
instability.

To generate, start from pure noise and repeatedly subtract a little of the predicted noise. After enough steps,
an image. Diffusion won over GANs not by being cleverer but by being *trainable*: a monotone loss that goes
down and stays down.`),

    jargon([
      ['forward process', 'Progressively adding noise to data until nothing is left but static. Fixed, with no learning involved.'],
      ['reverse process', 'The learned journey back: from pure noise to a sample, one denoising step at a time.'],
      ['timestep $t$', 'How far along the noising schedule you are. $t=0$ is a clean image; $t=T$ is pure noise.'],
      ['noise schedule', 'How much noise is added at each timestep. A design choice that materially affects quality.'],
      ['$\\epsilon$-prediction', 'Training the network to output the noise that was added, rather than the clean image. The standard parameterization.'],
      ['score', '$\\nabla_{\\mathbf{x}}\\log p(\\mathbf{x})$ — the direction of increasing probability density. Predicting noise turns out to be equivalent to estimating this.'],
      ['sampler', 'The algorithm running the reverse process. DDPM, DDIM, DPM-Solver — they trade steps against quality.'],
      ['classifier-free guidance', 'Running the model with and without the text prompt and amplifying the difference. How text-to-image models are made to obey their prompts.'],
      ['guidance scale', 'How much to amplify that difference. High values give prompt-faithful but less diverse images.'],
      ['latent diffusion', 'Running diffusion in a compressed latent space instead of on pixels. Roughly 50× cheaper; what Stable Diffusion is.'],
    ]),

    t(`## The idea

Generating an image directly is hard. But *removing a little noise* from a slightly noisy image is easy. Diffusion
turns the hard problem into many easy ones.

**Forward process** (fixed, no learning): progressively add Gaussian noise until the data is indistinguishable from
$\\mathcal{N}(0,I)$.

**Reverse process** (learned): start from pure noise and denoise step by step back to a sample.`),

    viz('diffusion-forward'),

    t(`The forward process has a closed form that lets you jump to any timestep in one step:

$$\\mathbf{x}_t = \\sqrt{\\bar\\alpha_t}\\,\\mathbf{x}_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\boldsymbol\\epsilon,
\\qquad \\boldsymbol\\epsilon\\sim\\mathcal{N}(0,I)$$

That is essential for training: you sample a random $t$, corrupt directly to it, and train — no simulation of the
chain required.`),

    t(`## The training objective is absurdly simple

$$\\mathcal{L} = \\mathbb{E}_{t,\\mathbf{x}_0,\\boldsymbol\\epsilon}\\Big[\\big\\|\\boldsymbol\\epsilon - \\boldsymbol\\epsilon_\\theta(\\mathbf{x}_t, t)\\big\\|^2\\Big]$$

**Predict the noise that was added.** That is the whole loss — a mean squared error. The remarkable part is that this
simple regression objective is equivalent (after a reweighting Ho et al. found works better than the theoretically
derived one) to a variational bound on the log-likelihood.`),

    deriv('Noise prediction = score estimation', `The **score** of a distribution is $\\nabla_{\\mathbf{x}}\\log p(\\mathbf{x})$ — the direction of increasing density.

For the noised marginal $\\mathbf{x}_t = \\sqrt{\\bar\\alpha_t}\\mathbf{x}_0 + \\sqrt{1-\\bar\\alpha_t}\\boldsymbol\\epsilon$,
conditioned on $\\mathbf{x}_0$ the distribution is Gaussian with mean $\\sqrt{\\bar\\alpha_t}\\mathbf{x}_0$ and variance
$(1-\\bar\\alpha_t)I$, so

$$\\nabla_{\\mathbf{x}_t}\\log p(\\mathbf{x}_t\\mid\\mathbf{x}_0)
= -\\frac{\\mathbf{x}_t - \\sqrt{\\bar\\alpha_t}\\mathbf{x}_0}{1-\\bar\\alpha_t}
= -\\frac{\\boldsymbol\\epsilon}{\\sqrt{1-\\bar\\alpha_t}}$$

Taking the expectation over $\\mathbf{x}_0$ given $\\mathbf{x}_t$ (Tweedie's formula) gives the marginal score:

$$\\nabla_{\\mathbf{x}_t}\\log p(\\mathbf{x}_t) = -\\frac{\\mathbb{E}[\\boldsymbol\\epsilon\\mid\\mathbf{x}_t]}{\\sqrt{1-\\bar\\alpha_t}}$$

So a network trained to predict $\\boldsymbol\\epsilon$ by least squares is estimating $\\mathbb{E}[\\boldsymbol\\epsilon\\mid\\mathbf{x}_t]$,
which **is** the score up to a known constant. ∎

This is why "denoising diffusion" (Ho et al.) and "score-based generative modeling" (Song & Ermon) — developed
independently with different motivations — turned out to be the same algorithm.`),

    viz('score-matching'),

    t(`That figure explains why a *range* of noise levels is necessary. At $\\sigma = 0.02$ the score is sharp but
defined only in a thin shell around the data — start anywhere else and there is no signal. At $\\sigma = 1.5$ every
point gets a clear signal that merely points at the global mean. Neither works alone. Diffusion learns the score at
**all** noise levels and anneals from coarse to fine.`),

    t(`## Sampling`),

    viz('diffusion-reverse'),

    t(`**DDPM** takes the stochastic route, adding fresh noise at each step. High quality, ~1000 steps.

**DDIM** reinterprets the process as a deterministic ODE, which allows large steps. Twenty steps of DDIM approaches
1000 steps of DDPM — and because it is deterministic, the same initial noise always yields the same image, which makes
latent-space interpolation meaningful.

Modern solvers (DPM-Solver, Euler-ancestral, Heun) apply better ODE integration and get to 10–20 steps routinely.

**Classifier-free guidance** is how text conditioning actually works. Train one model on both conditional and
unconditional objectives (randomly dropping the prompt ~10% of the time), then at sampling extrapolate:

$$\\tilde{\\boldsymbol\\epsilon} = \\boldsymbol\\epsilon_\\theta(\\mathbf{x}_t,\\varnothing) + w\\big(\\boldsymbol\\epsilon_\\theta(\\mathbf{x}_t,c) - \\boldsymbol\\epsilon_\\theta(\\mathbf{x}_t,\\varnothing)\\big)$$

$w>1$ amplifies the direction the prompt points in. Turn up guidance in the figure: samples collapse toward one mode.
**Higher guidance means better prompt adherence and less diversity**, quantitatively — that is the knob every image
tool exposes.`),

    t(`## Latent diffusion

Diffusion in pixel space is brutally expensive at high resolution. **Latent diffusion** (Stable Diffusion) runs the
whole process inside a pretrained VAE's latent space — 512×512×3 becomes 64×64×4, a 48× reduction in elements — with a
U-Net operating there and cross-attention layers injecting text conditioning.

This is the architecture that made high-quality image generation runnable on consumer hardware, and it is why the VAE
survived as a component.`),

    t(`## Flow matching

A reframing that has largely taken over. Instead of a noise schedule and a score, define a straight-line path from
noise to data and regress the velocity field:

$$\\mathbf{x}_t = (1-t)\\mathbf{x}_0 + t\\mathbf{x}_1, \\qquad
\\mathcal{L} = \\mathbb{E}\\big\\|v_\\theta(\\mathbf{x}_t,t) - (\\mathbf{x}_1-\\mathbf{x}_0)\\big\\|^2$$`),

    viz('flow-matching'),

    t(`Same family, simpler formulation: no variance schedule to tune, no SNR weighting, a plain regression objective,
and — because the learned paths are nearly straight — far fewer solver steps. Stable Diffusion 3, Flux, and most
current video models use it.`),

    code('Diffusion on 2-D data, end to end', `import numpy as np
rng = np.random.default_rng(0)

# --- target: three Gaussian blobs ---
MU = np.array([[-1.3,-1.0], [1.4,-0.8], [0.0,1.35]])
W, S = np.array([0.3,0.3,0.4]), 0.22
def sample_data(n):
    k = rng.choice(3, n, p=W)
    return MU[k] + rng.normal(0, S, (n,2))

T = 200
betas = np.linspace(1e-4, 0.02, T)
alphas = 1 - betas
abar = np.cumprod(alphas)

# --- a tiny MLP epsilon-predictor ---
H = 96
def init(a,b): return rng.normal(0, np.sqrt(2/a), (a,b)), np.zeros(b)
W1,b1 = init(4,H); W2,b2 = init(H,H); W3,b3 = init(H,2)

def feats(x, t):
    tt = (t/T)[:,None]
    return np.c_[x, tt, np.sin(6*tt)]

def fwd(x, t):
    h1 = np.tanh(feats(x,t) @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return h2 @ W3 + b3, (h1, h2)

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
    if step % 1500 == 0:
        print(f"step {step:5d}  loss {loss:.4f}")

# --- DDIM sampling ---
def sample(n, steps):
    x = rng.normal(size=(n,2))
    ts = np.linspace(T-1, 0, steps).astype(int)
    for i, t in enumerate(ts):
        e, _ = fwd(x, np.full(n, t))
        a = abar[t]
        x0 = (x - np.sqrt(1-a)*e) / np.sqrt(a)
        a_next = abar[ts[i+1]] if i+1 < len(ts) else 1.0
        x = np.sqrt(a_next)*x0 + np.sqrt(1-a_next)*e
    return x

print("\\nmode coverage (target 30/30/40%):")
for steps in [5, 20, 100]:
    s = sample(3000, steps)
    assign = ((s[:,None,:]-MU[None])**2).sum(-1).argmin(1)
    frac = np.bincount(assign, minlength=3)/len(s)
    print(f"  {steps:3d} DDIM steps -> {np.round(frac*100,1)}%  "
          f"mean dist to nearest mode {np.sqrt(((s-MU[assign])**2).sum(1)).mean():.3f}")`,
      'Notice how little the network is asked to do. It never sees a clean sample paired with a label, never learns a likelihood, and never learns to generate. It answers exactly one question — *how much noise is in this, given how far along the schedule we are* — and generation is what you get by running that answer backwards from pure noise. The training loop is a plain regression, which is most of why diffusion is so much more stable to train than a GAN.'),

    quiz('Why must a diffusion model be trained across many noise levels rather than one?',
      ['At low noise the score is only defined near the data; at high noise it is uninformative. Sampling needs both.',
       'To make training faster',
       'Because the forward process is stochastic',
       'To prevent overfitting'],
      0,
      'At small $\\sigma$ the score is accurate but essentially zero away from the data manifold — a random starting point gets no gradient to follow. At large $\\sigma$ every point gets a signal, but it only points at the overall mean. Learning all levels and annealing from high to low gives you a usable signal everywhere: coarse structure first, then detail. This was the key insight of Song & Ermon 2019.'),

    recap(`- State the core reframing: one hard generation problem becomes many easy denoising problems.
- Write the training objective and say why "predict the noise" is a plain regression.
- Explain why the closed-form forward process is what makes training practical.
- Say what classifier-free guidance does and what raising the guidance scale trades away.
- Explain why latent diffusion is ~50× cheaper, and what component makes it possible.
- Compare diffusion and GANs on training stability, sample diversity, and inference cost.`),
  ],
  refs: [
    paper('Denoising Diffusion Probabilistic Models', 'Ho, Jain & Abbeel', 2020, 'https://arxiv.org/abs/2006.11239', 'DDPM. The paper that made diffusion work.'),
    paper('Generative Modeling by Estimating Gradients of the Data Distribution', 'Song & Ermon', 2019, 'https://arxiv.org/abs/1907.05600', 'The score-based view, arrived at independently.'),
    paper('Denoising Diffusion Implicit Models', 'Song, Meng & Ermon', 2020, 'https://arxiv.org/abs/2010.02502', 'DDIM — deterministic sampling in far fewer steps.'),
    paper('High-Resolution Image Synthesis with Latent Diffusion Models', 'Rombach et al.', 2021, 'https://arxiv.org/abs/2112.10752', 'Stable Diffusion.'),
    paper('Classifier-Free Diffusion Guidance', 'Ho & Salimans', 2022, 'https://arxiv.org/abs/2207.12598', 'How text conditioning actually gets its strength.'),
    paper('Flow Matching for Generative Modeling', 'Lipman et al.', 2022, 'https://arxiv.org/abs/2210.02747', 'The reformulation most current models use.'),
    blog('What are Diffusion Models?', 'Lilian Weng', 2021, 'https://lilianweng.github.io/posts/2021-07-11-diffusion-models/', 'The best written survey, with all derivations worked.'),
  ],
},

];

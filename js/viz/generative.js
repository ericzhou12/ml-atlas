/* ============================================================
   viz/generative.js — autoencoders, VAEs, GANs, diffusion, flows.

   The diffusion demos use a 2-D Gaussian-mixture target whose score
   function is available in closed form, so the "denoiser" here is the
   exact optimal one — no training, no hand-waving.
   ============================================================ */

import { panel } from '../ui.js';
import { cssVar, alpha, mix, seqMap, divergeMap, LA, rng, fmt, SERIES } from '../plot.js';

const V = {};
export default V;

/* ---------- the target distribution: a mixture of Gaussians ---------- */

const MODES = [
  { mu: [-1.3, -1.0], w: .3, s: .22 },
  { mu: [1.4, -0.8], w: .3, s: .22 },
  { mu: [0.0, 1.35], w: .4, s: .22 },
];

/** p_t(x) for the variance-preserving forward process at noise level sigma. */
function mixDensity(x, y, extraVar) {
  let d = 0;
  for (const m of MODES) {
    const v = m.s * m.s + extraVar;
    d += m.w * Math.exp(-((x - m.mu[0]) ** 2 + (y - m.mu[1]) ** 2) / (2 * v)) / (2 * Math.PI * v);
  }
  return d;
}
/** ∇ log p_t(x) — exact score of the noised mixture. */
function mixScore(x, y, extraVar, shrink = 1) {
  let num = [0, 0], den = 0;
  for (const m of MODES) {
    const mu = [m.mu[0] * shrink, m.mu[1] * shrink];
    const v = m.s * m.s * shrink * shrink + extraVar;
    const w = m.w * Math.exp(-((x - mu[0]) ** 2 + (y - mu[1]) ** 2) / (2 * v)) / v;
    num[0] += w * (mu[0] - x) / v * v;
    num[1] += w * (mu[1] - y) / v * v;
    den += w * v;
    // accumulate properly below
  }
  // recompute cleanly (weighted average of per-component scores)
  let wsum = 0, sx = 0, sy = 0;
  for (const m of MODES) {
    const mu = [m.mu[0] * shrink, m.mu[1] * shrink];
    const v = m.s * m.s * shrink * shrink + extraVar;
    const q = m.w * Math.exp(-((x - mu[0]) ** 2 + (y - mu[1]) ** 2) / (2 * v)) / (2 * Math.PI * v);
    wsum += q;
    sx += q * (mu[0] - x) / v;
    sy += q * (mu[1] - y) / v;
  }
  if (wsum < 1e-300) return [0, 0];
  return [sx / wsum, sy / wsum];
}

function sampleTarget(r) {
  let u = r(), acc = 0;
  for (const m of MODES) { acc += m.w; if (u <= acc) return [m.mu[0] + r.normal(0, m.s), m.mu[1] + r.normal(0, m.s)]; }
  const m = MODES[MODES.length - 1];
  return [m.mu[0] + r.normal(0, m.s), m.mu[1] + r.normal(0, m.s)];
}

/* ============================================================
   Forward diffusion: destroying structure
   ============================================================ */

V['diffusion-forward'] = (host) => {
  const r = rng(12);
  const base = Array.from({ length: 400 }, () => sampleTarget(r));
  const noise = base.map(() => [r.normal(), r.normal()]);

  panel(host, {
    title: 'Forward process: adding noise on a schedule',
    height: 320,
    plot: { xlim: [-3, 3], ylim: [-2.6, 2.6], equal: true },
    controls: [
      { type: 'slider', key: 't', label: 'time t (0 = data, 1 = pure noise)', min: 0, max: 1, step: .005, value: 0 },
      { type: 'check', key: 'dens', label: 'show density p_t(x)', value: true },
      { type: 'play' },
    ],
    animate(s) { s.t = (s.t + .006) % 1; },
    draw(p, s, P) {
      // variance-preserving: x_t = sqrt(abar) x_0 + sqrt(1-abar) eps
      const abar = Math.cos(s.t * Math.PI / 2) ** 2;
      const sa = Math.sqrt(abar), sn = Math.sqrt(1 - abar);
      p.clear();
      if (s.dens) p.heat((x, y) => mixDensity(x / Math.max(sa, 1e-6), y / Math.max(sa, 1e-6), (1 - abar) / Math.max(abar, 1e-6)),
        { step: 6, cmap: (t) => mix(cssVar('--bg-inset'), cssVar('--s4'), t * .6) });
      p.axes({ nx: 5, ny: 4 });
      p.clip();
      const pts = base.map((b, i) => [sa * b[0] + sn * noise[i][0], sa * b[1] + sn * noise[i][1]]);
      p.points(pts, { r: 2.6, color: (_, i) => alpha(SERIES()[i % 3], .7) });
      p.clip(false);
      P.readout({
        't': fmt(s.t, 3),
        'ᾱ(t)': fmt(abar, 4),
        'signal √ᾱ': fmt(sa, 3),
        'noise √(1−ᾱ)': fmt(sn, 3),
        'SNR': fmt(abar / Math.max(1 - abar, 1e-9), 4),
      });
    },
    caption: 'Colors mark which mode each point started in. Slide $t$ forward: the three clusters blur, merge, and by $t=1$ the colors are completely interleaved inside a single standard Gaussian. **This direction requires no learning at all** — it is a fixed formula, $x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$, and you can jump to any $t$ in one step. The whole trick of diffusion is that the hard direction is the one you *learn*.',
  });
};

/* ============================================================
   Reverse process: sampling
   ============================================================ */

V['diffusion-reverse'] = (host) => {
  let parts = [], tNow = 1, trail = [];
  const reset = (n) => {
    const r = rng(Math.floor(Math.random() * 9999));
    parts = Array.from({ length: n }, () => [r.normal(), r.normal()]);
    trail = parts.map((p) => [p.slice()]);
    tNow = 1;
  };
  reset(120);

  panel(host, {
    title: 'Reverse process: following the score back to the data',
    height: 330,
    plot: { xlim: [-3, 3], ylim: [-2.6, 2.6], equal: true },
    controls: [
      { type: 'slider', key: 'steps', label: 'sampling steps', min: 5, max: 200, step: 5, value: 60 },
      { type: 'select', key: 'sampler', label: 'sampler', value: 'ddpm', options: [{ value: 'ddpm', label: 'DDPM (stochastic)' }, { value: 'ddim', label: 'DDIM (deterministic)' }] },
      { type: 'slider', key: 'guide', label: 'guidance toward mode ▲ (CFG-like)', min: 0, max: 4, step: .05, value: 0 },
      { type: 'check', key: 'field', label: 'show score field', value: true },
      { type: 'check', key: 'trails', label: 'show trajectories', value: true },
      { type: 'play' },
      { type: 'button', label: '↺ restart from noise', onClick: () => reset(120) },
    ],
    animate(s, P) {
      const N = Math.round(s.steps);
      const dt = 1 / N;
      if (tNow <= 1e-4) { P.play(false); return; }
      const tNext = Math.max(0, tNow - dt);
      const abar = Math.cos(tNow * Math.PI / 2) ** 2;
      const abarNext = Math.cos(tNext * Math.PI / 2) ** 2;
      const varNow = (1 - abar) / Math.max(abar, 1e-8);
      const sa = Math.sqrt(abar), saN = Math.sqrt(abarNext);
      const rr = rng(Math.floor(Math.random() * 1e6));
      parts = parts.map((q, i) => {
        // score in the "scaled" coordinate system
        let [gx, gy] = mixScore(q[0] / Math.max(sa, 1e-6), q[1] / Math.max(sa, 1e-6), varNow);
        gx /= Math.max(sa, 1e-6); gy /= Math.max(sa, 1e-6);
        if (s.guide > 0) {
          // pull toward the top mode: extra score term
          const m = MODES[2], v = m.s * m.s * abar + (1 - abar);
          const mu = [m.mu[0] * sa, m.mu[1] * sa];
          gx += s.guide * (mu[0] - q[0]) / v;
          gy += s.guide * (mu[1] - q[1]) / v;
        }
        // predict x0, then re-noise to t_next  (DDIM / DDPM step)
        const x0 = [(q[0] + (1 - abar) * gx) / Math.max(sa, 1e-6), (q[1] + (1 - abar) * gy) / Math.max(sa, 1e-6)];
        const eps = [(q[0] - sa * x0[0]) / Math.max(Math.sqrt(1 - abar), 1e-8), (q[1] - sa * x0[1]) / Math.max(Math.sqrt(1 - abar), 1e-8)];
        let nx, ny;
        if (s.sampler === 'ddim') {
          nx = saN * x0[0] + Math.sqrt(1 - abarNext) * eps[0];
          ny = saN * x0[1] + Math.sqrt(1 - abarNext) * eps[1];
        } else {
          const sigma = Math.sqrt(Math.max(0, (1 - abarNext) / (1 - abar) * (1 - abar / abarNext)));
          const c = Math.sqrt(Math.max(0, 1 - abarNext - sigma * sigma));
          nx = saN * x0[0] + c * eps[0] + sigma * rr.normal();
          ny = saN * x0[1] + c * eps[1] + sigma * rr.normal();
        }
        if (s.trails) { trail[i].push([nx, ny]); if (trail[i].length > 220) trail[i].shift(); }
        return [nx, ny];
      });
      tNow = tNext;
    },
    draw(p, s, P) {
      const abar = Math.cos(tNow * Math.PI / 2) ** 2;
      const sa = Math.max(Math.sqrt(abar), 1e-6);
      const varNow = (1 - abar) / abar;
      p.clear();
      p.heat((x, y) => mixDensity(x / sa, y / sa, varNow), { step: 6, cmap: (t) => mix(cssVar('--bg-inset'), cssVar('--s4'), t * .55) });
      p.axes({ nx: 5, ny: 4 });
      p.clip();
      if (s.field) {
        p.quiver((x, y) => {
          let [gx, gy] = mixScore(x / sa, y / sa, varNow);
          return [gx / sa, gy / sa];
        }, { color: alpha(cssVar('--s3'), .5), nx: 15, ny: 12, scale: .75, alphaBy: true });
      }
      if (s.trails) trail.forEach((tr, i) => p.line(tr, { color: alpha(SERIES()[i % 8], .28), width: 1 }));
      p.points(parts, { r: 3, color: cssVar('--s2') });
      // true modes
      p.points(MODES.map((m) => [m.mu[0] * sa, m.mu[1] * sa]), { r: 5, color: cssVar('--s5'), shape: 'cross', width: 2 });
      p.clip(false);
      // mode coverage
      const counts = [0, 0, 0];
      for (const q of parts) {
        let b = 0, bd = Infinity;
        MODES.forEach((m, k) => { const d = (q[0] - m.mu[0] * sa) ** 2 + (q[1] - m.mu[1] * sa) ** 2; if (d < bd) { bd = d; b = k; } });
        counts[b]++;
      }
      P.readout({
        't': fmt(tNow, 3),
        'sampler': s.sampler.toUpperCase(),
        'mode coverage': counts.map((c) => fmt(c / parts.length * 100, 0) + '%').join(' / '),
        'target': '30% / 30% / 40%',
        'guidance': s.guide > 0 ? `${fmt(s.guide, 2)} → mode collapse toward ▲` : 'none — unconditional',
      });
    },
    caption: 'The green arrows are $\\nabla\\log p_t(x)$ — the direction of increasing data density at the *current* noise level. A neural net is trained to predict exactly this (equivalently, to predict the noise). Watch three things: **(1)** early steps move through a smooth, nearly-Gaussian landscape; the modes only separate late. **(2)** DDIM with 20 steps gets close to DDPM with 200 — deterministic samplers are why generation got fast. **(3)** Turn up guidance: samples pile into one mode. Higher CFG means better prompt adherence and less diversity, quantitatively.',
  });
};

V['score-matching'] = (host) => {
  panel(host, {
    title: 'Why the noise level has to vary',
    height: 300,
    plot: { xlim: [-3, 3], ylim: [-2.6, 2.6], equal: true },
    controls: [
      { type: 'slider', key: 'sigma', label: 'noise level σ', min: .02, max: 1.6, step: .01, value: .05 },
      { type: 'check', key: 'field', label: 'score field', value: true },
      { type: 'check', key: 'samples', label: 'noisy samples', value: true },
    ],
    draw(p, s, P) {
      const v = s.sigma * s.sigma;
      p.clear();
      p.heat((x, y) => Math.pow(mixDensity(x, y, v), .35), { step: 5, cmap: (t) => mix(cssVar('--bg-inset'), cssVar('--s4'), t * .7) });
      p.axes({ nx: 5, ny: 4 });
      p.clip();
      if (s.field) p.quiver((x, y) => mixScore(x, y, v), { color: alpha(cssVar('--s3'), .55), nx: 16, ny: 13, scale: .8, alphaBy: true });
      if (s.samples) {
        const r = rng(8);
        const pts = Array.from({ length: 220 }, () => {
          const b = sampleTarget(r);
          return [b[0] + r.normal(0, s.sigma), b[1] + r.normal(0, s.sigma)];
        });
        p.points(pts, { r: 2.2, color: alpha(cssVar('--s2'), .6) });
      }
      p.clip(false);
      // fraction of the plane with usable (nonzero) score
      let live = 0, tot = 0;
      for (let x = -3; x <= 3; x += .12) for (let y = -2.6; y <= 2.6; y += .12) {
        const g = mixScore(x, y, v);
        if (Math.hypot(g[0], g[1]) > .05 && mixDensity(x, y, v) > 1e-4) live++;
        tot++;
      }
      P.readout({
        'σ': fmt(s.sigma, 3),
        'region with informative score': fmt(live / tot * 100, 1) + '%',
        'problem at small σ': 'the score is only defined near the data — a random start gets no signal',
        'problem at large σ': 'the score points to the mean — accurate, but uninformative about detail',
      });
    },
    caption: 'Set σ to 0.02. The score field is sharp and correct — **but only in a thin shell around the data**. Start a sample anywhere else and the arrows are essentially zero: you are lost with no gradient to follow. Now set σ to 1.5: every point gets a clear signal, but it just points at the overall centre. Neither level works alone. Diffusion\'s answer is to learn the score at *all* noise levels simultaneously and anneal from large to small — coarse structure first, then detail.',
  });
};

/* ============================================================
   Flow matching
   ============================================================ */

V['flow-matching'] = (host) => {
  let parts = [], t = 0, trail = [];
  const reset = () => {
    const r = rng(Math.floor(Math.random() * 9999));
    parts = Array.from({ length: 140 }, () => [r.normal(), r.normal()]);
    trail = parts.map((q) => [q.slice()]);
    t = 0;
  };
  reset();

  panel(host, {
    title: 'Flow matching: a straight-line path from noise to data',
    height: 320,
    plot: { xlim: [-3, 3], ylim: [-2.6, 2.6], equal: true },
    controls: [
      { type: 'slider', key: 'steps', label: 'ODE steps', min: 2, max: 100, step: 1, value: 24 },
      { type: 'check', key: 'field', label: 'show velocity field v(x,t)', value: true },
      { type: 'check', key: 'trails', label: 'trajectories', value: true },
      { type: 'play' },
      { type: 'button', label: '↺ restart', onClick: reset },
    ],
    animate(s, P) {
      const dt = 1 / Math.round(s.steps);
      if (t >= 1 - 1e-9) { P.play(false); return; }
      // conditional-OT probability path: x_t = (1-t) x0 + t x1, marginal velocity
      parts = parts.map((q, i) => {
        const v = velocity(q[0], q[1], t);
        const nx = q[0] + v[0] * dt, ny = q[1] + v[1] * dt;
        if (s.trails) trail[i].push([nx, ny]);
        return [nx, ny];
      });
      t = Math.min(1, t + dt);
    },
    draw(p, s, P) {
      p.clear();
      p.heat((x, y) => pathDensity(x, y, t), { step: 6, cmap: (c) => mix(cssVar('--bg-inset'), cssVar('--s4'), c * .55) });
      p.axes({ nx: 5, ny: 4 });
      p.clip();
      if (s.field) p.quiver((x, y) => velocity(x, y, t), { color: alpha(cssVar('--s3'), .55), nx: 15, ny: 12, scale: .7 });
      if (s.trails) trail.forEach((tr, i) => p.line(tr, { color: alpha(SERIES()[i % 8], .3), width: 1 }));
      p.points(parts, { r: 3, color: cssVar('--s2') });
      p.points(MODES.map((m) => m.mu), { r: 5, color: cssVar('--s5'), shape: 'cross', width: 2 });
      p.clip(false);
      P.readout({
        't': fmt(t, 3),
        'objective': 'regress v_θ(x,t) onto (x₁ − x₀) — a plain MSE',
        'paths': 'nearly straight → few steps needed',
        'vs diffusion': 'same family; flow matching just chooses a straighter path',
      });
    },
    caption: 'Flow matching trains a velocity field to transport noise to data along **straight conditional paths** $x_t=(1-t)x_0+tx_1$. The training objective is an ordinary regression — no score, no SDE, no variance schedule to tune. Because the learned paths are close to straight, an ODE solver needs far fewer steps than a diffusion sampler. This (as rectified flow / stochastic interpolants) is what Stable Diffusion 3 and most current image and video models use.',
  });

  function velocity(x, y, tt) {
    // marginal velocity for a Gaussian-mixture endpoint with Gaussian source
    let wsum = 0, vx = 0, vy = 0;
    for (const m of MODES) {
      const sd2 = (1 - tt) ** 2 + (tt * m.s) ** 2;
      const mu = [tt * m.mu[0], tt * m.mu[1]];
      const q = m.w * Math.exp(-((x - mu[0]) ** 2 + (y - mu[1]) ** 2) / (2 * sd2)) / (2 * Math.PI * sd2);
      // E[x1 - x0 | x_t] for this component
      const d = [x - mu[0], y - mu[1]];
      const a = (tt * m.s * m.s - (1 - tt)) / sd2;
      const cvx = m.mu[0] + a * d[0] * 1 - 0;
      const cvy = m.mu[1] + a * d[1] * 1 - 0;
      // v = (x1 - x0); with x0 = (x_t - t x1)/(1-t) this reduces to below
      const ex1 = [mu[0] / Math.max(tt, 1e-6), mu[1] / Math.max(tt, 1e-6)];
      const num = [(m.mu[0] * (tt * m.s * m.s) + (1 - tt) * 0), 0];
      // use the standard closed form for conditional-OT paths
      const denom = Math.max(sd2, 1e-9);
      const x1hat = [
        (m.mu[0] * (1 - tt) ** 2 + x * tt * m.s * m.s) / denom,
        (m.mu[1] * (1 - tt) ** 2 + y * tt * m.s * m.s) / denom,
      ];
      const x0hat = [(x - tt * x1hat[0]) / Math.max(1 - tt, 1e-6), (y - tt * x1hat[1]) / Math.max(1 - tt, 1e-6)];
      wsum += q;
      vx += q * (x1hat[0] - x0hat[0]);
      vy += q * (x1hat[1] - x0hat[1]);
    }
    if (wsum < 1e-300) return [0, 0];
    return [vx / wsum, vy / wsum];
  }
  function pathDensity(x, y, tt) {
    let d = 0;
    for (const m of MODES) {
      const sd2 = (1 - tt) ** 2 + (tt * m.s) ** 2;
      d += m.w * Math.exp(-((x - tt * m.mu[0]) ** 2 + (y - tt * m.mu[1]) ** 2) / (2 * sd2)) / (2 * Math.PI * sd2);
    }
    return d;
  }
};

/* ============================================================
   Autoencoders & VAEs
   ============================================================ */

V['autoencoder'] = (host) => {
  panel(host, {
    title: 'Autoencoder: a bottleneck forces compression',
    height: 300,
    plot: { xlim: [-3, 3], ylim: [-2.4, 2.4], equal: true },
    controls: [
      { type: 'slider', key: 'dim', label: 'latent dimensions', min: 1, max: 2, step: 1, value: 1 },
      { type: 'slider', key: 'nonlin', label: 'nonlinearity strength', min: 0, max: 1, step: .02, value: 0 },
      { type: 'check', key: 'recon', label: 'show reconstructions', value: true },
    ],
    draw(p, s, P) {
      const r = rng(15);
      // data on a curved 1-D manifold
      const data = Array.from({ length: 160 }, () => {
        const t = (r() - .5) * 4;
        const bend = s.nonlin;
        return [t + r.normal(0, .12), .45 * t + bend * .55 * t * t - bend * 1.4 + r.normal(0, .12)];
      });
      const mx = data.reduce((a, d) => a + d[0], 0) / data.length;
      const my = data.reduce((a, d) => a + d[1], 0) / data.length;
      let c00 = 0, c01 = 0, c11 = 0;
      for (const d of data) { const a = d[0] - mx, b = d[1] - my; c00 += a * a; c01 += a * b; c11 += b * b; }
      const { vecs } = LA.eig2([[c00 / data.length, c01 / data.length], [c01 / data.length, c11 / data.length]]);
      const u = vecs[0];
      p.clear().axes({ nx: 5, ny: 4 });
      p.clip();
      p.points(data, { r: 3, color: alpha(cssVar('--s1'), .7) });
      if (s.dim === 1) {
        p.line([[mx - u[0] * 4, my - u[1] * 4], [mx + u[0] * 4, my + u[1] * 4]], { color: cssVar('--s3'), width: 2.2 });
        if (s.recon) {
          const rec = data.map((d) => {
            const t = (d[0] - mx) * u[0] + (d[1] - my) * u[1];
            return [mx + u[0] * t, my + u[1] * t];
          });
          data.forEach((d, i) => p.line([d, rec[i]], { color: alpha(cssVar('--s2'), .3), width: 1 }));
          p.points(rec, { r: 2.4, color: cssVar('--s2') });
        }
      }
      p.clip(false);
      let err = 0;
      for (const d of data) {
        const t = (d[0] - mx) * u[0] + (d[1] - my) * u[1];
        err += (d[0] - mx - u[0] * t) ** 2 + (d[1] - my - u[1] * t) ** 2;
      }
      P.readout({
        'latent dim': s.dim,
        'compression': s.dim === 1 ? '2 → 1' : 'none (identity is optimal)',
        'reconstruction MSE': s.dim === 1 ? fmt(err / data.length, 4) : '0',
        'note': s.nonlin > .3 && s.dim === 1 ? 'a LINEAR autoencoder cannot follow the curve — this is where depth pays' : '—',
      });
    },
    caption: 'A linear autoencoder with a $k$-dimensional bottleneck learns exactly the PCA subspace — nothing more. Now bend the data manifold: the straight line can no longer track it, and reconstruction error jumps. **That gap is what nonlinear encoders buy you.** A plain autoencoder gives you compression but not a *generative* model: nothing tells you which latent codes decode to something sensible, which is precisely the problem the VAE fixes.',
  });
};

V['vae-latent'] = (host) => {
  panel(host, {
    title: 'VAE: a latent space you can actually sample from',
    height: 320,
    plot: { xlim: [-3.4, 3.4], ylim: [-2.6, 2.6], equal: true, xlabel: 'z₁', ylabel: 'z₂' },
    controls: [
      { type: 'slider', key: 'beta', label: 'β (KL weight)', min: 0, max: 4, step: .05, value: 1 },
      { type: 'check', key: 'prior', label: 'show prior N(0, I)', value: true },
      { type: 'check', key: 'post', label: 'show per-example posteriors', value: true },
    ],
    draw(p, s, P) {
      const r = rng(19);
      const K = 8;
      // as beta grows, posteriors are pulled toward the prior and widen
      const pull = s.beta / (1 + s.beta);
      const centers = Array.from({ length: K }, (_, k) => {
        const a = k / K * 6.2832;
        const R = 2.1 * (1 - pull * .85);
        return [Math.cos(a) * R, Math.sin(a) * R * .8];
      });
      const sd = .12 + pull * .85;
      p.clear().axes({ nx: 5, ny: 4 });
      p.clip();
      if (s.prior) {
        for (const nsig of [1, 2]) p.circle(0, 0, nsig * p.sx, { stroke: alpha(cssVar('--s4'), nsig === 1 ? .8 : .4), width: 1.6, dash: [5, 4] });
      }
      const cols = SERIES();
      centers.forEach((c, k) => {
        if (s.post) p.ellipse(c[0], c[1], [[sd * sd, 0], [0, sd * sd]], { n: 2, stroke: alpha(cols[k % cols.length], .8), fill: alpha(cols[k % cols.length], .12), width: 1.4 });
        const pts = Array.from({ length: 24 }, () => [c[0] + r.normal(0, sd), c[1] + r.normal(0, sd)]);
        p.points(pts, { r: 2.2, color: alpha(cols[k % cols.length], .8) });
      });
      p.clip(false);
      // coverage: fraction of prior samples landing near some posterior
      let hit = 0;
      const rr = rng(3);
      for (let i = 0; i < 400; i++) {
        const z = [rr.normal(), rr.normal()];
        if (centers.some((c) => Math.hypot(z[0] - c[0], z[1] - c[1]) < sd * 2.2)) hit++;
      }
      P.readout({
        'β': fmt(s.beta, 2),
        'posterior width σ': fmt(sd, 3),
        'prior samples that decode to something seen': fmt(hit / 400 * 100, 0) + '%',
        'regime': s.beta < .3 ? 'autoencoder-like: sharp reconstructions, holes in latent space' : s.beta > 2.5 ? 'posterior collapse: latents carry no information ✗' : 'balanced',
      });
    },
    caption: 'The ELBO has two terms pulling against each other. **Reconstruction** wants each example\'s posterior to be a tight, distinctive blob — good for copying, terrible for sampling, because most of the prior lands in empty space. **KL** pulls every posterior toward $\\mathcal N(0,I)$ so the latent space is filled and samplable. Set β=0 and you get an autoencoder with holes; set β=4 and the posteriors merge entirely — *posterior collapse*, where the decoder ignores $z$ and the model degenerates.',
  });
};

/* ============================================================
   GANs
   ============================================================ */

V['gan-training'] = (host) => {
  let gen = { mu: -1.8, sd: .5 }, disc = { w: 1.2, b: 0 }, hist = [], step = 0;
  const target = { mu: 1.2, sd: .45 };
  const reset = () => { gen = { mu: -1.8, sd: .5 }; disc = { w: 1.2, b: 0 }; hist = []; step = 0; };

  panel(host, {
    title: 'GAN: a generator and a critic chasing each other',
    height: 300,
    plot: { xlim: [-4, 4], ylim: [0, 1.2], xlabel: 'x', ylabel: 'density / D(x)' },
    controls: [
      { type: 'slider', key: 'lrG', label: 'generator LR', min: .001, max: .1, step: .001, value: .02, fmt: (v) => v.toFixed(3) },
      { type: 'slider', key: 'lrD', label: 'discriminator LR', min: .001, max: .3, step: .001, value: .06, fmt: (v) => v.toFixed(3) },
      { type: 'slider', key: 'dsteps', label: 'D steps per G step', min: 1, max: 8, step: 1, value: 2 },
      { type: 'play' },
      { type: 'button', label: '↺ reset', onClick: reset },
    ],
    animate(s) {
      const r = rng(1000 + step);
      const sigmoid = (z) => 1 / (1 + Math.exp(-z));
      const D = (x) => sigmoid(disc.w * x + disc.b);
      // D steps
      for (let k = 0; k < Math.round(s.dsteps); k++) {
        let gw = 0, gb = 0;
        for (let i = 0; i < 32; i++) {
          const xr = target.mu + r.normal(0, target.sd);
          const xf = gen.mu + r.normal(0, gen.sd);
          const dr = D(xr), df = D(xf);
          gw += (1 - dr) * xr - df * xf;
          gb += (1 - dr) - df;
        }
        disc.w += s.lrD * gw / 32;
        disc.b += s.lrD * gb / 32;
        disc.w = Math.max(-14, Math.min(14, disc.w));
      }
      // G step (non-saturating)
      let gmu = 0;
      for (let i = 0; i < 32; i++) {
        const z = r.normal();
        const xf = gen.mu + gen.sd * z;
        const df = D(xf);
        gmu += (1 - df) * disc.w;
      }
      gen.mu += s.lrG * gmu / 32;
      step++;
      hist.push(gen.mu);
      if (hist.length > 400) hist.shift();
    },
    draw(p, s, P) {
      const N = (x, m, sd) => Math.exp(-((x - m) ** 2) / (2 * sd * sd)) / (sd * Math.sqrt(2 * Math.PI));
      p.clear().axes();
      p.clip();
      const a = [], b = [], d = [];
      for (let x = -4; x <= 4; x += .02) {
        a.push([x, N(x, target.mu, target.sd)]);
        b.push([x, N(x, gen.mu, gen.sd)]);
        d.push([x, 1 / (1 + Math.exp(-(disc.w * x + disc.b)))]);
      }
      p.area(a, { color: alpha(cssVar('--s1'), .2) }); p.line(a, { color: cssVar('--s1'), width: 2.4 });
      p.area(b, { color: alpha(cssVar('--s2'), .2) }); p.line(b, { color: cssVar('--s2'), width: 2.4 });
      p.line(d, { color: cssVar('--s3'), width: 2, dash: [5, 3] });
      p.clip(false);
      p.legend([
        { label: 'real data', color: cssVar('--s1') },
        { label: 'generator', color: cssVar('--s2') },
        { label: 'D(x) — prob. real', color: cssVar('--s3'), dash: true },
      ], { pos: 'tl' });
      P.readout({
        'step': step,
        'generator mean': fmt(gen.mu, 3),
        'target mean': fmt(target.mu, 3),
        'D weight': fmt(disc.w, 3),
        'D at equilibrium should be': '0.5 everywhere',
        'status': Math.abs(gen.mu - target.mu) < .1 ? 'converged ✓' : Math.abs(disc.w) > 10 ? 'D too strong — G gets no gradient ✗' : 'training',
      });
    },
    caption: 'Two networks, one objective, opposite signs. Press play: the discriminator learns to separate the distributions, and its gradient tells the generator which way to move. Set **D steps to 8 with a high D learning rate** — the discriminator becomes perfect, $D(x)$ turns into a step function, and the generator\'s gradient vanishes. That instability (plus mode collapse, invisible in 1-D) is why GANs were replaced by diffusion for most generation, and why WGAN and gradient penalties were invented.',
  });
};

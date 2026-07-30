/* ============================================================
   viz/classical.js — figures for classical ML.
   ============================================================ */

import { panel } from '../ui.js';
import { cssVar, alpha, mix, seqMap, divergeMap, LA, rng, fmt, SERIES } from '../plot.js';

const V = {};
export default V;

/* ---------- helpers shared by several demos ---------- */

function polyFit(xs, ys, deg, lam = 0) {
  const n = xs.length, d = deg + 1;
  const X = xs.map((x) => Array.from({ length: d }, (_, j) => x ** j));
  const XtX = LA.zeros(d, d), Xty = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < d; a++) {
      Xty[a] += X[i][a] * ys[i];
      for (let b = 0; b < d; b++) XtX[a][b] += X[i][a] * X[i][b];
    }
  }
  for (let a = 0; a < d; a++) XtX[a][a] += lam + 1e-9;
  return LA.solve(XtX, Xty);
}
const polyEval = (w, x) => w.reduce((s, c, j) => s + c * x ** j, 0);

/** Blobs / moons / circles / xor generators for classifier demos. */
function makeData(kind, n, noise, seed = 5) {
  const r = rng(seed);
  const pts = [];
  for (let i = 0; i < n; i++) {
    let x, y, c;
    if (kind === 'blobs') {
      c = i % 2;
      const cx = c ? 1.1 : -1.1, cy = c ? .8 : -.7;
      x = cx + r.normal(0, .55 + noise); y = cy + r.normal(0, .55 + noise);
    } else if (kind === 'moons') {
      c = i % 2;
      const t = r() * Math.PI;
      if (c) { x = 1 - Math.cos(t) - .5; y = -Math.sin(t) + .35; }
      else { x = Math.cos(t) - .5; y = Math.sin(t) - .35; }
      x = x * 1.4 + r.normal(0, noise); y = y * 1.4 + r.normal(0, noise);
    } else if (kind === 'circles') {
      c = i % 2;
      const t = r() * 6.2832, rad = (c ? 1.75 : .75) + r.normal(0, noise * 1.5);
      x = Math.cos(t) * rad; y = Math.sin(t) * rad;
    } else if (kind === 'xor') {
      x = (r() - .5) * 4; y = (r() - .5) * 4;
      c = (x > 0) === (y > 0) ? 1 : 0;
      x += r.normal(0, noise); y += r.normal(0, noise);
    } else if (kind === 'spiral') {
      c = i % 2;
      const t = (i / n) * 4.2 + r() * .25;
      const sgn = c ? 1 : -1;
      x = sgn * t * Math.cos(t * 1.4) * .55 + r.normal(0, noise);
      y = sgn * t * Math.sin(t * 1.4) * .55 + r.normal(0, noise);
    }
    pts.push({ x, y, c });
  }
  return pts;
}

const DATASETS = [
  { value: 'blobs', label: 'two blobs (linearly separable)' },
  { value: 'moons', label: 'two moons' },
  { value: 'circles', label: 'concentric circles' },
  { value: 'xor', label: 'XOR' },
  { value: 'spiral', label: 'spirals' },
];

function drawPoints(p, pts, o = {}) {
  const c0 = cssVar('--s1'), c1 = cssVar('--s2');
  p.points(pts.map((q) => [q.x, q.y]), {
    r: o.r || 4,
    color: (_, i) => (pts[i].c ? c1 : c0),
    stroke: alpha(cssVar('--bg-inset'), .9), strokeWidth: 1.2,
  });
}

/** Paint a decision surface from a score function. */
function paintDecision(p, score, o = {}) {
  p.heat((x, y) => score(x, y), {
    step: o.step || 5,
    lo: o.lo != null ? o.lo : -1, hi: o.hi != null ? o.hi : 1,
    cmap: (t) => mix(mix(cssVar('--bg-inset'), cssVar('--s1'), .42), mix(cssVar('--bg-inset'), cssVar('--s2'), .42), t),
  });
}

/* ============================================================
   Linear regression
   ============================================================ */

V['linear-regression'] = (host, params = {}) => {
  const r = rng(21);
  let pts = Array.from({ length: 14 }, (_, i) => {
    const x = -2.4 + i * .38 + r.normal(0, .1);
    return [x, 1.15 * x + .6 + r.normal(0, .7)];
  });

  panel(host, {
    title: 'Least squares — drag any point',
    height: 320,
    plot: { xlim: [-3.2, 3.2], ylim: [-4, 4.5], xlabel: 'x', ylabel: 'y' },
    controls: [
      { type: 'check', key: 'resid', label: 'squared residuals', value: true },
      { type: 'check', key: 'manual', label: 'manual line (compare to optimum)', value: false },
      { type: 'slider', key: 'w0', label: 'manual intercept', min: -3, max: 3, step: .02, value: 0 },
      { type: 'slider', key: 'w1', label: 'manual slope', min: -3, max: 3, step: .02, value: 0.4 },
      { type: 'button', label: '+ outlier', onClick: () => pts.push([2.6, -3.2]) },
      { type: 'button', label: '↺ reset', onClick: () => { pts = Array.from({ length: 14 }, (_, i) => { const x = -2.4 + i * .38 + r.normal(0, .1); return [x, 1.15 * x + .6 + r.normal(0, .7)]; }); } },
    ],
    interact: (s, P) => ({
      down(x, y, e, cx, cy) { P._i = P.plot.nearest(pts, x, y, 16); return P._i >= 0; },
      move(x, y) { if (P._i >= 0) { pts[P._i] = [x, y]; P.redraw(); } },
      up() { P._i = -1; },
      cursor: 'grab',
    }),
    draw(p, s, P) {
      const n = pts.length;
      const mx = pts.reduce((a, q) => a + q[0], 0) / n;
      const my = pts.reduce((a, q) => a + q[1], 0) / n;
      let sxy = 0, sxx = 0, syy = 0;
      for (const q of pts) { sxy += (q[0] - mx) * (q[1] - my); sxx += (q[0] - mx) ** 2; syy += (q[1] - my) ** 2; }
      const b1 = sxx > 1e-9 ? sxy / sxx : 0, b0 = my - b1 * mx;

      p.clear().axes();
      p.clip();
      const line = (w0, w1, col, w, dash) => p.fn((x) => w0 + w1 * x, { color: col, width: w, dash });

      if (s.resid) {
        const g = p.ctx;
        for (const q of pts) {
          const yh = s.manual ? s.w0 + s.w1 * q[0] : b0 + b1 * q[0];
          const res = q[1] - yh;
          // square with side = |residual|
          const x1 = p.px(q[0]), y1 = p.py(q[1]), y2 = p.py(yh);
          const side = Math.abs(y1 - y2);
          g.fillStyle = alpha(cssVar('--s2'), .16);
          g.strokeStyle = alpha(cssVar('--s2'), .45);
          g.lineWidth = 1;
          g.fillRect(x1, Math.min(y1, y2), side * Math.sign(1), Math.abs(y1 - y2));
          g.strokeRect(x1, Math.min(y1, y2), side, Math.abs(y1 - y2));
          p.line([[q[0], q[1]], [q[0], yh]], { color: alpha(cssVar('--s2'), .8), width: 1.4 });
        }
      }
      line(b0, b1, cssVar('--s3'), 2.6);
      if (s.manual) line(s.w0, s.w1, cssVar('--s4'), 2, [6, 3]);
      p.points(pts, { r: 5, color: cssVar('--s1'), stroke: cssVar('--bg-inset'), strokeWidth: 1.5 });
      p.points([[mx, my]], { r: 4, color: cssVar('--s5'), shape: 'cross' });
      p.clip(false);
      p.legend([
        { label: 'least-squares fit', color: cssVar('--s3') },
        ...(s.manual ? [{ label: 'your line', color: cssVar('--s4'), dash: true }] : []),
        { label: 'centroid (x̄, ȳ)', color: cssVar('--s5'), shape: 'dot' },
      ], { pos: 'tl' });

      const mse = (w0, w1) => pts.reduce((a, q) => a + (q[1] - w0 - w1 * q[0]) ** 2, 0) / n;
      const best = mse(b0, b1);
      const r2 = syy > 1e-9 ? 1 - best * n / syy : 0;
      P.readout({
        'ŵ₀ (intercept)': fmt(b0, 4), 'ŵ₁ (slope)': fmt(b1, 4),
        'MSE': fmt(best, 4), 'R²': fmt(r2, 4),
        ...(s.manual ? { 'your MSE': fmt(mse(s.w0, s.w1), 4), 'excess': fmt(mse(s.w0, s.w1) - best, 4) } : {}),
      });
    },
    caption: 'Every orange square is a **squared** residual — least squares minimizes their total area, which is why a single far-away point (try **+ outlier**) can swing the whole line. The fitted line always passes through the centroid $(\\bar x, \\bar y)$, and the slope is $\\hat\\beta_1 = \\mathrm{Cov}(x,y)/\\mathrm{Var}(x)$. Turn on the manual line and try to beat the optimum: you cannot.',
  });
};

/* ---------- loss surface for linear regression ---------- */

V['regression-loss-surface'] = (host) => {
  const r = rng(21);
  const pts = Array.from({ length: 14 }, (_, i) => {
    const x = -2.4 + i * .38 + r.normal(0, .1);
    return [x, 1.15 * x + .6 + r.normal(0, .7)];
  });
  const mse = (w0, w1) => pts.reduce((a, q) => a + (q[1] - w0 - w1 * q[0]) ** 2, 0) / pts.length;
  let path = [];
  let w = [-2.5, -2];

  panel(host, {
    title: 'The same problem, seen in parameter space',
    height: 320,
    plot: { xlim: [-3, 3], ylim: [-2.5, 3], xlabel: 'w₀ (intercept)', ylabel: 'w₁ (slope)' },
    controls: [
      { type: 'slider', key: 'lr', label: 'learning rate', min: .005, max: .3, step: .005, value: .06 },
      { type: 'play' },
      { type: 'button', label: '↺ reset', onClick: () => { w = [-2.5, -2]; path = []; } },
      { type: 'label', label: 'click to place the starting weights' },
    ],
    interact: (s, P) => ({
      down(x, y) { w = [x, y]; path = [[x, y]]; P.redraw(); return true; },
      move(x, y) { w = [x, y]; path = [[x, y]]; P.redraw(); },
    }),
    animate(s) {
      const n = pts.length;
      let g0 = 0, g1 = 0;
      for (const q of pts) { const e = w[0] + w[1] * q[0] - q[1]; g0 += 2 * e / n; g1 += 2 * e * q[0] / n; }
      w = [w[0] - s.lr * g0, w[1] - s.lr * g1];
      path.push([...w]);
      if (path.length > 600) path.shift();
    },
    draw(p, s, P) {
      p.clear();
      p.heat((a, b) => Math.log1p(mse(a, b)), { step: 5, cmap: (t) => mix(cssVar('--bg-inset'), cssVar('--s1'), t * .55) });
      p.axes();
      p.clip();
      const lv = [];
      for (let i = 1; i <= 14; i++) lv.push((i / 14) ** 2 * 30);
      p.contour(mse, lv, { color: alpha(cssVar('--axis'), .5) });
      // exact optimum
      const n = pts.length;
      const mx = pts.reduce((a, q) => a + q[0], 0) / n, my = pts.reduce((a, q) => a + q[1], 0) / n;
      let sxy = 0, sxx = 0;
      for (const q of pts) { sxy += (q[0] - mx) * (q[1] - my); sxx += (q[0] - mx) ** 2; }
      const b1 = sxy / sxx, b0 = my - b1 * mx;
      p.points([[b0, b1]], { r: 6, color: cssVar('--s3'), shape: 'cross', width: 2.5 });
      p.line(path, { color: cssVar('--s2'), width: 2 });
      p.circle(w[0], w[1], 5, { fill: cssVar('--s2'), stroke: cssVar('--bg-inset'), width: 2 });
      p.clip(false);
      P.readout({ 'w': `(${fmt(w[0], 3)}, ${fmt(w[1], 3)})`, 'MSE': fmt(mse(w[0], w[1]), 5), 'optimum': `(${fmt(b0, 3)}, ${fmt(b1, 3)})`, 'steps': path.length });
    },
    caption: 'Same fourteen points, different view: each **position** here is a whole line in the previous figure, and the height is its MSE. For linear regression the surface is an exact paraboloid — convex, one minimum, reachable in closed form. Everything after this lesson is about what happens when the surface stops being this nice.',
  });
};

/* ============================================================
   Polynomial fitting: underfit → overfit
   ============================================================ */

V['polynomial-overfit'] = (host) => {
  const truth = (x) => Math.sin(x * 2.2) * 1.3 + .35 * x;
  let seed = 4;
  let train, test;
  const regen = (n, noise) => {
    const r = rng(seed);
    train = Array.from({ length: n }, () => { const x = (r() - .5) * 5; return [x, truth(x) + r.normal(0, noise)]; });
    test = Array.from({ length: 120 }, () => { const x = (r() - .5) * 5; return [x, truth(x) + r.normal(0, noise)]; });
  };
  regen(12, .35);

  panel(host, {
    title: 'Model complexity: underfitting, fitting, memorizing',
    height: 320,
    plot: { xlim: [-2.8, 2.8], ylim: [-3.2, 3.2], xlabel: 'x', ylabel: 'y' },
    controls: [
      { type: 'slider', key: 'deg', label: 'polynomial degree', min: 0, max: 14, step: 1, value: 3 },
      { type: 'slider', key: 'n', label: 'training points', min: 4, max: 60, step: 1, value: 12, onChange: (v, P) => regen(Math.round(v), P.state.noise) },
      { type: 'slider', key: 'noise', label: 'label noise σ', min: 0, max: 1, step: .02, value: .35, onChange: (v, P) => regen(Math.round(P.state.n), v) },
      { type: 'slider', key: 'lam', label: 'ridge λ (log₁₀)', min: -8, max: 2, step: .1, value: -8, fmt: (v) => (v <= -7.9 ? '0' : `1e${v.toFixed(1)}`) },
      { type: 'check', key: 'showTest', label: 'show held-out points', value: false },
      { type: 'button', label: '↻ new sample', onClick: (s) => { seed++; regen(Math.round(s.n), s.noise); } },
    ],
    draw(p, s, P) {
      const lam = s.lam <= -7.9 ? 0 : 10 ** s.lam;
      const w = polyFit(train.map((d) => d[0]), train.map((d) => d[1]), Math.round(s.deg), lam);
      p.clear().axes();
      p.clip();
      p.fn(truth, { color: alpha(cssVar('--s3'), .8), width: 2, dash: [6, 4] });
      p.fn((x) => polyEval(w, x), { color: cssVar('--s2'), width: 2.6, n: 400 });
      if (s.showTest) p.points(test, { r: 2.2, color: alpha(cssVar('--s4'), .45) });
      p.points(train, { r: 4.6, color: cssVar('--s1'), stroke: cssVar('--bg-inset'), strokeWidth: 1.4 });
      p.clip(false);
      p.legend([
        { label: 'true function', color: cssVar('--s3'), dash: true },
        { label: `degree-${Math.round(s.deg)} fit`, color: cssVar('--s2') },
        { label: 'training data', color: cssVar('--s1'), shape: 'dot' },
      ], { pos: 'tl' });
      const err = (D) => D.reduce((a, d) => a + (polyEval(w, d[0]) - d[1]) ** 2, 0) / D.length;
      const tr = err(train), te = err(test);
      P.readout({
        'params': Math.round(s.deg) + 1,
        'train MSE': fmt(tr, 4),
        'test MSE': fmt(te, 4),
        'gap': fmt(te - tr, 4),
        'verdict': te > tr * 3 && s.deg > 4 ? 'overfitting ✗' : s.deg < 2 ? 'underfitting ✗' : 'reasonable ✓',
        '‖w‖': fmt(Math.sqrt(w.reduce((a, c) => a + c * c, 0)), 3),
      });
    },
    caption: 'Degree 0–1: the model is too rigid to see the wave (**underfitting** — high bias). Degree 12+ with 12 points: the curve snakes through every point and train error hits ~0 while test error explodes (**overfitting** — high variance). Now raise **ridge λ** on the wiggly fit and watch it settle down without changing the degree. That is regularization: constraining the *effective* capacity rather than the parameter count.',
  });
};

/* ---------- bias-variance decomposition ---------- */

V['bias-variance'] = (host) => {
  const truth = (x) => Math.sin(x * 2.2) * 1.3 + .35 * x;
  panel(host, {
    title: 'Bias and variance, measured over many datasets',
    height: 330,
    plot: { xlim: [-2.6, 2.6], ylim: [-3, 3], xlabel: 'x', ylabel: 'y' },
    controls: [
      { type: 'slider', key: 'deg', label: 'degree', min: 0, max: 12, step: 1, value: 1 },
      { type: 'slider', key: 'n', label: 'points per dataset', min: 5, max: 40, step: 1, value: 10 },
      { type: 'slider', key: 'reps', label: 'datasets drawn', min: 3, max: 60, step: 1, value: 25 },
    ],
    draw(p, s, P) {
      const deg = Math.round(s.deg), n = Math.round(s.n), R = Math.round(s.reps);
      const fits = [];
      for (let k = 0; k < R; k++) {
        const r = rng(100 + k);
        const xs = [], ys = [];
        for (let i = 0; i < n; i++) { const x = (r() - .5) * 5; xs.push(x); ys.push(truth(x) + r.normal(0, .35)); }
        fits.push(polyFit(xs, ys, deg, 1e-7));
      }
      p.clear().axes();
      p.clip();
      for (const w of fits) p.fn((x) => polyEval(w, x), { color: alpha(cssVar('--s2'), .22), width: 1.2, n: 200 });
      // mean prediction
      const meanF = (x) => fits.reduce((a, w) => a + polyEval(w, x), 0) / fits.length;
      p.fn(meanF, { color: cssVar('--s2'), width: 2.8, n: 300 });
      p.fn(truth, { color: cssVar('--s3'), width: 2.4, dash: [6, 4] });
      p.clip(false);
      p.legend([
        { label: 'true f(x)', color: cssVar('--s3'), dash: true },
        { label: 'average prediction', color: cssVar('--s2') },
        { label: 'individual fits', color: alpha(cssVar('--s2'), .35) },
      ], { pos: 'tl' });
      // decomposition on a grid
      let bias2 = 0, varr = 0, cnt = 0;
      for (let x = -2.4; x <= 2.4; x += .05) {
        const m = meanF(x);
        bias2 += (m - truth(x)) ** 2;
        varr += fits.reduce((a, w) => a + (polyEval(w, x) - m) ** 2, 0) / fits.length;
        cnt++;
      }
      bias2 /= cnt; varr /= cnt;
      P.readout({
        'bias²': fmt(bias2, 4), 'variance': fmt(varr, 4),
        'noise σ²': fmt(.35 ** 2, 4),
        'expected test MSE': fmt(bias2 + varr + .35 ** 2, 4),
      });
    },
    caption: 'Each faint curve is the model fit to a *different* random draw of the same size. **Bias²** is how far the bold average curve sits from the dashed truth; **variance** is how much the faint curves scatter around that average. Degree 1: huge bias, tiny variance. Degree 10: near-zero bias, wild variance. Test error is their sum plus irreducible noise — and the minimum is somewhere in between. Now raise **points per dataset** and watch variance shrink: more data buys you complexity.',
  });
};

/* ---------- double descent ---------- */

V['double-descent'] = (host) => {
  panel(host, {
    title: 'Double descent: the classical picture is incomplete',
    height: 300,
    plot: { xlim: [0, 45], ylim: [0, 1.6], xlabel: 'model capacity (# features / # training points)', ylabel: 'error' },
    controls: [
      { type: 'slider', key: 'n', label: 'training points n', min: 8, max: 40, step: 1, value: 20 },
      { type: 'slider', key: 'noise', label: 'label noise', min: 0, max: .6, step: .02, value: .25 },
      { type: 'slider', key: 'lam', label: 'ridge λ (log₁₀)', min: -8, max: 0, step: .2, value: -8, fmt: (v) => (v <= -7.9 ? '0' : `1e${v.toFixed(1)}`) },
    ],
    draw(p, s, P) {
      const n = Math.round(s.n);
      const lam = s.lam <= -7.9 ? 1e-9 : 10 ** s.lam;
      const r = rng(9);
      // random-features regression on a fixed 1-D target
      const truth = (x) => Math.sin(3 * x) + .4 * x;
      const xs = Array.from({ length: n }, () => (r() - .5) * 4);
      const ys = xs.map((x) => truth(x) + r.normal(0, s.noise));
      const xt = Array.from({ length: 200 }, () => (r() - .5) * 4);
      const yt = xt.map((x) => truth(x));
      const W = Array.from({ length: 60 }, () => [r.normal(0, 1.2), r.normal(0, 1.5)]);
      const feat = (x, d) => Array.from({ length: d }, (_, j) => Math.tanh(W[j][0] * x + W[j][1]));

      const trainCurve = [], testCurve = [];
      for (let d = 1; d <= 60; d++) {
        const X = xs.map((x) => feat(x, d));
        const XtX = LA.zeros(d, d), Xty = new Array(d).fill(0);
        for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) {
          Xty[a] += X[i][a] * ys[i];
          for (let b = 0; b < d; b++) XtX[a][b] += X[i][a] * X[i][b];
        }
        for (let a = 0; a < d; a++) XtX[a][a] += lam;
        const w = LA.solve(XtX, Xty);
        const pred = (x) => LA.dot(feat(x, d), w);
        const tr = xs.reduce((a, x, i) => a + (pred(x) - ys[i]) ** 2, 0) / n;
        const te = xt.reduce((a, x, i) => a + (pred(x) - yt[i]) ** 2, 0) / xt.length;
        const ratio = d / n * 20;   // x-axis in units where n maps to 20
        trainCurve.push([ratio, Math.min(tr, 1.55)]);
        testCurve.push([ratio, Math.min(te, 1.55)]);
      }
      p.clear().axes();
      p.clip();
      // interpolation threshold
      p.line([[20, 0], [20, 1.6]], { color: alpha(cssVar('--danger'), .7), width: 1.6, dash: [5, 4] });
      p.line(trainCurve, { color: cssVar('--s1'), width: 2.4 });
      p.line(testCurve, { color: cssVar('--s2'), width: 2.6 });
      p.clip(false);
      p.text(20, 1.5, ' d = n  (interpolation threshold)', { color: cssVar('--danger'), size: 10 });
      p.legend([{ label: 'train error', color: cssVar('--s1') }, { label: 'test error', color: cssVar('--s2') }], { pos: 'tr' });
      P.readout({
        'n': n,
        'test error at d=n': fmt(testCurve[Math.min(n - 1, 59)][1], 3),
        'test error at d=60': fmt(testCurve[59][1], 3),
        'note': lam > 1e-6 ? 'ridge damps the spike' : 'no regularization → sharp peak',
      });
    },
    caption: 'The U-curve you were taught is only the left half. Right at $d = n$ the model can *just barely* interpolate the data and it does so with enormous weights — test error spikes. Push past it and error **descends again**, often below the classical minimum, because among the many interpolating solutions the minimum-norm one is smooth. Add ridge λ and the spike melts away. This is the regime deep networks live in.',
  });
};

/* ---------- regularization geometry ---------- */

V['regularization-geometry'] = (host) => {
  panel(host, {
    title: 'Why L1 zeroes coefficients and L2 does not',
    height: 330,
    plot: { xlim: [-2.2, 2.6], ylim: [-1.8, 2.2], equal: true, xlabel: 'w₁', ylabel: 'w₂' },
    controls: [
      { type: 'select', key: 'norm', label: 'penalty', value: 'l1', options: [{ value: 'l1', label: 'L1 (Lasso)' }, { value: 'l2', label: 'L2 (Ridge)' }, { value: 'elastic', label: 'Elastic net' }] },
      { type: 'slider', key: 't', label: 'budget size', min: .1, max: 2.2, step: .02, value: .9 },
      { type: 'slider', key: 'ang', label: 'data correlation (tilts contours)', min: -1.2, max: 1.2, step: .02, value: .5 },
    ],
    draw(p, s, P) {
      const th = s.ang;
      const A = [[1 + 1.6 * Math.cos(th) ** 2, 1.6 * Math.cos(th) * Math.sin(th)], [1.6 * Math.cos(th) * Math.sin(th), 1 + 1.6 * Math.sin(th) ** 2]];
      const wOLS = [1.5, 1.0];
      const loss = (a, b) => {
        const d = [a - wOLS[0], b - wOLS[1]];
        return A[0][0] * d[0] * d[0] + 2 * A[0][1] * d[0] * d[1] + A[1][1] * d[1] * d[1];
      };
      p.clear().axes();
      p.clip();
      p.contour(loss, [.05, .2, .5, 1, 1.8, 3, 4.5, 6.5, 9], { color: alpha(cssVar('--s1'), .55), width: 1.3 });
      // constraint region
      const g = p.ctx;
      g.save();
      g.beginPath();
      const t = s.t;
      if (s.norm === 'l2') {
        for (let k = 0; k <= 96; k++) {
          const a = k / 96 * 6.2832;
          const X = p.px(Math.cos(a) * t), Y = p.py(Math.sin(a) * t);
          k ? g.lineTo(X, Y) : g.moveTo(X, Y);
        }
      } else if (s.norm === 'l1') {
        const pts = [[t, 0], [0, t], [-t, 0], [0, -t]];
        pts.forEach((q, k) => { const X = p.px(q[0]), Y = p.py(q[1]); k ? g.lineTo(X, Y) : g.moveTo(X, Y); });
      } else {
        for (let k = 0; k <= 160; k++) {
          const a = k / 160 * 6.2832;
          const cx = Math.cos(a), cy = Math.sin(a);
          // solve for radius where .5*|w|_1 + .5*|w|^2 = t
          let lo = 0, hi = 5;
          for (let it = 0; it < 30; it++) {
            const m = (lo + hi) / 2;
            const v = .5 * (Math.abs(cx * m) + Math.abs(cy * m)) + .5 * m * m;
            v > t ? (hi = m) : (lo = m);
          }
          const X = p.px(cx * lo), Y = p.py(cy * lo);
          k ? g.lineTo(X, Y) : g.moveTo(X, Y);
        }
      }
      g.closePath();
      g.fillStyle = alpha(cssVar('--s3'), .16);
      g.strokeStyle = cssVar('--s3'); g.lineWidth = 2;
      g.fill(); g.stroke();
      g.restore();

      // constrained optimum by dense search on the boundary
      let best = null, bl = Infinity;
      for (let k = 0; k < 2000; k++) {
        const a = k / 2000 * 6.2832;
        let q;
        if (s.norm === 'l2') q = [Math.cos(a) * t, Math.sin(a) * t];
        else if (s.norm === 'l1') {
          const c = Math.cos(a), sn = Math.sin(a);
          const scale = t / (Math.abs(c) + Math.abs(sn));
          q = [c * scale, sn * scale];
        } else {
          const c = Math.cos(a), sn = Math.sin(a);
          let lo = 0, hi = 5;
          for (let it = 0; it < 26; it++) {
            const m = (lo + hi) / 2;
            (.5 * (Math.abs(c * m) + Math.abs(sn * m)) + .5 * m * m) > t ? (hi = m) : (lo = m);
          }
          q = [c * lo, sn * lo];
        }
        const l = loss(q[0], q[1]);
        if (l < bl) { bl = l; best = q; }
      }
      p.points([wOLS], { r: 5, color: cssVar('--s4') });
      p.text(wOLS[0], wOLS[1], '  OLS', { color: cssVar('--s4'), size: 11, weight: '600' });
      p.points([best], { r: 6, color: cssVar('--s2'), stroke: cssVar('--bg-inset'), strokeWidth: 2 });
      p.clip(false);
      P.readout({
        'ŵ (penalized)': `(${fmt(best[0], 3)}, ${fmt(best[1], 3)})`,
        'exactly zero?': Math.abs(best[0]) < 1e-3 || Math.abs(best[1]) < 1e-3 ? 'YES — sparse ✓' : 'no',
        'shrinkage vs OLS': fmt(1 - Math.hypot(best[0], best[1]) / Math.hypot(...wOLS), 3),
      });
    },
    caption: 'The green region is the parameter budget; the blue ovals are equal-loss contours around the unpenalized solution. The answer is where the smallest oval **touches** the region. The L1 diamond has corners *on the axes*, so contact happens at a corner and a coefficient becomes exactly zero. The L2 circle is smooth — contact almost never lands on an axis, so ridge shrinks everything but zeroes nothing. Shrink the budget and watch it happen.',
  });
};

/* ---------- coefficient paths ---------- */

V['regularization-path'] = (host) => {
  panel(host, {
    title: 'Coefficient paths as the penalty tightens',
    height: 290,
    plot: { xlim: [-4, 2], ylim: [-1.5, 2.2], xlabel: 'log₁₀ λ', ylabel: 'coefficient value' },
    controls: [
      { type: 'select', key: 'norm', label: 'penalty', value: 'l2', options: [{ value: 'l2', label: 'Ridge (L2)' }, { value: 'l1', label: 'Lasso (L1)' }] },
      { type: 'slider', key: 'corr', label: 'feature correlation', min: 0, max: .95, step: .01, value: .3 },
    ],
    draw(p, s, P) {
      const r = rng(17);
      const n = 60, d = 8;
      const wTrue = [2, -1.4, .9, 0, 0, 0, 0, .4];
      const X = [], y = [];
      for (let i = 0; i < n; i++) {
        const base = r.normal(0, 1);
        const row = Array.from({ length: d }, (_, j) => s.corr * base + Math.sqrt(1 - s.corr ** 2) * r.normal(0, 1));
        X.push(row);
        y.push(LA.dot(row, wTrue) + r.normal(0, .5));
      }
      const XtX = LA.zeros(d, d), Xty = new Array(d).fill(0);
      for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) {
        Xty[a] += X[i][a] * y[i];
        for (let b = 0; b < d; b++) XtX[a][b] += X[i][a] * X[i][b];
      }
      const paths = Array.from({ length: d }, () => []);
      for (let L = -4; L <= 2; L += .06) {
        const lam = 10 ** L * n;
        let w;
        if (s.norm === 'l2') {
          const M = XtX.map((row, a) => row.map((v, b) => v + (a === b ? lam : 0)));
          w = LA.solve(M, Xty);
        } else {
          // coordinate descent for lasso
          w = new Array(d).fill(0);
          for (let it = 0; it < 220; it++) {
            for (let j = 0; j < d; j++) {
              let rho = Xty[j];
              for (let k = 0; k < d; k++) if (k !== j) rho -= XtX[j][k] * w[k];
              const z = XtX[j][j] || 1;
              w[j] = Math.sign(rho) * Math.max(0, Math.abs(rho) - lam) / z;
            }
          }
        }
        for (let j = 0; j < d; j++) paths[j].push([L, w[j]]);
      }
      p.clear().axes();
      p.clip();
      const cols = SERIES();
      paths.forEach((pa, j) => {
        p.line(pa, { color: cols[j % cols.length], width: wTrue[j] !== 0 ? 2.4 : 1.4, alpha: wTrue[j] !== 0 ? 1 : .55 });
      });
      p.clip(false);
      p.legend([
        { label: 'true signal features (bold)', color: cssVar('--s1') },
        { label: 'true zero features (faint)', color: alpha(cssVar('--s2'), .55) },
      ], { pos: 'bl' });
      const atMid = paths.map((pa) => pa[Math.floor(pa.length * .55)][1]);
      P.readout({
        'exactly-zero coefficients at λ=10^-0.7': atMid.filter((v) => Math.abs(v) < 1e-6).length + ` / ${d}`,
        'penalty': s.norm === 'l1' ? 'L1 → hits zero and stays' : 'L2 → shrinks but never reaches zero',
      });
    },
    caption: 'Read right-to-left as the penalty loosens. **Lasso paths hit exactly zero and stop** — it performs feature selection. **Ridge paths approach zero asymptotically** — every feature keeps a small vote. Turn up feature correlation and watch Lasso become unstable, arbitrarily picking one of a correlated pair — the failure mode elastic net was invented to fix.',
  });
};

/* ============================================================
   Logistic regression
   ============================================================ */

V['logistic-regression'] = (host) => {
  let pts = makeData('blobs', 60, .1, 3);
  let w = [0, 0, 0]; // b, w1, w2

  const fit = (iters, lr, l2) => {
    for (let it = 0; it < iters; it++) {
      const g = [0, 0, 0];
      for (const q of pts) {
        const z = w[0] + w[1] * q.x + w[2] * q.y;
        const pr = 1 / (1 + Math.exp(-z));
        const e = pr - q.c;
        g[0] += e; g[1] += e * q.x; g[2] += e * q.y;
      }
      const n = pts.length;
      for (let k = 0; k < 3; k++) w[k] -= lr * (g[k] / n + (k ? l2 * w[k] : 0));
    }
  };

  panel(host, {
    title: 'Logistic regression — a linear boundary with calibrated confidence',
    height: 330,
    plot: { xlim: [-3.4, 3.4], ylim: [-2.6, 2.6], equal: true, xlabel: 'x₁', ylabel: 'x₂' },
    controls: [
      { type: 'select', key: 'data', label: 'dataset', value: 'blobs', options: DATASETS.slice(0, 4), onChange: (v, P) => { pts = makeData(v, 60, .1, 3); w = [0, 0, 0]; } },
      { type: 'slider', key: 'l2', label: 'L2 penalty', min: 0, max: .5, step: .005, value: 0 },
      { type: 'check', key: 'prob', label: 'show probability field', value: true },
      { type: 'button', label: 'train 200 steps', onClick: (s) => fit(200, .5, s.l2), primary: true },
      { type: 'button', label: '↺ reset weights', onClick: () => (w = [0, 0, 0]) },
      { type: 'label', label: 'click the plot to add a point of the majority class' },
    ],
    interact: (s, P) => ({
      down(x, y, e) { pts.push({ x, y, c: e.shiftKey ? 1 : 0 }); P.redraw(); return false; },
    }),
    draw(p, s, P) {
      const sig = (z) => 1 / (1 + Math.exp(-z));
      p.clear();
      if (s.prob) p.heat((x, y) => sig(w[0] + w[1] * x + w[2] * y), {
        step: 4, lo: 0, hi: 1,
        cmap: (t) => mix(mix(cssVar('--bg-inset'), cssVar('--s1'), .5), mix(cssVar('--bg-inset'), cssVar('--s2'), .5), t),
      });
      p.axes({ nx: 5, ny: 4 });
      p.clip();
      // boundary + margins
      if (Math.abs(w[2]) > 1e-6 || Math.abs(w[1]) > 1e-6) {
        for (const [lvl, dash, col] of [[0, null, cssVar('--text')], [Math.log(3), [5, 4], alpha(cssVar('--text'), .4)], [-Math.log(3), [5, 4], alpha(cssVar('--text'), .4)]]) {
          p.contour((x, y) => w[0] + w[1] * x + w[2] * y - lvl, [0], { color: col, width: lvl === 0 ? 2.4 : 1.2 });
        }
      }
      drawPoints(p, pts, { r: 4.5 });
      p.clip(false);
      let ll = 0, acc = 0;
      for (const q of pts) {
        const pr = sig(w[0] + w[1] * q.x + w[2] * q.y);
        ll += q.c ? Math.log(Math.max(pr, 1e-12)) : Math.log(Math.max(1 - pr, 1e-12));
        acc += (pr > .5) === !!q.c ? 1 : 0;
      }
      P.readout({
        'w': `(${fmt(w[1], 3)}, ${fmt(w[2], 3)})`, 'b': fmt(w[0], 3),
        'mean log-lik': fmt(ll / pts.length, 4),
        'accuracy': fmt(acc / pts.length * 100, 1) + '%',
        '‖w‖ (confidence scale)': fmt(Math.hypot(w[1], w[2]), 3),
      });
    },
    caption: 'Press **train** repeatedly. The solid line is $p=0.5$; the dashed lines are $p=0.25$ and $p=0.75$. Notice what $\\|w\\|$ controls: the *direction* sets the boundary, the *magnitude* sets how fast probability swings from 0 to 1 across it. On separable data $\\|w\\|$ grows without bound (log-loss keeps improving) — which is exactly why you need the L2 penalty. Try the **circles** dataset to see the model fail: no line can do it.',
  });
};

V['sigmoid-softmax'] = (host) => {
  panel(host, {
    title: 'From scores to probabilities',
    height: 270,
    plot: { xlim: [-6, 6], ylim: [-0.05, 1.05], xlabel: 'z (logit)', ylabel: 'probability' },
    controls: [
      { type: 'slider', key: 'temp', label: 'temperature T', min: .1, max: 4, step: .05, value: 1 },
      { type: 'check', key: 'grad', label: "show σ'(z)", value: true },
    ],
    draw(p, s, P) {
      p.clear().axes();
      p.clip();
      const sig = (z) => 1 / (1 + Math.exp(-z / s.temp));
      p.fn(sig, { color: cssVar('--s1'), width: 2.8 });
      if (s.grad) {
        p.fn((z) => { const t = sig(z); return t * (1 - t) / s.temp; }, { color: cssVar('--s2'), width: 2, dash: [5, 3] });
      }
      p.line([[-6, .5], [6, .5]], { color: alpha(cssVar('--axis'), .5), width: 1, dash: [2, 3] });
      p.clip(false);
      p.legend([{ label: 'σ(z/T)', color: cssVar('--s1') }, ...(s.grad ? [{ label: "σ'(z) — the gradient", color: cssVar('--s2'), dash: true }] : [])], { pos: 'tl' });
      P.readout({
        'σ(0)': '0.5', 'max slope': fmt(.25 / s.temp, 4),
        'σ(4)': fmt(sig(4), 4),
        'saturation': `|z| > ${fmt(4 * s.temp, 1)} → gradient < ${fmt(.018 / s.temp, 4)}`,
      });
    },
    caption: 'The sigmoid squashes any real number into $(0,1)$. Its derivative $\\sigma(1-\\sigma)$ peaks at $0.25$ and **collapses toward zero at both ends** — the saturation that kills gradients in deep sigmoid networks. Lower the temperature and the transition sharpens toward a step function; raise it and everything drifts toward $0.5$. Softmax is the same construction with $K$ classes: $\\mathrm{softmax}(z)_i = e^{z_i/T}/\\sum_j e^{z_j/T}$.',
  });
};

/* ============================================================
   kNN, trees, SVM, kernels
   ============================================================ */

V['knn-boundary'] = (host) => {
  let pts = makeData('moons', 70, .18, 8);
  panel(host, {
    title: 'k-nearest neighbours: no training, all memory',
    height: 320,
    plot: { xlim: [-3.2, 3.2], ylim: [-2.4, 2.4], equal: true },
    controls: [
      { type: 'slider', key: 'k', label: 'k', min: 1, max: 31, step: 2, value: 1 },
      { type: 'select', key: 'data', label: 'dataset', value: 'moons', options: DATASETS, onChange: (v, P) => { pts = makeData(v, 70, .18, 8); } },
      { type: 'slider', key: 'noise', label: 'noise', min: .02, max: .6, step: .02, value: .18, onChange: (v, P) => { pts = makeData(P.state.data, 70, v, 8); } },
      { type: 'check', key: 'soft', label: 'show vote fraction (not just class)', value: true },
    ],
    draw(p, s, P) {
      const k = Math.round(s.k);
      const score = (x, y) => {
        const d = pts.map((q) => ({ d2: (q.x - x) ** 2 + (q.y - y) ** 2, c: q.c }));
        d.sort((a, b) => a.d2 - b.d2);
        let v = 0;
        for (let i = 0; i < Math.min(k, d.length); i++) v += d[i].c;
        const frac = v / Math.min(k, d.length);
        return s.soft ? frac : (frac > .5 ? 1 : 0);
      };
      p.clear();
      p.heat(score, {
        step: 6, lo: 0, hi: 1,
        cmap: (t) => mix(mix(cssVar('--bg-inset'), cssVar('--s1'), .45), mix(cssVar('--bg-inset'), cssVar('--s2'), .45), t),
      });
      p.axes({ nx: 5, ny: 4 });
      p.clip();
      p.contour((x, y) => score(x, y) - .5, [0], { color: cssVar('--text'), width: 2 });
      drawPoints(p, pts, { r: 4.2 });
      p.clip(false);
      // leave-one-out accuracy
      let loo = 0;
      for (let i = 0; i < pts.length; i++) {
        const d = pts.map((q, j) => ({ d2: (q.x - pts[i].x) ** 2 + (q.y - pts[i].y) ** 2, c: q.c, j })).filter((z) => z.j !== i);
        d.sort((a, b) => a.d2 - b.d2);
        let v = 0;
        for (let t = 0; t < Math.min(k, d.length); t++) v += d[t].c;
        if ((v / Math.min(k, d.length) > .5) === !!pts[i].c) loo++;
      }
      P.readout({
        'k': k,
        'training accuracy': k === 1 ? '100% (always — it memorizes)' : '—',
        'leave-one-out accuracy': fmt(loo / pts.length * 100, 1) + '%',
        'boundary': k < 5 ? 'jagged — high variance' : k > 20 ? 'smooth — high bias' : 'balanced',
      });
    },
    caption: 'At $k=1$ every training point owns a little island — training accuracy is a perfect and completely meaningless 100%. Raise $k$ and the boundary smooths as votes average out. **Leave-one-out accuracy is the honest number** and it peaks somewhere in the middle. kNN has no parameters and no training; the cost is that prediction touches the whole dataset, and in high dimensions "nearest" stops meaning anything.',
  });
};

V['decision-tree'] = (host) => {
  let pts = makeData('xor', 90, .25, 12);

  function buildTree(data, depth, maxDepth, minLeaf) {
    const n = data.length;
    const pos = data.filter((d) => d.c).length;
    const node = { p: n ? pos / n : .5, n };
    if (depth >= maxDepth || n < minLeaf * 2 || pos === 0 || pos === n) return node;
    let best = null;
    const gini = (a, b) => { const t = a + b; return t ? 1 - (a / t) ** 2 - (b / t) ** 2 : 0; };
    for (const dim of ['x', 'y']) {
      const vals = [...new Set(data.map((d) => d[dim]))].sort((a, b) => a - b);
      for (let i = 1; i < vals.length; i++) {
        const th = (vals[i - 1] + vals[i]) / 2;
        const L = data.filter((d) => d[dim] <= th), R = data.filter((d) => d[dim] > th);
        if (L.length < minLeaf || R.length < minLeaf) continue;
        const lp = L.filter((d) => d.c).length, rp = R.filter((d) => d.c).length;
        const imp = (L.length * gini(lp, L.length - lp) + R.length * gini(rp, R.length - rp)) / n;
        if (!best || imp < best.imp) best = { imp, dim, th, L, R };
      }
    }
    if (!best) return node;
    node.dim = best.dim; node.th = best.th;
    node.L = buildTree(best.L, depth + 1, maxDepth, minLeaf);
    node.R = buildTree(best.R, depth + 1, maxDepth, minLeaf);
    return node;
  }
  const predict = (t, x, y) => (t.dim === undefined ? t.p : predict((t.dim === 'x' ? x : y) <= t.th ? t.L : t.R, x, y));
  const countLeaves = (t) => (t.dim === undefined ? 1 : countLeaves(t.L) + countLeaves(t.R));

  panel(host, {
    title: 'A decision tree carves space into boxes',
    height: 320,
    plot: { xlim: [-2.6, 2.6], ylim: [-2.4, 2.4], equal: true },
    controls: [
      { type: 'slider', key: 'depth', label: 'max depth', min: 1, max: 10, step: 1, value: 3 },
      { type: 'slider', key: 'minLeaf', label: 'min samples per leaf', min: 1, max: 15, step: 1, value: 1 },
      { type: 'select', key: 'data', label: 'dataset', value: 'xor', options: DATASETS, onChange: (v, P) => { pts = makeData(v, 90, .25, 12); } },
    ],
    draw(p, s, P) {
      const T = buildTree(pts, 0, Math.round(s.depth), Math.round(s.minLeaf));
      p.clear();
      p.heat((x, y) => predict(T, x, y), {
        step: 4, lo: 0, hi: 1,
        cmap: (t) => mix(mix(cssVar('--bg-inset'), cssVar('--s1'), .45), mix(cssVar('--bg-inset'), cssVar('--s2'), .45), t),
      });
      p.axes({ nx: 5, ny: 4 });
      p.clip();
      // draw split lines
      const drawSplits = (t, x0, x1, y0, y1, d) => {
        if (t.dim === undefined) return;
        const col = alpha(cssVar('--text'), Math.max(.25, .9 - d * .12));
        if (t.dim === 'x') {
          p.line([[t.th, y0], [t.th, y1]], { color: col, width: Math.max(1, 2.6 - d * .3) });
          drawSplits(t.L, x0, t.th, y0, y1, d + 1); drawSplits(t.R, t.th, x1, y0, y1, d + 1);
        } else {
          p.line([[x0, t.th], [x1, t.th]], { color: col, width: Math.max(1, 2.6 - d * .3) });
          drawSplits(t.L, x0, x1, y0, t.th, d + 1); drawSplits(t.R, x0, x1, t.th, y1, d + 1);
        }
      };
      drawSplits(T, -3, 3, -3, 3, 0);
      drawPoints(p, pts, { r: 4 });
      p.clip(false);
      const acc = pts.filter((q) => (predict(T, q.x, q.y) > .5) === !!q.c).length / pts.length;
      P.readout({
        'depth': Math.round(s.depth), 'leaves': countLeaves(T),
        'training accuracy': fmt(acc * 100, 1) + '%',
        'boundary': 'axis-aligned only',
      });
    },
    caption: 'Every split is a single yes/no question about one feature, so the boundary is always made of **axis-aligned rectangles**. That is why XOR is trivial for a tree (two splits) and a rotated diagonal boundary is awkward (a staircase). Crank the depth: leaves multiply until each holds one point and training accuracy hits 100% — pure memorization. Trees are high-variance by nature, which is exactly why we average them (random forests) or add them slowly (boosting).',
  });
};

V['svm-margin'] = (host) => {
  let pts = makeData('blobs', 40, .05, 2);
  let w = [0, 0, 0];
  const train = (C, iters) => {
    // hinge loss + L2 via subgradient descent
    for (let it = 0; it < iters; it++) {
      const lr = .05 / (1 + it * .002);
      const g = [0, 0, 0];
      for (const q of pts) {
        const yy = q.c ? 1 : -1;
        const m = yy * (w[0] + w[1] * q.x + w[2] * q.y);
        if (m < 1) { g[0] -= yy; g[1] -= yy * q.x; g[2] -= yy * q.y; }
      }
      w[0] -= lr * (C * g[0] / pts.length);
      w[1] -= lr * (w[1] + C * g[1] / pts.length);
      w[2] -= lr * (w[2] + C * g[2] / pts.length);
    }
  };
  panel(host, {
    title: 'Support vector machine: the widest possible street',
    height: 320,
    plot: { xlim: [-3.4, 3.4], ylim: [-2.6, 2.6], equal: true },
    controls: [
      { type: 'slider', key: 'C', label: 'C (log₁₀) — tolerance for violations', min: -1, max: 3, step: .1, value: 1, fmt: (v) => (10 ** v).toFixed(1) },
      { type: 'button', label: 'train', primary: true, onClick: (s) => train(10 ** s.C, 4000) },
      { type: 'button', label: '↺ new data', onClick: (s) => { pts = makeData('blobs', 40, .05 + Math.random() * .3, Math.floor(Math.random() * 99)); w = [0, 0, 0]; } },
      { type: 'label', label: 'press train after changing C' },
    ],
    init: (P, s) => train(10 ** s.C, 4000),
    draw(p, s, P) {
      p.clear().axes({ nx: 5, ny: 4 });
      p.clip();
      const f = (x, y) => w[0] + w[1] * x + w[2] * y;
      const nw = Math.hypot(w[1], w[2]) || 1e-9;
      // margin band
      const g = p.ctx;
      g.save();
      g.beginPath();
      const band = [];
      for (const lvl of [1, -1]) {
        p.contour((x, y) => f(x, y) - lvl, [0], { color: alpha(cssVar('--s3'), .8), width: 1.5 });
      }
      g.restore();
      p.contour(f, [0], { color: cssVar('--text'), width: 2.6 });
      // support vectors
      const sv = pts.filter((q) => (q.c ? 1 : -1) * f(q.x, q.y) <= 1.02);
      p.points(sv.map((q) => [q.x, q.y]), { r: 8, color: 'transparent', stroke: cssVar('--s5'), strokeWidth: 2.2 });
      drawPoints(p, pts, { r: 4.5 });
      p.clip(false);
      p.legend([
        { label: 'decision boundary', color: cssVar('--text') },
        { label: 'margins (f = ±1)', color: cssVar('--s3') },
        { label: 'support vectors', color: cssVar('--s5'), shape: 'dot' },
      ], { pos: 'tl' });
      const acc = pts.filter((q) => (f(q.x, q.y) > 0) === !!q.c).length / pts.length;
      P.readout({
        'margin width 2/‖w‖': fmt(2 / nw, 3),
        '‖w‖': fmt(nw, 3),
        'support vectors': `${sv.length} / ${pts.length}`,
        'accuracy': fmt(acc * 100, 1) + '%',
        'C': fmt(10 ** s.C, 1),
      });
    },
    caption: 'The SVM does not just separate the classes — it separates them by the **widest street** it can, because maximizing the margin $2/\\|w\\|$ is what generalizes. Only the circled points touch the margin; delete any other point and the solution is unchanged. That is what "support vector" means. Small $C$ = a wide street that tolerates violations (more regularization); large $C$ = a narrow street that insists on getting every point right.',
  });
};

V['kernel-trick'] = (host) => {
  panel(host, {
    title: 'The kernel trick: separability is a matter of dimension',
    height: 300,
    plot: { xlim: [-2.6, 2.6], ylim: [-2.4, 2.4], equal: true },
    controls: [
      { type: 'slider', key: 'lift', label: 'lift into 3rd dimension (z = x² + y²)', min: 0, max: 1, step: .01, value: 0 },
      { type: 'slider', key: 'tilt', label: 'viewing angle', min: 0, max: 1.2, step: .01, value: .55 },
    ],
    draw(p, s, P) {
      const r = rng(4);
      const pts = [];
      for (let i = 0; i < 90; i++) {
        const c = i % 2;
        const t = r() * 6.2832, rad = (c ? 1.85 : .7) + r.normal(0, .16);
        pts.push({ x: Math.cos(t) * rad, y: Math.sin(t) * rad, c });
      }
      p.clear().axes({ nx: 5, ny: 4 });
      p.clip();
      const a = s.tilt * s.lift;
      const proj = (x, y) => {
        const z = (x * x + y * y) * .42;
        return [x, y * Math.cos(a) - z * s.lift * Math.sin(a) * 1.4 + z * 0];
      };
      // separating plane in the lifted space appears as a line
      if (s.lift > .05) {
        const zc = 1.55 * .42;
        const seg = [];
        for (let x = -2.6; x <= 2.6; x += .05) {
          const yy = -zc * s.lift * Math.sin(a) * 1.4 / Math.cos(a);
          seg.push([x, yy]);
        }
        p.line(seg, { color: alpha(cssVar('--s3'), .9), width: 2.4 });
      }
      const pp = pts.map((q) => proj(q.x, q.y));
      p.points(pp, { r: 4.2, color: (_, i) => (pts[i].c ? cssVar('--s2') : cssVar('--s1')), stroke: alpha(cssVar('--bg-inset'), .8), strokeWidth: 1.2 });
      if (s.lift < .05) p.contour((x, y) => x * x + y * y - 1.55 ** 2, [0], { color: alpha(cssVar('--s3'), .9), width: 2.2 });
      p.clip(false);
      P.readout({
        'input space': 'not linearly separable',
        'feature map φ(x) = (x₁, x₂, x₁²+x₂²)': 'linearly separable ✓',
        'kernel': 'k(x,x′) = (1 + xᵀx′)² computes the dot product without ever forming φ',
      });
    },
    caption: 'Slide the lift. In the plane no line separates the rings; raise each point by $x_1^2+x_2^2$ and a **flat plane** slices cleanly between them. The trick is that SVM training and prediction only ever need *inner products* $\\phi(x)^{\\mathsf T}\\phi(x\')$ — and a kernel computes those directly, so you get the benefits of a high- (even infinite-) dimensional space without ever visiting it. The RBF kernel corresponds to an infinite-dimensional $\\phi$.',
  });
};

/* ============================================================
   Unsupervised
   ============================================================ */

V['kmeans'] = (host) => {
  let data = [], cents = [], assign = [], iter = 0, converged = false;
  const init = (k, seed) => {
    const r = rng(seed);
    data = [];
    const centers = [[-1.5, -1], [1.6, .9], [-.2, 1.7], [1.4, -1.5], [-2.2, 1.2]];
    for (let i = 0; i < 180; i++) {
      const c = centers[i % 4];
      data.push([c[0] + r.normal(0, .48), c[1] + r.normal(0, .48)]);
    }
    cents = Array.from({ length: k }, () => [(r() - .5) * 5, (r() - .5) * 4]);
    assign = new Array(data.length).fill(0);
    iter = 0; converged = false;
  };
  init(4, 3);

  const stepE = () => {
    let changed = false;
    assign = data.map((d, i) => {
      let b = 0, bd = Infinity;
      cents.forEach((c, k) => { const dd = (d[0] - c[0]) ** 2 + (d[1] - c[1]) ** 2; if (dd < bd) { bd = dd; b = k; } });
      if (assign[i] !== b) changed = true;
      return b;
    });
    return changed;
  };
  const stepM = () => {
    cents = cents.map((c, k) => {
      const mem = data.filter((_, i) => assign[i] === k);
      if (!mem.length) return c;
      return [mem.reduce((a, d) => a + d[0], 0) / mem.length, mem.reduce((a, d) => a + d[1], 0) / mem.length];
    });
  };

  panel(host, {
    title: 'k-means: alternate between assigning and averaging',
    height: 320,
    plot: { xlim: [-3.4, 3.4], ylim: [-2.6, 2.6], equal: true },
    controls: [
      { type: 'slider', key: 'k', label: 'k (clusters)', min: 1, max: 8, step: 1, value: 4, onChange: (v, P) => init(Math.round(v), P.state.seed || 3) },
      { type: 'check', key: 'cells', label: 'show Voronoi cells', value: true },
      { type: 'play' },
      { type: 'button', label: 'one step', onClick: () => { const ch = stepE(); stepM(); iter++; converged = !ch; } },
      { type: 'button', label: '↻ new init (watch it change!)', onClick: (s) => { s.seed = Math.floor(Math.random() * 999); init(Math.round(s.k), s.seed); } },
    ],
    animate(s, P) {
      if (converged) { P.play(false); return; }
      const ch = stepE(); stepM(); iter++;
      if (!ch) converged = true;
    },
    draw(p, s, P) {
      const cols = SERIES();
      p.clear();
      if (s.cells) {
        p.heat((x, y) => {
          let b = 0, bd = Infinity;
          cents.forEach((c, k) => { const dd = (x - c[0]) ** 2 + (y - c[1]) ** 2; if (dd < bd) { bd = dd; b = k; } });
          return b;
        }, { step: 6, lo: 0, hi: Math.max(cents.length - 1, 1), cmap: (t) => alpha(cols[Math.round(t * (cents.length - 1)) % cols.length], .16) });
      }
      p.axes({ nx: 5, ny: 4 });
      p.clip();
      p.points(data, { r: 3.4, color: (_, i) => alpha(cols[assign[i] % cols.length], .85) });
      cents.forEach((c, k) => {
        p.points([c], { r: 9, color: cols[k % cols.length], shape: 'cross', width: 3 });
        p.circle(c[0], c[1], 9, { stroke: cssVar('--bg-inset'), width: 1 });
      });
      p.clip(false);
      let inertia = 0;
      data.forEach((d, i) => { const c = cents[assign[i]]; inertia += (d[0] - c[0]) ** 2 + (d[1] - c[1]) ** 2; });
      P.readout({
        'iteration': iter, 'k': cents.length,
        'inertia (within-cluster SS)': fmt(inertia, 2),
        'status': converged ? 'converged ✓ (a local optimum)' : 'running',
      });
    },
    caption: 'Two steps, forever: **assign** each point to its nearest centre, then **move** each centre to the mean of its points. Inertia can only decrease, so it always converges — but usually to a *local* optimum. Press "new init" a few times on the same data and watch the final clustering change. That sensitivity is why k-means++ initialization and multiple restarts exist. Note the boundaries are always straight lines (a Voronoi diagram): k-means cannot find elongated or nested clusters.',
  });
};

V['gmm-em'] = (host) => {
  let data = [], comps = [], iter = 0, resp = [];
  const init = (K, seed) => {
    const r = rng(seed);
    data = [];
    const specs = [[[-1.2, -.6], [[.7, .35], [.35, .3]]], [[1.4, .8], [[.35, -.25], [-.25, .8]]], [[-.3, 1.6], [[.25, 0], [0, .25]]]];
    for (let i = 0; i < 220; i++) {
      const [m, C] = specs[i % 3];
      const L = [[Math.sqrt(C[0][0]), 0], [C[0][1] / Math.sqrt(C[0][0]), Math.sqrt(Math.max(C[1][1] - C[0][1] ** 2 / C[0][0], 1e-6))]];
      const z = [r.normal(), r.normal()];
      data.push([m[0] + L[0][0] * z[0], m[1] + L[1][0] * z[0] + L[1][1] * z[1]]);
    }
    comps = Array.from({ length: K }, (_, k) => ({
      mu: [(r() - .5) * 4, (r() - .5) * 3.4],
      S: [[.6, 0], [0, .6]],
      pi: 1 / K,
    }));
    iter = 0;
  };
  init(3, 6);

  const gauss = (x, mu, S) => {
    const det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    if (det <= 1e-9) return 1e-12;
    const inv = [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]];
    const d = [x[0] - mu[0], x[1] - mu[1]];
    const q = d[0] * (inv[0][0] * d[0] + inv[0][1] * d[1]) + d[1] * (inv[1][0] * d[0] + inv[1][1] * d[1]);
    return Math.exp(-.5 * q) / (2 * Math.PI * Math.sqrt(det));
  };

  const em = () => {
    // E
    resp = data.map((x) => {
      const w = comps.map((c) => c.pi * gauss(x, c.mu, c.S));
      const z = w.reduce((a, b) => a + b, 0) || 1e-12;
      return w.map((v) => v / z);
    });
    // M
    comps.forEach((c, k) => {
      const Nk = resp.reduce((a, r) => a + r[k], 0) + 1e-9;
      c.pi = Nk / data.length;
      c.mu = [resp.reduce((a, r, i) => a + r[k] * data[i][0], 0) / Nk, resp.reduce((a, r, i) => a + r[k] * data[i][1], 0) / Nk];
      let s00 = 0, s01 = 0, s11 = 0;
      data.forEach((x, i) => {
        const d0 = x[0] - c.mu[0], d1 = x[1] - c.mu[1], w = resp[i][k];
        s00 += w * d0 * d0; s01 += w * d0 * d1; s11 += w * d1 * d1;
      });
      c.S = [[s00 / Nk + 1e-4, s01 / Nk], [s01 / Nk, s11 / Nk + 1e-4]];
    });
    iter++;
  };

  panel(host, {
    title: 'Gaussian mixtures + EM: soft clustering with shape',
    height: 320,
    plot: { xlim: [-3.4, 3.4], ylim: [-2.6, 2.6], equal: true },
    controls: [
      { type: 'slider', key: 'K', label: 'components K', min: 1, max: 6, step: 1, value: 3, onChange: (v, P) => init(Math.round(v), P.state.seed || 6) },
      { type: 'play' },
      { type: 'button', label: 'one EM step', onClick: () => em() },
      { type: 'button', label: '↻ new init', onClick: (s) => { s.seed = Math.floor(Math.random() * 999); init(Math.round(s.K), s.seed); } },
    ],
    animate() { em(); },
    draw(p, s, P) {
      const cols = SERIES();
      p.clear().axes({ nx: 5, ny: 4 });
      p.clip();
      p.points(data, { r: 3.2, color: (_, i) => {
        if (!resp.length) return alpha(cssVar('--text-faint'), .6);
        // blend component colors by responsibility
        let best = 0, bv = 0;
        resp[i].forEach((v, k) => { if (v > bv) { bv = v; best = k; } });
        return alpha(cols[best % cols.length], .35 + bv * .55);
      } });
      comps.forEach((c, k) => {
        for (const nsig of [1, 2]) {
          p.ellipse(c.mu[0], c.mu[1], c.S, { n: nsig, stroke: alpha(cols[k % cols.length], nsig === 1 ? .95 : .45), width: nsig === 1 ? 2.2 : 1.2 });
        }
        p.points([c.mu], { r: 5, color: cols[k % cols.length], shape: 'cross', width: 2.5 });
      });
      p.clip(false);
      let ll = 0;
      for (const x of data) ll += Math.log(Math.max(comps.reduce((a, c) => a + c.pi * gauss(x, c.mu, c.S), 0), 1e-300));
      P.readout({
        'EM iteration': iter,
        'log-likelihood': fmt(ll, 2),
        'mixing weights π': comps.map((c) => fmt(c.pi, 2)).join(', '),
      });
    },
    caption: 'k-means is GMM\'s hard-assignment cousin. **E-step**: compute each point\'s *responsibility* — the posterior probability it came from each component. **M-step**: refit each Gaussian using those responsibilities as weights. Log-likelihood increases monotonically (that is the theorem EM rests on). Unlike k-means, components can be elongated and tilted, and a point near a boundary genuinely belongs partly to both.',
  });
};

V['pca'] = (host) => {
  let data = [];
  const gen = (n, corr, seed) => {
    const r = rng(seed);
    data = [];
    for (let i = 0; i < n; i++) {
      const a = r.normal(0, 1.5), b = r.normal(0, .55);
      const th = .55;
      data.push([a * Math.cos(th) - b * Math.sin(th) * (1 - corr + .2), a * Math.sin(th) * corr + b * Math.cos(th)]);
    }
  };
  gen(140, 1, 5);

  panel(host, {
    title: 'PCA finds the directions of greatest variance',
    height: 330,
    plot: { xlim: [-4, 4], ylim: [-3, 3], equal: true, xlabel: 'x₁', ylabel: 'x₂' },
    controls: [
      { type: 'slider', key: 'ang', label: 'your projection direction (°)', min: 0, max: 180, step: 1, value: 20 },
      { type: 'check', key: 'showPC', label: 'show principal components', value: true },
      { type: 'check', key: 'showProj', label: 'show projections + reconstruction error', value: true },
      { type: 'button', label: '↻ new data', onClick: () => gen(140, 1, Math.floor(Math.random() * 999)) },
    ],
    draw(p, s, P) {
      const n = data.length;
      const mx = data.reduce((a, d) => a + d[0], 0) / n, my = data.reduce((a, d) => a + d[1], 0) / n;
      let c00 = 0, c01 = 0, c11 = 0;
      for (const d of data) { const a = d[0] - mx, b = d[1] - my; c00 += a * a; c01 += a * b; c11 += b * b; }
      c00 /= n; c01 /= n; c11 /= n;
      const { vals, vecs } = LA.eig2([[c00, c01], [c01, c11]]);
      const th = s.ang * Math.PI / 180;
      const u = [Math.cos(th), Math.sin(th)];

      p.clear().axes({ nx: 5, ny: 4 });
      p.clip();
      if (s.showProj) {
        for (const d of data) {
          const t = (d[0] - mx) * u[0] + (d[1] - my) * u[1];
          const pr = [mx + u[0] * t, my + u[1] * t];
          p.line([d, pr], { color: alpha(cssVar('--s2'), .3), width: 1 });
        }
      }
      p.points(data, { r: 3.2, color: alpha(cssVar('--s1'), .75) });
      // user direction
      p.line([[mx - u[0] * 5, my - u[1] * 5], [mx + u[0] * 5, my + u[1] * 5]], { color: cssVar('--s2'), width: 2.2 });
      // projected points on the user line
      if (s.showProj) {
        p.points(data.map((d) => {
          const t = (d[0] - mx) * u[0] + (d[1] - my) * u[1];
          return [mx + u[0] * t, my + u[1] * t];
        }), { r: 2.4, color: cssVar('--s2') });
      }
      if (s.showPC) {
        vals.forEach((l, i) => {
          const v = vecs[i], sc = Math.sqrt(Math.max(l, 0)) * 2;
          p.arrow(mx, my, mx + v[0] * sc, my + v[1] * sc, { color: i ? cssVar('--s6') : cssVar('--s3'), width: 3 });
          p.text(mx + v[0] * sc, my + v[1] * sc, ` PC${i + 1}`, { color: i ? cssVar('--s6') : cssVar('--s3'), size: 11, weight: '700' });
        });
      }
      p.clip(false);
      // variance along u & reconstruction error
      let varU = 0, err = 0;
      for (const d of data) {
        const a = d[0] - mx, b = d[1] - my;
        const t = a * u[0] + b * u[1];
        varU += t * t;
        err += (a - u[0] * t) ** 2 + (b - u[1] * t) ** 2;
      }
      varU /= n; err /= n;
      P.readout({
        'variance captured by your line': fmt(varU, 4),
        'max possible (λ₁)': fmt(vals[0], 4),
        'reconstruction MSE': fmt(err, 4),
        'explained variance ratio': fmt(varU / (vals[0] + vals[1]) * 100, 1) + '%',
        'PC1 angle': fmt(Math.atan2(vecs[0][1], vecs[0][0]) * 180 / Math.PI, 1) + '°',
      });
    },
    caption: 'Rotate your line and watch two numbers move in opposite directions: **variance captured** goes up exactly as **reconstruction error** goes down. They are the same objective. The maximum sits precisely on PC1 — the top eigenvector of the covariance matrix. That equivalence ("maximize spread" = "minimize squared distance to the subspace") is why PCA shows up as both a compression method and an analysis tool.',
  });
};

/* ---------- evaluation: ROC / PR / calibration ---------- */

V['roc-curve'] = (host) => {
  panel(host, {
    title: 'Thresholds, ROC, and the base-rate trap',
    height: 320,
    plot: { xlim: [0, 1], ylim: [0, 1], xlabel: 'false positive rate', ylabel: 'true positive rate' },
    controls: [
      { type: 'slider', key: 'sep', label: 'class separation (model skill)', min: 0, max: 4, step: .05, value: 1.6 },
      { type: 'slider', key: 'thr', label: 'decision threshold', min: -4, max: 6, step: .05, value: 1 },
      { type: 'slider', key: 'prev', label: 'positive class prevalence', min: .01, max: .5, step: .005, value: .3 },
      { type: 'select', key: 'view', label: 'view', value: 'roc', options: [{ value: 'roc', label: 'ROC curve' }, { value: 'pr', label: 'precision-recall' }, { value: 'dist', label: 'score distributions' }] },
    ],
    draw(p, s, P) {
      const N = 4000;
      const nPos = Math.round(N * s.prev), nNeg = N - nPos;
      const r = rng(2);
      const phi = (z) => Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI);
      const Phi = (z) => { // normal cdf via erf approx
        const t = 1 / (1 + .2316419 * Math.abs(z));
        const d = phi(z);
        let pr = d * t * (.319381530 + t * (-.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        return z > 0 ? 1 - pr : pr;
      };
      const tpr = (th) => 1 - Phi(th - s.sep);
      const fpr = (th) => 1 - Phi(th);

      if (s.view === 'dist') {
        p.setLim([-4, 6], [0, .55]);
        p.clear().axes();
        p.clip();
        const a = [], b = [];
        for (let x = -4; x <= 6; x += .02) { a.push([x, phi(x) * (1 - s.prev)]); b.push([x, phi(x - s.sep) * s.prev]); }
        p.area(a, { color: alpha(cssVar('--s1'), .25) }); p.line(a, { color: cssVar('--s1'), width: 2 });
        p.area(b, { color: alpha(cssVar('--s2'), .25) }); p.line(b, { color: cssVar('--s2'), width: 2 });
        p.line([[s.thr, 0], [s.thr, 1]], { color: cssVar('--text'), width: 2 });
        p.clip(false);
        p.legend([{ label: 'negatives', color: cssVar('--s1') }, { label: 'positives', color: cssVar('--s2') }, { label: 'threshold', color: cssVar('--text') }], { pos: 'tr' });
        p.xlabel = 'model score';
      } else if (s.view === 'roc') {
        p.setLim([0, 1], [0, 1]);
        p.clear().axes();
        p.clip();
        const curve = [];
        for (let th = 8; th >= -6; th -= .02) curve.push([fpr(th), tpr(th)]);
        p.line([[0, 0], [1, 1]], { color: alpha(cssVar('--text-faint'), .6), width: 1.4, dash: [4, 4] });
        p.area(curve, { color: alpha(cssVar('--s1'), .12), base: 0 });
        p.line(curve, { color: cssVar('--s1'), width: 2.6 });
        p.points([[fpr(s.thr), tpr(s.thr)]], { r: 6, color: cssVar('--s2'), stroke: cssVar('--bg-inset'), strokeWidth: 2 });
        p.clip(false);
      } else {
        p.setLim([0, 1], [0, 1]);
        p.clear().axes();
        p.clip();
        const curve = [];
        for (let th = 8; th >= -6; th -= .02) {
          const tp = tpr(th) * s.prev, fp = fpr(th) * (1 - s.prev);
          const prec = tp + fp > 1e-9 ? tp / (tp + fp) : 1;
          curve.push([tpr(th), prec]);
        }
        p.line([[0, s.prev], [1, s.prev]], { color: alpha(cssVar('--text-faint'), .6), width: 1.4, dash: [4, 4] });
        p.line(curve, { color: cssVar('--s3'), width: 2.6 });
        const tp = tpr(s.thr) * s.prev, fp = fpr(s.thr) * (1 - s.prev);
        p.points([[tpr(s.thr), tp + fp > 1e-9 ? tp / (tp + fp) : 1]], { r: 6, color: cssVar('--s2'), stroke: cssVar('--bg-inset'), strokeWidth: 2 });
        p.clip(false);
        p.xlabel = 'recall'; p.ylabel = 'precision';
      }
      // metrics at threshold
      const TP = tpr(s.thr) * nPos, FN = nPos - TP;
      const FP = fpr(s.thr) * nNeg, TN = nNeg - FP;
      const prec = TP + FP > 1e-9 ? TP / (TP + FP) : 0;
      const rec = TP / Math.max(nPos, 1e-9);
      // AUC = Phi(sep/sqrt2)
      P.readout({
        'AUC': fmt(Phi(s.sep / Math.SQRT2), 4),
        'accuracy': fmt((TP + TN) / N * 100, 1) + '%',
        'precision': fmt(prec * 100, 1) + '%',
        'recall (TPR)': fmt(rec * 100, 1) + '%',
        'F1': fmt(2 * prec * rec / Math.max(prec + rec, 1e-9), 3),
        'always-predict-negative accuracy': fmt((1 - s.prev) * 100, 1) + '%',
      });
    },
    caption: 'A model has **one** ROC curve but infinitely many operating points — moving the threshold slides you along it, trading recall for precision. Now drop prevalence to 2%: AUC and accuracy barely flinch while **precision collapses**, because the few true positives get swamped by false ones from a much larger negative pool. This is why ROC flatters rare-event models and precision-recall does not.',
  });
};

/* ---------- cross-validation ---------- */

V['cross-validation'] = (host) => {
  panel(host, {
    title: 'k-fold cross-validation',
    height: 250,
    noCanvas: false,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'k', label: 'folds k', min: 2, max: 10, step: 1, value: 5 },
      { type: 'slider', key: 'n', label: 'dataset size', min: 10, max: 60, step: 1, value: 30 },
      { type: 'check', key: 'holdout', label: 'reserve a final test set', value: true },
    ],
    draw(p, s, P) {
      const k = Math.round(s.k), n = Math.round(s.n);
      p.clear();
      const g = p.ctx;
      const W = p.w - 110, x0 = 96;
      const nTest = s.holdout ? Math.round(n * .2) : 0;
      const nCV = n - nTest;
      const rowH = Math.min(26, (p.h - 46) / (k + 1));
      g.font = '11px -apple-system, sans-serif';
      g.textBaseline = 'middle';
      for (let f = 0; f < k; f++) {
        const y = 24 + f * rowH;
        g.fillStyle = cssVar('--text-dim');
        g.textAlign = 'right';
        g.fillText(`fold ${f + 1}`, x0 - 10, y + rowH / 2 - 4);
        const cw = (W * (nCV / n)) / nCV;
        for (let i = 0; i < nCV; i++) {
          const isVal = Math.floor(i * k / nCV) === f;
          g.fillStyle = isVal ? cssVar('--s2') : alpha(cssVar('--s1'), .55);
          g.fillRect(x0 + i * cw, y, Math.max(cw - 1, 1), rowH - 8);
        }
        if (nTest) {
          g.fillStyle = alpha(cssVar('--s6'), .35);
          g.fillRect(x0 + W * (nCV / n) + 6, y, W * (nTest / n) - 6, rowH - 8);
        }
      }
      const yl = 24 + k * rowH + 8;
      g.textAlign = 'left';
      const legend = [['train', alpha(cssVar('--s1'), .55)], ['validate', cssVar('--s2')], ...(nTest ? [['held-out test (touch once)', alpha(cssVar('--s6'), .35)]] : [])];
      let lx = x0;
      for (const [lab, col] of legend) {
        g.fillStyle = col; g.fillRect(lx, yl, 14, 10);
        g.fillStyle = cssVar('--text-dim'); g.fillText(lab, lx + 19, yl + 5);
        lx += 24 + g.measureText(lab).width + 14;
      }
      P.readout({
        'train size per fold': Math.round(nCV * (k - 1) / k),
        'validation size per fold': Math.round(nCV / k),
        'models trained': k,
        'test set': nTest || 'none — you will overfit your model selection',
      });
    },
    caption: 'Every point gets to be validation data exactly once, so you use all $n$ points for evaluation while never testing on what you trained on. Larger $k$ means more training data per fold (less pessimistic bias) but $k$ times the compute and more correlated estimates; $k=5$ or $10$ is the usual compromise. **The held-out test set is not part of CV** — if you tune hyperparameters against it, it stops being a test set.',
  });
};

/* ---------- ensembles / boosting ---------- */

V['boosting'] = (host) => {
  const truth = (x) => Math.sin(x * 1.6) * 1.2 + .3 * x;
  const r = rng(14);
  const data = Array.from({ length: 40 }, () => { const x = (r() - .5) * 5; return [x, truth(x) + r.normal(0, .25)]; });

  // depth-1 regression stump
  const fitStump = (xs, res) => {
    let best = null;
    const cand = [...xs].sort((a, b) => a - b);
    for (let i = 1; i < cand.length; i++) {
      const th = (cand[i - 1] + cand[i]) / 2;
      const L = [], R = [];
      xs.forEach((x, j) => (x <= th ? L : R).push(res[j]));
      if (!L.length || !R.length) continue;
      const lm = L.reduce((a, b) => a + b, 0) / L.length, rm = R.reduce((a, b) => a + b, 0) / R.length;
      let e = 0;
      xs.forEach((x, j) => { e += (res[j] - (x <= th ? lm : rm)) ** 2; });
      if (!best || e < best.e) best = { e, th, lm, rm };
    }
    return best || { th: 0, lm: 0, rm: 0 };
  };

  panel(host, {
    title: 'Gradient boosting: a committee of stumps, each fixing the last one\'s mistakes',
    height: 320,
    plot: { xlim: [-2.8, 2.8], ylim: [-3, 3], xlabel: 'x', ylabel: 'y' },
    controls: [
      { type: 'slider', key: 'M', label: 'number of trees', min: 0, max: 120, step: 1, value: 1 },
      { type: 'slider', key: 'lr', label: 'learning rate (shrinkage)', min: .02, max: 1, step: .02, value: .3 },
      { type: 'check', key: 'resid', label: 'show current residuals', value: true },
    ],
    draw(p, s, P) {
      const xs = data.map((d) => d[0]), ys = data.map((d) => d[1]);
      const M = Math.round(s.M);
      const base = ys.reduce((a, b) => a + b, 0) / ys.length;
      let pred = xs.map(() => base);
      const stumps = [];
      for (let m = 0; m < M; m++) {
        const res = ys.map((y, i) => y - pred[i]);
        const st = fitStump(xs, res);
        stumps.push(st);
        pred = pred.map((v, i) => v + s.lr * (xs[i] <= st.th ? st.lm : st.rm));
      }
      const F = (x) => base + stumps.reduce((a, st) => a + s.lr * (x <= st.th ? st.lm : st.rm), 0);
      p.clear().axes();
      p.clip();
      p.fn(truth, { color: alpha(cssVar('--s3'), .8), width: 2, dash: [6, 4] });
      p.fn(F, { color: cssVar('--s2'), width: 2.6, n: 600 });
      if (s.resid) data.forEach((d, i) => p.line([[d[0], d[1]], [d[0], F(d[0])]], { color: alpha(cssVar('--s4'), .45), width: 1 }));
      p.points(data, { r: 3.8, color: cssVar('--s1'), stroke: cssVar('--bg-inset'), strokeWidth: 1.2 });
      p.clip(false);
      p.legend([{ label: 'truth', color: cssVar('--s3'), dash: true }, { label: `boosted (${M} stumps)`, color: cssVar('--s2') }], { pos: 'tl' });
      const mse = data.reduce((a, d) => a + (F(d[0]) - d[1]) ** 2, 0) / data.length;
      P.readout({ 'trees': M, 'train MSE': fmt(mse, 4), 'effective step': fmt(s.lr, 2), 'note': M > 60 && s.lr > .6 ? 'fitting noise — overfitting ✗' : 'ok' });
    },
    caption: 'Each new stump is fit to the **residuals** — the part of the target the committee still gets wrong — which is exactly gradient descent in function space. The learning rate shrinks each contribution so no single tree dominates; small rate + many trees generalizes better than the reverse. Push trees to 120 with rate 1.0 and watch the curve start chasing individual noise points.',
  });
};

/* ---------- naive bayes / generative vs discriminative ---------- */

V['generative-discriminative'] = (host) => {
  panel(host, {
    title: 'Generative vs discriminative: two ways to draw the same boundary',
    height: 320,
    plot: { xlim: [-4, 4], ylim: [-3, 3], equal: true },
    controls: [
      { type: 'slider', key: 'n', label: 'training points', min: 6, max: 200, step: 2, value: 20 },
      { type: 'check', key: 'gauss', label: 'show fitted Gaussians (generative)', value: true },
      { type: 'slider', key: 'seed', label: 'resample', min: 1, max: 60, step: 1, value: 5 },
    ],
    draw(p, s, P) {
      const r = rng(Math.round(s.seed));
      const n = Math.round(s.n);
      const pts = [];
      for (let i = 0; i < n; i++) {
        const c = i % 2;
        pts.push({ x: (c ? 1.3 : -1.3) + r.normal(0, .9), y: (c ? .7 : -.7) + r.normal(0, .9), c });
      }
      // generative: class-conditional Gaussians with shared covariance (LDA)
      const grp = [pts.filter((q) => !q.c), pts.filter((q) => q.c)];
      const mus = grp.map((G) => [G.reduce((a, q) => a + q.x, 0) / Math.max(G.length, 1), G.reduce((a, q) => a + q.y, 0) / Math.max(G.length, 1)]);
      let S = [[0, 0], [0, 0]];
      grp.forEach((G, k) => G.forEach((q) => {
        const d = [q.x - mus[k][0], q.y - mus[k][1]];
        S[0][0] += d[0] * d[0]; S[0][1] += d[0] * d[1]; S[1][0] += d[0] * d[1]; S[1][1] += d[1] * d[1];
      }));
      S = S.map((row) => row.map((v) => v / Math.max(n, 1) + 1e-3));
      const det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
      const inv = [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]];
      const dmu = [mus[1][0] - mus[0][0], mus[1][1] - mus[0][1]];
      const wG = [inv[0][0] * dmu[0] + inv[0][1] * dmu[1], inv[1][0] * dmu[0] + inv[1][1] * dmu[1]];
      const mid = [(mus[0][0] + mus[1][0]) / 2, (mus[0][1] + mus[1][1]) / 2];
      const bG = -(wG[0] * mid[0] + wG[1] * mid[1]);

      // discriminative: logistic regression
      let w = [0, 0, 0];
      for (let it = 0; it < 900; it++) {
        const g = [0, 0, 0];
        for (const q of pts) {
          const pr = 1 / (1 + Math.exp(-(w[0] + w[1] * q.x + w[2] * q.y)));
          const e = pr - q.c;
          g[0] += e; g[1] += e * q.x; g[2] += e * q.y;
        }
        for (let k = 0; k < 3; k++) w[k] -= .35 * (g[k] / n + (k ? 1e-3 * w[k] : 0));
      }
      p.clear().axes({ nx: 5, ny: 4 });
      p.clip();
      if (s.gauss) grp.forEach((G, k) => {
        for (const ns of [1, 2]) p.ellipse(mus[k][0], mus[k][1], S, { n: ns, stroke: alpha(k ? cssVar('--s2') : cssVar('--s1'), ns === 1 ? .8 : .35), width: ns === 1 ? 1.8 : 1 });
      });
      p.contour((x, y) => bG + wG[0] * x + wG[1] * y, [0], { color: cssVar('--s3'), width: 2.4 });
      p.contour((x, y) => w[0] + w[1] * x + w[2] * y, [0], { color: cssVar('--s5'), width: 2.4 });
      drawPoints(p, pts, { r: 4.4 });
      p.clip(false);
      p.legend([
        { label: 'generative (LDA)', color: cssVar('--s3') },
        { label: 'discriminative (logistic)', color: cssVar('--s5') },
      ], { pos: 'tl' });
      P.readout({
        'n': n,
        'generative': 'models p(x|y) and p(y), then flips it with Bayes',
        'discriminative': 'models p(y|x) directly — no distribution over x',
      });
    },
    caption: 'Both draw a line; they get there differently. **Generative** (LDA/naive Bayes) fits a distribution to each class and lets Bayes rule produce the boundary — this uses strong assumptions, so with very few points (drag n to 6) it is often *better*. **Discriminative** (logistic regression) optimizes the boundary directly and wins as data grows, because it never wastes capacity modeling things you did not ask about. Resample repeatedly at small n to see the variance difference.',
  });
};

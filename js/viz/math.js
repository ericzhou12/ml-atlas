/* ============================================================
   viz/math.js — figures for the mathematics track.
   ============================================================ */

import { panel } from '../ui.js';
import { cssVar, alpha, mix, seqMap, divergeMap, LA, rng, fmt, SERIES } from '../plot.js';

const V = {};
export default V;

/* ---------- vectors: add, scale, dot, projection ---------- */

V['vector-playground'] = (host) => {
  let a = [2.2, 1.4], b = [-1.1, 2.0];
  panel(host, {
    title: 'Vectors — drag the arrow tips',
    height: 330,
    plot: { xlim: [-4, 4], ylim: [-3, 3], equal: true, xlabel: 'x₁', ylabel: 'x₂' },
    controls: [
      { type: 'check', key: 'sum', label: 'a + b', value: true },
      { type: 'check', key: 'proj', label: 'projection of a onto b', value: false },
      { type: 'check', key: 'perp', label: 'residual (a − proj)', value: false },
    ],
    interact: (s, P) => ({
      down(x, y) {
        const da = (x - a[0]) ** 2 + (y - a[1]) ** 2, db = (x - b[0]) ** 2 + (y - b[1]) ** 2;
        P._drag = Math.min(da, db) < 0.25 ? (da < db ? 'a' : 'b') : null;
        return !!P._drag;
      },
      move(x, y, e, cx, cy) {
        const v = [Math.round(x * 20) / 20, Math.round(y * 20) / 20];
        if (P._drag === 'a') a = v; else if (P._drag === 'b') b = v;
        P.redraw();
      },
    }),
    draw(p, s, P) {
      const A = cssVar('--s1'), B = cssVar('--s2'), C = cssVar('--s3'), D = cssVar('--s4');
      p.clear().axes();
      if (s.sum) {
        p.arrow(a[0], a[1], a[0] + b[0], a[1] + b[1], { color: alpha(B, .35), dash: [4, 3], width: 1.5 });
        p.arrow(b[0], b[1], a[0] + b[0], a[1] + b[1], { color: alpha(A, .35), dash: [4, 3], width: 1.5 });
        p.arrow(0, 0, a[0] + b[0], a[1] + b[1], { color: C, width: 2.5 });
        p.text(a[0] + b[0], a[1] + b[1], ' a+b', { color: C, size: 12, weight: '600' });
      }
      const dot = LA.dot(a, b), nb2 = LA.dot(b, b);
      if (s.proj || s.perp) {
        const k = nb2 > 1e-9 ? dot / nb2 : 0;
        const pr = [b[0] * k, b[1] * k];
        if (s.proj) {
          p.arrow(0, 0, pr[0], pr[1], { color: D, width: 3 });
          p.text(pr[0], pr[1], ' proj', { color: D, size: 11 });
        }
        if (s.perp) p.line([pr, a], { color: alpha(D, .7), dash: [3, 3], width: 1.5 });
      }
      p.arrow(0, 0, a[0], a[1], { color: A, width: 3 });
      p.arrow(0, 0, b[0], b[1], { color: B, width: 3 });
      p.circle(a[0], a[1], 5, { fill: A });
      p.circle(b[0], b[1], 5, { fill: B });
      p.text(a[0], a[1], ' a', { color: A, size: 13, weight: '700' });
      p.text(b[0], b[1], ' b', { color: B, size: 13, weight: '700' });

      const na = LA.norm(a), nb = LA.norm(b);
      const cos = na * nb > 1e-9 ? dot / (na * nb) : 0;
      P.readout({
        'a': `(${fmt(a[0], 2)}, ${fmt(a[1], 2)})`,
        'b': `(${fmt(b[0], 2)}, ${fmt(b[1], 2)})`,
        '‖a‖': fmt(na, 3), '‖b‖': fmt(nb, 3),
        'a·b': fmt(dot, 3),
        'cos θ': fmt(cos, 3),
        'θ': fmt(Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI, 1) + '°',
      });
    },
    caption: 'The dot product $a\\cdot b = \\|a\\|\\|b\\|\\cos\\theta$ is a **similarity meter**: maximal when the arrows point the same way, zero when perpendicular, negative when opposed. Drag `b` until `a·b` hits 0 — that is orthogonality, and it is the single most reused fact in all of ML.',
  });
};

/* ---------- 2x2 linear map on a grid ---------- */

V['matrix-transform'] = (host) => {
  panel(host, {
    title: 'A matrix is a transformation of space',
    height: 340,
    plot: { xlim: [-3, 3], ylim: [-2.4, 2.4], equal: true },
    controls: [
      { type: 'slider', key: 'a', label: 'a₁₁', min: -2, max: 2, step: .05, value: 1 },
      { type: 'slider', key: 'b', label: 'a₁₂', min: -2, max: 2, step: .05, value: 0.6 },
      { type: 'slider', key: 'c', label: 'a₂₁', min: -2, max: 2, step: .05, value: 0.2 },
      { type: 'slider', key: 'd', label: 'a₂₂', min: -2, max: 2, step: .05, value: 1 },
      { type: 'slider', key: 't', label: 'animate ▸ identity → A', min: 0, max: 1, step: .01, value: 1 },
      { type: 'check', key: 'eig', label: 'show eigenvectors', value: true },
      { type: 'check', key: 'circ', label: 'unit circle → ellipse', value: true },
      {
        type: 'select', key: 'preset', label: 'preset', value: 'custom',
        options: [
          { value: 'custom', label: 'custom' }, { value: 'rot', label: 'rotation 40°' },
          { value: 'scale', label: 'scale (2, 0.5)' }, { value: 'shear', label: 'shear' },
          { value: 'proj', label: 'projection (rank 1)' }, { value: 'reflect', label: 'reflection' },
        ],
        onChange(v, P) {
          const M = { rot: [Math.cos(.7), -Math.sin(.7), Math.sin(.7), Math.cos(.7)], scale: [2, 0, 0, .5], shear: [1, 1, 0, 1], proj: [1, 0, 0, 0], reflect: [0, 1, 1, 0] }[v];
          if (M) { P.set('a', M[0], 1); P.set('b', M[1], 1); P.set('c', M[2], 1); P.set('d', M[3], 1); }
        },
      },
    ],
    draw(p, s, P) {
      const t = s.t;
      const M = [[1 + (s.a - 1) * t, s.b * t], [s.c * t, 1 + (s.d - 1) * t]];
      p.clear().axes({ nx: 5, ny: 4 });
      p.clip();
      // transformed grid
      const G = cssVar('--s1');
      for (let i = -6; i <= 6; i++) {
        const h = [], v = [];
        for (let u = -6; u <= 6; u += .5) {
          h.push([M[0][0] * u + M[0][1] * i, M[1][0] * u + M[1][1] * i]);
          v.push([M[0][0] * i + M[0][1] * u, M[1][0] * i + M[1][1] * u]);
        }
        const w = i === 0 ? 1.8 : .8;
        const al = i === 0 ? .75 : .28;
        p.line(h, { color: alpha(G, al), width: w });
        p.line(v, { color: alpha(G, al), width: w });
      }
      // unit square image
      const sq = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]].map(([x, y]) => [M[0][0] * x + M[0][1] * y, M[1][0] * x + M[1][1] * y]);
      p.area(sq, { color: alpha(cssVar('--s3'), .22) });
      p.line(sq, { color: cssVar('--s3'), width: 2 });

      if (s.circ) {
        const ci = [];
        for (let k = 0; k <= 72; k++) {
          const th = k / 72 * 6.2832, x = Math.cos(th), y = Math.sin(th);
          ci.push([M[0][0] * x + M[0][1] * y, M[1][0] * x + M[1][1] * y]);
        }
        p.line(ci, { color: alpha(cssVar('--s5'), .9), width: 1.6, dash: [5, 3] });
      }
      p.clip(false);

      const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
      const tr = M[0][0] + M[1][1];
      const disc = tr * tr / 4 - det;
      let eigTxt = 'complex';
      if (s.eig && disc >= 0) {
        const l1 = tr / 2 + Math.sqrt(disc), l2 = tr / 2 - Math.sqrt(disc);
        eigTxt = `${fmt(l1, 3)}, ${fmt(l2, 3)}`;
        for (const [l, col] of [[l1, cssVar('--s2')], [l2, cssVar('--s6')]]) {
          // eigenvector of [[a,b],[c,d]] for eigenvalue l
          let v = Math.abs(M[0][1]) > 1e-9 ? [M[0][1], l - M[0][0]] : (Math.abs(M[1][0]) > 1e-9 ? [l - M[1][1], M[1][0]] : [1, 0]);
          const n = Math.hypot(v[0], v[1]) || 1;
          v = [v[0] / n * 2.6, v[1] / n * 2.6];
          p.line([[-v[0], -v[1]], [v[0], v[1]]], { color: alpha(col, .35), width: 1, dash: [3, 3] });
          p.arrow(0, 0, v[0] / 2.6, v[1] / 2.6, { color: alpha(col, .9), width: 1.5 });
          p.arrow(0, 0, v[0] / 2.6 * l, v[1] / 2.6 * l, { color: col, width: 2.6 });
        }
      }
      P.readout({ 'det A': fmt(det, 3), 'trace': fmt(tr, 3), 'eigenvalues': eigTxt, 'area scale': `${fmt(Math.abs(det), 3)}×`, 'invertible': Math.abs(det) > 1e-6 ? 'yes' : 'no — rank deficient' });
    },
    caption: 'Grid lines stay straight and evenly spaced — that *is* linearity. **det A** is the signed area scale factor: watch it hit 0 as the transformation squashes the plane onto a line (the rank-1 preset). The thick arrows are eigenvectors: directions the matrix only stretches, never rotates.',
  });
};

/* ---------- SVD ---------- */

V['svd-demo'] = (host) => {
  panel(host, {
    title: 'SVD: every matrix is rotate → stretch → rotate',
    height: 300,
    plot: { xlim: [-3, 3], ylim: [-2.2, 2.2], equal: true },
    controls: [
      { type: 'slider', key: 'th1', label: 'Vᵀ rotation', min: 0, max: 6.28, step: .01, value: 0.6 },
      { type: 'slider', key: 's1', label: 'σ₁', min: 0, max: 2.5, step: .05, value: 1.9 },
      { type: 'slider', key: 's2', label: 'σ₂', min: 0, max: 2.5, step: .05, value: 0.6 },
      { type: 'slider', key: 'th2', label: 'U rotation', min: 0, max: 6.28, step: .01, value: 1.1 },
      { type: 'slider', key: 'stage', label: 'stage: I → Vᵀ → ΣVᵀ → UΣVᵀ', min: 0, max: 3, step: .01, value: 3 },
    ],
    draw(p, s, P) {
      const rot = (t) => [[Math.cos(t), -Math.sin(t)], [Math.sin(t), Math.cos(t)]];
      const st = s.stage;
      const f1 = Math.min(st, 1), f2 = Math.max(0, Math.min(st - 1, 1)), f3 = Math.max(0, Math.min(st - 2, 1));
      const Vt = rot(-s.th1 * f1);
      const S = [[1 + (s.s1 - 1) * f2, 0], [0, 1 + (s.s2 - 1) * f2]];
      const U = rot(s.th2 * f3);
      const M = LA.matmul(U, LA.matmul(S, Vt));
      p.clear().axes({ nx: 5, ny: 4 });
      p.clip();
      const pts = [], colors = [];
      for (let k = 0; k <= 96; k++) {
        const th = k / 96 * 6.2832;
        const v = [Math.cos(th), Math.sin(th)];
        pts.push([M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]]);
        colors.push(seqMap(k / 96));
      }
      p.line([[-3, 0], [3, 0]], { color: alpha(cssVar('--axis'), .3), width: 1 });
      p.points(pts, { r: 2.6, color: (_, i) => colors[i] });
      // singular directions
      for (const [i, col] of [[0, cssVar('--s2')], [1, cssVar('--s6')]]) {
        const e = i === 0 ? [1, 0] : [0, 1];
        const uv = [U[0][0] * e[0] + U[0][1] * e[1], U[1][0] * e[0] + U[1][1] * e[1]];
        const sv = i === 0 ? S[0][0] : S[1][1];
        p.arrow(0, 0, uv[0] * sv, uv[1] * sv, { color: col, width: 2.6 });
        p.text(uv[0] * sv, uv[1] * sv, ` σ${i + 1}u${i + 1}`, { color: col, size: 11, weight: '600' });
      }
      p.clip(false);
      const stageName = st < 1 ? 'rotating (Vᵀ)' : st < 2 ? 'stretching (Σ)' : 'rotating (U)';
      P.readout({ stage: stageName, 'σ₁/σ₂ (condition κ)': fmt(s.s1 / Math.max(s.s2, 1e-6), 2), 'rank': (s.s1 > 1e-3 ? 1 : 0) + (s.s2 > 1e-3 ? 1 : 0) });
    },
    caption: 'Colors track where each point on the unit circle ends up. Slide **stage** to watch $A = U\\Sigma V^{\\mathsf T}$ happen in three acts. Push $\\sigma_2 \\to 0$: the circle collapses to a segment — that is a rank-1 matrix, and it is exactly what low-rank adaptation (LoRA) exploits.',
  });
};

/* ---------- derivative & tangent ---------- */

const FUNCS = {
  'x^2': { f: (x) => x * x, d: (x) => 2 * x, tex: 'f(x)=x^2' },
  'sin(x)': { f: Math.sin, d: Math.cos, tex: 'f(x)=\\sin x' },
  'x^3-2x': { f: (x) => x ** 3 - 2 * x, d: (x) => 3 * x * x - 2, tex: 'f(x)=x^3-2x' },
  'e^x': { f: Math.exp, d: Math.exp, tex: 'f(x)=e^x' },
  'log(1+e^x)': { f: (x) => Math.log1p(Math.exp(x)), d: (x) => 1 / (1 + Math.exp(-x)), tex: 'f(x)=\\log(1+e^x)' },
  '|x| (relu-ish)': { f: (x) => Math.abs(x), d: (x) => Math.sign(x), tex: 'f(x)=|x|' },
};

V['derivative-tangent'] = (host) => {
  panel(host, {
    title: 'The derivative is the slope of the best local line',
    height: 300,
    plot: { xlim: [-3, 3], ylim: [-3, 4], xlabel: 'x' },
    controls: [
      { type: 'select', key: 'fn', label: 'function', value: 'x^2', options: Object.keys(FUNCS) },
      { type: 'slider', key: 'x0', label: 'x₀', min: -3, max: 3, step: .01, value: 1 },
      { type: 'slider', key: 'h', label: 'h (secant gap)', min: 0.001, max: 2, step: .001, value: 1, fmt: (v) => v.toFixed(3) },
      { type: 'check', key: 'sec', label: 'show secant', value: true },
    ],
    draw(p, s, P) {
      const { f, d } = FUNCS[s.fn];
      const x0 = s.x0, h = s.h;
      p.clear().axes();
      p.clip();
      p.fn(f, { color: cssVar('--s1'), width: 2.4 });
      const slope = d(x0), y0 = f(x0);
      p.fn((x) => y0 + slope * (x - x0), { color: cssVar('--s2'), width: 2, dash: [] });
      if (s.sec) {
        const y1 = f(x0 + h), sec = (y1 - y0) / h;
        p.fn((x) => y0 + sec * (x - x0), { color: alpha(cssVar('--s4'), .85), width: 1.6, dash: [5, 4] });
        p.points([[x0 + h, y1]], { r: 4.5, color: cssVar('--s4') });
        p.line([[x0, y0], [x0 + h, y0]], { color: alpha(cssVar('--text-faint'), .6), width: 1 });
        p.line([[x0 + h, y0], [x0 + h, y1]], { color: alpha(cssVar('--text-faint'), .6), width: 1 });
        p.text(x0 + h / 2, y0, 'h', { color: cssVar('--text-faint'), size: 10, align: 'center', baseline: 'top' });
        P.readout({
          'f′(x₀) exact': fmt(slope, 4),
          'secant (f(x₀+h)−f(x₀))/h': fmt(sec, 4),
          'error': fmt(Math.abs(sec - slope), 4),
        });
      } else P.readout({ 'f(x₀)': fmt(y0, 4), 'f′(x₀)': fmt(slope, 4) });
      p.points([[x0, y0]], { r: 5.5, color: cssVar('--s2'), stroke: cssVar('--bg-inset'), strokeWidth: 2 });
      p.clip(false);
    },
    caption: 'Shrink **h** and watch the dashed secant swing onto the solid tangent. That limit is the whole definition: $f\'(x)=\\lim_{h\\to 0}\\frac{f(x+h)-f(x)}{h}$. Notice the error shrinks roughly *linearly* in $h$ — which is why finite-difference gradient checks need small-but-not-too-small $h$.',
  });
};

/* ---------- gradient field ---------- */

const SURFACES = {
  bowl: { f: (x, y) => 0.5 * (x * x + y * y), g: (x, y) => [x, y], name: 'isotropic bowl', tex: '\\tfrac12(x^2+y^2)' },
  ellipse: { f: (x, y) => 0.5 * (0.15 * x * x + 4 * y * y), g: (x, y) => [0.15 * x, 4 * y], name: 'ill-conditioned valley', tex: '\\tfrac12(0.15x^2+4y^2)' },
  rosen: { f: (x, y) => 0.02 * ((1 - x) ** 2 + 40 * (y - x * x) ** 2), g: (x, y) => [0.02 * (-2 * (1 - x) - 160 * x * (y - x * x)), 0.02 * 80 * (y - x * x)], name: 'Rosenbrock banana', tex: '(1-x)^2+40(y-x^2)^2' },
  saddle: { f: (x, y) => 0.5 * (x * x - y * y), g: (x, y) => [x, -y], name: 'saddle', tex: '\\tfrac12(x^2-y^2)' },
  multi: {
    f: (x, y) => -Math.exp(-((x - 1.4) ** 2 + (y - 1) ** 2)) * 1.4 - Math.exp(-((x + 1.6) ** 2 + (y + 1.2) ** 2) / 1.6) * 1.1 + 0.06 * (x * x + y * y),
    g: (x, y) => {
      const e = 1e-4;
      const F = SURFACES.multi.f;
      return [(F(x + e, y) - F(x - e, y)) / (2 * e), (F(x, y + e) - F(x, y - e)) / (2 * e)];
    },
    name: 'two basins', tex: '\\text{mixture of Gaussians}',
  },
};

V['gradient-field'] = (host, params = {}) => {
  panel(host, {
    title: 'Gradients point uphill, steepest-first',
    height: 340,
    plot: { xlim: [-3, 3], ylim: [-2.4, 2.4], equal: true, xlabel: 'θ₁', ylabel: 'θ₂' },
    controls: [
      { type: 'select', key: 'surf', label: 'surface', value: params.surf || 'ellipse', options: Object.keys(SURFACES).map((k) => ({ value: k, label: SURFACES[k].name })) },
      { type: 'check', key: 'field', label: 'gradient field', value: true },
      { type: 'check', key: 'heat', label: 'heatmap', value: true },
      { type: 'check', key: 'neg', label: 'show −∇ (descent) at cursor', value: true },
    ],
    interact: (s, P) => ({
      hover(x, y) { P.state.cx = x; P.state.cy = y; P.redraw(); },
      leave() { P.state.cx = null; P.redraw(); },
    }),
    draw(p, s, P) {
      const S = SURFACES[s.surf];
      p.clear();
      if (s.heat) p.heat((x, y) => S.f(x, y), { step: 5, cmap: (t) => mix(cssVar('--bg-inset'), cssVar('--s1'), t * .55), alpha: 1 });
      p.axes({ nx: 5, ny: 4 });
      p.clip();
      const vals = [];
      for (let i = 1; i <= 11; i++) vals.push(S.f(0, 0) + (i / 11) ** 2 * 8);
      p.contour(S.f, vals, { color: alpha(cssVar('--axis'), .5), width: 1 });
      if (s.field) p.quiver((x, y) => S.g(x, y), { color: alpha(cssVar('--s2'), .6), nx: 15, ny: 11, scale: .8, alphaBy: true });
      if (s.neg && s.cx != null) {
        const g = S.g(s.cx, s.cy);
        const n = Math.hypot(g[0], g[1]) || 1;
        p.arrow(s.cx, s.cy, s.cx - g[0] / n * .9, s.cy - g[1] / n * .9, { color: cssVar('--s3'), width: 3, head: 10 });
        p.circle(s.cx, s.cy, 4, { fill: cssVar('--s3') });
        P.readout({ 'θ': `(${fmt(s.cx, 2)}, ${fmt(s.cy, 2)})`, 'L(θ)': fmt(S.f(s.cx, s.cy), 4), '∇L': `(${fmt(g[0], 3)}, ${fmt(g[1], 3)})`, '‖∇L‖': fmt(n, 3) });
      }
      p.clip(false);
    },
    caption: 'Move your cursor over the surface. The green arrow is $-\\nabla L$ — the direction that decreases the loss fastest *right there*. Note it is always perpendicular to the contour line through that point, and it gets short near the minimum. Gradient descent is nothing more than repeatedly taking a small step along the green arrow.',
  });
};

/* ---------- gradient descent / optimizers ---------- */

V['gradient-descent'] = (host, params = {}) => {
  const makeOpt = (kind, lr, x0, y0) => ({
    kind, lr, x: x0, y: y0, vx: 0, vy: 0, mx: 0, my: 0, sx: 0, sy: 0, t: 0, path: [[x0, y0]],
  });
  let opts = [];

  const step = (o, S, s) => {
    const [gx0, gy0] = S.g(o.x, o.y);
    let gx = gx0, gy = gy0;
    if (s.noise > 0) { gx += (Math.random() - .5) * s.noise * 2; gy += (Math.random() - .5) * s.noise * 2; }
    o.t++;
    const lr = o.lr;
    if (o.kind === 'sgd') { o.x -= lr * gx; o.y -= lr * gy; }
    else if (o.kind === 'momentum') {
      o.vx = s.beta * o.vx - lr * gx; o.vy = s.beta * o.vy - lr * gy;
      o.x += o.vx; o.y += o.vy;
    } else if (o.kind === 'nesterov') {
      const lx = o.x + s.beta * o.vx, ly = o.y + s.beta * o.vy;
      const [lgx, lgy] = S.g(lx, ly);
      o.vx = s.beta * o.vx - lr * lgx; o.vy = s.beta * o.vy - lr * lgy;
      o.x += o.vx; o.y += o.vy;
    } else if (o.kind === 'rmsprop') {
      o.sx = .9 * o.sx + .1 * gx * gx; o.sy = .9 * o.sy + .1 * gy * gy;
      o.x -= lr * gx / (Math.sqrt(o.sx) + 1e-8); o.y -= lr * gy / (Math.sqrt(o.sy) + 1e-8);
    } else if (o.kind === 'adam') {
      const b1 = .9, b2 = .999;
      o.mx = b1 * o.mx + (1 - b1) * gx; o.my = b1 * o.my + (1 - b1) * gy;
      o.sx = b2 * o.sx + (1 - b2) * gx * gx; o.sy = b2 * o.sy + (1 - b2) * gy * gy;
      const mhx = o.mx / (1 - b1 ** o.t), mhy = o.my / (1 - b1 ** o.t);
      const shx = o.sx / (1 - b2 ** o.t), shy = o.sy / (1 - b2 ** o.t);
      o.x -= lr * mhx / (Math.sqrt(shx) + 1e-8); o.y -= lr * mhy / (Math.sqrt(shy) + 1e-8);
    }
    if (!isFinite(o.x) || Math.abs(o.x) > 40) { o.x = Math.sign(o.x) * 40; o.diverged = true; }
    if (!isFinite(o.y) || Math.abs(o.y) > 40) { o.y = Math.sign(o.y) * 40; o.diverged = true; }
    o.path.push([o.x, o.y]);
    if (o.path.length > 900) o.path.shift();
  };

  const reset = (s) => {
    const kinds = s.compare ? ['sgd', 'momentum', 'rmsprop', 'adam'] : [s.opt];
    opts = kinds.map((k) => makeOpt(k, s.lr * (k === 'adam' || k === 'rmsprop' ? 1 : 1), s.x0, s.y0));
  };

  panel(host, {
    title: 'Gradient descent — watch the ball roll',
    height: 340,
    plot: { xlim: [-3, 3], ylim: [-2.4, 2.4], equal: true, xlabel: 'θ₁', ylabel: 'θ₂' },
    state: { x0: params.x0 != null ? params.x0 : -2.5, y0: params.y0 != null ? params.y0 : 2 },
    controls: [
      { type: 'select', key: 'surf', label: 'loss surface', value: params.surf || 'ellipse', options: Object.keys(SURFACES).map((k) => ({ value: k, label: SURFACES[k].name })), onChange: (v, P) => reset(P.state) },
      { type: 'select', key: 'opt', label: 'optimizer', value: params.opt || 'sgd', options: [{ value: 'sgd', label: 'plain GD' }, { value: 'momentum', label: 'momentum' }, { value: 'nesterov', label: 'Nesterov' }, { value: 'rmsprop', label: 'RMSProp' }, { value: 'adam', label: 'Adam' }], onChange: (v, P) => reset(P.state) },
      { type: 'slider', key: 'lr', label: 'learning rate η', min: 0.001, max: 0.9, step: .001, value: params.lr || 0.1, fmt: (v) => v.toFixed(3), onChange: (v, P) => reset(P.state) },
      { type: 'slider', key: 'beta', label: 'momentum β', min: 0, max: .99, step: .01, value: .9 },
      { type: 'slider', key: 'noise', label: 'gradient noise (→ SGD)', min: 0, max: 3, step: .05, value: 0 },
      { type: 'check', key: 'compare', label: 'race all four', value: !!params.compare, onChange: (v, P) => reset(P.state) },
      { type: 'play' },
      { type: 'button', label: '↺ reset', onClick: (s) => reset(s) },
    ],
    init: (P, s) => reset(s),
    interact: (s, P) => ({
      down(x, y) { s.x0 = x; s.y0 = y; reset(s); P.redraw(); return true; },
      move(x, y) { s.x0 = x; s.y0 = y; reset(s); P.redraw(); },
    }),
    animate(s, P) { const S = SURFACES[s.surf]; for (const o of opts) if (!o.diverged) step(o, S, s); },
    draw(p, s, P) {
      const S = SURFACES[s.surf];
      p.clear();
      p.heat((x, y) => S.f(x, y), { step: 5, cmap: (t) => mix(cssVar('--bg-inset'), cssVar('--s1'), t * .5) });
      p.axes({ nx: 5, ny: 4 });
      p.clip();
      const vals = [];
      for (let i = 1; i <= 12; i++) vals.push(S.f(0, 0) + (i / 12) ** 2 * 8);
      p.contour(S.f, vals, { color: alpha(cssVar('--axis'), .45) });
      const cols = SERIES();
      const names = { sgd: 'plain GD', momentum: 'momentum', nesterov: 'Nesterov', rmsprop: 'RMSProp', adam: 'Adam' };
      opts.forEach((o, i) => {
        const c = cols[i % cols.length];
        p.line(o.path, { color: alpha(c, .85), width: 1.8 });
        p.points(o.path.filter((_, k) => k % 6 === 0), { r: 1.6, color: alpha(c, .5) });
        p.circle(o.x, o.y, 5.5, { fill: c, stroke: cssVar('--bg-inset'), width: 2 });
      });
      p.clip(false);
      if (opts.length > 1) p.legend(opts.map((o, i) => ({ label: `${names[o.kind]} — L=${o.diverged ? '∞' : fmt(S.f(o.x, o.y), 3)}`, color: cols[i % cols.length], shape: 'dot' })), { pos: 'tr' });
      const o = opts[0];
      P.readout(opts.length === 1 ? {
        step: o.t, 'θ': `(${fmt(o.x, 3)}, ${fmt(o.y, 3)})`, 'L(θ)': o.diverged ? 'diverged ✗' : fmt(S.f(o.x, o.y), 5),
        '‖∇L‖': fmt(Math.hypot(...S.g(o.x, o.y)), 4),
      } : { step: o.t, hint: 'click the surface to move the start point' });
    },
    caption: 'Click anywhere to drop the ball there, then press play. Try: η too large on the **ill-conditioned valley** (it oscillates across the narrow axis and crawls along the long one) — then switch on momentum or Adam and watch the pathology disappear. Add gradient noise to see why SGD escapes shallow traps that full-batch GD sits in.',
  });
};

/* ---------- learning rate: 1D stability ---------- */

V['lr-stability'] = (host) => {
  panel(host, {
    title: 'Why too-large a learning rate explodes',
    height: 280,
    plot: { xlim: [-4, 4], ylim: [-0.5, 9], xlabel: 'θ', ylabel: 'L(θ)' },
    controls: [
      { type: 'slider', key: 'lr', label: 'learning rate η', min: 0.01, max: 1.3, step: .01, value: .3 },
      { type: 'slider', key: 'curv', label: 'curvature L″ = a', min: .2, max: 4, step: .05, value: 1 },
      { type: 'slider', key: 'steps', label: 'steps shown', min: 1, max: 30, step: 1, value: 12 },
    ],
    draw(p, s, P) {
      const a = s.curv;
      const f = (x) => .5 * a * x * x, g = (x) => a * x;
      p.setLim([-4, 4], [-.5, Math.min(9, .5 * a * 16)]);
      p.clear().axes();
      p.clip();
      p.fn(f, { color: cssVar('--s1'), width: 2.4 });
      let x = 3.4;
      const pts = [[x, f(x)]];
      for (let i = 0; i < s.steps; i++) {
        const nx = x - s.lr * g(x);
        p.line([[x, f(x)], [nx, f(x)]], { color: alpha(cssVar('--s2'), .5), width: 1.2, dash: [3, 3] });
        p.line([[nx, f(x)], [nx, f(nx)]], { color: alpha(cssVar('--s2'), .5), width: 1.2, dash: [3, 3] });
        x = nx;
        if (!isFinite(x) || Math.abs(x) > 100) break;
        pts.push([x, f(x)]);
      }
      p.points(pts, { r: 3.4, color: (_, i) => mix(cssVar('--s2'), cssVar('--s3'), i / Math.max(pts.length - 1, 1)) });
      p.clip(false);
      const crit = 2 / a;
      P.readout({
        'stable if η <': fmt(crit, 3), 'current η': fmt(s.lr, 3),
        'ratio η·a': fmt(s.lr * a, 3),
        regime: s.lr * a < 1 ? 'monotone descent' : s.lr * a < 2 ? 'oscillating but converging' : 'divergent ✗',
        ['|θ| after ' + s.steps + ' steps']: isFinite(x) ? fmt(Math.abs(x), 4) : '∞',
      });
    },
    caption: 'On a quadratic $L=\\tfrac12 a\\theta^2$, one GD step maps $\\theta \\mapsto (1-\\eta a)\\theta$. Convergence needs $|1-\\eta a| < 1$, i.e. $\\eta < 2/a$. Push η past that line and the iterates blow up geometrically. **In deep nets $a$ is the largest eigenvalue of the Hessian**, which is why the usable learning rate is set by the sharpest direction in the whole loss landscape — and why normalization, which tames curvature, lets you raise it.',
  });
};

/* ---------- Taylor ---------- */

V['taylor-approx'] = (host) => {
  panel(host, {
    title: 'Taylor series: building a function from its derivatives',
    height: 290,
    plot: { xlim: [-4, 4], ylim: [-2.5, 4], xlabel: 'x' },
    controls: [
      { type: 'select', key: 'fn', label: 'function', value: 'sin', options: [{ value: 'sin', label: 'sin x' }, { value: 'exp', label: 'eˣ' }, { value: 'log', label: 'log(1+eˣ) softplus' }, { value: 'sig', label: 'σ(x) sigmoid' }] },
      { type: 'slider', key: 'order', label: 'order n', min: 0, max: 9, step: 1, value: 1 },
      { type: 'slider', key: 'a', label: 'expansion point a', min: -3, max: 3, step: .05, value: 0 },
    ],
    draw(p, s, P) {
      const F = { sin: Math.sin, exp: Math.exp, log: (x) => Math.log1p(Math.exp(x)), sig: (x) => 1 / (1 + Math.exp(-x)) }[s.fn];
      // numeric derivatives via central differences on a fine stencil
      const deriv = (f, x, n) => {
        if (n === 0) return f(x);
        const h = 0.35;
        let sum = 0;
        for (let k = 0; k <= n; k++) {
          sum += (-1) ** k * binom(n, k) * f(x + (n / 2 - k) * h);
        }
        return sum / h ** n;
      };
      const a = s.a;
      const coef = [];
      let fact = 1;
      for (let n = 0; n <= s.order; n++) {
        if (n > 0) fact *= n;
        coef.push(deriv(F, a, n) / fact);
      }
      const T = (x) => coef.reduce((acc, c, n) => acc + c * (x - a) ** n, 0);
      p.clear().axes();
      p.clip();
      p.fn(F, { color: cssVar('--s1'), width: 2.6 });
      p.fn(T, { color: cssVar('--s2'), width: 2.2, dash: [6, 3] });
      p.points([[a, F(a)]], { r: 5, color: cssVar('--s3') });
      p.clip(false);
      p.legend([{ label: 'f(x)', color: cssVar('--s1') }, { label: `order-${s.order} Taylor`, color: cssVar('--s2'), dash: true }], { pos: 'tl' });
      let err = 0;
      for (let x = a - 1; x <= a + 1; x += .05) err = Math.max(err, Math.abs(F(x) - T(x)));
      P.readout({ 'max |f − T| on [a−1, a+1]': fmt(err, 5), 'terms': s.order + 1 });
    },
    caption: 'Order 0 is a flat line (the value). Order 1 is the tangent — **this is the approximation gradient descent implicitly trusts**. Order 2 adds curvature, which is what Newton\'s method and second-order optimizers use. Notice how the approximation is excellent near $a$ and falls apart far from it: that is exactly why you take *small* steps.',
  });
};

function binom(n, k) { let r = 1; for (let i = 0; i < k; i++) r = r * (n - i) / (i + 1); return r; }

/* ---------- chain rule ---------- */

V['chain-rule'] = (host) => {
  panel(host, {
    title: 'The chain rule as a gain cascade',
    height: 250,
    plot: { xlim: [-3, 3], ylim: [-1.5, 3], xlabel: 'x' },
    controls: [
      { type: 'slider', key: 'x', label: 'x', min: -3, max: 3, step: .01, value: 0.8 },
      { type: 'slider', key: 'w', label: 'inner weight w', min: -3, max: 3, step: .05, value: 1.5 },
      { type: 'select', key: 'outer', label: 'outer g(u)', value: 'sq', options: [{ value: 'sq', label: 'u²' }, { value: 'sig', label: 'σ(u)' }, { value: 'tanh', label: 'tanh u' }, { value: 'relu', label: 'ReLU u' }] },
    ],
    draw(p, s, P) {
      const inner = (x) => s.w * x;
      const G = {
        sq: [(u) => u * u, (u) => 2 * u, 'u^2'],
        sig: [(u) => 1 / (1 + Math.exp(-u)), (u) => { const t = 1 / (1 + Math.exp(-u)); return t * (1 - t); }, '\\sigma(u)'],
        tanh: [Math.tanh, (u) => 1 - Math.tanh(u) ** 2, '\\tanh u'],
        relu: [(u) => Math.max(0, u), (u) => (u > 0 ? 1 : 0), '\\mathrm{ReLU}(u)'],
      }[s.outer];
      const h = (x) => G[0](inner(x));
      p.clear().axes();
      p.clip();
      p.fn(inner, { color: alpha(cssVar('--s4'), .8), width: 1.8, dash: [5, 3] });
      p.fn(h, { color: cssVar('--s1'), width: 2.6 });
      const u = inner(s.x);
      const dg = G[1](u), du = s.w, dh = dg * du;
      p.fn((x) => h(s.x) + dh * (x - s.x), { color: cssVar('--s2'), width: 1.8 });
      p.points([[s.x, h(s.x)]], { r: 5, color: cssVar('--s2') });
      p.clip(false);
      p.legend([{ label: 'u = wx', color: cssVar('--s4'), dash: true }, { label: 'h = g(u)', color: cssVar('--s1') }, { label: 'tangent', color: cssVar('--s2') }], { pos: 'tl' });
      P.readout({
        'u = wx': fmt(u, 3),
        'dh/du = g′(u)': fmt(dg, 4),
        'du/dx = w': fmt(du, 3),
        'dh/dx = g′(u)·w': fmt(dh, 4),
      });
    },
    caption: 'Each layer multiplies the signal that flows back. Pick $\\sigma(u)$ and slide $w$ up: the outer gain $g\'(u)$ collapses toward 0 as the sigmoid saturates, so $dh/dx \\to 0$ **no matter how big $w$ is**. Stack forty of those factors and you have the vanishing gradient problem in one picture.',
  });
};

/* ---------- probability distributions ---------- */

const DISTS = {
  normal: {
    name: 'Normal', params: [['μ', -3, 3, 0], ['σ', .2, 3, 1]],
    pdf: (x, [m, s]) => Math.exp(-((x - m) ** 2) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI)),
    support: [-6, 6],
    note: 'Maximum entropy given a mean and variance. Sums of many independent things converge to it (CLT). Log-density is a parabola — which is why Gaussian MLE = least squares.',
  },
  bernoulli: {
    name: 'Bernoulli / Binomial', params: [['p', 0, 1, .5], ['n', 1, 30, 10]],
    discrete: true,
    pmf: (k, [p, n]) => { n = Math.round(n); if (k < 0 || k > n) return 0; return binom(n, k) * p ** k * (1 - p) ** (n - k); },
    support: [0, 30],
    note: 'Counts of successes. As n grows with p fixed it approaches a Normal; with np fixed it approaches Poisson.',
  },
  beta: {
    name: 'Beta', params: [['α', .3, 10, 2], ['β', .3, 10, 2]],
    pdf: (x, [a, b]) => (x <= 0 || x >= 1 ? 0 : Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lbeta(a, b))),
    support: [0, 1],
    note: 'The conjugate prior for a Bernoulli probability. α−1 and β−1 act like "pseudo-counts" of prior successes and failures.',
  },
  exponential: {
    name: 'Exponential', params: [['λ', .2, 4, 1]],
    pdf: (x, [l]) => (x < 0 ? 0 : l * Math.exp(-l * x)),
    support: [0, 8],
    note: 'Waiting time with no memory. The maximum-entropy distribution on the positive reals given a mean.',
  },
  laplace: {
    name: 'Laplace', params: [['μ', -3, 3, 0], ['b', .2, 3, 1]],
    pdf: (x, [m, b]) => Math.exp(-Math.abs(x - m) / b) / (2 * b),
    support: [-6, 6],
    note: 'Heavier tails than a Normal, and a sharp peak. Its negative log-density is |x|, so a Laplace prior gives you **L1 / Lasso**.',
  },
  student: {
    name: "Student's t", params: [['ν (dof)', 1, 30, 3]],
    pdf: (x, [v]) => Math.exp(lgamma((v + 1) / 2) - lgamma(v / 2)) / Math.sqrt(v * Math.PI) * (1 + x * x / v) ** (-(v + 1) / 2),
    support: [-6, 6],
    note: 'A Normal with an uncertain variance. Heavy tails make it robust to outliers; ν→∞ recovers the Normal.',
  },
};
function lgamma(z) {
  const g = [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  z -= 1;
  let x = 0.99999999999980993;
  for (let i = 0; i < g.length; i++) x += g[i] / (z + i + 1);
  const t = z + g.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}
function lbeta(a, b) { return lgamma(a) + lgamma(b) - lgamma(a + b); }

V['distributions'] = (host) => {
  let sliders = [];
  const P = panel(host, {
    title: 'The distributions you actually meet',
    height: 280,
    plot: { xlim: [-6, 6], ylim: [0, .7], xlabel: 'x', ylabel: 'density' },
    controls: [
      {
        type: 'select', key: 'd', label: 'distribution', value: 'normal',
        options: Object.keys(DISTS).map((k) => ({ value: k, label: DISTS[k].name })),
        onChange(v, P) { syncSliders(v, P); },
      },
      { type: 'slider', key: 'p0', label: 'param 1', min: -3, max: 3, step: .01, value: 0 },
      { type: 'slider', key: 'p1', label: 'param 2', min: .2, max: 3, step: .01, value: 1 },
      { type: 'check', key: 'samples', label: 'draw 300 samples', value: false },
    ],
    draw(p, s, PP) {
      const D = DISTS[s.d];
      const pv = D.params.map((pd, i) => (i === 0 ? s.p0 : s.p1));
      p.setLim(D.support, [0, 1]);
      // autoscale y
      let mx = 0;
      if (D.discrete) {
        const n = Math.round(pv[1] || 10);
        for (let k = 0; k <= n; k++) mx = Math.max(mx, D.pmf(k, pv));
        p.setLim([-0.5, n + .5], [0, mx * 1.25]);
        p.clear().axes({ nx: Math.min(n, 10) });
        const vals = [];
        for (let k = 0; k <= n; k++) vals.push(D.pmf(k, pv));
        p.bars(vals, { x0: -0.5, color: cssVar('--s1'), gap: .25 });
        const mean = pv[0] * n, sd = Math.sqrt(n * pv[0] * (1 - pv[0]));
        PP.readout({ mean: fmt(mean, 3), sd: fmt(sd, 3), 'P(k=mode)': fmt(mx, 4) });
      } else {
        for (let x = D.support[0]; x <= D.support[1]; x += .01) mx = Math.max(mx, D.pdf(x, pv));
        p.setLim(D.support, [0, mx * 1.2]);
        p.clear().axes();
        p.clip();
        const pts = [];
        for (let x = D.support[0]; x <= D.support[1]; x += (D.support[1] - D.support[0]) / 400) pts.push([x, D.pdf(x, pv)]);
        p.area(pts, { color: alpha(cssVar('--s1'), .2), base: 0 });
        p.line(pts, { color: cssVar('--s1'), width: 2.4 });
        if (s.samples) {
          const r = rng(7);
          const smp = [];
          // rejection sampling
          let guard = 0;
          while (smp.length < 300 && guard++ < 40000) {
            const x = D.support[0] + r() * (D.support[1] - D.support[0]);
            if (r() * mx < D.pdf(x, pv)) smp.push([x, (r() * .12 + .02) * mx]);
          }
          p.points(smp, { r: 2, color: alpha(cssVar('--s2'), .55) });
        }
        p.clip(false);
        // numeric mean/var
        let m = 0, z = 0, v = 0;
        const dx = (D.support[1] - D.support[0]) / 2000;
        for (let x = D.support[0]; x <= D.support[1]; x += dx) { const d = D.pdf(x, pv); z += d * dx; m += x * d * dx; }
        m /= z || 1;
        for (let x = D.support[0]; x <= D.support[1]; x += dx) { const d = D.pdf(x, pv); v += (x - m) ** 2 * d * dx; }
        v /= z || 1;
        PP.readout({ mean: fmt(m, 3), variance: fmt(v, 3), 'peak density': fmt(mx, 3) });
      }
    },
    caption: '',
  });

  function syncSliders(k, P) {
    const D = DISTS[k];
    const wraps = P.el.querySelectorAll('.controls .ctl');
    D.params.forEach((pd, i) => {
      const w = wraps[i + 1];
      if (!w) return;
      w.style.display = '';
      w.querySelector('.ctl-label span').textContent = pd[0];
      const inp = w.querySelector('input');
      inp.min = pd[1]; inp.max = pd[2]; inp.value = pd[3];
      inp.step = (pd[2] - pd[1]) / 200;
      P.state[i === 0 ? 'p0' : 'p1'] = pd[3];
      w.querySelector('b').textContent = pd[3];
    });
    if (D.params.length < 2) wraps[2].style.display = 'none';
    const cap = P.el.querySelector('.viz-caption');
    if (cap) cap.innerHTML = `<strong>${D.name}.</strong> ${D.note}`;
    P.redraw();
  }
  syncSliders('normal', P);
};

/* ---------- Bayes updating ---------- */

V['bayes-coin'] = (host) => {
  let data = [];
  panel(host, {
    title: 'Bayesian updating — click to flip',
    height: 280,
    plot: { xlim: [0, 1], ylim: [0, 6], xlabel: 'θ  (probability of heads)', ylabel: 'density' },
    controls: [
      { type: 'slider', key: 'a0', label: 'prior α₀', min: .5, max: 20, step: .5, value: 1 },
      { type: 'slider', key: 'b0', label: 'prior β₀', min: .5, max: 20, step: .5, value: 1 },
      { type: 'slider', key: 'truth', label: 'true θ (hidden coin bias)', min: 0, max: 1, step: .01, value: .7 },
      { type: 'button', label: 'flip once', onClick: (s) => data.push(Math.random() < s.truth ? 1 : 0) },
      { type: 'button', label: 'flip ×10', onClick: (s) => { for (let i = 0; i < 10; i++) data.push(Math.random() < s.truth ? 1 : 0); } },
      { type: 'button', label: '↺ forget data', onClick: () => (data = []) },
    ],
    draw(p, s, P) {
      const h = data.filter((d) => d).length, t = data.length - h;
      const a = s.a0 + h, b = s.b0 + t;
      const post = (x) => (x <= 0 || x >= 1 ? 0 : Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lbeta(a, b)));
      const prior = (x) => (x <= 0 || x >= 1 ? 0 : Math.exp((s.a0 - 1) * Math.log(x) + (s.b0 - 1) * Math.log(1 - x) - lbeta(s.a0, s.b0)));
      let mx = 0;
      for (let x = 0; x <= 1; x += .002) mx = Math.max(mx, post(x), prior(x));
      p.setLim([0, 1], [0, Math.min(mx * 1.15, 22)]);
      p.clear().axes();
      p.clip();
      const pp = [], qq = [];
      for (let x = 0; x <= 1; x += .002) { pp.push([x, prior(x)]); qq.push([x, post(x)]); }
      p.line(pp, { color: alpha(cssVar('--s4'), .9), width: 1.8, dash: [5, 3] });
      p.area(qq, { color: alpha(cssVar('--s1'), .2) });
      p.line(qq, { color: cssVar('--s1'), width: 2.6 });
      const mle = data.length ? h / data.length : null;
      if (mle != null) p.line([[mle, 0], [mle, 99]], { color: cssVar('--s2'), width: 1.6, dash: [4, 3] });
      p.line([[s.truth, 0], [s.truth, 99]], { color: cssVar('--s3'), width: 1.6 });
      p.clip(false);
      p.legend([
        { label: 'prior', color: cssVar('--s4'), dash: true },
        { label: 'posterior', color: cssVar('--s1') },
        { label: 'MLE (h/n)', color: cssVar('--s2'), dash: true },
        { label: 'true θ', color: cssVar('--s3') },
      ], { pos: 'tl' });
      P.readout({
        'flips': data.length, 'heads': h, 'tails': t,
        'posterior': `Beta(${fmt(a, 1)}, ${fmt(b, 1)})`,
        'mean': fmt(a / (a + b), 4),
        'sd': fmt(Math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))), 4),
      });
    },
    caption: 'Beta is *conjugate* to Bernoulli: the posterior is Beta$(\\alpha_0+h,\\ \\beta_0+t)$ — updating is literally counting. Start with a strong prior (α₀=β₀=20) and flip a biased coin: the posterior resists at first, then data overwhelms it. **Notice the MLE and the posterior mean converge as n grows** — that is why the prior stops mattering with enough data, and why it matters enormously without it.',
  });
};

/* ---------- MLE ---------- */

V['mle-fit'] = (host) => {
  const r = rng(11);
  const data = Array.from({ length: 24 }, () => r.normal(1.2, 0.9));
  panel(host, {
    title: 'Maximum likelihood: slide until the data looks least surprising',
    height: 300,
    plot: { xlim: [-3, 5], ylim: [0, .7], xlabel: 'x' },
    controls: [
      { type: 'slider', key: 'mu', label: 'μ', min: -2, max: 4, step: .02, value: -0.6 },
      { type: 'slider', key: 'sd', label: 'σ', min: .2, max: 3, step: .02, value: 2 },
      { type: 'check', key: 'sticks', label: 'show per-point log-likelihood', value: true },
      {
        type: 'button', label: 'snap to MLE', onClick: (s) => {
          const m = data.reduce((a, b) => a + b, 0) / data.length;
          s.mu = m; s.sd = Math.sqrt(data.reduce((a, b) => a + (b - m) ** 2, 0) / data.length);
        },
      },
    ],
    draw(p, s, P) {
      const pdf = (x) => Math.exp(-((x - s.mu) ** 2) / (2 * s.sd ** 2)) / (s.sd * Math.sqrt(2 * Math.PI));
      p.setLim([-3, 5], [0, Math.max(.7, 1 / (s.sd * 2.5))]);
      p.clear().axes();
      p.clip();
      const pts = [];
      for (let x = -3; x <= 5; x += .02) pts.push([x, pdf(x)]);
      p.area(pts, { color: alpha(cssVar('--s1'), .18) });
      p.line(pts, { color: cssVar('--s1'), width: 2.4 });
      let ll = 0;
      for (const d of data) {
        const den = pdf(d);
        ll += Math.log(Math.max(den, 1e-300));
        if (s.sticks) p.line([[d, 0], [d, den]], { color: alpha(cssVar('--s2'), .45), width: 1 });
        p.points([[d, den]], { r: 3.2, color: cssVar('--s2') });
      }
      p.points(data.map((d) => [d, 0]), { r: 3, color: alpha(cssVar('--s2'), .8), shape: 'square' });
      p.clip(false);
      const m = data.reduce((a, b) => a + b, 0) / data.length;
      const v = data.reduce((a, b) => a + (b - m) ** 2, 0) / data.length;
      let llStar = 0;
      for (const d of data) llStar += Math.log(Math.exp(-((d - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v));
      P.readout({
        'log-likelihood Σ log p(xᵢ)': fmt(ll, 3),
        'best possible (MLE)': fmt(llStar, 3),
        'gap': fmt(llStar - ll, 3),
        'MLE μ̂': fmt(m, 3), 'MLE σ̂': fmt(Math.sqrt(v), 3),
      });
    },
    caption: 'Each orange stick is one data point\'s density under the current curve. Maximum likelihood slides and widens the bell to make the **product** of those heights as large as possible — equivalently, the sum of their logs. For a Gaussian, the answer is closed-form: $\\hat\\mu$ is the sample mean and $\\hat\\sigma^2$ the sample variance. Note the MLE variance is the *biased* one (divides by $n$, not $n-1$).',
  });
};

/* ---------- entropy / KL / cross-entropy ---------- */

V['entropy-kl'] = (host) => {
  panel(host, {
    title: 'Entropy, cross-entropy, KL — all in one picture',
    height: 300,
    plot: { xlim: [-.5, 5.5], ylim: [0, 1], xlabel: 'outcome', ylabel: 'probability' },
    controls: [
      { type: 'slider', key: 'pTemp', label: 'true p — spread', min: .15, max: 4, step: .05, value: 1 },
      { type: 'slider', key: 'qTemp', label: 'model q — spread', min: .15, max: 4, step: .05, value: 1 },
      { type: 'slider', key: 'qShift', label: 'model q — shift', min: -2, max: 2, step: .05, value: 0 },
    ],
    draw(p, s, P) {
      const n = 6;
      const base = [2.2, 1.4, .3, -.4, -1, -1.6];
      const P_ = LA.softmax(base.map((v) => v), s.pTemp);
      const Q_ = LA.softmax(base.map((v, i) => v + s.qShift * (i - 2.5)), s.qTemp);
      p.setLim([-.5, n - .5], [0, Math.max(...P_, ...Q_) * 1.25]);
      p.clear().axes({ nx: n });
      const bw = .38;
      for (let i = 0; i < n; i++) {
        p.bars([P_[i]], { x0: i - bw - .06, gap: 0, color: cssVar('--s1') });
        p.bars([Q_[i]], { x0: i + .06, gap: 0, color: cssVar('--s2') });
      }
      // redraw with proper widths
      p.clear().axes({ nx: n });
      const g = p.ctx;
      for (let i = 0; i < n; i++) {
        for (const [val, col, off] of [[P_[i], cssVar('--s1'), -.44], [Q_[i], cssVar('--s2'), .04]]) {
          const L = p.px(i + off), R = p.px(i + off + .4);
          g.fillStyle = col;
          g.fillRect(L, p.py(val), R - L, p.py(0) - p.py(val));
        }
      }
      p.legend([{ label: 'p  (truth)', color: cssVar('--s1') }, { label: 'q  (model)', color: cssVar('--s2') }], { pos: 'tr' });
      const H = -P_.reduce((a, v) => a + v * Math.log2(Math.max(v, 1e-12)), 0);
      const CE = -P_.reduce((a, v, i) => a + v * Math.log2(Math.max(Q_[i], 1e-12)), 0);
      const Hq = -Q_.reduce((a, v) => a + v * Math.log2(Math.max(v, 1e-12)), 0);
      P.readout({
        'H(p) entropy': fmt(H, 4) + ' bits',
        'H(p,q) cross-entropy': fmt(CE, 4) + ' bits',
        'KL(p‖q) = H(p,q) − H(p)': fmt(CE - H, 4) + ' bits',
        'H(q)': fmt(Hq, 4),
        'perplexity 2^H(p,q)': fmt(2 ** CE, 3),
      });
    },
    caption: 'Cross-entropy $H(p,q)$ is the *average bits* you pay coding samples from $p$ using a code built for $q$. It never goes below $H(p)$; the excess **is** the KL divergence. Training a classifier or a language model minimizes exactly this cross-entropy — and since $H(p)$ is fixed by the data, that is the same as minimizing $\\mathrm{KL}(p\\|q)$. Note it is asymmetric: make $q$ narrow where $p$ is wide and the penalty explodes.',
  });
};

/* ---------- convexity ---------- */

V['convexity'] = (host) => {
  panel(host, {
    title: 'Convex vs non-convex, and why anyone cares',
    height: 280,
    plot: { xlim: [-3, 3], ylim: [-1.5, 4], xlabel: 'θ', ylabel: 'L(θ)' },
    controls: [
      { type: 'select', key: 'fn', label: 'loss', value: 'cvx', options: [{ value: 'cvx', label: 'convex (θ²)' }, { value: 'cvx2', label: 'convex (|θ|)' }, { value: 'noncvx', label: 'non-convex (many minima)' }, { value: 'plateau', label: 'plateau + cliff' }] },
      { type: 'slider', key: 'a', label: 'point a', min: -3, max: 3, step: .05, value: -2 },
      { type: 'slider', key: 'b', label: 'point b', min: -3, max: 3, step: .05, value: 2.2 },
    ],
    draw(p, s, P) {
      const F = {
        cvx: (x) => .5 * x * x,
        cvx2: (x) => Math.abs(x) * 1.2,
        noncvx: (x) => .6 * Math.sin(3 * x) + .18 * x * x + .5,
        plateau: (x) => 1.6 / (1 + Math.exp(-4 * (x - .6))) + .05 * x * x,
      }[s.fn];
      p.clear().axes();
      p.clip();
      p.fn(F, { color: cssVar('--s1'), width: 2.6 });
      const fa = F(s.a), fb = F(s.b);
      p.line([[s.a, fa], [s.b, fb]], { color: cssVar('--s2'), width: 2, dash: [5, 3] });
      p.points([[s.a, fa], [s.b, fb]], { r: 5, color: cssVar('--s2') });
      // sample the chord vs function
      let viol = 0, maxGap = 0;
      for (let t = 0.02; t < 1; t += .01) {
        const x = s.a + (s.b - s.a) * t;
        const chord = fa + (fb - fa) * t;
        const gap = F(x) - chord;
        if (gap > 1e-9) { viol++; maxGap = Math.max(maxGap, gap); }
      }
      const mid = (s.a + s.b) / 2;
      p.line([[mid, F(mid)], [mid, fa + (fb - fa) * .5]], { color: viol ? cssVar('--danger') : cssVar('--s3'), width: 2 });
      p.clip(false);
      P.readout({
        'chord above curve everywhere?': viol ? 'NO — not convex here ✗' : 'yes ✓',
        'max violation': fmt(maxGap, 4),
        'verdict': viol ? 'local minima possible; where you start matters' : 'any local min is global',
      });
    },
    caption: 'A function is convex iff the straight line between *any* two points on it never dips below the curve. Drag `a` and `b` across the non-convex loss to find a violating pair. **Linear/logistic regression and SVMs are convex** — one minimum, solvers just work. **Neural nets are wildly non-convex** — and yet SGD finds good solutions anyway, a fact that took the field twenty years to stop worrying about.',
  });
};

/* ---------- expectation / variance / CLT ---------- */

V['clt'] = (host) => {
  panel(host, {
    title: 'The central limit theorem, live',
    height: 290,
    plot: { xlim: [-4, 4], ylim: [0, .6], xlabel: 'sample mean (standardized)', ylabel: 'density' },
    controls: [
      { type: 'select', key: 'src', label: 'source distribution', value: 'uniform', options: [{ value: 'uniform', label: 'uniform' }, { value: 'exp', label: 'exponential (skewed)' }, { value: 'bern', label: 'coin flip (discrete)' }, { value: 'bimodal', label: 'bimodal' }, { value: 'cauchy', label: 'Cauchy (heavy tail ✗)' }] },
      { type: 'slider', key: 'n', label: 'sample size n', min: 1, max: 60, step: 1, value: 1 },
      { type: 'slider', key: 'trials', label: 'number of samples', min: 200, max: 8000, step: 100, value: 3000 },
    ],
    draw(p, s, P) {
      const r = rng(3);
      const draw1 = {
        uniform: () => (r() - .5) * Math.sqrt(12),
        exp: () => -Math.log(r()) - 1,
        bern: () => (r() < .5 ? -1 : 1),
        bimodal: () => (r() < .5 ? r.normal(-1.6, .35) : r.normal(1.6, .35)) / 1.65,
        cauchy: () => Math.tan(Math.PI * (r() - .5)),
      }[s.src];
      const n = Math.round(s.n), T = Math.round(s.trials);
      const means = [];
      for (let t = 0; t < T; t++) {
        let acc = 0;
        for (let i = 0; i < n; i++) acc += draw1();
        means.push(acc / Math.sqrt(n));  // standardized: var stays 1
      }
      const bins = 61, lo = -4, hi = 4;
      const hist = new Array(bins).fill(0);
      let outl = 0;
      for (const m of means) {
        const k = Math.floor((m - lo) / (hi - lo) * bins);
        if (k >= 0 && k < bins) hist[k]++; else outl++;
      }
      const wid = (hi - lo) / bins;
      const dens = hist.map((c) => c / (T * wid));
      p.setLim([lo, hi], [0, Math.max(.55, Math.max(...dens) * 1.15)]);
      p.clear().axes();
      p.clip();
      const g = p.ctx;
      for (let i = 0; i < bins; i++) {
        const L = p.px(lo + i * wid), R = p.px(lo + (i + 1) * wid);
        g.fillStyle = alpha(cssVar('--s1'), .75);
        g.fillRect(L, p.py(dens[i]), Math.max(R - L - .5, 1), p.py(0) - p.py(dens[i]));
      }
      p.fn((x) => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI), { color: cssVar('--s2'), width: 2.4 });
      p.clip(false);
      p.legend([{ label: `mean of n=${n}`, color: cssVar('--s1') }, { label: 'standard normal', color: cssVar('--s2') }], { pos: 'tr' });
      P.readout({
        n, 'samples': T,
        'off-chart (|z|>4)': outl + (s.src === 'cauchy' ? '  ← heavy tails' : ''),
        'note': s.src === 'cauchy' ? 'Cauchy has no finite mean — CLT does not apply' : 'converging to Normal',
      });
    },
    caption: 'Slide **n** from 1 upward. Whatever the source shape — skewed, discrete, bimodal — the distribution of the mean marches toward a Gaussian at rate $1/\\sqrt{n}$. This is why noise in ML is so often modeled as Gaussian, and why minibatch gradients (averages!) are approximately Gaussian around the true gradient. The Cauchy option shows the fine print: **the theorem needs finite variance.**',
  });
};

/* ---------- matrix calculus: Jacobian ---------- */

V['jacobian'] = (host) => {
  panel(host, {
    title: 'The Jacobian is the local linear map',
    height: 320,
    plot: { xlim: [-2.2, 2.2], ylim: [-1.8, 1.8], equal: true },
    controls: [
      { type: 'select', key: 'fn', label: 'f: ℝ² → ℝ²', value: 'polar', options: [{ value: 'polar', label: '(x²−y², 2xy)' }, { value: 'sin', label: '(sin x, y + x²/2)' }, { value: 'tanh', label: '(tanh x, tanh y) — a layer' }] },
      { type: 'slider', key: 'x0', label: 'x₀', min: -2, max: 2, step: .02, value: .7 },
      { type: 'slider', key: 'y0', label: 'y₀', min: -1.6, max: 1.6, step: .02, value: .5 },
      { type: 'slider', key: 'eps', label: 'neighborhood size', min: .05, max: 1, step: .01, value: .35 },
    ],
    draw(p, s, P) {
      const F = {
        polar: (x, y) => [x * x - y * y, 2 * x * y],
        sin: (x, y) => [Math.sin(x), y + x * x / 2],
        tanh: (x, y) => [Math.tanh(x), Math.tanh(y)],
      }[s.fn];
      const e = 1e-5;
      const J = [
        [(F(s.x0 + e, s.y0)[0] - F(s.x0 - e, s.y0)[0]) / (2 * e), (F(s.x0, s.y0 + e)[0] - F(s.x0, s.y0 - e)[0]) / (2 * e)],
        [(F(s.x0 + e, s.y0)[1] - F(s.x0 - e, s.y0)[1]) / (2 * e), (F(s.x0, s.y0 + e)[1] - F(s.x0, s.y0 - e)[1]) / (2 * e)],
      ];
      p.clear().axes({ nx: 5, ny: 4 });
      p.clip();
      const f0 = F(s.x0, s.y0);
      // exact image of a small circle vs Jacobian's linear prediction
      const exact = [], lin = [];
      for (let k = 0; k <= 64; k++) {
        const th = k / 64 * 6.2832;
        const dx = Math.cos(th) * s.eps, dy = Math.sin(th) * s.eps;
        const fe = F(s.x0 + dx, s.y0 + dy);
        exact.push([fe[0] - f0[0] + s.x0, fe[1] - f0[1] + s.y0]);
        lin.push([J[0][0] * dx + J[0][1] * dy + s.x0, J[1][0] * dx + J[1][1] * dy + s.y0]);
      }
      // input circle
      const inc = [];
      for (let k = 0; k <= 64; k++) inc.push([s.x0 + Math.cos(k / 64 * 6.2832) * s.eps, s.y0 + Math.sin(k / 64 * 6.2832) * s.eps]);
      p.line(inc, { color: alpha(cssVar('--text-faint'), .8), width: 1.4, dash: [4, 3] });
      p.line(exact, { color: cssVar('--s1'), width: 2.4 });
      p.line(lin, { color: cssVar('--s2'), width: 2, dash: [6, 3] });
      p.points([[s.x0, s.y0]], { r: 4.5, color: cssVar('--s3') });
      p.clip(false);
      p.legend([
        { label: 'input neighborhood', color: cssVar('--text-faint'), dash: true },
        { label: 'exact image Δf', color: cssVar('--s1') },
        { label: 'Jacobian prediction JΔx', color: cssVar('--s2'), dash: true },
      ], { pos: 'tl' });
      const det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
      P.readout({
        'J': `[[${fmt(J[0][0], 2)}, ${fmt(J[0][1], 2)}], [${fmt(J[1][0], 2)}, ${fmt(J[1][1], 2)}]]`,
        'det J': fmt(det, 3),
        'volume change': `${fmt(Math.abs(det), 3)}×`,
      });
    },
    caption: 'Shrink the neighborhood and the exact image (solid) converges onto the Jacobian\'s linear prediction (dashed). **That is all a Jacobian is**: the matrix of the best local linear approximation. Backprop through a layer is a Jacobian-vector product — and it never builds the matrix, it just applies it. $|\\det J|$ tells you how the layer expands or contracts volume, which is the entire basis of normalizing flows.',
  });
};

/* ---------- condition number / numerical precision ---------- */

V['float-precision'] = (host) => {
  panel(host, {
    title: 'Where your bits go: fp32 vs fp16 vs bf16',
    height: 260,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    noCanvas: false,
    controls: [
      { type: 'slider', key: 'val', label: 'value to represent (log₁₀)', min: -10, max: 10, step: .1, value: 0 },
    ],
    draw(p, s, P) {
      const formats = [
        { n: 'fp32', exp: 8, man: 23, max: 3.4e38, min: 1.2e-38, eps: 1.2e-7 },
        { n: 'tf32', exp: 8, man: 10, max: 3.4e38, min: 1.2e-38, eps: 9.8e-4 },
        { n: 'bf16', exp: 8, man: 7, max: 3.4e38, min: 1.2e-38, eps: 7.8e-3 },
        { n: 'fp16', exp: 5, man: 10, max: 65504, min: 6.1e-5, eps: 9.8e-4 },
        { n: 'fp8 e4m3', exp: 4, man: 3, max: 448, min: 1.5e-3, eps: 0.125 },
      ];
      p.clear();
      const g = p.ctx;
      const W = p.w, rowH = 40, x0 = 66;
      const v = 10 ** s.val;
      formats.forEach((f, i) => {
        const y = 26 + i * rowH;
        g.fillStyle = cssVar('--text');
        g.font = '600 12px ui-monospace, monospace';
        g.textAlign = 'left'; g.textBaseline = 'middle';
        g.fillText(f.n, 8, y);
        // bit layout: sign | exponent | mantissa
        const total = 1 + f.exp + f.man;
        const bw = (W - x0 - 100) / 32;
        let x = x0;
        const seg = [[1, cssVar('--s6'), 'S'], [f.exp, cssVar('--s2'), 'exponent (range)'], [f.man, cssVar('--s1'), 'mantissa (precision)']];
        for (const [n, col, lab] of seg) {
          g.fillStyle = col;
          g.fillRect(x, y - 9, n * bw - 1.5, 18);
          if (n > 3) {
            g.fillStyle = cssVar('--bg');
            g.font = '9px -apple-system, sans-serif';
            g.textAlign = 'center';
            g.fillText(lab.split(' ')[0], x + n * bw / 2, y);
            g.textAlign = 'left';
          }
          x += n * bw;
        }
        // representability
        const ok = v <= f.max && v >= f.min;
        g.fillStyle = ok ? cssVar('--ok') : cssVar('--danger');
        g.font = '10.5px ui-monospace, monospace';
        g.fillText(ok ? `rel. err ≤ ${f.eps.toExponential(1)}` : (v > f.max ? 'OVERFLOW → inf' : 'UNDERFLOW → 0'), x + 10, y);
      });
      g.fillStyle = cssVar('--text-dim');
      g.font = '11px -apple-system, sans-serif';
      g.fillText(`representing ${v.toExponential(2)}`, 8, p.h - 10);
      P.readout({
        'value': v.toExponential(3),
        'fp16 max': '65 504', 'bf16 max': '3.4×10³⁸',
        'why bf16 won': 'same exponent range as fp32 → no loss scaling needed',
      });
    },
    caption: 'Slide the value past $10^5$: **fp16 overflows** while bf16 is fine — bf16 simply truncated fp32\'s mantissa and kept all 8 exponent bits. That single design choice is why bf16 became the default for training large models: you get half the memory without the loss-scaling gymnastics fp16 requires. The price is precision, which matters for optimizer state — hence master weights in fp32.',
  });
};

/* ---------- eigen / PSD quadratic form ---------- */

V['quadratic-form'] = (host) => {
  panel(host, {
    title: 'Quadratic forms, curvature, and conditioning',
    height: 330,
    plot: { xlim: [-3, 3], ylim: [-2.4, 2.4], equal: true, xlabel: 'θ₁', ylabel: 'θ₂' },
    controls: [
      { type: 'slider', key: 'a', label: 'H₁₁', min: -2, max: 4, step: .05, value: 1 },
      { type: 'slider', key: 'b', label: 'H₁₂ = H₂₁', min: -2, max: 2, step: .05, value: 0.6 },
      { type: 'slider', key: 'd', label: 'H₂₂', min: -2, max: 4, step: .05, value: 2 },
      { type: 'check', key: 'newton', label: 'show Newton step from a point', value: true },
    ],
    interact: (s, P) => ({
      hover(x, y) { s.cx = x; s.cy = y; P.redraw(); },
    }),
    draw(p, s, P) {
      const H = [[s.a, s.b], [s.b, s.d]];
      const f = (x, y) => .5 * (H[0][0] * x * x + 2 * H[0][1] * x * y + H[1][1] * y * y);
      p.clear();
      p.heat(f, { step: 5, cmap: divergeMap });
      p.axes({ nx: 5, ny: 4 });
      p.clip();
      const lv = [];
      for (let i = 1; i <= 10; i++) lv.push((i / 10) ** 2 * 6);
      for (let i = 1; i <= 5; i++) lv.push(-((i / 5) ** 2) * 3);
      p.contour(f, lv, { color: alpha(cssVar('--text'), .25) });
      const { vals, vecs } = LA.eig2(H);
      const cols = [cssVar('--s3'), cssVar('--s6')];
      vals.forEach((l, i) => {
        const v = vecs[i];
        p.arrow(0, 0, v[0] * 1.6, v[1] * 1.6, { color: cols[i], width: 2.6 });
        p.text(v[0] * 1.75, v[1] * 1.75, `λ${i + 1}=${fmt(l, 2)}`, { color: cols[i], size: 11, weight: '600', align: 'center' });
      });
      if (s.newton && s.cx != null) {
        const gx = H[0][0] * s.cx + H[0][1] * s.cy, gy = H[1][0] * s.cx + H[1][1] * s.cy;
        const ng = Math.hypot(gx, gy) || 1;
        p.arrow(s.cx, s.cy, s.cx - gx / ng * .8, s.cy - gy / ng * .8, { color: cssVar('--s2'), width: 2.4 });
        const step = LA.solve(H.map((r) => r.slice()), [gx, gy]);
        if (isFinite(step[0])) p.arrow(s.cx, s.cy, s.cx - step[0], s.cy - step[1], { color: cssVar('--s5'), width: 2.4, dash: [5, 3] });
        p.circle(s.cx, s.cy, 4, { fill: cssVar('--text') });
      }
      p.clip(false);
      p.legend([
        { label: '−∇L (gradient step)', color: cssVar('--s2') },
        { label: '−H⁻¹∇L (Newton step)', color: cssVar('--s5'), dash: true },
      ], { pos: 'bl' });
      const kappa = Math.abs(vals[0] / vals[1]);
      P.readout({
        'λ₁, λ₂': `${fmt(vals[0], 3)}, ${fmt(vals[1], 3)}`,
        'definiteness': vals[1] > 0 ? 'positive definite → bowl (min)' : vals[0] < 0 ? 'negative definite → dome (max)' : 'indefinite → saddle',
        'condition κ = λ₁/λ₂': isFinite(kappa) ? fmt(kappa, 2) : '∞',
        'GD steps ∝ κ': isFinite(kappa) ? fmt(kappa, 0) : '∞',
      });
    },
    caption: 'The Hessian eigenvectors are the principal axes of the contours; the eigenvalues are the curvatures along them. Make $\\lambda_1 \\gg \\lambda_2$ and the bowl becomes a **taco** — gradient descent zig-zags because $-\\nabla L$ (orange) points across the valley, not down it. The Newton step $-H^{-1}\\nabla L$ (yellow) aims straight at the minimum. Flip $H_{22}$ negative to make a saddle: the eigenvalue signs tell you everything.',
  });
};

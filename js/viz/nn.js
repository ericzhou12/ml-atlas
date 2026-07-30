/* ============================================================
   viz/nn.js — neural network figures.
   Includes a real MLP trained in-browser with hand-written backprop.
   ============================================================ */

import { panel } from '../ui.js';
import { cssVar, alpha, mix, seqMap, divergeMap, LA, rng, fmt, SERIES } from '../plot.js';

const V = {};
export default V;

/* ---------- activation zoo ---------- */

export const ACT = {
  relu: { f: (x) => Math.max(0, x), d: (x) => (x > 0 ? 1 : 0), name: 'ReLU', tex: '\\max(0,x)' },
  leaky: { f: (x) => (x > 0 ? x : .1 * x), d: (x) => (x > 0 ? 1 : .1), name: 'Leaky ReLU', tex: '\\max(0.1x, x)' },
  gelu: {
    f: (x) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x ** 3))),
    d: (x) => { const e = 1e-4; return (ACT.gelu.f(x + e) - ACT.gelu.f(x - e)) / (2 * e); },
    name: 'GELU', tex: 'x\\,\\Phi(x)',
  },
  silu: {
    f: (x) => x / (1 + Math.exp(-x)),
    d: (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 + x * (1 - s)); },
    name: 'SiLU / Swish', tex: 'x\\,\\sigma(x)',
  },
  tanh: { f: Math.tanh, d: (x) => 1 - Math.tanh(x) ** 2, name: 'tanh', tex: '\\tanh x' },
  sigmoid: { f: (x) => 1 / (1 + Math.exp(-x)), d: (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); }, name: 'sigmoid', tex: '\\sigma(x)' },
  softplus: { f: (x) => Math.log1p(Math.exp(Math.min(x, 30))), d: (x) => 1 / (1 + Math.exp(-x)), name: 'softplus', tex: '\\log(1+e^x)' },
  identity: { f: (x) => x, d: () => 1, name: 'identity (linear)', tex: 'x' },
};

V['activations'] = (host) => {
  panel(host, {
    title: 'Activation functions and their gradients',
    height: 300,
    plot: { xlim: [-4, 4], ylim: [-1.5, 3], xlabel: 'x' },
    controls: [
      { type: 'select', key: 'a', label: 'activation', value: 'relu', options: Object.keys(ACT).map((k) => ({ value: k, label: ACT[k].name })) },
      { type: 'check', key: 'grad', label: 'show derivative', value: true },
      { type: 'check', key: 'all', label: 'overlay all', value: false },
    ],
    draw(p, s, P) {
      p.clear().axes();
      p.clip();
      if (s.all) {
        const cols = SERIES();
        Object.keys(ACT).slice(0, 6).forEach((k, i) => p.fn(ACT[k].f, { color: cols[i], width: 2 }));
        p.clip(false);
        p.legend(Object.keys(ACT).slice(0, 6).map((k, i) => ({ label: ACT[k].name, color: SERIES()[i] })), { pos: 'tl' });
        P.readout({ note: 'all modern choices are ≈linear for large positive x and ≈0 for large negative x' });
        return;
      }
      const A = ACT[s.a];
      p.fn(A.f, { color: cssVar('--s1'), width: 2.8 });
      if (s.grad) p.fn(A.d, { color: cssVar('--s2'), width: 2, dash: [5, 3] });
      p.clip(false);
      p.legend([{ label: A.name, color: cssVar('--s1') }, ...(s.grad ? [{ label: 'derivative', color: cssVar('--s2'), dash: true }] : [])], { pos: 'tl' });
      // dead-zone / saturation diagnostics
      let deadFrac = 0, samples = 0;
      for (let x = -4; x <= 4; x += .01) { if (Math.abs(A.d(x)) < .01) deadFrac++; samples++; }
      P.readout({
        'max derivative': fmt(Math.max(...Array.from({ length: 801 }, (_, i) => A.d(-4 + i * .01))), 3),
        'fraction of inputs with ~zero gradient': fmt(deadFrac / samples * 100, 1) + '%',
        'zero-centered output': ['tanh', 'identity'].includes(s.a) ? 'yes' : 'no',
        'smooth at 0': ['relu', 'leaky'].includes(s.a) ? 'no (kink)' : 'yes',
      });
    },
    caption: 'The derivative is what actually matters — it is the multiplier backprop applies. **Sigmoid and tanh saturate**: past $|x|\\approx 4$ the gradient is effectively zero and learning stops. **ReLU** has a perfect gradient of 1 on the positive side (which is why it made deep nets trainable) but exactly 0 on the negative side — a unit pushed there can die permanently. **GELU and SiLU** smooth that corner and let a little gradient through, which is why transformers use them.',
  });
};

/* ---------- MLP playground: real training in the browser ---------- */

function makeNet(sizes, act, seed, initScale = 1) {
  const r = rng(seed);
  const W = [], b = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const fanIn = sizes[l];
    // He init for ReLU-family, Xavier otherwise
    const g = ['relu', 'leaky'].includes(act) ? Math.sqrt(2 / fanIn) : Math.sqrt(1 / fanIn);
    W.push(Array.from({ length: sizes[l + 1] }, () => Array.from({ length: sizes[l] }, () => r.normal(0, g * initScale))));
    b.push(new Array(sizes[l + 1]).fill(0));
  }
  return { W, b, sizes, act };
}

function forward(net, x) {
  const A = ACT[net.act];
  const acts = [x];
  const pre = [];
  let h = x;
  for (let l = 0; l < net.W.length; l++) {
    const z = net.W[l].map((row, i) => row.reduce((s, w, j) => s + w * h[j], 0) + net.b[l][i]);
    pre.push(z);
    h = l === net.W.length - 1 ? z : z.map(A.f);
    acts.push(h);
  }
  return { acts, pre, out: h };
}

/** One SGD step on binary cross-entropy over a minibatch. Returns loss. */
function trainStep(net, batch, lr, l2 = 0) {
  const A = ACT[net.act];
  const L = net.W.length;
  const gW = net.W.map((m) => m.map((r) => r.map(() => 0)));
  const gb = net.b.map((v) => v.map(() => 0));
  let loss = 0;
  for (const { x, y } of batch) {
    const { acts, pre, out } = forward(net, x);
    const p = 1 / (1 + Math.exp(-out[0]));
    loss += -(y * Math.log(Math.max(p, 1e-12)) + (1 - y) * Math.log(Math.max(1 - p, 1e-12)));
    // dL/dz at output (sigmoid + BCE collapses beautifully)
    let delta = [p - y];
    for (let l = L - 1; l >= 0; l--) {
      for (let i = 0; i < net.W[l].length; i++) {
        gb[l][i] += delta[i];
        for (let j = 0; j < net.W[l][i].length; j++) gW[l][i][j] += delta[i] * acts[l][j];
      }
      if (l > 0) {
        const prev = new Array(net.sizes[l]).fill(0);
        for (let j = 0; j < net.sizes[l]; j++) {
          let s = 0;
          for (let i = 0; i < net.W[l].length; i++) s += net.W[l][i][j] * delta[i];
          prev[j] = s * A.d(pre[l - 1][j]);
        }
        delta = prev;
      }
    }
  }
  const m = batch.length;
  for (let l = 0; l < L; l++) {
    for (let i = 0; i < net.W[l].length; i++) {
      net.b[l][i] -= lr * gb[l][i] / m;
      for (let j = 0; j < net.W[l][i].length; j++) {
        net.W[l][i][j] -= lr * (gW[l][i][j] / m + l2 * net.W[l][i][j]);
      }
    }
  }
  return loss / m;
}

function genData(kind, n, noise, seed) {
  const r = rng(seed);
  const out = [];
  for (let i = 0; i < n; i++) {
    let x, y, c;
    if (kind === 'moons') {
      c = i % 2;
      const t = r() * Math.PI;
      if (c) { x = 1 - Math.cos(t) - .5; y = -Math.sin(t) + .35; } else { x = Math.cos(t) - .5; y = Math.sin(t) - .35; }
      x = x * 1.6; y = y * 1.6;
    } else if (kind === 'circles') {
      c = i % 2;
      const t = r() * 6.2832, rad = (c ? 1.9 : .8);
      x = Math.cos(t) * rad; y = Math.sin(t) * rad;
    } else if (kind === 'xor') {
      x = (r() - .5) * 4; y = (r() - .5) * 4; c = (x > 0) === (y > 0) ? 1 : 0;
    } else if (kind === 'spiral') {
      c = i % 2;
      const t = (i / n) * 3.6 + .4;
      const sg = c ? 1 : -1;
      x = sg * t * Math.cos(t * 1.6) * .62; y = sg * t * Math.sin(t * 1.6) * .62;
    } else { // blobs
      c = i % 2;
      x = (c ? 1.2 : -1.2) + r.normal(0, .55); y = (c ? .8 : -.8) + r.normal(0, .55);
    }
    out.push({ x: [x + r.normal(0, noise), y + r.normal(0, noise)], y: c });
  }
  return out;
}

V['mlp-playground'] = (host, params = {}) => {
  let net, data, testData, step = 0, lossHist = [], accHist = [];
  const build = (s) => {
    const hidden = [];
    for (let i = 0; i < Math.round(s.layers); i++) hidden.push(Math.round(s.width));
    net = makeNet([2, ...hidden, 1], s.act, Math.round(s.seed), s.initScale);
    data = genData(s.data, 120, s.noise, 3);
    testData = genData(s.data, 120, s.noise, 77);
    step = 0; lossHist = []; accHist = [];
  };

  panel(host, {
    title: 'Train a neural net, here, now',
    height: 340,
    plot: { xlim: [-3, 3], ylim: [-2.4, 2.4], equal: true },
    state: { seed: 1 },
    controls: [
      { type: 'select', key: 'data', label: 'dataset', value: params.data || 'moons', options: [{ value: 'blobs', label: 'blobs' }, { value: 'moons', label: 'moons' }, { value: 'circles', label: 'circles' }, { value: 'xor', label: 'XOR' }, { value: 'spiral', label: 'spirals' }], onChange: (v, P) => build(P.state) },
      { type: 'slider', key: 'layers', label: 'hidden layers', min: 0, max: 4, step: 1, value: params.layers || 2, onChange: (v, P) => build(P.state) },
      { type: 'slider', key: 'width', label: 'units per layer', min: 1, max: 16, step: 1, value: params.width || 8, onChange: (v, P) => build(P.state) },
      { type: 'select', key: 'act', label: 'activation', value: params.act || 'relu', options: ['relu', 'tanh', 'gelu', 'sigmoid', 'identity'].map((k) => ({ value: k, label: ACT[k].name })), onChange: (v, P) => build(P.state) },
      { type: 'slider', key: 'lr', label: 'learning rate', min: .005, max: 2, step: .005, value: .5 },
      { type: 'slider', key: 'noise', label: 'data noise', min: 0, max: .5, step: .01, value: .1, onChange: (v, P) => build(P.state) },
      { type: 'slider', key: 'l2', label: 'weight decay', min: 0, max: .02, step: .0005, value: 0, fmt: (v) => v.toFixed(4) },
      { type: 'slider', key: 'initScale', label: 'init scale', min: .05, max: 3, step: .05, value: 1, onChange: (v, P) => build(P.state) },
      { type: 'play' },
      { type: 'button', label: '↻ re-init', onClick: (s) => { s.seed = Math.floor(Math.random() * 999); build(s); } },
    ],
    init: (P, s) => build(s),
    animate(s, P) {
      for (let k = 0; k < 6; k++) {
        // minibatch
        const batch = [];
        for (let i = 0; i < 24; i++) batch.push(data[Math.floor(Math.random() * data.length)]);
        const l = trainStep(net, batch, s.lr, s.l2);
        step++;
        if (step % 4 === 0) {
          lossHist.push(l);
          if (lossHist.length > 260) lossHist.shift();
        }
      }
    },
    draw(p, s, P) {
      p.clear();
      const pred = (x, y) => {
        const o = forward(net, [x, y]).out[0];
        return 1 / (1 + Math.exp(-o));
      };
      p.heat(pred, {
        step: 5, lo: 0, hi: 1,
        cmap: (t) => mix(mix(cssVar('--bg-inset'), cssVar('--s1'), .5), mix(cssVar('--bg-inset'), cssVar('--s2'), .5), t),
      });
      p.axes({ nx: 5, ny: 4 });
      p.clip();
      p.contour((x, y) => pred(x, y) - .5, [0], { color: cssVar('--text'), width: 2 });
      p.points(data.map((d) => d.x), { r: 4, color: (_, i) => (data[i].y ? cssVar('--s2') : cssVar('--s1')), stroke: alpha(cssVar('--bg-inset'), .9), strokeWidth: 1.2 });
      p.clip(false);

      // inset loss curve
      if (lossHist.length > 2) {
        const g = p.ctx;
        const W = 120, H = 46, X = p.w - p.pad.r - W - 8, Y = p.pad.t + 8;
        g.save();
        g.fillStyle = alpha(cssVar('--bg-elev'), .92);
        g.strokeStyle = cssVar('--border');
        g.fillRect(X, Y, W, H); g.strokeRect(X, Y, W, H);
        const mx = Math.max(...lossHist), mn = Math.min(...lossHist);
        g.beginPath();
        lossHist.forEach((l, i) => {
          const px = X + i / (lossHist.length - 1) * W;
          const py = Y + H - 4 - (l - mn) / Math.max(mx - mn, 1e-6) * (H - 10);
          i ? g.lineTo(px, py) : g.moveTo(px, py);
        });
        g.strokeStyle = cssVar('--s3'); g.lineWidth = 1.6; g.stroke();
        g.fillStyle = cssVar('--text-faint');
        g.font = '9px ui-monospace, monospace';
        g.fillText('training loss', X + 4, Y + 10);
        g.restore();
      }

      const acc = (D) => D.filter((d) => (pred(d.x[0], d.x[1]) > .5) === !!d.y).length / D.length;
      const nParams = net.W.reduce((a, m) => a + m.length * m[0].length + m.length, 0);
      P.readout({
        'steps': step,
        'parameters': nParams,
        'loss': lossHist.length ? fmt(lossHist[lossHist.length - 1], 4) : '—',
        'train acc': fmt(acc(data) * 100, 1) + '%',
        'test acc': fmt(acc(testData) * 100, 1) + '%',
        'architecture': `2 → ${net.sizes.slice(1, -1).join(' → ') || '(none)'} → 1`,
      });
    },
    caption: 'This is a real network — forward pass, backprop, and SGD all running in your browser. Things worth trying: **(1)** set hidden layers to 0 (or activation to *identity*) and try spirals — a linear model cannot do it, no matter how long you wait. **(2)** Set learning rate to 2.0 and watch training destabilize. **(3)** Set init scale to 0.05 with 4 layers — the signal vanishes and learning stalls. **(4)** Fit spirals with width 16, then check the train/test gap.',
  });
};

/* ---------- what one hidden unit does ---------- */

V['hidden-units'] = (host) => {
  panel(host, {
    title: 'How bumps are built from ReLUs',
    height: 300,
    plot: { xlim: [-4, 4], ylim: [-2, 3], xlabel: 'x', ylabel: 'output' },
    controls: [
      { type: 'slider', key: 'n', label: 'hidden units', min: 1, max: 12, step: 1, value: 4 },
      { type: 'slider', key: 'seed', label: 'random weights', min: 1, max: 40, step: 1, value: 7 },
      { type: 'check', key: 'parts', label: 'show individual units', value: true },
      { type: 'select', key: 'act', label: 'activation', value: 'relu', options: ['relu', 'tanh', 'gelu', 'sigmoid'].map((k) => ({ value: k, label: ACT[k].name })) },
    ],
    draw(p, s, P) {
      const r = rng(Math.round(s.seed));
      const n = Math.round(s.n);
      const A = ACT[s.act];
      const units = Array.from({ length: n }, () => ({ w: r.normal(0, 2.2), b: r.normal(0, 2.4), v: r.normal(0, 1.4) }));
      p.clear().axes();
      p.clip();
      const cols = SERIES();
      if (s.parts) units.forEach((u, i) => p.fn((x) => u.v * A.f(u.w * x + u.b), { color: alpha(cols[i % cols.length], .45), width: 1.4 }));
      p.fn((x) => units.reduce((a, u) => a + u.v * A.f(u.w * x + u.b), 0) / Math.sqrt(n), { color: cssVar('--text'), width: 2.8 });
      p.clip(false);
      P.readout({
        'units': n,
        'kinks in the sum': s.act === 'relu' ? n + ' (one per unit)' : 'smooth',
        'universal approximation': 'any continuous function on a bounded interval, given enough units',
      });
    },
    caption: 'A ReLU unit is a hinge: flat, then a straight ramp starting wherever $wx+b$ crosses zero. Add several with different slopes and offsets and you get a **piecewise-linear function with one kink per unit**. That is the whole mechanism behind universal approximation — with enough hinges you can trace any curve. The theorem says one hidden layer suffices; it does not say the width is reasonable, which is why we go deep instead.',
  });
};

/* ---------- backprop computational graph ---------- */

V['backprop-graph'] = (host) => {
  panel(host, {
    title: 'Backpropagation on a concrete graph',
    height: 320,
    noCanvas: false,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'x', label: 'x', min: -2, max: 2, step: .1, value: 1 },
      { type: 'slider', key: 'w1', label: 'w₁', min: -2, max: 2, step: .1, value: 1.5 },
      { type: 'slider', key: 'w2', label: 'w₂', min: -2, max: 2, step: .1, value: -1 },
      { type: 'slider', key: 'y', label: 'target y', min: -2, max: 2, step: .1, value: 1 },
      { type: 'check', key: 'back', label: 'show backward pass', value: true },
    ],
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const W = p.w, H = p.h;
      // forward values
      const a = s.w1 * s.x;
      const h = Math.tanh(a);
      const o = s.w2 * h;
      const L = (o - s.y) ** 2;
      // backward
      const dL_do = 2 * (o - s.y);
      const dL_dw2 = dL_do * h;
      const dL_dh = dL_do * s.w2;
      const dL_da = dL_dh * (1 - h * h);
      const dL_dw1 = dL_da * s.x;
      const dL_dx = dL_da * s.w1;

      const nodes = [
        { id: 'x', x: .07, y: .28, label: 'x', val: s.x, grad: dL_dx },
        { id: 'w1', x: .07, y: .72, label: 'w₁', val: s.w1, grad: dL_dw1, param: true },
        { id: 'a', x: .3, y: .5, label: '×', val: a, grad: dL_da, op: 'a = w₁x' },
        { id: 'h', x: .5, y: .5, label: 'tanh', val: h, grad: dL_dh, op: 'h = tanh(a)' },
        { id: 'w2', x: .5, y: .84, label: 'w₂', val: s.w2, grad: dL_dw2, param: true },
        { id: 'o', x: .72, y: .5, label: '×', val: o, grad: dL_do, op: 'o = w₂h' },
        { id: 'L', x: .92, y: .5, label: 'L', val: L, grad: 1, op: 'L = (o−y)²' },
      ];
      const edges = [['x', 'a'], ['w1', 'a'], ['a', 'h'], ['h', 'o'], ['w2', 'o'], ['o', 'L']];
      const N = Object.fromEntries(nodes.map((n) => [n.id, n]));
      const PX = (n) => n.x * W, PY = (n) => n.y * (H - 30) + 12;

      // edges
      for (const [u, v] of edges) {
        const A = N[u], B = N[v];
        g.strokeStyle = alpha(cssVar('--s1'), .7); g.lineWidth = 1.8;
        g.beginPath(); g.moveTo(PX(A) + 20, PY(A)); g.lineTo(PX(B) - 20, PY(B)); g.stroke();
        // arrowhead
        const ang = Math.atan2(PY(B) - PY(A), PX(B) - PX(A));
        g.fillStyle = alpha(cssVar('--s1'), .7);
        g.beginPath();
        g.moveTo(PX(B) - 20, PY(B));
        g.lineTo(PX(B) - 20 - 7 * Math.cos(ang - .4), PY(B) - 7 * Math.sin(ang - .4));
        g.lineTo(PX(B) - 20 - 7 * Math.cos(ang + .4), PY(B) - 7 * Math.sin(ang + .4));
        g.closePath(); g.fill();
        if (s.back) {
          g.strokeStyle = alpha(cssVar('--s2'), .75); g.lineWidth = 1.6;
          g.setLineDash([4, 3]);
          g.beginPath();
          g.moveTo(PX(B) - 20, PY(B) + 9); g.lineTo(PX(A) + 20, PY(A) + 9);
          g.stroke(); g.setLineDash([]);
        }
      }
      // nodes
      for (const n of nodes) {
        const cx = PX(n), cy = PY(n);
        g.beginPath(); g.arc(cx, cy, 19, 0, 6.2832);
        g.fillStyle = n.param ? alpha(cssVar('--s4'), .3) : n.id === 'L' ? alpha(cssVar('--s2'), .3) : cssVar('--bg-elev-2');
        g.fill();
        g.strokeStyle = n.param ? cssVar('--s4') : cssVar('--border'); g.lineWidth = 1.6; g.stroke();
        g.fillStyle = cssVar('--text');
        g.font = '600 12px -apple-system, sans-serif';
        g.textAlign = 'center'; g.textBaseline = 'middle';
        g.fillText(n.label, cx, cy);
        g.font = '10.5px ui-monospace, monospace';
        g.fillStyle = cssVar('--s3');
        g.fillText(fmt(n.val, 3), cx, cy - 30);
        if (s.back) {
          g.fillStyle = cssVar('--s2');
          g.fillText('∂L/∂ = ' + fmt(n.grad, 3), cx, cy + 32);
        }
      }
      g.fillStyle = cssVar('--text-faint');
      g.font = '10px -apple-system, sans-serif';
      g.textAlign = 'left';
      g.fillText('forward: values →', 8, H - 8);
      if (s.back) { g.fillStyle = cssVar('--s2'); g.fillText('← backward: gradients', 130, H - 8); }

      P.readout({
        'L': fmt(L, 4),
        '∂L/∂w₁ = ∂L/∂a · x': `${fmt(dL_da, 3)} × ${fmt(s.x, 2)} = ${fmt(dL_dw1, 4)}`,
        '∂L/∂w₂ = ∂L/∂o · h': `${fmt(dL_do, 3)} × ${fmt(h, 3)} = ${fmt(dL_dw2, 4)}`,
      });
    },
    caption: 'Green numbers flow forward and are **cached**; orange numbers flow backward. Each node needs only two things: its local derivative, and the gradient arriving from downstream — it multiplies them and passes the result on. Set $w_1$ to 2 and $x$ to 2: $\\tanh$ saturates, its local derivative $1-h^2$ collapses, and **every gradient upstream of it dies**. That single multiplication is the vanishing gradient problem.',
  });
};

/* ---------- vanishing / exploding gradients across depth ---------- */

V['vanishing-gradients'] = (host) => {
  panel(host, {
    title: 'Signal and gradient magnitude across depth',
    height: 300,
    plot: { xlim: [0, 30], ylim: [-8, 4], xlabel: 'layer', ylabel: 'log₁₀ magnitude' },
    controls: [
      { type: 'select', key: 'act', label: 'activation', value: 'tanh', options: ['tanh', 'sigmoid', 'relu', 'identity'].map((k) => ({ value: k, label: ACT[k].name })) },
      { type: 'slider', key: 'gain', label: 'init std multiplier', min: .2, max: 2.5, step: .02, value: 1 },
      { type: 'slider', key: 'width', label: 'layer width', min: 8, max: 256, step: 8, value: 64 },
      { type: 'check', key: 'norm', label: 'insert LayerNorm each layer', value: false },
      { type: 'check', key: 'res', label: 'add residual connections', value: false },
    ],
    draw(p, s, P) {
      const A = ACT[s.act];
      const n = Math.round(s.width), depth = 30;
      const r = rng(5);
      // He/Xavier baseline gain
      const base = ['relu'].includes(s.act) ? Math.sqrt(2 / n) : Math.sqrt(1 / n);
      const std = base * s.gain;
      let h = Array.from({ length: n }, () => r.normal(0, 1));
      const actMag = [[0, Math.log10(LA.norm(h) / Math.sqrt(n))]];
      const Ws = [];
      const pres = [];
      for (let l = 0; l < depth; l++) {
        const W = Array.from({ length: n }, () => Array.from({ length: n }, () => r.normal(0, std)));
        Ws.push(W);
        const z = LA.matvec(W, h);
        pres.push(z);
        let out = z.map(A.f);
        if (s.res) out = out.map((v, i) => v + h[i]);
        if (s.norm) {
          const m = out.reduce((a, b) => a + b, 0) / n;
          const sd = Math.sqrt(out.reduce((a, b) => a + (b - m) ** 2, 0) / n) + 1e-6;
          out = out.map((v) => (v - m) / sd);
        }
        h = out;
        actMag.push([l + 1, Math.log10(Math.max(LA.norm(h) / Math.sqrt(n), 1e-30))]);
      }
      // backward
      let d = Array.from({ length: n }, () => r.normal(0, 1));
      const gradMag = [[depth, Math.log10(LA.norm(d) / Math.sqrt(n))]];
      for (let l = depth - 1; l >= 0; l--) {
        let dd = d.map((v, i) => v * A.d(pres[l][i]));
        const Wt = LA.transpose(Ws[l]);
        let prev = LA.matvec(Wt, dd);
        if (s.res) prev = prev.map((v, i) => v + d[i]);
        d = prev;
        gradMag.push([l, Math.log10(Math.max(LA.norm(d) / Math.sqrt(n), 1e-30))]);
      }
      p.clear().axes();
      p.clip();
      p.line([[0, 0], [30, 0]], { color: alpha(cssVar('--axis'), .6), width: 1, dash: [3, 3] });
      p.line(actMag, { color: cssVar('--s1'), width: 2.6 });
      p.line(gradMag.reverse(), { color: cssVar('--s2'), width: 2.6 });
      p.clip(false);
      p.legend([{ label: 'forward activation ‖h‖/√n', color: cssVar('--s1') }, { label: 'backward gradient ‖δ‖/√n', color: cssVar('--s2') }], { pos: 'bl' });
      const g0 = gradMag[gradMag.length - 1][1], gN = gradMag[0][1];
      P.readout({
        'gradient at layer 0': `10^${fmt(g0, 2)}`,
        'ratio over 30 layers': `10^${fmt(g0 - gN, 2)}`,
        'verdict': g0 < -4 ? 'VANISHED — early layers cannot learn ✗' : g0 > 3 ? 'EXPLODED ✗' : 'healthy ✓',
      });
    },
    caption: 'Both curves should stay flat near 0 for a trainable network. With tanh and a gain of 0.5 the signal decays geometrically and the gradient reaching layer 0 is astronomically small. Push the gain to 2 and it explodes instead. **Now switch on residual connections**: the gradient gets an identity path home and the curve flattens instantly. Add LayerNorm and the forward signal is pinned too. Those two tricks are why 100-layer networks are routine.',
  });
};

/* ---------- initialization ---------- */

V['initialization'] = (host) => {
  panel(host, {
    title: 'Why initialization scale is not a detail',
    height: 290,
    plot: { xlim: [-3, 3], ylim: [0, 1.2], xlabel: 'pre-activation value', ylabel: 'density (layer 8)' },
    controls: [
      { type: 'select', key: 'scheme', label: 'scheme', value: 'xavier', options: [{ value: 'small', label: 'std = 0.01 (too small)' }, { value: 'xavier', label: 'Xavier/Glorot 1/√n' }, { value: 'he', label: 'He 2/√n (for ReLU)' }, { value: 'big', label: 'std = 1.0 (too big)' }] },
      { type: 'select', key: 'act', label: 'activation', value: 'tanh', options: ['tanh', 'relu', 'sigmoid'].map((k) => ({ value: k, label: ACT[k].name })) },
      { type: 'slider', key: 'layer', label: 'layer to inspect', min: 1, max: 12, step: 1, value: 8 },
    ],
    draw(p, s, P) {
      const n = 96, depth = Math.round(s.layer);
      const A = ACT[s.act];
      const r = rng(3);
      const std = { small: .01, xavier: Math.sqrt(1 / n), he: Math.sqrt(2 / n), big: 1 }[s.scheme];
      let H = Array.from({ length: 400 }, () => Array.from({ length: n }, () => r.normal(0, 1)));
      let zs = [];
      for (let l = 0; l < depth; l++) {
        const W = Array.from({ length: n }, () => Array.from({ length: n }, () => r.normal(0, std)));
        H = H.map((h) => {
          const z = LA.matvec(W, h);
          if (l === depth - 1) zs.push(...z);
          return z.map(A.f);
        });
      }
      // histogram
      const lo = -3, hi = 3, bins = 60;
      const hist = new Array(bins).fill(0);
      let sat = 0;
      for (const v of zs) {
        if (Math.abs(v) > 2.5) sat++;
        const k = Math.floor((v - lo) / (hi - lo) * bins);
        if (k >= 0 && k < bins) hist[k]++;
      }
      const mxc = Math.max(...hist) || 1;
      p.clear().axes();
      p.clip();
      const g = p.ctx;
      for (let i = 0; i < bins; i++) {
        const L = p.px(lo + i * (hi - lo) / bins), R = p.px(lo + (i + 1) * (hi - lo) / bins);
        g.fillStyle = alpha(cssVar('--s1'), .8);
        const hgt = hist[i] / mxc;
        g.fillRect(L, p.py(hgt), Math.max(R - L - .5, 1), p.py(0) - p.py(hgt));
      }
      p.clip(false);
      const mean = zs.reduce((a, b) => a + b, 0) / zs.length;
      const sd = Math.sqrt(zs.reduce((a, b) => a + (b - mean) ** 2, 0) / zs.length);
      P.readout({
        'std of pre-activations': fmt(sd, 5),
        'target': '≈ 1 (stable across depth)',
        'fraction |z| > 2.5': fmt(sat / zs.length * 100, 1) + '%',
        'diagnosis': sd < .02 ? 'collapsed to zero — no signal ✗' : sd > 4 ? 'blown up — saturation ✗' : 'healthy ✓',
      });
    },
    caption: 'Look at the *width* of the histogram. With std 0.01, activations at layer 8 have collapsed onto a spike at zero — every input produces the same output and there is nothing to learn from. With std 1.0 they blow up and saturate the nonlinearity. **Xavier/He initialization picks the one scale that keeps variance constant layer to layer**: $1/\\sqrt{n}$ for symmetric activations, $\\sqrt{2/n}$ for ReLU (which discards half the signal, so it needs the factor of 2).',
  });
};

/* ---------- normalization ---------- */

V['normalization'] = (host) => {
  panel(host, {
    title: 'BatchNorm vs LayerNorm vs RMSNorm — what gets averaged',
    height: 300,
    noCanvas: false,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'select', key: 'kind', label: 'normalization', value: 'layer', options: [{ value: 'none', label: 'none' }, { value: 'batch', label: 'BatchNorm' }, { value: 'layer', label: 'LayerNorm' }, { value: 'rms', label: 'RMSNorm' }, { value: 'group', label: 'GroupNorm' }] },
    ],
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const B = 5, F = 8;
      const cw = Math.min(38, (p.w - 130) / F), ch = 26;
      const x0 = 96, y0 = 44;
      const r = rng(9);
      const M = Array.from({ length: B }, () => Array.from({ length: F }, () => r.normal(0, 1)));
      const inGroup = (b, f) => {
        if (s.kind === 'batch') return true;      // highlight column-wise below
        if (s.kind === 'layer' || s.kind === 'rms') return true;
        return true;
      };
      g.font = '11px -apple-system, sans-serif';
      g.textBaseline = 'middle';
      // header
      g.fillStyle = cssVar('--text-dim');
      g.textAlign = 'center';
      g.fillText('features / channels →', x0 + F * cw / 2, 18);
      g.save();
      g.translate(20, y0 + B * ch / 2);
      g.rotate(-Math.PI / 2);
      g.textAlign = 'center';
      g.fillText('batch →', 0, 0);
      g.restore();

      for (let b = 0; b < B; b++) {
        for (let f = 0; f < F; f++) {
          const X = x0 + f * cw, Y = y0 + b * ch;
          let inSet = false;
          if (s.kind === 'batch') inSet = f === 2;
          else if (s.kind === 'layer' || s.kind === 'rms') inSet = b === 2;
          else if (s.kind === 'group') inSet = b === 2 && f >= 4;
          g.fillStyle = inSet ? alpha(cssVar('--s2'), .75) : alpha(cssVar('--s1'), .22);
          g.fillRect(X, Y, cw - 2, ch - 2);
          g.fillStyle = inSet ? cssVar('--bg') : cssVar('--text-faint');
          g.font = '9.5px ui-monospace, monospace';
          g.textAlign = 'center';
          g.fillText(M[b][f].toFixed(1), X + (cw - 2) / 2, Y + (ch - 2) / 2);
        }
      }
      const desc = {
        none: ['no normalization', 'activations drift as weights update — "internal covariate shift"'],
        batch: ['statistics over the BATCH, per feature', 'depends on batch size; needs running averages at inference; awkward for sequences'],
        layer: ['statistics over the FEATURES, per example', 'batch-size independent → the default in transformers'],
        rms: ['divide by RMS only, no mean subtraction', 'cheaper than LayerNorm, works just as well — used in Llama, T5'],
        group: ['statistics over a GROUP of features, per example', 'compromise for vision when batches are small'],
      }[s.kind];
      g.fillStyle = cssVar('--s2');
      g.font = '600 12px -apple-system, sans-serif';
      g.textAlign = 'left';
      g.fillText('■ ' + desc[0], x0, y0 + B * ch + 22);
      g.fillStyle = cssVar('--text-dim');
      g.font = '11px -apple-system, sans-serif';
      g.fillText(desc[1], x0, y0 + B * ch + 40);

      P.readout({
        'formula': s.kind === 'rms' ? 'x / √(mean(x²) + ε) · g' : 'γ (x − μ) / √(σ² + ε) + β',
        'learnable': s.kind === 'rms' ? 'gain g' : 'scale γ, shift β',
        'why it helps': 'smooths the loss landscape → larger stable learning rates',
      });
    },
    caption: 'The only difference between these is **which cells get averaged together** — highlighted in orange. BatchNorm reaches across examples, which couples them and breaks when the batch is small or the sequence length varies. LayerNorm stays inside one example, which is why transformers use it. RMSNorm drops the mean-subtraction entirely and loses nothing measurable while saving a pass over the data.',
  });
};

/* ---------- dropout ---------- */

V['dropout'] = (host) => {
  panel(host, {
    title: 'Dropout: train an ensemble of subnetworks',
    height: 280,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'p', label: 'drop probability p', min: 0, max: .8, step: .05, value: .5 },
      { type: 'check', key: 'train', label: 'training mode (off = inference)', value: true },
      { type: 'button', label: 'resample mask', onClick: (s) => (s.tick = (s.tick || 0) + 1) },
      { type: 'play' },
    ],
    animate(s) { s.tick = (s.tick || 0) + 1; },
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const layers = [4, 7, 7, 3];
      const W = p.w, H = p.h;
      const r = rng(1000 + (s.tick || 0));
      const drop = layers.map((n, li) => Array.from({ length: n }, () => (li > 0 && li < layers.length - 1 && s.train ? r() < s.p : false)));
      const pos = layers.map((n, li) => Array.from({ length: n }, (_, i) => [
        60 + li * ((W - 120) / (layers.length - 1)),
        H * (i + 1) / (n + 1),
      ]));
      // edges
      for (let l = 0; l < layers.length - 1; l++) {
        for (let i = 0; i < layers[l]; i++) {
          for (let j = 0; j < layers[l + 1]; j++) {
            const dead = drop[l][i] || drop[l + 1][j];
            g.strokeStyle = dead ? alpha(cssVar('--text-faint'), .07) : alpha(cssVar('--s1'), .28);
            g.lineWidth = dead ? .5 : 1;
            g.beginPath(); g.moveTo(pos[l][i][0], pos[l][i][1]); g.lineTo(pos[l + 1][j][0], pos[l + 1][j][1]); g.stroke();
          }
        }
      }
      for (let l = 0; l < layers.length; l++) {
        for (let i = 0; i < layers[l]; i++) {
          const [x, y] = pos[l][i];
          g.beginPath(); g.arc(x, y, 9, 0, 6.2832);
          if (drop[l][i]) { g.fillStyle = alpha(cssVar('--text-faint'), .12); g.strokeStyle = alpha(cssVar('--text-faint'), .35); }
          else { g.fillStyle = alpha(cssVar('--s1'), .55); g.strokeStyle = cssVar('--s1'); }
          g.lineWidth = 1.4;
          g.fill(); g.stroke();
          if (drop[l][i]) {
            g.strokeStyle = cssVar('--danger'); g.lineWidth = 1.6;
            g.beginPath(); g.moveTo(x - 5, y - 5); g.lineTo(x + 5, y + 5); g.moveTo(x + 5, y - 5); g.lineTo(x - 5, y + 5); g.stroke();
          }
        }
      }
      const alive = drop.flat().filter((d) => !d).length;
      P.readout({
        'mode': s.train ? 'training — units dropped, survivors scaled by 1/(1−p)' : 'inference — all units on, no scaling',
        'active units': `${alive} / ${drop.flat().length}`,
        'scale factor': s.train ? fmt(1 / (1 - s.p), 3) : '1.0',
        'effective ensemble size': s.train ? `~2^${drop.flat().length - layers[0] - layers[layers.length - 1]} subnetworks` : '—',
      });
    },
    caption: 'Each training step samples a *different* thinned network, so no unit can rely on any specific partner being present — co-adaptation breaks down and features become individually useful. At inference you use the full network, which approximates averaging over all those subnetworks. The $1/(1-p)$ scaling during training ("inverted dropout") keeps the expected activation constant so nothing needs to change at test time.',
  });
};

/* ---------- convolution ---------- */

V['convolution'] = (host) => {
  const KERNELS = {
    identity: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    blur: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
    sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    edgeH: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    edgeV: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    laplace: [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    emboss: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
  };
  panel(host, {
    title: 'Convolution: one small kernel, slid everywhere',
    height: 300,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'select', key: 'k', label: 'kernel', value: 'edgeV', options: Object.keys(KERNELS) },
      { type: 'slider', key: 'pos', label: 'kernel position', min: 0, max: 1, step: .002, value: .35 },
      { type: 'check', key: 'anim', label: 'sweep', value: false },
      { type: 'play' },
    ],
    animate(s) { s.pos = (s.pos + .004) % 1; },
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const N = 26;
      // synthetic image: shapes with clear edges
      const img = [];
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let j = 0; j < N; j++) {
          let v = .12;
          if (j > 5 && j < 13 && i > 4 && i < 20) v = .82;               // rectangle
          if ((i - 15) ** 2 + (j - 19) ** 2 < 20) v = .55;               // circle
          if (Math.abs(i - j) < 1.2) v = Math.max(v, .95);               // diagonal line
          row.push(v);
        }
        img.push(row);
      }
      const K = KERNELS[s.k];
      const out = [];
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let j = 0; j < N; j++) {
          let acc = 0;
          for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) {
            const ii = Math.min(N - 1, Math.max(0, i + a)), jj = Math.min(N - 1, Math.max(0, j + b));
            acc += img[ii][jj] * K[a + 1][b + 1];
          }
          row.push(acc);
        }
        out.push(row);
      }
      const cell = Math.min((p.w / 2 - 40) / N, (p.h - 60) / N);
      const drawGrid = (M, x0, y0, title, lo, hi) => {
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
          const t = (M[i][j] - lo) / (hi - lo);
          g.fillStyle = mix(cssVar('--bg-inset'), cssVar('--text'), Math.max(0, Math.min(1, t)) * .95);
          g.fillRect(x0 + j * cell, y0 + i * cell, cell + .5, cell + .5);
        }
        g.fillStyle = cssVar('--text-dim');
        g.font = '11px -apple-system, sans-serif';
        g.textAlign = 'left';
        g.fillText(title, x0, y0 - 8);
      };
      const x0 = 20, y0 = 34, x1 = p.w / 2 + 14;
      drawGrid(img, x0, y0, 'input', 0, 1);
      let lo = Infinity, hi = -Infinity;
      for (const r of out) for (const v of r) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      drawGrid(out, x1, y0, `output (${s.k})`, lo, hi);

      // kernel window
      const idx = Math.floor(s.pos * (N * N - 1));
      const ki = Math.floor(idx / N), kj = idx % N;
      g.strokeStyle = cssVar('--s2'); g.lineWidth = 2;
      g.strokeRect(x0 + (kj - 1) * cell, y0 + (ki - 1) * cell, cell * 3, cell * 3);
      g.strokeStyle = cssVar('--s3');
      g.strokeRect(x1 + kj * cell, y0 + ki * cell, cell, cell);

      // kernel values
      const kx = 20, ky = y0 + N * cell + 20;
      g.font = '10px ui-monospace, monospace';
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        g.fillStyle = alpha(K[a][b] > 0 ? cssVar('--s1') : cssVar('--s2'), Math.min(1, Math.abs(K[a][b]) * .8 + .15));
        g.fillRect(kx + b * 22, ky + a * 16, 21, 15);
        g.fillStyle = cssVar('--text');
        g.textAlign = 'center';
        g.fillText(K[a][b].toFixed(2).replace('0.00', '0').replace(/^(-?)0\./, '$1.'), kx + b * 22 + 10, ky + a * 16 + 11);
      }
      g.fillStyle = cssVar('--text-faint');
      g.font = '10.5px -apple-system, sans-serif';
      g.textAlign = 'left';
      g.fillText('kernel', kx, ky - 5);
      let acc = 0;
      for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) {
        const ii = Math.min(N - 1, Math.max(0, ki + a)), jj = Math.min(N - 1, Math.max(0, kj + b));
        acc += img[ii][jj] * K[a + 1][b + 1];
      }
      g.fillText(`output at this position = Σ (input ⊙ kernel) = ${acc.toFixed(3)}`, kx + 90, ky + 24);

      P.readout({
        'parameters in this layer': '9 (+1 bias) — regardless of image size',
        'a dense layer on 26×26 would need': (26 * 26) ** 2 + ' weights',
        'properties': 'translation equivariance + locality + weight sharing',
      });
    },
    caption: 'The same nine numbers are applied at every position — that is **weight sharing**, and it is why a conv layer needs 9 parameters where a dense layer would need 450,000. Because the kernel is identical everywhere, an edge detected in the corner is detected the same way in the centre: **translation equivariance**. Note that `edgeV` responds to vertical intensity changes — early CNN layers learn exactly these filters from scratch.',
  });
};

V['pooling-receptive-field'] = (host) => {
  panel(host, {
    title: 'Stacking convolutions grows the receptive field',
    height: 270,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'layers', label: 'conv layers', min: 1, max: 8, step: 1, value: 3 },
      { type: 'slider', key: 'k', label: 'kernel size', min: 3, max: 7, step: 2, value: 3 },
      { type: 'slider', key: 'stride', label: 'stride', min: 1, max: 3, step: 1, value: 1 },
      { type: 'check', key: 'dilate', label: 'dilated (rate 2^l)', value: false },
    ],
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const L = Math.round(s.layers), k = Math.round(s.k), st = Math.round(s.stride);
      // receptive field growth
      let rf = 1, jump = 1;
      const rows = [{ l: 0, rf: 1 }];
      for (let l = 1; l <= L; l++) {
        const d = s.dilate ? 2 ** (l - 1) : 1;
        rf = rf + ((k - 1) * d) * jump;
        jump *= st;
        rows.push({ l, rf });
      }
      const maxRF = rows[rows.length - 1].rf;
      const W = p.w - 120, x0 = 80;
      const rowH = Math.min(30, (p.h - 40) / (L + 1));
      g.font = '11px -apple-system, sans-serif';
      g.textBaseline = 'middle';
      rows.forEach((r, i) => {
        const y = 22 + i * rowH;
        g.fillStyle = cssVar('--text-dim'); g.textAlign = 'right';
        g.fillText(i === 0 ? 'input' : `layer ${i}`, x0 - 10, y);
        const w = W * (r.rf / maxRF);
        g.fillStyle = alpha(cssVar('--s1'), .25 + .55 * (i / rows.length));
        g.fillRect(x0 + (W - w) / 2, y - rowH / 2 + 4, w, rowH - 10);
        g.fillStyle = cssVar('--text'); g.textAlign = 'left';
        g.font = '10.5px ui-monospace, monospace';
        g.fillText(`${r.rf}×${r.rf} px`, x0 + W + 8, y);
        g.font = '11px -apple-system, sans-serif';
      });
      const params = L * k * k;
      P.readout({
        'receptive field': `${maxRF} × ${maxRF} pixels`,
        'parameters (per channel pair)': params,
        'one big kernel of the same reach would cost': maxRF * maxRF,
        'insight': `${L}× ${k}×${k} is cheaper AND more nonlinear than 1× ${maxRF}×${maxRF}`,
      });
    },
    caption: 'Each layer only looks at a $k\\times k$ window, but a neuron in layer 3 looks at layer 2 neurons that each looked at layer 1 neurons — the reach compounds. Two 3×3 convs see 5×5 using 18 weights instead of 25, **with a nonlinearity in between**. That observation is the entire argument of the VGG paper, and dilation (try it) buys exponential reach without extra parameters.',
  });
};

/* ---------- RNN unrolling ---------- */

V['rnn-unroll'] = (host) => {
  panel(host, {
    title: 'Recurrence: one cell, applied over and over',
    height: 290,
    plot: { xlim: [0, 20], ylim: [-1.2, 1.2], xlabel: 'time step', ylabel: 'hidden state h' },
    controls: [
      { type: 'slider', key: 'w', label: 'recurrent weight w', min: -1.5, max: 1.5, step: .01, value: .9 },
      { type: 'slider', key: 'u', label: 'input weight u', min: 0, max: 2, step: .05, value: .5 },
      { type: 'select', key: 'act', label: 'cell activation', value: 'tanh', options: ['tanh', 'identity', 'relu'].map((k) => ({ value: k, label: ACT[k].name })) },
      { type: 'check', key: 'grad', label: 'show gradient flow to t=0', value: true },
    ],
    draw(p, s, P) {
      const A = ACT[s.act];
      const T = 20;
      const x = Array.from({ length: T }, (_, t) => (t === 2 ? 1 : t === 9 ? -.8 : 0));
      const h = [0];
      const pre = [];
      for (let t = 0; t < T; t++) {
        const z = s.w * h[t] + s.u * x[t];
        pre.push(z);
        h.push(A.f(z));
      }
      p.clear().axes();
      p.clip();
      p.line(h.map((v, t) => [t, v]), { color: cssVar('--s1'), width: 2.6 });
      p.points(h.map((v, t) => [t, v]), { r: 3, color: cssVar('--s1') });
      // input spikes
      x.forEach((v, t) => { if (v !== 0) p.line([[t, 0], [t, v]], { color: cssVar('--s3'), width: 2.5 }); });
      if (s.grad) {
        // dh_T/dh_t = prod w*act'(z)
        const gr = [];
        let g = 1;
        for (let t = T - 1; t >= 0; t--) {
          g *= s.w * A.d(pre[t]);
          gr.push([t, Math.max(-1.15, Math.min(1.15, g))]);
        }
        p.line(gr, { color: cssVar('--s2'), width: 2, dash: [5, 3] });
      }
      p.clip(false);
      p.legend([
        { label: 'hidden state hₜ', color: cssVar('--s1') },
        { label: 'input spikes', color: cssVar('--s3') },
        ...(s.grad ? [{ label: '∂h₂₀/∂hₜ', color: cssVar('--s2'), dash: true }] : []),
      ], { pos: 'bl' });
      let g = 1;
      for (let t = T - 1; t >= 0; t--) g *= s.w * A.d(pre[t]);
      P.readout({
        'w': fmt(s.w, 2),
        '∂h₂₀/∂h₀': Math.abs(g) < 1e-6 ? g.toExponential(2) : fmt(g, 6),
        'memory': Math.abs(s.w) < .8 ? 'forgets quickly' : Math.abs(s.w) < 1 ? 'holds for a while' : 'unstable / explodes',
        'the problem': 'gradient ≈ wᵀ — exponential in sequence length either way',
      });
    },
    caption: 'The same weight is applied at every step, so after $T$ steps the gradient carries a factor of $w^T$. With $w=0.9$ over 20 steps that is $0.12$; over 100 steps, $10^{-5}$. **The information from an early token simply cannot reach the loss.** Set $w=1.05$ and it explodes instead. There is no good value — which is exactly why LSTMs added a gated, additive memory path, and why attention eventually replaced recurrence entirely.',
  });
};

V['lstm-gates'] = (host) => {
  panel(host, {
    title: 'LSTM gates: an additive highway for memory',
    height: 290,
    plot: { xlim: [0, 30], ylim: [-.2, 2.2], xlabel: 'time step', ylabel: 'cell state c' },
    controls: [
      { type: 'slider', key: 'f', label: 'forget gate f', min: 0, max: 1, step: .01, value: .95 },
      { type: 'slider', key: 'i', label: 'input gate i', min: 0, max: 1, step: .01, value: .5 },
      { type: 'slider', key: 'o', label: 'output gate o', min: 0, max: 1, step: .01, value: 1 },
      { type: 'check', key: 'cmp', label: 'compare with vanilla RNN', value: true },
    ],
    draw(p, s, P) {
      const T = 30;
      const x = Array.from({ length: T }, (_, t) => (t === 3 ? 1.5 : t === 14 ? .8 : 0));
      const c = [0], hh = [0], rnn = [0];
      for (let t = 0; t < T; t++) {
        c.push(s.f * c[t] + s.i * x[t]);
        hh.push(s.o * Math.tanh(c[t + 1]));
        rnn.push(Math.tanh(.95 * rnn[t] + .5 * x[t]));
      }
      p.clear().axes();
      p.clip();
      x.forEach((v, t) => { if (v !== 0) p.line([[t, 0], [t, v]], { color: cssVar('--s3'), width: 2.5 }); });
      p.line(c.map((v, t) => [t, v]), { color: cssVar('--s1'), width: 2.8 });
      p.line(hh.map((v, t) => [t, v]), { color: cssVar('--s4'), width: 2, dash: [4, 3] });
      if (s.cmp) p.line(rnn.map((v, t) => [t, v]), { color: alpha(cssVar('--s2'), .8), width: 2, dash: [6, 3] });
      p.clip(false);
      p.legend([
        { label: 'cell state c (the highway)', color: cssVar('--s1') },
        { label: 'hidden output h = o·tanh(c)', color: cssVar('--s4'), dash: true },
        ...(s.cmp ? [{ label: 'vanilla RNN', color: cssVar('--s2'), dash: true }] : []),
      ], { pos: 'tr' });
      P.readout({
        'c update': 'cₜ = f ⊙ cₜ₋₁ + i ⊙ c̃ₜ  ← addition, not multiplication by a weight matrix',
        'gradient over 30 steps': fmt(s.f ** 30, 6),
        'with f = 1.0': '1.0 — perfect memory, no decay',
      });
    },
    caption: 'The cell state is updated by **adding**, and it is scaled only by a gate the network itself controls. Set $f=1$ and information from step 3 survives to step 30 completely intact — the gradient path is multiplication by 1, thirty times. Compare the vanilla RNN (orange), whose memory of the first spike has already faded. This gating idea — *let the network choose what to keep* — is the direct ancestor of residual connections.',
  });
};

/* ---------- loss functions ---------- */

V['loss-functions'] = (host) => {
  panel(host, {
    title: 'Loss functions and what they punish',
    height: 290,
    plot: { xlim: [-3, 3], ylim: [0, 4], xlabel: 'margin  y·f(x)   /   residual  (ŷ − y)', ylabel: 'loss' },
    controls: [
      { type: 'select', key: 'task', label: 'task', value: 'cls', options: [{ value: 'cls', label: 'classification' }, { value: 'reg', label: 'regression' }] },
      { type: 'check', key: 'grad', label: 'show gradient', value: false },
      { type: 'slider', key: 'delta', label: 'Huber δ', min: .2, max: 2, step: .05, value: 1 },
    ],
    draw(p, s, P) {
      p.clear().axes();
      p.clip();
      const cols = SERIES();
      let items;
      if (s.task === 'cls') {
        items = [
          ['0-1 loss', (m) => (m > 0 ? 0 : 1)],
          ['hinge (SVM)', (m) => Math.max(0, 1 - m)],
          ['logistic / BCE', (m) => Math.log(1 + Math.exp(-m)) / Math.log(2)],
          ['exponential (AdaBoost)', (m) => Math.exp(-m)],
          ['squared', (m) => (1 - m) ** 2],
        ];
      } else {
        items = [
          ['squared (L2)', (r) => r * r],
          ['absolute (L1)', (r) => Math.abs(r)],
          ['Huber', (r) => (Math.abs(r) <= s.delta ? .5 * r * r : s.delta * (Math.abs(r) - .5 * s.delta))],
          ['log-cosh', (r) => Math.log(Math.cosh(r))],
        ];
      }
      items.forEach(([name, f], i) => {
        if (s.grad) {
          p.fn((x) => { const e = 1e-3; return (f(x + e) - f(x - e)) / (2 * e); }, { color: cols[i], width: 2 });
        } else p.fn(f, { color: cols[i], width: 2.4 });
      });
      p.clip(false);
      p.legend(items.map(([n], i) => ({ label: n, color: cols[i] })), { pos: 'tr' });
      P.readout(s.task === 'cls' ? {
        'margin > 0': 'correct classification',
        'hinge': 'zero loss once margin ≥ 1 → sparse support vectors',
        'logistic': 'never exactly zero → keeps pushing confident points further',
        'exponential': 'punishes outliers hardest → AdaBoost is outlier-sensitive',
      } : {
        'L2': 'differentiable everywhere, but a single outlier dominates (grows quadratically)',
        'L1': 'robust to outliers; gradient is ±1 everywhere → constant-size steps',
        'Huber': `quadratic inside |r| < ${fmt(s.delta, 2)}, linear outside — best of both`,
      });
    },
    caption: 'Turn on **show gradient** — that is what actually reaches the weights. The squared loss\'s gradient grows without bound, so one mislabeled point can hijack training; L1\'s gradient is constant, so it shrugs. Hinge loss is exactly zero past margin 1, meaning correctly-and-confidently classified points contribute nothing — that is what makes SVM solutions sparse. Logistic loss never quite reaches zero, which is why unregularized logistic regression on separable data drives $\\|w\\|\\to\\infty$.',
  });
};

/* ---------- optimizer / lr schedules ---------- */

V['lr-schedules'] = (host) => {
  panel(host, {
    title: 'Learning-rate schedules',
    height: 270,
    plot: { xlim: [0, 1], ylim: [0, 1.15], xlabel: 'fraction of training', ylabel: 'learning rate (relative)' },
    controls: [
      { type: 'slider', key: 'warm', label: 'warmup fraction', min: 0, max: .3, step: .005, value: .03 },
      { type: 'slider', key: 'floor', label: 'final LR fraction', min: 0, max: .5, step: .01, value: .1 },
      { type: 'slider', key: 'steps', label: 'step-decay drops', min: 1, max: 5, step: 1, value: 3 },
    ],
    draw(p, s, P) {
      p.clear().axes();
      p.clip();
      const cols = SERIES();
      const warm = (t) => (s.warm > 0 && t < s.warm ? t / s.warm : 1);
      const scheds = [
        ['constant', (t) => warm(t)],
        ['cosine', (t) => warm(t) * (s.floor + (1 - s.floor) * .5 * (1 + Math.cos(Math.PI * Math.min(1, Math.max(0, (t - s.warm) / (1 - s.warm))))))],
        ['linear decay', (t) => warm(t) * (s.floor + (1 - s.floor) * (1 - Math.min(1, Math.max(0, (t - s.warm) / (1 - s.warm)))))],
        ['step decay', (t) => warm(t) * Math.pow(.3, Math.floor(t * (s.steps + 1) / 1.0001))],
        ['inverse sqrt', (t) => warm(t) * Math.min(1, 1 / Math.sqrt(Math.max(t, 1e-3) / .1))],
      ];
      scheds.forEach(([n, f], i) => p.fn(f, { color: cols[i], width: 2.2, n: 500 }));
      if (s.warm > 0) p.line([[s.warm, 0], [s.warm, 1.15]], { color: alpha(cssVar('--text-faint'), .5), width: 1, dash: [3, 3] });
      p.clip(false);
      p.legend(scheds.map(([n], i) => ({ label: n, color: cols[i] })), { pos: 'tr' });
      P.readout({
        'warmup': 'prevents early divergence when Adam\'s second-moment estimate is still noisy',
        'cosine': 'the default for LLM pretraining — smooth, ends near zero',
        'inverse sqrt': 'the original transformer schedule (Vaswani et al.)',
        'why decay at all': 'large LR explores, small LR settles into a minimum',
      });
    },
    caption: 'Every large training run uses a schedule, not a constant. **Warmup** matters most with Adam: at step 0 the second-moment estimate is based on almost no data, so the effective step size is wild — ramping in avoids blowing up. **Decay** matters because the noise floor of SGD scales with the learning rate; to actually converge you must shrink it. Cosine decay to ~10% of peak is the current default for language model pretraining.',
  });
};

/* ---------- batch size / noise ---------- */

V['batch-size'] = (host) => {
  panel(host, {
    title: 'Minibatch size: noise, compute, and the linear scaling rule',
    height: 300,
    plot: { xlim: [-3, 3], ylim: [-2.4, 2.4], equal: true, xlabel: 'θ₁', ylabel: 'θ₂' },
    controls: [
      { type: 'slider', key: 'bs', label: 'batch size', min: 1, max: 128, step: 1, value: 8 },
      { type: 'slider', key: 'lr', label: 'learning rate', min: .005, max: .4, step: .005, value: .08 },
      { type: 'check', key: 'scale', label: 'apply linear scaling rule (lr ∝ batch)', value: false },
      { type: 'play' },
      { type: 'button', label: '↺ reset', onClick: (s) => { s.w = [-2.4, 1.9]; s.path = []; } },
    ],
    state: { w: [-2.4, 1.9], path: [] },
    animate(s) {
      const N = 256;
      // fixed synthetic dataset defining a noisy quadratic loss
      const r = rng(31);
      if (!s.data) s.data = Array.from({ length: N }, () => [r.normal(0, 1), r.normal(0, 1)]);
      const bs = Math.round(s.bs);
      const lr = s.scale ? s.lr * bs / 8 : s.lr;
      let g0 = 0, g1 = 0;
      for (let i = 0; i < bs; i++) {
        const d = s.data[Math.floor(Math.random() * N)];
        g0 += .3 * (s.w[0] - d[0] * .35);
        g1 += 1.6 * (s.w[1] - d[1] * .3);
      }
      s.w = [s.w[0] - lr * g0 / bs, s.w[1] - lr * g1 / bs];
      s.path.push([...s.w]);
      if (s.path.length > 400) s.path.shift();
    },
    draw(p, s, P) {
      const f = (x, y) => .5 * (.3 * x * x + 1.6 * y * y);
      p.clear();
      p.heat(f, { step: 5, cmap: (t) => mix(cssVar('--bg-inset'), cssVar('--s1'), t * .5) });
      p.axes({ nx: 5, ny: 4 });
      p.clip();
      p.contour(f, [.1, .4, 1, 2, 3.5, 5.5, 8], { color: alpha(cssVar('--axis'), .5) });
      p.line(s.path, { color: cssVar('--s2'), width: 1.8 });
      p.circle(s.w[0], s.w[1], 5, { fill: cssVar('--s2'), stroke: cssVar('--bg-inset'), width: 2 });
      p.clip(false);
      const bs = Math.round(s.bs);
      P.readout({
        'batch size': bs,
        'effective lr': fmt(s.scale ? s.lr * bs / 8 : s.lr, 4),
        'gradient noise ∝ 1/√B': fmt(1 / Math.sqrt(bs), 3),
        'loss': fmt(f(s.w[0], s.w[1]), 5),
        'compute per step': `${bs}× — but parallelizable`,
      });
    },
    caption: 'Small batches take a jagged path — that noise is a real regularizer, and it helps escape sharp minima. Large batches take smooth steps but each costs proportionally more compute, and beyond a critical size you get almost nothing extra per step. The **linear scaling rule** (lr ∝ batch size, with warmup) keeps the *per-example* step consistent; turn it on and watch the large-batch trajectory recover its speed.',
  });
};

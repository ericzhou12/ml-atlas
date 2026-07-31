/* ============================================================
   viz/foundations.js — beginner-facing figures.

   These exist to make the *mechanics* visible for someone meeting
   linear algebra for the first time: what matrix multiplication
   actually does step by step, what "rank" looks like geometrically,
   and why a low-rank approximation is a compression scheme.
   ============================================================ */

import { panel } from '../ui.js';
import { cssVar, alpha, seqMap, divergeMap, LA, rng, fmt } from '../plot.js';

const V = {};
export default V;

/* ------------------------------------------------------------
   shared: draw a matrix as a bracketed grid of numbers.
   x, y are pixel coords of the top-left corner.
   o.tint(i,j)   -> a colour to wash the cell with, or null
   o.weight(i,j) -> 0..1 strength of that wash
   o.dim(i,j)    -> true to grey the number out
   o.digits      -> decimals shown
   ------------------------------------------------------------ */
function grid(p, M, x, y, cw, ch, o = {}) {
  const g = p.ctx;
  const rows = M.length, cols = M[0].length;
  const W = cols * cw, H = rows * ch;
  g.save();

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const c = o.tint ? o.tint(i, j) : null;
      const wgt = c ? (o.weight ? o.weight(i, j) : 0.2) : 0;
      g.fillStyle = c ? alpha(c, wgt) : cssVar('--bg-elev');
      g.fillRect(x + j * cw + 1, y + i * ch + 1, cw - 2, ch - 2);
      g.strokeStyle = cssVar('--border-soft');
      g.lineWidth = 1;
      g.strokeRect(x + j * cw + 1.5, y + i * ch + 1.5, cw - 3, ch - 3);

      const dimmed = o.dim ? o.dim(i, j) : false;
      g.fillStyle = dimmed ? cssVar('--text-faint') : (c ? c : cssVar('--text'));
      g.font = `${c && !dimmed ? '600 ' : ''}11.5px ui-monospace, Menlo, monospace`;
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(fmt(M[i][j], o.digits != null ? o.digits : 0),
                 x + j * cw + cw / 2, y + i * ch + ch / 2);
    }
  }

  // square brackets, so it reads as a matrix rather than a spreadsheet
  g.strokeStyle = cssVar('--text-faint');
  g.lineWidth = 1.4;
  const ear = 5;
  g.beginPath();
  g.moveTo(x + ear, y - 2); g.lineTo(x - 1, y - 2); g.lineTo(x - 1, y + H + 2); g.lineTo(x + ear, y + H + 2);
  g.moveTo(x + W - ear, y - 2); g.lineTo(x + W + 1, y - 2); g.lineTo(x + W + 1, y + H + 2); g.lineTo(x + W - ear, y + H + 2);
  g.stroke();

  if (o.label) {
    g.fillStyle = o.labelColor || cssVar('--text-dim');
    g.font = '600 12px ui-monospace, Menlo, monospace';
    g.textAlign = 'center'; g.textBaseline = 'bottom';
    g.fillText(o.label, x + W / 2, y - 8);
  }
  if (o.sub) {
    g.fillStyle = cssVar('--text-faint');
    g.font = '10.5px system-ui, sans-serif';
    g.textAlign = 'center'; g.textBaseline = 'top';
    g.fillText(o.sub, x + W / 2, y + H + 8);
  }
  g.restore();
  return { x, y, w: W, h: H };
}

function symbol(p, s, x, y, o = {}) {
  const g = p.ctx;
  g.save();
  g.fillStyle = o.color || cssVar('--text-dim');
  g.font = `${o.size || 17}px system-ui, sans-serif`;
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(s, x, y);
  g.restore();
}

function centerText(p, s, x, y, o = {}) {
  const g = p.ctx;
  g.save();
  g.fillStyle = o.color || cssVar('--text-dim');
  g.font = `${o.weight || ''} ${o.size || 12}px ${o.mono ? 'ui-monospace, Menlo, monospace' : 'system-ui, sans-serif'}`.trim();
  g.textAlign = o.align || 'center'; g.textBaseline = 'middle';
  g.fillText(s, x, y);
  g.restore();
}

/* ============================================================
   1. Matrix multiplication, one step at a time.
   Two readings of the same product:
     "dot"   — each output cell is one row · one column
     "outer" — the whole output is a sum of rank-1 layers
   The second reading is the one that makes rank and LoRA obvious,
   and almost nobody is taught it first.
   ============================================================ */

V['matmul-walkthrough'] = (host, params = {}) => {
  const R = rng(7);
  const mk = (r, c) => Array.from({ length: r }, () => Array.from({ length: c }, () => Math.round(R.normal() * 2)));
  let A = [[2, 0, 1], [-1, 3, 2], [0, 1, -2]];
  let B = [[1, 2, 0], [3, -1, 1], [0, 2, 2]];

  const C = () => LA.matmul(A, B);

  panel(host, {
    title: 'Matrix multiplication, one step at a time',
    height: 380,
    plot: { pad: { l: 0, r: 0, t: 0, b: 0 } },
    controls: [
      { type: 'select', key: 'mode', label: 'reading', value: params.mode || 'dot', options: [
        { value: 'dot', label: 'row · column  (one cell at a time)' },
        { value: 'outer', label: 'sum of layers  (one rank-1 slab at a time)' },
      ] },
      { type: 'slider', key: 'step', label: 'step', min: 0, max: 9, step: 1, value: 0 },
      { type: 'play' },
      { type: 'button', label: '⟳ new numbers', onClick: (s, P) => { A = mk(3, 3); B = mk(3, 3); P.set('step', 0, true); } },
    ],

    animate(s, P) {
      s._t = (s._t || 0) + 1;
      if (s._t % 45) return;
      const max = s.mode === 'dot' ? 9 : 3;
      P.set('step', (s.step + 1) % (max + 1), true);
    },

    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const Cm = C();
      const maxStep = s.mode === 'dot' ? 9 : 3;
      const step = Math.min(Math.round(s.step), maxStep);

      const cA = cssVar('--s1'), cB = cssVar('--s2'), cC = cssVar('--s3');
      const cw = 34, ch = 30;
      const mw = 3 * cw;
      const gap = 34;
      const total = mw * 3 + gap * 2 + 34;
      const x0 = Math.max(12, (p.w - total) / 2);
      const yTop = 52;

      if (s.mode === 'dot') {
        /* ---- reading 1: every output cell is a dot product ---- */
        const i = step === 9 ? 2 : Math.floor(step / 3);
        const j = step === 9 ? 2 : step % 3;
        const done = (r, c) => r * 3 + c < step;

        grid(p, A, x0, yTop, cw, ch, {
          label: 'A', labelColor: cA,
          tint: (r) => (r === i ? cA : null), weight: () => 0.18,
          sub: 'rows →',
        });
        symbol(p, '×', x0 + mw + gap / 2, yTop + 45);
        grid(p, B, x0 + mw + gap, yTop, cw, ch, {
          label: 'B', labelColor: cB,
          tint: (r, c) => (c === j ? cB : null), weight: () => 0.18,
          sub: 'columns ↓',
        });
        symbol(p, '=', x0 + 2 * mw + 1.5 * gap, yTop + 45);
        grid(p, Cm, x0 + 2 * (mw + gap), yTop, cw, ch, {
          label: 'C = AB', labelColor: cC,
          tint: (r, c) => (r === i && c === j ? cC : null), weight: () => 0.4,
          dim: (r, c) => !(done(r, c) || (r === i && c === j)),
        });

        // the arithmetic, spelled out
        const terms = [0, 1, 2].map((k) => `${fmt(A[i][k], 0)}·${fmt(B[k][j], 0)}`);
        const vals = [0, 1, 2].map((k) => A[i][k] * B[k][j]);
        const yA = yTop + 3 * ch + 54;

        centerText(p, `cell C[${i + 1}][${j + 1}]  =  row ${i + 1} of A  ·  column ${j + 1} of B`,
          p.w / 2, yA, { size: 12.5, color: cssVar('--text-dim') });
        centerText(p, `= ${terms.join('  +  ')}`, p.w / 2, yA + 26, { size: 14, mono: true, color: cssVar('--text') });
        centerText(p, `= ${vals.join('  +  ')}   =   ${Cm[i][j]}`, p.w / 2, yA + 50,
          { size: 14, mono: true, weight: '600', color: cC });

        P.readout({
          'reading': 'row · column',
          'cell': `C[${i + 1}][${j + 1}]`,
          'value': String(Cm[i][j]),
          'cells done': `${Math.min(step + 1, 9)} / 9`,
        });

      } else {
        /* ---- reading 2: the product is a stack of rank-1 layers ---- */
        const k = Math.min(step, 2);
        const layers = [0, 1, 2].map((q) =>
          A.map((row) => B[0].map((_, c) => row[q] * B[q][c])));
        const partial = layers.slice(0, step + 1).reduce(
          (acc, L) => acc.map((row, r) => row.map((v, c) => v + L[r][c])),
          LA.zeros(3, 3));

        grid(p, A, x0, yTop, cw, ch, {
          label: 'A', labelColor: cA,
          tint: (r, c) => (c === k ? cA : null), weight: () => 0.22,
          dim: (r, c) => c !== k,
          sub: `column ${k + 1} of A`,
        });
        symbol(p, '⊗', x0 + mw + gap / 2, yTop + 45);
        grid(p, B, x0 + mw + gap, yTop, cw, ch, {
          label: 'B', labelColor: cB,
          tint: (r) => (r === k ? cB : null), weight: () => 0.22,
          dim: (r) => r !== k,
          sub: `row ${k + 1} of B`,
        });
        symbol(p, '=', x0 + 2 * mw + 1.5 * gap, yTop + 45);
        grid(p, layers[k], x0 + 2 * (mw + gap), yTop, cw, ch, {
          label: `layer ${k + 1}`, labelColor: cssVar('--s4'),
          tint: () => cssVar('--s4'), weight: () => 0.14,
          sub: 'rank 1 — every row is a multiple of the same row',
        });

        const yB = yTop + 3 * ch + 72;
        const xr = Math.max(12, (p.w - (mw * 2 + gap + 30)) / 2);
        grid(p, partial, xr, yB, cw, ch, {
          label: `layers 1…${step + 1} added up`, labelColor: cssVar('--text-dim'),
        });
        symbol(p, step === 2 ? '=' : '→', xr + mw + gap / 2 + 15, yB + 45);
        grid(p, Cm, xr + mw + gap + 30, yB, cw, ch, {
          label: 'C = AB', labelColor: cC,
          tint: () => cC, weight: () => (step === 2 ? 0.18 : 0.05),
        });

        centerText(p, step === 2
          ? 'All three layers added — this is exactly AB.'
          : `${2 - step} layer${step === 1 ? '' : 's'} still to add.`,
          p.w / 2, yB + 3 * ch + 26,
          { size: 12, color: step === 2 ? cC : cssVar('--text-faint') });

        P.readout({
          'reading': 'sum of rank-1 layers',
          'layer': `${k + 1} of 3`,
          'each layer rank': '1',
          'layers ⇒ rank': `≤ ${step + 1}`,
        });
      }
    },

    caption: `Switch **reading** to \`sum of layers\` and step through. Same product, but now
\`AB\` is three rank-1 slabs stacked on top of each other — column *k* of A times row *k* of B.

That is the whole idea behind rank. A 3×3 product built from a 3-wide inner dimension needs 3 layers;
if the inner dimension were 1, you would get **one** layer and the answer would be rank 1 no matter how
big the outer dimensions are. Hold that picture — it is what [low-rank structure](#/l/math-matrices)
and LoRA are about.`,
  });
};

/* ============================================================
   2. What rank looks like: a 2×2 matrix built from its own SVD,
   with the singular values on sliders. Drag σ₂ to zero and the
   plane collapses onto a line — that *is* rank deficiency.
   ============================================================ */

V['rank-collapse'] = (host, params = {}) => {
  // deterministic ring + interior grid, so the squash is legible
  const ring = Array.from({ length: 160 }, (_, i) => {
    const a = (i / 160) * Math.PI * 2;
    return [Math.cos(a), Math.sin(a)];
  });
  const gridPts = [];
  for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) gridPts.push([i / 2, j / 2]);

  panel(host, {
    title: 'Rank, seen directly: how much space survives',
    height: 340,
    plot: { xlim: [-3, 3], ylim: [-2.2, 2.2], equal: true, xlabel: 'x₁', ylabel: 'x₂' },
    controls: [
      { type: 'slider', key: 's1', label: 'σ₁  (stretch along direction 1)', min: 0, max: 2.5, step: .01, value: params.s1 != null ? params.s1 : 1.8 },
      { type: 'slider', key: 's2', label: 'σ₂  (stretch along direction 2)', min: 0, max: 2.5, step: .01, value: params.s2 != null ? params.s2 : 0.9 },
      { type: 'slider', key: 'th', label: 'orientation', min: 0, max: 3.14, step: .01, value: 0.5 },
      { type: 'check', key: 'showIn', label: 'show the input circle', value: true },
    ],

    draw(p, s, P) {
      p.clear().axes({ nx: 7, ny: 5 });
      p.clip();

      const c = Math.cos(s.th), sn = Math.sin(s.th);
      const u1 = [c, sn], u2 = [-sn, c];
      // A = σ₁ u₁u₁ᵀ + σ₂ u₂u₂ᵀ  — symmetric, so its own SVD, easy to read off
      const A = [
        [s.s1 * u1[0] * u1[0] + s.s2 * u2[0] * u2[0], s.s1 * u1[0] * u1[1] + s.s2 * u2[0] * u2[1]],
        [s.s1 * u1[1] * u1[0] + s.s2 * u2[1] * u2[0], s.s1 * u1[1] * u1[1] + s.s2 * u2[1] * u2[1]],
      ];

      if (s.showIn) {
        p.line(ring.concat([ring[0]]), { color: cssVar('--text-faint'), width: 1.2, dash: [3, 3], alpha: .8 });
        p.points(gridPts, { r: 1.8, color: alpha(cssVar('--text-faint'), .5) });
      }

      const out = ring.map((v) => LA.matvec(A, v));
      const outGrid = gridPts.map((v) => LA.matvec(A, v));
      p.line(out.concat([out[0]]), { color: cssVar('--s1'), width: 2.4 });
      p.points(outGrid, { r: 2.6, color: cssVar('--s1') });

      // the two principal directions, scaled by their singular values
      p.arrow(0, 0, s.s1 * u1[0], s.s1 * u1[1], { color: cssVar('--s2'), width: 2.2, head: 8 });
      p.arrow(0, 0, s.s2 * u2[0], s.s2 * u2[1], { color: cssVar('--s4'), width: 2.2, head: 8 });

      p.clip(false);
      p.legend([
        { label: 'unit circle in', color: cssVar('--text-faint') },
        { label: 'image out', color: cssVar('--s1') },
        { label: 'σ₁ direction', color: cssVar('--s2') },
        { label: 'σ₂ direction', color: cssVar('--s4') },
      ], { pos: 'tl' });

      const eps = 0.02;
      const rank = (s.s1 > eps ? 1 : 0) + (s.s2 > eps ? 1 : 0);
      const verdict = rank === 2 ? 'full rank — nothing lost'
        : rank === 1 ? 'rank 1 — the plane is now a LINE'
        : 'rank 0 — everything maps to the origin';

      P.readout({
        'σ₁': fmt(s.s1, 2),
        'σ₂': fmt(s.s2, 2),
        'det = σ₁σ₂': fmt(s.s1 * s.s2, 3),
        'rank': String(rank),
        '': verdict,
      });
    },

    caption: `Drag **σ₂** down to 0. The output circle flattens into a line segment: every point in the
plane, all infinitely many of them, now land on a single one-dimensional line. That is what
**rank 1** means — and notice the determinant hits 0 at the same moment, because an area squashed
onto a line has zero area.

Rank counts *how many independent directions survive*, nothing more. σ₂ = 0.05 is technically
rank 2, but for any practical purpose that second direction is noise — which is why real data is
described as "approximately low rank".`,
  });
};

/* ============================================================
   3. Low-rank approximation: the compression argument, and the
   one that makes LoRA's parameter count obvious.
   The matrix is *constructed* as a sum of orthonormal rank-1
   layers with decaying weights, so truncating at r really is the
   optimal rank-r approximation (Eckart–Young), no SVD solver needed.
   ============================================================ */

V['low-rank-approx'] = (host, params = {}) => {
  const N = 48;
  const K = 24;
  // orthonormal cosine (DCT-II) basis: u_k[i] = cos(pi k (i+.5)/N), normalised
  const basis = Array.from({ length: K }, (_, k) => {
    const v = Array.from({ length: N }, (_, i) => Math.cos(Math.PI * k * (i + 0.5) / N));
    const n = Math.hypot(...v);
    return v.map((x) => x / n);
  });
  const basis2 = Array.from({ length: K }, (_, k) => {
    const v = Array.from({ length: N }, (_, i) => Math.cos(Math.PI * (k + 0.5) * (i + 0.7) / N));
    const n = Math.hypot(...v);
    return v.map((x) => x / n);
  });
  const R = rng(3);
  const sig = Array.from({ length: K }, (_, k) => 12 / Math.pow(k + 1, 1.35) + 0.35 * Math.abs(R.normal()));
  sig.sort((a, b) => b - a);

  const build = (r) => {
    const M = LA.zeros(N, N);
    for (let k = 0; k < r; k++) {
      const u = basis[k], v = basis2[k], sg = sig[k];
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) M[i][j] += sg * u[i] * v[j];
    }
    return M;
  };
  const full = build(K);
  let lo = Infinity, hi = -Infinity;
  for (const row of full) for (const v of row) { if (v < lo) lo = v; if (v > hi) hi = v; }

  panel(host, {
    title: 'Low-rank approximation — keep the big layers, drop the rest',
    height: 320,
    plot: { pad: { l: 8, r: 8, t: 8, b: 8 } },
    controls: [
      { type: 'slider', key: 'r', label: 'rank r kept', min: 1, max: K, step: 1, value: params.r || 3 },
      { type: 'check', key: 'err', label: 'show what was thrown away', value: false },
    ],

    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const r = Math.round(s.r);
      const approx = build(r);

      const side = Math.min(190, (p.w - 90) / (s.err ? 3 : 2));
      const gap = 34;
      const nPanel = s.err ? 3 : 2;
      const x0 = (p.w - (side * nPanel + gap * (nPanel - 1))) / 2;
      const y0 = 34;
      const cell = side / N;

      const paint = (M, x, label, sub, cmap) => {
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
          const t = (M[i][j] - lo) / (hi - lo || 1);
          g.fillStyle = cmap(Math.max(0, Math.min(1, t)));
          g.fillRect(x + j * cell, y0 + i * cell, cell + .6, cell + .6);
        }
        g.strokeStyle = cssVar('--border');
        g.strokeRect(x + .5, y0 + .5, side, side);
        centerText(p, label, x + side / 2, y0 - 14, { size: 12, weight: '600', color: cssVar('--text') });
        centerText(p, sub, x + side / 2, y0 + side + 16, { size: 11, color: cssVar('--text-faint') });
      };

      const full2 = full;
      paint(full2, x0, 'the real matrix', `rank ${K} · ${N * N} numbers`, seqMap);
      paint(approx, x0 + side + gap, `rank-${r} approximation`,
        `${2 * N * r} numbers · ${((2 * N * r) / (N * N) * 100).toFixed(0)}% of the original`, seqMap);

      let sse = 0, tot = 0;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const d = full2[i][j] - approx[i][j];
        sse += d * d; tot += full2[i][j] * full2[i][j];
      }

      if (s.err) {
        const diff = full2.map((row, i) => row.map((v, j) => v - approx[i][j]));
        let m = 0;
        for (const row of diff) for (const v of row) m = Math.max(m, Math.abs(v));
        paint(diff.map((row) => row.map((v) => lo + (hi - lo) * (0.5 + 0.5 * v / (m || 1)))),
          x0 + 2 * (side + gap), 'what was discarded',
          `${(100 * sse / tot).toFixed(1)}% of the energy`, divergeMap);
      }

      P.readout({
        'rank kept': String(r),
        'stored numbers': `${2 * N * r} vs ${N * N}`,
        'compression': `${(N * N / (2 * N * r)).toFixed(1)}×`,
        'error': `${(100 * Math.sqrt(sse / tot)).toFixed(1)}%`,
      });
    },

    caption: `Slide **r** from 1 upward. By about r = 4 the approximation is already hard to tell from the
original, because the singular values decay fast — the first few layers carry nearly all the structure.

Now read the parameter count. Storing a rank-*r* approximation of an *N×N* matrix costs $2Nr$ numbers
instead of $N^2$. That single ratio is the entire argument for **LoRA**: a 4096×4096 weight matrix is
16.8M numbers, but a rank-8 update to it is only $2 \\times 4096 \\times 8 = 65{,}536$ — 0.4% as many —
and empirically that is enough room to hold what fine-tuning needs to change.`,
  });
};

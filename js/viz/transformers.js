/* ============================================================
   viz/transformers.js — attention, tokenization, LLM mechanics.
   ============================================================ */

import { panel } from '../ui.js';
import { cssVar, alpha, mix, seqMap, divergeMap, LA, rng, fmt, SERIES } from '../plot.js';

const V = {};
export default V;

/* ============================================================
   Attention
   ============================================================ */

V['attention-basics'] = (host) => {
  panel(host, {
    title: 'Attention is a soft dictionary lookup',
    height: 300,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'qx', label: 'query direction', min: 0, max: 6.28, step: .01, value: .6 },
      { type: 'slider', key: 'temp', label: 'scale 1/√d (sharpness)', min: .2, max: 4, step: .05, value: 1 },
      { type: 'check', key: 'showV', label: 'show weighted output', value: true },
    ],
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const keys = [
        { name: 'the', v: [.2, .8], ang: 0.2 },
        { name: 'cat', v: [.9, .1], ang: 1.4 },
        { name: 'sat', v: [.4, -.7], ang: 3.0 },
        { name: 'mat', v: [-.6, .5], ang: 4.4 },
        { name: 'on', v: [-.3, -.8], ang: 5.6 },
      ];
      const q = [Math.cos(s.qx), Math.sin(s.qx)];
      const scores = keys.map((k) => (Math.cos(k.ang) * q[0] + Math.sin(k.ang) * q[1]) * 3 * s.temp);
      const w = LA.softmax(scores);
      const out = [0, 0];
      keys.forEach((k, i) => { out[0] += w[i] * k.v[0]; out[1] += w[i] * k.v[1]; });

      // left: vectors; right: weights
      const cx = p.w * .27, cy = p.h * .5, R = Math.min(cx, cy) * .68;
      g.strokeStyle = cssVar('--border'); g.lineWidth = 1;
      g.beginPath(); g.arc(cx, cy, R, 0, 6.2832); g.stroke();
      const cols = SERIES();
      keys.forEach((k, i) => {
        const x = cx + Math.cos(k.ang) * R, y = cy - Math.sin(k.ang) * R;
        g.strokeStyle = alpha(cols[i], .35 + w[i] * .65);
        g.lineWidth = 1 + w[i] * 5;
        g.beginPath(); g.moveTo(cx, cy); g.lineTo(x, y); g.stroke();
        g.fillStyle = cols[i];
        g.beginPath(); g.arc(x, y, 4 + w[i] * 7, 0, 6.2832); g.fill();
        g.font = '11px -apple-system, sans-serif';
        g.textAlign = Math.cos(k.ang) > 0 ? 'left' : 'right';
        g.textBaseline = 'middle';
        g.fillText(' ' + k.name + ' ', x + Math.cos(k.ang) * 10, y - Math.sin(k.ang) * 10);
      });
      // query
      g.strokeStyle = cssVar('--text'); g.lineWidth = 2.6;
      g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + q[0] * R * .85, cy - q[1] * R * .85); g.stroke();
      g.fillStyle = cssVar('--text');
      g.font = '600 12px -apple-system, sans-serif';
      g.textAlign = 'center';
      g.fillText('query', cx + q[0] * R * 1.02, cy - q[1] * R * 1.02);

      // bars
      const bx = p.w * .55, bw = p.w * .34;
      g.font = '11px -apple-system, sans-serif';
      keys.forEach((k, i) => {
        const y = 34 + i * 30;
        g.fillStyle = cssVar('--text-dim'); g.textAlign = 'right'; g.textBaseline = 'middle';
        g.fillText(k.name, bx - 8, y);
        g.fillStyle = alpha(cols[i], .25);
        g.fillRect(bx, y - 9, bw, 18);
        g.fillStyle = cols[i];
        g.fillRect(bx, y - 9, bw * w[i], 18);
        g.fillStyle = cssVar('--text'); g.textAlign = 'left';
        g.font = '10.5px ui-monospace, monospace';
        g.fillText(w[i].toFixed(3), bx + bw + 7, y);
        g.font = '11px -apple-system, sans-serif';
      });
      g.fillStyle = cssVar('--text-faint');
      g.textAlign = 'left';
      g.font = '10.5px -apple-system, sans-serif';
      g.fillText('softmax(q·kᵢ/√d)  — the attention weights', bx - 40, 16);
      if (s.showV) {
        g.fillStyle = cssVar('--s3');
        g.font = '600 11px ui-monospace, monospace';
        g.fillText(`output = Σ wᵢ vᵢ = (${out[0].toFixed(3)}, ${out[1].toFixed(3)})`, bx - 40, p.h - 14);
      }
      P.readout({
        'entropy of weights': fmt(-w.reduce((a, v) => a + v * Math.log2(Math.max(v, 1e-12)), 0), 3) + ' bits',
        'max weight': fmt(Math.max(...w), 3),
        'effective # attended': fmt(2 ** (-w.reduce((a, v) => a + v * Math.log2(Math.max(v, 1e-12)), 0)), 2),
      });
    },
    caption: 'A query vector is compared against every key by dot product; softmax turns those scores into weights that sum to 1; the output is the weighted average of the **values**. Rotate the query and watch attention slide from one token to another. Raise the sharpness (that is the $1/\\sqrt{d}$ scaling in reverse) and it becomes nearly a hard lookup — which is why the scaling factor exists: without it, large $d$ makes dot products huge, softmax saturates, and gradients vanish.',
  });
};

V['attention-matrix'] = (host) => {
  const SENT = ['The', 'cat', 'that', 'the', 'dog', 'chased', 'ran', 'away'];
  panel(host, {
    title: 'The attention matrix — who looks at whom',
    height: 340,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      {
        type: 'select', key: 'pattern', label: 'head type', value: 'previous',
        options: [
          { value: 'previous', label: 'previous-token head' },
          { value: 'first', label: 'attend-to-first (attention sink)' },
          { value: 'induction', label: 'induction head (copy after match)' },
          { value: 'syntactic', label: 'syntactic (subject–verb)' },
          { value: 'diffuse', label: 'diffuse / averaging' },
        ],
      },
      { type: 'check', key: 'causal', label: 'causal mask (decoder)', value: true },
      { type: 'slider', key: 'temp', label: 'temperature', min: .2, max: 3, step: .05, value: 1 },
    ],
    draw(p, s, P) {
      const n = SENT.length;
      const raw = LA.zeros(n, n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        let sc = 0;
        if (s.pattern === 'previous') sc = j === i - 1 ? 5 : (j === i ? 1 : 0);
        else if (s.pattern === 'first') sc = j === 0 ? 5 : 0.6;
        else if (s.pattern === 'induction') {
          // attend to the token AFTER a previous occurrence of the current token
          sc = 0.3;
          for (let k = 1; k < i; k++) if (SENT[k - 1].toLowerCase() === SENT[i].toLowerCase() && j === k) sc = 5;
        } else if (s.pattern === 'syntactic') {
          const heads = { 5: 4, 6: 1, 7: 6, 2: 1, 4: 3, 1: 0 };
          sc = heads[i] === j ? 5 : .4;
        } else sc = 1 + Math.sin(i * .7 + j * 1.3) * .4;
        raw[i][j] = sc / s.temp;
      }
      const A = raw.map((row, i) => {
        const masked = row.map((v, j) => (s.causal && j > i ? -1e9 : v));
        return LA.softmax(masked);
      });

      p.clear();
      const g = p.ctx;
      const pad = 62;
      const size = Math.min(p.w - pad - 30, p.h - pad - 26);
      const cw = size / n;
      const x0 = pad, y0 = 30;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const v = A[i][j];
        g.fillStyle = s.causal && j > i ? alpha(cssVar('--bg'), 1) : mix(cssVar('--bg-inset'), cssVar('--s1'), Math.min(1, v * 1.6));
        g.fillRect(x0 + j * cw, y0 + i * cw, cw - .5, cw - .5);
        if (cw > 26 && !(s.causal && j > i)) {
          g.fillStyle = v > .5 ? cssVar('--bg') : alpha(cssVar('--text-faint'), .85);
          g.font = '9px ui-monospace, monospace';
          g.textAlign = 'center'; g.textBaseline = 'middle';
          g.fillText(v.toFixed(2).slice(1), x0 + j * cw + cw / 2, y0 + i * cw + cw / 2);
        }
      }
      g.font = '10.5px -apple-system, sans-serif';
      g.fillStyle = cssVar('--text-dim');
      for (let i = 0; i < n; i++) {
        g.textAlign = 'right'; g.textBaseline = 'middle';
        g.fillText(SENT[i], x0 - 6, y0 + i * cw + cw / 2);
        g.save();
        g.translate(x0 + i * cw + cw / 2, y0 + size + 6);
        g.rotate(-Math.PI / 4);
        g.textAlign = 'right'; g.textBaseline = 'middle';
        g.fillText(SENT[i], 0, 0);
        g.restore();
      }
      g.fillStyle = cssVar('--text-faint');
      g.font = '10px -apple-system, sans-serif';
      g.textAlign = 'left';
      g.fillText('key (attended to) →', x0, 18);
      g.save();
      g.translate(14, y0 + size / 2); g.rotate(-Math.PI / 2);
      g.textAlign = 'center';
      g.fillText('query (doing the looking) →', 0, 0);
      g.restore();

      const desc = {
        previous: 'Copies information from the token immediately before. The simplest useful head, and it appears in layer 0 of nearly every trained transformer.',
        first: 'Dumps attention on the first token. Looks useless — it is actually the model\'s "do nothing" option, a place to park probability mass. Critical for quantization and streaming.',
        induction: 'Finds a previous occurrence of the current token and attends to what followed it. This two-head circuit is the mechanism behind much of in-context learning.',
        syntactic: 'Links a word to its grammatical head. Emerges without any parse-tree supervision.',
        diffuse: 'Averages broadly — closer to a bag-of-words summary than a lookup.',
      }[s.pattern];
      P.readout({ 'pattern': desc });
    },
    caption: 'Row $i$ is where token $i$ **looks**; it sums to 1. The **causal mask** blanks the upper triangle: during training the model predicts every position at once, and a position must never see its own future. Toggle it off and you have an encoder (BERT-style) that sees the whole sentence. These named patterns are idealized versions of heads that mechanistic interpretability actually finds inside trained models.',
  });
};

V['multi-head'] = (host) => {
  panel(host, {
    title: 'Multi-head attention: split the budget, not the width',
    height: 280,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'dmodel', label: 'd_model', min: 64, max: 1024, step: 64, value: 512 },
      { type: 'slider', key: 'heads', label: 'number of heads', min: 1, max: 16, step: 1, value: 8 },
      { type: 'select', key: 'kind', label: 'variant', value: 'mha', options: [{ value: 'mha', label: 'MHA (each head has own K,V)' }, { value: 'gqa', label: 'GQA (groups share K,V)' }, { value: 'mqa', label: 'MQA (all share one K,V)' }] },
      { type: 'slider', key: 'groups', label: 'KV groups (GQA)', min: 1, max: 8, step: 1, value: 2 },
    ],
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const H = Math.round(s.heads), d = Math.round(s.dmodel);
      const dh = Math.floor(d / H);
      const kvHeads = s.kind === 'mha' ? H : s.kind === 'mqa' ? 1 : Math.min(Math.round(s.groups), H);
      const W = p.w - 60, x0 = 30, y = 56, barH = 34;
      const cw = W / H;
      // Q heads
      g.font = '10px -apple-system, sans-serif';
      g.textAlign = 'center'; g.textBaseline = 'middle';
      const cols = SERIES();
      for (let h = 0; h < H; h++) {
        g.fillStyle = alpha(cols[h % cols.length], .8);
        g.fillRect(x0 + h * cw, y, cw - 2, barH);
        if (cw > 26) { g.fillStyle = cssVar('--bg'); g.fillText('Q' + h, x0 + h * cw + cw / 2 - 1, y + barH / 2); }
      }
      // KV heads
      const kvcw = W / kvHeads;
      for (let h = 0; h < kvHeads; h++) {
        g.fillStyle = alpha(cols[(h * Math.ceil(H / kvHeads)) % cols.length], .35);
        g.fillRect(x0 + h * kvcw, y + barH + 26, kvcw - 2, barH);
        g.fillStyle = cssVar('--text');
        if (kvcw > 34) g.fillText('K,V' + (kvHeads > 1 ? h : ''), x0 + h * kvcw + kvcw / 2 - 1, y + barH + 26 + barH / 2);
      }
      // links
      g.strokeStyle = alpha(cssVar('--text-faint'), .5); g.lineWidth = 1;
      for (let h = 0; h < H; h++) {
        const kv = Math.floor(h / Math.ceil(H / kvHeads));
        g.beginPath();
        g.moveTo(x0 + h * cw + cw / 2, y + barH);
        g.lineTo(x0 + kv * kvcw + kvcw / 2, y + barH + 26);
        g.stroke();
      }
      g.fillStyle = cssVar('--text-dim');
      g.textAlign = 'left';
      g.font = '11px -apple-system, sans-serif';
      g.fillText(`${H} query heads × d_head = ${dh}`, x0, y - 14);
      g.fillText(`${kvHeads} key/value head${kvHeads > 1 ? 's' : ''} — this is what the KV cache stores`, x0, y + barH + 14);

      const kvPerTok = 2 * kvHeads * dh * 2; // bytes at fp16
      g.fillStyle = cssVar('--s3');
      g.font = '600 11.5px ui-monospace, monospace';
      g.fillText(`KV cache: ${kvPerTok} bytes/token/layer  (fp16)`, x0, p.h - 22);
      P.readout({
        'd_head': dh,
        'total Q params': (d * d).toLocaleString(),
        'total K+V params': (2 * d * kvHeads * dh).toLocaleString(),
        'KV cache vs MHA': fmt(kvHeads / H * 100, 1) + '%',
        'quality cost': s.kind === 'mqa' ? 'noticeable' : s.kind === 'gqa' ? 'small' : 'none (baseline)',
      });
    },
    caption: 'Total width stays $d_{\\text{model}}$ — heads *split* it rather than adding to it, so multi-head attention costs the same as single-head but can attend to several things at once. The catch shows up at inference: the KV cache stores keys and values for every past token, and it dominates memory. **GQA** lets several query heads share one KV head, cutting the cache by the sharing factor with minimal quality loss — which is why nearly every modern open model uses it.',
  });
};

V['attention-cost'] = (host) => {
  panel(host, {
    title: 'Why context length is expensive',
    height: 280,
    plot: { xlim: [2, 6], ylim: [0, 8], xlabel: 'log₁₀ sequence length', ylabel: 'log₁₀ relative cost' },
    controls: [
      { type: 'slider', key: 'd', label: 'd_model', min: 256, max: 8192, step: 256, value: 4096 },
      { type: 'slider', key: 'L', label: 'layers', min: 4, max: 120, step: 4, value: 32 },
      { type: 'check', key: 'flash', label: 'FlashAttention (memory, not FLOPs)', value: false },
    ],
    draw(p, s, P) {
      const d = s.d, L = s.L;
      p.clear().axes();
      p.clip();
      // per-layer FLOPs: attention 4*n^2*d ; MLP+proj ~ 12*n*d^2  (relative units)
      const attn = (n) => 4 * n * n * d;
      const mlp = (n) => 12 * n * d * d;
      const pts1 = [], pts2 = [], pts3 = [];
      for (let lg = 2; lg <= 6; lg += .02) {
        const n = 10 ** lg;
        pts1.push([lg, Math.log10(attn(n))-6]);
        pts2.push([lg, Math.log10(mlp(n))-6]);
        pts3.push([lg, Math.log10(s.flash ? n * d * 4 : n * n * 4)-6]);
      }
      p.line(pts1, { color: cssVar('--s2'), width: 2.6 });
      p.line(pts2, { color: cssVar('--s1'), width: 2.6 });
      p.line(pts3, { color: cssVar('--s3'), width: 2, dash: [5, 3] });
      // crossover
      const cross = 3 * d;
      if (cross > 100 && cross < 1e6) {
        p.line([[Math.log10(cross), 0], [Math.log10(cross), 8]], { color: alpha(cssVar('--danger'), .7), width: 1.4, dash: [4, 4] });
        p.text(Math.log10(cross), 7.4, ` attention overtakes MLP at n≈${Math.round(cross).toLocaleString()}`, { color: cssVar('--danger'), size: 10 });
      }
      p.clip(false);
      p.legend([
        { label: 'attention FLOPs  O(n²d)', color: cssVar('--s2') },
        { label: 'MLP FLOPs  O(nd²)', color: cssVar('--s1') },
        { label: s.flash ? 'attention memory (Flash) O(nd)' : 'attention memory O(n²)', color: cssVar('--s3'), dash: true },
      ], { pos: 'tl' });
      P.readout({
        'crossover n = 3·d_model': Math.round(3 * d).toLocaleString(),
        'at n = 128k, attention share': fmt(4 * 131072 * 131072 * d / (4 * 131072 ** 2 * d + 12 * 131072 * d * d) * 100, 1) + '%',
        'note': 'below the crossover, the MLP — not attention — dominates compute',
      });
    },
    caption: 'The famous $O(n^2)$ is real but often misunderstood. For a 4096-wide model, attention only overtakes the MLP in FLOPs past **~12k tokens** — below that, the quadratic term is not your bottleneck. What *was* catastrophic is the $n^2$ **memory** for the attention matrix, and that is exactly what FlashAttention removed by never materializing it (tiling + recomputation), turning memory back into $O(n)$ without changing the math at all.',
  });
};

/* ============================================================
   Positional encodings
   ============================================================ */

V['positional-encoding'] = (host) => {
  panel(host, {
    title: 'Positional encodings',
    height: 320,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'select', key: 'kind', label: 'scheme', value: 'sinusoid', options: [{ value: 'sinusoid', label: 'sinusoidal (original transformer)' }, { value: 'learned', label: 'learned absolute' }, { value: 'rope', label: 'RoPE (rotary)' }, { value: 'alibi', label: 'ALiBi (linear bias)' }] },
      { type: 'slider', key: 'seq', label: 'sequence length', min: 16, max: 128, step: 8, value: 64 },
      { type: 'slider', key: 'dim', label: 'dimensions shown', min: 8, max: 64, step: 4, value: 32 },
    ],
    draw(p, s, P) {
      const T = Math.round(s.seq), D = Math.round(s.dim);
      p.clear();
      const g = p.ctx;
      const M = [];
      if (s.kind === 'sinusoid') {
        for (let t = 0; t < T; t++) {
          const row = [];
          for (let i = 0; i < D; i++) {
            const k = Math.floor(i / 2);
            const w = 1 / 10000 ** (2 * k / D);
            row.push(i % 2 === 0 ? Math.sin(t * w) : Math.cos(t * w));
          }
          M.push(row);
        }
      } else if (s.kind === 'learned') {
        const r = rng(4);
        for (let t = 0; t < T; t++) M.push(Array.from({ length: D }, () => r.normal(0, .6)));
      } else if (s.kind === 'rope') {
        // show relative rotation: dot product between position m and n as function of m-n
        for (let t = 0; t < T; t++) {
          const row = [];
          for (let i = 0; i < D; i++) {
            const k = Math.floor(i / 2);
            const th = t / 10000 ** (2 * k / D);
            row.push(i % 2 === 0 ? Math.cos(th) : Math.sin(th));
          }
          M.push(row);
        }
      } else {
        for (let t = 0; t < T; t++) {
          const row = [];
          for (let i = 0; i < D; i++) {
            const slope = 2 ** (-(i + 1) * 8 / D);
            row.push(i < D ? -slope * t : 0);
          }
          M.push(row);
        }
      }
      const x0 = 46, y0 = 24;
      const cw = (p.w - x0 - 120) / D, ch = (p.h - y0 - 40) / T;
      let lo = Infinity, hi = -Infinity;
      for (const r of M) for (const v of r) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      for (let t = 0; t < T; t++) for (let i = 0; i < D; i++) {
        g.fillStyle = divergeMap((M[t][i] - lo) / (hi - lo || 1));
        g.fillRect(x0 + i * cw, y0 + t * ch, cw + .5, ch + .5);
      }
      g.fillStyle = cssVar('--text-dim');
      g.font = '10px -apple-system, sans-serif';
      g.textAlign = 'center';
      g.fillText('dimension →', x0 + D * cw / 2, p.h - 22);
      g.save(); g.translate(14, y0 + T * ch / 2); g.rotate(-Math.PI / 2);
      g.fillText('position →', 0, 0); g.restore();

      // side plot: similarity vs distance
      const sx = x0 + D * cw + 26, sw = p.w - sx - 14, sh = p.h - y0 - 44;
      g.strokeStyle = cssVar('--border');
      g.strokeRect(sx, y0, sw, sh);
      g.beginPath();
      for (let dlt = 0; dlt < Math.min(T, 60); dlt++) {
        let dot = 0;
        if (s.kind === 'alibi') dot = -(2 ** -4) * dlt;
        else dot = LA.dot(M[0], M[dlt] || M[0]) / D;
        const X = sx + dlt / Math.min(T - 1, 59) * sw;
        const Y = y0 + sh / 2 - dot * sh * .42;
        dlt ? g.lineTo(X, Y) : g.moveTo(X, Y);
      }
      g.strokeStyle = cssVar('--s2'); g.lineWidth = 2; g.stroke();
      g.fillStyle = cssVar('--text-faint');
      g.font = '9.5px -apple-system, sans-serif';
      g.textAlign = 'left';
      g.fillText('score vs. distance', sx + 3, y0 - 6);

      const notes = {
        sinusoid: 'Fixed, not learned. Different dimensions oscillate at geometrically spaced frequencies, so position is encoded like a binary clock. Extrapolates to unseen lengths in principle.',
        learned: 'A plain lookup table of vectors. Simple and effective — but there is no row for position 5000 if you only trained to 4096. Hard length limit.',
        rope: 'Rotates the query and key vectors by an angle proportional to position. The dot product then depends only on the RELATIVE offset — position information without any added vector.',
        alibi: 'Adds a distance-proportional penalty straight into the attention scores. No vectors at all, and it extrapolates to longer contexts remarkably well.',
      };
      P.readout({ 'how it works': notes[s.kind] });
    },
    caption: 'Attention is permutation-invariant — shuffle the tokens and the output shuffles identically. Something must inject order. **RoPE won** because rotating $q$ and $k$ makes $q_m^{\\mathsf T}k_n$ depend only on $m-n$: relative position, for free, with no extra parameters and no added vector to dilute the embedding. Modern long-context tricks (NTK scaling, YaRN) all work by stretching RoPE\'s frequencies.',
  });
};

V['rope-rotation'] = (host) => {
  panel(host, {
    title: 'RoPE: position as rotation',
    height: 300,
    plot: { xlim: [-1.6, 1.6], ylim: [-1.3, 1.3], equal: true },
    controls: [
      { type: 'slider', key: 'm', label: 'query position m', min: 0, max: 40, step: 1, value: 8 },
      { type: 'slider', key: 'n', label: 'key position n', min: 0, max: 40, step: 1, value: 3 },
      { type: 'slider', key: 'freq', label: 'frequency θ for this dim pair', min: .02, max: .8, step: .01, value: .25 },
    ],
    draw(p, s, P) {
      p.clear().axes({ nx: 5, ny: 4 });
      p.clip();
      const q0 = [.9, .3], k0 = [.7, -.5];
      const rot = (v, a) => [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)];
      const qm = rot(q0, s.m * s.freq), kn = rot(k0, s.n * s.freq);
      // draw arcs
      const g = p.ctx;
      g.strokeStyle = alpha(cssVar('--s1'), .25); g.lineWidth = 1;
      g.beginPath(); g.arc(p.px(0), p.py(0), Math.hypot(q0[0], q0[1]) * p.sx, 0, 6.2832); g.stroke();
      p.arrow(0, 0, q0[0], q0[1], { color: alpha(cssVar('--s1'), .45), width: 1.6, dash: [4, 3] });
      p.arrow(0, 0, k0[0], k0[1], { color: alpha(cssVar('--s2'), .45), width: 1.6, dash: [4, 3] });
      p.arrow(0, 0, qm[0], qm[1], { color: cssVar('--s1'), width: 3 });
      p.arrow(0, 0, kn[0], kn[1], { color: cssVar('--s2'), width: 3 });
      p.text(qm[0], qm[1], ' q rotated by mθ', { color: cssVar('--s1'), size: 11, weight: '600' });
      p.text(kn[0], kn[1], ' k rotated by nθ', { color: cssVar('--s2'), size: 11, weight: '600' });
      p.clip(false);
      const dot = LA.dot(qm, kn), dot0 = LA.dot(q0, k0);
      P.readout({
        'q·k at (m,n)': fmt(dot, 5),
        'relative offset m−n': s.m - s.n,
        'depends only on m−n': 'yes — try (m,n)=(8,3) then (30,25)',
        'unrotated q·k': fmt(dot0, 5),
      });
    },
    caption: 'Both vectors get rotated by an angle proportional to their position. Because rotation preserves angles between vectors, the dot product $q_m \\cdot k_n$ ends up depending **only on $m-n$**. Verify it: set $(m,n) = (8,3)$, note the value, then set $(30,25)$ — identical. Real RoPE does this on every pair of dimensions with a different $\\theta$, giving a multi-scale relative position code.',
  });
};

/* ============================================================
   Tokenization — a real BPE trainer
   ============================================================ */

V['bpe-tokenizer'] = (host) => {
  const DEFAULT = 'the cat sat on the mat. the cat ate the rat. a rat sat on a hat.';
  let merges = [], vocab = [], corpus = DEFAULT, tokensNow = [];

  function train(text, nMerges) {
    // word-level BPE, GPT-style: words keep a leading-space marker
    const words = text.toLowerCase().match(/\S+|\s+/g) || [];
    let seqs = [];
    const counts = new Map();
    for (const w of words) {
      if (/^\s+$/.test(w)) continue;
      const key = '▁' + w;
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    for (const [w, c] of counts) seqs.push({ sym: w.split(''), count: c });
    const base = new Set();
    for (const s of seqs) for (const ch of s.sym) base.add(ch);
    merges = [];
    for (let it = 0; it < nMerges; it++) {
      const pairs = new Map();
      for (const s of seqs) {
        for (let i = 0; i + 1 < s.sym.length; i++) {
          const k = s.sym[i] + ' ' + s.sym[i + 1];
          pairs.set(k, (pairs.get(k) || 0) + s.count);
        }
      }
      if (!pairs.size) break;
      let bk = null, bv = 0;
      for (const [k, v] of pairs) if (v > bv) { bv = v; bk = k; }
      if (bv < 2) break;
      const [a, b] = bk.split(' ');
      merges.push({ a, b, count: bv, token: a + b });
      for (const s of seqs) {
        const out = [];
        for (let i = 0; i < s.sym.length; i++) {
          if (i + 1 < s.sym.length && s.sym[i] === a && s.sym[i + 1] === b) { out.push(a + b); i++; }
          else out.push(s.sym[i]);
        }
        s.sym = out;
      }
    }
    vocab = [...base, ...merges.map((m) => m.token)];
    return seqs;
  }

  function encode(text) {
    const words = text.toLowerCase().match(/\S+/g) || [];
    const out = [];
    for (const w of words) {
      let sym = ('▁' + w).split('');
      let changed = true;
      while (changed) {
        changed = false;
        for (const m of merges) {
          for (let i = 0; i + 1 < sym.length; i++) {
            if (sym[i] === m.a && sym[i + 1] === m.b) {
              sym = [...sym.slice(0, i), m.token, ...sym.slice(i + 2)];
              changed = true;
              break;
            }
          }
          if (changed) break;
        }
      }
      out.push(...sym);
    }
    return out;
  }

  const P = panel(host, {
    title: 'Byte-pair encoding, trained live',
    height: 200,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'nm', label: 'number of merges', min: 0, max: 60, step: 1, value: 12 },
    ],
    draw(p, s, PP) {
      train(corpus, Math.round(s.nm));
      tokensNow = encode(testEl.value || corpus);
      p.clear();
      const g = p.ctx;
      // merge list
      g.font = '11px ui-monospace, monospace';
      g.textBaseline = 'middle';
      const show = merges.slice(-24);
      const colw = p.w / 3;
      g.fillStyle = cssVar('--text-faint');
      g.font = '10px -apple-system, sans-serif';
      g.fillText('learned merges (most recent last) — each adds one token to the vocabulary', 8, 12);
      g.font = '11px ui-monospace, monospace';
      show.forEach((m, i) => {
        const col = Math.floor(i / 8), row = i % 8;
        const x = 10 + col * colw, y = 32 + row * 19;
        g.fillStyle = cssVar('--text-dim');
        g.fillText(`${(merges.length - show.length + i + 1).toString().padStart(2)}.`, x, y);
        g.fillStyle = cssVar('--s1');
        g.fillText(`"${m.a}" + "${m.b}"`, x + 24, y);
        g.fillStyle = cssVar('--s3');
        g.fillText(`→ "${m.token}"`, x + 24 + g.measureText(`"${m.a}" + "${m.b}"`).width + 8, y);
        g.fillStyle = cssVar('--text-faint');
        g.fillText(`×${m.count}`, x + colw - 42, y);
      });
      const chars = (testEl.value || corpus).replace(/\s/g, '').length;
      PP.readout({
        'vocabulary size': vocab.length,
        'merges applied': merges.length,
        'tokens for the test string': tokensNow.length,
        'characters': chars,
        'compression': fmt(chars / Math.max(tokensNow.length, 1), 2) + ' chars/token',
      });
      renderTokens();
    },
    caption: 'BPE starts from characters and greedily merges the most frequent adjacent pair, over and over. Common words collapse into single tokens; rare ones stay in pieces. **Slide merges from 0 to 60** and watch "the" become one token. This is exactly how GPT-family tokenizers are built (over bytes rather than characters, with ~100k merges) — and it explains why models are bad at spelling and arithmetic: they never see the letters.',
  });

  // custom DOM: editable corpus + test string + token chips
  const extra = P.extra;
  extra.innerHTML = `
    <div style="display:grid;gap:8px;margin-top:6px">
      <label style="font-size:11px;color:var(--text-faint)">training corpus
        <textarea id="bpe-corpus" spellcheck="false" style="width:100%;height:52px;margin-top:3px;background:var(--bg-inset);border:1px solid var(--border);border-radius:6px;color:var(--text);font-family:var(--mono);font-size:11.5px;padding:6px;resize:vertical;outline:none"></textarea>
      </label>
      <label style="font-size:11px;color:var(--text-faint)">tokenize this
        <input id="bpe-test" spellcheck="false" style="width:100%;margin-top:3px;height:28px;background:var(--bg-inset);border:1px solid var(--border);border-radius:6px;color:var(--text);font-family:var(--mono);font-size:12px;padding:0 8px;outline:none">
      </label>
      <div id="bpe-out" style="display:flex;flex-wrap:wrap;gap:3px;min-height:26px"></div>
    </div>`;
  const corpusEl = extra.querySelector('#bpe-corpus');
  const testEl = extra.querySelector('#bpe-test');
  const outEl = extra.querySelector('#bpe-out');
  corpusEl.value = DEFAULT;
  testEl.value = 'the cat sat on a rat';

  function renderTokens() {
    const cols = SERIES();
    outEl.innerHTML = tokensNow.map((t, i) =>
      `<span style="font-family:var(--mono);font-size:11.5px;padding:2px 6px;border-radius:4px;background:${alpha(cols[i % cols.length], .22)};border:1px solid ${alpha(cols[i % cols.length], .5)};color:var(--text)">${t.replace(/▁/g, '␣').replace(/</g, '&lt;')}</span>`).join('');
  }
  corpusEl.addEventListener('input', () => { corpus = corpusEl.value; P.redraw(); });
  testEl.addEventListener('input', () => P.redraw());
  P.redraw();
};

/* ============================================================
   Language modeling: n-gram model you can sample from
   ============================================================ */

V['language-model'] = (host) => {
  const TEXT = `the quick brown fox jumps over the lazy dog. the dog barks and the fox runs.
a quick fox is a clever fox. the lazy dog sleeps under the old tree.
the tree is tall and the dog is small. every dog has its day and every fox has its den.`;
  let model = null, order = 3;

  function build(n) {
    const t = TEXT.toLowerCase().replace(/\s+/g, ' ');
    const m = new Map();
    for (let i = 0; i + n < t.length; i++) {
      const ctx = t.slice(i, i + n), nxt = t[i + n];
      if (!m.has(ctx)) m.set(ctx, new Map());
      const d = m.get(ctx);
      d.set(nxt, (d.get(nxt) || 0) + 1);
    }
    return m;
  }

  panel(host, {
    title: 'A language model is a next-token probability table',
    height: 280,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'order', label: 'context length (characters)', min: 1, max: 6, step: 1, value: 3 },
      { type: 'slider', key: 'temp', label: 'temperature', min: .1, max: 2.5, step: .05, value: 1 },
      { type: 'button', label: 'generate 90 chars', primary: true, onClick: (s) => (s.gen = generate(s)) },
    ],
    draw(p, s, P) {
      order = Math.round(s.order);
      model = build(order);
      p.clear();
      const g = p.ctx;
      const ctx = (s.ctxStr || 'the').slice(-order).padStart(order, ' ');
      const dist = model.get(ctx);
      g.font = '12px ui-monospace, monospace';
      g.textBaseline = 'middle';
      g.fillStyle = cssVar('--text-dim');
      g.fillText(`context: "${ctx}"  →  next character distribution`, 10, 16);
      if (!dist) {
        g.fillStyle = cssVar('--danger');
        g.fillText('unseen context — the model has no idea (this is the sparsity problem)', 10, 44);
      } else {
        const entries = [...dist.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
        const tot = [...dist.values()].reduce((a, b) => a + b, 0);
        const logits = entries.map(([c, v]) => Math.log(v / tot));
        const probs = LA.softmax(logits, s.temp);
        const bw = p.w - 130;
        entries.forEach(([c, v], i) => {
          const y = 42 + i * 21;
          g.fillStyle = cssVar('--text');
          g.textAlign = 'right';
          g.fillText(c === ' ' ? '␣' : c, 40, y);
          g.fillStyle = alpha(cssVar('--s1'), .2);
          g.fillRect(52, y - 8, bw, 16);
          g.fillStyle = cssVar('--s1');
          g.fillRect(52, y - 8, bw * probs[i], 16);
          g.fillStyle = cssVar('--text-dim');
          g.textAlign = 'left';
          g.font = '10.5px ui-monospace, monospace';
          g.fillText(probs[i].toFixed(3), 52 + bw + 8, y);
          g.font = '12px ui-monospace, monospace';
        });
        const H = -probs.reduce((a, v) => a + v * Math.log2(Math.max(v, 1e-12)), 0);
        P.readout({
          'contexts memorized': model.size,
          'entropy of this distribution': fmt(H, 3) + ' bits',
          'perplexity': fmt(2 ** H, 2),
          'generated': s.gen || '(press generate)',
        });
      }
    },
    caption: 'This is a genuine — if tiny — language model: count how often each character follows each context, normalize, sample. Raise the context length and generations get more fluent *and* more plagiarized, because longer contexts appear fewer times (at order 6 most contexts have exactly one continuation). **That trade-off is the entire motivation for neural language models**: they generalize across similar contexts instead of memorizing exact strings.',
  });

  function generate(s) {
    let out = 'the ';
    for (let i = 0; i < 90; i++) {
      const ctx = out.slice(-order);
      const dist = model.get(ctx);
      if (!dist) break;
      const entries = [...dist.entries()];
      const tot = entries.reduce((a, e) => a + e[1], 0);
      const probs = LA.softmax(entries.map((e) => Math.log(e[1] / tot)), s.temp);
      let r = Math.random(), k = 0;
      while (k < probs.length - 1 && r > probs[k]) { r -= probs[k]; k++; }
      out += entries[k][0];
    }
    s.ctxStr = out.slice(-order);
    return out;
  }
};

/* ============================================================
   Sampling / decoding
   ============================================================ */

V['sampling-strategies'] = (host) => {
  const WORDS = ['the', 'a', 'this', 'my', 'our', 'their', 'some', 'one', 'that', 'every', 'zebra', 'quantum', 'plausible', 'green'];
  const BASE = [3.9, 3.2, 2.4, 2.0, 1.6, 1.3, 1.0, 0.7, 0.4, 0.1, -1.4, -2.2, -2.8, -3.5];

  panel(host, {
    title: 'Decoding: turning logits into a choice',
    height: 300,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'temp', label: 'temperature', min: .05, max: 2.5, step: .05, value: 1 },
      { type: 'slider', key: 'topk', label: 'top-k (0 = off)', min: 0, max: 14, step: 1, value: 0 },
      { type: 'slider', key: 'topp', label: 'top-p / nucleus', min: .1, max: 1, step: .01, value: 1 },
      { type: 'slider', key: 'rep', label: 'repetition penalty', min: 1, max: 2, step: .05, value: 1 },
      { type: 'check', key: 'greedy', label: 'greedy (argmax)', value: false },
    ],
    draw(p, s, P) {
      let logits = BASE.slice();
      // pretend 'the' was already used
      if (s.rep > 1) logits[0] /= s.rep;
      let probs = LA.softmax(logits, s.greedy ? .02 : s.temp);
      const idx = probs.map((v, i) => i).sort((a, b) => probs[b] - probs[a]);
      const kept = new Set();
      if (s.topk > 0) idx.slice(0, Math.round(s.topk)).forEach((i) => kept.add(i));
      else idx.forEach((i) => kept.add(i));
      if (s.topp < 1) {
        const nuc = new Set();
        let cum = 0;
        for (const i of idx) { if (!kept.has(i)) continue; nuc.add(i); cum += probs[i]; if (cum >= s.topp) break; }
        for (const i of [...kept]) if (!nuc.has(i)) kept.delete(i);
      }
      const renorm = probs.map((v, i) => (kept.has(i) ? v : 0));
      const z = renorm.reduce((a, b) => a + b, 0) || 1;
      const final = renorm.map((v) => v / z);

      p.clear();
      const g = p.ctx;
      const n = WORDS.length;
      const rowH = Math.min(19, (p.h - 30) / n);
      const bw = p.w - 190;
      g.font = '11px ui-monospace, monospace';
      g.textBaseline = 'middle';
      for (let r = 0; r < n; r++) {
        const i = idx[r];
        const y = 16 + r * rowH;
        const on = kept.has(i);
        g.fillStyle = on ? cssVar('--text') : cssVar('--text-faint');
        g.textAlign = 'right';
        g.fillText(WORDS[i], 74, y);
        // original prob (faint) + final prob
        g.fillStyle = alpha(cssVar('--text-faint'), .22);
        g.fillRect(82, y - rowH / 2 + 2, bw * probs[i] / Math.max(...probs), rowH - 4);
        g.fillStyle = on ? cssVar('--s1') : alpha(cssVar('--danger'), .3);
        g.fillRect(82, y - rowH / 2 + 2, bw * final[i] / Math.max(...final, 1e-9), rowH - 4);
        g.fillStyle = on ? cssVar('--text-dim') : cssVar('--text-faint');
        g.textAlign = 'left';
        g.font = '10px ui-monospace, monospace';
        g.fillText(on ? final[i].toFixed(3) : 'cut', 82 + bw + 8, y);
        g.font = '11px ui-monospace, monospace';
      }
      const H = -final.filter((v) => v > 0).reduce((a, v) => a + v * Math.log2(v), 0);
      P.readout({
        'candidates kept': kept.size + ' / ' + n,
        'entropy': fmt(H, 3) + ' bits',
        'perplexity': fmt(2 ** H, 2),
        'P(top token)': fmt(Math.max(...final), 3),
        'style': s.greedy ? 'deterministic — repetitive but safe' : s.temp < .5 ? 'conservative' : s.temp > 1.3 ? 'creative / risky' : 'balanced',
      });
    },
    caption: 'Faint bars are the raw distribution; solid bars are what you actually sample from. **Temperature** rescales the logits before softmax — below 1 it sharpens, above 1 it flattens. **Top-k** keeps a fixed number of candidates; **top-p** keeps however many are needed to reach probability mass $p$, which adapts to how confident the model is. Push temperature to 2 with top-p off and watch "zebra" become reachable — that is where hallucinated nonsense comes from.',
  });
};

/* ============================================================
   Scaling laws
   ============================================================ */

V['scaling-laws'] = (host) => {
  panel(host, {
    title: 'Scaling laws and the compute-optimal frontier',
    height: 310,
    plot: { xlim: [17, 26], ylim: [1.4, 4.2], xlabel: 'log₁₀ training compute (FLOPs)', ylabel: 'loss (nats/token)' },
    controls: [
      { type: 'slider', key: 'budget', label: 'compute budget (log₁₀ FLOPs)', min: 18, max: 26, step: .1, value: 22 },
      { type: 'select', key: 'view', label: 'view', value: 'frontier', options: [{ value: 'frontier', label: 'loss vs compute' }, { value: 'alloc', label: 'params vs tokens allocation' }] },
    ],
    draw(p, s, P) {
      // Chinchilla-style parametric fit:  L(N,D) = E + A/N^a + B/D^b
      const E = 1.69, A = 406.4, a = 0.34, B = 410.7, b = 0.28;
      const L = (N, D) => E + A / N ** a + B / D ** b;
      const C = (N, D) => 6 * N * D;

      if (s.view === 'frontier') {
        p.setLim([17, 26], [1.4, 4.2]);
        p.clear().axes();
        p.clip();
        // several fixed model sizes, each trained for varying tokens
        const sizes = [1e8, 1e9, 1e10, 7e10, 5e11];
        const cols = SERIES();
        sizes.forEach((N, i) => {
          const pts = [];
          for (let lgD = 8.5; lgD <= 13.5; lgD += .05) {
            const D = 10 ** lgD;
            pts.push([Math.log10(C(N, D)), L(N, D)]);
          }
          p.line(pts, { color: alpha(cols[i], .8), width: 1.8 });
        });
        // optimal frontier
        const front = [];
        for (let lgC = 17; lgC <= 26; lgC += .05) {
          const Cb = 10 ** lgC;
          let best = Infinity, bN = 0;
          for (let lgN = 6; lgN <= 13; lgN += .02) {
            const N = 10 ** lgN, D = Cb / (6 * N);
            if (D < 1e6) continue;
            const l = L(N, D);
            if (l < best) { best = l; bN = N; }
          }
          front.push([lgC, best]);
        }
        p.line(front, { color: cssVar('--text'), width: 2.8 });
        p.line([[s.budget, 1.4], [s.budget, 4.2]], { color: cssVar('--danger'), width: 1.4, dash: [4, 4] });
        p.clip(false);
        p.legend([...sizes.map((N, i) => ({ label: `N = ${fmtN(N)} params`, color: SERIES()[i] })), { label: 'compute-optimal frontier', color: cssVar('--text') }], { pos: 'tr' });
      } else {
        p.setLim([18, 26], [7, 14]);
        p.xlabel = 'log₁₀ compute (FLOPs)'; p.ylabel = 'log₁₀ count';
        p.clear().axes();
        p.clip();
        const pN = [], pD = [];
        for (let lgC = 18; lgC <= 26; lgC += .05) {
          const Cb = 10 ** lgC;
          let best = Infinity, bN = 0, bD = 0;
          for (let lgN = 6; lgN <= 13; lgN += .02) {
            const N = 10 ** lgN, D = Cb / (6 * N);
            if (D < 1e6) continue;
            const l = L(N, D);
            if (l < best) { best = l; bN = N; bD = D; }
          }
          pN.push([lgC, Math.log10(bN)]);
          pD.push([lgC, Math.log10(bD)]);
        }
        p.line(pN, { color: cssVar('--s1'), width: 2.6 });
        p.line(pD, { color: cssVar('--s2'), width: 2.6 });
        p.line([[s.budget, 7], [s.budget, 14]], { color: cssVar('--danger'), width: 1.4, dash: [4, 4] });
        p.clip(false);
        p.legend([{ label: 'optimal parameters N', color: cssVar('--s1') }, { label: 'optimal tokens D', color: cssVar('--s2') }], { pos: 'tl' });
      }

      // report optimum at the chosen budget
      const Cb = 10 ** s.budget;
      let best = Infinity, bN = 0, bD = 0;
      for (let lgN = 6; lgN <= 13; lgN += .005) {
        const N = 10 ** lgN, D = Cb / (6 * N);
        if (D < 1e6) continue;
        const l = L(N, D);
        if (l < best) { best = l; bN = N; bD = D; }
      }
      P.readout({
        'budget': `10^${fmt(s.budget, 1)} FLOPs`,
        'optimal params': fmtN(bN),
        'optimal tokens': fmtN(bD),
        'tokens per param': fmt(bD / bN, 1),
        'predicted loss': fmt(best, 4),
        'GPT-3 (175B, 300B tok)': 'well left of optimal — undertrained',
      });
    },
    caption: 'Each colored curve is one model size trained for more and more tokens; the black curve is the lower envelope — the best loss any model achieves at that compute. **Chinchilla\'s finding**: the optimum sits at roughly 20 tokens per parameter, meaning GPT-3-era models were far too large for their token budgets. Switch to the allocation view and note both curves have slope ≈0.5: double the compute, and you should grow the model *and* the data by ~√2 each. (Inference cost pushes production models even smaller and more data-heavy than this.)',
  });
};

function fmtN(x) {
  if (x >= 1e12) return (x / 1e12).toFixed(1) + 'T';
  if (x >= 1e9) return (x / 1e9).toFixed(1) + 'B';
  if (x >= 1e6) return (x / 1e6).toFixed(1) + 'M';
  if (x >= 1e3) return (x / 1e3).toFixed(1) + 'K';
  return x.toFixed(0);
}

/* ---------- transformer size calculator ---------- */

V['transformer-calculator'] = (host) => {
  panel(host, {
    title: 'Transformer budget calculator',
    height: 250,
    noCanvas: true,
    controls: [
      { type: 'slider', key: 'L', label: 'layers', min: 2, max: 128, step: 1, value: 32 },
      { type: 'slider', key: 'd', label: 'd_model', min: 128, max: 16384, step: 128, value: 4096 },
      { type: 'slider', key: 'ff', label: 'FFN multiplier', min: 2, max: 8, step: .5, value: 4 },
      { type: 'slider', key: 'vocab', label: 'vocab (×1000)', min: 8, max: 256, step: 8, value: 128 },
      { type: 'slider', key: 'seq', label: 'context length (×1024)', min: 1, max: 128, step: 1, value: 8 },
      { type: 'slider', key: 'tok', label: 'training tokens (log₁₀)', min: 9, max: 14, step: .1, value: 12.3 },
    ],
    draw(p, s, P) {
      const L = Math.round(s.L), d = Math.round(s.d), V = Math.round(s.vocab) * 1000;
      const seq = Math.round(s.seq) * 1024;
      const attnP = 4 * d * d;
      const ffP = 2 * s.ff * d * d;
      const perLayer = attnP + ffP;
      const embed = V * d;
      const N = L * perLayer + embed * 2;
      const D = 10 ** s.tok;
      const flops = 6 * N * D;
      const kvBytes = 2 * L * d * 2; // fp16, MHA
      const gpuHours = flops / (4e14 * 0.4) / 3600;  // H100 bf16 ~ 400 TFLOP/s @ 40% MFU

      const out = P.extra;
      const row = (k, v, hi) => `<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--border-soft);font-size:13px">
        <span style="color:var(--text-dim)">${k}</span><b style="font-family:var(--mono);color:${hi ? 'var(--s3)' : 'var(--text)'}">${v}</b></div>`;
      out.innerHTML = `
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0 22px">
          <div>
            ${row('parameters per layer', fmtN(perLayer))}
            ${row('embedding + unembedding', fmtN(embed * 2))}
            ${row('total parameters N', fmtN(N), true)}
            ${row('memory, bf16 weights', fmtN(N * 2) + 'B')}
            ${row('memory, Adam training', fmtN(N * 16) + 'B', true)}
          </div>
          <div>
            ${row('training FLOPs ≈ 6ND', flops.toExponential(2), true)}
            ${row('H100-hours @ 40% MFU', fmtN(gpuHours))}
            ${row('KV cache per token', (kvBytes / 1024).toFixed(1) + ' KB')}
            ${row('KV cache at full context', fmtN(kvBytes * seq) + 'B', true)}
            ${row('Chinchilla-optimal tokens', fmtN(N * 20))}
          </div>
        </div>
        <div style="font-size:12px;color:var(--text-faint);margin-top:10px">
          Training memory assumes fp32 master weights + Adam moments (≈16 bytes/param) before any sharding.
          Real runs cut this with ZeRO/FSDP, activation checkpointing, and 8-bit optimizers.
        </div>`;
      P.readout({
        'N': fmtN(N),
        'tokens/param': fmt(D / N, 1),
        'verdict': D / N < 10 ? 'undertrained for this size' : D / N > 100 ? 'small model, lots of data (inference-optimal)' : 'near Chinchilla-optimal',
      });
    },
    caption: 'Two formulas do most of the work in practice: **parameters** $\\approx 12Ld^2$ for a standard block (4 for attention projections, 8 for a 4× FFN), and **training compute** $\\approx 6ND$ (2 for the forward pass, 4 for backward). Try d_model 8192 / 80 layers to land near a 70B model. Then look at the KV cache line — at long context it can exceed the weights themselves, which is why GQA, MQA, and cache quantization exist.',
  });
};

/* ---------- mixture of experts ---------- */

V['moe-routing'] = (host) => {
  panel(host, {
    title: 'Mixture of experts: more parameters, same FLOPs',
    height: 300,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'E', label: 'experts', min: 2, max: 16, step: 1, value: 8 },
      { type: 'slider', key: 'k', label: 'top-k routed', min: 1, max: 4, step: 1, value: 2 },
      { type: 'slider', key: 'balance', label: 'load-balancing strength', min: 0, max: 1, step: .02, value: .6 },
      { type: 'play' },
    ],
    state: { t: 0 },
    animate(s) { s.t += 1; },
    draw(p, s, P) {
      const E = Math.round(s.E), k = Math.round(s.k);
      const nTok = 12;
      const r = rng(20 + Math.floor((s.t || 0) / 20));
      p.clear();
      const g = p.ctx;
      const y0 = 34, tokY = y0, expY = p.h - 60;
      const tokW = (p.w - 60) / nTok, expW = (p.w - 60) / E;
      const cols = SERIES();
      // routing: skewed logits, optionally balanced
      const load = new Array(E).fill(0);
      const routes = [];
      for (let t = 0; t < nTok; t++) {
        let logits = Array.from({ length: E }, (_, e) => r.normal(0, 1) + (e < 2 ? 1.6 : 0) * (1 - s.balance));
        if (s.balance > 0) logits = logits.map((v, e) => v - s.balance * load[e] * 1.5);
        const idx = logits.map((v, i) => i).sort((x, y2) => logits[y2] - logits[x]).slice(0, k);
        const w = LA.softmax(idx.map((i) => logits[i]));
        idx.forEach((e) => load[e]++);
        routes.push({ idx, w });
      }
      // draw
      for (let t = 0; t < nTok; t++) {
        const x = 30 + t * tokW + tokW / 2;
        g.fillStyle = cssVar('--s1');
        g.beginPath(); g.arc(x, tokY, 7, 0, 6.2832); g.fill();
        routes[t].idx.forEach((e, j) => {
          const ex = 30 + e * expW + expW / 2;
          g.strokeStyle = alpha(cols[e % cols.length], .35 + routes[t].w[j] * .5);
          g.lineWidth = .8 + routes[t].w[j] * 3;
          g.beginPath(); g.moveTo(x, tokY + 8); g.lineTo(ex, expY - 14); g.stroke();
        });
      }
      const maxLoad = Math.max(...load, 1);
      for (let e = 0; e < E; e++) {
        const x = 30 + e * expW;
        const h = 34;
        g.fillStyle = alpha(cols[e % cols.length], .3 + .5 * (load[e] / maxLoad));
        g.fillRect(x + 3, expY - 12, expW - 6, h);
        g.strokeStyle = cols[e % cols.length]; g.lineWidth = 1.2;
        g.strokeRect(x + 3, expY - 12, expW - 6, h);
        g.fillStyle = cssVar('--text');
        g.font = '10px -apple-system, sans-serif';
        g.textAlign = 'center'; g.textBaseline = 'middle';
        if (expW > 30) g.fillText('E' + e, x + expW / 2, expY + 5);
        g.fillStyle = cssVar('--text-faint');
        g.font = '9px ui-monospace, monospace';
        g.fillText(load[e], x + expW / 2, expY + 30);
      }
      g.fillStyle = cssVar('--text-dim');
      g.font = '11px -apple-system, sans-serif';
      g.textAlign = 'left';
      g.fillText('tokens', 8, tokY);
      g.fillText('experts', 8, expY + 5);
      const cv = Math.sqrt(load.reduce((a, v) => a + (v - nTok * k / E) ** 2, 0) / E) / (nTok * k / E);
      P.readout({
        'total FFN parameters': `${E}×`,
        'FLOPs per token': `${k}× (only ${k} of ${E} experts run)`,
        'load imbalance (CV)': fmt(cv, 3),
        'risk': cv > .6 ? 'expert collapse — a few experts get everything ✗' : 'reasonably balanced ✓',
      });
    },
    caption: 'A router picks $k$ experts per token, so the model holds $E\\times$ the FFN parameters while each token only pays for $k$ of them. The failure mode is **expert collapse**: without pressure, the router sends everything to a couple of favourites and the rest never train. Drag load-balancing to 0 and watch the counts skew. Real systems add an auxiliary balancing loss (or a bias-based scheme, as in DeepSeek-V3) and cap each expert\'s capacity.',
  });
};

/* ---------- KV cache ---------- */

V['kv-cache'] = (host) => {
  panel(host, {
    title: 'Why generation is memory-bound',
    height: 290,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'pos', label: 'tokens generated', min: 1, max: 14, step: 1, value: 5 },
      { type: 'check', key: 'cache', label: 'use KV cache', value: true },
    ],
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const n = Math.round(s.pos), N = 14;
      const cw = (p.w - 80) / N, ch = 22;
      const y0 = 40;
      g.font = '10px ui-monospace, monospace';
      g.textBaseline = 'middle';
      g.textAlign = 'center';
      for (let step = 0; step < n; step++) {
        const y = y0 + step * (ch + 3);
        g.fillStyle = cssVar('--text-faint');
        g.textAlign = 'right';
        g.fillText(`step ${step + 1}`, 56, y + ch / 2);
        g.textAlign = 'center';
        for (let t = 0; t <= step; t++) {
          const recompute = !s.cache;
          const isNew = t === step;
          g.fillStyle = isNew ? cssVar('--s2') : (recompute ? alpha(cssVar('--s2'), .55) : alpha(cssVar('--s1'), .28));
          g.fillRect(64 + t * cw, y, cw - 2, ch);
          if (cw > 22) {
            g.fillStyle = isNew || recompute ? cssVar('--bg') : cssVar('--text-faint');
            g.fillText(isNew ? 'new' : (recompute ? 'redo' : '✓'), 64 + t * cw + cw / 2 - 1, y + ch / 2);
          }
        }
      }
      const work = s.cache ? n : n * (n + 1) / 2;
      g.fillStyle = cssVar('--text-dim');
      g.font = '11.5px -apple-system, sans-serif';
      g.textAlign = 'left';
      g.fillText(`key/value vectors computed: ${work}`, 64, y0 + n * (ch + 3) + 18);
      g.fillStyle = cssVar('--s3');
      g.fillText(s.cache ? 'orange = computed now · blue = read from cache' : 'orange = recomputed from scratch every step', 64, y0 + n * (ch + 3) + 36);
      P.readout({
        'KV vectors computed': work,
        'without cache': `O(n²) — ${n * (n + 1) / 2}`,
        'with cache': `O(n) — ${n}`,
        'the new bottleneck': 'reading the whole cache + all weights from HBM for ONE token',
        'arithmetic intensity': 'terrible → generation is memory-bandwidth-bound, not compute-bound',
      });
    },
    caption: 'Without a cache, generating token $n$ recomputes keys and values for all $n-1$ previous tokens — quadratic waste, since those never change. Caching makes it linear. But now note what each step actually does: it reads *every weight and the entire cache* from memory to produce **one token**. The GPU\'s FLOPs sit idle waiting on bandwidth. That single fact explains batching, speculative decoding, quantization, and paged attention.',
  });
};

/* ---------- embeddings ---------- */

V['embedding-space'] = (host) => {
  const WORDS = {
    king: [2.2, 1.6], queen: [1.5, 2.3], man: [1.9, .3], woman: [1.2, 1.0],
    prince: [2.6, 1.2], princess: [1.9, 1.9], boy: [2.3, -.2], girl: [1.6, .5],
    paris: [-1.8, 1.9], france: [-2.4, 1.2], rome: [-1.2, 2.2], italy: [-1.8, 1.5],
    tokyo: [-.8, 2.6], japan: [-1.4, 1.9],
    dog: [.4, -1.9], cat: [.9, -1.7], puppy: [.1, -2.4], kitten: [.6, -2.2],
    run: [-2.6, -1.4], running: [-2.2, -1.9], walk: [-2.9, -.9], walking: [-2.5, -1.4],
  };
  panel(host, {
    title: 'Embeddings: meaning as direction',
    height: 330,
    plot: { xlim: [-3.6, 3.6], ylim: [-3, 3], equal: true },
    controls: [
      { type: 'select', key: 'a', label: 'A', value: 'king', options: Object.keys(WORDS) },
      { type: 'select', key: 'b', label: '− B', value: 'man', options: Object.keys(WORDS) },
      { type: 'select', key: 'c', label: '+ C', value: 'woman', options: Object.keys(WORDS) },
      { type: 'check', key: 'arrows', label: 'show analogy arrows', value: true },
    ],
    draw(p, s, P) {
      p.clear().axes({ nx: 5, ny: 4 });
      p.clip();
      const A = WORDS[s.a], B = WORDS[s.b], C = WORDS[s.c];
      const target = [A[0] - B[0] + C[0], A[1] - B[1] + C[1]];
      // nearest word to target, excluding inputs
      let best = null, bd = Infinity;
      for (const [w, v] of Object.entries(WORDS)) {
        if ([s.a, s.b, s.c].includes(w)) continue;
        const d = (v[0] - target[0]) ** 2 + (v[1] - target[1]) ** 2;
        if (d < bd) { bd = d; best = w; }
      }
      for (const [w, v] of Object.entries(WORDS)) {
        const hot = [s.a, s.b, s.c, best].includes(w);
        p.points([v], { r: hot ? 5.5 : 3.5, color: hot ? cssVar('--s2') : alpha(cssVar('--s1'), .55) });
        p.text(v[0], v[1] + .16, w, { color: hot ? cssVar('--text') : cssVar('--text-faint'), size: hot ? 11.5 : 10, align: 'center', weight: hot ? '600' : '' });
      }
      if (s.arrows) {
        p.arrow(B[0], B[1], A[0], A[1], { color: cssVar('--s3'), width: 2 });
        p.arrow(C[0], C[1], target[0], target[1], { color: cssVar('--s3'), width: 2, dash: [5, 3] });
        p.points([target], { r: 6, color: 'transparent', stroke: cssVar('--s5'), strokeWidth: 2 });
      }
      p.clip(false);
      const cos = (u, v) => LA.dot(u, v) / (LA.norm(u) * LA.norm(v));
      P.readout({
        'A − B + C': `${s.a} − ${s.b} + ${s.c}`,
        'nearest word': best,
        'distance to target': fmt(Math.sqrt(bd), 3),
        'cos(A, C)': fmt(cos(A, C), 3),
      });
    },
    caption: 'These coordinates are hand-placed to make the geometry legible, but the structure is real: in trained embeddings, **consistent relationships become consistent directions**. "king − man + woman ≈ queen" works because the male→female offset is roughly the same vector everywhere. Try `paris − france + japan`. The caveat worth knowing: analogy accuracy in real embeddings is far shakier than the famous examples suggest, and the offsets also encode social biases present in the training text.',
  });
};

/* ---------- in-context learning ---------- */

V['in-context-learning'] = (host) => {
  panel(host, {
    title: 'In-context learning: the prompt is a training set',
    height: 280,
    plot: { xlim: [0, 32], ylim: [0, 1], xlabel: 'examples in the prompt (shots)', ylabel: 'accuracy' },
    controls: [
      { type: 'slider', key: 'size', label: 'model size (log₁₀ params)', min: 7, max: 12, step: .1, value: 10 },
      { type: 'check', key: 'cot', label: 'chain-of-thought prompting', value: false },
    ],
    draw(p, s, P) {
      p.clear().axes();
      p.clip();
      const cols = SERIES();
      const sizes = [8, 9, 10, 11, 12];
      sizes.forEach((lg, i) => {
        const cap = 1 / (1 + Math.exp(-(lg - 9.6) * 1.5)) * .92;
        const rate = .18 + (lg - 7) * .04;
        const pts = [];
        for (let k = 0; k <= 32; k += .5) {
          let acc = cap * (1 - Math.exp(-rate * k)) + .05;
          if (s.cot) acc = Math.min(.97, acc * (lg > 10 ? 1.35 : 0.95));
          pts.push([k, Math.min(acc, .98)]);
        }
        p.line(pts, { color: alpha(cols[i], Math.abs(lg - s.size) < .3 ? 1 : .35), width: Math.abs(lg - s.size) < .3 ? 3 : 1.6 });
      });
      p.clip(false);
      p.legend(sizes.map((lg, i) => ({ label: `10^${lg} params`, color: cols[i] })), { pos: 'br' });
      const cap = 1 / (1 + Math.exp(-(s.size - 9.6) * 1.5)) * .92;
      P.readout({
        'model': `10^${fmt(s.size, 1)} params`,
        'few-shot ceiling': fmt(cap * 100, 0) + '%',
        'CoT effect': s.cot ? (s.size > 10 ? 'large gain — emergent' : 'no gain or worse at this scale') : 'off',
        'note': 'no weights change — the "learning" happens entirely in the forward pass',
      });
    },
    caption: 'Curves here are illustrative of the published shape, not measured. The important facts they encode: **(1)** more examples in the prompt help, with diminishing returns; **(2)** the ceiling rises sharply with scale; **(3)** chain-of-thought *hurts* small models and helps large ones — a genuine discontinuity. What is remarkable is that no gradient step occurs. The forward pass alone implements something functionally like learning from the examples, and induction heads are a big part of the mechanism.',
  });
};

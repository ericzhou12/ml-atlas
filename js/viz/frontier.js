/* ============================================================
   viz/frontier.js — vision, VLMs, systems, interpretability,
   and current-frontier figures.
   ============================================================ */

import { panel } from '../ui.js';
import { cssVar, alpha, mix, seqMap, divergeMap, LA, rng, fmt, SERIES } from '../plot.js';

const V = {};
export default V;

/* ============================================================
   Vision
   ============================================================ */

V['vit-patches'] = (host) => {
  panel(host, {
    title: 'Vision Transformer: an image is a sequence of patches',
    height: 300,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'patch', label: 'patch size', min: 4, max: 32, step: 4, value: 16 },
      { type: 'slider', key: 'img', label: 'image size', min: 64, max: 512, step: 32, value: 224 },
      { type: 'check', key: 'split', label: 'show the split', value: true },
    ],
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const N = 16;    // display grid resolution
      const img = [];
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let j = 0; j < N; j++) {
          const cx = j - 7.5, cy = i - 7.5;
          let v = .18 + .5 * Math.exp(-(cx * cx + cy * cy) / 30);
          if (Math.abs(cy + cx * .4) < 1.4) v += .3;
          if (i > 11) v = .1 + .04 * ((i + j) % 3);
          row.push(Math.min(1, v));
        }
        img.push(row);
      }
      const size = Math.min(p.w / 2 - 40, p.h - 70);
      const cell = size / N;
      const x0 = 24, y0 = 34;
      const nPatchDisp = Math.max(1, Math.round(N / (s.patch / (s.img / N) / (s.img / N) * 1)));
      const patchesPerSide = Math.max(1, Math.round(s.img / s.patch));
      const dispPatch = Math.max(1, Math.round(N / Math.min(patchesPerSide, N)));

      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        g.fillStyle = mix(cssVar('--bg-inset'), cssVar('--text'), img[i][j] * .9);
        g.fillRect(x0 + j * cell, y0 + i * cell, cell + .5, cell + .5);
      }
      if (s.split) {
        g.strokeStyle = cssVar('--s2'); g.lineWidth = 1.4;
        for (let k = 0; k <= N; k += dispPatch) {
          g.beginPath(); g.moveTo(x0 + k * cell, y0); g.lineTo(x0 + k * cell, y0 + size); g.stroke();
          g.beginPath(); g.moveTo(x0, y0 + k * cell); g.lineTo(x0 + size, y0 + k * cell); g.stroke();
        }
      }
      // sequence view
      const sx = x0 + size + 34;
      const nShow = Math.min(patchesPerSide * patchesPerSide, 24);
      const tw = 22;
      g.font = '9px ui-monospace, monospace';
      g.textAlign = 'center'; g.textBaseline = 'middle';
      for (let k = 0; k < nShow; k++) {
        const col = k % Math.max(1, Math.floor((p.w - sx - 20) / (tw + 4)));
        const row = Math.floor(k / Math.max(1, Math.floor((p.w - sx - 20) / (tw + 4))));
        const X = sx + col * (tw + 4), Y = y0 + row * (tw + 4);
        if (Y > p.h - 40) break;
        g.fillStyle = alpha(cssVar('--s1'), .45);
        g.fillRect(X, Y, tw, tw);
        g.fillStyle = cssVar('--text-dim');
        g.fillText(k, X + tw / 2, Y + tw / 2);
      }
      g.fillStyle = cssVar('--text-faint');
      g.font = '10px -apple-system, sans-serif';
      g.textAlign = 'left';
      g.fillText('image', x0, 20);
      g.fillText(`→ ${patchesPerSide * patchesPerSide} patch tokens (first ${nShow} shown)`, sx, 20);

      const nTok = patchesPerSide * patchesPerSide;
      P.readout({
        'patches': `${patchesPerSide} × ${patchesPerSide} = ${nTok} tokens`,
        'per-patch dimension': `${s.patch}×${s.patch}×3 = ${s.patch * s.patch * 3}`,
        'attention cost ∝ n²': (nTok * nTok).toLocaleString(),
        'halving patch size': '→ 4× the tokens, 16× the attention cost',
      });
    },
    caption: 'A ViT does something almost insultingly simple: cut the image into fixed squares, flatten each one, project it to $d_{\\text{model}}$, add a position embedding, and run a standard transformer. **All the visual inductive bias of a CNN is thrown away** — which is why ViTs lose to CNNs on small datasets and win on large ones, where they learn better biases than we could design. Note the cost: patch 16 on 224px gives 196 tokens; patch 8 gives 784, and attention costs 16× more.',
  });
};

V['clip-contrastive'] = (host) => {
  panel(host, {
    title: 'CLIP: line up images and captions in one space',
    height: 320,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'train', label: 'training progress', min: 0, max: 1, step: .01, value: 0 },
      { type: 'slider', key: 'temp', label: 'temperature τ', min: .02, max: 1, step: .01, value: .1 },
    ],
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const N = 6;
      const labels = ['a dog', 'a beach', 'a car', 'a cake', 'a violin', 'a mountain'];
      const r = rng(11);
      // similarity matrix: converges to identity as training progresses
      const S = [];
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let j = 0; j < N; j++) {
          const noise = r.normal(0, .35);
          const diag = i === j ? 1 : 0;
          row.push((1 - s.train) * noise + s.train * (diag * 1.0 - .18));
        }
        S.push(row);
      }
      const probs = S.map((row) => LA.softmax(row.map((v) => v / s.temp)));
      const size = Math.min(p.w - 200, p.h - 70);
      const cw = size / N;
      const x0 = 118, y0 = 34;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        g.fillStyle = mix(cssVar('--bg-inset'), cssVar('--s1'), Math.min(1, probs[i][j] * 1.5));
        g.fillRect(x0 + j * cw, y0 + i * cw, cw - 1, cw - 1);
        if (i === j) { g.strokeStyle = cssVar('--s3'); g.lineWidth = 1.8; g.strokeRect(x0 + j * cw, y0 + i * cw, cw - 1, cw - 1); }
        if (cw > 30) {
          g.fillStyle = probs[i][j] > .5 ? cssVar('--bg') : alpha(cssVar('--text-faint'), .8);
          g.font = '9px ui-monospace, monospace';
          g.textAlign = 'center'; g.textBaseline = 'middle';
          g.fillText(probs[i][j].toFixed(2).slice(1), x0 + j * cw + cw / 2, y0 + i * cw + cw / 2);
        }
      }
      g.font = '10.5px -apple-system, sans-serif';
      g.fillStyle = cssVar('--text-dim');
      for (let i = 0; i < N; i++) {
        g.textAlign = 'right'; g.textBaseline = 'middle';
        g.fillText('🖼 image ' + (i + 1), x0 - 8, y0 + i * cw + cw / 2);
        g.save();
        g.translate(x0 + i * cw + cw / 2, y0 + size + 8);
        g.rotate(-Math.PI / 4);
        g.textAlign = 'right';
        g.fillText(labels[i], 0, 0);
        g.restore();
      }
      const diag = probs.reduce((a, row, i) => a + row[i], 0) / N;
      P.readout({
        'mean P(correct caption)': fmt(diag, 3),
        'objective': 'symmetric cross-entropy over the N×N similarity matrix',
        'batch size matters': 'each row is an N-way classification — bigger N = harder = better features',
        'zero-shot classification': 'swap the captions for "a photo of a {class}" and take the argmax',
      });
    },
    caption: 'Drag training to 1. The objective is exactly a classification problem read twice: for each image, pick its caption out of the batch; for each caption, pick its image. Nothing labels what a dog *is* — the supervision is only "these two go together." **This is why CLIP does zero-shot classification**: at test time you write the class names as captions and ask which one matches. The temperature is learned, and the batch size is a genuine capability lever — CLIP used 32,768.',
  });
};

V['vlm-architecture'] = (host) => {
  panel(host, {
    title: 'How a vision-language model is wired',
    height: 300,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      {
        type: 'select', key: 'kind', label: 'architecture', value: 'proj', options: [
          { value: 'proj', label: 'projection / adapter (LLaVA-style)' },
          { value: 'resampler', label: 'perceiver resampler (Flamingo)' },
          { value: 'xattn', label: 'interleaved cross-attention' },
          { value: 'native', label: 'natively multimodal (early fusion)' },
        ],
      },
    ],
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const box = (x, y, w, h, label, col, sub) => {
        g.fillStyle = alpha(col, .18);
        g.strokeStyle = col; g.lineWidth = 1.6;
        g.fillRect(x, y, w, h); g.strokeRect(x, y, w, h);
        g.fillStyle = cssVar('--text');
        g.font = '600 11px -apple-system, sans-serif';
        g.textAlign = 'center'; g.textBaseline = 'middle';
        g.fillText(label, x + w / 2, y + h / 2 - (sub ? 7 : 0));
        if (sub) {
          g.fillStyle = cssVar('--text-faint');
          g.font = '9.5px -apple-system, sans-serif';
          g.fillText(sub, x + w / 2, y + h / 2 + 8);
        }
      };
      const arrow = (x1, y1, x2, y2, col) => {
        g.strokeStyle = col || cssVar('--text-faint'); g.lineWidth = 1.5;
        g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
        const a = Math.atan2(y2 - y1, x2 - x1);
        g.fillStyle = col || cssVar('--text-faint');
        g.beginPath();
        g.moveTo(x2, y2);
        g.lineTo(x2 - 6 * Math.cos(a - .4), y2 - 6 * Math.sin(a - .4));
        g.lineTo(x2 - 6 * Math.cos(a + .4), y2 - 6 * Math.sin(a + .4));
        g.fill();
      };
      const W = p.w, H = p.h;
      const c1 = cssVar('--s1'), c2 = cssVar('--s2'), c3 = cssVar('--s3'), c4 = cssVar('--s4');

      if (s.kind === 'native') {
        box(20, 40, 90, 40, 'image', c2, 'patches');
        box(20, 100, 90, 40, 'text', c1, 'tokens');
        box(140, 60, 100, 60, 'shared tokenizer', c4, 'one vocabulary');
        box(280, 60, W - 320, 60, 'single transformer trained on both from scratch', c3);
        arrow(110, 60, 138, 78); arrow(110, 120, 138, 102);
        arrow(240, 90, 278, 90);
      } else {
        box(20, 30, 100, 44, 'image', c2, '');
        box(20, 130, 100, 44, 'text prompt', c1, '');
        box(148, 30, 110, 44, 'vision encoder', c2, 'ViT / SigLIP');
        const mid = { proj: ['MLP projector', '2-layer, trained'], resampler: ['perceiver resampler', '→ 64 latent queries'], xattn: ['cross-attn adapters', 'inserted in every block'] }[s.kind];
        box(286, 30, 120, 44, mid[0], c4, mid[1]);
        box(286, 110, W - 326, 80, 'language model (usually frozen at first)', c3, s.kind === 'xattn' ? 'image enters via cross-attention inside each layer' : 'image tokens are prepended to the text tokens');
        arrow(120, 52, 146, 52);
        arrow(258, 52, 284, 52);
        arrow(346, 74, 346, 108, c4);
        arrow(120, 152, 284, 152);
        g.fillStyle = cssVar('--text-faint');
        g.font = '9.5px -apple-system, sans-serif';
        g.textAlign = 'left';
        g.fillText('frozen ❄', 150, 90);
      }
      const notes = {
        proj: ['Cheapest by far: freeze a CLIP/SigLIP encoder and an LLM, train only a small MLP that maps image patches into the LLM\'s token space.', 'Image patches literally become tokens in the prompt. Two-stage training: align the projector, then instruction-tune. This is LLaVA, and it is why VLM research became accessible.'],
        resampler: ['A fixed set of learned queries cross-attends to the (many) image features and compresses them to a small constant number of tokens.', 'Decouples image resolution from sequence length — essential for video, where naive patching explodes the token count.'],
        xattn: ['The image never enters the token stream. Instead, gated cross-attention layers are inserted between the LLM\'s existing blocks.', 'Preserves the LLM\'s text ability exactly (the gates start at zero) at the cost of new parameters in every layer.'],
        native: ['No separate encoder and no bridge. Images are tokenized alongside text and the whole model is trained on mixed data from the start.', 'Best quality ceiling and true any-to-any generation; also the most expensive, and you cannot reuse an existing LLM.'],
      }[s.kind];
      g.fillStyle = cssVar('--text-dim');
      g.font = '11.5px -apple-system, sans-serif';
      g.textAlign = 'left';
      wrapText(g, notes[0], 20, H - 52, W - 40, 15);
      g.fillStyle = cssVar('--text-faint');
      wrapText(g, notes[1], 20, H - 22, W - 40, 15);
      P.readout({ 'trainable in stage 1': s.kind === 'proj' ? 'the projector only (~10M params)' : s.kind === 'native' ? 'everything' : 'adapter + resampler' });
    },
    caption: 'Every design here is answering the same question: **how do pixels get into a language model that only understands token embeddings?** The projector approach won on cost — you can build a competent VLM by training a two-layer MLP on top of two frozen models. Frontier systems have since moved toward native multimodality, because a bridge bolted onto a frozen encoder inherits whatever that encoder discarded.',
  });
};

function wrapText(g, text, x, y, maxW, lh) {
  const words = text.split(' ');
  let line = '', yy = y;
  for (const w of words) {
    const t = line + w + ' ';
    if (g.measureText(t).width > maxW && line) { g.fillText(line, x, yy); line = w + ' '; yy += lh; }
    else line = t;
  }
  g.fillText(line, x, yy);
}

/* ============================================================
   Systems
   ============================================================ */

V['quantization'] = (host) => {
  panel(host, {
    title: 'Quantization: fewer bits per weight',
    height: 300,
    plot: { xlim: [-4, 4], ylim: [0, .55], xlabel: 'weight value', ylabel: 'density' },
    controls: [
      { type: 'slider', key: 'bits', label: 'bits', min: 2, max: 8, step: 1, value: 4 },
      { type: 'check', key: 'outlier', label: 'add an outlier channel', value: false },
      { type: 'select', key: 'scheme', label: 'scheme', value: 'group', options: [{ value: 'tensor', label: 'per-tensor scale' }, { value: 'group', label: 'per-group scale (g=64)' }] },
    ],
    draw(p, s, P) {
      const r = rng(6);
      const n = 2000;
      const w = Array.from({ length: n }, (_, i) => (s.outlier && i % 97 === 0 ? r.normal(0, 1) * 6 : r.normal(0, 1)));
      const bits = Math.round(s.bits);
      const levels = 2 ** bits;
      const quantize = (arr) => {
        const mx = Math.max(...arr.map(Math.abs));
        const scale = mx / (levels / 2 - 1);
        return arr.map((v) => Math.round(v / scale) * scale);
      };
      let q;
      if (s.scheme === 'tensor') q = quantize(w);
      else {
        q = [];
        for (let i = 0; i < n; i += 64) q.push(...quantize(w.slice(i, i + 64)));
      }
      p.clear().axes();
      p.clip();
      // histogram of original
      const bins = 70, lo = -4, hi = 4;
      const h1 = new Array(bins).fill(0);
      for (const v of w) { const k = Math.floor((v - lo) / (hi - lo) * bins); if (k >= 0 && k < bins) h1[k]++; }
      const mx1 = Math.max(...h1);
      const g = p.ctx;
      for (let i = 0; i < bins; i++) {
        const L = p.px(lo + i * (hi - lo) / bins), R = p.px(lo + (i + 1) * (hi - lo) / bins);
        g.fillStyle = alpha(cssVar('--s1'), .3);
        const ht = h1[i] / mx1 * .5;
        g.fillRect(L, p.py(ht), Math.max(R - L - .5, 1), p.py(0) - p.py(ht));
      }
      // quantization levels
      const uniq = [...new Set(q.map((v) => Math.round(v * 1e4) / 1e4))].sort((a, b) => a - b);
      for (const v of uniq.slice(0, 400)) {
        if (v < lo || v > hi) continue;
        p.line([[v, 0], [v, .52]], { color: alpha(cssVar('--s2'), .55), width: 1 });
      }
      p.clip(false);
      p.legend([{ label: 'weight distribution', color: alpha(cssVar('--s1'), .5) }, { label: 'representable levels', color: cssVar('--s2') }], { pos: 'tr' });
      const err = Math.sqrt(w.reduce((a, v, i) => a + (v - q[i]) ** 2, 0) / n);
      P.readout({
        'bits': bits,
        'levels': levels,
        'memory vs fp16': fmt(bits / 16 * 100, 1) + '%',
        'RMS quantization error': fmt(err, 4),
        'verdict': err > .3 ? 'severe distortion ✗' : err > .1 ? 'noticeable' : 'acceptable ✓',
      });
    },
    caption: 'Weights are roughly Gaussian, so a uniform grid wastes most of its levels on the empty tails. Now switch on the **outlier channel**: with a single per-tensor scale, one huge value stretches the grid and everything else collapses onto a few levels — this is precisely why naive INT8 broke on large transformers. Per-group scales (the "g=64" you see in GPTQ/AWQ checkpoints) contain the damage locally, which is what makes 4-bit inference viable at all.',
  });
};

V['gpu-memory'] = (host) => {
  panel(host, {
    title: 'Where the memory actually goes',
    height: 290,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'params', label: 'parameters (B)', min: .5, max: 200, step: .5, value: 7 },
      { type: 'select', key: 'mode', label: 'scenario', value: 'train', options: [{ value: 'train', label: 'full fine-tuning (Adam)' }, { value: 'lora', label: 'LoRA fine-tuning' }, { value: 'infer16', label: 'inference (bf16)' }, { value: 'infer4', label: 'inference (4-bit)' }] },
      { type: 'slider', key: 'batch', label: 'batch × sequence (k tokens)', min: 1, max: 128, step: 1, value: 8 },
    ],
    draw(p, s, P) {
      const N = s.params * 1e9;
      const items = [];
      if (s.mode === 'train') {
        items.push(['weights (bf16)', N * 2, cssVar('--s1')]);
        items.push(['gradients (bf16)', N * 2, cssVar('--s2')]);
        items.push(['Adam m + v (fp32)', N * 8, cssVar('--s4')]);
        items.push(['fp32 master weights', N * 4, cssVar('--s6')]);
        items.push(['activations', s.batch * 1000 * 4096 * 2 * 30, cssVar('--s3')]);
      } else if (s.mode === 'lora') {
        items.push(['frozen weights (4-bit)', N * 0.55, cssVar('--s1')]);
        items.push(['LoRA params + grads + Adam', N * 0.01 * 14, cssVar('--s4')]);
        items.push(['activations', s.batch * 1000 * 4096 * 2 * 30, cssVar('--s3')]);
      } else if (s.mode === 'infer16') {
        items.push(['weights (bf16)', N * 2, cssVar('--s1')]);
        items.push(['KV cache', s.batch * 1000 * 2 * 32 * 4096 * 2, cssVar('--s5')]);
      } else {
        items.push(['weights (4-bit + scales)', N * 0.55, cssVar('--s1')]);
        items.push(['KV cache', s.batch * 1000 * 2 * 32 * 4096 * 2, cssVar('--s5')]);
      }
      const total = items.reduce((a, i) => a + i[1], 0);
      p.clear();
      const g = p.ctx;
      const W = p.w - 40, x0 = 20, barY = 44, barH = 40;
      let x = x0;
      for (const [name, bytes, col] of items) {
        const w = W * bytes / total;
        g.fillStyle = alpha(col, .75);
        g.fillRect(x, barY, w - 1, barH);
        x += w;
      }
      g.font = '11px -apple-system, sans-serif';
      g.textBaseline = 'middle';
      let Y = barY + barH + 24;
      for (const [name, bytes, col] of items) {
        g.fillStyle = col; g.fillRect(x0, Y - 5, 11, 11);
        g.fillStyle = cssVar('--text-dim'); g.textAlign = 'left';
        g.fillText(name, x0 + 18, Y);
        g.fillStyle = cssVar('--text'); g.textAlign = 'right';
        g.font = '11px ui-monospace, monospace';
        g.fillText(gb(bytes), x0 + W, Y);
        g.font = '11px -apple-system, sans-serif';
        Y += 19;
      }
      // GPU capacity markers
      const gpus = [['A100 40GB', 40], ['A100/H100 80GB', 80], ['8×H100', 640]];
      g.font = '10px -apple-system, sans-serif';
      g.textAlign = 'left';
      let gy = barY - 10;
      for (const [n2, cap] of gpus) {
        const fits = total / 1e9 <= cap;
        g.fillStyle = fits ? cssVar('--ok') : cssVar('--danger');
        g.fillText(`${fits ? '✓' : '✗'} ${n2}`, x0 + (gpus.indexOf(gpus.find((z) => z[0] === n2))) * 120, gy);
      }
      P.readout({
        'total': gb(total),
        'per parameter': fmt(total / N, 2) + ' bytes',
        'rule of thumb': s.mode === 'train' ? '≈16 bytes/param before activations' : s.mode === 'infer16' ? '≈2 bytes/param + KV cache' : '≈0.6 bytes/param + KV cache',
      });
    },
    caption: 'Full fine-tuning needs roughly **16 bytes per parameter** before a single activation — 2 for bf16 weights, 2 for gradients, 8 for Adam\'s two moments in fp32, 4 for master weights. That is why a 7B model does not fine-tune on a 40GB card, and why QLoRA (4-bit frozen base + tiny trainable adapters) collapses the requirement by more than 10×. At inference the picture inverts: weights are cheap and the **KV cache** is what eats your batch size.',
  });

  function gb(b) { return b >= 1e9 ? (b / 1e9).toFixed(1) + ' GB' : (b / 1e6).toFixed(0) + ' MB'; }
};

V['parallelism'] = (host) => {
  panel(host, {
    title: 'Four ways to split a model across GPUs',
    height: 290,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      {
        type: 'select', key: 'kind', label: 'strategy', value: 'data', options: [
          { value: 'data', label: 'data parallel' },
          { value: 'fsdp', label: 'FSDP / ZeRO-3 (sharded)' },
          { value: 'tensor', label: 'tensor parallel' },
          { value: 'pipeline', label: 'pipeline parallel' },
        ],
      },
      { type: 'slider', key: 'gpus', label: 'GPUs', min: 2, max: 8, step: 1, value: 4 },
    ],
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const G = Math.round(s.gpus);
      const W = p.w - 40, x0 = 20;
      const gw = W / G;
      const y0 = 46, gh = 90;
      const cols = SERIES();
      for (let i = 0; i < G; i++) {
        const X = x0 + i * gw;
        g.strokeStyle = cssVar('--border'); g.lineWidth = 1.4;
        g.fillStyle = cssVar('--bg-inset');
        g.fillRect(X + 4, y0, gw - 10, gh);
        g.strokeRect(X + 4, y0, gw - 10, gh);
        g.fillStyle = cssVar('--text-faint');
        g.font = '10px -apple-system, sans-serif';
        g.textAlign = 'center'; g.textBaseline = 'middle';
        g.fillText('GPU ' + i, X + gw / 2 - 1, y0 - 10);

        const L = 4;   // model layers
        for (let l = 0; l < L; l++) {
          const ly = y0 + 8 + l * ((gh - 16) / L);
          const lh = (gh - 16) / L - 3;
          let show = true, frac = 1, col = cols[l % cols.length];
          if (s.kind === 'pipeline') { show = Math.floor(l * G / L) === i; }
          else if (s.kind === 'tensor') { frac = 1 / G; }
          else if (s.kind === 'fsdp') { frac = 1 / G; }
          if (!show) {
            g.fillStyle = alpha(cssVar('--text-faint'), .08);
            g.fillRect(X + 10, ly, gw - 22, lh);
            continue;
          }
          g.fillStyle = alpha(col, .6);
          if (s.kind === 'tensor' || s.kind === 'fsdp') {
            g.fillRect(X + 10, ly, (gw - 22) * .96, lh);
            g.fillStyle = alpha(cssVar('--bg'), .55);
            for (let k = 0; k < G; k++) if (k !== i) g.fillRect(X + 10 + (gw - 22) * .96 * k / G, ly, (gw - 22) * .96 / G, lh);
          } else {
            g.fillRect(X + 10, ly, gw - 22, lh);
          }
        }
        if (s.kind === 'data' || s.kind === 'fsdp') {
          g.fillStyle = cssVar('--s5');
          g.font = '9px -apple-system, sans-serif';
          g.fillText(`batch ${i + 1}`, X + gw / 2 - 1, y0 + gh + 12);
        }
      }
      const info = {
        data: ['Every GPU holds a FULL copy of the model and processes a different slice of the batch.', 'Communication: all-reduce the gradients each step (2× model size). Simple, but memory scales with nothing — the model must fit on one GPU.'],
        fsdp: ['Parameters, gradients and optimizer state are SHARDED across GPUs; each layer is gathered just before use and freed after.', 'Communication: all-gather weights + reduce-scatter gradients. Memory per GPU drops ~G×, which is what lets a 70B model train on 8 cards.'],
        tensor: ['Individual matrices are split — e.g. the FFN\'s columns across GPUs. Every GPU works on every layer.', 'Communication: an all-reduce INSIDE each layer, twice per block. Needs very fast interconnect (NVLink) — usually kept within a single node.'],
        pipeline: ['Different GPUs own different LAYERS. Activations are passed down the chain.', 'Communication is tiny (just activations at the boundaries), but naive scheduling leaves GPUs idle — the "bubble". Micro-batching shrinks it.'],
      }[s.kind];
      g.fillStyle = cssVar('--text-dim');
      g.font = '11.5px -apple-system, sans-serif';
      g.textAlign = 'left';
      wrapText(g, info[0], 20, y0 + gh + 34, p.w - 40, 15);
      g.fillStyle = cssVar('--text-faint');
      wrapText(g, info[1], 20, y0 + gh + 68, p.w - 40, 15);
      P.readout({
        'memory per GPU': { data: '1× full model', fsdp: `1/${G} of model+optimizer`, tensor: `1/${G} of each layer`, pipeline: `1/${G} of the layers` }[s.kind],
        'communication': { data: 'gradient all-reduce, once per step', fsdp: 'all-gather + reduce-scatter, per layer', tensor: 'all-reduce, twice per block', pipeline: 'activations at stage boundaries' }[s.kind],
      });
    },
    caption: 'Real training runs combine all of these — "3D parallelism" typically means tensor-parallel within a node (fast NVLink), pipeline-parallel across nodes, and data-parallel over the whole thing, with FSDP sharding the optimizer state. The choice is always a trade between **memory per device** and **bytes on the wire**, and the interconnect topology decides which axis you can afford to stretch.',
  });
};

V['speculative-decoding'] = (host) => {
  panel(host, {
    title: 'Speculative decoding: guess ahead, verify in parallel',
    height: 280,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'k', label: 'draft length k', min: 1, max: 8, step: 1, value: 4 },
      { type: 'slider', key: 'acc', label: 'per-token acceptance rate', min: .1, max: .98, step: .01, value: .7 },
      { type: 'slider', key: 'cost', label: 'draft cost (fraction of target)', min: .01, max: .4, step: .01, value: .1 },
    ],
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const k = Math.round(s.k), a = s.acc;
      // expected accepted tokens per verification: (1 - a^(k+1)) / (1 - a)
      const exp = (1 - a ** (k + 1)) / (1 - a);
      const costPer = k * s.cost + 1;
      const speedup = exp / costPer;
      // draw a timeline
      const r = rng(4);
      const rows = 4;
      const rowH = 28, x0 = 90, cw = Math.min(30, (p.w - x0 - 30) / 12);
      g.font = '10px -apple-system, sans-serif';
      g.textBaseline = 'middle';
      let tok = 0;
      for (let row = 0; row < rows; row++) {
        const y = 34 + row * (rowH + 8);
        g.fillStyle = cssVar('--text-faint'); g.textAlign = 'right';
        g.fillText(`round ${row + 1}`, x0 - 8, y + rowH / 2);
        let accepted = 0;
        for (let i = 0; i < k; i++) { if (r() < a) accepted++; else break; }
        for (let i = 0; i < k; i++) {
          const X = x0 + i * (cw + 3);
          const ok = i < accepted;
          g.fillStyle = ok ? alpha(cssVar('--ok'), .7) : alpha(cssVar('--danger'), .4);
          g.fillRect(X, y, cw, rowH);
          g.fillStyle = cssVar('--bg');
          g.textAlign = 'center';
          g.font = '9px -apple-system, sans-serif';
          g.fillText(ok ? '✓' : '✗', X + cw / 2, y + rowH / 2);
        }
        // the free bonus token from the target model
        const X = x0 + Math.min(accepted, k) * (cw + 3);
        g.fillStyle = alpha(cssVar('--s1'), .8);
        g.fillRect(X, y, cw, rowH);
        g.fillStyle = cssVar('--bg');
        g.fillText('+1', X + cw / 2, y + rowH / 2);
        tok += accepted + 1;
      }
      g.fillStyle = cssVar('--text-dim');
      g.font = '10.5px -apple-system, sans-serif';
      g.textAlign = 'left';
      g.fillText('✓ draft token accepted   ✗ rejected (and everything after it)   +1 free token from the target model', x0, 18);
      P.readout({
        'expected tokens per round': fmt(exp, 3),
        'cost per round (target passes)': fmt(costPer, 3),
        'speedup': fmt(speedup, 2) + '×',
        'output distribution': 'provably IDENTICAL to the target model',
        'optimal k here': bestK(a, s.cost),
      });
    },
    caption: 'A small draft model proposes $k$ tokens; the big model checks all of them in **one forward pass** (it can, because verification is parallel — generation is not). A clever accept/reject rule makes the output distribution *exactly* the target model\'s, so this is free speed, not a quality trade. The catch: one rejection discards every token after it, so the gain depends on how well the draft model imitates the target. Push acceptance below 0.4 and the speedup evaporates.',
  });

  function bestK(a, c) {
    let bk = 1, bs = 0;
    for (let k = 1; k <= 16; k++) {
      const sp = ((1 - a ** (k + 1)) / (1 - a)) / (k * c + 1);
      if (sp > bs) { bs = sp; bk = k; }
    }
    return `${bk}  (${fmt(bs, 2)}×)`;
  }
};

/* ============================================================
   Interpretability
   ============================================================ */

V['superposition'] = (host) => {
  panel(host, {
    title: 'Superposition: more features than dimensions',
    height: 320,
    plot: { xlim: [-1.4, 1.4], ylim: [-1.15, 1.15], equal: true },
    controls: [
      { type: 'slider', key: 'nfeat', label: 'features to represent', min: 2, max: 12, step: 1, value: 5 },
      { type: 'slider', key: 'sparsity', label: 'feature sparsity (1−p active)', min: 0, max: .99, step: .01, value: .9 },
    ],
    draw(p, s, P) {
      const n = Math.round(s.nfeat);
      p.clear().axes({ nx: 5, ny: 4 });
      p.clip();
      // At high sparsity a 2-D model packs n features as near-uniformly spaced directions.
      // At low sparsity it can only afford 2 orthogonal ones.
      const dense = s.sparsity < .4;
      const kept = dense ? 2 : n;
      const cols = SERIES();
      p.circle(0, 0, p.sx, { stroke: alpha(cssVar('--text-faint'), .4), width: 1, dash: [3, 3] });
      for (let i = 0; i < n; i++) {
        const active = i < kept;
        const ang = active ? (i / kept) * Math.PI * (kept === 2 ? 1 : 2) / (kept === 2 ? 2 : 1) * (kept === 2 ? 1 : 1) : 0;
        const a2 = kept === 2 ? i * Math.PI / 2 : (i / kept) * 6.2832;
        const mag = active ? (dense ? 1 : .96) : .04;
        p.arrow(0, 0, Math.cos(a2) * mag, Math.sin(a2) * mag, {
          color: active ? cols[i % cols.length] : alpha(cssVar('--text-faint'), .4),
          width: active ? 2.6 : 1.2,
        });
        if (active) p.text(Math.cos(a2) * mag * 1.13, Math.sin(a2) * mag * 1.13, 'f' + i, { color: cols[i % cols.length], size: 11, align: 'center', weight: '600' });
      }
      p.clip(false);
      // interference
      let interf = 0, cnt = 0;
      for (let i = 0; i < kept; i++) for (let j = i + 1; j < kept; j++) {
        const ai = kept === 2 ? i * Math.PI / 2 : (i / kept) * 6.2832;
        const aj = kept === 2 ? j * Math.PI / 2 : (j / kept) * 6.2832;
        interf += Math.abs(Math.cos(ai - aj)); cnt++;
      }
      P.readout({
        'model dimensions': 2,
        'features stored': kept,
        'mean |cosine| between features': cnt ? fmt(interf / cnt, 3) : '0',
        'regime': dense ? 'dense features → only 2 fit, orthogonally' : `sparse → ${n} features packed with tolerable interference`,
      });
    },
    caption: 'A 2-dimensional layer can only hold 2 features *orthogonally*. But if features are **sparse** — rarely active at the same time — the model can pack many more as non-orthogonal directions and simply accept the occasional interference, because collisions are rare. Slide sparsity from 0.2 to 0.95 and watch the model switch strategies. This is why individual neurons are usually polysemantic, and it is the direct motivation for sparse autoencoders: recover the overcomplete feature basis the model is secretly using.',
  });
};

V['sparse-autoencoder'] = (host) => {
  panel(host, {
    title: 'Sparse autoencoders: unpacking a polysemantic neuron',
    height: 300,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'l1', label: 'L1 sparsity penalty', min: 0, max: 1, step: .01, value: .3 },
      { type: 'slider', key: 'expand', label: 'dictionary expansion factor', min: 1, max: 16, step: 1, value: 8 },
    ],
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const nIn = 8, nHid = Math.min(Math.round(s.expand) * 4, 48);
      const r = rng(13);
      // activation vector, and a sparse code whose sparsity depends on l1
      const active = Math.max(1, Math.round(nHid * (1 - s.l1) * .4));
      const code = new Array(nHid).fill(0);
      const idx = r.shuffle(Array.from({ length: nHid }, (_, i) => i)).slice(0, active);
      for (const i of idx) code[i] = .3 + r() * .7;
      const x0 = 24, y0 = 46;
      const inW = 26;
      g.font = '10px -apple-system, sans-serif';
      g.textAlign = 'center'; g.textBaseline = 'middle';
      // input activations
      for (let i = 0; i < nIn; i++) {
        const v = .2 + r() * .8;
        g.fillStyle = mix(cssVar('--bg-inset'), cssVar('--s1'), v);
        g.fillRect(x0, y0 + i * 20, inW, 18);
      }
      g.fillStyle = cssVar('--text-faint');
      g.fillText('MLP', x0 + inW / 2, y0 - 14);
      g.fillText('acts', x0 + inW / 2, y0 - 3);
      // sparse code
      const cx = x0 + inW + 70;
      const ch = Math.min(14, (p.h - y0 - 40) / nHid);
      for (let i = 0; i < nHid; i++) {
        g.fillStyle = code[i] > 0 ? mix(cssVar('--bg-inset'), cssVar('--s3'), code[i]) : alpha(cssVar('--text-faint'), .08);
        g.fillRect(cx, y0 + i * ch, 22, Math.max(ch - 1.5, 2));
      }
      g.fillStyle = cssVar('--text-faint');
      g.fillText('sparse features', cx + 11, y0 - 10);
      // reconstruction
      const rx = cx + 100;
      for (let i = 0; i < nIn; i++) {
        const v = .2 + r() * .8;
        g.fillStyle = mix(cssVar('--bg-inset'), cssVar('--s1'), v * (1 - s.l1 * .3));
        g.fillRect(rx, y0 + i * 20, inW, 18);
      }
      g.fillStyle = cssVar('--text-faint');
      g.fillText('recon', rx + inW / 2, y0 - 8);
      // arrows
      g.strokeStyle = cssVar('--text-faint'); g.lineWidth = 1.2;
      for (const [a, b] of [[x0 + inW + 6, cx - 6], [cx + 28, rx - 6]]) {
        g.beginPath(); g.moveTo(a, y0 + 70); g.lineTo(b, y0 + 70); g.stroke();
      }
      g.fillStyle = cssVar('--text-dim');
      g.font = '10px -apple-system, sans-serif';
      g.fillText('encode', (x0 + inW + cx) / 2, y0 + 58);
      g.fillText('decode', (cx + 28 + rx) / 2, y0 + 58);

      const reconErr = s.l1 * .55;
      P.readout({
        'dictionary size': `${nHid} features from ${nIn} dimensions (${Math.round(s.expand)}× overcomplete)`,
        'active features': `${active} / ${nHid}`,
        'L0 sparsity': fmt(active, 0),
        'reconstruction loss': fmt(reconErr, 3),
        'the tradeoff': 'more sparsity → more interpretable features, worse reconstruction',
      });
    },
    caption: 'The SAE is trained on a single objective — reconstruct the activations, with an L1 penalty pushing the code toward sparsity — and it is *wider* than the layer it reads, deliberately. The bet is that the model stores many sparse features in superposition, and that a wide, sparse basis will recover them one per unit. In practice the recovered features are often strikingly interpretable (a "Golden Gate Bridge" feature, a "code that will error" feature). The open problems: reconstruction is never perfect, features split unpredictably with dictionary size, and there is no ground truth to check against.',
  });
};

V['grokking'] = (host) => {
  panel(host, {
    title: 'Grokking: generalization long after memorization',
    height: 290,
    plot: { xlim: [1, 6], ylim: [0, 1.05], xlabel: 'log₁₀ optimization steps', ylabel: 'accuracy' },
    controls: [
      { type: 'slider', key: 'wd', label: 'weight decay', min: 0, max: 1, step: .02, value: .5 },
      { type: 'slider', key: 'data', label: 'training fraction', min: .2, max: .9, step: .02, value: .5 },
    ],
    draw(p, s, P) {
      p.clear().axes();
      p.clip();
      const memStep = 2.4 - s.data * .6;
      const genStep = s.wd < .05 ? 9 : 5.2 - s.wd * 1.6 - s.data * 1.2;
      const sig = (x, c, w) => 1 / (1 + Math.exp(-(x - c) / w));
      p.fn((x) => .02 + .97 * sig(x, memStep, .18), { color: cssVar('--s1'), width: 2.6 });
      p.fn((x) => .02 + .96 * sig(x, genStep, .22), { color: cssVar('--s2'), width: 2.6 });
      p.clip(false);
      p.legend([{ label: 'train accuracy', color: cssVar('--s1') }, { label: 'test accuracy', color: cssVar('--s2') }], { pos: 'br' });
      P.readout({
        'memorization at': `~10^${fmt(memStep, 1)} steps`,
        'generalization at': s.wd < .05 ? 'never (within budget)' : `~10^${fmt(genStep, 1)} steps`,
        'gap': s.wd < .05 ? '∞' : `${fmt(10 ** genStep / 10 ** memStep, 0)}× longer`,
        'mechanism': 'weight decay slowly replaces the memorizing circuit with a smaller generalizing one',
      });
    },
    caption: 'Curves are a stylized reproduction of the published phenomenon, not measured here. On small algorithmic tasks (modular arithmetic), a transformer hits **100% train accuracy while test accuracy sits at chance** — and then, tens of thousands of steps later, test accuracy suddenly jumps to 100%. Set weight decay to 0 and it never happens. The current explanation: memorizing and generalizing circuits both fit the data, but the generalizing one has smaller norm, so regularization eventually wins. It is the sharpest available evidence that "training loss stopped improving" ≠ "learning stopped."',
  });
};

V['ssm-vs-attention'] = (host) => {
  panel(host, {
    title: 'State-space models: recurrence with a long memory',
    height: 290,
    plot: { xlim: [0, 60], ylim: [-.05, 1.05], xlabel: 'distance from the current token', ylabel: 'influence' },
    controls: [
      { type: 'slider', key: 'decay', label: 'state decay a', min: .8, max: .999, step: .001, value: .96, fmt: (v) => v.toFixed(3) },
      { type: 'slider', key: 'nstate', label: 'state dimensions', min: 1, max: 8, step: 1, value: 4 },
      { type: 'check', key: 'attn', label: 'compare with attention', value: true },
    ],
    draw(p, s, P) {
      p.clear().axes();
      p.clip();
      const cols = SERIES();
      // several state channels with different decay rates (HiPPO-like)
      const N = Math.round(s.nstate);
      for (let k = 0; k < N; k++) {
        const a = s.decay ** (1 + k * 1.6);
        const pts = [];
        for (let d = 0; d <= 60; d++) pts.push([d, a ** d]);
        p.line(pts, { color: alpha(cols[k % cols.length], .85), width: 2 });
      }
      if (s.attn) {
        const pts = [];
        for (let d = 0; d <= 60; d++) pts.push([d, d < 45 ? .82 : 0]);
        p.line(pts, { color: cssVar('--text'), width: 2.4, dash: [5, 3] });
      }
      p.clip(false);
      p.legend([
        { label: 'SSM channels (geometric decay)', color: cssVar('--s1') },
        ...(s.attn ? [{ label: 'attention (flat within context, then nothing)', color: cssVar('--text'), dash: true }] : []),
      ], { pos: 'tr' });
      P.readout({
        'inference cost per token': 'O(1) state update — no growing cache',
        'attention cost per token': 'O(n) — must read the whole KV cache',
        'training': 'parallelizable via a scan / convolution, unlike an RNN',
        'weakness': 'fixed-size state → exact recall of arbitrary earlier tokens is impossible',
      });
    },
    caption: 'An SSM is a linear recurrence, $h_t = A h_{t-1} + B x_t$, which means it has an $O(1)$ state instead of a growing KV cache — enormously cheaper at long context. The trick that made them work is twofold: **structure $A$** so different channels decay at very different rates (giving multi-timescale memory), and **make the parameters input-dependent** (Mamba\'s selectivity) so the model can choose what to keep. The limitation is intrinsic: a fixed state cannot losslessly store an unbounded past, which is why hybrid architectures interleave a few attention layers for exact recall.',
  });
};

V['test-time-compute'] = (host) => {
  panel(host, {
    title: 'Spending compute at inference instead of training',
    height: 290,
    plot: { xlim: [0, 3], ylim: [0, 1], xlabel: 'log₁₀ samples / reasoning tokens', ylabel: 'solve rate' },
    controls: [
      { type: 'select', key: 'method', label: 'method', value: 'maj', options: [{ value: 'best1', label: 'single sample' }, { value: 'maj', label: 'majority vote (self-consistency)' }, { value: 'bon', label: 'best-of-n with a verifier' }, { value: 'cot', label: 'long chain-of-thought (RL-trained)' }] },
      { type: 'slider', key: 'p', label: 'per-attempt success probability', min: .05, max: .9, step: .01, value: .35 },
      { type: 'slider', key: 'verifier', label: 'verifier accuracy', min: .5, max: 1, step: .01, value: .9 },
    ],
    draw(p, s, P) {
      p.clear().axes();
      p.clip();
      const cols = SERIES();
      const curves = {
        best1: (lg) => s.p,
        maj: (lg) => {
          const n = 10 ** lg;
          // rough: majority over n samples where correct answer has prob p and errors scatter
          return Math.min(.98, s.p + (1 - s.p) * (1 - Math.exp(-0.55 * Math.log10(n + 1))) * (s.p > .25 ? .85 : .3));
        },
        bon: (lg) => {
          const n = 10 ** lg;
          const anyCorrect = 1 - (1 - s.p) ** n;
          return anyCorrect * (s.verifier ** Math.min(Math.log2(n + 1), 6));
        },
        cot: (lg) => Math.min(.97, s.p + (1 - s.p) * (1 - Math.exp(-1.05 * lg))),
      };
      Object.entries(curves).forEach(([k, f], i) => {
        p.fn(f, { color: alpha(cols[i], k === s.method ? 1 : .3), width: k === s.method ? 3 : 1.5, n: 200 });
      });
      // pass@n ceiling
      p.fn((lg) => 1 - (1 - s.p) ** (10 ** lg), { color: alpha(cssVar('--text-faint'), .7), width: 1.6, dash: [4, 3] });
      p.clip(false);
      p.legend([
        ...Object.keys(curves).map((k, i) => ({ label: { best1: 'single sample', maj: 'majority vote', bon: 'best-of-n + verifier', cot: 'long CoT (RL-trained)' }[k], color: cols[i] })),
        { label: 'pass@n ceiling (any sample correct)', color: cssVar('--text-faint'), dash: true },
      ], { pos: 'br' });
      P.readout({
        'at 100 samples': fmt(curves[s.method](2) * 100, 1) + '%',
        'ceiling (pass@100)': fmt((1 - (1 - s.p) ** 100) * 100, 1) + '%',
        'the gap': 'the model can often FIND the answer — the hard part is knowing which one is right',
        'note': s.method === 'bon' && s.verifier < .8 ? 'a weak verifier destroys best-of-n at large n ✗' : '',
      });
    },
    caption: 'Curves are illustrative of the shape reported across the literature, not measured. The key structural fact is the gap between **pass@n** (the dashed ceiling — did *any* sample get it right) and what you can actually extract. Majority voting closes some of it for free; a verifier closes more but degrades at large $n$ because you start selecting for whatever fools the verifier. Reasoning models trained with RL on verifiable rewards learn to spend those tokens productively in a *single* sample, which is why o-series and R1-style models changed the shape of this plot.',
  });
};

V['hallucination'] = (host) => {
  panel(host, {
    title: 'Calibration and the shape of hallucination',
    height: 290,
    plot: { xlim: [0, 1], ylim: [0, 1], xlabel: 'model-stated confidence', ylabel: 'actual accuracy' },
    controls: [
      { type: 'select', key: 'stage', label: 'model', value: 'base', options: [{ value: 'base', label: 'base (pretrained only)' }, { value: 'rlhf', label: 'after RLHF' }, { value: 'rlhf-cal', label: 'RLHF + calibration training' }] },
      { type: 'slider', key: 'know', label: 'fraction of questions the model knows', min: .1, max: .95, step: .01, value: .6 },
    ],
    draw(p, s, P) {
      p.clear().axes();
      p.clip();
      p.line([[0, 0], [1, 1]], { color: alpha(cssVar('--text-faint'), .7), width: 1.6, dash: [4, 4] });
      const f = {
        base: (c) => Math.max(0, Math.min(1, c * .96 + .02)),
        rlhf: (c) => Math.max(0, Math.min(1, c * .55 + .05)),
        'rlhf-cal': (c) => Math.max(0, Math.min(1, c * .88 + .04)),
      }[s.stage];
      p.fn(f, { color: cssVar('--s2'), width: 3 });
      // confidence histogram along the bottom
      const g = p.ctx;
      for (let i = 0; i < 20; i++) {
        const c = (i + .5) / 20;
        const mass = s.stage === 'base'
          ? Math.exp(-((c - s.know) ** 2) / .08)
          : Math.exp(-((c - .92) ** 2) / .01) + .25 * Math.exp(-((c - s.know) ** 2) / .1);
        const h = mass * .1;
        g.fillStyle = alpha(cssVar('--s1'), .5);
        g.fillRect(p.px(c - .022), p.py(h), p.px(.044) - p.px(0), p.py(0) - p.py(h));
      }
      p.clip(false);
      p.legend([{ label: 'perfect calibration', color: cssVar('--text-faint'), dash: true }, { label: 'this model', color: cssVar('--s2') }, { label: 'where its confidence sits', color: alpha(cssVar('--s1'), .5) }], { pos: 'tl' });
      let ece = 0;
      for (let i = 0; i < 20; i++) { const c = (i + .5) / 20; ece += Math.abs(f(c) - c) / 20; }
      P.readout({
        'expected calibration error': fmt(ece, 3),
        'diagnosis': s.stage === 'rlhf' ? 'overconfident — states 0.9 when right 0.5 of the time ✗' : 'reasonably calibrated ✓',
        'why RLHF hurts calibration': 'human raters prefer confident-sounding answers, so confidence gets rewarded independently of correctness',
      });
    },
    caption: 'Points below the diagonal mean **overconfidence**: the model says 90% and is right 50% of the time. Pretrained base models are often startlingly well calibrated — next-token probabilities are trained on exactly this. RLHF systematically damages it, because "sounds certain" is rewarded by raters whether or not it is true. Hallucination is not one bug: it is training on text that never says "I don\'t know", an objective that always demands *some* next token, and a preference signal that punishes hedging.',
  });
};

V['agent-loop'] = (host) => {
  panel(host, {
    title: 'The agent loop, and where it breaks',
    height: 290,
    plot: { xlim: [0, 30], ylim: [0, 1], xlabel: 'steps in the trajectory', ylabel: 'probability the whole task is still on track' },
    controls: [
      { type: 'slider', key: 'acc', label: 'per-step reliability', min: .8, max: .999, step: .001, value: .95, fmt: (v) => v.toFixed(3) },
      { type: 'check', key: 'recover', label: 'model can detect and recover from errors', value: false },
      { type: 'slider', key: 'rec', label: 'recovery probability', min: 0, max: .95, step: .01, value: .6 },
    ],
    draw(p, s, P) {
      p.clear().axes();
      p.clip();
      const cols = SERIES();
      for (const [i, a] of [.9, .95, .99, .999].entries()) {
        const pts = [];
        for (let n = 0; n <= 30; n++) {
          const eff = s.recover ? a + (1 - a) * s.rec : a;
          pts.push([n, eff ** n]);
        }
        p.line(pts, { color: alpha(cols[i], Math.abs(a - s.acc) < .005 ? 1 : .35), width: Math.abs(a - s.acc) < .005 ? 3 : 1.5 });
      }
      p.clip(false);
      p.legend([.9, .95, .99, .999].map((a, i) => ({ label: `p = ${a}`, color: cols[i] })), { pos: 'tr' });
      const eff = s.recover ? s.acc + (1 - s.acc) * s.rec : s.acc;
      P.readout({
        'per-step reliability': fmt(s.acc, 3),
        'effective with recovery': fmt(eff, 4),
        'success at 10 steps': fmt(eff ** 10 * 100, 1) + '%',
        'success at 30 steps': fmt(eff ** 30 * 100, 1) + '%',
        'steps until 50% failure': fmt(Math.log(.5) / Math.log(eff), 0),
      });
    },
    caption: 'An agent is a loop: think → call a tool → read the result → repeat. The arithmetic is unforgiving — at 95% per-step reliability, a 30-step task succeeds **21%** of the time. This is why agent work is dominated not by reasoning quality but by **error recovery**: checkpointing, verification steps, letting the model notice and undo a bad action. Switch on recovery and watch the curve lift. It is also why "reliability per step" is the metric that actually matters for long-horizon tasks.',
  });
};

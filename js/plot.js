/* ============================================================
   plot.js — tiny canvas plotting + interaction toolkit.

   Everything visual in the app is built from this: axes, curves,
   scatter, vector fields, heatmaps, contours, bars, arrows.
   Theme-aware (reads CSS custom properties), HiDPI-correct,
   and supports drag interaction in data coordinates.
   ============================================================ */

/* ---------- theme colors ---------- */

let _cssCache = {};
export function cssVar(name) {
  if (_cssCache[name]) return _cssCache[name];
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  _cssCache[name] = v || '#888';
  return _cssCache[name];
}
export function clearColorCache() { _cssCache = {}; }

export const SERIES = () => [
  cssVar('--s1'), cssVar('--s2'), cssVar('--s3'), cssVar('--s4'),
  cssVar('--s5'), cssVar('--s6'), cssVar('--s7'), cssVar('--s8'),
];

/** Mix a hex color toward another by t in [0,1]. */
export function mix(a, b, t) {
  const pa = hex2rgb(a), pb = hex2rgb(b);
  return `rgb(${Math.round(pa[0] + (pb[0] - pa[0]) * t)},${Math.round(pa[1] + (pb[1] - pa[1]) * t)},${Math.round(pa[2] + (pb[2] - pa[2]) * t)})`;
}
export function alpha(c, a) {
  const p = hex2rgb(c);
  return `rgba(${p[0]},${p[1]},${p[2]},${a})`;
}
function hex2rgb(h) {
  h = h.trim();
  if (h.startsWith('rgb')) return h.match(/[\d.]+/g).slice(0, 3).map(Number);
  h = h.replace('#', '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Diverging blue↔orange colormap, t in [0,1]. */
export function divergeMap(t) {
  t = Math.max(0, Math.min(1, t));
  const lo = cssVar('--s1'), mid = cssVar('--bg-elev'), hi = cssVar('--s2');
  return t < 0.5 ? mix(lo, mid, t * 2) : mix(mid, hi, (t - 0.5) * 2);
}
/** Sequential viridis-ish colormap, t in [0,1]. */
export function seqMap(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  const x = t * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2);
  const f = x - i;
  const c = stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/* ---------- Plot ---------- */

export class Plot {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} o  {xlim,ylim,pad,xlabel,ylabel,grid,equal,height}
   */
  constructor(canvas, o = {}) {
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.xlim = o.xlim || [-1, 1];
    this.ylim = o.ylim || [-1, 1];
    this.pad = Object.assign({ l: 42, r: 12, t: 12, b: 32 }, o.pad);
    this.xlabel = o.xlabel || '';
    this.ylabel = o.ylabel || '';
    this.showGrid = o.grid !== false;
    this.equal = !!o.equal;
    this.aspect = o.aspect || 0.62;   // height / width when auto-sizing
    this.height = o.height || null;
    this.resize();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const cssW = this.c.clientWidth || this.c.parentElement.clientWidth || 600;
    const cssH = this.height || Math.round(cssW * this.aspect);
    this.c.style.height = cssH + 'px';
    this.c.width = Math.round(cssW * dpr);
    this.c.height = Math.round(cssH * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = cssW; this.h = cssH;
    if (this.equal) this._applyEqual();
    return this;
  }

  _applyEqual() {
    const pw = this.w - this.pad.l - this.pad.r;
    const ph = this.h - this.pad.t - this.pad.b;
    const dx = this.xlim[1] - this.xlim[0], dy = this.ylim[1] - this.ylim[0];
    const sx = pw / dx, sy = ph / dy;
    const s = Math.min(sx, sy);
    const cx = (this.xlim[0] + this.xlim[1]) / 2, cy = (this.ylim[0] + this.ylim[1]) / 2;
    this.xlim = [cx - pw / (2 * s), cx + pw / (2 * s)];
    this.ylim = [cy - ph / (2 * s), cy + ph / (2 * s)];
  }

  setLim(xlim, ylim) {
    this.xlim = xlim; this.ylim = ylim;
    if (this.equal) this._applyEqual();
    return this;
  }

  /* coordinate transforms */
  px(x) { return this.pad.l + (x - this.xlim[0]) / (this.xlim[1] - this.xlim[0]) * (this.w - this.pad.l - this.pad.r); }
  py(y) { return this.h - this.pad.b - (y - this.ylim[0]) / (this.ylim[1] - this.ylim[0]) * (this.h - this.pad.t - this.pad.b); }
  ix(px) { return this.xlim[0] + (px - this.pad.l) / (this.w - this.pad.l - this.pad.r) * (this.xlim[1] - this.xlim[0]); }
  iy(py) { return this.ylim[0] + (this.h - this.pad.b - py) / (this.h - this.pad.t - this.pad.b) * (this.ylim[1] - this.ylim[0]); }
  /** pixels per data unit (x) */
  get sx() { return (this.w - this.pad.l - this.pad.r) / (this.xlim[1] - this.xlim[0]); }
  get sy() { return (this.h - this.pad.t - this.pad.b) / (this.ylim[1] - this.ylim[0]); }

  clear(bg) {
    const g = this.ctx;
    g.clearRect(0, 0, this.w, this.h);
    g.fillStyle = bg || cssVar('--bg-inset');
    g.fillRect(0, 0, this.w, this.h);
    return this;
  }

  clip(on = true) {
    const g = this.ctx;
    if (on) {
      g.save();
      g.beginPath();
      g.rect(this.pad.l, this.pad.t, this.w - this.pad.l - this.pad.r, this.h - this.pad.t - this.pad.b);
      g.clip();
    } else g.restore();
    return this;
  }

  /* ---------- axes ---------- */

  axes(o = {}) {
    const g = this.ctx;
    const ticks = o.ticks !== false;
    const nx = o.nx || 6, ny = o.ny || 5;
    const xs = niceTicks(this.xlim[0], this.xlim[1], nx);
    const ys = niceTicks(this.ylim[0], this.ylim[1], ny);

    if (this.showGrid) {
      g.strokeStyle = cssVar('--grid'); g.lineWidth = 1;
      g.beginPath();
      for (const x of xs) { const p = Math.round(this.px(x)) + .5; g.moveTo(p, this.pad.t); g.lineTo(p, this.h - this.pad.b); }
      for (const y of ys) { const p = Math.round(this.py(y)) + .5; g.moveTo(this.pad.l, p); g.lineTo(this.w - this.pad.r, p); }
      g.stroke();
    }

    // zero lines
    g.strokeStyle = cssVar('--axis'); g.lineWidth = 1;
    g.beginPath();
    if (this.ylim[0] < 0 && this.ylim[1] > 0) { const p = Math.round(this.py(0)) + .5; g.moveTo(this.pad.l, p); g.lineTo(this.w - this.pad.r, p); }
    if (this.xlim[0] < 0 && this.xlim[1] > 0) { const p = Math.round(this.px(0)) + .5; g.moveTo(p, this.pad.t); g.lineTo(p, this.h - this.pad.b); }
    g.stroke();

    // frame
    g.strokeStyle = cssVar('--border');
    g.strokeRect(this.pad.l + .5, this.pad.t + .5, this.w - this.pad.l - this.pad.r - 1, this.h - this.pad.t - this.pad.b - 1);

    if (ticks) {
      g.fillStyle = cssVar('--text-faint');
      g.font = '10px ' + MONO;
      g.textAlign = 'center'; g.textBaseline = 'top';
      for (const x of xs) g.fillText(fmt(x), this.px(x), this.h - this.pad.b + 5);
      g.textAlign = 'right'; g.textBaseline = 'middle';
      for (const y of ys) g.fillText(fmt(y), this.pad.l - 6, this.py(y));
    }

    if (this.xlabel) {
      g.fillStyle = cssVar('--text-dim'); g.font = '11px ' + SANS;
      g.textAlign = 'center'; g.textBaseline = 'bottom';
      g.fillText(this.xlabel, (this.pad.l + this.w - this.pad.r) / 2, this.h - 1);
    }
    if (this.ylabel) {
      g.save();
      g.fillStyle = cssVar('--text-dim'); g.font = '11px ' + SANS;
      g.translate(10, (this.pad.t + this.h - this.pad.b) / 2);
      g.rotate(-Math.PI / 2);
      g.textAlign = 'center'; g.textBaseline = 'top';
      g.fillText(this.ylabel, 0, 0);
      g.restore();
    }
    return this;
  }

  /* ---------- primitives (data coords) ---------- */

  line(pts, o = {}) {
    if (!pts || pts.length < 2) return this;
    const g = this.ctx;
    g.save();
    g.strokeStyle = o.color || cssVar('--s1');
    g.lineWidth = o.width || 2;
    g.lineJoin = 'round'; g.lineCap = 'round';
    if (o.dash) g.setLineDash(o.dash);
    if (o.alpha != null) g.globalAlpha = o.alpha;
    g.beginPath();
    let started = false;
    for (const p of pts) {
      const x = p[0], y = p[1];
      if (!isFinite(x) || !isFinite(y)) { started = false; continue; }
      const X = this.px(x), Y = this.py(y);
      if (!started) { g.moveTo(X, Y); started = true; } else g.lineTo(X, Y);
    }
    g.stroke();
    g.restore();
    return this;
  }

  /** Plot y = f(x) sampled across the x range. */
  fn(f, o = {}) {
    const n = o.n || 240;
    const a = o.from != null ? o.from : this.xlim[0];
    const b = o.to != null ? o.to : this.xlim[1];
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const x = a + (b - a) * i / n;
      let y = f(x);
      if (!isFinite(y)) y = NaN;
      // avoid drawing wild vertical jumps off-screen
      if (y > this.ylim[1] + (this.ylim[1] - this.ylim[0]) * 4) y = NaN;
      if (y < this.ylim[0] - (this.ylim[1] - this.ylim[0]) * 4) y = NaN;
      pts.push([x, y]);
    }
    return this.line(pts, o);
  }

  area(pts, o = {}) {
    if (!pts || pts.length < 2) return this;
    const g = this.ctx;
    const base = o.base != null ? o.base : Math.max(this.ylim[0], 0);
    g.save();
    g.fillStyle = o.color || alpha(cssVar('--s1'), .18);
    g.beginPath();
    g.moveTo(this.px(pts[0][0]), this.py(base));
    for (const p of pts) g.lineTo(this.px(p[0]), this.py(p[1]));
    g.lineTo(this.px(pts[pts.length - 1][0]), this.py(base));
    g.closePath(); g.fill();
    g.restore();
    return this;
  }

  points(pts, o = {}) {
    const g = this.ctx;
    const r = o.r || 4;
    g.save();
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const col = typeof o.color === 'function' ? o.color(p, i) : (o.color || cssVar('--s1'));
      const rr = typeof r === 'function' ? r(p, i) : r;
      g.fillStyle = col;
      g.beginPath();
      const X = this.px(p[0]), Y = this.py(p[1]);
      if (o.shape === 'square') g.rect(X - rr, Y - rr, rr * 2, rr * 2);
      else if (o.shape === 'cross') {
        g.strokeStyle = col; g.lineWidth = o.width || 2;
        g.moveTo(X - rr, Y - rr); g.lineTo(X + rr, Y + rr);
        g.moveTo(X + rr, Y - rr); g.lineTo(X - rr, Y + rr);
        g.stroke(); continue;
      } else g.arc(X, Y, rr, 0, 6.2832);
      g.fill();
      if (o.stroke) { g.strokeStyle = o.stroke; g.lineWidth = o.strokeWidth || 1.5; g.stroke(); }
    }
    g.restore();
    return this;
  }

  arrow(x0, y0, x1, y1, o = {}) {
    const g = this.ctx;
    const X0 = this.px(x0), Y0 = this.py(y0), X1 = this.px(x1), Y1 = this.py(y1);
    const dx = X1 - X0, dy = Y1 - Y0;
    const len = Math.hypot(dx, dy);
    if (len < .5) return this;
    const head = Math.min(o.head || 8, len * .5);
    const ang = Math.atan2(dy, dx);
    g.save();
    g.strokeStyle = o.color || cssVar('--s2');
    g.fillStyle = o.color || cssVar('--s2');
    g.lineWidth = o.width || 2;
    g.lineCap = 'round';
    if (o.alpha != null) g.globalAlpha = o.alpha;
    if (o.dash) g.setLineDash(o.dash);
    g.beginPath();
    g.moveTo(X0, Y0);
    g.lineTo(X1 - Math.cos(ang) * head * .8, Y1 - Math.sin(ang) * head * .8);
    g.stroke();
    g.setLineDash([]);
    g.beginPath();
    g.moveTo(X1, Y1);
    g.lineTo(X1 - head * Math.cos(ang - .4), Y1 - head * Math.sin(ang - .4));
    g.lineTo(X1 - head * Math.cos(ang + .4), Y1 - head * Math.sin(ang + .4));
    g.closePath(); g.fill();
    g.restore();
    return this;
  }

  /** Filled/stroked circle in data coords with pixel radius. */
  circle(x, y, rPx, o = {}) {
    const g = this.ctx;
    g.save();
    g.beginPath();
    g.arc(this.px(x), this.py(y), rPx, 0, 6.2832);
    if (o.fill) { g.fillStyle = o.fill; g.fill(); }
    if (o.stroke) { g.strokeStyle = o.stroke; g.lineWidth = o.width || 1.5; if (o.dash) g.setLineDash(o.dash); g.stroke(); }
    g.restore();
    return this;
  }

  /** Ellipse from a 2x2 covariance matrix (for Gaussians). */
  ellipse(cx, cy, cov, o = {}) {
    const [a, b, , d] = [cov[0][0], cov[0][1], cov[1][0], cov[1][1]];
    const tr = a + d, det = a * d - b * b;
    const disc = Math.max(0, tr * tr / 4 - det);
    const l1 = tr / 2 + Math.sqrt(disc), l2 = tr / 2 - Math.sqrt(disc);
    let vx = 1, vy = 0;
    if (Math.abs(b) > 1e-12) { vx = l1 - d; vy = b; }
    const ang = Math.atan2(vy, vx);
    const g = this.ctx;
    const k = o.n || 2; // n-sigma
    g.save();
    g.translate(this.px(cx), this.py(cy));
    g.rotate(-ang);
    g.beginPath();
    g.ellipse(0, 0, Math.sqrt(Math.max(l1, 1e-9)) * k * this.sx, Math.sqrt(Math.max(l2, 1e-9)) * k * this.sy, 0, 0, 6.2832);
    if (o.fill) { g.fillStyle = o.fill; g.fill(); }
    g.strokeStyle = o.stroke || cssVar('--s4'); g.lineWidth = o.width || 1.5;
    if (o.dash) g.setLineDash(o.dash);
    g.stroke();
    g.restore();
    return this;
  }

  bars(vals, o = {}) {
    const g = this.ctx;
    const n = vals.length;
    const x0 = o.x0 != null ? o.x0 : 0;
    const gap = o.gap != null ? o.gap : 0.18;
    const base = o.base != null ? o.base : 0;
    g.save();
    for (let i = 0; i < n; i++) {
      const v = vals[i];
      const col = typeof o.color === 'function' ? o.color(v, i) : (o.color || cssVar('--s1'));
      g.fillStyle = col;
      const L = this.px(x0 + i + gap / 2), R = this.px(x0 + i + 1 - gap / 2);
      const T = this.py(Math.max(v, base)), B = this.py(Math.min(v, base));
      g.fillRect(L, T, Math.max(R - L, 1), Math.max(B - T, 0.5));
      if (o.stroke) { g.strokeStyle = o.stroke; g.lineWidth = 1; g.strokeRect(L, T, R - L, B - T); }
    }
    g.restore();
    return this;
  }

  /** Scalar field as an image. f(x,y)->value; norm maps value->[0,1]. */
  heat(f, o = {}) {
    const step = o.step || 4;
    const g = this.ctx;
    const L = this.pad.l, T = this.pad.t;
    const W = this.w - this.pad.l - this.pad.r, H = this.h - this.pad.t - this.pad.b;
    const cmap = o.cmap || divergeMap;

    // first pass: find range if not given
    let lo = o.lo, hi = o.hi;
    const vals = [];
    for (let py = 0; py < H; py += step) {
      const row = [];
      for (let px = 0; px < W; px += step) {
        row.push(f(this.ix(L + px + step / 2), this.iy(T + py + step / 2)));
      }
      vals.push(row);
    }
    if (lo == null || hi == null) {
      let mn = Infinity, mx = -Infinity;
      for (const r of vals) for (const v of r) { if (isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; } }
      lo = lo == null ? mn : lo; hi = hi == null ? mx : hi;
    }
    const range = (hi - lo) || 1;

    g.save();
    if (o.alpha != null) g.globalAlpha = o.alpha;
    for (let i = 0; i < vals.length; i++) {
      for (let j = 0; j < vals[i].length; j++) {
        let t = (vals[i][j] - lo) / range;
        if (!isFinite(t)) continue;
        g.fillStyle = cmap(Math.max(0, Math.min(1, t)));
        g.fillRect(L + j * step, T + i * step, step + 1, step + 1);
      }
    }
    g.restore();
    this._heatRange = [lo, hi];
    return this;
  }

  /** Contour lines of f via marching squares. */
  contour(f, levels, o = {}) {
    const g = this.ctx;
    const nx = o.nx || 90, ny = o.ny || 70;
    const L = this.pad.l, T = this.pad.t;
    const W = this.w - this.pad.l - this.pad.r, H = this.h - this.pad.t - this.pad.b;
    const gx = [], gy = [], Z = [];
    for (let i = 0; i <= nx; i++) gx.push(this.ix(L + W * i / nx));
    for (let j = 0; j <= ny; j++) gy.push(this.iy(T + H * j / ny));
    for (let j = 0; j <= ny; j++) {
      const row = [];
      for (let i = 0; i <= nx; i++) row.push(f(gx[i], gy[j]));
      Z.push(row);
    }
    g.save();
    g.lineWidth = o.width || 1;
    if (o.alpha != null) g.globalAlpha = o.alpha;
    for (let li = 0; li < levels.length; li++) {
      const lv = levels[li];
      g.strokeStyle = typeof o.color === 'function' ? o.color(lv, li) : (o.color || alpha(cssVar('--axis'), .8));
      g.beginPath();
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const v = [Z[j][i], Z[j][i + 1], Z[j + 1][i + 1], Z[j + 1][i]];
          const X = [gx[i], gx[i + 1], gx[i + 1], gx[i]];
          const Y = [gy[j], gy[j], gy[j + 1], gy[j + 1]];
          const seg = [];
          for (let k = 0; k < 4; k++) {
            const k2 = (k + 1) % 4;
            const a = v[k], b = v[k2];
            if ((a < lv) !== (b < lv)) {
              const t = (lv - a) / (b - a);
              seg.push([X[k] + (X[k2] - X[k]) * t, Y[k] + (Y[k2] - Y[k]) * t]);
            }
          }
          for (let k = 0; k + 1 < seg.length; k += 2) {
            g.moveTo(this.px(seg[k][0]), this.py(seg[k][1]));
            g.lineTo(this.px(seg[k + 1][0]), this.py(seg[k + 1][1]));
          }
        }
      }
      g.stroke();
    }
    g.restore();
    return this;
  }

  /** Vector field: fn(x,y) -> [u,v]. */
  quiver(fn, o = {}) {
    const nx = o.nx || 14, ny = o.ny || 10;
    const scale = o.scale || 1;
    let maxLen = 0;
    const arr = [];
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const x = this.xlim[0] + (this.xlim[1] - this.xlim[0]) * (i + .5) / nx;
        const y = this.ylim[0] + (this.ylim[1] - this.ylim[0]) * (j + .5) / ny;
        const [u, v] = fn(x, y);
        const L = Math.hypot(u * this.sx, v * this.sy);
        if (isFinite(L)) maxLen = Math.max(maxLen, L);
        arr.push([x, y, u, v]);
      }
    }
    const cell = Math.min((this.w - this.pad.l - this.pad.r) / nx, (this.h - this.pad.t - this.pad.b) / ny) * 0.82 * scale;
    const k = maxLen > 0 ? cell / maxLen : 0;
    for (const [x, y, u, v] of arr) {
      const L = Math.hypot(u * this.sx, v * this.sy);
      if (!isFinite(L) || L < 1e-9) continue;
      const dx = u * k, dy = v * k;
      this.arrow(x, y, x + dx, y + dy, {
        color: o.color || alpha(cssVar('--s1'), .55),
        width: o.width || 1.3,
        head: Math.min(6, Math.max(3, L * k * .35)),
        alpha: o.alphaBy ? Math.max(.25, Math.min(1, L / maxLen)) : (o.alpha != null ? o.alpha : .8),
      });
    }
    return this;
  }

  /** Matrix as a grid of colored cells (attention maps, weights). */
  matrix(M, o = {}) {
    const g = this.ctx;
    const rows = M.length, cols = M[0].length;
    const L = o.x0 != null ? this.px(o.x0) : this.pad.l;
    const T = o.y0 != null ? this.py(o.y0) : this.pad.t;
    const W = o.w != null ? o.w * this.sx : this.w - this.pad.l - this.pad.r;
    const H = o.h != null ? o.h * this.sy : this.h - this.pad.t - this.pad.b;
    const cw = W / cols, ch = H / rows;
    let lo = o.lo, hi = o.hi;
    if (lo == null || hi == null) {
      let mn = Infinity, mx = -Infinity;
      for (const r of M) for (const v of r) { if (v < mn) mn = v; if (v > mx) mx = v; }
      lo = lo == null ? mn : lo; hi = hi == null ? mx : hi;
    }
    const rng = (hi - lo) || 1;
    const cmap = o.cmap || seqMap;
    g.save();
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        g.fillStyle = cmap((M[i][j] - lo) / rng);
        g.fillRect(L + j * cw, T + i * ch, cw + .6, ch + .6);
        if (o.showVals && cw > 26 && ch > 14) {
          g.fillStyle = (M[i][j] - lo) / rng > .55 ? '#0b0f14' : '#e6edf3';
          g.font = `${Math.min(10, ch * .5)}px ` + MONO;
          g.textAlign = 'center'; g.textBaseline = 'middle';
          g.fillText(fmt(M[i][j], o.digits || 2), L + j * cw + cw / 2, T + i * ch + ch / 2);
        }
      }
    }
    if (o.grid) {
      g.strokeStyle = alpha(cssVar('--bg'), .5); g.lineWidth = .5;
      g.beginPath();
      for (let i = 0; i <= rows; i++) { g.moveTo(L, T + i * ch); g.lineTo(L + W, T + i * ch); }
      for (let j = 0; j <= cols; j++) { g.moveTo(L + j * cw, T); g.lineTo(L + j * cw, T + H); }
      g.stroke();
    }
    g.restore();
    this._matBox = { L, T, W, H, cw, ch, rows, cols };
    return this;
  }

  /* ---------- pixel-space helpers ---------- */

  text(x, y, str, o = {}) {
    const g = this.ctx;
    g.save();
    g.fillStyle = o.color || cssVar('--text-dim');
    g.font = `${o.weight || ''} ${o.size || 11}px ${o.mono ? MONO : SANS}`.trim();
    g.textAlign = o.align || 'left';
    g.textBaseline = o.baseline || 'middle';
    const X = o.pixel ? x : this.px(x), Y = o.pixel ? y : this.py(y);
    if (o.bg) {
      const m = g.measureText(str);
      const pad = 3;
      g.fillStyle = o.bg;
      const tw = m.width, th = (o.size || 11) + 2;
      let bx = X; if (g.textAlign === 'center') bx = X - tw / 2; if (g.textAlign === 'right') bx = X - tw;
      g.fillRect(bx - pad, Y - th / 2 - pad, tw + pad * 2, th + pad * 2);
      g.fillStyle = o.color || cssVar('--text-dim');
    }
    g.fillText(str, X, Y);
    g.restore();
    return this;
  }

  legend(items, o = {}) {
    const g = this.ctx;
    const pad = 7, lh = 15;
    g.save();
    g.font = '11px ' + SANS;
    let wMax = 0;
    for (const it of items) wMax = Math.max(wMax, g.measureText(it.label).width);
    const bw = wMax + 30, bh = items.length * lh + pad * 2 - 3;
    const pos = o.pos || 'tr';
    let X = this.w - this.pad.r - bw - 6, Y = this.pad.t + 6;
    if (pos.includes('l')) X = this.pad.l + 6;
    if (pos.includes('b')) Y = this.h - this.pad.b - bh - 6;
    g.fillStyle = alpha(cssVar('--bg-elev'), .92);
    g.strokeStyle = cssVar('--border');
    g.lineWidth = 1;
    roundRect(g, X, Y, bw, bh, 6); g.fill(); g.stroke();
    items.forEach((it, i) => {
      const cy = Y + pad + i * lh + 5;
      g.fillStyle = it.color;
      if (it.dash) {
        g.strokeStyle = it.color; g.lineWidth = 2; g.setLineDash([4, 3]);
        g.beginPath(); g.moveTo(X + pad, cy); g.lineTo(X + pad + 14, cy); g.stroke();
        g.setLineDash([]);
      } else if (it.shape === 'dot') {
        g.beginPath(); g.arc(X + pad + 7, cy, 3.5, 0, 6.2832); g.fill();
      } else {
        g.fillRect(X + pad, cy - 2, 14, 4);
      }
      g.fillStyle = cssVar('--text-dim');
      g.textAlign = 'left'; g.textBaseline = 'middle';
      g.fillText(it.label, X + pad + 20, cy);
    });
    g.restore();
    return this;
  }

  /* ---------- interaction ---------- */

  /**
   * Enable pointer interaction.
   * @param {object} h handlers {down(x,y,e), move(x,y,e,dragging), up(x,y,e), hover(x,y)}
   */
  interact(h) {
    const toData = (e) => {
      const r = this.c.getBoundingClientRect();
      const cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
      const cy = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
      return [this.ix(cx), this.iy(cy), cx, cy];
    };
    let dragging = false;
    const down = (e) => {
      const [x, y, cx, cy] = toData(e);
      dragging = h.down ? h.down(x, y, e, cx, cy) !== false : true;
      if (dragging) e.preventDefault();
    };
    const move = (e) => {
      const [x, y, cx, cy] = toData(e);
      if (dragging && h.move) { h.move(x, y, e, cx, cy); e.preventDefault(); }
      else if (h.hover) h.hover(x, y, e, cx, cy);
    };
    const up = (e) => {
      if (dragging && h.up) { const [x, y] = toData(e); h.up(x, y, e); }
      dragging = false;
    };
    this.c.addEventListener('pointerdown', down);
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    this.c.addEventListener('pointerleave', () => { if (h.leave) h.leave(); });
    this._cleanup = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    this.c.style.cursor = h.cursor || 'crosshair';
    return this;
  }

  /** Find index of nearest point (data coords) within pixel radius. */
  nearest(pts, x, y, rPx = 14) {
    let best = -1, bd = rPx * rPx;
    const X = this.px(x), Y = this.py(y);
    for (let i = 0; i < pts.length; i++) {
      const d = (this.px(pts[i][0]) - X) ** 2 + (this.py(pts[i][1]) - Y) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }
}

const MONO = 'ui-monospace, SFMono-Regular, Menlo, monospace';
const SANS = '-apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif';

function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}

export function niceTicks(lo, hi, n) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return out;
}

export function fmt(v, digits) {
  if (!isFinite(v)) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (digits != null) return v.toFixed(digits);
  if (a >= 1e6 || a < 1e-4) return v.toExponential(1).replace('e+', 'e');
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
}

/* ---------- deterministic RNG (reproducible demos) ---------- */

export function rng(seed = 42) {
  let s = seed >>> 0;
  const f = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  f.normal = (mu = 0, sd = 1) => {
    let u = 0, v = 0;
    while (u === 0) u = f();
    while (v === 0) v = f();
    return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  f.pick = (arr) => arr[Math.floor(f() * arr.length)];
  f.int = (a, b) => a + Math.floor(f() * (b - a));
  f.shuffle = (arr) => {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(f() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  };
  return f;
}

/* ---------- small linear algebra used across demos ---------- */

export const LA = {
  zeros: (n, m) => (m == null ? new Array(n).fill(0) : Array.from({ length: n }, () => new Array(m).fill(0))),
  matmul(A, B) {
    const n = A.length, k = B.length, m = B[0].length;
    const C = LA.zeros(n, m);
    for (let i = 0; i < n; i++) for (let p = 0; p < k; p++) {
      const a = A[i][p]; if (a === 0) continue;
      for (let j = 0; j < m; j++) C[i][j] += a * B[p][j];
    }
    return C;
  },
  matvec: (A, x) => A.map((r) => r.reduce((s, v, j) => s + v * x[j], 0)),
  transpose: (A) => A[0].map((_, j) => A.map((r) => r[j])),
  dot: (a, b) => a.reduce((s, v, i) => s + v * b[i], 0),
  norm: (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0)),
  add: (a, b) => a.map((v, i) => v + b[i]),
  sub: (a, b) => a.map((v, i) => v - b[i]),
  scale: (a, k) => a.map((v) => v * k),
  /** Solve Ax=b by Gaussian elimination with partial pivoting. */
  solve(A, b) {
    const n = A.length;
    const M = A.map((r, i) => r.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      if (Math.abs(M[c][c]) < 1e-12) continue;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    return M.map((r, i) => (Math.abs(M[i][i]) < 1e-12 ? 0 : r[n] / M[i][i]));
  },
  /** Symmetric 2x2 eigen-decomposition -> {vals:[l1,l2], vecs:[[..],[..]]} (l1>=l2). */
  eig2(M) {
    const a = M[0][0], b = M[0][1], d = M[1][1];
    const tr = a + d, det = a * d - b * b;
    const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
    const v = (l) => {
      if (Math.abs(b) > 1e-12) { const n = Math.hypot(l - d, b); return [(l - d) / n, b / n]; }
      return Math.abs(a - l) < 1e-12 ? [1, 0] : [0, 1];
    };
    return { vals: [l1, l2], vecs: [v(l1), v(l2)] };
  },
  softmax(z, temp = 1) {
    const m = Math.max(...z);
    const e = z.map((v) => Math.exp((v - m) / temp));
    const s = e.reduce((a, v) => a + v, 0);
    return e.map((v) => v / s);
  },
};

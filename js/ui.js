/* ============================================================
   ui.js — builds the interactive figure panels.

   A "panel" = titled card + canvas + control strip + readout line.
   Handles: state from controls, redraw on change, animation loop,
   resize, theme changes, and cleanup when the lesson unmounts.
   ============================================================ */

import { Plot, clearColorCache } from './plot.js';
import { md, tex } from './md.js';

const live = new Set();   // panels currently mounted

export function disposeAll() {
  for (const p of live) p.dispose();
  live.clear();
}

/**
 * @param {HTMLElement} host  container to fill
 * @param {object} o
 *   title, caption, height, aspect, plot: {xlim,ylim,...},
 *   controls: [...], draw(plot, state, panel), init(panel),
 *   animate(state, panel, dt) -> called each frame when playing,
 *   interact: {down,move,up,hover}, noCanvas: bool
 */
export function panel(host, o = {}) {
  const el = document.createElement('div');
  el.className = 'viz';

  if (o.title) {
    const head = document.createElement('div');
    head.className = 'viz-head';
    head.innerHTML = `<span class="viz-title">${o.title}</span>`;
    if (o.headRight) {
      const r = document.createElement('div');
      r.className = 'btn-row';
      r.innerHTML = o.headRight;
      head.appendChild(r);
    }
    el.appendChild(head);
  }

  const body = document.createElement('div');
  body.className = 'viz-body';
  el.appendChild(body);

  const canvas = document.createElement('canvas');
  if (!o.noCanvas) body.appendChild(canvas);

  const extra = document.createElement('div');
  extra.className = 'viz-extra';
  body.appendChild(extra);

  const readoutEl = document.createElement('div');
  readoutEl.className = 'viz-readout';
  readoutEl.hidden = true;
  el.appendChild(readoutEl);

  const ctlEl = document.createElement('div');
  ctlEl.className = 'controls';
  if (o.controls && o.controls.length) el.appendChild(ctlEl);

  if (o.caption) {
    const cap = document.createElement('div');
    cap.className = 'viz-caption';
    cap.innerHTML = md(o.caption);
    el.appendChild(cap);
  }

  host.appendChild(el);

  /* ---- state ---- */
  const state = {};
  const setters = {};
  for (const c of o.controls || []) {
    if (c.key !== undefined && c.value !== undefined) state[c.key] = c.value;
  }
  Object.assign(state, o.state || {});

  const P = {
    el, canvas, extra, state, playing: false,
    plot: null,
    _raf: null, _last: 0, _disposed: false,
  };

  if (!o.noCanvas) {
    P.plot = new Plot(canvas, Object.assign({ height: o.height, aspect: o.aspect }, o.plot));
  }

  P.set = (k, v, silent) => {
    state[k] = v;
    if (setters[k]) setters[k](v);
    if (!silent) P.redraw();
  };

  P.readout = (obj) => {
    if (!obj) { readoutEl.hidden = true; return; }
    readoutEl.hidden = false;
    readoutEl.innerHTML = Object.entries(obj)
      .map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join('');
  };

  P.redraw = () => {
    if (P._disposed) return;
    if (P.plot && o.draw) {
      try { o.draw(P.plot, state, P); }
      catch (e) { console.error('viz draw error:', o.title, e); }
    } else if (o.draw) {
      try { o.draw(null, state, P); } catch (e) { console.error(e); }
    }
  };

  P.play = (on) => {
    if (on === undefined) on = !P.playing;
    P.playing = on;
    for (const b of el.querySelectorAll('[data-play]')) b.textContent = on ? '❚❚ Pause' : '▶ Play';
    if (on && !P._raf) {
      P._last = performance.now();
      const step = (t) => {
        if (!P.playing || P._disposed) { P._raf = null; return; }
        const dt = Math.min((t - P._last) / 1000, 0.05);
        P._last = t;
        if (o.animate) {
          try { o.animate(state, P, dt); } catch (e) { console.error(e); P.playing = false; }
        }
        P.redraw();
        P._raf = requestAnimationFrame(step);
      };
      P._raf = requestAnimationFrame(step);
    }
  };

  P.dispose = () => {
    P._disposed = true;
    P.playing = false;
    if (P._raf) cancelAnimationFrame(P._raf);
    if (P.plot && P.plot._cleanup) P.plot._cleanup();
    ro && ro.disconnect();
    live.delete(P);
  };

  /* ---- controls ---- */
  for (const c of o.controls || []) {
    const wrap = document.createElement('div');

    if (c.type === 'slider') {
      wrap.className = 'ctl';
      const fmtv = c.fmt || ((v) => String(Math.round(v * 1000) / 1000));
      wrap.innerHTML = `<span class="ctl-label"><span>${c.label}</span><b></b></span>`;
      const inp = document.createElement('input');
      inp.type = 'range';
      inp.min = c.min; inp.max = c.max; inp.step = c.step != null ? c.step : 'any';
      inp.value = c.value;
      const out = wrap.querySelector('b');
      out.textContent = fmtv(c.value);
      inp.addEventListener('input', () => {
        state[c.key] = +inp.value;
        out.textContent = fmtv(+inp.value);
        if (c.onChange) c.onChange(+inp.value, P);
        P.redraw();
      });
      setters[c.key] = (v) => { inp.value = v; out.textContent = fmtv(v); };
      wrap.appendChild(inp);

    } else if (c.type === 'select') {
      wrap.className = 'ctl';
      wrap.innerHTML = `<span class="ctl-label"><span>${c.label}</span></span>`;
      const sel = document.createElement('select');
      for (const opt of c.options) {
        const oEl = document.createElement('option');
        const val = opt.value !== undefined ? opt.value : opt;
        oEl.value = val; oEl.textContent = opt.label !== undefined ? opt.label : val;
        sel.appendChild(oEl);
      }
      sel.value = c.value;
      sel.addEventListener('change', () => {
        state[c.key] = sel.value;
        if (c.onChange) c.onChange(sel.value, P);
        P.redraw();
      });
      setters[c.key] = (v) => { sel.value = v; };
      wrap.appendChild(sel);

    } else if (c.type === 'check') {
      wrap.className = 'ctl ctl-check';
      const id = 'chk' + Math.random().toString(36).slice(2, 8);
      wrap.innerHTML = `<input type="checkbox" id="${id}"><label for="${id}">${c.label}</label>`;
      const inp = wrap.querySelector('input');
      inp.checked = !!c.value;
      inp.addEventListener('change', () => {
        state[c.key] = inp.checked;
        if (c.onChange) c.onChange(inp.checked, P);
        P.redraw();
      });
      setters[c.key] = (v) => { inp.checked = !!v; };

    } else if (c.type === 'button') {
      wrap.className = 'ctl-check';
      const b = document.createElement('button');
      b.className = 'btn' + (c.primary ? ' primary' : '');
      b.textContent = c.label;
      b.addEventListener('click', () => { c.onClick(state, P); P.redraw(); });
      wrap.appendChild(b);

    } else if (c.type === 'play') {
      wrap.className = 'ctl-check';
      const b = document.createElement('button');
      b.className = 'btn primary';
      b.dataset.play = '1';
      b.textContent = '▶ Play';
      b.addEventListener('click', () => P.play());
      wrap.appendChild(b);

    } else if (c.type === 'label') {
      wrap.className = 'ctl-check';
      wrap.innerHTML = `<span style="font-size:11.5px;color:var(--text-faint)">${c.label}</span>`;
    }
    ctlEl.appendChild(wrap);
  }

  /* ---- resize + theme ---- */
  let ro = null;
  if (P.plot) {
    let last = 0;
    ro = new ResizeObserver(() => {
      const w = canvas.clientWidth;
      if (Math.abs(w - last) < 2) return;
      last = w;
      P.plot.resize();
      P.redraw();
    });
    ro.observe(body);
    if (o.interact) P.plot.interact(o.interact(state, P));
  }

  live.add(P);
  if (o.init) o.init(P, state);
  P.redraw();
  if (o.autoplay) P.play(true);
  return P;
}

/** Rebuild all live panels' colors after a theme switch. */
export function refreshTheme() {
  clearColorCache();
  for (const p of live) { if (p.plot) p.plot.resize(); p.redraw(); }
}

/** Inline HTML fragment builder for extras inside a panel. */
export function html(strings, ...vals) {
  const d = document.createElement('div');
  d.innerHTML = String.raw({ raw: strings }, ...vals);
  return d;
}

export { md, tex };

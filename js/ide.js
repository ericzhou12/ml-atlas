/* ============================================================
   ide.js — the code lab panel.

   Two runtimes:
     • Python  — CPython 3.12 + NumPy via Pyodide (WebAssembly),
                 served from vendor/ so it works with no network.
     • JS      — instant sandbox with a small linear-algebra helper.

   Both capture stdout/stderr into the output pane.
   ============================================================ */

import { LA } from './plot.js';
import { highlight } from './md.js';

const el = (id) => document.getElementById(id);

let mode = 'python';
let pyodide = null;
let booting = null;
let lastSnippets = [];

const DEFAULT_PY = `# Python 3.12 + NumPy, running locally in your browser.
# Press Cmd/Ctrl+Enter to run.

import numpy as np

# Least squares by hand: fit y = w0 + w1*x
rng = np.random.default_rng(0)
x = np.linspace(0, 1, 40)
y = 2.5 * x + 0.7 + rng.normal(0, 0.2, 40)

X = np.column_stack([np.ones_like(x), x])          # design matrix (40, 2)
w = np.linalg.solve(X.T @ X, X.T @ y)              # normal equations

print("w (intercept, slope) =", np.round(w, 4))
print("train MSE            =", np.mean((X @ w - y) ** 2).round(5))
print()
aplot(y, X @ w)   # helper: ascii scatter of truth vs fit
`;

const DEFAULT_JS = `// JavaScript sandbox — instant, no boot time.
// LA gives you matmul, matvec, dot, norm, solve, eig2, softmax.

const X = [[1,0],[1,1],[1,2],[1,3]];
const y = [1.0, 2.9, 5.2, 6.8];

const Xt  = LA.transpose(X);
const XtX = LA.matmul(Xt, X);
const Xty = LA.matvec(Xt, y);
const w   = LA.solve(XtX, Xty);

console.log("weights:", w.map(v => v.toFixed(4)));
console.log("preds:  ", LA.matvec(X, w).map(v => v.toFixed(3)));
`;

/* ---------------- package loading ----------------

   Everything below numpy is vendored but loaded ON DEMAND, so booting the
   sandbox stays fast. We scan the source for imports and pull only what it
   needs. `deps` lists what pyodide must fetch (it resolves transitively). */

const PKG_TRIGGERS = [
  { pkg: 'matplotlib', re: /\b(matplotlib|pyplot|\bplt\b)/ },
  { pkg: 'scipy', re: /\bscipy\b/ },
  { pkg: 'scikit-learn', re: /\b(sklearn|scikit[-_]learn)\b/ },
  { pkg: 'pandas', re: /\b(pandas|\bpd\b)\b/ },
  { pkg: 'sympy', re: /\bsympy\b/ },
  { pkg: 'micropip', re: /\bmicropip\b/ },
];

const loaded = new Set(['numpy']);

/** Figure out which vendored packages a snippet needs. */
function neededPackages(code) {
  const out = [];
  for (const { pkg, re } of PKG_TRIGGERS) {
    if (!loaded.has(pkg) && re.test(code)) out.push(pkg);
  }
  return out;
}

/* Matplotlib runs headless (AGG) and we lift the rendered figures out as PNGs
   after each run. Nothing tries to drive a live canvas, which keeps show()
   working the way people expect in a notebook. */
const PY_MPL_SETUP = `
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as _plt
import io as _io, base64 as _b64

_plt.show = lambda *a, **k: None      # figures are captured automatically

def _mlatlas_figures():
    out = []
    for num in _plt.get_fignums():
        fig = _plt.figure(num)
        buf = _io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                        facecolor=fig.get_facecolor(), edgecolor="none")
            out.append(_b64.b64encode(buf.getvalue()).decode())
        except Exception as e:
            print("figure capture failed:", e)
    _plt.close("all")
    return out
`;

/** Match matplotlib's colours to the app's current theme. */
function pyMplTheme(dark) {
  const c = dark
    ? { fg: '#e6edf3', dim: '#9aa8b8', bg: '#0a0e14', grid: '#202a36' }
    : { fg: '#16202b', dim: '#566172', bg: '#ffffff', grid: '#e7ecf2' };
  const cycle = ['#5aa9ff', '#ff8f5a', '#37d6a8', '#b78bff', '#ffd35a', '#ff6b9d', '#4fd3e8', '#a3b18a'];
  return `
import matplotlib as _mpl
_mpl.rcParams.update({
    "figure.facecolor": "${c.bg}", "axes.facecolor": "${c.bg}",
    "savefig.facecolor": "${c.bg}",
    "text.color": "${c.fg}", "axes.labelcolor": "${c.fg}",
    "axes.edgecolor": "${c.dim}", "axes.titlecolor": "${c.fg}",
    "xtick.color": "${c.dim}", "ytick.color": "${c.dim}",
    "grid.color": "${c.grid}", "axes.grid": True, "grid.alpha": 0.6,
    "legend.facecolor": "${c.bg}", "legend.edgecolor": "${c.grid}",
    "legend.labelcolor": "${c.fg}",
    "axes.prop_cycle": _mpl.cycler(color=${JSON.stringify(cycle).replace(/"/g, "'")}),
    "figure.figsize": (6.0, 3.6), "figure.dpi": 110, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})
`;
}

/* ASCII plotting + tiny helpers injected into the Python namespace. */
const PY_PRELUDE = `
import sys as _sys

def aplot(*series, width=64, height=14, labels=None):
    """Quick ASCII line/scatter plot of one or more 1-D sequences."""
    import numpy as _np
    ss = [_np.asarray(s, dtype=float).ravel() for s in series]
    if not ss: return
    n = max(len(s) for s in ss)
    lo = min(float(_np.nanmin(s)) for s in ss)
    hi = max(float(_np.nanmax(s)) for s in ss)
    if hi - lo < 1e-12: hi = lo + 1.0
    marks = "*o+x.#@"
    grid = [[" "] * width for _ in range(height)]
    for k, s in enumerate(ss):
        for i, v in enumerate(s):
            if not _np.isfinite(v): continue
            cx = int(round(i / max(len(s) - 1, 1) * (width - 1)))
            cy = int(round((1 - (v - lo) / (hi - lo)) * (height - 1)))
            grid[cy][cx] = marks[k % len(marks)]
    top, bot = f"{hi:.3g}", f"{lo:.3g}"
    pad = max(len(top), len(bot))
    print(f"{top:>{pad}} ┤" + "".join(grid[0]))
    for r in grid[1:-1]:
        print(" " * pad + " │" + "".join(r))
    print(f"{bot:>{pad}} ┤" + "".join(grid[-1]))
    print(" " * pad + " └" + "─" * width)
    if labels:
        print(" " * pad + "  " + "   ".join(f"{marks[i % len(marks)]} {l}" for i, l in enumerate(labels)))

def describe(a, name="array"):
    """Shape / dtype / range / mean / std summary — the thing you always print."""
    import numpy as _np
    a = _np.asarray(a)
    print(f"{name}: shape={a.shape} dtype={a.dtype} "
          f"min={a.min():.4g} max={a.max():.4g} mean={a.mean():.4g} std={a.std():.4g}")
`;

/* ---------------- editor ---------------- */

const editor = () => el('ide-editor');
const output = () => el('ide-output');

function setStatus(msg, cls = '') {
  const s = el('ide-status');
  s.className = 'ide-status ' + cls;
  s.innerHTML = msg;
}

function print(text, cls = '') {
  const o = output();
  const span = document.createElement('span');
  if (cls) span.className = cls;
  span.textContent = text;
  o.appendChild(span);
  o.scrollTop = o.scrollHeight;
}

/** Render a matplotlib figure captured as base64 PNG. */
function printImage(b64) {
  const o = output();
  const img = document.createElement('img');
  img.className = 'ide-figure';
  img.src = 'data:image/png;base64,' + b64;
  img.alt = 'matplotlib figure';
  o.appendChild(img);
  o.scrollTop = o.scrollHeight;
}

/** Repaint line numbers and the syntax-highlighted layer under the caret. */
function syncEditor() {
  const ed = editor();
  const n = ed.value.split('\n').length;

  const g = el('ide-gutter');
  if (g.childElementCount !== n) {
    g.innerHTML = Array.from({ length: n }, (_, i) => `<div>${i + 1}</div>`).join('');
  }

  const hl = el('ide-highlight');
  if (hl) {
    // A trailing newline collapses in <pre>; pad so the last line still scrolls.
    hl.firstElementChild.innerHTML = highlight(ed.value + '\n', mode === 'python' ? 'python' : 'js');
  }
  syncScroll();
}

function syncScroll() {
  const ed = editor();
  const hl = el('ide-highlight');
  if (hl) { hl.scrollTop = ed.scrollTop; hl.scrollLeft = ed.scrollLeft; }
  el('ide-gutter').scrollTop = ed.scrollTop;
}

// kept for call sites that only care about the gutter
const syncGutter = syncEditor;

/* ---------------- python ---------------- */

async function bootPython() {
  if (pyodide) return pyodide;
  if (booting) return booting;

  booting = (async () => {
    setStatus('<span class="spinner"></span> loading CPython 3.12 (WebAssembly)…');
    if (typeof loadPyodide === 'undefined') {
      await new Promise((res, rej) => {
        const s = document.createElement('script');
        s.src = 'vendor/pyodide/pyodide.js';
        s.onload = res; s.onerror = () => rej(new Error('could not load vendor/pyodide/pyodide.js'));
        document.head.appendChild(s);
      });
    }
    const py = await loadPyodide({
      indexURL: 'vendor/pyodide/',
      stdout: (t) => print(t + '\n'),
      stderr: (t) => print(t + '\n', 'err'),
    });
    setStatus('<span class="spinner"></span> loading NumPy…');
    await py.loadPackage('numpy');
    await py.runPythonAsync(PY_PRELUDE);
    pyodide = py;
    setStatusReady();
    return py;
  })();

  try { return await booting; }
  catch (e) { booting = null; setStatus('Python failed to start: ' + e.message, 'err'); throw e; }
}

async function runPython(code) {
  const py = await bootPython();

  // Pull in any vendored package this snippet needs, on first use only.
  const need = neededPackages(code);
  if (need.length) {
    setStatus(`<span class="spinner"></span> loading ${need.join(', ')}…`);
    try {
      await py.loadPackage(need);
      for (const p of need) loaded.add(p);
      if (need.includes('matplotlib')) {
        await py.runPythonAsync(PY_MPL_SETUP);
      }
    } catch (e) {
      print(`could not load ${need.join(', ')}: ${e.message}\n`, 'err');
    }
    setStatusReady();
  }

  // Keep figure colours in step with the app theme.
  if (loaded.has('matplotlib')) {
    try {
      await py.runPythonAsync(pyMplTheme(document.documentElement.dataset.theme !== 'light'));
    } catch {}
  }

  const t0 = performance.now();
  try {
    const res = await py.runPythonAsync(code);
    if (res !== undefined && res !== null) print(String(res) + '\n');

    if (loaded.has('matplotlib')) {
      const figs = await py.runPythonAsync('_mlatlas_figures()');
      const arr = figs && figs.toJs ? figs.toJs() : figs;
      if (arr && arr.length) {
        for (const b64 of arr) printImage(b64);
        print(`\n[${arr.length} figure${arr.length > 1 ? 's' : ''}]\n`, 'ok');
      }
      if (figs && figs.destroy) figs.destroy();
    }

    const dt = (performance.now() - t0).toFixed(0);
    print(`\n[ok · ${dt} ms]\n`, 'ok');
  } catch (e) {
    print(String(e.message || e) + '\n', 'err');
  }
}

function setStatusReady() {
  const extra = [...loaded].filter((p) => p !== 'numpy');
  setStatus(
    `Python 3.12 · NumPy${extra.length ? ' · ' + extra.join(' · ') : ''} · helpers: <code>aplot()</code>, <code>describe()</code>`,
    'ok');
}

/* ---------------- javascript ---------------- */

function runJS(code) {
  const t0 = performance.now();
  const log = (...a) => print(a.map(fmtJS).join(' ') + '\n');
  const sandboxConsole = { log, info: log, warn: log, debug: log, error: (...a) => print(a.map(fmtJS).join(' ') + '\n', 'err') };
  try {
    const fn = new Function('console', 'LA', 'Math', '"use strict";' + code);
    const r = fn(sandboxConsole, LA, Math);
    if (r !== undefined) print(fmtJS(r) + '\n');
    print(`\n[ok · ${(performance.now() - t0).toFixed(1)} ms]\n`, 'ok');
  } catch (e) {
    print((e.stack || String(e)).split('\n').slice(0, 4).join('\n') + '\n', 'err');
  }
}

function fmtJS(v) {
  if (typeof v === 'string') return v;
  if (Array.isArray(v)) {
    if (Array.isArray(v[0])) return '[\n  ' + v.map((r) => JSON.stringify(r.map(round6))).join('\n  ') + '\n]';
    return JSON.stringify(v.map(round6));
  }
  if (typeof v === 'object' && v !== null) { try { return JSON.stringify(v, null, 1); } catch { return String(v); } }
  return String(v);
}
const round6 = (x) => (typeof x === 'number' ? Math.round(x * 1e6) / 1e6 : x);

/* ---------------- public API ---------------- */

export function openInLab(code, lang = 'python') {
  showIDE(true);
  setMode(lang === 'js' || lang === 'javascript' ? 'js' : 'python');
  editor().value = code;
  syncGutter();
  editor().focus();
}

export function setLessonSnippets(snips) {
  lastSnippets = snips || [];
  const host = el('ide-snips');
  if (!host) return;
  host.innerHTML = '';
  if (!lastSnippets.length) { host.hidden = true; return; }
  host.hidden = false;
  const lbl = document.createElement('span');
  lbl.style.cssText = 'font-size:10.5px;color:var(--text-faint);align-self:center;margin-right:2px';
  lbl.textContent = 'this lesson:';
  host.appendChild(lbl);
  lastSnippets.forEach((s, i) => {
    const b = document.createElement('button');
    b.className = 'snip-btn';
    b.textContent = s.title.length > 30 ? s.title.slice(0, 28) + '…' : s.title;
    b.onclick = () => { editor().value = s.code; syncGutter(); };
    host.appendChild(b);
  });
}

function setMode(m) {
  mode = m;
  for (const t of document.querySelectorAll('.ide-tab')) t.classList.toggle('active', t.dataset.mode === m);
  syncEditor();   // relex the buffer against the new language
  if (m === 'js') setStatus('JavaScript sandbox — <code>LA</code> (matmul, solve, eig2, softmax) is in scope.');
  else if (pyodide) setStatusReady();
  else setStatus('Python sandbox not started — press ▶ Run to boot NumPy (~10 s, first time only).');
}

function showIDE(on) {
  const app = el('app'), ide = el('ide');
  const want = on === undefined ? ide.hidden : on;
  ide.hidden = !want;
  el('ide-resizer').hidden = !want;
  app.classList.toggle('ide-open', want);
  try { localStorage.setItem('mlatlas.ideOpen', JSON.stringify(want)); } catch {}
  if (want && !editor().value) {
    editor().value = mode === 'python' ? DEFAULT_PY : DEFAULT_JS;
    syncGutter();
  }
}

async function run() {
  const btn = el('ide-run');
  const code = editor().value;
  output().textContent = '';
  btn.disabled = true;
  try {
    if (mode === 'python') await runPython(code);
    else runJS(code);
  } finally { btn.disabled = false; }
}

export function initIDE() {
  const ed = editor();
  ed.value = DEFAULT_PY;
  syncGutter();

  ed.addEventListener('input', syncEditor);
  ed.addEventListener('scroll', syncScroll);
  ed.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const s = ed.selectionStart, t = ed.selectionEnd;
      if (s !== t && ed.value.slice(s, t).includes('\n')) {
        // indent/dedent block
        const a = ed.value.lastIndexOf('\n', s - 1) + 1;
        const block = ed.value.slice(a, t);
        const nb = e.shiftKey
          ? block.replace(/^ {1,4}/gm, '')
          : block.replace(/^/gm, '    ');
        ed.setRangeText(nb, a, t, 'select');
      } else {
        ed.setRangeText('    ', s, t, 'end');
      }
      syncGutter();
    }
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); run(); }
    if (e.key === 'Enter' && !e.metaKey && !e.ctrlKey) {
      // keep indentation, and add one level after ':' (python) or '{' (js)
      const s = ed.selectionStart;
      const lineStart = ed.value.lastIndexOf('\n', s - 1) + 1;
      const line = ed.value.slice(lineStart, s);
      const indent = (line.match(/^\s*/) || [''])[0];
      const extra = /[:{[(]\s*$/.test(line) ? '    ' : '';
      if (indent || extra) {
        e.preventDefault();
        ed.setRangeText('\n' + indent + extra, s, ed.selectionEnd, 'end');
        syncGutter();
      }
    }
  });

  el('ide-run').onclick = run;
  el('ide-clear').onclick = () => { output().textContent = ''; };
  el('ide-close').onclick = () => showIDE(false);
  el('ide-toggle').onclick = () => showIDE();
  el('ide-reset').onclick = () => {
    ed.value = lastSnippets.length ? lastSnippets[0].code : (mode === 'python' ? DEFAULT_PY : DEFAULT_JS);
    syncGutter();
  };
  for (const t of document.querySelectorAll('.ide-tab')) {
    t.onclick = () => {
      const prev = mode;
      setMode(t.dataset.mode);
      const isDefault = ed.value.trim() === (prev === 'python' ? DEFAULT_PY : DEFAULT_JS).trim();
      if (isDefault) { ed.value = mode === 'python' ? DEFAULT_PY : DEFAULT_JS; syncGutter(); }
    };
  }

  // drag to resize
  const rez = el('ide-resizer');
  rez.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = el('ide').offsetWidth;
    const move = (ev) => {
      const w = Math.max(320, Math.min(window.innerWidth - 420, startW - (ev.clientX - startX)));
      document.documentElement.style.setProperty('--ide-w', w + 'px');
    };
    const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  });

  let open = false;
  try { open = JSON.parse(localStorage.getItem('mlatlas.ideOpen') || 'false'); } catch {}
  if (open) showIDE(true);
}

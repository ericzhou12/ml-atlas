/* ============================================================
   main.js — app shell: routing, sidebar, search, progress, render.
   ============================================================ */

import { TRACKS, LESSONS, byId, trackOf, GLOSSARY, PATHS, allRefs, CHALLENGES } from './content/index.js';
import { VIZ } from './viz/index.js';
import { md, highlight, escapeHTML } from './md.js';
import { disposeAll, panel, refreshTheme } from './ui.js';
import { initIDE, openInLab, setLessonSnippets } from './ide.js';

/* ---------------- persistent state ---------------- */

const store = {
  get(k, d) { try { const v = localStorage.getItem('mlatlas.' + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('mlatlas.' + k, JSON.stringify(v)); } catch {} },
};

let done = new Set(store.get('done', []));
const saveDone = () => store.set('done', [...done]);

/* ---------------- theme ---------------- */

function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  store.set('theme', t);
  refreshTheme();
}
applyTheme(store.get('theme', 'dark'));

document.getElementById('theme-toggle').onclick = () =>
  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');

/* ---------------- sidebar ---------------- */

const tocEl = document.getElementById('toc');

function buildTOC() {
  tocEl.innerHTML = '';
  const open = new Set(store.get('openTracks', [TRACKS[0].id]));
  for (const tr of TRACKS) {
    const d = document.createElement('div');
    d.className = 'track' + (open.has(tr.id) ? ' open' : '');
    d.dataset.track = tr.id;
    const nDone = tr.lessons.filter((l) => done.has(l.id)).length;
    d.innerHTML = `
      <div class="track-head">
        <span class="track-chev">▶</span>
        <span class="track-dot" style="background:${tr.color}"></span>
        <span class="track-name">${tr.name}</span>
        <span class="track-count">${nDone}/${tr.lessons.length}</span>
      </div>
      <div class="track-body"></div>`;
    const body = d.querySelector('.track-body');
    tr.lessons.forEach((l, i) => {
      const a = document.createElement('a');
      a.className = 'toc-item' + (done.has(l.id) ? ' done' : '');
      a.href = '#/l/' + l.id;
      a.dataset.id = l.id;
      a.innerHTML = `<span class="toc-num">${tr.num}.${i + 1}</span><span class="toc-label">${l.title}</span>`;
      body.appendChild(a);
    });
    d.querySelector('.track-head').onclick = () => {
      d.classList.toggle('open');
      const o = new Set(store.get('openTracks', []));
      d.classList.contains('open') ? o.add(tr.id) : o.delete(tr.id);
      store.set('openTracks', [...o]);
    };
    tocEl.appendChild(d);
  }
  markActive();
}

function markActive() {
  const id = currentLessonId();
  for (const a of tocEl.querySelectorAll('.toc-item')) {
    const on = a.dataset.id === id;
    a.classList.toggle('active', on);
    if (on) {
      const tr = a.closest('.track');
      if (!tr.classList.contains('open')) tr.classList.add('open');
      requestAnimationFrame(() => a.scrollIntoView({ block: 'nearest' }));
    }
  }
}

function refreshProgress() {
  const total = LESSONS.length;
  const n = LESSONS.filter((l) => done.has(l.id)).length;
  document.getElementById('progress-text').textContent = `${n} / ${total}`;
  const ring = document.getElementById('ring-fg');
  const C = 2 * Math.PI * 15;
  ring.style.strokeDasharray = C;
  ring.style.strokeDashoffset = C * (1 - n / total);
  for (const tr of TRACKS) {
    const el = tocEl.querySelector(`.track[data-track="${tr.id}"] .track-count`);
    if (el) el.textContent = `${tr.lessons.filter((l) => done.has(l.id)).length}/${tr.lessons.length}`;
  }
}

document.getElementById('nav-toggle').onclick = () =>
  document.getElementById('app').classList.toggle('nav-hidden');

document.getElementById('expand-all').onclick = () => {
  const anyClosed = [...tocEl.querySelectorAll('.track')].some((t) => !t.classList.contains('open'));
  tocEl.querySelectorAll('.track').forEach((t) => t.classList.toggle('open', anyClosed));
  store.set('openTracks', anyClosed ? TRACKS.map((t) => t.id) : []);
};

/* ---------------- search ---------------- */

const searchEl = document.getElementById('search');
const srEl = document.getElementById('search-results');

const INDEX = LESSONS.map((l) => {
  const parts = [l.title, l.sub || '', (l.tags || []).join(' ')];
  for (const s of l.sections || []) {
    if (s.md) parts.push(s.md);
    if (s.q) parts.push(s.q);
    if (s.title) parts.push(s.title);
  }
  for (const r of l.refs || []) parts.push(r.title + ' ' + (r.author || ''));
  return { l, text: parts.join(' \n ').toLowerCase() };
});

function search(q) {
  q = q.trim().toLowerCase();
  if (!q) return [];
  const terms = q.split(/\s+/);
  const out = [];
  for (const { l, text } of INDEX) {
    let score = 0, ok = true;
    for (const t of terms) {
      const inTitle = l.title.toLowerCase().includes(t);
      const inSub = (l.sub || '').toLowerCase().includes(t);
      const inBody = text.includes(t);
      if (!inBody) { ok = false; break; }
      score += inTitle ? 100 : inSub ? 30 : 1;
      if (l.title.toLowerCase().startsWith(t)) score += 60;
      // frequency bonus, capped
      score += Math.min(10, (text.split(t).length - 1) * 0.6);
    }
    if (ok) out.push({ l, score });
  }
  // glossary hits
  const gl = GLOSSARY.filter((g) => g.term.toLowerCase().includes(q)).slice(0, 3)
    .map((g) => ({ gloss: g, score: 80 }));
  return [...out, ...gl].sort((a, b) => b.score - a.score).slice(0, 12);
}

function hl(text, q) {
  const terms = q.trim().split(/\s+/).filter(Boolean).map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  if (!terms.length) return escapeHTML(text);
  return escapeHTML(text).replace(new RegExp(`(${terms.join('|')})`, 'gi'), '<mark>$1</mark>');
}

let selIdx = 0;
function renderSearch() {
  const q = searchEl.value;
  const res = search(q);
  if (!q.trim()) { srEl.hidden = true; return; }
  srEl.hidden = false;
  selIdx = 0;
  if (!res.length) { srEl.innerHTML = `<div class="sr-empty">No matches for “${escapeHTML(q)}”.</div>`; return; }
  srEl.innerHTML = res.map((r, i) => {
    if (r.gloss) {
      return `<a class="sr-item${i === 0 ? ' sel' : ''}" href="#/glossary">
        <div class="sr-title">${hl(r.gloss.term, q)}</div>
        <div class="sr-meta">glossary · ${escapeHTML(r.gloss.def.slice(0, 90))}…</div></a>`;
    }
    const tr = trackOf(r.l.id);
    return `<a class="sr-item${i === 0 ? ' sel' : ''}" href="#/l/${r.l.id}">
      <div class="sr-title">${hl(r.l.title, q)}</div>
      <div class="sr-meta"><span style="color:${tr.color}">●</span> ${tr.name} · ${r.l.mins || 15} min</div></a>`;
  }).join('');
}

searchEl.addEventListener('input', renderSearch);
searchEl.addEventListener('focus', () => { if (searchEl.value) renderSearch(); });
searchEl.addEventListener('keydown', (e) => {
  const items = [...srEl.querySelectorAll('.sr-item')];
  if (e.key === 'Escape') { searchEl.blur(); srEl.hidden = true; searchEl.value = ''; }
  if (!items.length) return;
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    items[selIdx]?.classList.remove('sel');
    selIdx = (selIdx + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length;
    items[selIdx].classList.add('sel');
    items[selIdx].scrollIntoView({ block: 'nearest' });
  }
  if (e.key === 'Enter') { e.preventDefault(); items[selIdx]?.click(); }
});
document.addEventListener('click', (e) => {
  if (!e.target.closest('.search-wrap')) srEl.hidden = true;
});

/* ---------------- routing ---------------- */

function currentLessonId() {
  const m = /^#\/l\/([\w.-]+)/.exec(location.hash);
  return m ? m[1] : null;
}

const contentEl = document.getElementById('content');
const mainEl = document.getElementById('main');

function route() {
  disposeAll();
  srEl.hidden = true;
  const h = location.hash || '#/';
  contentEl.innerHTML = '';

  if (h.startsWith('#/l/')) {
    const l = byId(currentLessonId());
    if (l) renderLesson(l);
    else renderNotFound();
  } else if (h.startsWith('#/track/')) {
    const tr = TRACKS.find((t) => t.id === h.slice(8));
    tr ? renderTrack(tr) : renderNotFound();
  } else if (h.startsWith('#/library')) renderLibrary();
  else if (h.startsWith('#/glossary')) renderGlossary();
  else if (h.startsWith('#/paths')) renderPaths();
  else if (h.startsWith('#/about')) renderAbout();
  else renderHome();

  markActive();
  mainEl.scrollTop = 0;
  const anchor = h.split('#')[2];
  if (anchor) {
    const t = document.getElementById('s-' + anchor);
    if (t) setTimeout(() => t.scrollIntoView({ block: 'start' }), 30);
  }
}

window.addEventListener('hashchange', route);

/* ---------------- renderers ---------------- */

function renderNotFound() {
  contentEl.innerHTML = `<div class="center-msg"><h2>Not found</h2><p>That page doesn't exist. <a href="#/">Back to the atlas</a></p></div>`;
}

function renderHome() {
  const nRef = allRefs().length;
  const nViz = LESSONS.reduce((s, l) => s + (l.sections || []).filter((x) => x.t === 'viz').length, 0);
  const nDone = LESSONS.filter((l) => done.has(l.id)).length;
  const last = store.get('last', null);
  const lastL = last ? byId(last) : null;

  contentEl.innerHTML = `
    <div class="hero">
      <h1>Machine learning, from first gradients&nbsp;to the frontier.</h1>
      <p class="lede">A self-contained atlas: the math derived rather than asserted, every core idea attached to
      something you can <em>move with your hands</em>, runnable code beside the theory, and a path from
      “what is a derivative” to “how does a reasoning VLM actually work.”</p>
      <div class="hero-stats">
        <div class="stat"><div class="stat-n">${LESSONS.length}</div><div class="stat-l">lessons</div></div>
        <div class="stat"><div class="stat-n">${TRACKS.length}</div><div class="stat-l">tracks</div></div>
        <div class="stat"><div class="stat-n">${nViz}</div><div class="stat-l">interactive figures</div></div>
        <div class="stat"><div class="stat-n">${nRef}</div><div class="stat-l">sources</div></div>
        <div class="stat"><div class="stat-n" style="color:var(--ok)">${nDone}</div><div class="stat-l">completed</div></div>
      </div>
      <div class="btn-row" style="margin-top:22px">
        ${lastL ? `<a class="btn primary" href="#/l/${lastL.id}">Resume: ${escapeHTML(lastL.title)}</a>` : ''}
        <a class="btn ${lastL ? '' : 'primary'}" href="#/l/${LESSONS[0].id}">Start from the beginning</a>
        <a class="btn" href="#/paths">Pick a learning path</a>
        <a class="btn" href="#/about">How to use this</a>
      </div>
    </div>
    <h2>Tracks</h2>
    <div class="track-grid">
      ${TRACKS.map((t) => {
        const d = t.lessons.filter((l) => done.has(l.id)).length;
        return `<a class="track-card" href="#/track/${t.id}">
          <div class="tc-top"><span class="track-dot" style="background:${t.color}"></span>
            <span class="tc-name">${t.num}. ${t.name}</span></div>
          <div class="tc-desc">${t.desc}</div>
          <div class="tc-bar"><div class="tc-fill" style="width:${(d / t.lessons.length * 100).toFixed(0)}%;background:${t.color}"></div></div>
          <div class="tc-meta"><span>${t.lessons.length} lessons</span><span>${d} done</span></div>
        </a>`;
      }).join('')}
    </div>
    ${md(`## How this is built

Every lesson follows the same rhythm, because that is the order in which things actually become clear:

1. **Intuition first** — a picture or an analogy you can hold onto, before any symbols.
2. **The math, derived** — not a wall of notation but the steps, with the algebra you'd skip on a first pass folded into collapsible *derivation* blocks.
3. **Something to play with** — sliders, draggable data, live training. If a concept has a knob, the knob is here.
4. **Code you can run** — NumPy in the browser, in the panel on the right (press <span class="kbd">⌘J</span>).
5. **Where it came from** — the actual papers, books, and lectures, linked.

Nothing here calls out to a server. Progress is stored in your browser only.`)}
  `;
}

function renderTrack(tr) {
  contentEl.innerHTML = `
    <div class="crumbs"><a href="#/">Atlas</a> <span>›</span> <span>${tr.name}</span></div>
    <h1 class="lesson-title">${tr.num}. ${tr.name}</h1>
    <p class="lesson-sub">${tr.desc}</p>
    ${tr.intro ? md(tr.intro) : ''}
    <div class="lesson-list">
      ${tr.lessons.map((l, i) => `
        <a class="ll-item${done.has(l.id) ? ' done' : ''}" href="#/l/${l.id}">
          <span class="ll-n">${tr.num}.${i + 1}</span>
          <span class="ll-body">
            <span class="ll-title">${l.title}</span>
            <div class="ll-sum">${l.sub || ''}</div>
          </span>
          <span class="ll-time">${l.mins || 15}m</span>
        </a>`).join('')}
    </div>`;
}

function renderLesson(l) {
  store.set('last', l.id);
  const tr = trackOf(l.id);
  const idx = tr.lessons.indexOf(l);
  const flat = LESSONS;
  const fi = flat.indexOf(l);
  const prev = flat[fi - 1], next = flat[fi + 1];

  const head = document.createElement('div');
  head.className = 'lesson-head';
  head.innerHTML = `
    <div class="crumbs">
      <a href="#/">Atlas</a> <span>›</span>
      <a href="#/track/${tr.id}">${tr.name}</a> <span>›</span>
      <span>${tr.num}.${idx + 1}</span>
    </div>
    <h1 class="lesson-title">${l.title}</h1>
    ${l.sub ? `<p class="lesson-sub">${md(l.sub).replace(/^<p>|<\/p>$/g, '')}</p>` : ''}
    <div class="lesson-meta">
      <span class="chip" style="border-color:${tr.color};color:${tr.color}">${tr.name}</span>
      <span class="chip">${l.mins || 15} min</span>
      ${l.level ? `<span class="chip">${l.level}</span>` : ''}
      ${(l.tags || []).slice(0, 4).map((t) => `<span class="chip">${t}</span>`).join('')}
    </div>
    ${l.prereq && l.prereq.length ? `<div class="prereqs">Assumes: ${l.prereq.map((p) => {
      const pl = byId(p);
      return pl ? `<a href="#/l/${p}">${pl.title}</a>` : p;
    }).join(' · ')}</div>` : ''}`;
  contentEl.appendChild(head);

  for (const s of l.sections || []) renderSection(s, contentEl, l);

  const ch = CHALLENGES[l.id];
  if (ch) {
    const h = document.createElement('h2');
    h.textContent = 'Try it yourself';
    h.id = 's-try-it-yourself';
    contentEl.appendChild(h);
    contentEl.appendChild(challengeEl(ch, l));
  }

  if (l.refs && l.refs.length) {
    const h = document.createElement('h2');
    h.textContent = 'Sources & further reading';
    h.id = 's-sources';
    contentEl.appendChild(h);
    contentEl.appendChild(refsEl(l.refs));
  }

  // footer
  const foot = document.createElement('div');
  foot.className = 'lesson-foot';
  const isDone = done.has(l.id);
  foot.innerHTML = `
    <div class="complete-row">
      <button class="btn ${isDone ? '' : 'primary'}" id="mark-done">${isDone ? '✓ Completed — unmark' : 'Mark as complete'}</button>
      <span style="font-size:12.5px;color:var(--text-faint)">Progress is saved in this browser.</span>
    </div>
    <div class="pager">
      ${prev ? `<a href="#/l/${prev.id}"><div class="dir">← Previous</div><div class="ttl">${escapeHTML(prev.title)}</div></a>` : '<span></span>'}
      ${next ? `<a class="next" href="#/l/${next.id}"><div class="dir">Next →</div><div class="ttl">${escapeHTML(next.title)}</div></a>` : '<span></span>'}
    </div>`;
  contentEl.appendChild(foot);
  foot.querySelector('#mark-done').onclick = (e) => {
    done.has(l.id) ? done.delete(l.id) : done.add(l.id);
    saveDone(); refreshProgress(); buildTOC();
    const nowDone = done.has(l.id);
    e.target.textContent = nowDone ? '✓ Completed — unmark' : 'Mark as complete';
    e.target.classList.toggle('primary', !nowDone);
  };

  // lesson code snippets → IDE quick-load buttons
  setLessonSnippets((l.sections || [])
    .filter((s) => s.t === 'code' && (s.lang === 'python' || !s.lang) && s.lab !== false)
    .map((s) => ({ title: s.title || 'snippet', code: s.code })));
}

function renderSection(s, host, lesson) {
  if (s.t === 'md') {
    const d = document.createElement('div');
    d.innerHTML = md(s.md);
    host.appendChild(d);

  } else if (s.t === 'note') {
    const d = document.createElement('div');
    d.className = 'callout ' + (s.kind || 'key');
    const labels = { intuition: '◈ Intuition', warning: '⚠ Watch out', key: '★ Key idea', history: '⏱ History', math: '∑ Math note' };
    d.innerHTML = `<div class="callout-label">${s.title || labels[s.kind] || '★ Key idea'}</div>${md(s.md)}`;
    host.appendChild(d);

  } else if (s.t === 'deriv') {
    const d = document.createElement('details');
    d.className = 'deriv';
    d.innerHTML = `<summary>${s.title || 'Derivation'}</summary><div class="deriv-body">${md(s.md)}</div>`;
    host.appendChild(d);

  } else if (s.t === 'code') {
    host.appendChild(codeEl(s));

  } else if (s.t === 'viz') {
    const f = VIZ[s.id];
    const holder = document.createElement('div');
    host.appendChild(holder);
    if (!f) {
      holder.innerHTML = `<div class="callout warning"><div class="callout-label">Missing figure</div><p>No visualization registered for <code>${s.id}</code>.</p></div>`;
    } else {
      try { f(holder, s.params || {}, { panel, lesson }); }
      catch (e) {
        console.error('viz failed:', s.id, e);
        holder.innerHTML = `<div class="callout warning"><div class="callout-label">Figure error</div><p><code>${s.id}</code> — ${escapeHTML(e.message)}</p></div>`;
      }
    }

  } else if (s.t === 'quiz') {
    host.appendChild(quizEl(s, lesson));

  } else if (s.t === 'refs') {
    host.appendChild(refsEl(s.items));
  }
}

function codeEl(s) {
  const d = document.createElement('div');
  d.className = 'codeblock';
  const lang = s.lang || 'python';
  d.innerHTML = `
    <div class="codeblock-head">
      <span class="codeblock-title">${s.title || lang}</span>
      <span class="codeblock-actions">
        ${lang === 'python' || lang === 'js' ? '<button class="mini-btn lab">▶ open in lab</button>' : ''}
        <button class="mini-btn copy">copy</button>
      </span>
    </div>
    <pre><code>${highlight(s.code, lang)}</code></pre>
    ${s.explain ? `<div class="code-explain">${md(s.explain)}</div>` : ''}`;
  const copy = d.querySelector('.copy');
  copy.onclick = () => {
    navigator.clipboard.writeText(s.code);
    copy.textContent = 'copied ✓';
    setTimeout(() => (copy.textContent = 'copy'), 1400);
  };
  const lab = d.querySelector('.lab');
  if (lab) lab.onclick = () => openInLab(s.code, lang);
  return d;
}

function quizEl(s, lesson) {
  const d = document.createElement('div');
  d.className = 'quiz';
  const key = `quiz.${lesson ? lesson.id : 'x'}.${(s.q || '').slice(0, 24)}`;
  d.innerHTML = `<div class="quiz-q">${md(s.q).replace(/^<p>|<\/p>$/g, '')}</div>`;
  const opts = [];
  s.options.forEach((o, i) => {
    const b = document.createElement('div');
    b.className = 'quiz-opt';
    b.innerHTML = `<span class="mark">${'ABCD'[i]}</span><span>${md(o).replace(/^<p>|<\/p>$/g, '')}</span>`;
    b.onclick = () => reveal(i);
    d.appendChild(b);
    opts.push(b);
  });
  const ex = document.createElement('div');
  ex.className = 'quiz-explain';
  ex.hidden = true;
  ex.innerHTML = md(s.explain || '');
  d.appendChild(ex);

  function reveal(pick) {
    opts.forEach((b, i) => {
      b.classList.toggle('correct', i === s.answer);
      b.classList.toggle('wrong', i === pick && pick !== s.answer);
    });
    ex.hidden = false;
    store.set(key, pick === s.answer);
  }
  return d;
}

/** A small programming exercise: prompt, starter code, hint, solution. */
function challengeEl(c, lesson) {
  const d = document.createElement('div');
  d.className = 'challenge';
  d.innerHTML = `
    <div class="challenge-head">
      <div>
        <div class="challenge-label">⌨ Challenge</div>
        <div class="challenge-title">${escapeHTML(c.title)}</div>
      </div>
    </div>
    <div class="challenge-body">${md(c.prompt)}</div>
    <div class="challenge-actions">
      <button class="btn primary open-lab">▶ Open in lab</button>
      <button class="btn copy-starter">copy starter</button>
    </div>
    ${c.hint ? `<details><summary>Hint</summary><div class="deriv-body">${md(c.hint)}</div></details>` : ''}
    <details><summary>Solution</summary></details>`;

  d.querySelector('.open-lab').onclick = () => openInLab(c.starter, 'python');
  const cp = d.querySelector('.copy-starter');
  cp.onclick = () => {
    navigator.clipboard.writeText(c.starter);
    cp.textContent = 'copied ✓';
    setTimeout(() => (cp.textContent = 'copy starter'), 1400);
  };

  // Build the solution lazily so a page full of challenges stays cheap.
  const sol = d.querySelectorAll('details')[c.hint ? 1 : 0];
  let built = false;
  sol.addEventListener('toggle', () => {
    if (!sol.open || built) return;
    built = true;
    sol.appendChild(codeEl({ code: c.solution, lang: 'python', title: 'solution', explain: c.explain }));
  });
  return d;
}

const KIND_LABEL = { paper: 'paper', book: 'book', course: 'course', blog: 'blog', video: 'video', code: 'code', interactive: 'demo' };

function refsEl(items) {
  const d = document.createElement('div');
  d.className = 'refs';
  d.innerHTML = items.map((r) => `
    <div class="ref">
      <div class="ref-kind k-${r.kind || 'paper'}">${KIND_LABEL[r.kind] || r.kind || 'paper'}</div>
      <div class="ref-main">
        <div class="ref-title">${r.url ? `<a href="${r.url}" target="_blank" rel="noopener noreferrer">${escapeHTML(r.title)}</a>` : escapeHTML(r.title)}</div>
        <div class="ref-meta">${[r.author, r.year, r.venue].filter(Boolean).map(escapeHTML).join(' · ')}</div>
        ${r.note ? `<div class="ref-note">${md(r.note).replace(/^<p>|<\/p>$/g, '')}</div>` : ''}
      </div>
    </div>`).join('');
  return d;
}

/* ---------------- library / glossary / paths / about ---------------- */

function renderLibrary() {
  const refs = allRefs();
  const kinds = [...new Set(refs.map((r) => r.kind || 'paper'))];
  contentEl.innerHTML = `
    <div class="crumbs"><a href="#/">Atlas</a> <span>›</span> <span>Reference library</span></div>
    <h1 class="lesson-title">Reference library</h1>
    <p class="lesson-sub">Every source cited across the ${LESSONS.length} lessons — ${refs.length} entries, deduplicated,
    grouped by type. This is the reading list.</p>
    <div class="filter-row" id="lib-filters">
      <span class="filter-chip on" data-k="all">all (${refs.length})</span>
      ${kinds.map((k) => `<span class="filter-chip" data-k="${k}">${KIND_LABEL[k] || k} (${refs.filter((r) => (r.kind || 'paper') === k).length})</span>`).join('')}
    </div>
    <input id="lib-search" type="search" placeholder="filter by title or author…" style="width:100%;height:32px;padding:0 11px;background:var(--bg-inset);border:1px solid var(--border);border-radius:8px;color:var(--text);margin-bottom:16px;font-family:var(--sans);font-size:13px;outline:none">
    <div id="lib-list"></div>`;

  const list = document.getElementById('lib-list');
  let kind = 'all', q = '';
  const draw = () => {
    const sel = refs.filter((r) => (kind === 'all' || (r.kind || 'paper') === kind) &&
      (!q || (r.title + ' ' + (r.author || '')).toLowerCase().includes(q)));
    sel.sort((a, b) => (b.year || 0) - (a.year || 0) || a.title.localeCompare(b.title));
    list.innerHTML = '';
    if (!sel.length) { list.innerHTML = '<div class="center-msg">No matching sources.</div>'; return; }
    const el = refsEl(sel);
    // append "used in" links
    sel.forEach((r, i) => {
      const row = el.children[i];
      if (r.lessons && r.lessons.length) {
        const u = document.createElement('div');
        u.className = 'ref-note';
        u.innerHTML = 'in: ' + r.lessons.slice(0, 4).map((id) => {
          const L = byId(id);
          return L ? `<a href="#/l/${id}">${escapeHTML(L.title)}</a>` : '';
        }).filter(Boolean).join(', ');
        row.querySelector('.ref-main').appendChild(u);
      }
    });
    list.appendChild(el);
  };
  document.getElementById('lib-filters').onclick = (e) => {
    const c = e.target.closest('.filter-chip'); if (!c) return;
    document.querySelectorAll('#lib-filters .filter-chip').forEach((x) => x.classList.toggle('on', x === c));
    kind = c.dataset.k; draw();
  };
  document.getElementById('lib-search').oninput = (e) => { q = e.target.value.toLowerCase(); draw(); };
  draw();
}

function renderGlossary() {
  const letters = [...new Set(GLOSSARY.map((g) => g.term[0].toUpperCase()))].sort();
  contentEl.innerHTML = `
    <div class="crumbs"><a href="#/">Atlas</a> <span>›</span> <span>Glossary</span></div>
    <h1 class="lesson-title">Glossary</h1>
    <p class="lesson-sub">${GLOSSARY.length} terms and the notation that goes with them. When a symbol shows up
    without warning in a paper, look here first.</p>
    <input id="gl-search" type="search" placeholder="search terms…" style="width:100%;height:32px;padding:0 11px;background:var(--bg-inset);border:1px solid var(--border);border-radius:8px;color:var(--text);margin-bottom:8px;font-family:var(--sans);font-size:13px;outline:none">
    <div class="filter-row">${letters.map((L) => `<a class="filter-chip" href="#/glossary#${L}">${L}</a>`).join('')}</div>
    <div id="gl-list"></div>`;
  const list = document.getElementById('gl-list');
  const draw = (q = '') => {
    const sel = GLOSSARY.filter((g) => !q || (g.term + ' ' + g.def).toLowerCase().includes(q))
      .sort((a, b) => a.term.localeCompare(b.term));
    list.innerHTML = sel.map((g) => `
      <div class="gloss-item" id="${g.term[0].toUpperCase()}">
        <div class="gloss-term">${escapeHTML(g.term)}${g.sym ? `<span class="gloss-sym">${md(g.sym).replace(/<\/?p>/g, '')}</span>` : ''}</div>
        <div class="gloss-def">${md(g.def).replace(/^<p>|<\/p>$/g, '')}${g.see ? ` <a href="#/l/${g.see}">→ lesson</a>` : ''}</div>
      </div>`).join('') || '<div class="center-msg">No matches.</div>';
  };
  document.getElementById('gl-search').oninput = (e) => draw(e.target.value.toLowerCase());
  draw();
}

function renderPaths() {
  contentEl.innerHTML = `
    <div class="crumbs"><a href="#/">Atlas</a> <span>›</span> <span>Learning paths</span></div>
    <h1 class="lesson-title">Learning paths</h1>
    <p class="lesson-sub">The atlas is a graph, not a line. These are routes through it for different starting points and goals.</p>
    ${PATHS.map((p) => `
      <div class="path-card">
        <h3>${p.name}</h3>
        <p style="color:var(--text-dim);font-size:13.5px;margin-bottom:4px">${p.desc}</p>
        <div style="font-size:12px;color:var(--text-faint)">${p.hours} · ${p.lessons.length} lessons</div>
        <div class="path-steps">${p.lessons.map((id, i) => {
          const L = byId(id);
          return L ? `<a class="path-step" href="#/l/${id}">${i + 1}. ${escapeHTML(L.title)}</a>` : '';
        }).join('')}</div>
      </div>`).join('')}`;
}

function renderAbout() {
  contentEl.innerHTML = `<div class="crumbs"><a href="#/">Atlas</a> <span>›</span> <span>About</span></div>` + md(`
# About this atlas

## What it is

A single-page, fully offline reference-and-course for machine learning: ${LESSONS.length} lessons spanning
vectors and derivatives through transformers, diffusion, RL, vision-language models, and current frontier
research. No account, no server, no network calls. Everything — including the math renderer — is local.

## How to use it

**If you're starting from scratch**, go in order. Tracks 1–3 are the load-bearing ones; everything later
leans on them. Don't skip the derivation blocks on your second pass.

**If you're filling gaps**, use search (<span class="kbd">/</span>) or the [glossary](#/glossary). Each lesson
lists its prerequisites at the top and is written to be readable on its own.

**If you have a goal** — "understand LLMs", "prep for interviews", "read papers" — see [learning paths](#/paths).

## The three panes

- **Left**: the curriculum. Click a track to expand it. ✓ marks lessons you've completed.
- **Middle**: the lesson. Interactive figures are live — drag the points, move the sliders. That is where the
  intuition actually comes from; reading past them costs you most of the value.
- **Right**: the code lab (<span class="kbd">⌘J</span> / <span class="kbd">Ctrl+J</span>). Real CPython with NumPy,
  compiled to WebAssembly, running in your tab. First boot takes ~10 seconds; after that it's instant.
  Every Python block in a lesson has an **open in lab** button.

## Keyboard

| Key | Action |
|---|---|
| <span class="kbd">/</span> | focus search |
| <span class="kbd">←</span> <span class="kbd">→</span> | previous / next lesson |
| <span class="kbd">⌘B</span> | toggle sidebar |
| <span class="kbd">⌘J</span> | toggle code lab |
| <span class="kbd">⌘↵</span> | run code (in the lab) |
| <span class="kbd">Esc</span> | close search |

## Honest caveats

- The interactive demos are **toy-scale by design**. A 40-parameter network trained in your browser teaches the
  mechanism, not the phenomenology of scale. Where behavior only appears at scale, the text says so.
- Frontier material (track 10) moves fast. Treat it as a map of the terrain as of the writing, with the primary
  sources linked so you can check what's changed.
- Every visualization computes real math — no faked curves — but numbers use float64 in JS and small grids, so
  they're illustrative rather than benchmark-accurate.

## Reset

<button class="btn" onclick="if(confirm('Clear all progress?')){localStorage.clear();location.reload()}">Clear all progress</button>
`);
}

/* ---------------- keyboard ---------------- */

document.addEventListener('keydown', (e) => {
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
  if (e.key === '/' && !typing) { e.preventDefault(); searchEl.focus(); searchEl.select(); }
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'b') {
    e.preventDefault(); document.getElementById('app').classList.toggle('nav-hidden');
  }
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'j') {
    e.preventDefault(); document.getElementById('ide-toggle').click();
  }
  if (!typing && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
    const id = currentLessonId(); if (!id) return;
    const i = LESSONS.findIndex((l) => l.id === id);
    const t = LESSONS[i + (e.key === 'ArrowRight' ? 1 : -1)];
    if (t) location.hash = '#/l/' + t.id;
  }
});

/* ---------------- boot ---------------- */

buildTOC();
refreshProgress();
initIDE();
route();

// expose for console tinkering
window.MLAtlas = { TRACKS, LESSONS, byId, VIZ, done };

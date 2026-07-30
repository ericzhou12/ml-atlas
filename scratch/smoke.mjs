/* Headless integrity check for ML Atlas: content graph + markdown renderer. */
globalThis.katex = { renderToString: (t) => `<span class="katex">${t}</span>` };

const base = new URL('../js/', import.meta.url).href;
const { TRACKS, LESSONS, byId, GLOSSARY, PATHS, allRefs, CHALLENGES } = await import(base + 'content/index.js');
const { VIZ, VIZ_IDS } = await import(base + 'viz/index.js');
const { md } = await import(base + 'md.js');

const err = [];
const warn = [];

// 1. unique lesson ids
const ids = LESSONS.map((l) => l.id);
const dupes = ids.filter((id, i) => ids.indexOf(id) !== i);
if (dupes.length) err.push(`duplicate lesson ids: ${[...new Set(dupes)].join(', ')}`);

// 2. every viz referenced by a lesson exists in the registry
const used = new Set();
for (const l of LESSONS) {
  for (const s of l.sections || []) {
    if (s.t === 'viz') {
      used.add(s.id);
      if (!VIZ[s.id]) err.push(`${l.id}: missing viz "${s.id}"`);
    }
  }
}

// 3. prereqs point at real lessons
for (const l of LESSONS) {
  for (const p of l.prereq || []) {
    if (!byId(p)) err.push(`${l.id}: prereq "${p}" does not exist`);
  }
}

// 4. in-text #/l/ links resolve
const linkRe = /#\/l\/([\w.-]+)/g;
for (const l of LESSONS) {
  const blob = JSON.stringify(l);
  let m;
  while ((m = linkRe.exec(blob))) {
    if (!byId(m[1])) err.push(`${l.id}: dead link to "${m[1]}"`);
  }
}

// 5. glossary "see" targets resolve
for (const g of GLOSSARY) {
  if (g.see && !byId(g.see)) err.push(`glossary "${g.term}": see -> "${g.see}" missing`);
}

// 6. learning path lessons resolve
for (const p of PATHS) {
  for (const id of p.lessons) if (!byId(id)) err.push(`path "${p.name}": missing "${id}"`);
}

// 7. quizzes have a valid answer index and an explanation
for (const l of LESSONS) {
  for (const s of l.sections || []) {
    if (s.t === 'quiz') {
      if (!(s.answer >= 0 && s.answer < s.options.length)) err.push(`${l.id}: quiz answer index out of range`);
      if (!s.explain) warn.push(`${l.id}: quiz has no explanation`);
    }
  }
}

// 8. every lesson has refs, a summary, and at least one figure
for (const l of LESSONS) {
  if (!l.sub) warn.push(`${l.id}: no sub-heading`);
  if (!(l.refs || []).length) warn.push(`${l.id}: no references`);
  const nviz = (l.sections || []).filter((s) => s.t === 'viz').length;
  if (!nviz) warn.push(`${l.id}: no interactive figure`);
}

// 9. render every markdown block — catches TeX/parse crashes
let blocks = 0, chars = 0;
for (const l of LESSONS) {
  for (const s of l.sections || []) {
    for (const field of ['md', 'explain', 'q']) {
      if (typeof s[field] === 'string') {
        try {
          const out = md(s[field]);
          blocks++; chars += out.length;
          if (out.includes('math-err')) warn.push(`${l.id}: KaTeX failed in a ${field} block`);
        } catch (e) {
          err.push(`${l.id}: md() threw on ${field}: ${e.message}`);
        }
      }
    }
  }
}

// 9b. challenges: every lesson should have one, and each must be well-formed
if (!CHALLENGES) err.push('CHALLENGES is not exported from content/index.js');
else {
  for (const id of Object.keys(CHALLENGES)) {
    if (!byId(id)) err.push(`challenge "${id}" has no matching lesson`);
  }
  for (const l of LESSONS) {
    const c = CHALLENGES[l.id];
    if (!c) { warn.push(`${l.id}: no challenge`); continue; }
    for (const f of ['title', 'prompt', 'starter', 'solution']) {
      if (!c[f] || typeof c[f] !== 'string') err.push(`challenge ${l.id}: missing ${f}`);
    }
    if (c.starter === c.solution) warn.push(`${l.id}: starter and solution are identical`);
    if (c.starter && !/TODO|challenge|implement/i.test(c.starter)) {
      warn.push(`${l.id}: starter has no obvious TODO for the learner`);
    }
    for (const f of ['prompt', 'hint', 'explain']) {
      if (typeof c[f] === 'string') {
        try { md(c[f]); } catch (e) { err.push(`challenge ${l.id}: md() threw on ${f}`); }
      }
    }
  }
}

// 10. unused viz (built but never referenced)
const unused = VIZ_IDS.filter((id) => !used.has(id));

// ---- report ----
const nViz = LESSONS.reduce((a, l) => a + (l.sections || []).filter((s) => s.t === 'viz').length, 0);
const nCode = LESSONS.reduce((a, l) => a + (l.sections || []).filter((s) => s.t === 'code').length, 0);
const nQuiz = LESSONS.reduce((a, l) => a + (l.sections || []).filter((s) => s.t === 'quiz').length, 0);
const nDeriv = LESSONS.reduce((a, l) => a + (l.sections || []).filter((s) => s.t === 'deriv').length, 0);
const words = LESSONS.reduce((a, l) => a + JSON.stringify(l).split(/\s+/).length, 0);

console.log(`tracks              ${TRACKS.length}`);
console.log(`lessons             ${LESSONS.length}`);
console.log(`figures placed      ${nViz}   (registry has ${VIZ_IDS.length})`);
console.log(`code blocks         ${nCode}`);
console.log(`derivations         ${nDeriv}`);
console.log(`quizzes             ${nQuiz}`);
console.log(`unique references   ${allRefs().length}`);
console.log(`glossary terms      ${GLOSSARY.length}`);
console.log(`learning paths      ${PATHS.length}`);
console.log(`challenges          ${Object.keys(CHALLENGES || {}).length}`);
console.log(`markdown blocks OK  ${blocks}  (${(chars / 1000).toFixed(0)}k chars rendered)`);
console.log(`approx word count   ${(words / 1000).toFixed(0)}k`);

if (unused.length) console.log(`\nbuilt but not yet placed in a lesson (${unused.length}):\n  ${unused.join(', ')}`);
if (warn.length) console.log(`\nWARNINGS (${warn.length}):\n  ` + warn.join('\n  '));
console.log(err.length ? `\nERRORS (${err.length}):\n  ` + err.join('\n  ') : '\nNo errors.');
process.exit(err.length ? 1 : 0);

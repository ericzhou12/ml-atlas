/* audit.mjs — flag common content defects across tracks.
   usage: node scratch/audit.mjs [trackId ...] */
import { TRACKS } from '../js/content/index.js';
import { CHALLENGES } from '../js/content/challenges/index.js';

const only = process.argv.slice(2);
for (const tr of TRACKS) {
  if (only.length && !only.includes(tr.id)) continue;
  for (const l of tr.lessons) {
    const flags = [];
    const codes = l.sections.filter((s) => s.t === 'code');
    const noExplain = codes.filter((s) => !s.explain).length;
    if (noExplain) flags.push(`${noExplain} code block(s) with no explanation`);
    if (!l.sections.some((s) => s.t === 'deriv')) flags.push('no derivation');
    const ch = CHALLENGES[l.id];
    if (!ch) flags.push('NO CHALLENGE');
    else {
      if (!ch.explain) flags.push('challenge has no explain');
      if (!/TODO/.test(ch.starter || '')) flags.push('challenge starter has no TODO');
      if (!/assert|PASS/.test(ch.starter || '')) flags.push('challenge has no self-check');
    }
    const md = JSON.stringify(l.sections);
    if (/\\\\`/.test(md)) flags.push('over-escaped backtick');
    if (flags.length) console.log(`${tr.id.padEnd(10)} ${l.id.padEnd(26)} ${flags.join(' · ')}`);
  }
}

/* runsol.mjs — extract every challenge `solution` and run it under python3.
   usage: node scratch/runsol.mjs [lessonIdSubstring]
   Requires numpy (and matplotlib/scipy if a solution uses them). */
import { CHALLENGES } from '../js/content/challenges/index.js';
import { execFileSync } from 'node:child_process';
import { writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const filter = process.argv[2] || '';
const dir = mkdtempSync(join(tmpdir(), 'sol-'));
let pass = 0, fail = 0, skip = 0;

for (const [id, ch] of Object.entries(CHALLENGES)) {
  if (filter && !id.includes(filter)) continue;
  const src = ch.solution;
  if (!src) { skip++; continue; }
  if (/^\s*(\/\/|const |let |function )/m.test(src) && !/^import /m.test(src)) { skip++; continue; }
  const f = join(dir, `${id.replace(/[^\w]/g, '_')}.py`);
  writeFileSync(f, src);
  try {
    const out = execFileSync('python3', [f], { encoding: 'utf8', timeout: 120000, stdio: ['ignore', 'pipe', 'pipe'] });
    if (/\bFAIL\b/.test(out)) { console.log(`FAIL(assert) ${id}`); console.log(out.split('\n').slice(-12).join('\n')); fail++; }
    else pass++;
  } catch (e) {
    fail++;
    console.log(`ERROR ${id}`);
    console.log(String(e.stderr || e.message).split('\n').slice(-8).join('\n'));
  }
}
console.log(`\nsolutions: ${pass} ok, ${fail} failing, ${skip} skipped (non-python)`);
process.exit(fail ? 1 : 0);

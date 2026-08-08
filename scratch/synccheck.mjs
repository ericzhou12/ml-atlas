/* synccheck.mjs — report challenges whose starter has no self-check but whose
   solution does, and print the trailing assertion block to copy across. */
import { CHALLENGES } from '../js/content/challenges/index.js';

for (const [id, c] of Object.entries(CHALLENGES)) {
  if (!c.starter || !c.solution) continue;
  const hasStarter = /^\s*assert |PASS/m.test(c.starter);
  if (hasStarter) continue;
  const lines = c.solution.split('\n');
  let i = lines.length - 1;
  while (i >= 0 && !/^assert /.test(lines[i])) i--;
  if (i < 0) { console.log(`${id}: solution has no assertions either`); continue; }
  let start = i;
  while (start > 0 && lines[start - 1].trim() !== '' ) start--;
  const block = lines.slice(start).join('\n');
  console.log(`===== ${id}\n${block}\n`);
}

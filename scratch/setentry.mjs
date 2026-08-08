/* setentry.mjs — replace a whole `'lesson-id': { ... },` entry in a challenges file.
   usage: node scratch/setentry.mjs <challengesFile> <lessonId> <replacementFile>
   The replacement file must contain the full entry, including the `'id': {` opener
   and the closing `},`. Boundaries are found by locating the entry's opening line
   and the next top-level entry (or the final `};`). */
import { readFileSync, writeFileSync } from 'node:fs';

const [file, id, repl] = process.argv.slice(2);
const lines = readFileSync(file, 'utf8').split('\n');

const openRe = /^'([\w-]+)':\s*\{\s*$/;
const starts = [];
lines.forEach((l, i) => { const m = l.match(openRe); if (m) starts.push({ id: m[1], i }); });

const k = starts.findIndex((s) => s.id === id);
if (k < 0) { console.error(`entry '${id}' not found in ${file}`); process.exit(1); }

const from = starts[k].i;
let to;
if (k + 1 < starts.length) {
  to = starts[k + 1].i - 1;                      // line before the next entry
  while (to > from && lines[to].trim() === '') to--;   // trim blank lines
} else {
  to = lines.length - 1;
  while (to > from && lines[to].trim() !== '};') to--;
  to--;                                          // stop before the closing `};`
  while (to > from && lines[to].trim() === '') to--;
}
if (lines[to].trim() !== '},') {
  console.error(`expected '},' at line ${to + 1}, found: ${JSON.stringify(lines[to])}`);
  process.exit(1);
}

const body = readFileSync(repl, 'utf8').replace(/\n+$/, '').split('\n');
lines.splice(from, to - from + 1, ...body);
writeFileSync(file, lines.join('\n'));
console.error(`replaced '${id}' (was lines ${from + 1}-${to + 1}, now ${body.length} lines)`);

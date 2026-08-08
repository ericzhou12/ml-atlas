/* appendcheck.mjs — append a self-check block to both the starter and the solution
   of a named challenge.
   usage: node scratch/appendcheck.mjs <file> <lessonId> <blockFile> */
import { readFileSync, writeFileSync } from 'node:fs';

const [file, id, blockFile] = process.argv.slice(2);
// The block is inserted into a JS template literal, so backslashes, backticks and
// ${ must be escaped or they will be interpreted by JS instead of reaching Python.
const block = readFileSync(blockFile, 'utf8')
  .replace(/\n+$/, '')
  .replaceAll('\\', '\\\\')
  .replaceAll('`', '\\`')
  .replaceAll('${', '\\${');
let s = readFileSync(file, 'utf8');

const key = `\n'${id}': {`;
const i = s.indexOf(key);
if (i < 0) { console.error(`entry ${id} not found`); process.exit(1); }
const rest = s.slice(i + key.length);
const m = rest.match(/\n'[\w-]+':\s*\{/);
const end = m ? i + key.length + m.index : s.lastIndexOf('\n};');
let seg = s.slice(i, end);

// insert before the closing backtick of `starter:` and of `solution:`
for (const field of ['starter', 'solution']) {
  const fi = seg.indexOf(`  ${field}: \``);
  if (fi < 0) { console.error(`no ${field} in ${id}`); process.exit(1); }
  // find the terminating "`," of that template literal, skipping escaped backticks
  let j = fi + `  ${field}: \``.length;
  for (;;) {
    j = seg.indexOf('`', j);
    if (j < 0) { console.error(`unterminated ${field}`); process.exit(1); }
    if (seg[j - 1] !== '\\') break;
    j++;
  }
  seg = seg.slice(0, j) + '\n' + block + '\n' + seg.slice(j);
}
s = s.slice(0, i) + seg + s.slice(end);
writeFileSync(file, s);
console.error(`appended self-check to ${id}`);

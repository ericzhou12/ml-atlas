/* splice.mjs — replace an inclusive line range of a file with the contents of another.
   usage: node scratch/splice.mjs <target> <startLine> <endLine> <replacementFile>
   Line numbers are 1-based and inclusive. */
import { readFileSync, writeFileSync } from 'node:fs';

const [target, a, b, repl] = process.argv.slice(2);
const lines = readFileSync(target, 'utf8').split('\n');
const start = Number(a), end = Number(b);
if (!(start >= 1 && end >= start && end <= lines.length)) {
  console.error(`bad range ${start}-${end} for ${target} (${lines.length} lines)`);
  process.exit(1);
}
const body = readFileSync(repl, 'utf8').replace(/\n$/, '').split('\n');
lines.splice(start - 1, end - start + 1, ...body);
writeFileSync(target, lines.join('\n'));
console.error(`spliced ${repl} into ${target}:${start}-${end} (${body.length} lines)`);

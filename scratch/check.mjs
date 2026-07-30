import { readdirSync, statSync } from 'fs';
import { join } from 'path';
import { pathToFileURL } from 'url';

const root = new URL('../js', import.meta.url).pathname;
const files = [];
(function walk(d) {
  for (const f of readdirSync(d)) {
    const p = join(d, f);
    if (statSync(p).isDirectory()) walk(p);
    else if (f.endsWith('.js')) files.push(p);
  }
})(root);

let bad = 0;
for (const f of files.sort()) {
  try {
    await import(pathToFileURL(f).href);
  } catch (e) {
    const msg = e.message.split('\n')[0];
    // runtime-only browser deps are fine
    if (/is not defined|Cannot find module|katex/.test(msg) && e.name !== 'SyntaxError') continue;
    console.log(`\n=== ${f.replace(root, 'js')}`);
    console.log(`    ${e.name}: ${msg}`);
    if (e.stack) console.log('    ' + e.stack.split('\n').slice(0, 4).join('\n    '));
    bad++;
  }
}
console.log(bad ? `\n${bad} file(s) failed` : '\nAll modules parse cleanly.');

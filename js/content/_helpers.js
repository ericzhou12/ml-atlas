/* Shorthand constructors for lesson sections. */

export const t = (md) => ({ t: 'md', md });
export const note = (kind, md, title) => ({ t: 'note', kind, md, title });
export const key = (md) => note('key', md);
export const intuition = (md) => note('intuition', md);
export const warn = (md) => note('warning', md);
export const hist = (md) => note('history', md);
export const mathnote = (md) => note('math', md);
export const recap = (md) => note('recap', md);
export const viz = (id, params) => ({ t: 'viz', id, params });
export const deriv = (title, md) => ({ t: 'deriv', title, md });

/* Pedagogy scaffolding.

   jargon(items)  — a decoder ring. items: [[term, plain-English gloss], ...].
                    Put it BEFORE the notation that uses the words, not after.
   steps(title, items) — a numbered walkthrough. items: ['md', ...] or [{h, md}, ...].
   diagram(title, svg, caption) — a static SVG figure. Use CSS vars for every colour
                    (var(--s1)…var(--s8), var(--text), var(--border)) so both themes work.
   tldr(md)       — the whole lesson in a few lines, placed first.
   recap(md)      — what the reader can now say, placed last.                       */
export const jargon = (items) => ({ t: 'jargon', items });
export const steps = (title, items) => ({ t: 'steps', title, items });
export const diagram = (title, svg, caption) => ({ t: 'diagram', title, svg, caption });
export const tldr = (md) => note('tldr', md);
export const code = (title, src, explain, lang = 'python') => ({ t: 'code', title, code: src, explain, lang });
export const quiz = (q, options, answer, explain) => ({ t: 'quiz', q, options, answer, explain });
export const refs = (items) => ({ t: 'refs', items });

/* Reference shorthands */
export const paper = (title, author, year, url, note) => ({ kind: 'paper', title, author, year, url, note });
export const book = (title, author, year, url, note) => ({ kind: 'book', title, author, year, url, note });
export const course = (title, author, year, url, note) => ({ kind: 'course', title, author, year, url, note });
export const blog = (title, author, year, url, note) => ({ kind: 'blog', title, author, year, url, note });
export const video = (title, author, year, url, note) => ({ kind: 'video', title, author, year, url, note });
export const codeRef = (title, author, year, url, note) => ({ kind: 'code', title, author, year, url, note });
export const demo = (title, author, year, url, note) => ({ kind: 'interactive', title, author, year, url, note });

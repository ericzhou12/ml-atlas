# Adding a Track

How to add a new track to ML Atlas. Written for future-you (or a future agent) picking this up cold.

All ten planned tracks now exist (61 lessons). This document is for adding an eleventh, splitting an existing
track, or extending one. `node scratch/smoke.mjs` (see §7) reports current counts and any unused figures.

---

## 1. The shape of the thing

```
ml-learning/
  index.html            # app shell — you will not need to touch this
  styles/app.css        # ditto
  js/
    main.js             # router, sidebar, search, lesson renderer
    md.js               # markdown + KaTeX renderer
    plot.js             # canvas plotting toolkit (Plot class, LA, rng, colors)
    ui.js               # panel() — builds a figure card with controls
    ide.js              # Pyodide code lab
    content/
      _helpers.js       # t(), key(), viz(), code(), quiz(), paper() ...
      index.js          # ← TRACKS registry, GLOSSARY, PATHS       ★ edit this
      math.js           # ← one file per track                      ★ add this
      classical.js
      nn.js
    viz/
      index.js          # ← VIZ registry                            ★ edit this
      math.js           # ← figures, grouped by theme               ★ add here
      classical.js  nn.js  transformers.js  generative.js  rl.js  frontier.js
  vendor/               # KaTeX + Pyodide + NumPy, all local, no network needed
```

Adding a track touches **three files**: a new `js/content/<track>.js`, plus registrations in
`js/content/index.js` and (if you add figures) `js/viz/index.js`.

---

## 2. Write the lesson file

Create `js/content/<track>.js`. It default-exports an **array of lesson objects**.

```js
import { t, key, intuition, warn, hist, mathnote, viz, deriv, code, quiz,
         paper, book, course, blog, video, demo, codeRef } from './_helpers.js';

export default [
{
  id: 'llm-attention',            // unique across ALL tracks; kebab-case, track-prefixed
  title: 'Attention',
  sub: 'One sentence that says what this lesson is actually about.',
  mins: 25,                       // honest reading time
  level: 'core',                  // 'foundations' | 'core' | 'advanced' | 'frontier'
  prereq: ['nn-backprop'],        // lesson ids — validated by the smoke test
  tags: ['attention', 'transformers'],
  sections: [ /* see §3 */ ],
  refs:     [ /* see §4 */ ],
},
// ... more lessons
];
```

`id` is permanent — it is the URL (`#/l/llm-attention`) and the progress key in localStorage. Renaming one
orphans a user's completion mark and breaks any inbound link.

---

## 3. Section types

`sections` is an ordered array. These are all of them:

| Helper | Renders as |
|---|---|
| `tldr(md)` | Blue "◎ The short version" callout. **Required, and must be the first section.** |
| `jargon([[term, gloss], …])` | The "⌘ Words you will need first" decoder table. **Required.** |
| `t(md)` | Body prose. Markdown + `$inline$` / `$$display$$` TeX. |
| `key(md)` | Purple "★ Key idea" callout. |
| `intuition(md)` | Green "◈ Intuition" callout. |
| `warn(md)` | Orange "⚠ Watch out" callout. |
| `hist(md)` | Grey "⏱ History" callout. |
| `mathnote(md)` | Cyan "∑ Math note" callout. |
| `steps(title, items)` | Numbered walkthrough. `items` is `['md', …]` or `[{h, md}, …]`. |
| `diagram(title, svg, caption)` | A static inline-SVG figure. |
| `deriv(title, md)` | Collapsed `<details>` block. Put the algebra here. |
| `viz(id, params?)` | An interactive figure from the `VIZ` registry. |
| `code(title, src, explain?, lang?)` | Syntax-highlighted block with **copy** and **open in lab** buttons. `lang` defaults to `'python'`. |
| `quiz(q, options, answerIdx, explain)` | Multiple choice, 3–4 options, reveals on click. |
| `refs(items)` | An inline reference list mid-lesson (rare — normally use the lesson-level `refs`). |
| `recap(md)` | Green "✓ You can now say" checklist. **Required, and must be the last section.** |

### The three required blocks

`smoke.mjs` warns if any lesson is missing `tldr`, `jargon`, or `recap`, or if the first/last section is wrong.
They exist because the atlas is read by people who do not already know the vocabulary:

- **`tldr`** — the whole lesson in three or four short paragraphs, in plain language, before any notation. If a
  reader stops here they should still have gained something true.
- **`jargon`** — every term the lesson uses that a CS-literate beginner would not already know, glossed in one
  sentence each, in the order they appear. Put it **before** the notation that uses the words, not after.
  Gloss the *symbols* too: `$\mathbf{x} \in \mathbb{R}^d$` is jargon, and reading it aloud is a skill.
- **`recap`** — what the reader can now *do* or *say*, as a checklist of verbs. Not a summary of the content;
  a list of capabilities they can test themselves against.

### Diagrams

`diagram()` takes raw inline SVG, so it lives in the content file and needs no viz registration. Two rules:

1. **Never a hex colour.** Use `style="fill: var(--s1)"` — a *style attribute*, not `fill="var(--s1)"`, which
   browsers do not resolve. `smoke.mjs` warns on hex literals.
2. Use the shared classes: `dtitle` (bold label), `dlabel` (dim caption), `dmono` (monospace). Set a `viewBox`
   and no width/height so it scales.

Use a diagram where a *static* picture carries the idea (a shape, a comparison, a decomposition) and a `viz()`
where the reader needs to change something and watch it respond.

### Markdown notes

The renderer in `md.js` is hand-rolled and supports exactly: headings, paragraphs, nested lists, tables,
blockquotes, fenced and inline code, bold/italic/strike, links, `---`, and TeX. Not supported: footnotes,
HTML blocks, reference-style links.

**The two things that will bite you:**

1. Lesson content lives in JS template literals, so a literal backtick must be escaped: `` \`np.linalg.lstsq\` ``.
   An unescaped one silently terminates the string and produces a baffling syntax error. This has happened.
2. Escape backslashes for TeX: write `$\\nabla f$`, not `$\nabla f$`. In a template literal `\n` is a newline.

Custom KaTeX macros are already defined in `md.js` — `\R \E \P \N \L \KL \argmin \argmax \T \bx \bw \btheta` and
friends. Add more there rather than repeating `\mathbb{R}` everywhere.

Cross-link other lessons with `[text](#/l/lesson-id)`. The smoke test validates every one.

---

## 4. References

Every lesson should carry real sources. Helpers set the `kind` badge and feed the global
[Reference library](#) page, which dedupes by title and back-links to lessons.

```js
refs: [
  paper('Attention Is All You Need', 'Vaswani et al.', 2017,
        'https://arxiv.org/abs/1706.03762',
        'Why it matters, in one or two sentences. This note is displayed.'),
  book('Deep Learning', 'Goodfellow, Bengio & Courville', 2016, 'https://www.deeplearningbook.org/'),
  blog(...), course(...), video(...), demo(...), codeRef(...),
]
```

Prefer primary sources with stable URLs (arXiv, official book sites, university course pages). The `note`
field is the most valuable part — say *why* someone should read it, not what it is about.

---

## 5. Figures

### Reusing an existing one

```js
viz('gradient-descent', { surf: 'rosen', opt: 'adam', lr: 0.05, compare: true })
```

The `params` object is passed to the figure factory; most accept initial-state overrides. Check the factory
signature in `js/viz/*.js` for what each takes.

### Writing a new one

Add it to the thematically appropriate `js/viz/*.js` (or a new file, imported in `js/viz/index.js`):

```js
V['my-figure'] = (host, params = {}) => {
  panel(host, {
    title: 'Shown in the card header',
    height: 300,
    plot: { xlim: [-3, 3], ylim: [-2, 2], equal: true, xlabel: 'x', ylabel: 'y' },

    controls: [
      { type: 'slider', key: 'k', label: 'steepness', min: 0, max: 5, step: .1, value: 1 },
      { type: 'select', key: 'mode', label: 'mode', value: 'a',
        options: [{ value: 'a', label: 'first' }, { value: 'b', label: 'second' }] },
      { type: 'check',  key: 'grid', label: 'show grid', value: true },
      { type: 'play' },                                        // animation toggle
      { type: 'button', label: '↺ reset', onClick: (s, P) => { /* mutate s */ } },
    ],

    // called on every redraw. p is a Plot, s is the control state.
    draw(p, s, P) {
      p.clear().axes();
      p.clip();                                                // clip to the plot box
      p.fn((x) => Math.tanh(s.k * x), { color: cssVar('--s1'), width: 2.4 });
      p.clip(false);
      p.legend([{ label: 'tanh(kx)', color: cssVar('--s1') }], { pos: 'tl' });
      P.readout({ 'k': fmt(s.k, 2), 'slope at 0': fmt(s.k, 3) });   // the mono strip below
    },

    animate(s, P) { s.k = (s.k + 0.02) % 5; },                 // per-frame, only while playing
    interact: (s, P) => ({ down(x, y) {...}, move(x, y) {...}, hover(x, y) {...} }),
    caption: 'Markdown. This is where the *insight* goes — see §6.',
  });
};
```

Then register it in `js/viz/index.js` if you created a new file.

**`Plot` API** (`js/plot.js`) — all coordinates are data coordinates unless noted:

`clear()` `axes({nx,ny,ticks})` `clip(bool)` `setLim(xlim,ylim)` `line(pts,{color,width,dash,alpha})`
`fn(f,{n,from,to})` `area(pts,{base})` `points(pts,{r,color,shape,stroke})` `arrow(x0,y0,x1,y1,{head})`
`circle(x,y,rPx,{fill,stroke})` `ellipse(cx,cy,cov,{n})` `bars(vals,{x0,gap})` `heat(f,{step,lo,hi,cmap})`
`contour(f,levels,{color})` `quiver(fn,{nx,ny,scale})` `matrix(M,{showVals,cmap})` `text(x,y,s,{pixel,align})`
`legend(items,{pos})` `nearest(pts,x,y,rPx)` · transforms `px() py() ix() iy()` · scales `.sx .sy`

Also exported: `cssVar('--s1')` (theme colors `--s1`…`--s8`), `alpha()`, `mix()`, `seqMap()`, `divergeMap()`,
`fmt()`, `rng(seed)` (seeded, with `.normal() .shuffle() .pick()`), and `LA` (`matmul`, `matvec`, `solve`,
`eig2`, `softmax`, …).

**Rules that keep figures consistent:**

- Never hard-code a hex color. Use `cssVar('--s1')` etc. so light/dark themes both work.
- Use `rng(seed)` rather than `Math.random()` in `draw()` — an unseeded RNG makes the figure flicker on every
  redraw and resize.
- Compute real math. Every figure in the atlas runs the actual algorithm; where a curve is stylized (the
  grokking and test-time-compute figures), the caption says so explicitly. Keep that norm.
- Put the numbers in `P.readout({...})`, not on the canvas.

---

## 6. The voice

This is the part that makes it worth reading. The pattern every lesson follows:

1. **Intuition before notation.** A picture or an analogy, then the symbols.
2. **Derive, do not assert.** If a formula appears, either derive it inline or put the derivation in a
   `deriv()` block. "It can be shown that" is a failure.
3. **Say what is contested.** Normalization works and nobody fully agrees why. Say so. The atlas is more
   trustworthy for admitting the gaps than for papering over them.
4. **Captions carry the teaching.** A caption should not describe the figure — it should tell the reader what
   to *do* with it and what they will see. "Set the forget gate to 1.0 and compare against the vanilla RNN" is
   the format. This is where most of the value is.
5. **Quizzes target the common misconception**, not recall. The `explain` field should be a short paragraph
   that teaches, and it is shown whether the reader was right or wrong.
6. **Code should run.** Every Python block runs in the lab as-is, against NumPy only (no torch, no sklearn, no
   matplotlib). Helpers `aplot()` and `describe()` are pre-injected. Test anything non-trivial before shipping.
7. **Every code block needs an `explain`.** Not a description of what the code does — the reader can see that.
   Tell them which number in the output to look at and what it means. `audit.mjs` warns if one is missing.

### The reader you are writing for

Assume single-variable calculus and an introductory statistics course, and nothing else. Concretely, before you
ship a paragraph, ask two questions of every sentence:

1. **Is every concept in it either common knowledge, or taught earlier in the atlas?** If not, gloss it inline or
   add it to `jargon()`. This includes notation: `$\mathbf{x} \in \mathbb{R}^d$` needs reading aloud the first
   time, and so does `$\arg\max$`, `$\odot$`, and `$\propto$`.
2. **Does the sentence do work?** Not "is it true" — is it load-bearing for the explanation? Cut sentences that
   only demonstrate that the author knows something.

The tools available for rule 1 are: derive it (best), gloss it in one clause, or link to the lesson that covers
it. Naming a technique without explaining it is fine *only* when the name is the point — "this is called
superposition, and there is a whole lesson on it later" is useful; a list of five method names is not.

### Challenges

The rule is that a challenge must be **completable from the lesson it belongs to**, and it should test a claim
the lesson actually made rather than exercise generic coding. Every entry needs four things:

- a `starter` with `TODO`s and a **self-check** — assertions with messages, ending in `print("PASS")`, so the
  learner gets a verdict without reading the solution;
- a `solution` that runs (`runsol.mjs` is the gate);
- an `explain` that says what the output *means*, referring to the actual numbers it produces;
- a prompt that, wherever there is a surprise coming, asks the reader to **predict before running**.

The best challenges measure something the lesson asserted: that a gradient vanishes, that two definitions
coincide, that a method fails on a case it was never designed for. If an experiment does not support the claim
you wrote, change the claim — several explanations in this atlas were rewritten after the numbers disagreed with
them.

---

## 7. Register and verify

**a. Register the track** in `js/content/index.js`:

```js
import llm from './llm.js';

export const TRACKS = [
  /* ... */
  { id: 'llm', num: 4, name: 'Large Language Models',
    color: '#ff8f5a',                     // pick from the --s1..--s8 palette
    desc: 'One or two sentences for the home page card.',
    intro: `Optional longer intro shown on the track page.`,
    lessons: llm },
];
```

**b. Add glossary entries** for new jargon (`GLOSSARY` in the same file), each with a `see:` pointing at the
lesson that explains it.

**c. Add or extend a learning path** in `PATHS` if the track fits an existing route.

**d. Run the checks.** All of them live in `scratch/`:

```bash
node scratch/check.mjs      # every module parses (catches the backtick and ** bugs)
node scratch/smoke.mjs      # content graph integrity
node scratch/runsol.mjs     # every challenge solution actually runs, under python3 + numpy
node scratch/audit.mjs      # missing code explanations, challenge explains, self-checks
```

`smoke.mjs` validates: unique lesson ids · every `viz()` id exists in the registry · every `prereq` resolves ·
every `#/l/` link resolves · glossary `see:` targets · path lesson ids · quiz answer indices in range · every
markdown block renders without throwing · and it lists figures built but never placed.

`runsol.mjs` extracts each challenge's `solution` and executes it. It is the gate that matters most, because a
challenge whose own reference solution fails is worse than no challenge — and three of them did, silently, before
it existed. It needs `numpy` (`pip install numpy`); a few solutions also want `matplotlib`.

All of them exit non-zero on any error. Treat that as the gate.

**Editing helpers** (also in `scratch/`), useful because lesson and challenge entries are large template
literals that are awkward to edit by hand:

```bash
node scratch/splice.mjs <file> <startLine> <endLine> <replacementFile>   # replace a line range
node scratch/setentry.mjs <challengesFile> <lessonId> <replacementFile>  # replace a whole challenge entry
node scratch/appendcheck.mjs <challengesFile> <lessonId> <blockFile>     # append a self-check to both
                                                                        # starter and solution
```

`appendcheck.mjs` escapes the block for embedding in a template literal. Doing that by hand is how you end up
with a `\n` that reaches Python as a real newline and produces an unterminated string.

**e. Serve it.**

```bash
python3 -m http.server 8000     # ES modules need http://, not file://
```

Then click through the new track: expand it in the sidebar, load each lesson, poke every figure, press
**▶ Run** in the code lab on at least one Python block, and toggle the theme to check figure colors.

---

## 8. Current state and where to extend

All ten tracks are written and all 95 figures are placed. Six lessons are deliberately figure-free
(`llm-rag`, `llm-evaluation`, `sys-gpu`, and the three `pr-*` lessons) — they lean on code blocks and tables
instead, which is fine, but each would take one well.

Gaps worth filling, roughly by value:

| Area | What is missing |
|---|---|
| **Graph neural networks** | No coverage at all. Message passing, GCN/GAT, and the over-smoothing problem. Would slot after track 3. |
| **Time series / forecasting** | Classical (ARIMA, state space) through to transformer forecasters. |
| **Speech and audio** | Whisper-style ASR, neural vocoders, audio tokenization. Track 8 is vision-only. |
| **Causal inference** | Confounding, do-calculus, and why regression coefficients are not causal — `ml-linear-regression` gestures at this and stops. |
| **Fairness and privacy** | Differential privacy, membership inference, fairness metrics and their impossibility results. |
| **Tabular deep learning** | `ml-trees-ensembles` explains why trees still win; a lesson on TabPFN and friends would balance it. |
| **World models / robotics** | Track 7 stops at RLHF; embodied RL and model-based methods are absent. |

Existing tracks that could take another lesson or two: track 5 (only 4 lessons — a kernels/CUDA lesson and a
compiler/`torch.compile` lesson would fit), track 6 (autoregressive image generation, consistency models),
track 8 (video, any-to-any models).

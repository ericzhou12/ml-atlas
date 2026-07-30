# ML Atlas

An offline, single-page course and reference for machine learning — from vectors and derivatives through
neural networks, with interactive figures, derivations, runnable NumPy, and linked primary sources.

No build step, no server, no network. KaTeX and CPython+NumPy (via Pyodide) are vendored locally.

## Run it

```bash
python3 -m http.server 8000
open http://localhost:8000
```

ES modules require `http://` — opening `index.html` as a `file://` URL will not work.

## What's here

| | |
|---|---|
| Tracks | 11 |
| Lessons | 66 |
| Interactive figures | 102 built, 105 placements |
| Runnable code blocks | 65 |
| Derivations | 14 |
| Quizzes | 66 |
| Cited sources | 305 |
| Glossary terms | 97 |
| Learning paths | 12 |

1. **Mathematical Foundations** (9) — vectors, matrices, eigen/SVD, derivatives, Jacobians, probability,
   information theory, optimization, numerical precision.
2. **Classical ML** (10) — framing, linear regression, overfitting and double descent, regularization, logistic
   regression, trees and boosting, SVMs and kernels, clustering and PCA, evaluation, generative vs discriminative.
3. **Neural Networks** (10) — perceptron to MLP, backprop, activations, initialization, normalization,
   regularization, losses and the training loop, CNNs, RNNs and LSTMs, embeddings.
4. **Transformers & LLMs** (10) — tokenization, attention, the transformer block, pretraining, scaling laws,
   fine-tuning and LoRA, decoding, prompting, RAG, MoE, evaluation.
5. **Training & Inference Systems** (4) — GPUs and the roofline, distributed training, quantization, serving.
6. **Generative Models** (3) — autoencoders and VAEs, GANs, diffusion and flow matching.
7. **RL & Alignment** (4) — MDPs and Bellman, Q-learning and exploration, policy gradients and PPO, RLHF/DPO/RLVR.
8. **Vision & VLMs** (3) — vision transformers, CLIP, vision-language architectures.
9. **Frontier** (4) — mechanistic interpretability, reasoning and test-time compute, SSMs, open problems.
10. **Physical & Embodied Intelligence** (5) — why robots are different, visuomotor policies (ACT, Diffusion Policy),
    vision-language-action models (RT-2, OpenVLA, π₀), world models, and BEHAVIOR/sim-to-real evaluation.
11. **Practice** (3) — debugging, running trustworthy experiments, reading papers.

## The three panes

- **Left** — curriculum. ✓ marks completed lessons; progress is stored in localStorage only.
- **Middle** — the lesson. Figures are live: drag the points, move the sliders, press play.
- **Right** — the code lab (<kbd>⌘J</kbd>). Real CPython 3.12 + NumPy in WebAssembly, with syntax highlighting,
  ~10 s first boot, then instant. Every Python block has an **open in lab** button. Helpers `aplot()` and
  `describe()` are preloaded. A JavaScript tab runs instantly with a small linear-algebra library in scope.

Keyboard: <kbd>/</kbd> search · <kbd>←</kbd><kbd>→</kbd> prev/next lesson · <kbd>⌘B</kbd> sidebar ·
<kbd>⌘J</kbd> lab · <kbd>⌘↵</kbd> run code.

## Extending it

Read `AUTHORING.md`. Adding a track means one new file in `js/content/` plus two registrations. Before
committing:

```bash
node scratch/check.mjs   # every module parses
node scratch/smoke.mjs   # content graph integrity — exits non-zero on error
```

## Layout

```
index.html          app shell
styles/app.css      all styling, light + dark themes
js/main.js          router, sidebar, search, lesson renderer
js/md.js            markdown + KaTeX renderer
js/plot.js          canvas plotting toolkit
js/ui.js            figure panels and controls
js/ide.js           Pyodide code lab
js/content/         curriculum — one file per track
js/viz/             interactive figures — one file per theme
vendor/             KaTeX (1.4 MB), Pyodide + NumPy (25 MB)
```

## Caveats

Demos are toy-scale by design; where behavior only emerges at scale, the text says so. Two figures
(grokking, test-time compute) plot stylized reproductions of published results rather than live computation,
and say so in their captions. Everything else computes real math.

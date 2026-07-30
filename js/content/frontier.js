/* ============================================================
   Track 9 — Frontier Topics
   ============================================================ */

import { t, key, intuition, warn, hist, mathnote, viz, deriv, code, quiz, paper, book, course, blog, video, demo, codeRef } from './_helpers.js';

export default [

/* ---------------------------------------------------------- */
{
  id: 'fr-interpretability',
  title: 'Mechanistic Interpretability',
  sub: 'Reverse-engineering the algorithms a network actually learned.',
  mins: 26, level: 'frontier',
  prereq: ['llm-transformer', 'math-vectors'],
  tags: ['interpretability', 'superposition', 'SAE'],
  sections: [
    t(`## The ambition

Not "which input pixels mattered" (saliency maps) but "**what algorithm is this network running, expressed in terms a
human can check?**" The aspiration is closer to decompiling a binary than to correlational analysis.

It has actually worked in places. Researchers have identified curve detectors in vision models, induction heads that
implement in-context copying, and a complete circuit for indirect object identification in GPT-2. These are real
mechanistic accounts, verified by intervention.`),

    t(`## Superposition: why neurons are not features

The first obstacle is that individual neurons rarely correspond to individual concepts. A neuron fires for Chinese
text, DNA sequences, and HTTP requests. This is **polysemanticity**, and it is not sloppiness — it is optimal.`),

    viz('superposition'),

    key(`A $d$-dimensional layer holds only $d$ orthogonal directions. But if features are **sparse** — rarely active at
the same time — the model can pack many more as *non-orthogonal* directions and tolerate the rare collision.

Slide sparsity from 0.2 to 0.95 in the figure and watch the model switch strategies: dense features get orthogonal
storage and there is room for only $d$ of them; sparse features get packed in.

This is a compression scheme the network discovered on its own, and it is the reason "just look at what neuron 4,213
does" fails.`),

    t(`## Sparse autoencoders

If features are superposed in a $d$-dimensional space, recover them by projecting into a *wider*, sparse basis. Train
an autoencoder on the activations with an L1 penalty pushing the hidden code toward sparsity:

$$\\mathcal{L} = \\|\\mathbf{x} - \\hat{\\mathbf{x}}\\|^2 + \\lambda\\|\\mathbf{f}\\|_1$$

with the hidden layer 8–64× wider than the input.`),

    viz('sparse-autoencoder'),

    t(`The recovered features are often strikingly interpretable — Anthropic's Claude 3 Sonnet work surfaced features
for the Golden Gate Bridge, for code with security vulnerabilities, for sycophantic praise. Crucially they could be
**intervened on**: clamping the Golden Gate feature high made the model bring the bridge into every response, which is
a causal test rather than a correlational story.

Honest limitations, all active research:

- **Reconstruction is never perfect**, and the residual may contain exactly what you care about.
- **Feature splitting** — increasing dictionary size splits one feature into several, with no principled stopping
  point. Is "the" one feature or twelve?
- **No ground truth.** You cannot check the recovered features against what the model "really" uses, because that is
  the thing you were trying to find out.
- **Coverage.** Even large SAEs capture a minority of what a frontier model computes.
- Newer variants (top-k SAEs, transcoders, crosscoders) address parts of this.`),

    t(`## Circuits

The other half is tracing *computation*, not just representation. The method:

1. **Activation patching** — run the model on a corrupted input, patch in clean activations at one location, and see
   whether the output recovers. This localizes causally responsible components.
2. **Path patching** — the same, restricted to specific paths between components, to isolate the edges of the circuit.
3. **Ablation** — remove a component and check what breaks.

The IOI circuit ("When Mary and John went to the store, John gave a drink to ___") was mapped to about 26 heads with
identified roles: duplicate-token heads, S-inhibition heads, name-mover heads. It is a genuine algorithmic account of
one behavior in one small model.

The scaling problem is severe: this took months of human effort for one task in a 117M-parameter model. Automating it
is the central open problem.`),

    t(`## Why this matters

Beyond curiosity: if you cannot inspect what a model computes, you cannot verify it is not deceiving you, detect
dangerous capabilities before deployment, or debug failures in any principled way. Every current safety technique is
behavioral — we test outputs. Interpretability is the only path to checking the mechanism.

It is also genuinely unfinished, which makes it one of the more honest areas of the field about what it does not know.`),

    code('Superposition, reproduced', `import numpy as np
rng = np.random.default_rng(0)

# Toy model of superposition (Elhage et al. 2022):
# n features -> m dimensions -> reconstruct. m < n forces choices.
n_feat, m_dim = 12, 4

def train(sparsity, steps=4000, lr=0.02):
    W = rng.normal(0, 0.3, (n_feat, m_dim))
    b = np.zeros(n_feat)
    imp = 0.85 ** np.arange(n_feat)               # features differ in importance
    for _ in range(steps):
        B = 256
        x = rng.random((B, n_feat)) * (rng.random((B, n_feat)) > sparsity)
        h = x @ W
        out = np.maximum(0, h @ W.T + b)
        err = (out - x) * imp
        d = 2 * err * (out > 0) / B
        gW = h.T @ d + (d @ W).T * 0
        W -= lr * (d.T @ h + (x.T @ (d @ W)))
        b -= lr * d.sum(0)
    return W

print(f"{n_feat} features into {m_dim} dimensions\\n")
print(f"{'sparsity':>9} {'features stored':>16} {'mean |cos| between them':>25}")
for sp in [0.0, 0.5, 0.8, 0.95, 0.99]:
    W = train(sp)
    norms = np.linalg.norm(W, axis=1)
    stored = (norms > 0.35 * norms.max()).sum()
    Wn = W / (norms[:, None] + 1e-9)
    G = np.abs(Wn @ Wn.T)
    off = G[~np.eye(n_feat, dtype=bool)]
    print(f"{sp:9.2f} {stored:16d} {off.mean():24.3f}")

print("\\nDense features (sparsity 0): only ~4 stored, near-orthogonally.")
print("Sparse features (0.95+): many more packed in, accepting interference.")
print("The model is choosing compression over fidelity, and it is right to.")`),

    quiz('Why do sparse autoencoders use a hidden layer WIDER than their input?',
      ['The hypothesis is that the model stores more features than it has dimensions, so recovering them needs an overcomplete basis',
       'Wider layers train faster',
       'To increase reconstruction accuracy',
       'To match the transformer\'s FFN dimension'],
      0,
      'Superposition means the layer holds more features than dimensions, packed as non-orthogonal directions. An autoencoder with a bottleneck could not represent them separately. Going 8–64× wider, plus an L1 penalty forcing only a few to activate at once, gives each feature room to claim its own unit. Overcompleteness is the entire point.'),
  ],
  refs: [
    paper('Toy Models of Superposition', 'Elhage et al.', 2022, 'https://transformer-circuits.pub/2022/toy_model/index.html', 'The clearest possible treatment. The code above reproduces its central experiment.'),
    paper('Towards Monosemanticity', 'Bricken et al.', 2023, 'https://transformer-circuits.pub/2023/monosemantic-features', 'Sparse autoencoders working on a real model.'),
    paper('Scaling Monosemanticity', 'Templeton et al.', 2024, 'https://transformer-circuits.pub/2024/scaling-monosemanticity/', 'SAEs on Claude 3 Sonnet, with causal interventions.'),
    paper('Interpretability in the Wild: IOI in GPT-2', 'Wang et al.', 2022, 'https://arxiv.org/abs/2211.00593', 'A complete circuit, end to end.'),
    paper('A Mathematical Framework for Transformer Circuits', 'Elhage et al.', 2021, 'https://transformer-circuits.pub/2021/framework/index.html', 'The residual-stream framing that the whole field now uses.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'fr-reasoning',
  title: 'Reasoning and Test-Time Compute',
  sub: 'Buying accuracy with inference instead of parameters.',
  mins: 24, level: 'frontier',
  prereq: ['llm-prompting', 'rl-rlhf'],
  tags: ['reasoning', 'test-time compute'],
  sections: [
    t(`## A second scaling axis

For a decade, better meant bigger — more parameters, more pretraining data. Reasoning models opened a second axis:
**spend more compute at inference on a single problem**.`),

    viz('test-time-compute'),

    t(`The structural fact in that figure is the gap between **pass@n** (did *any* of $n$ samples get it right) and
what you can actually extract. The model frequently *can* find the answer; the difficulty is knowing which sample is
correct.

- **Majority voting / self-consistency** closes some of the gap for free, exploiting the fact that there are many ways
  to be wrong and usually one way to be right.
- **Best-of-n with a verifier** closes more, but degrades at large $n$ — you begin selecting for whatever fools the
  verifier rather than what is correct.
- **Long chain-of-thought trained with RL** closes it inside a *single* sample, which is the qualitative change.`),

    t(`## How reasoning models are trained

The recipe that emerged in 2024–25 is **RL with verifiable rewards** (RLVR):

1. Take problems with checkable answers — math with known solutions, code with test suites, logic puzzles.
2. Sample many attempts per problem.
3. Reward correctness. No learned reward model, so no reward hacking of the usual kind.
4. Optimize with PPO or GRPO.

DeepSeek-R1 documented something remarkable: with **no reasoning demonstrations at all**, models spontaneously develop
long chains of thought, backtracking, and self-verification. Response length grows steadily over training as the model
discovers that thinking longer pays. The paper shows an "aha moment" where a model mid-training writes out a
realization that its earlier approach was wrong.

The distinction that matters: **RLHF shapes behavior against human taste; RLVR improves capability against ground
truth.** They are different bets with different failure modes.`),

    warn(`**What this does not fix.** Reasoning models are better at math and code and only somewhat better at tasks
without verifiable answers — which is most tasks. Reward hacking reappears in new forms (writing code that special-cases
the tests). Long chains are expensive and slow. And there is evidence the visible chain-of-thought is not always a
faithful account of the computation that produced the answer, which matters a great deal if you were hoping to use it
for oversight.`),

    t(`## Agents

An agent is a loop: think → act (call a tool) → observe → repeat. Tools give the model access to search, code
execution, and APIs, which addresses knowledge cutoffs and arithmetic in one move.

The arithmetic of the loop is unforgiving.`),

    viz('agent-loop'),

    t(`At 95% per-step reliability a 30-step task succeeds **21%** of the time. This is why agent engineering is
dominated not by reasoning quality but by **error recovery** — checkpointing, verification steps, letting the model
notice and undo bad actions. Turn on recovery in the figure and watch the curve lift.

The practical consequences:

- **Reliability per step is the metric that matters**, not average capability.
- **Bounded, verifiable actions** beat open-ended ones. A tool that can only do one safe thing is easier to trust.
- **Human checkpoints** at consequential steps.
- **Prompt injection is unsolved.** An agent that reads a webpage is executing instructions from that webpage. Never
  let model output take an irreversible action without validation outside the model.`),

    code('The economics of test-time compute', `import numpy as np
rng = np.random.default_rng(0)

def trial(p_correct, n, strategy, verifier_acc=0.9, n_wrong=10):
    answers = []
    for _ in range(n):
        answers.append("right" if rng.random() < p_correct
                       else f"wrong_{rng.integers(n_wrong)}")
    if strategy == "single":
        return answers[0] == "right"
    if strategy == "majority":
        c = {}
        for a in answers: c[a] = c.get(a, 0) + 1
        return max(c, key=c.get) == "right"
    if strategy == "verifier":
        best, best_s = None, -1
        for a in answers:
            s = rng.random() * (verifier_acc if a == "right" else 1 - verifier_acc + 0.35)
            if s > best_s: best, best_s = a, s
        return best == "right"
    if strategy == "oracle":                       # pass@n ceiling
        return "right" in answers

print(f"per-attempt accuracy 0.35, wrong answers scattered over 10 modes\\n")
print(f"{'n':>5} {'single':>8} {'majority':>10} {'verifier':>10} {'pass@n':>9}")
for n in [1, 4, 16, 64, 256]:
    row = []
    for s in ["single", "majority", "verifier", "oracle"]:
        row.append(np.mean([trial(0.35, n, s) for _ in range(2000)]))
    print(f"{n:5d} {row[0]:8.3f} {row[1]:10.3f} {row[2]:10.3f} {row[3]:9.3f}")

print("\\nA weak verifier stops helping as n grows:")
for va in [0.95, 0.85, 0.70, 0.60]:
    accs = [np.mean([trial(0.35, n, "verifier", va) for _ in range(1500)])
            for n in [4, 64, 256]]
    print(f"  verifier acc {va:.2f}: n=4 {accs[0]:.3f}  n=64 {accs[1]:.3f}  n=256 {accs[2]:.3f}")

# --- agent reliability ---
print("\\nagent success vs steps:")
print(f"{'per-step':>9}" + "".join(f"{s:>8}" for s in [5,10,20,50]))
for p in [0.90, 0.95, 0.99, 0.999]:
    print(f"{p:9.3f}" + "".join(f"{p**s:8.3f}" for s in [5,10,20,50]))`),

    quiz('Best-of-64 with a verifier performs worse than best-of-16. What is happening?',
      ['With more candidates you increasingly select for whatever exploits the verifier rather than what is correct',
       'The model gets worse with more samples',
       'The verifier is being overloaded',
       'This cannot happen'],
      0,
      'Classic Goodhart. Each sample is a draw; picking the max of $n$ verifier scores means picking the maximum of $n$ noisy estimates, and as $n$ grows you are increasingly sampling the verifier\'s error tail rather than genuine correctness. Gao et al. quantified exactly this for reward models. Mitigations: a stronger verifier, ensembling verifiers, or capping $n$.'),
  ],
  refs: [
    paper('DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL', 'DeepSeek-AI', 2025, 'https://arxiv.org/abs/2501.12948', 'Reasoning emerging from pure RL. Unusually transparent about method.'),
    paper('Let\'s Verify Step by Step', 'Lightman et al.', 2023, 'https://arxiv.org/abs/2305.20050', 'Process supervision beats outcome supervision.'),
    paper('Scaling LLM Test-Time Compute Optimally', 'Snell et al.', 2024, 'https://arxiv.org/abs/2408.03314', 'When inference compute beats a bigger model.'),
    paper('Measuring Faithfulness in Chain-of-Thought Reasoning', 'Lanham et al.', 2023, 'https://arxiv.org/abs/2307.13702', 'The stated reasoning is not always the actual reasoning.'),
    paper('ReAct: Synergizing Reasoning and Acting', 'Yao et al.', 2022, 'https://arxiv.org/abs/2210.03629', 'The think-act-observe loop.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'fr-architectures',
  title: 'Beyond Transformers: SSMs and Efficient Attention',
  sub: 'What might replace quadratic attention, and why nothing has yet.',
  mins: 22, level: 'frontier',
  prereq: ['nn-rnn', 'llm-attention'],
  tags: ['SSM', 'Mamba', 'architecture'],
  sections: [
    t(`## The target

Attention is $O(n^2)$ in compute and needs a KV cache growing linearly with context. An RNN is $O(n)$ with $O(1)$
state — but cannot be parallelized over the sequence during training, which is why transformers won.

**Can you have both?**`),

    t(`## State-space models

An SSM is a linear recurrence:

$$\\mathbf{h}_t = A\\mathbf{h}_{t-1} + B\\mathbf{x}_t, \\qquad y_t = C\\mathbf{h}_t$$

Linearity is the key. Unrolled, it becomes a **convolution** with kernel $(CB, CAB, CA^2B, \\ldots)$, computable with an
FFT — so training parallelizes over the sequence like a transformer, while inference runs as a recurrence with
constant state.

Two ingredients made it work:

- **Structured $A$ (HiPPO/S4).** Random $A$ forgets almost immediately. HiPPO initialization gives channels
  geometrically spaced decay rates, so the state holds a multi-scale summary of the past.
- **Selectivity (Mamba).** Make $B$, $C$, and the timestep **input-dependent**, so the model can choose what to keep
  and what to discard. This breaks the convolution equivalence, so Mamba uses a hardware-aware parallel scan instead.`),

    viz('ssm-vs-attention'),

    key(`The tradeoff is intrinsic and worth stating plainly:

**Attention** stores everything and can retrieve any past token exactly, at $O(n)$ cost per token.
**SSMs** compress the past into a fixed-size state, at $O(1)$ cost per token — and a fixed state **cannot** losslessly
store an unbounded past.

Mamba is measurably worse at tasks requiring exact recall of arbitrary earlier content — copying, retrieval from long
context, in-context learning that depends on precise lookup. That is not an implementation gap; it is information
theory.

Hence **hybrids**. Jamba, Samba, and Zamba interleave a few attention layers among many SSM layers. Empirically, even
one attention layer in ten recovers most of the recall ability while keeping most of the efficiency. This is where the
architecture space seems to be settling.`),

    t(`## The other efficient-attention families

- **Sparse attention** (Longformer, BigBird) — attend to a local window plus a few global tokens. Linear, and works
  well when the relevant context is local.
- **Linear attention** — drop the softmax so $(QK^{\\mathsf T})V$ can be reassociated as $Q(K^{\\mathsf T}V)$, making it
  linear. Consistently a few points worse in quality; the softmax is doing more than normalization.
- **Retention / RWKV / RetNet** — RNN-transformer hybrids with parallel training and recurrent inference.
- **FlashAttention** — not an approximation at all. Exact attention, IO-aware, and it removed the memory problem so
  effectively that much of the motivation for approximate methods evaporated.`),

    intuition(`There is a pattern here worth noting. Between roughly 2020 and 2022 dozens of "efficient transformer"
variants were published, most claiming linear complexity with comparable quality. Almost none are in use.

Two reasons. First, careful benchmarking (Long Range Arena, and follow-ups) found the quality gaps were larger than
claimed. Second, FlashAttention made exact attention fast enough that the approximations were solving a problem that
had shrunk considerably.

The lesson generalizes: **an architectural improvement must beat the well-engineered baseline, not the naive one.**
Hardware-aware implementations of simple algorithms routinely outperform clever algorithms with poor memory
behavior.`),

    code('SSM decay, and the recall tradeoff', `import numpy as np

# --- how far back can a fixed state remember? ---
print("influence of a token d steps back, by decay rate a:")
print(f"{'a':>7}" + "".join(f"{'d='+str(d):>10}" for d in [10, 50, 200, 1000]))
for a in [0.9, 0.99, 0.999, 0.9999]:
    print(f"{a:7.4f}" + "".join(f"{a**d:10.2e}" for d in [10, 50, 200, 1000]))

# --- HiPPO-style multi-scale channels ---
print("\\nmulti-scale state: 8 channels with geometrically spaced decay")
rates = 1 - np.logspace(-4, -0.5, 8)
for i, a in enumerate(rates):
    half_life = np.log(0.5)/np.log(a)
    print(f"  channel {i}: a={a:.5f}  half-life {half_life:8.1f} tokens")

# --- the recall test that separates the families ---
def copy_task(seq_len, state_dim, n_symbols=64):
    """Can a fixed-size state losslessly store a random sequence?"""
    bits_needed = seq_len * np.log2(n_symbols)
    bits_available = state_dim * 32          # float32 state
    return bits_needed, bits_available

print(f"\\n{'seq len':>9} {'bits needed':>13} {'bits in a 256-dim state':>25} {'possible?':>11}")
for L in [10, 100, 1000, 10000]:
    need, have = copy_task(L, 256)
    print(f"{L:9d} {need:13.0f} {have:25.0f} {'yes' if need <= have else 'NO':>11}")

print("\\nA transformer's KV cache grows with the sequence, so it never hits this wall.")
print("An SSM's state does not — which is both its advantage and its ceiling.")

# --- cost comparison ---
print(f"\\n{'context':>9} {'attention FLOPs':>18} {'SSM FLOPs':>14} {'KV cache (GB)':>15}")
d, L = 4096, 32
for n in [2**k for k in (10, 13, 16, 19)]:
    attn = 4*n*n*d*L
    ssm  = 20*n*d*L
    kv   = 2*L*8*128*n*2/1e9
    print(f"{n:9d} {attn:18.3e} {ssm:14.3e} {kv:15.2f}")`),

    quiz('Why do hybrid models interleave a few attention layers among many SSM layers?',
      ['A fixed-size state cannot losslessly recall arbitrary earlier tokens; a few attention layers restore exact lookup cheaply',
       'Attention layers train faster',
       'SSMs cannot be parallelized during training',
       'To reduce the parameter count'],
      0,
      'The SSM\'s constant-size state is exactly what makes it cheap and exactly what limits it — you cannot compress an unbounded past into a fixed number of bits without loss. Tasks needing verbatim recall (copying, retrieval, precise in-context lookup) suffer. A small number of attention layers provides exact lookup where it is needed, while the SSM layers carry the bulk of the sequence processing at $O(1)$ per-token cost.'),
  ],
  refs: [
    paper('Mamba: Linear-Time Sequence Modeling with Selective State Spaces', 'Gu & Dao', 2023, 'https://arxiv.org/abs/2312.00752', 'Selectivity, and the hardware-aware scan.'),
    paper('Efficiently Modeling Long Sequences with Structured State Spaces (S4)', 'Gu, Goel & Ré', 2021, 'https://arxiv.org/abs/2111.00396', 'The HiPPO-structured predecessor.'),
    paper('Jamba: A Hybrid Transformer-Mamba Language Model', 'Lieber et al.', 2024, 'https://arxiv.org/abs/2403.19887', 'Where the architecture seems to be settling.'),
    paper('Long Range Arena', 'Tay et al.', 2020, 'https://arxiv.org/abs/2011.04006', 'The benchmark that deflated many efficient-transformer claims.'),
    blog('The Annotated S4', 'Sasha Rush', 2022, 'https://srush.github.io/annotated-s4/', 'S4 built up in runnable code.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'fr-limits',
  title: 'Hallucination, Calibration, and Open Problems',
  sub: 'What these systems reliably get wrong, and what nobody has solved.',
  mins: 22, level: 'frontier',
  prereq: ['llm-evaluation', 'rl-rlhf'],
  tags: ['hallucination', 'safety', 'open problems'],
  sections: [
    t(`## Hallucination is not one bug

It is the confluence of several structural facts:

1. **The objective always demands a token.** There is no "abstain" action in next-token prediction. The model must
   produce something, and the most probable something is a fluent, plausible continuation — whether or not it is true.
2. **Training text rarely says "I don't know."** People write what they know. The distribution of expressed
   uncertainty in the corpus does not match the model's actual uncertainty.
3. **RLHF rewards confidence.** Raters prefer decisive answers, and that preference is orthogonal to correctness.
4. **There is no ground-truth signal at inference.** Nothing checks the output against reality unless you build it.`),

    viz('hallucination'),

    t(`## Calibration

A well-calibrated model is right 70% of the time when it says 70%. Base models are often startlingly well calibrated —
next-token probabilities are trained on exactly this signal. **RLHF systematically damages it**, pushing the curve
below the diagonal into overconfidence. Switch between the model stages in the figure.

What helps, in rough order of practicality:

- **Ask for probabilities explicitly** and calibrate them against a validation set.
- **Sample consistency** — sample $n$ answers; disagreement is a usable uncertainty signal.
- **Retrieval with citation requirements**, so claims are checkable against a source.
- **Explicit permission to abstain** — "if you do not know, say so" measurably reduces confabulation.
- **Verification for anything checkable** — run the code, follow the link, check the arithmetic.`),

    t(`## The open problems

An honest list of what is not solved. All of these are active research, and none has a consensus answer.

**Hallucination.** Reduced substantially by retrieval, verification, and training on abstention. Not eliminated, and
there is no known method that eliminates it.

**Prompt injection.** A model cannot reliably distinguish instructions from data, because both are text in the same
context. Every proposed defence has been bypassed. This is a genuine blocker for autonomous agents with real
permissions.

**Faithful reasoning.** A model's stated chain-of-thought is not reliably a description of the computation that
produced its answer. If you were counting on reading the reasoning to verify the conclusion, that is on shaky ground.

**Evaluation.** Contamination is pervasive, benchmarks saturate, LLM judges have biases, and human evaluation is
expensive and inconsistent. We do not have a trustworthy way to measure how good these systems actually are.

**Continual learning.** Models are frozen at their training cutoff. Fine-tuning causes catastrophic forgetting.
Retrieval patches this without solving it.

**Data limits.** High-quality text is finite and roughly exhausted at the frontier. Synthetic data works partially;
model collapse from training on generated output is a real, measured risk.

**Interpretability at scale.** SAEs and circuit analysis work but capture a fraction of a frontier model's
computation, with no ground truth to check against.

**Alignment beyond human oversight.** RLHF works because humans can judge the outputs. For tasks where they cannot —
which is the direction capabilities are heading — the approach does not obviously extend. Scalable oversight, debate,
and weak-to-strong generalization are proposals, not solutions.`),

    warn(`**On reading the field.** Progress reporting is noisy in specific ways worth compensating for: benchmark
numbers are upper bounds because of contamination; demos are selected; ablations are frequently missing; negative
results are rarely published; and the gap between a paper's headline claim and its reproducible result is often
substantial.

Useful habits: check whether a baseline was tuned as carefully as the proposed method, whether variance across seeds
is reported, whether the evaluation set was held out properly, and whether anyone independent has reproduced it.`),

    t(`## Where to keep learning

The field moves fast enough that a curriculum like this one is a foundation, not a current picture. The parts that
age well are the mathematics, the mechanics of optimization and backprop, and the way to think about evaluation.
The parts that age fast are architectures, benchmarks, and best practices.

For staying current: read the primary papers rather than summaries of them, reproduce results you find surprising,
and be suspicious of any claim you have not seen replicated. The [reference library](#/library) collects everything
cited across these lessons.`),

    code('Calibration, measured', `import numpy as np
rng = np.random.default_rng(0)

def make_model(overconfidence):
    """Returns (stated_confidence, was_correct) pairs."""
    true_p = rng.beta(2, 2, 4000)
    stated = np.clip(true_p ** (1/overconfidence), 0, 1)
    correct = rng.random(4000) < true_p
    return stated, correct

def ece(conf, correct, bins=10):
    """Expected calibration error."""
    total = 0
    edges = np.linspace(0, 1, bins+1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0: continue
        total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total

print(f"{'model':22s} {'ECE':>8} {'accuracy':>10} {'mean stated conf':>18}")
for name, oc in [("base (calibrated)", 1.0), ("mild overconfidence", 2.0),
                 ("RLHF-like", 4.0), ("extreme", 8.0)]:
    c, k = make_model(oc)
    print(f"{name:22s} {ece(c,k):8.4f} {k.mean():10.3f} {c.mean():18.3f}")

print("\\nreliability diagram for the RLHF-like model:")
c, k = make_model(4.0)
for lo in np.arange(0, 1, 0.2):
    m = (c >= lo) & (c < lo+0.2)
    if m.sum() > 20:
        gap = k[m].mean() - c[m].mean()
        bar = "#" * int(abs(gap)*60)
        print(f"  says {lo:.1f}-{lo+0.2:.1f}: actually right {k[m].mean():.3f}  "
              f"gap {gap:+.3f} {bar}")

# --- temperature scaling: the standard post-hoc fix ---
logits = np.log(c/(1-c+1e-9) + 1e-9)
best_T, best_e = 1.0, 1e9
for T in np.arange(0.5, 5, 0.05):
    p = 1/(1+np.exp(-logits/T))
    e = ece(p, k)
    if e < best_e: best_T, best_e = T, e
print(f"\\ntemperature scaling: T={best_T:.2f} reduces ECE "
      f"{ece(c,k):.4f} -> {best_e:.4f}")`),

    quiz('Why does RLHF tend to make a model less well calibrated than its base version?',
      ['Human raters prefer confident-sounding answers, so confidence gets reinforced independently of correctness',
       'RL training uses a lower learning rate',
       'The reward model is smaller than the policy',
       'Calibration is not affected by RLHF'],
      0,
      'Base models are trained purely on next-token likelihood, which rewards honest probabilities. RLHF optimizes for human preference, and humans reliably prefer decisive answers — a preference uncorrelated with truth. The optimizer finds that confidence scores well, so it produces confidence. This is measured, not speculative, and it is a clean example of an objective producing a side effect nobody asked for.'),
  ],
  refs: [
    paper('Language Models (Mostly) Know What They Know', 'Kadavath et al.', 2022, 'https://arxiv.org/abs/2207.05221', 'Models have usable internal uncertainty signals even when their outputs do not express it.'),
    paper('Survey of Hallucination in Natural Language Generation', 'Ji et al.', 2022, 'https://arxiv.org/abs/2202.03629', 'Taxonomy and mitigations.'),
    paper('Universal and Transferable Adversarial Attacks on Aligned Language Models', 'Zou et al.', 2023, 'https://arxiv.org/abs/2307.15043', 'Automated jailbreaks that transfer across models.'),
    paper('Weak-to-Strong Generalization', 'Burns et al.', 2023, 'https://arxiv.org/abs/2312.09390', 'Can a weak supervisor align a stronger model? An honest attempt at the scalable-oversight problem.'),
    paper('The Curse of Recursion: Training on Generated Data Makes Models Forget', 'Shumailov et al.', 2023, 'https://arxiv.org/abs/2305.17493', 'Model collapse.'),
    paper('On Calibration of Modern Neural Networks', 'Guo et al.', 2017, 'https://arxiv.org/abs/1706.04599', 'Temperature scaling, and the general phenomenon.'),
  ],
},

];

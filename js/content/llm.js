/* ============================================================
   Track 4 — Transformers and Large Language Models
   ============================================================ */

import { t, key, intuition, warn, hist, mathnote, viz, deriv, code, quiz, paper, book, course, blog, video, demo, codeRef,
         tldr, recap, jargon, steps, diagram } from './_helpers.js';

export default [

/* ---------------------------------------------------------- */
{
  id: 'llm-tokenization',
  title: 'Tokenization',
  sub: 'Before a model sees text, something has to chop it up. That choice leaks into everything.',
  mins: 20, level: 'core',
  prereq: ['nn-embeddings'],
  tags: ['tokenization', 'BPE'],
  sections: [
    tldr(`A language model cannot read text. It reads integers. **Tokenization** is the step that turns
\`"hello world"\` into \`[15339, 1917]\` — and it happens before the model exists, using a fixed table built
by counting character frequencies in a corpus.

It sounds like plumbing and it is not. A startling number of LLM quirks — inability to count letters, patchy
arithmetic, non-English text costing 3× more to process, entire words that make models behave bizarrely — are
tokenization artefacts rather than reasoning failures. Knowing this changes what you conclude when a model fails
at something.`),

    jargon([
      ['token', 'One unit of text the model sees. Usually a word fragment: "tokenization" might be `token` + `ization`. Roughly 0.75 words on average for English.'],
      ['vocabulary', 'The fixed list of all tokens the model knows. Typically 50,000–200,000 entries. Set before training and unchangeable afterwards.'],
      ['token id', 'The integer index of a token in the vocabulary. What actually gets fed to the model.'],
      ['subword', 'A token smaller than a word. The compromise that makes everything work: common words stay whole, rare words split into pieces.'],
      ['BPE', 'Byte-Pair Encoding. The dominant algorithm: repeatedly merge the most frequent adjacent pair of symbols.'],
      ['merge', 'One learned rule, like "`t` followed by `h` → `th`". A vocabulary is built by applying tens of thousands of these in order.'],
      ['`<UNK>`', 'The "unknown token" of older systems, used when a word was not in the vocabulary. Modern byte-level tokenizers never need it.'],
      ['byte-level', 'Starting from the 256 possible bytes rather than characters, so *any* input — emoji, Chinese, binary — encodes without failure.'],
      ['WordPiece / SentencePiece / Unigram', 'The main BPE alternatives. Used by BERT, T5, and Llama respectively.'],
      ['glitch token', 'A token that appeared in the tokenizer\'s training data but almost never in the model\'s, leaving its embedding essentially untrained. Causes strange behaviour.'],
    ]),

    t(`## The vocabulary problem

A model needs a **finite** vocabulary — a fixed list of symbols, because the output layer is a softmax with one
slot per symbol and that layer's size is baked into the architecture.

So: what should the symbols be? Two obvious choices, both bad:

- **Words.** English has hundreds of thousands, plus every name, typo, and neologism. Anything unseen becomes
  \`<UNK>\` and its information is gone.
- **Characters.** Tiny vocabulary, no unknowns — but sequences become 5× longer, and attention is quadratic in
  length. The model must also relearn that "c-a-t" is a unit, every time.

**Subword tokenization** splits the difference: frequent words stay whole, rare ones decompose into pieces.
Nothing is ever unknown, and sequences stay short.`),

    t(`## Byte-pair encoding

BPE is the algorithm that finds the subwords for you, and it is simple enough to describe completely in four
steps.`),

    steps('How BPE builds a vocabulary', [
      { h: 'Start from the smallest possible units', md: `Split all your training text into individual bytes. Your vocabulary is now 256 entries and can represent literally anything, but every word is many tokens long.` },
      { h: 'Count every adjacent pair', md: `Across the whole corpus, count how often each pair of adjacent symbols occurs. In English text, \`t\` followed by \`h\` will be near the top.` },
      { h: 'Merge the most frequent pair', md: `Add \`th\` to the vocabulary as a new single symbol, and rewrite the corpus using it. Record the merge rule — the ordered list of merges *is* the tokenizer.` },
      { h: 'Repeat 50,000 times', md: `Each merge adds one vocabulary entry and shortens the corpus a little. Frequent sequences get absorbed into single tokens; rare ones never do and stay as fragments. Nobody decided that " the" should be one token — frequency decided.` },
    ]),

    t(`That is the entire algorithm: greedy, frequency-driven merging. No linguistics, no notion of what a
morpheme is. And it is what GPT-2, GPT-4, Llama, and essentially every modern model use.`),

    viz('bpe-tokenizer'),

    t(`Slide the merge count from 0 to 60 in that figure and watch " the" collapse from four tokens into one.
Real tokenizers run 50k–200k merges.

**Why bytes rather than characters?** With a 256-byte base alphabet, *any* possible input encodes without an
unknown token — emoji, Chinese, Cyrillic, malformed UTF-8, raw binary. There is no such thing as an
out-of-vocabulary input, which eliminates an entire class of failure that plagued earlier NLP systems.

The main alternatives you will encounter: **WordPiece** (BERT) merges by likelihood gain rather than raw frequency;
**Unigram/SentencePiece** (T5, Llama) starts from a large vocabulary and prunes, and handles whitespace as a regular
character so detokenization is exactly reversible.`),

    key(`**Tokenization explains a surprising number of model failures.**

- **Spelling and character counting.** "strawberry" may be 2–3 tokens. The model never sees the letters, so counting
  the r's is genuinely hard — it is doing it from memorized facts about the *word*, not by looking.
- **Arithmetic.** If "1234" tokenizes as "123"+"4" but "1235" as "12"+"35", digit alignment is inconsistent and
  carrying is harder than it should be. Models that tokenize digits individually do noticeably better at math.
- **Non-English cost.** A tokenizer trained mostly on English spends 2–4× more tokens per word on other languages.
  That is a direct penalty in context length, latency, and API price.
- **Prompt injection via odd tokens.** Rare tokens ("glitch tokens" like \`SolidGoldMagikarp\`) appeared in the
  tokenizer's training data but almost never in the model's, so their embeddings are essentially untrained and produce
  bizarre behavior.
- **Trailing whitespace.** " the" and "the" are different tokens. A prompt ending in a space can measurably degrade
  completions.`),

    code('BPE, implemented', `# Train BPE on a small corpus and watch the vocabulary form.
from collections import Counter

corpus = ("the cat sat on the mat the cat ate the rat "
          "a rat sat on a hat the dog sat on the log") .split()

# start from characters; the leading marker keeps word boundaries visible
words = Counter("_" + w for w in corpus)
splits = {w: list(w) for w in words}

merges = []
for step in range(14):
    pairs = Counter()
    for w, freq in words.items():
        s = splits[w]
        for i in range(len(s) - 1):
            pairs[(s[i], s[i+1])] += freq
    if not pairs: break
    (a, b), count = pairs.most_common(1)[0]
    if count < 2: break
    merges.append((a, b))
    for w in words:
        s, out, i = splits[w], [], 0
        while i < len(s):
            if i + 1 < len(s) and s[i] == a and s[i+1] == b:
                out.append(a + b); i += 2
            else:
                out.append(s[i]); i += 1
        splits[w] = out
    print(f"merge {step+1:2d}: {a!r} + {b!r} -> {a+b!r}  (seen {count}x)")

print("\\nfinal tokenization:")
for w in ["_the", "_cat", "_rat", "_dog"]:
    if w in splits:
        print(f"  {w:8s} -> {splits[w]}")

def encode(word):
    s = list("_" + word)
    for a, b in merges:
        out, i = [], 0
        while i < len(s):
            if i + 1 < len(s) and s[i] == a and s[i+1] == b:
                out.append(a + b); i += 2
            else:
                out.append(s[i]); i += 1
        s = out
    return s

print("\\nunseen words still encode, just into more pieces:")
for w in ["cats", "splat"]:
    print(f"  {w:8s} -> {encode(w)}")`,
      'Watch the merge list form. The early merges are the pairs this corpus repeats most — `at`, `_t`, `_the` — and nothing in the code knows they are English, or that " the" is a word. The last block is the property that makes byte-level BPE robust: a word the tokenizer has never seen does not fail, it simply costs more tokens. There is no unknown input, only an expensive one, and the challenge turns that observation into the reason non-English text costs more to process.'),

    quiz('Why do language models struggle to count the letters in a word?',
      ['They see tokens, not characters — the letters of a multi-character token are not individually represented',
       'Their context window is too short',
       'Counting requires recursion, which transformers lack',
       'The training data contains few examples of counting'],
      0,
      'A token like "berry" arrives as a single embedding vector. There is no representation of "b, e, r, r, y" as separate items to iterate over, so the model answers from memorized facts about the word rather than by inspection. Chain-of-thought helps because spelling the word out one letter at a time forces the characters into separate tokens, where they *can* be counted.'),

    recap(`- Explain why neither words nor characters work as a vocabulary, and what subwords fix about each.
- Describe the BPE algorithm in four steps, and say what a "merge" is.
- Say why byte-level tokenization eliminates unknown tokens entirely.
- Attribute a model failure to tokenization when appropriate — letter counting, arithmetic, non-English cost —
  rather than to reasoning.
- Explain what a glitch token is and why its embedding is untrained.
- Know that a trailing space in your prompt changes the tokens and can degrade output.`),
  ],
  refs: [
    paper('Neural Machine Translation of Rare Words with Subword Units', 'Sennrich, Haddow & Birch', 2015, 'https://arxiv.org/abs/1508.07909', 'BPE applied to NMT — the paper that made subword tokenization standard.'),
    paper('SentencePiece', 'Kudo & Richardson', 2018, 'https://arxiv.org/abs/1808.06226', 'Language-independent tokenization with lossless detokenization.'),
    blog('SolidGoldMagikarp', 'Rumbelow & Watkins', 2023, 'https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation', 'Glitch tokens: what happens when a token exists in the tokenizer but not the training data.'),
    demo('Tiktokenizer', 'Diagnostics', 2023, 'https://tiktokenizer.vercel.app/', 'Paste text, see exactly how GPT models tokenize it. Worth trying with numbers and non-English.'),
    video('Let\'s build the GPT Tokenizer', 'Andrej Karpathy', 2024, 'https://www.youtube.com/watch?v=zduSFxRajkE', 'Two hours, builds BPE from scratch and walks through every failure mode above.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'llm-attention',
  title: 'Attention',
  sub: 'The mechanism. A soft, differentiable dictionary lookup.',
  mins: 30, level: 'core',
  prereq: ['nn-embeddings', 'math-jacobian'],
  tags: ['attention', 'transformers'],
  sections: [
    tldr(`This is the mechanism everything else in the track is built on, so it is worth slowing down for.

The problem: to understand "it" in a sentence, a token needs information from *some other* token — and which
one depends on the content, not on any fixed position. A convolution cannot do that (it always looks at fixed
neighbours). An RNN cannot do it well (information has to survive being squeezed through every intervening
step).

Attention's answer is a **soft dictionary lookup**. Every token broadcasts a "here is what I am" key and a
"here is what I am looking for" query. Compare each query against every key, turn the match scores into
weights, and take a weighted average. Because the weights are computed from content and are differentiable, the
model can *learn what to look for*.`),

    jargon([
      ['query, key, value', 'Three vectors each token produces. Query = what I am looking for. Key = what I advertise about myself. Value = what I hand over if selected. Borrowed from database terminology.'],
      ['attention score / logit', 'The raw match between one query and one key: their dot product. Higher means more relevant.'],
      ['attention weights', 'The scores after softmax — non-negative and summing to 1 across a row. How much of each token to blend in.'],
      ['soft lookup', 'Instead of picking one match (hard), take a weighted blend of everything (soft). Softness is what makes it differentiable and therefore learnable.'],
      ['$d_k$', 'The dimension of the query and key vectors. Appears in the $\\sqrt{d_k}$ scaling for a specific and derivable reason.'],
      ['causal mask', 'Blocking each position from seeing later positions, so the model cannot cheat while learning to predict the next token.'],
      ['head', 'One independent attention operation. Models run many in parallel, each free to track a different kind of relationship.'],
      ['multi-head attention (MHA)', 'Running $h$ heads in parallel on slices of the representation, then concatenating.'],
      ['KV cache', 'Storing past keys and values during generation so they need not be recomputed for every new token. Dominates memory at long context.'],
      ['GQA / MQA', 'Grouped-Query / Multi-Query Attention. Several query heads sharing one key/value head, shrinking the KV cache.'],
      ['FlashAttention', 'An exact reimplementation of attention that never builds the full $n \\times n$ matrix in memory. Same numbers, far less memory, faster.'],
      ['induction head', 'A learned two-head circuit that finds an earlier copy of the current token and predicts whatever followed it last time. Strongly linked to in-context learning.'],
    ]),

    t(`## The idea

Every token needs information from other tokens, and *which* ones depends on content rather than position. In
"the trophy did not fit in the suitcase because **it** was too big", resolving "it" requires reaching back to
"trophy" — seven tokens away. Change one word to "small" and it refers to the suitcase instead. No fixed wiring
can capture that.

Attention frames this as a **lookup with soft keys**. Each token produces three different vectors from its
representation, each with a distinct job:

- **Query** $\\mathbf{q}$ — *what am I looking for?* ("I am a pronoun; I need a noun to refer to.")
- **Key** $\\mathbf{k}$ — *what do I offer?* ("I am a concrete singular noun.")
- **Value** $\\mathbf{v}$ — *what do I actually pass along if selected?* (The content itself.)

Separating key from value is the subtle part and it is what makes the mechanism flexible: **what makes a token
findable need not be what it contributes.** A token can advertise "I am a date" while handing over the actual
date.

The operation is then: score every query against every key, softmax the scores into weights that sum to 1, and
return the weighted average of the values.`),

    viz('attention-basics'),

    t(`In matrix form, doing all positions at once:

$$\\text{Attention}(Q,K,V) = \\text{softmax}\\!\\left(\\frac{QK^{\\mathsf T}}{\\sqrt{d_k}}\\right)V$$

Take that apart piece by piece — it is four steps you already know:

| Piece | What it does |
|---|---|
| $Q = XW_Q,\\ K = XW_K,\\ V = XW_V$ | Three learned linear projections of the same input. The *only* parameters here. |
| $QK^{\\mathsf T}$ | Every query dotted with every key — an $n \\times n$ grid of similarity scores. |
| $/\\sqrt{d_k}$ | A scaling factor. Derived below; it is not arbitrary. |
| $\\text{softmax}(\\cdot)$ | Turns each row of scores into weights that are positive and sum to 1. |
| $\\times V$ | Weighted average of the values. The output. |

Note what is *not* in that formula: nothing about position, and no learned parameters except the three
projections. Attention is permutation-invariant by construction — which is exactly why transformers need
positional encodings bolted on separately.`),

    deriv('Why divide by $\\sqrt{d_k}$', `Suppose the entries of $\\mathbf{q}$ and $\\mathbf{k}$ are independent with mean 0 and variance 1. Then

$$\\mathbf{q}\\cdot\\mathbf{k} = \\sum_{i=1}^{d_k} q_i k_i$$

is a sum of $d_k$ independent zero-mean terms each with variance 1, so

$$\\text{Var}(\\mathbf{q}\\cdot\\mathbf{k}) = d_k, \\qquad \\text{sd} = \\sqrt{d_k}$$

With $d_k = 128$ the dot products have a standard deviation of ~11, so typical logit *gaps* are on the order of 15–20.
Softmax on logits that far apart is numerically a hard argmax: one weight ≈ 1, the rest ≈ 0.

And a saturated softmax has a **vanishing gradient** — recall $\\partial p_i/\\partial z_j = p_i(\\delta_{ij}-p_j)$, which
is ≈ 0 when any $p_i \\approx 1$. The model would be unable to learn to attend differently.

Dividing by $\\sqrt{d_k}$ restores unit variance, keeping the softmax in its responsive range. It is a one-symbol fix
for a problem that would otherwise scale with model width. ∎`),

    t(`## Reading an attention matrix

Row $i$ of the attention matrix is where token $i$ looks; it sums to 1.`),

    viz('attention-matrix'),

    t(`### The causal mask

The **causal mask** sets the upper triangle of the score matrix to $-\\infty$ *before* the softmax — and since
$e^{-\\infty} = 0$, those positions get exactly zero weight.

Why this is needed comes down to a training efficiency trick. During training the model predicts the next token
at *every* position simultaneously, in one forward pass — otherwise you would need a separate pass per position
and training would be hundreds of times slower. But that means position 5 is computing its prediction while
tokens 6, 7, 8 are sitting right there in the same sequence. Without a mask, predicting "the next token" would
be a lookup rather than a prediction, and the model would learn nothing and then fail completely at generation
time when the future is genuinely absent.

Toggle the mask off and you have an **encoder** (BERT-style): every position sees the whole sequence. That is
the right choice for *understanding* tasks — classification, retrieval — where you have the full text in hand
and no need to generate.

The named patterns in that figure are idealized versions of heads that interpretability research actually finds inside
trained models. **Induction heads** are the most consequential: a two-head circuit that finds a previous occurrence of
the current token and copies whatever followed it. They form abruptly during training, and their formation coincides
with the emergence of in-context learning.`),

    t(`## Multi-head attention

One attention operation computes one kind of relationship. Run $h$ of them in parallel on slices of the representation:

$$\\text{MHA}(X) = \\text{Concat}(\\text{head}_1,\\ldots,\\text{head}_h)W_O, \\qquad \\text{head}_i = \\text{Attention}(XW_Q^i, XW_K^i, XW_V^i)$$

Crucially the heads **split** $d_{\\text{model}}$ rather than adding to it — each head works in $d_{\\text{model}}/h$
dimensions — so multi-head costs the same as single-head but can attend to several things at once.`),

    viz('multi-head'),

    t(`### The KV cache, and why GQA exists

Generation is sequential: you produce one token, append it, and run the model again. Naively that recomputes the
keys and values for every previous token at every step — enormously wasteful, since those tokens have not
changed.

So you cache them. The **KV cache** stores every past token's keys and values, and each new token only computes
its own. This turns generation from quadratic to linear in total work, and it is why every inference server does
it.

The cost is memory, and at long context that cost dominates everything else — the cache grows with (batch ×
sequence length × layers × heads × head dimension), and for a long conversation it can exceed the model weights
themselves.

**GQA** (grouped-query attention) attacks this directly: let several query heads *share* one key/value head. With
8 query heads per KV head, the cache shrinks 8×. The quality cost is small, apparently because the queries
carried most of the useful diversity anyway. **MQA** pushes it to a single shared KV head — maximum saving,
slightly more quality loss. Nearly every modern open model uses GQA.`),

    t(`## The cost`),

    viz('attention-cost'),

    warn(`The famous $O(n^2)$ is real but routinely misunderstood. For a 4096-wide model, attention only overtakes the
MLP in FLOPs beyond about 12k tokens — below that the feed-forward layers dominate compute.

What *was* catastrophic is the $n^2$ **memory** for the attention matrix. **FlashAttention** removed it by tiling the
computation and never materializing the full matrix, recomputing pieces during the backward pass instead. Same math,
exact same outputs, $O(n)$ memory, and faster in wall-clock because it is IO-aware rather than FLOP-bound.`),

    code('Attention from scratch', `import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def attention(X, Wq, Wk, Wv, causal=True):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)                   # (T, T)
    if causal:
        T = scores.shape[0]
        scores = np.where(np.tril(np.ones((T, T))) == 1, scores, -np.inf)
    A = softmax(scores)
    return A @ V, A

rng = np.random.default_rng(0)
T, d, dk = 6, 32, 16
X = rng.normal(size=(T, d))
Wq, Wk, Wv = (rng.normal(0, d**-0.5, (d, dk)) for _ in range(3))

out, A = attention(X, Wq, Wk, Wv)
print("attention matrix (rows sum to 1, upper triangle masked):")
print(np.round(A, 3))
print("\\nrow sums:", A.sum(1).round(6))

# --- what happens without the 1/sqrt(d_k) scaling ---
for dk_test in [16, 128, 1024]:
    q = rng.normal(size=dk_test); k = rng.normal(size=(8, dk_test))
    raw = k @ q
    print(f"\\nd_k={dk_test:5d}  logit sd={raw.std():7.2f}")
    print(f"  unscaled softmax max weight: {softmax(raw).max():.4f}")
    print(f"  scaled   softmax max weight: {softmax(raw/np.sqrt(dk_test)).max():.4f}")
print("\\nUnscaled, large d_k saturates the softmax -> no gradient -> no learning.")`),

    quiz('Why must the attention matrix rows sum to 1?',
      ['Softmax makes the output a convex combination of value vectors, keeping its scale independent of sequence length',
       'It is required for the gradient to exist',
       'To make the matrix invertible',
       'It is a convention with no functional consequence'],
      0,
      'The output is $\\sum_j a_{ij}\\mathbf v_j$ with $\\sum_j a_{ij}=1$ — a weighted average, so it stays in the same range as the values regardless of whether you attend over 10 tokens or 100,000. Without normalization the output magnitude would grow with sequence length and destabilize every downstream layer.'),

    recap(`- Explain query, key, and value in plain language, and say why key and value are kept separate.
- Walk through the attention formula one factor at a time and say what each does.
- Derive the $\\sqrt{d_k}$ scaling from the variance of a dot product, and say what breaks without it.
- Explain what the causal mask prevents and why training would otherwise be trivially easy and useless.
- Say why multi-head attention costs the same as single-head despite running $h$ operations.
- Explain what the KV cache stores, why it dominates long-context memory, and what GQA trades away.
- State what FlashAttention changed — and, importantly, what it did *not* change (the maths).`),
  ],
  refs: [
    paper('Attention Is All You Need', 'Vaswani et al.', 2017, 'https://arxiv.org/abs/1706.03762', 'The transformer. Among the most consequential papers in the field.'),
    blog('The Illustrated Transformer', 'Jay Alammar', 2018, 'https://jalammar.github.io/illustrated-transformer/', 'The diagrams that taught a generation. Read this alongside the paper.'),
    paper('FlashAttention', 'Dao et al.', 2022, 'https://arxiv.org/abs/2205.14135', 'IO-aware exact attention. Same math, dramatically less memory.'),
    paper('GQA: Training Generalized Multi-Query Transformer Models', 'Ainslie et al.', 2023, 'https://arxiv.org/abs/2305.13245', 'The KV-cache/quality tradeoff that every modern model now takes.'),
    paper('In-context Learning and Induction Heads', 'Olsson et al.', 2022, 'https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/', 'The mechanism behind in-context learning, found inside real models.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'llm-transformer',
  title: 'The Transformer Block',
  sub: 'Attention plus an MLP plus residuals — assembled, and every modern variation on it.',
  mins: 28, level: 'core',
  prereq: ['llm-attention', 'nn-normalization'],
  tags: ['transformers', 'architecture'],
  sections: [
    tldr(`A transformer is one small block, repeated. That is genuinely the whole architecture — GPT-4 and a
tutorial implementation differ in size and detail, not in structure.

The block has exactly two moving parts. **Attention** lets tokens exchange information with each other. An
**MLP** processes each token on its own. Both are wrapped in residual connections, so nothing overwrites the
running state — layers only ever *add* to it.

The single most useful mental model: think of the flowing representation as a **shared bus** that every layer
reads from and writes to, rather than a pipeline that transforms and hands along. Interpretability research is
built on that picture and it explains a lot of otherwise-strange behaviour.`),

    jargon([
      ['block / layer', 'One attention sublayer plus one MLP sublayer, with residuals and normalization. A "40-layer model" means 40 of these stacked.'],
      ['sublayer', 'One of the two components inside a block.'],
      ['MLP / FFN', 'The position-wise feed-forward network: two linear layers with a nonlinearity between. "Feed-forward network" is the same thing.'],
      ['position-wise', 'Applied to each token independently, with the same weights. The MLP has no idea other tokens exist.'],
      ['residual stream', 'The running vector carried through the network by the residual connections. The "bus" every layer reads and writes.'],
      ['$d_{\\text{model}}$', 'The width of the residual stream — how many numbers represent each token. 768 for GPT-2 small, 4096+ for large models.'],
      ['permutation-equivariant', 'Shuffle the inputs and the outputs shuffle identically. True of attention, which is why position must be added separately.'],
      ['positional encoding', 'Information about *where* a token sits, injected because attention cannot tell.'],
      ['RoPE', 'Rotary Position Embedding. Encodes position by rotating query and key vectors, so only relative distance matters. The current default.'],
      ['ALiBi', 'Adding a distance-based penalty directly to attention scores. Extrapolates to longer sequences than trained on.'],
      ['pre-norm / post-norm', 'Whether normalization comes before a sublayer or after the residual add. Pre-norm is what makes very deep models trainable.'],
    ]),

    t(`## The block

A transformer is one block repeated $L$ times. Modern (pre-norm) form:

$$\\begin{aligned}
\\mathbf{x} &\\leftarrow \\mathbf{x} + \\text{MHA}(\\text{Norm}(\\mathbf{x}))\\\\
\\mathbf{x} &\\leftarrow \\mathbf{x} + \\text{MLP}(\\text{Norm}(\\mathbf{x}))
\\end{aligned}$$

Two sublayers, each wrapped in a residual connection, each preceded by normalization. That is the whole architecture.

The **MLP** is position-wise — the same two-layer network applied independently at every position:

$$\\text{MLP}(\\mathbf{x}) = W_2\\,\\phi(W_1\\mathbf{x})$$

with a hidden dimension typically $4\\times d_{\\text{model}}$.`),

    key(`**Attention moves information between positions. The MLP processes it within a position.**

That division of labour is worth memorising, because it tells you where to look when reasoning about a model.
Attention is the *only* place in the entire architecture where tokens exchange information — remove it and each
token is processed in complete isolation, as if the others did not exist. Everything else operates per-token.`),

    intuition(`The residual stream is the other half of the picture, and reframing it is worth doing
deliberately.

The naive reading of $\\mathbf{x} \\leftarrow \\mathbf{x} + \\text{MHA}(\\ldots)$ is "a layer transforms the
representation and passes it on". A better reading: $\\mathbf{x}$ is a **shared bus**. Each layer *reads* from
it, computes something, and *adds* its contribution back. Nothing is ever overwritten.

Three consequences that this picture explains and the pipeline picture does not:

- **Layers can communicate at a distance.** Layer 3 can write into some subspace that layer 27 reads from,
  with the twenty-three layers in between neither knowing nor caring.
- **Layers can abstain.** Contributing approximately zero is a perfectly good option, and analysis shows many
  layers do exactly that on many inputs. This is why you can often prune or skip layers with little damage.
- **The stream grows.** Since everything adds, the magnitude of $\\mathbf{x}$ increases with depth — which is
  why a final normalization before the output head is standard.

This "residual stream" framing is the foundation of mechanistic interpretability, and it is the reason people
talk about models as having *circuits* spread across layers rather than a sequence of processing stages.`),

    t(`## Where the parameters go

Per block, with hidden dimension $4d$:

| Component | Parameters |
|---|---|
| $W_Q, W_K, W_V, W_O$ | $4d^2$ |
| MLP ($W_1, W_2$) | $8d^2$ |
| **Total** | $\\approx 12d^2$ |

So a model is roughly $12Ld^2$ parameters plus embeddings. Note the MLP holds **two-thirds** of them — attention gets
the attention, but most of the capacity is in the feed-forward layers, and interpretability work increasingly suggests
that is where factual knowledge lives.`),

    viz('transformer-calculator'),

    t(`## Positional information

Go back and look at the attention formula: there is nothing in it that refers to position. Every token is
compared against every other by *content only*. So attention is **permutation-equivariant** — shuffle the input
tokens and the outputs shuffle identically, with no other change.

Which means, on its own, a transformer cannot distinguish "dog bites man" from "man bites dog". Order has to be
injected explicitly, and how to do it well took the field several attempts.`),

    viz('positional-encoding'),

    t(`The lineage:

- **Sinusoidal** (original) — fixed sin/cos at geometrically spaced frequencies. Like a binary clock.
- **Learned absolute** (BERT, GPT-2) — a lookup table. Simple, but there is no row for position 5000 if you trained to
  4096. Hard length limit.
- **RoPE** (now dominant) — rotate $\\mathbf{q}$ and $\\mathbf{k}$ by an angle proportional to their position:
  the vector at position $m$ is turned by $m\\theta$. Here is why that gives relative position for free. Rotating
  two vectors by the *same* angle leaves the angle between them unchanged, so rotating $\\mathbf{q}$ by $m\\theta$
  and $\\mathbf{k}$ by $n\\theta$ leaves a dot product that depends only on the *difference* in how far each was
  turned — that is, on $m - n$. Tokens 5 apart look 5 apart wherever they sit in the sequence. No parameters, and
  nothing added to the embedding to dilute its content.
- **ALiBi** — add a distance-proportional penalty straight into the attention scores. Extrapolates well.`),

    viz('rope-rotation'),

    t(`## The modern block, annotated

Comparing "Attention Is All You Need" (2017) to a current model:

| | 2017 | Now |
|---|---|---|
| Norm placement | post-norm | **pre-norm** (trains deeper without warmup) |
| Norm type | LayerNorm | **RMSNorm** (cheaper, equivalent) |
| Position | sinusoidal | **RoPE** |
| MLP activation | ReLU | **SwiGLU** |
| Attention | MHA | **GQA** |
| Bias terms | yes | **usually removed** (no measurable loss) |
| Structure | encoder–decoder | **decoder-only** |

Each change is worth a fraction of a percent. Together they are worth a lot, and none of them altered the fundamental
shape: attention, MLP, residual, repeat.`),

    hist(`The original transformer was an encoder–decoder for translation. Three lineages split off:

- **Encoder-only** (BERT, 2018) — bidirectional, masked-language-model pretraining. Great for classification and
  embedding, cannot generate.
- **Decoder-only** (GPT, 2018) — causal, next-token prediction. Won, decisively.
- **Encoder–decoder** (T5, BART) — still used for translation and some structured tasks.

Decoder-only won partly because next-token prediction is a task you can run on all text with no labels, and partly
because one model that generates can be prompted into any task, while a classifier can only classify.`),

    code('A transformer block, forward pass', `import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)

def rmsnorm(x, g, eps=1e-6):
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps) * g

def rope(x, base=10000):
    """Rotate pairs of dimensions by an angle proportional to position."""
    T, d = x.shape
    pos = np.arange(T)[:, None]
    i = np.arange(0, d, 2)[None, :]
    theta = pos / base ** (i / d)
    c, s = np.cos(theta), np.sin(theta)
    out = x.copy()
    out[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s
    out[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c
    return out

def block(x, P, n_heads=4):
    T, d = x.shape
    dh = d // n_heads

    # ---- attention sublayer ----
    h = rmsnorm(x, P["g1"])
    Q, K, V = h @ P["Wq"], h @ P["Wk"], h @ P["Wv"]
    out = np.zeros_like(Q)
    for i in range(n_heads):
        sl = slice(i*dh, (i+1)*dh)
        q, k, v = rope(Q[:, sl]), rope(K[:, sl]), V[:, sl]
        sc = q @ k.T / np.sqrt(dh)
        sc = np.where(np.tril(np.ones((T, T))) == 1, sc, -np.inf)
        out[:, sl] = softmax(sc) @ v
    x = x + out @ P["Wo"]                       # residual

    # ---- MLP sublayer (SwiGLU) ----
    h = rmsnorm(x, P["g2"])
    a, b = h @ P["W1"], h @ P["W3"]
    swish = a / (1 + np.exp(-a))
    x = x + (swish * b) @ P["W2"]               # residual
    return x

rng = np.random.default_rng(0)
T, d, dff = 8, 64, 176                          # 176 ~ (2/3)*4d, the SwiGLU convention
P = {"g1": np.ones(d), "g2": np.ones(d)}
for n, shape in [("Wq",(d,d)), ("Wk",(d,d)), ("Wv",(d,d)), ("Wo",(d,d)),
                 ("W1",(d,dff)), ("W3",(d,dff)), ("W2",(dff,d))]:
    P[n] = rng.normal(0, shape[0]**-0.5, shape)

x = rng.normal(size=(T, d))
for layer in range(6):
    x = block(x, P)
    print(f"after layer {layer+1}: residual stream norm = {np.linalg.norm(x, axis=-1).mean():.3f}")

print("\\nNote the norm growing with depth — that is why a final norm before")
print("the output head is standard in pre-norm architectures.")`),

    quiz('In a pre-norm transformer, why is a final normalization layer applied before the output head?',
      ['Each block adds to the residual stream without rescaling it, so its magnitude grows with depth',
       'To make the softmax numerically stable',
       'Because RMSNorm requires it',
       'It is optional and mostly historical'],
      0,
      'Pre-norm normalizes the *input* to each sublayer, but the residual stream itself is only ever added to — nothing rescales it. Its magnitude therefore grows roughly with $\\sqrt L$. Without a final norm, the output head would see inputs whose scale depends on depth. You can watch this in the code above.'),

    recap(`- Write out a transformer block from memory and name what each of the two sublayers is responsible for.
- Explain the residual stream as a shared bus, and give two things that picture predicts.
- Say where two-thirds of a transformer's parameters live, and why that is surprising given what gets discussed.
- Explain why positional information must be added separately, from the structure of the attention formula.
- Say what RoPE encodes and why rotation makes the result depend only on relative distance.
- List three changes between the 2017 transformer and a current one, and say what each fixed.`),
  ],
  refs: [
    paper('Attention Is All You Need', 'Vaswani et al.', 2017, 'https://arxiv.org/abs/1706.03762', ''),
    paper('RoFormer: Enhanced Transformer with Rotary Position Embedding', 'Su et al.', 2021, 'https://arxiv.org/abs/2104.09864', 'RoPE.'),
    paper('Train Short, Test Long (ALiBi)', 'Press, Smith & Lewis', 2021, 'https://arxiv.org/abs/2108.12409', 'Positional bias that extrapolates.'),
    paper('LLaMA: Open and Efficient Foundation Language Models', 'Touvron et al.', 2023, 'https://arxiv.org/abs/2302.13971', 'The reference modern decoder-only recipe: RMSNorm, RoPE, SwiGLU, no biases.'),
    blog('The Annotated Transformer', 'Harvard NLP', 2018, 'https://nlp.seas.harvard.edu/annotated-transformer/', 'The paper, line by line, as runnable code.'),
    video('Let\'s build GPT: from scratch, in code, spelled out', 'Andrej Karpathy', 2023, 'https://www.youtube.com/watch?v=kCc8FmEb1nY', 'Builds a working GPT in two hours. The single best way to make this concrete.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'llm-pretraining',
  title: 'Language Modeling and Pretraining',
  sub: 'Predict the next token, at scale, and something unreasonable happens.',
  mins: 24, level: 'core',
  prereq: ['llm-transformer', 'math-information'],
  tags: ['pretraining', 'language modeling'],
  sections: [
    tldr(`The training objective for every large language model is one sentence: **predict the next token.** No
labels, no task definitions, no human annotation — just text, and the question "what comes after this?"

The surprising part is not that it works, but what falls out of it. To predict text well you have to model
whatever produced that text, and text was produced by people reasoning, computing, remembering, and arguing. So
a system optimised purely for compression ends up doing all of those things as a side effect.

Whether that constitutes understanding is genuinely contested. That it produces capability is not.`),

    jargon([
      ['pretraining', 'The initial, enormous training run on raw text. Everything afterwards — instruction tuning, RLHF — is comparatively tiny.'],
      ['next-token prediction', 'The objective: given all previous tokens, put a probability on each possible next one.'],
      ['self-supervised', 'Learning from labels the data supplies itself. The "label" here is just the next token, which is free.'],
      ['chain rule (of probability)', 'Factoring $p(\\text{whole sequence})$ into a product of one-token-at-a-time conditionals. Different from calculus\' chain rule.'],
      ['$x_{<t}$', 'All tokens before position $t$. The context the model conditions on.'],
      ['corpus', 'The body of training text. Frontier runs use roughly $10^{13}$ tokens.'],
      ['Common Crawl', 'A free periodic scrape of the public web. The bulk of most pretraining corpora, and mostly junk before filtering.'],
      ['deduplication', 'Removing repeated documents and passages. Duplicated text causes memorization and wastes compute.'],
      ['decontamination', 'Removing benchmark test sets from training data, so evaluation scores mean something. Done imperfectly by everyone.'],
      ['data mixture', 'The proportions of code, web text, books, and languages. Tuned empirically, and models are sensitive to it.'],
      ['emergent capability', 'A skill absent in smaller models that appears above some scale. Real, though partly an artefact of how the metric is measured.'],
      ['model collapse', 'Degradation from training on the outputs of other models, round after round, until diversity is lost.'],
    ]),

    t(`## The objective

Start with what we want: a probability for an entire sequence, $p(x_1,\\ldots,x_T)$. That is a distribution over
every possible document, which is not something you can write down directly.

The chain rule of probability breaks it into pieces that are individually manageable:

$$p(x_1,\\ldots,x_T) = p(x_1)\\cdot p(x_2\\mid x_1)\\cdot p(x_3 \\mid x_1,x_2)\\cdots = \\prod_{t=1}^{T}p(x_t\\mid x_{<t})$$

This is exact, not an approximation — it is just "the probability of the whole thing equals the probability of
the first bit, times the probability of the next given the first, and so on." Each factor is a much smaller
question: *given everything so far, what comes next?*

Maximising the log of that product (logs, for the [same reasons as always](#/l/math-probability)) gives the
training loss — which is exactly cross-entropy against the token that actually occurred, averaged over every
position:

$$\\mathcal{L} = -\\frac{1}{T}\\sum_{t=1}^{T}\\log p_\\theta(x_t\\mid x_{<t})$$

That is the entire pretraining objective. Every frontier model costing hundreds of millions of dollars is
minimising this one expression. No labels, no task, no human input — just: what comes next.`),

    viz('language-model'),

    t(`The figure above is a genuine character-level n-gram model. Raise the context length and generations get more
fluent *and* more plagiarized — because longer contexts appear fewer times, and at order 6 most have exactly one
continuation. **That trade-off is the whole motivation for neural language models**: they generalize across *similar*
contexts instead of memorizing exact strings.`),

    key(`Why does next-token prediction produce so much capability? Because doing it well requires everything else.

To predict the last token of "The murderer was, in fact, the ___" you need to have tracked the plot. To finish
"import numpy as np; a = np.array([1,2,3]); print(a.sum()) # ___" you need to execute Python. To complete a proof you
need to do mathematics.

Compression and understanding converge. A model that predicts text perfectly must have modeled whatever process
generated it — and that process is people, thinking.`),

    t(`## Data

Pretraining corpora are on the order of $10^{13}$ tokens, and the pipeline matters as much as the architecture:

- **Sources** — Common Crawl (the bulk), curated web (Wikipedia, arXiv, StackExchange), books, code. Code is included
  even for non-coding models; it appears to improve reasoning.
- **Quality filtering** — classifiers trained to recognize "high-quality" text, perplexity filters, heuristics. This
  is where much of the differentiation between labs lives, and it is rarely published.
- **Deduplication** — near-duplicate removal at document and substring level. Duplicated data causes memorization and
  wastes compute; dedup consistently improves results per token.
- **Decontamination** — removing benchmark test sets. Done imperfectly, which is a real problem for interpreting
  published evaluation numbers.
- **Mixing** — how much code, how much of each language, sampling weights across sources. Tuned empirically, and models
  are sensitive to it.`),

    warn(`**The public web is finite.** High-quality text is estimated in the low tens of trillions of tokens, and
frontier runs are already there. Responses: multi-epoch training (works for a few epochs, then degrades), synthetic
data (works, with model-collapse risk if done naively), and shifting the marginal compute from pretraining toward
post-training and inference-time reasoning. This constraint is shaping the field's direction right now.`),

    t(`## Practical mechanics

- **Batch size** in the millions of tokens, assembled from many sequences packed to a fixed length.
- **AdamW**, $\\beta_2 = 0.95$ (lower than the 0.999 default — large-batch gradients are less noisy), weight decay 0.1.
- **Cosine schedule** decaying to ~10% of peak, with a few thousand warmup steps.
- **Gradient clipping** at global norm 1.0. Non-negotiable; a single bad batch otherwise ends the run.
- **bf16** with fp32 master weights.
- **One epoch**, usually. With trillions of unique tokens there is nothing to overfit to — which is why dropout is
  typically 0.
- **Checkpoint constantly.** Runs take months and hardware fails. Loss spikes are routine; the standard response is to
  roll back a few thousand steps and skip the offending data.`),

    code('Training a tiny language model', `import numpy as np

text = ("the quick brown fox jumps over the lazy dog. "
        "the lazy dog sleeps. the quick fox runs. "
        "a brown dog jumps over a lazy fox. ") * 40

chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
V, CTX, D = len(chars), 8, 24
data = np.array([stoi[c] for c in text])

rng = np.random.default_rng(0)
E  = rng.normal(0, 0.1, (V, D))            # token embeddings
Pe = rng.normal(0, 0.1, (CTX, D))          # learned positional embeddings
W1 = rng.normal(0, (CTX*D)**-0.5, (CTX*D, 64))
W2 = rng.normal(0, 64**-0.5, (64, V))

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

def batch(n=64):
    i = rng.integers(0, len(data) - CTX - 1, n)
    X = np.stack([data[j:j+CTX] for j in i])
    y = np.array([data[j+CTX] for j in i])
    return X, y

lr = 0.2
for step in range(3001):
    X, y = batch()
    h0 = (E[X] + Pe[None]).reshape(len(X), -1)      # embed + position, flatten
    h1 = np.maximum(0, h0 @ W1)
    logits = h1 @ W2
    p = softmax(logits)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()

    dlogits = p.copy(); dlogits[np.arange(len(y)), y] -= 1; dlogits /= len(y)
    dW2 = h1.T @ dlogits
    dh1 = (dlogits @ W2.T) * (h1 > 0)
    dW1 = h0.T @ dh1
    dh0 = (dh1 @ W1.T).reshape(len(X), CTX, D)
    W2 -= lr*dW2; W1 -= lr*dW1
    np.add.at(E, X, -lr*dh0)
    Pe -= lr*dh0.sum(0)

    if step % 750 == 0:
        print(f"step {step:4d}  loss {loss:.4f}  perplexity {np.exp(loss):6.2f}")

ctx = [stoi[c] for c in "the quic"]
out = "the quic"
for _ in range(60):
    x = np.array(ctx[-CTX:])
    h = np.maximum(0, (E[x] + Pe).reshape(1, -1) @ W1)
    p = softmax(h @ W2)[0]
    nxt = rng.choice(V, p=p)
    out += chars[nxt]; ctx.append(nxt)
print("\\ngenerated:", out)`,
      'A complete language model: embed, positional-encode, one hidden layer, predict, sample. No attention — this is the pre-transformer architecture (Bengio et al. 2003). Watch perplexity fall from ~30 toward ~4.'),

    quiz('Why does a language model trained only to predict the next token acquire abilities like arithmetic and translation?',
      ['Those abilities are required to predict text well, so the objective forces the model to acquire them',
       'They are explicitly included as auxiliary training objectives',
       'They emerge from the attention mechanism specifically',
       'They are fine-tuned in afterwards'],
      0,
      'The training text contains arithmetic, translations, code, and arguments. Minimizing prediction loss on that text is only possible by modeling the processes that produced it. The objective is narrow; the data is not, and capability comes from the data. This is the core insight — sometimes phrased as "compression is understanding."'),

    recap(`- Write the pretraining objective and explain the chain-rule factorization it comes from.
- Explain why a narrow objective produces broad capability, in terms of what the *data* contains.
- Name the five stages of a data pipeline and say what each is protecting against.
- Explain why decontamination failures make published benchmark numbers hard to interpret.
- State the data-supply problem and name the three responses the field is trying.`),
  ],
  refs: [
    paper('Language Models are Unsupervised Multitask Learners', 'Radford et al.', 2019, 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf', 'GPT-2. The argument that task-specific training is unnecessary.'),
    paper('The Pile', 'Gao et al.', 2020, 'https://arxiv.org/abs/2101.00027', 'An open pretraining corpus, documented in detail. Read it to see what actually goes into one.'),
    paper('Deduplicating Training Data Makes Language Models Better', 'Lee et al.', 2021, 'https://arxiv.org/abs/2107.06499', ''),
    paper('Will we run out of data?', 'Villalobos et al.', 2022, 'https://arxiv.org/abs/2211.04325', 'Quantifies the high-quality text supply against projected demand.'),
    paper('A Neural Probabilistic Language Model', 'Bengio et al.', 2003, 'https://www.jmlr.org/papers/v3/bengio03a.html', 'The architecture in the code block above, twenty years early.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'llm-scaling',
  title: 'Scaling Laws',
  sub: 'Loss is a predictable power law in compute — which is why anyone dared spend a billion dollars.',
  mins: 22, level: 'core',
  prereq: ['llm-pretraining'],
  tags: ['scaling', 'Chinchilla'],
  sections: [
    tldr(`Here is the finding that made the last five years possible: **model loss is predictable.** Not roughly,
not qualitatively — it follows a clean power law in model size, data size, and compute, holding over many orders
of magnitude.

That predictability is the entire reason anyone was willing to spend a hundred million dollars on a single
training run. You can train a series of small cheap models, fit a curve, and know what the enormous one will
achieve before you start it. Without scaling laws, frontier AI would be gambling; with them, it is capital
allocation.

The second half of the lesson is about **how to spend** a fixed budget: more parameters or more data? The
field's first answer was wrong, and correcting it in 2022 changed how every model since has been trained.`),

    jargon([
      ['scaling law', 'An empirical formula predicting model loss from size, data, and compute. Discovered by measurement, not derived from theory.'],
      ['power law', 'A relationship of the form $y = x^{-\\alpha}$ — a straight line on a log-log plot. Means steady proportional returns, never a cliff.'],
      ['$N$, $D$, $C$', 'Number of parameters, number of training tokens, total compute. The three quantities everything is expressed in.'],
      ['FLOPs', 'Floating-point operations — the unit of compute. Training a frontier model is roughly $10^{25}$ of them.'],
      ['irreducible loss $E$', 'The floor: language\'s own entropy. A perfect model still pays it, because text is not deterministic.'],
      ['compute-optimal', 'The allocation of a fixed compute budget between model size and data that minimises loss. What "Chinchilla-optimal" refers to.'],
      ['Chinchilla', 'The 2022 DeepMind paper that corrected the scaling recipe to roughly 20 tokens per parameter.'],
      ['tokens per parameter', 'The ratio $D/N$. Chinchilla says ~20 for training-optimal; deployment often justifies far more.'],
      ['emergence', 'A capability appearing abruptly above some scale rather than improving gradually. Partly real, partly a measurement artefact.'],
      ['overtraining', 'Deliberately training past the compute-optimal point to get a smaller model of the same quality. Standard practice for models that will be served a lot.'],
    ]),

    t(`## The empirical finding

Test loss falls as a **power law** in model size $N$, dataset size $D$, and compute $C$, and does so over many
orders of magnitude with no sign of a cliff:

$$L(N, D) = \\underbrace{E}_{\\text{the floor}} + \\underbrace{\\frac{A}{N^{\\alpha}}}_{\\text{model too small}} + \\underbrace{\\frac{B}{D^{\\beta}}}_{\\text{data too little}}$$

Three terms, each with a clean interpretation. $E$ is the **irreducible entropy of language itself** — no model
beats it, because text genuinely is not deterministic ([the same $H(p)$](#/l/math-information) from the
information lesson). The other two are the penalties for having a finite model and finite data, and both shrink
toward zero as you scale.

Now look at the exponents, which are fitted from experiments: around $\\alpha = 0.34$ and $\\beta = 0.28$. Those
are *small*, and small exponents mean expensive progress — halving the reducible loss costs roughly **8× the
compute**. There is no shortcut hiding in the curve.

But the practical significance is not the rate, it is the **reliability**. Because the relationship is a clean
power law, you can train a ladder of small models, fit the curve, and extrapolate 1000× beyond anything you have
run. Being able to predict the outcome of a $100M experiment before running it is what turned frontier-scale
spending from a gamble into an investment case.`),

    viz('scaling-laws'),

    t(`## Kaplan vs Chinchilla

**Kaplan et al. (2020)** concluded that model size should grow much faster than data. The field followed, producing
GPT-3 (175B parameters, 300B tokens) and a generation of models trained the same way.

**Hoffmann et al. (2022)** — "Chinchilla" — redid the analysis with the learning-rate schedule handled correctly, and
found $N$ and $D$ should scale **roughly equally**: about **20 tokens per parameter**.

The demonstration was blunt: Chinchilla (70B params, 1.4T tokens) beat Gopher (280B params, 300B tokens) on nearly
everything, using the same compute and being 4× cheaper to run.

Switch the figure above to the allocation view: both curves have slope ≈ 0.5. Double the compute and you should grow
the model and the data by ~$\\sqrt2$ each.`),

    key(`**Chinchilla-optimal is not deployment-optimal.** It minimizes loss for a fixed *training* budget and ignores
inference entirely. If you will serve a model billions of times, a smaller model trained far past 20 tokens/parameter
is better: slightly worse loss, much cheaper forever.

This is why Llama-3-8B saw 15T tokens — nearly 2000 tokens per parameter, a hundred times "optimal." It is not a
mistake; it is optimizing a different objective. The correct frame is total cost of ownership, not training FLOPs.`),

    t(`## Emergence, and the argument about it

Some capabilities appear to jump discontinuously with scale — three-digit arithmetic, chain-of-thought benefiting
rather than hurting, certain reasoning benchmarks.

Schaeffer et al. (2023) argued much of this is a **measurement artifact**: metrics like exact-match are discontinuous
by construction, and switching to a smooth metric (token-level log-likelihood, edit distance) often reveals gradual
improvement underneath. The underlying capability was rising smoothly; the *scoring* was a step function.

The honest position: some apparent emergence is metric artifact, and it is genuinely unresolved whether all of it is.
Either way, the practical problem stands — you cannot always predict from small models which capabilities a large one
will have, which is a real difficulty for safety evaluation.`),

    t(`## The shifting frontier

The pure scaling story has been complicated by three things:

1. **Data is finite.** See the previous lesson. Compute can keep growing; unique high-quality tokens cannot.
2. **Post-training gives large gains cheaply.** RLHF and reasoning-focused RL move benchmarks far more per FLOP than
   additional pretraining.
3. **Inference-time compute has its own scaling law.** Letting a model think longer, or sample many times and select,
   buys accuracy that would otherwise require a much larger model. This is the axis frontier labs are currently
   pushing hardest.`),

    code('Fitting and extrapolating a scaling law', `import numpy as np

# Chinchilla's fitted parametric form
E, A, alpha, B, beta = 1.69, 406.4, 0.34, 410.7, 0.28
L = lambda N, D: E + A/N**alpha + B/D**beta
C = lambda N, D: 6 * N * D

def optimal(budget):
    """Best (N, D) split for a given compute budget."""
    best = (np.inf, 0, 0)
    for lgN in np.arange(7, 13, 0.005):
        N = 10**lgN
        D = budget / (6*N)
        if D < 1e8: continue
        l = L(N, D)
        if l < best[0]: best = (l, N, D)
    return best

def fmt(x):
    for div, s in [(1e12,'T'), (1e9,'B'), (1e6,'M')]:
        if x >= div: return f"{x/div:.1f}{s}"
    return f"{x:.0f}"

print(f"{'budget':>10} {'params':>9} {'tokens':>9} {'tok/param':>10} {'loss':>7}")
for lgC in [20, 21, 22, 23, 24, 25]:
    loss, N, D = optimal(10**lgC)
    print(f"  1e{lgC:<7} {fmt(N):>9} {fmt(D):>9} {D/N:>10.1f} {loss:>7.4f}")

print("\\n--- real models against the optimum ---")
for name, N, D in [("GPT-3", 175e9, 300e9), ("Chinchilla", 70e9, 1.4e12),
                   ("Llama-3-8B", 8e9, 15e12), ("Llama-3-70B", 70e9, 15e12)]:
    budget = C(N, D)
    opt_loss, oN, oD = optimal(budget)
    print(f"{name:12s} {D/N:7.1f} tok/param  loss {L(N,D):.4f}  "
          f"vs optimal {opt_loss:.4f} at {fmt(oN)} params")`,
      'GPT-3 sits well left of optimal — badly undertrained for its size. Llama-3-8B sits far right, deliberately: it trades a little loss for permanently cheaper inference.'),

    quiz('You have compute for a 10B-parameter Chinchilla-optimal model, but will serve 100 billion requests. What should you do?',
      ['Train a smaller model on far more tokens — slightly higher loss, but much lower inference cost forever',
       'Follow Chinchilla exactly; it is the optimum',
       'Train a larger model on fewer tokens',
       'Split the budget across an ensemble'],
      0,
      'Chinchilla optimizes training loss per training FLOP and ignores serving entirely. At 100B requests, inference dominates total cost by orders of magnitude, so a 3B model trained on 5× the tokens — worse loss, 3× cheaper per request — usually wins on total cost. This is exactly the reasoning behind the Llama-3 token counts.'),

    recap(`- Explain the three terms of the scaling law and say what the irreducible term corresponds to.
- Say why predictability, rather than the rate of improvement, is what mattered commercially.
- State the Chinchilla ratio and what Kaplan et al. got wrong.
- Argue for deliberately "overtraining" a model, using total cost of ownership rather than training FLOPs.
- Give the strongest version of the argument that emergence is a measurement artefact — and say what it does
  not settle.`),
  ],
  refs: [
    paper('Scaling Laws for Neural Language Models', 'Kaplan et al.', 2020, 'https://arxiv.org/abs/2001.08361', 'The original. Set the field\'s direction for two years.'),
    paper('Training Compute-Optimal Large Language Models', 'Hoffmann et al.', 2022, 'https://arxiv.org/abs/2203.15556', 'Chinchilla. Corrected the allocation and reshaped every training run since.'),
    paper('Emergent Abilities of Large Language Models', 'Wei et al.', 2022, 'https://arxiv.org/abs/2206.07682', 'The emergence claim.'),
    paper('Are Emergent Abilities of LLMs a Mirage?', 'Schaeffer, Miranda & Koyejo', 2023, 'https://arxiv.org/abs/2304.15004', 'The rebuttal. Read both.'),
    paper('Scaling Laws for Precision', 'Kumar et al.', 2024, 'https://arxiv.org/abs/2411.04330', 'Extends the framework to quantization and low-precision training.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'llm-finetuning',
  title: 'Fine-Tuning, Instruction Tuning, and PEFT',
  sub: 'Turning a text predictor into something that answers questions.',
  mins: 24, level: 'core',
  prereq: ['llm-pretraining'],
  tags: ['fine-tuning', 'LoRA', 'SFT'],
  sections: [
    tldr(`Pretraining produces a model with enormous capability and no manners. It completes text; it does not
answer questions, because nothing ever asked it to.

**Post-training** fixes the behaviour. It costs under 1% of pretraining's compute and accounts for most of what
users experience as quality. Two stages: **supervised fine-tuning** (show it thousands of good answers and have
it imitate) and then preference optimization like RLHF, which gets past imitation's ceiling.

The other half of this lesson is **PEFT** — how to fine-tune a model without the GPUs to hold its gradients.
LoRA, derived back in [the matrices lesson](#/l/math-matrices), is the answer, and it is why fine-tuning is
something you can do on a single consumer GPU.`),

    jargon([
      ['base model', 'The raw output of pretraining. Completes text, does not follow instructions.'],
      ['post-training', 'Everything after pretraining: SFT, preference optimization, safety tuning. Cheap in compute, decisive for quality.'],
      ['SFT', 'Supervised Fine-Tuning. Continue training on (prompt, ideal response) pairs.'],
      ['instruction tuning', 'SFT specifically aimed at making the model follow instructions rather than continue text.'],
      ['loss masking', 'Computing the loss only on the response tokens, not the prompt. You want it to learn to *answer*, not to predict questions.'],
      ['chat template', 'The special-token format marking who said what. Model-specific, and getting it wrong silently degrades quality.'],
      ['RLHF', 'Reinforcement Learning from Human Feedback. Training against a learned model of human preference rather than fixed examples.'],
      ['PEFT', 'Parameter-Efficient Fine-Tuning. Training a small number of extra parameters and freezing the rest.'],
      ['LoRA', 'Low-Rank Adaptation. Represent the weight update as a product of two skinny matrices. The dominant PEFT method.'],
      ['rank $r$', 'The width of LoRA\'s bottleneck. 8–64 typically. Higher = more capacity and more parameters.'],
      ['QLoRA', 'LoRA on top of a 4-bit quantized base model. Lets a 70B model be fine-tuned on one consumer GPU.'],
      ['catastrophic forgetting', 'Losing previously-learned abilities while fine-tuning on something new.'],
    ]),

    t(`## The problem with a base model

A pretrained model completes text. Ask it "What is the capital of France?" and a plausible completion is:

> What is the capital of Germany? What is the capital of Italy?

It is not being unhelpful — it is doing exactly what it was trained to do. It has seen far more lists of quiz
questions than answered ones. **The capability is there; the behavior is not.**

Fixing that is *post-training*, and it costs less than 1% of the compute of pretraining while accounting for most of
what users experience as quality.`),

    t(`## Supervised fine-tuning

Continue training on curated (prompt, ideal response) pairs, with the loss masked to the **response tokens only** —
you want the model to learn to produce answers, not to predict prompts.

Data is the whole game here. Quality dominates quantity, sometimes dramatically: LIMA achieved strong results with
1,000 carefully written examples. Typical volumes are 10k–1M, increasingly generated or filtered by other models.

**The ceiling of SFT is the quality of its demonstrations.** Imitation cannot exceed what it imitates: if your
annotators write mediocre answers, you get a model that writes mediocre answers, and no amount of additional
data changes that.

Worse, there is a subtler failure. Training the model to confidently produce answers a *human expert* would give
teaches it to sound like it knows things — including on questions where it does not. Imitating the *form* of
expertise is easier than acquiring the substance, and SFT cannot tell the difference.

That ceiling is the reason RLHF exists: instead of showing the model what to say, you let it generate and score
it, so it can find answers better than any single demonstration.`),

    t(`## Chat templates

Instruction models use special tokens to mark roles:

\`\`\`
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
Paris.<|im_end|>
\`\`\`

The exact format is model-specific and **matters**. Using the wrong template — or hand-concatenating strings instead
of using the tokenizer's \`apply_chat_template\` — degrades quality noticeably and is a common silent bug. It is also
the boundary that prompt injection attacks try to cross.`),

    t(`## Parameter-efficient fine-tuning

Full fine-tuning of a 70B model needs ~1.1 TB of optimizer state and gradients. PEFT methods train a small number of
new parameters and freeze the rest.

**LoRA** is the dominant one. Freeze $W$, learn a low-rank update:

$$W' = W + \\frac{\\alpha}{r}BA, \\qquad B\\in\\mathbb{R}^{d\\times r},\\ A\\in\\mathbb{R}^{r\\times k},\\ r \\ll d$$

$A$ is initialized randomly and $B$ to zero, so training starts exactly at the pretrained model. At $r=16$ on a 7B
model you train ~0.1% of the parameters.

Why it works: the *update* a task requires is empirically low-rank, even though $W$ is not. You are not compressing
the model, you are compressing the change.

**QLoRA** goes further — the frozen base is quantized to 4 bits while the LoRA adapters stay in bf16. A 70B model
fine-tunes on a single 48 GB GPU.`),

    viz('gpu-memory', { mode: 'lora', params: 7 }),

    t(`Switch that figure between **full fine-tuning** and **LoRA** at 7B and watch the optimizer-state bar disappear —
that is the whole proposition.

Practical notes: adapters can be **merged** into $W$ after training, so inference has zero overhead. Or kept separate,
letting one served base model swap adapters per request. $\\alpha/r$ is the effective scale; the common convention
$\\alpha = 2r$ keeps it constant as you change rank.`),

    warn(`**Catastrophic forgetting.** Fine-tuning on a narrow dataset degrades general capability, sometimes sharply.
Mitigations: keep the learning rate low ($10^{-5}$ to $10^{-4}$, far below pretraining), train few epochs (1–3), mix in
general data, or use LoRA — which forgets less, partly because it simply cannot move the weights very far.

Related and underappreciated: fine-tuning on *any* narrow task can undo safety training. This has been demonstrated
repeatedly with very small numbers of examples.`),

    code('LoRA, implemented and verified', `import numpy as np
rng = np.random.default_rng(0)

d_in, d_out, r = 512, 512, 8

W = rng.normal(0, d_in**-0.5, (d_in, d_out))     # frozen pretrained weights
A = rng.normal(0, 0.01, (d_in, r))               # LoRA down-projection
B = np.zeros((r, d_out))                         # up-projection, zero-init
alpha = 16

def forward(x, merged=False):
    if merged:
        return x @ (W + (alpha/r) * (A @ B))
    return x @ W + (alpha/r) * ((x @ A) @ B)      # low-rank path, computed separately

x = rng.normal(size=(4, d_in))
print("at init, LoRA is exactly the identity:",
      np.allclose(forward(x), x @ W))

full = d_in * d_out
lora = d_in*r + r*d_out
print(f"\\nfull fine-tune params : {full:,}")
print(f"LoRA (r={r}) params    : {lora:,}  ({lora/full:.2%})")
print(f"optimizer memory saved : {(full-lora)*16/1e6:.0f} MB per layer (Adam, fp32)")

# --- train the adapter to fit a target update ---
target_delta = np.outer(rng.normal(size=d_in), rng.normal(size=d_out)) * 0.1
for step in range(600):
    delta = (alpha/r) * (A @ B)
    err = delta - target_delta
    A -= 0.5 * (alpha/r) * (err @ B.T)
    B -= 0.5 * (alpha/r) * (A.T @ err)
print(f"\\nfitting a rank-1 target with rank-{r} LoRA:")
print(f"  relative error: {np.linalg.norm((alpha/r)*(A@B) - target_delta)/np.linalg.norm(target_delta):.4f}")

print(f"\\nmerged and unmerged agree: {np.allclose(forward(x), forward(x, merged=True), atol=1e-9)}")
print("-> adapters can be folded into W after training, so inference costs nothing extra.")`),

    quiz('LoRA at rank 8 trains 0.1% of the parameters yet often matches full fine-tuning. Why?',
      ['The weight *update* a task needs has low intrinsic rank, even though the pretrained weights do not',
       'Most pretrained weights are zero anyway',
       'It only works for simple tasks',
       'The frozen weights are being implicitly updated too'],
      0,
      'Full fine-tuning finds some $\\Delta W$; the empirical observation behind LoRA is that a good $\\Delta W$ can be approximated well by a rank-8 matrix. You are constraining the *change*, not the model. It follows that harder tasks — and especially learning genuinely new knowledge rather than new behavior — need higher rank, and full fine-tuning still wins there.'),

    recap(`- Explain why a base model answers a question with more questions, and why that is correct behaviour
  rather than a bug.
- Say what loss masking does during SFT and why the prompt is excluded.
- State the ceiling of supervised fine-tuning, and why imitation cannot pass it.
- Explain LoRA's parameter count and the empirical bet underneath it — and when that bet fails.
- Say when to prefer prompting, RAG, LoRA, or full fine-tuning, given a described problem.
- Recognise "wrong chat template" as a cause of mysteriously poor output.`),
  ],
  refs: [
    paper('LoRA: Low-Rank Adaptation of Large Language Models', 'Hu et al.', 2021, 'https://arxiv.org/abs/2106.09685', ''),
    paper('QLoRA: Efficient Finetuning of Quantized LLMs', 'Dettmers et al.', 2023, 'https://arxiv.org/abs/2305.14314', '4-bit base + bf16 adapters. Put 70B fine-tuning on one GPU.'),
    paper('Training language models to follow instructions', 'Ouyang et al.', 2022, 'https://arxiv.org/abs/2203.02155', 'InstructGPT. SFT + RLHF, and the paper behind ChatGPT.'),
    paper('LIMA: Less Is More for Alignment', 'Zhou et al.', 2023, 'https://arxiv.org/abs/2305.11206', '1,000 examples. The argument that capability is in pretraining and SFT only surfaces it.'),
    paper('Fine-tuning Aligned Language Models Compromises Safety', 'Qi et al.', 2023, 'https://arxiv.org/abs/2310.03693', 'A handful of examples can undo safety training.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'llm-decoding',
  title: 'Decoding and Sampling',
  sub: 'The model gives a distribution. Turning it into text is a separate set of decisions.',
  mins: 18, level: 'core',
  prereq: ['llm-pretraining'],
  tags: ['decoding', 'sampling'],
  sections: [
    tldr(`A language model does not output text. It outputs a **probability for every token in the
vocabulary**, and something else has to pick one. That picking is *decoding*, it happens outside the model, and
it changes the output enormously.

This is genuinely under-appreciated: the difference between a model that seems creative and one that seems
robotic is often not the model at all, but the \`temperature\` and \`top_p\` values in the API call.

One tension runs through the whole lesson. Sampling from the low-probability tail is where creativity comes
from, and it is also where hallucination comes from. They are the *same mechanism*, and no decoding setting
separates them.`),

    jargon([
      ['decoding', 'The process of turning the model\'s probability distribution into actual output tokens. Not part of the model.'],
      ['logits', 'The raw unnormalized scores the model outputs, one per vocabulary token. Any real number.'],
      ['greedy decoding', 'Always take the single highest-probability token. Deterministic, and prone to loops.'],
      ['temperature $T$', 'Divides the logits before softmax. Below 1 sharpens the distribution (more predictable); above 1 flattens it (more random). $T = 0$ is greedy.'],
      ['top-k', 'Consider only the $k$ most likely tokens, discard the rest, renormalize.'],
      ['top-p / nucleus', 'Consider the smallest set of tokens whose probabilities add up to $p$. Adapts its size to how confident the model is.'],
      ['beam search', 'Keeping several candidate *sequences* alive and returning the most probable complete one.'],
      ['degeneration', 'The failure mode where output collapses into repetitive loops. What greedy and beam search do on open-ended text.'],
      ['repetition penalty', 'Reducing the score of tokens already used, to discourage loops. A blunt instrument.'],
      ['speculative decoding', 'A small fast model drafts several tokens; the big model verifies them all in one pass. Same output, several times faster.'],
      ['constrained decoding', 'Forcing output to match a grammar or JSON schema by zeroing out invalid tokens before sampling.'],
    ]),

    t(`## The choice

At each step, the model produces a vector of **logits** — one score per vocabulary token, typically 50,000 to
200,000 of them. Softmax turns those into probabilities. And then... something has to choose.

That choice is not part of the model, and the model has no opinion about it. It is a separate algorithm you
control, and it is why the same model can be dull or wild depending on two numbers in your API call.`),

    viz('sampling-strategies'),

    t(`**Greedy** takes the argmax. Deterministic and repetitive — greedy decoding is notorious for degenerating into
loops ("I think that I think that I think...").

**Temperature** rescales logits before softmax: $p_i \\propto \\exp(z_i/T)$. Below 1 sharpens, above 1 flattens. $T\\to0$
is greedy; $T\\to\\infty$ is uniform.

**Top-k** keeps the $k$ highest-probability tokens and renormalizes. Simple, but $k$ is fixed regardless of how
confident the model is.

**Top-p (nucleus)** keeps the smallest set of tokens whose cumulative probability reaches $p$ (typically 0.9 or
0.95), and renormalizes over those.

The word doing the work is **adaptive**. After "The capital of France is" the model is nearly certain, so the
nucleus contains one token and sampling is effectively deterministic. After "She opened the door and saw" there
are hundreds of reasonable continuations, so the nucleus expands to include them all. Top-k cannot do this — a
fixed $k=40$ either injects nonsense into the confident case or truncates the creative one.

That single property is why top-p became the default everywhere.

**Min-p** keeps tokens with probability at least $p \\cdot p_{\\max}$ — a newer variant that behaves better at high
temperature.

**Repetition and presence penalties** reduce the logits of already-used tokens. Blunt instruments; they suppress
legitimate repetition too (a name that should recur, code indentation).`),

    t(`## Beam search, and why generative models mostly abandoned it

Beam search keeps the $k$ highest-probability *sequences*, not tokens, and returns the best complete one. It is
standard for translation and summarization, where there is roughly one right answer.

For open-ended generation it fails in an interesting way: the highest-probability sequence is **boring**. Real human
text is not the mode of the distribution — it has surprise in it, and its per-token likelihood fluctuates. Beam search
optimizes for exactly the wrong thing and produces bland, repetitive output. This observation is the core of "The
Curious Case of Neural Text Degeneration," which introduced nucleus sampling.`),

    intuition(`There is a real tension here. Sampling from the tail is where creativity comes from *and* where
hallucination comes from — they are the same mechanism. A model asked for a citation it does not know will happily
sample a plausible-sounding one. Lowering the temperature reduces both.

Practical defaults: **temperature 0** for extraction, classification, and code where you want determinism;
**0.7–1.0 with top-p 0.9–0.95** for open-ended writing. Note that even temperature 0 is not perfectly deterministic in
practice — batched GPU inference has non-deterministic floating-point reduction order.`),

    t(`## Structured output

When you need valid JSON or a specific grammar, **constrained decoding** masks out any token that would violate it
before sampling. This makes malformed output structurally impossible rather than merely unlikely.

Caveat worth knowing: constraining can lower quality, because it forces the model off the token path it "wanted" and
into a region of the distribution it is less well-calibrated on. Asking for a format the model handles naturally
usually beats forcing an awkward one.`),

    code('Every sampler, from scratch', `import numpy as np
rng = np.random.default_rng(0)

vocab = ["the","a","this","my","our","cat","dog","quantum","zebra","xylophone"]
logits = np.array([3.9, 3.2, 2.4, 2.0, 1.6, 0.8, 0.5, -1.4, -2.2, -3.5])

def softmax(z, T=1.0):
    z = z / T; z = z - z.max(); e = np.exp(z); return e / e.sum()

def top_k(p, k):
    out = np.zeros_like(p)
    idx = np.argsort(-p)[:k]
    out[idx] = p[idx]
    return out / out.sum()

def top_p(p, thresh):
    idx = np.argsort(-p)
    cum = np.cumsum(p[idx])
    keep = idx[:np.searchsorted(cum, thresh) + 1]
    out = np.zeros_like(p); out[keep] = p[keep]
    return out / out.sum()

def min_p(p, frac):
    out = np.where(p >= frac * p.max(), p, 0.0)
    return out / out.sum()

def entropy(p):
    q = p[p > 0]; return -(q * np.log2(q)).sum()

print(f"{'strategy':22s} {'kept':>5} {'entropy':>8}  top-3")
for name, p in [
    ("raw (T=1)",        softmax(logits)),
    ("T=0.5",            softmax(logits, 0.5)),
    ("T=1.5",            softmax(logits, 1.5)),
    ("top-k=3",          top_k(softmax(logits), 3)),
    ("top-p=0.9",        top_p(softmax(logits), 0.9)),
    ("min-p=0.1",        min_p(softmax(logits), 0.1)),
    ("T=1.5 + top-p=0.9",top_p(softmax(logits, 1.5), 0.9)),
]:
    top3 = ", ".join(f"{vocab[i]}={p[i]:.2f}" for i in np.argsort(-p)[:3])
    print(f"{name:22s} {int((p>0).sum()):5d} {entropy(p):8.3f}  {top3}")

print("\\nnucleus size adapts to confidence:")
for desc, lg in [("confident", np.array([8.,1.,0.,0.,0.,0.,0.,0.,0.,0.])),
                 ("uncertain", np.array([1.,.9,.8,.8,.7,.7,.6,.6,.5,.5]))]:
    print(f"  {desc:10s} -> top-p 0.9 keeps {(top_p(softmax(lg), 0.9) > 0).sum()} tokens")`),

    quiz('Why does beam search produce bland text for open-ended generation?',
      ['The highest-probability sequence is not representative of human text, which contains genuine surprise',
       'Beam search is too slow',
       'It cannot handle long sequences',
       'It always produces grammatical errors'],
      0,
      'Beam search approximates the *mode* of the sequence distribution. But human text is not modal — its per-token surprisal fluctuates, and the globally most-likely continuation is a generic, repetitive one. Nucleus sampling was introduced on exactly this observation: match the distribution rather than maximize it. Beam search remains right where the answer is nearly unique, as in translation.'),

    recap(`- Explain that decoding is separate from the model, and name the two parameters that change output most.
- Say what temperature does to the logits, and what $T=0$ and $T\\to\\infty$ correspond to.
- Explain why top-p adapts where top-k cannot, with an example of each regime.
- Say why maximising sequence probability produces *worse* open-ended text, and where beam search is still
  right.
- State the creativity/hallucination tension in one sentence and say why no temperature setting resolves it.
- Explain how speculative decoding is faster without changing the output distribution.`),
  ],
  refs: [
    paper('The Curious Case of Neural Text Degeneration', 'Holtzman et al.', 2019, 'https://arxiv.org/abs/1904.09751', 'Nucleus sampling, and the clearest analysis of why likelihood maximization fails for open-ended text.'),
    paper('Hierarchical Neural Story Generation', 'Fan, Lewis & Dauphin', 2018, 'https://arxiv.org/abs/1805.04833', 'Top-k sampling.'),
    blog('How to generate text', 'Patrick von Platen (HuggingFace)', 2020, 'https://huggingface.co/blog/how-to-generate', 'Practical walkthrough with code.'),
    codeRef('Outlines', 'dottxt-ai', 2023, 'https://github.com/dottxt-ai/outlines', 'Constrained decoding to regex and JSON schemas.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'llm-prompting',
  title: 'Prompting and In-Context Learning',
  sub: 'Learning that happens in the forward pass, with no gradient step.',
  mins: 20, level: 'core',
  prereq: ['llm-finetuning'],
  tags: ['prompting', 'chain-of-thought', 'ICL'],
  sections: [
    tldr(`Show a model three worked examples in the prompt and it will do the fourth — **with no weight updates
at all**. Nothing is trained; the "learning" happens entirely inside one forward pass and is forgotten
immediately after.

This is strange enough that it took the field several years to get a partial explanation, and the current best
one involves a specific learned circuit called an induction head.

The second half of the lesson is **chain-of-thought**, whose mechanism is less mysterious than it sounds: a
transformer does a fixed amount of computation per token, so making it emit more tokens literally buys it more
computation. Thinking out loud converts *depth* into *length*.`),

    jargon([
      ['in-context learning (ICL)', 'Performing a task from examples given in the prompt, with no weight changes. Also called few-shot prompting.'],
      ['zero-shot / few-shot', 'No examples in the prompt / a handful of examples in the prompt.'],
      ['induction head', 'A learned two-head circuit that finds an earlier occurrence of the current token and predicts what followed it. The main known mechanism behind ICL.'],
      ['chain-of-thought (CoT)', 'Prompting the model to write out intermediate reasoning steps before its answer.'],
      ['serial computation', 'Steps that must happen one after another because each needs the previous result. A transformer\'s depth caps how many fit in one forward pass.'],
      ['self-consistency', 'Sampling several chains of thought and taking a majority vote on the final answer.'],
      ['system prompt', 'Standing instructions placed before the conversation, in a privileged role the model is trained to weight heavily.'],
      ['prompt injection', 'Untrusted input containing instructions that the model follows. A structural security problem, not a bug to patch.'],
    ]),

    t(`## The surprise

Put a few worked examples in the prompt and the model does the task. No fine-tuning, no gradient steps, **no
weight updates at all** — the parameters are frozen and identical before and after.

GPT-3 made this famous in 2020, and it is worth preserving how odd it is. Everything else in this atlas
describes learning as changing weights. Here, something functionally like learning happens inside a single
forward pass, in the activations, and vanishes the moment the context is cleared.`),

    viz('in-context-learning'),

    t(`Mechanistically, **induction heads** do much of the work: a circuit that finds a previous occurrence of the
current token and copies what followed. That is exactly the operation "match the pattern in the examples and continue
it" requires. These heads form abruptly during training, and their appearance coincides with a jump in in-context
learning ability.

Suggestively, a body of work shows transformers can implement gradient descent on the in-context examples inside the
forward pass for simple function classes. It is not settled that this is what large models do, but it is a striking
result.`),

    t(`## Chain-of-thought

Ask the model to reason step by step and accuracy on multi-step problems rises substantially.

The mechanism is not mystical: a transformer does a fixed amount of computation per token. A problem needing more
serial steps than the network has layers **cannot** be solved in one forward pass. Generating intermediate tokens
gives the model more forward passes to work with, and lets it write its intermediate state into the context where
later steps can read it.

**Chain-of-thought converts depth into length.** That framing predicts its behavior: it helps most on problems with
genuine serial structure (arithmetic, multi-hop reasoning, code tracing), and helps little on single-step recall.

It is also *emergent* — it hurts small models, which produce plausible-looking but wrong reasoning and then follow it.`),

    t(`## What actually works

Ordered roughly by how reliably it helps:

1. **Be specific about the task and the output format.** Most prompt failures are underspecification.
2. **Give examples** covering the edge cases you care about, and make sure the label distribution in them is balanced —
   models pick up on the format and the distribution, sometimes more than the label correctness.
3. **Ask for reasoning before the answer.** Order matters: reasoning after the answer does nothing, since the answer
   token was generated first.
4. **Decompose** multi-part tasks into separate calls rather than one heroic prompt.
5. **Self-consistency**: sample several chains and take the majority answer. Reliable, and costs $n\\times$.
6. **Give the model an out** — "if the context does not contain the answer, say so" measurably reduces confabulation.

Things that are oversold: elaborate personas, threats and bribes, and most "magic phrases." Some show small effects on
some benchmarks and do not transfer.`),

    warn(`**Prompts do not transfer between models.** A prompt tuned on one model can perform worse on another, and
model updates silently change behavior. If prompts are load-bearing in a product, you need an eval suite and you need
to re-run it on every model change. Treat prompts as code with no type system.

**Security.** Everything in the context is text, and the model cannot reliably distinguish your instructions from
instructions embedded in retrieved documents or user input. Prompt injection is not solved. Never let model output
take a consequential action without validation outside the model.`),

    code('Self-consistency, simulated', `import numpy as np
rng = np.random.default_rng(0)

def simulate(p_correct, n_samples, n_wrong_modes, trials=4000):
    """Majority voting over sampled chains, where wrong answers scatter."""
    wins = 0
    for _ in range(trials):
        counts = {}
        for _ in range(n_samples):
            if rng.random() < p_correct:
                ans = "correct"
            else:
                ans = f"wrong_{rng.integers(n_wrong_modes)}"
            counts[ans] = counts.get(ans, 0) + 1
        if max(counts, key=counts.get) == "correct":
            wins += 1
    return wins / trials

print("single-sample accuracy 0.4, wrong answers scattered over 8 modes:")
for n in [1, 3, 5, 11, 21, 41]:
    print(f"  {n:3d} samples -> {simulate(0.4, n, 8):.3f}")

print("\\nthe scatter matters — if the model is CONSISTENTLY wrong:")
for modes in [1, 2, 4, 8, 20]:
    print(f"  wrong answers over {modes:2d} mode(s) -> {simulate(0.4, 21, modes):.3f}")
print("\\nMajority voting exploits the fact that there are many ways to be wrong")
print("and only one way to be right. A systematically biased model gains nothing.")`),

    quiz('Why does chain-of-thought prompting improve multi-step reasoning?',
      ['It converts a depth-limited computation into a longer one — each generated token is another forward pass',
       'It makes the model more confident',
       'It retrieves relevant training examples',
       'It reduces the temperature'],
      0,
      'A transformer performs a fixed amount of serial computation per token, bounded by its layer count. A problem requiring more serial steps than that cannot be solved in one pass. Emitting intermediate tokens buys additional passes and externalizes state into the context. Depth becomes length — which also explains why CoT helps little on single-step recall.'),

    recap(`- State what makes in-context learning surprising, in terms of what does and does not change.
- Describe what an induction head does, and why that operation is what few-shot prompting needs.
- Explain chain-of-thought as converting depth into length, and use it to predict which tasks it helps.
- Say why chain-of-thought hurts small models.
- Explain why prompt injection is a structural problem rather than a bug that can be patched.`),
  ],
  refs: [
    paper('Language Models are Few-Shot Learners', 'Brown et al.', 2020, 'https://arxiv.org/abs/2005.14165', 'GPT-3. In-context learning as a headline result.'),
    paper('Chain-of-Thought Prompting Elicits Reasoning', 'Wei et al.', 2022, 'https://arxiv.org/abs/2201.11903', ''),
    paper('Self-Consistency Improves Chain of Thought Reasoning', 'Wang et al.', 2022, 'https://arxiv.org/abs/2203.11171', ''),
    paper('Rethinking the Role of Demonstrations', 'Min et al.', 2022, 'https://arxiv.org/abs/2202.12837', 'Randomizing labels in few-shot examples barely hurts. What ICL learns is not what you think.'),
    paper('Transformers learn in-context by gradient descent', 'von Oswald et al.', 2022, 'https://arxiv.org/abs/2212.07677', 'The mechanistic hypothesis.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'llm-rag',
  title: 'Retrieval-Augmented Generation',
  sub: 'Giving the model access to things it was not trained on.',
  mins: 22, level: 'core',
  prereq: ['nn-embeddings', 'llm-prompting'],
  tags: ['RAG', 'retrieval'],
  sections: [
    tldr(`A model only knows what was in its training data, frozen at a cutoff date, with no citations and no
access to your company's documents. **Retrieval-augmented generation** works around all of that without
retraining anything: look up relevant text at query time, paste it into the prompt, and let the model read it.

Structurally it is [kNN](#/l/ml-svm-knn) over embeddings with a language model reading the neighbours — which
means RAG failures are usually *retrieval* failures, not model failures. If the right passage never made it into
the context, no amount of prompt engineering will save you.`),

    jargon([
      ['RAG', 'Retrieval-Augmented Generation. Fetch relevant documents, put them in the prompt, generate an answer grounded in them.'],
      ['parametric knowledge', 'What the model knows from its weights. Frozen, uncitable, and impossible to update without retraining.'],
      ['chunk', 'A passage a document is split into for indexing. Chunk size and boundaries matter more than people expect.'],
      ['vector index', 'A data structure for fast approximate nearest-neighbour search over millions of embeddings. HNSW and IVF-PQ are the common ones.'],
      ['bi-encoder', 'Embeds query and document *separately*, so documents can be embedded once, offline. Fast, less accurate.'],
      ['cross-encoder', 'Scores a (query, document) pair *jointly*. Far more accurate, far too slow to run over a whole corpus — so it is used to rerank a shortlist.'],
      ['reranking', 'Re-scoring the top ~50 retrieved candidates with a cross-encoder to pick the best few.'],
      ['hybrid search', 'Combining dense (embedding) retrieval with keyword search like BM25. Catches exact terms embeddings miss.'],
      ['grounding', 'Requiring the answer to be supported by the retrieved text, ideally with citations.'],
      ['lost in the middle', 'The observed tendency of models to attend well to the start and end of a long context and poorly to the middle.'],
    ]),

    t(`## Why retrieve

Everything a model knows lives in its weights — its **parametric knowledge** — and that has four hard limits,
none of which more training fixes:

- **Frozen at the cutoff.** It cannot know about anything after training ended.
- **Cannot include private data.** Your company's wiki was not on the internet.
- **Cannot be updated cheaply.** Correcting one fact means retraining or fine-tuning, with no guarantee the
  edit sticks or stays local.
- **No citations.** The model cannot tell you where it learned something, because "where" is smeared across
  billions of weights.

RAG addresses all four with the same move: retrieve relevant text at query time, put it in the context window,
and let the model read it. The knowledge lives in a database you control rather than in weights you do not.`),

    t(`## The pipeline

**Indexing** (offline):
1. **Chunk** documents into passages.
2. **Embed** each chunk into a vector.
3. **Store** in a vector index (HNSW, IVF-PQ) for approximate nearest-neighbour search.

**Query time**:
4. **Embed the query**, retrieve top-$k$ chunks by cosine similarity.
5. **Rerank** with a cross-encoder that scores (query, chunk) pairs jointly — far more accurate than the bi-encoder
   used for retrieval, and affordable because it only sees $k$ candidates.
6. **Assemble** the prompt with the retrieved context.
7. **Generate**, ideally with citations.

Structurally this is kNN (from the [SVM/kNN lesson](#/l/ml-svm-knn)) in embedding space, with a language model reading
the neighbours.`),

    t(`## Where it goes wrong

Retrieval quality dominates. A perfect model cannot answer from bad context, and this is where nearly all RAG failures
originate.

**Chunking.** Too small and you sever the context a passage needs; too large and you dilute the embedding and waste
context. 200–500 tokens with overlap is a common starting point, but structure-aware chunking (by section, by
function) beats fixed windows when the documents have structure.

**Semantic search misses exact matches.** Embeddings are poor at rare identifiers, product codes, and names — a
model that has learned "error E1234" and "error E1235" are similar strings will happily return the wrong one.
**Hybrid search** fixes this by running a keyword search alongside the embedding search and combining the two.
It is the single highest-value improvement to a naive pipeline, and both halves are simple enough to state
here.

**How keyword search scores a document — BM25.** It is three intuitions multiplied together, and each one is
something you would have invented yourself.

- **Term frequency.** A document containing the query word many times is probably more about it. But the tenth
  occurrence tells you much less than the second, so the count is passed through a saturating function
  $\\frac{tf(k_1+1)}{tf + k_1}$ rather than used raw. The constant $k_1$ (around 1.5) sets how fast it saturates.
- **Inverse document frequency.** A word appearing in every document distinguishes nothing. Weight each word by
  roughly $\\log\\frac{N}{df}$, where $df$ is how many documents contain it — so "the" counts for almost nothing
  and "Chinchilla" counts for a lot. This is the same $-\\log p$ [surprise](#/l/math-information) as before: a
  rare word carries more information about which document you want.
- **Length normalization.** A long document contains more of every word by accident, so divide by its length
  relative to the average, mixed in with a knob $b$ (around 0.75) controlling how aggressively.

Sum that over the query's words and you have BM25. No training, no embeddings, and it beats dense retrieval
whenever the user typed the exact token that appears in the document.

**How to combine two rankings — reciprocal rank fusion.** BM25 returns scores in the range 0 to 30-ish; cosine
similarity returns 0 to 1. Adding them directly is meaningless, and calibrating them against each other is
fiddly and breaks whenever either system changes.

RRF sidesteps the problem by throwing the scores away and keeping only the **ranks**. Each list contributes
$\\frac{1}{k + \\text{rank}}$ to every document it ranked, with $k \\approx 60$, and the contributions are summed.
A document ranked first by either system gets a large contribution; one ranked highly by *both* wins outright.
Because only order is used, the two systems never need to agree on what a score means — which is why RRF is the
default way to fuse retrieval systems in practice.

**The query is not the document.** Users ask "why is my build failing" and the document says "compilation error
E1234." Fixes: HyDE (have the model write a hypothetical answer and embed *that*), query rewriting, or multi-query
expansion.

**Lost in the middle.** Models attend most reliably to the beginning and end of a long context. Put the highest-ranked
chunk last, adjacent to the question.

**Retrieval can hurt.** Irrelevant retrieved context measurably degrades answers on questions the model knew. Consider
letting the model decide whether to retrieve at all.`),

    key(`**RAG does not stop hallucination.** It reduces it when retrieval succeeds and can *increase* confident
wrongness when retrieval fails, because now there is authoritative-looking text supporting the wrong answer.

Mitigations that work: require citation of specific retrieved spans, instruct explicitly that "not in the provided
context" is an acceptable answer, and evaluate faithfulness (is the answer supported by the context?) separately from
correctness.`),

    t(`## Evaluating it

Evaluate the two halves separately, or you will not know which one is broken:

- **Retrieval** — recall@k (is the answer-bearing chunk in the top $k$?), MRR, nDCG. If recall@10 is 60%, no amount of
  prompt engineering will save you.
- **Generation** — faithfulness (grounded in context), answer relevance, and correctness against a reference.

Frameworks like RAGAS automate parts of this. Build a small gold set of question/answer/source triples early; it will
pay for itself immediately.`),

    t(`## RAG vs long context vs fine-tuning

| | Best for | Weak at |
|---|---|---|
| **RAG** | changing facts, large corpora, citation, access control | reasoning across many documents at once |
| **Long context** | deep reasoning over a bounded document set | cost and latency; degrades in the middle |
| **Fine-tuning** | format, style, domain vocabulary, behavior | injecting facts (unreliable, and hard to update) |

Long context did not kill RAG. Putting 2M tokens in every request is expensive and slow, and retrieval is a
prefilter — the two compose well: retrieve broadly, then let a long-context model read a lot of what you found.

**Fine-tuning is the wrong tool for facts.** It is the right tool for behavior.`),

    code('A minimal RAG pipeline with hybrid search', `import numpy as np
from collections import Counter
import re

docs = [
 "The transformer architecture was introduced in the paper Attention Is All You Need in 2017.",
 "LoRA freezes pretrained weights and injects trainable low-rank matrices into each layer.",
 "BPE tokenization merges the most frequent adjacent pair of symbols repeatedly.",
 "Chinchilla found that compute-optimal training uses about 20 tokens per parameter.",
 "FlashAttention reduces attention memory from quadratic to linear by tiling the computation.",
 "RMSNorm removes the mean-subtraction step from LayerNorm and is used in Llama.",
]

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())
vocab = sorted({w for d in docs for w in tok(d)})
vi = {w: i for i, w in enumerate(vocab)}

# --- sparse: BM25 ---
N, avgdl = len(docs), np.mean([len(tok(d)) for d in docs])
df = Counter(w for d in docs for w in set(tok(d)))
def bm25(query, k1=1.5, b=0.75):
    scores = np.zeros(N)
    for i, d in enumerate(docs):
        tf, dl = Counter(tok(d)), len(tok(d))
        for w in tok(query):
            if w not in tf: continue
            idf = np.log((N - df[w] + 0.5)/(df[w] + 0.5) + 1)
            scores[i] += idf * tf[w]*(k1+1) / (tf[w] + k1*(1 - b + b*dl/avgdl))
    return scores

# --- dense: a stand-in embedding (real systems use a trained encoder) ---
rng = np.random.default_rng(0)
W = rng.normal(size=(len(vocab), 32))
def embed(s):
    ws = [vi[w] for w in tok(s) if w in vi]
    if not ws: return np.zeros(32)
    v = W[ws].mean(0)
    return v / (np.linalg.norm(v) + 1e-9)
D = np.stack([embed(d) for d in docs])

def rrf(*rankings, k=60):
    """Reciprocal rank fusion — combines rankings without score calibration."""
    score = np.zeros(N)
    for r in rankings:
        for rank, idx in enumerate(np.argsort(-r)):
            score[idx] += 1 / (k + rank + 1)
    return score

for q in ["how many tokens per parameter should I train on?",
          "what does LoRA freeze?"]:
    s_sparse, s_dense = bm25(q), D @ embed(q)
    print(f"\\nQ: {q}")
    for name, s in [("BM25", s_sparse), ("dense", s_dense), ("hybrid (RRF)", rrf(s_sparse, s_dense))]:
        top = np.argsort(-s)[0]
        print(f"  {name:14s} -> {docs[top][:68]}...")`),

    quiz('Your RAG system answers wrongly. Retrieval recall@5 is 55%. What should you fix first?',
      ['Retrieval — nearly half of queries never see the answer, and no prompt can fix that',
       'The generation prompt',
       'Switch to a larger language model',
       'Increase the temperature'],
      0,
      'With recall@5 at 55%, 45% of queries have no chance regardless of the model. Work the retrieval side: add hybrid BM25+dense search, add a cross-encoder reranker, revisit chunking, and consider query rewriting. Only tune generation once retrieval recall is high — otherwise you are optimizing a component that is not the bottleneck.'),

    recap(`- Name the four limits of parametric knowledge that RAG works around.
- Describe the pipeline end to end, and say why a cross-encoder reranks rather than retrieves.
- Explain why retrieval recall is the first thing to measure when a RAG system answers badly.
- Say what hybrid search adds over pure embedding retrieval, and give a query where it matters.
- Explain "lost in the middle" and how it should change the way you order retrieved chunks.`),
  ],
  refs: [
    paper('Retrieval-Augmented Generation for Knowledge-Intensive NLP', 'Lewis et al.', 2020, 'https://arxiv.org/abs/2005.11401', 'The paper that named it.'),
    paper('Lost in the Middle', 'Liu et al.', 2023, 'https://arxiv.org/abs/2307.03172', 'Position within a long context strongly affects whether the model uses the information.'),
    paper('Precise Zero-Shot Dense Retrieval without Relevance Labels', 'Gao et al.', 2022, 'https://arxiv.org/abs/2212.10496', 'HyDE — embed a hypothetical answer instead of the question.'),
    paper('RAGAS: Automated Evaluation of RAG', 'Es et al.', 2023, 'https://arxiv.org/abs/2309.15217', ''),
    blog('Building RAG-based LLM Applications for Production', 'Anyscale', 2023, 'https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1', 'Careful, measured, with ablations rather than assertions.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'llm-moe',
  title: 'Mixture of Experts',
  sub: 'More parameters without more compute per token.',
  mins: 18, level: 'advanced',
  prereq: ['llm-transformer'],
  tags: ['MoE', 'architecture'],
  sections: [
    tldr(`A normal ("dense") model runs *every* parameter for *every* token. That is wasteful — a token in
Python code and a token in French plausibly want different machinery.

**Mixture of Experts** replaces each feed-forward layer with many parallel copies and a small **router** that
sends each token to just one or two of them. Total parameters grow enormously; compute per token barely moves.

The result is the trick behind most frontier models: hundreds of billions of parameters' worth of knowledge, at
the inference cost of a much smaller model. The catch is that everything about MoE is a systems problem — the
mathematics is easy, and making the routing balanced and the memory affordable is where the difficulty lives.`),

    jargon([
      ['dense model', 'A normal transformer, where every parameter participates in every token\'s computation.'],
      ['sparse / MoE model', 'A model where only a fraction of parameters are used per token.'],
      ['expert', 'One of the parallel feed-forward networks. Despite the name, experts rarely specialise in anything a human would name.'],
      ['router / gate', 'A small learned layer deciding which experts each token goes to.'],
      ['top-$k$ routing', 'Sending each token to its $k$ best-scoring experts. $k$ is 1 or 2 in practice.'],
      ['active parameters', 'How many parameters actually run per token. The number that determines inference speed — as opposed to total parameters, which determines memory.'],
      ['load balancing', 'Keeping tokens spread evenly across experts. Does not happen by itself.'],
      ['expert collapse', 'The failure where the router picks a few favourites, those improve from all the gradient, and the rest atrophy. A rich-get-richer loop.'],
      ['capacity factor', 'How many tokens each expert will accept per batch before overflow tokens get dropped.'],
      ['token dropping', 'Discarding tokens routed to an over-full expert. They skip the layer via the residual connection.'],
    ]),

    t(`## The idea

A dense model runs every parameter for every token, and that is the observation MoE attacks. Not every token
needs the same processing — so why pay for all of it every time?

MoE replaces the feed-forward layer with $E$ parallel expert FFNs plus a small **router** that sends each token
to its top-$k$ experts, usually $k = 1$ or $2$.

Now count the two things that used to move together:

- **Total parameters** grow by roughly $E\\times$. That is capacity — knowledge the model can store.
- **Compute per token** grows only by $k\\times$. That is cost — what you pay per token generated.

Decoupling those two is the entire point. A model can have 400B parameters of stored knowledge while doing 30B
parameters' worth of arithmetic per token. You still need memory for all 400B, which is why MoE is a win for
large-scale serving and not for running a model on your laptop.`),

    viz('moe-routing'),

    t(`Formally, with router logits $g(\\mathbf{x}) = W_r\\mathbf{x}$ and $\\mathcal{T}$ the top-$k$ indices:

$$\\mathbf{y} = \\sum_{i\\in\\mathcal{T}} \\text{softmax}(g(\\mathbf{x}))_i \\cdot E_i(\\mathbf{x})$$

Routing is per-token and per-layer, so a sequence's tokens fan out across many experts and the assignment changes at
every layer.`),

    t(`## The problems

**Load imbalance / expert collapse.** Nothing in the objective encourages balance. Left alone, the router develops
favourites, the popular experts get all the gradient and improve, and the rest atrophy. Drag the balancing slider to 0
in the figure to watch it.

Fixes: an **auxiliary load-balancing loss** penalizing the correlation between routing probability and actual load
(Switch Transformer), or **loss-free balancing** with a learned per-expert bias adjusted to equalize load
(DeepSeek-V3) — which avoids the auxiliary loss interfering with the language modeling objective.

**Capacity and dropping.** For efficient batched execution each expert gets a fixed token capacity. Overflow tokens are
*dropped* — they skip the FFN entirely and pass through on the residual. A capacity factor of 1.25 is typical.

**Memory.** All $E$ experts must be resident even though only $k$ run. MoE saves compute, not memory, which makes it a
poor fit for single-GPU deployment and a good fit for large distributed serving.

**Training instability.** The router is a discrete decision made differentiable by a softmax, and it is jumpy. Router
z-loss and careful initialization are standard.`),

    t(`## Where it landed

Mixtral 8×7B (2023) was the model that made open MoE credible: 47B total parameters, ~13B active, quality comparable
to a 70B dense model at far lower inference FLOPs.

DeepSeek-V3 pushed it further with 671B total / 37B active, fine-grained experts (many small ones rather than a few
large), and shared experts that every token uses for common processing.

Frontier proprietary models are widely believed to be MoE, though rarely confirmed.

The tradeoff in one line: **MoE buys parameter count cheaply and pays for it in memory and serving complexity.** For
large-scale serving where you can shard experts across devices, that is a good trade. For a single GPU, it usually is
not.`),

    code('MoE routing and the balancing problem', `import numpy as np
rng = np.random.default_rng(0)

E, k, d, T = 8, 2, 32, 2000

def route(x, Wr, bias=None):
    logits = x @ Wr
    if bias is not None: logits = logits + bias
    idx = np.argsort(-logits, axis=1)[:, :k]
    gates = np.take_along_axis(logits, idx, 1)
    gates = np.exp(gates - gates.max(1, keepdims=True))
    gates /= gates.sum(1, keepdims=True)
    return idx, gates

x = rng.normal(size=(T, d))
Wr = rng.normal(0, d**-0.5, (d, E))
Wr[:, :2] += 0.35                                 # a slight head start for experts 0,1

def load_of(idx):
    return np.bincount(idx.ravel(), minlength=E)

idx, _ = route(x, Wr)
load = load_of(idx)
cv = load.std() / load.mean()
print(f"no balancing:        load {load}  CV={cv:.3f}")

# --- auxiliary loss style: report the Switch Transformer penalty ---
probs = np.exp(x @ Wr); probs /= probs.sum(1, keepdims=True)
f = load / (T * k)                                # fraction of tokens per expert
P = probs.mean(0)                                 # mean routing probability
print(f"auxiliary loss = E * sum(f*P) = {E * (f * P).sum():.4f}  (1.0 is perfectly balanced)")

# --- loss-free balancing: adjust a per-expert bias until load equalizes ---
bias = np.zeros(E)
target = T * k / E
for step in range(60):
    idx, _ = route(x, Wr, bias)
    load = load_of(idx)
    bias -= 0.02 * np.sign(load - target)
print(f"after bias balancing: load {load}  CV={load.std()/load.mean():.3f}")

# --- what MoE actually buys ---
d_ff = 4*d
dense = d*d_ff*2
moe_total, moe_active = E*dense, k*dense
print(f"\\ndense FFN params      : {dense:,}")
print(f"MoE total params      : {moe_total:,}  ({E}x)")
print(f"MoE params per token  : {moe_active:,}  ({k}x)  <- compute grows by k, not E")`),

    quiz('An MoE model has 8 experts with top-2 routing. Compared to one dense expert, what changes?',
      ['8× the parameters and memory, but only 2× the compute per token',
       '8× the parameters and 8× the compute',
       '2× the parameters and 2× the compute',
       'The same parameters, distributed differently'],
      0,
      'All 8 experts exist and must be held in memory; only 2 run per token. That is the entire proposition — parameter count (which correlates with capability) is decoupled from per-token FLOPs. The cost is memory and serving complexity, which is why MoE suits large distributed inference and not single-GPU deployment.'),

    recap(`- Distinguish **total** from **active** parameters, and say which one governs memory and which governs speed.
- Explain what the router does and why routing is per-token *and* per-layer.
- Describe expert collapse as a rich-get-richer loop, and name the two ways it is prevented.
- Say why MoE suits large distributed serving and not a single GPU.
- Explain why "experts" rarely correspond to anything a human would name.`),
  ],
  refs: [
    paper('Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer', 'Shazeer et al.', 2017, 'https://arxiv.org/abs/1701.06538', 'The modern MoE formulation.'),
    paper('Switch Transformers', 'Fedus, Zoph & Shazeer', 2021, 'https://arxiv.org/abs/2101.03961', 'Top-1 routing, the auxiliary balance loss, and scale to a trillion parameters.'),
    paper('Mixtral of Experts', 'Jiang et al.', 2024, 'https://arxiv.org/abs/2401.04088', 'The open model that made MoE mainstream.'),
    paper('DeepSeek-V3 Technical Report', 'DeepSeek-AI', 2024, 'https://arxiv.org/abs/2412.19437', 'Fine-grained experts, shared experts, and auxiliary-loss-free load balancing. Unusually detailed.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'llm-evaluation',
  title: 'Evaluating Language Models',
  sub: 'The hardest unsolved problem in applied LLM work.',
  mins: 20, level: 'core',
  prereq: ['ml-evaluation', 'llm-prompting'],
  tags: ['evaluation', 'benchmarks'],
  sections: [
    tldr(`Evaluating a classifier is easy: compare the prediction to the label, count. Evaluating a language
model is the hardest unsolved problem in applied LLM work, because there *is* no label — a good answer to
"explain recursion" is not one string, it is a vast set of them.

Everything the field does about this is a workaround. Multiple-choice benchmarks are measurable but
contaminated and narrow. Human evaluation is the gold standard and does not scale. LLM-as-judge scales and has
known biases. Arena rankings measure what people prefer, which is not the same as what is correct.

The practical conclusion is unglamorous: **build a small gold set for your own task** and trust it over any
public leaderboard.`),

    jargon([
      ['benchmark', 'A fixed dataset of questions with known answers, used to compare models. MMLU, GSM8K, HumanEval.'],
      ['MMLU', 'Massive Multitask Language Understanding — 16,000 multiple-choice questions across 57 subjects. The most-cited general benchmark, and heavily contaminated.'],
      ['contamination', 'The benchmark appearing in the training data. Turns a test into a memory check, and is nearly impossible to rule out for anything public.'],
      ['LLM-as-judge', 'Using a strong model to grade another model\'s output. Scalable, correlates decently with humans, carries systematic biases.'],
      ['position bias', 'A judge preferring whichever answer it sees first. Corrected by running both orderings.'],
      ['arena / Elo', 'Ranking models by head-to-head human preference votes, scored like chess ratings.'],
      ['gold set', 'A small set of examples from *your* actual task with answers you trust. The most useful evaluation most teams never build.'],
      ['red-teaming', 'Deliberately trying to make a model fail or misbehave, to find problems before users do.'],
      ['Goodhart\'s law', '"When a measure becomes a target, it ceases to be a good measure." The governing dynamic of every public benchmark.'],
    ]),

    t(`## Why this is hard

Classical ML evaluation compares a prediction to a label and counts. That works because the answer is a single
value from a small set.

Open-ended generation breaks every part of that. There is no single right answer — a good explanation of
recursion is any of a huge family of strings. The output space is effectively infinite. Quality is
multidimensional (correct, clear, appropriately hedged, well-formatted, safe) and those dimensions trade against
each other. And the thing you actually care about — *is this a good answer?* — is precisely the judgement you
have no metric for.

So every method below is a compromise, and the useful skill is knowing which compromise each one makes.`),

    t(`## The tiers

**Perplexity** — the pretraining loss. Correlates with capability within a model family, and is not comparable across
tokenizers (a bigger vocabulary compresses more per token). Use bits-per-byte for cross-model comparison. It stops
tracking usefulness after post-training.

**Multiple-choice benchmarks** — MMLU, ARC, HellaSwag, GPQA. Cheap and automatic. Their problems are serious:
contamination (test items are in the pretraining data), sensitivity to prompt format and answer ordering, and
saturation. MMLU is largely saturated at the frontier; GPQA and similar exist because of that.

**Generative benchmarks with verification** — HumanEval and MBPP (run the code against tests), GSM8K and MATH (check
the final answer), SWE-bench (does the patch make the test suite pass?). **Verifiable rewards are the strongest signal
available**, which is precisely why reasoning models are trained against them.

**Human preference** — Chatbot Arena's pairwise comparisons with an Elo rating. Closest to what people mean by
quality, but slow, expensive, and biased toward length, formatting, and confident tone.

**LLM-as-judge** — a strong model scores outputs. Scalable and increasingly standard, with known biases: position
(prefers the first option), length (prefers longer), and self-preference (prefers its own family's outputs). Mitigate
by randomizing order, controlling for length, and calibrating against human labels on a subset.`),

    warn(`**Contamination is pervasive and mostly unmeasured.** Benchmark test sets are on the public web and therefore
in pretraining corpora. Decontamination is done with n-gram overlap filters that miss paraphrases and translations.

Signals that a number is contaminated: performance far above similar-difficulty held-out problems, a sharp drop on a
freshly-written variant of the same benchmark, and the model reproducing test items verbatim when prompted with a
prefix.

Treat any single benchmark number in a model release as an upper bound on real capability.`),

    t(`## Building your own evals

For a real application, public benchmarks are close to irrelevant. What you need:

1. **A gold set from real traffic.** 50–200 examples of what your users actually send. This is the highest-value
   artifact in an LLM project and almost nobody builds it early enough.
2. **Task-specific automatic checks** wherever possible — did it produce valid JSON, cite a real source, stay within
   the allowed vocabulary, get the arithmetic right.
3. **An LLM judge with a rubric**, calibrated against your own human labels on a subset so you know its error rate.
4. **Regression testing on every change** — prompt edits, model version bumps, retrieval changes. Model updates
   silently move behavior.
5. **Adversarial cases**: injection attempts, out-of-scope questions, ambiguous inputs, and inputs where the correct
   answer is "I don't know."

Report **variance**. A single number from one sampled generation is a draw from a distribution, and at temperature > 0
that distribution is wide.`),

    code('Judge bias, and how to measure it', `import numpy as np
rng = np.random.default_rng(0)

def judge(a_quality, b_quality, pos_bias=0.10, len_bias=0.0,
          a_len=1.0, b_len=1.0, noise=0.25):
    """Simulated pairwise judge. Returns True if it picks A."""
    s = (a_quality - b_quality)
    s += pos_bias                                  # prefers whatever is shown first
    s += len_bias * (a_len - b_len)
    s += rng.normal(0, noise)
    return s > 0

# two genuinely equal models
N = 4000
wins_naive = sum(judge(1.0, 1.0) for _ in range(N)) / N
print(f"equal models, fixed order   : A wins {wins_naive:.1%}  <- pure position bias")

# fix 1: randomize presentation order
wins = 0
for _ in range(N):
    if rng.random() < 0.5: wins += judge(1.0, 1.0)
    else:                  wins += not judge(1.0, 1.0)
print(f"equal models, randomized    : A wins {wins/N:.1%}  <- corrected")

# length bias: a slightly WORSE but longer answer
wins = 0
for _ in range(N):
    if rng.random() < 0.5: wins += judge(0.9, 1.0, len_bias=0.35, a_len=2.0)
    else:                  wins += not judge(1.0, 0.9, len_bias=0.35, b_len=2.0)
print(f"worse but 2x longer answer  : A wins {wins/N:.1%}  <- length bias wins")

# --- how many samples do you need to detect a real difference? ---
print("\\ndetecting a true 55% win rate:")
for n in [30, 100, 300, 1000]:
    se = np.sqrt(0.55*0.45/n)
    print(f"  n={n:5d}: 95% CI = [{0.55-1.96*se:.3f}, {0.55+1.96*se:.3f}]"
          f"  {'significant' if 0.55-1.96*se > 0.5 else 'NOT significant'}")`),

    quiz('A new model reports 89% on MMLU, up from 86%. What is the appropriate reaction?',
      ['Treat it as weak evidence — contamination, prompt format, and evaluation choices easily move MMLU by a few points',
       'It is clearly a substantially better model',
       'MMLU is meaningless and should be ignored',
       'Wait for the perplexity number instead'],
      0,
      'Three points on MMLU is within the range that prompt formatting, few-shot example choice, answer-order handling, and contamination can produce on their own. It is not nothing, but it is not a capability claim either. Look for consistency across several benchmarks, performance on freshly-written held-out sets, and — for your own use — results on your own gold set.'),

    recap(`- Say why evaluating an open-ended generator is harder than evaluating a classifier.
- Explain contamination and why it is nearly impossible to rule out for a public benchmark.
- Read a benchmark improvement sceptically, and list what else you would want to see.
- Describe the known biases of LLM-as-judge, and what to do about each.
- Argue for building your own small gold set over trusting public leaderboards for your use case.`),
  ],
  refs: [
    paper('Holistic Evaluation of Language Models (HELM)', 'Liang et al.', 2022, 'https://arxiv.org/abs/2211.09110', 'Multi-metric, multi-scenario evaluation done seriously.'),
    paper('Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena', 'Zheng et al.', 2023, 'https://arxiv.org/abs/2306.05685', 'Quantifies judge biases and how well they track humans.'),
    paper('SWE-bench', 'Jimenez et al.', 2023, 'https://arxiv.org/abs/2310.06770', 'Real GitHub issues, verified by the repository\'s own tests. The model for verifiable evaluation.'),
    paper('GPQA: A Graduate-Level Google-Proof Q&A Benchmark', 'Rein et al.', 2023, 'https://arxiv.org/abs/2311.12022', 'Deliberately built to resist contamination and search.'),
    paper('Investigating Data Contamination in Modern Benchmarks', 'Deng et al.', 2023, 'https://arxiv.org/abs/2311.09783', ''),
  ],
},

];

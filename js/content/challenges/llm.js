/* ============================================================
   Challenges — track 4, transformers & LLMs

   One entry per lesson id: { title, prompt, starter, solution, hint?, explain? }
   Starters carry their own PASS/FAIL check so the learner gets a verdict
   without leaving the lab.

   EDITOR NOTE: these are JS template literals, so an unescaped backtick
   silently terminates the string. Escape inline code in prose.
   ============================================================ */

export default {

'llm-tokenization': {
  title: 'Train a BPE tokenizer, then make it charge one language more than another',
  prompt: `1. Implement the merge loop: find the most frequent adjacent pair, merge it everywhere, record the rule.
   Then encode words the tokenizer has never seen and confirm they still decompose rather than failing.
2. **Where the language tax comes from.** The second corpus is 90% "English" and 10% another language. Train
   BPE on it, then measure how many tokens each language costs per word. **Predict the ratio before running** —
   and note that nothing in the algorithm was told which language to favour.`,
  hint: 'Use `pairs.most_common(1)[0]` to get the winning pair and its count. Stop when the best count drops below 2 — a pair seen once is not worth a vocabulary slot. Applying a merge means scanning each split left to right and joining the two symbols wherever they are adjacent.',
  starter: `from collections import Counter

def train_bpe(corpus, n_merges):
    words = Counter("_" + w for w in corpus)
    splits = {w: list(w) for w in words}
    merges = []
    for step in range(n_merges):
        pairs = Counter()
        for w, freq in words.items():
            s = splits[w]
            for i in range(len(s) - 1):
                pairs[(s[i], s[i+1])] += freq
        if not pairs: break
        # TODO: take the most frequent pair and its count; stop if the count < 2;
        #       append it to merges; then rewrite every entry of splits using it
        break
    return merges

def encode(word, merges):
    s = list("_" + word)
    for a, b in merges:
        out, i = [], 0
        while i < len(s):
            if i+1 < len(s) and s[i] == a and s[i+1] == b:
                out.append(a+b); i += 2
            else:
                out.append(s[i]); i += 1
        s = out
    return s

corpus = ("the cat sat on the mat the cat ate the rat "
          "a rat sat on a hat the dog sat on the log").split()
merges = train_bpe(corpus, 14)

print(f"merges learned: {len(merges)}")
for i, (a, b) in enumerate(merges[:8]):
    print(f"  {i+1}. {a!r} + {b!r} -> {a+b!r}")

print("\\nknown words are short; unseen ones still encode, just into more pieces:")
for w in ["the", "cat", "splat", "zebra"]:
    print(f"  {w:8s} -> {encode(w, merges)}")
assert len(merges) >= 8, "the merge loop does not look implemented"
assert len(encode("the", merges)) < len(encode("zebra", merges)), \\
    "a frequent word should cost fewer tokens than an unseen one"
print("PASS\\n")

# ---------- 2. the same algorithm, on a lopsided corpus ----------
eng = "the model reads the text and the model writes the text again".split()
oth = "qux zeb qux vlim zeb qux".split()
mixed = eng*9 + oth*1                    # 90% one language, 10% the other
big = train_bpe(mixed, 60)

def cost(words, merges):
    return sum(len(encode(w, merges)) for w in words) / len(words)

c_eng, c_oth = cost(set(eng), big), cost(set(oth), big)
print(f"tokens per word, the majority language: {c_eng:.2f}")
print(f"tokens per word, the minority language: {c_oth:.2f}")
print(f"the minority language costs {c_oth/c_eng:.1f}x more per word")
print("\\nNobody chose this. The merge rule only ever asked 'which pair is most frequent'.")`,
  solution: `from collections import Counter

def train_bpe(corpus, n_merges):
    words = Counter("_" + w for w in corpus)
    splits = {w: list(w) for w in words}
    merges = []
    for step in range(n_merges):
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
                if i+1 < len(s) and s[i] == a and s[i+1] == b:
                    out.append(a+b); i += 2
                else:
                    out.append(s[i]); i += 1
            splits[w] = out
    return merges

def encode(word, merges):
    s = list("_" + word)
    for a, b in merges:
        out, i = [], 0
        while i < len(s):
            if i+1 < len(s) and s[i] == a and s[i+1] == b:
                out.append(a+b); i += 2
            else:
                out.append(s[i]); i += 1
        s = out
    return s

corpus = ("the cat sat on the mat the cat ate the rat "
          "a rat sat on a hat the dog sat on the log").split()
merges = train_bpe(corpus, 14)

print(f"merges learned: {len(merges)}")
for i, (a, b) in enumerate(merges[:8]):
    print(f"  {i+1}. {a!r} + {b!r} -> {a+b!r}")

print("\\nknown words are short; unseen ones still encode, just into more pieces:")
for w in ["the", "cat", "splat", "zebra"]:
    print(f"  {w:8s} -> {encode(w, merges)}")
assert len(merges) >= 8
assert len(encode("the", merges)) < len(encode("zebra", merges))
print("PASS\\n")

eng = "the model reads the text and the model writes the text again".split()
oth = "qux zeb qux vlim zeb qux".split()
mixed = eng*9 + oth*1
big = train_bpe(mixed, 60)

def cost(words, merges):
    return sum(len(encode(w, merges)) for w in words) / len(words)

c_eng, c_oth = cost(set(eng), big), cost(set(oth), big)
print(f"tokens per word, the majority language: {c_eng:.2f}")
print(f"tokens per word, the minority language: {c_oth:.2f}")
print(f"the minority language costs {c_oth/c_eng:.1f}x more per word")
print("\\nNobody chose this. The merge rule only ever asked 'which pair is most frequent'.")`,
  explain: `Part 1: notice that "zebra" — a word the tokenizer has never seen — does not fail. It comes back as
several pieces. That is the property byte-level BPE was built for: there is no such thing as an unknown input,
only an expensive one.

Part 2 is where the practical consequences come from. The merge rule contains no notion of language, fairness,
or importance — it asks only which adjacent pair is most frequent, and a corpus that is 90% one language will
spend nearly all its merges on that language's patterns. The minority language never accumulates merges, so its
words stay split into small fragments and cost several times more tokens each.

Now follow that through to what a user experiences. More tokens per word means: fewer words fit in the context
window, generation is slower, and — since APIs bill per token — the same paragraph costs several times more to
process. Real tokenizers show a 2–4× penalty for languages under-represented in their training data. It is not
a policy decision anyone made; it is a frequency count, doing exactly what it was asked.`,
},

'llm-attention': {
  title: 'Build causal attention, break the scaling, then add a KV cache',
  prompt: `1. Implement scaled dot-product attention with a causal mask. The assertions check that rows sum to 1
   and that no position can see the future.
2. **Why the $\\sqrt{d_k}$ is there.** Remove the scaling and watch the softmax saturate into a hard argmax as
   $d_k$ grows — and then watch the *gradient* vanish along with it, using
   $\\partial p_i/\\partial z_i = p_i(1-p_i)$ from the softmax derivation.
3. **The KV cache.** Generate five tokens one at a time, once recomputing everything and once caching past keys
   and values. Confirm the outputs are identical, then count the work each did.`,
  hint: 'Masking: build a lower-triangular matrix of ones and use `np.where(mask == 1, scores, -np.inf)`, *before* the softmax — `exp(-inf)` is 0, so those positions get exactly no weight. For the cache, note that a past token\'s key and value never change when a new token is appended.',
  starter: `import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def attention(X, Wq, Wk, Wv, causal=True, scale=True):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    # TODO: scores = Q K^T, divided by sqrt(d_k) when scale=True;
    #       apply the causal mask when causal=True; softmax; then weight V.
    return X, None

rng = np.random.default_rng(0)
T, d, dk = 6, 32, 16
X = rng.normal(size=(T, d))
Wq, Wk, Wv = (rng.normal(0, d**-0.5, (d, dk)) for _ in range(3))

out, A = attention(X, Wq, Wk, Wv)
assert A is not None and A.shape == (T, T), "return the attention matrix as well"
assert np.allclose(A.sum(1), 1.0), "each row must sum to 1"
assert np.allclose(np.triu(A, 1), 0.0), "the causal mask is not applied"
print("attention matrix (row i = where token i looks):")
print(np.round(A, 3))
print("PASS\\n")

# ---------- 2. what the scaling prevents ----------
print(f"{'d_k':>6} {'logit sd':>10} {'max weight':>12} {'gradient p(1-p)':>17}   (unscaled)")
for dk_t in [16, 128, 1024, 8192]:
    q = rng.normal(size=dk_t); k = rng.normal(size=(8, dk_t))
    raw = k @ q
    p = softmax(raw).max()
    ps = softmax(raw/np.sqrt(dk_t)).max()
    print(f"{dk_t:6d} {raw.std():10.2f} {p:12.5f} {p*(1-p):17.2e}"
          f"      scaled: max {ps:.3f}, grad {ps*(1-ps):.3f}")

# ---------- 3. the KV cache ----------
def generate_naive(X0, steps):
    """Recompute keys and values for the whole prefix at every step."""
    X, work = X0.copy(), 0
    for _ in range(steps):
        o, _ = attention(X, Wq, Wk, Wv)
        work += len(X)                            # keys/values computed this step
        X = np.vstack([X, o[-1] @ np.linalg.pinv(Wv) ])   # feed the output back in
    return X, work

def generate_cached(X0, steps):
    """Compute each token's key and value once, then reuse them."""
    X, work = X0.copy(), 0
    Kc, Vc = X @ Wk, X @ Wv
    work += len(X)
    for _ in range(steps):
        q = (X[-1] @ Wq)
        s = Kc @ q / np.sqrt(dk)
        o = softmax(s) @ Vc
        nxt = o @ np.linalg.pinv(Wv)
        X = np.vstack([X, nxt])
        Kc = np.vstack([Kc, nxt @ Wk])            # one new key
        Vc = np.vstack([Vc, nxt @ Wv])            # one new value
        work += 1
    return X, work

a, w_naive  = generate_naive(X, 5)
b, w_cached = generate_cached(X, 5)
print(f"\\nsame tokens generated either way? {np.allclose(a, b)}")
print(f"key/value vectors computed -- without a cache: {w_naive}, with one: {w_cached}")`,
  solution: `import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def attention(X, Wq, Wk, Wv, causal=True, scale=True):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    d_k = Q.shape[-1]
    scores = Q @ K.T
    if scale: scores = scores / np.sqrt(d_k)
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
assert A is not None and A.shape == (T, T)
assert np.allclose(A.sum(1), 1.0)
assert np.allclose(np.triu(A, 1), 0.0)
print("attention matrix (row i = where token i looks):")
print(np.round(A, 3))
print("PASS\\n")

print(f"{'d_k':>6} {'logit sd':>10} {'max weight':>12} {'gradient p(1-p)':>17}   (unscaled)")
for dk_t in [16, 128, 1024, 8192]:
    q = rng.normal(size=dk_t); k = rng.normal(size=(8, dk_t))
    raw = k @ q
    p = softmax(raw).max()
    ps = softmax(raw/np.sqrt(dk_t)).max()
    print(f"{dk_t:6d} {raw.std():10.2f} {p:12.5f} {p*(1-p):17.2e}"
          f"      scaled: max {ps:.3f}, grad {ps*(1-ps):.3f}")

def generate_naive(X0, steps):
    X, work = X0.copy(), 0
    for _ in range(steps):
        o, _ = attention(X, Wq, Wk, Wv)
        work += len(X)
        X = np.vstack([X, o[-1] @ np.linalg.pinv(Wv)])
    return X, work

def generate_cached(X0, steps):
    X, work = X0.copy(), 0
    Kc, Vc = X @ Wk, X @ Wv
    work += len(X)
    for _ in range(steps):
        q = (X[-1] @ Wq)
        s = Kc @ q / np.sqrt(dk)
        o = softmax(s) @ Vc
        nxt = o @ np.linalg.pinv(Wv)
        X = np.vstack([X, nxt])
        Kc = np.vstack([Kc, nxt @ Wk])
        Vc = np.vstack([Vc, nxt @ Wv])
        work += 1
    return X, work

a, w_naive  = generate_naive(X, 5)
b, w_cached = generate_cached(X, 5)
print(f"\\nsame tokens generated either way? {np.allclose(a, b)}")
print(f"key/value vectors computed -- without a cache: {w_naive}, with one: {w_cached}")`,
  explain: `Part 2 makes the derivation's argument visible in three columns. The logit standard deviation grows
like $\\sqrt{d_k}$ exactly as predicted, and by $d_k = 1024$ the unscaled softmax is putting essentially all of
its weight on one key. Then look at the last column: $p(1-p)$ is the softmax's own derivative, and once $p$ is
pinned at 1 that derivative is effectively zero. The model is not merely attending sharply — it has lost the
ability to *change* where it attends, because no gradient reaches the scores. Dividing by $\\sqrt{d_k}$ holds the
logits at unit scale whatever the width, and the scaled columns stay responsive at every $d_k$.

Part 3 is the KV cache. The two routines produce identical tokens, because a past token's key and value do not
depend on anything that comes after it — recomputing them is pure waste. The work counts show the difference:
without a cache you recompute the whole prefix at every step, which grows quadratically with the length of what
you generate; with one, each token is computed once.

The catch is that you are now storing those keys and values, and that store grows with sequence length, layers,
and heads. For a long conversation the cache can exceed the model's own weights, which is the entire reason
grouped-query attention exists — several query heads sharing one key/value head shrinks it by the sharing
factor.`,
},

'llm-transformer': {
  title: 'Build the block, verify RoPE is relative, and watch the residual stream grow',
  prompt: `1. **RoPE.** Implement the rotation, then check the lesson's claim directly: a query at position 8 and a
   key at position 3 should score exactly the same as a query at 30 and a key at 25. Same gap, same score.
2. **The block.** Assemble a pre-norm block — RMSNorm, causal attention, residual add, RMSNorm, MLP, residual
   add — and stack eight of them.
3. **The bus.** The lesson said layers *add* to a shared stream rather than overwriting it. Measure the stream's
   size at every depth, and measure how much of the final state came from the original embedding.`,
  hint: 'RoPE treats the vector as $d/2$ pairs $(x_0,x_1), (x_2,x_3), \\ldots$ and rotates each pair by its own angle: $x\'_{even} = x_{even}\\cos\\theta - x_{odd}\\sin\\theta$ and $x\'_{odd} = x_{even}\\sin\\theta + x_{odd}\\cos\\theta$.',
  starter: `import numpy as np

def rmsnorm(x, g=1.0, eps=1e-6):
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps) * g

def rope(x, base=10000):
    T, d = x.shape
    pos = np.arange(T)[:, None]
    i = np.arange(0, d, 2)[None, :]
    theta = pos / base ** (i / d)
    c, s = np.cos(theta), np.sin(theta)
    out = x.copy()
    # TODO: rotate each pair (x[:, 0::2], x[:, 1::2]) by theta
    return out

# ---------- 1. is RoPE really relative? ----------
d = 16
q0 = np.random.default_rng(0).normal(size=d)
k0 = np.random.default_rng(1).normal(size=d)

def score_at(m, n):
    a = np.zeros((max(m, n)+1, d)); a[m] = q0
    b = np.zeros((max(m, n)+1, d)); b[n] = k0
    return float(rope(a)[m] @ rope(b)[n])

print(f"query at 8,  key at 3   -> {score_at(8, 3):.6f}")
print(f"query at 30, key at 25  -> {score_at(30, 25):.6f}    same gap, same score")
print(f"query at 8,  key at 2   -> {score_at(8, 2):.6f}    different gap, different score")
assert abs(score_at(8, 3) - score_at(30, 25)) < 1e-6, "RoPE must depend only on m - n"
assert abs(score_at(8, 3) - score_at(8, 2)) > 1e-3,  "a different gap should give a different score"
print("PASS\\n")

# ---------- 2. a stack of pre-norm blocks ----------
rng = np.random.default_rng(0)
T, D, H = 12, 32, 4
def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

def make_block():
    return dict(Wq=rng.normal(0, D**-0.5, (D, D)), Wk=rng.normal(0, D**-0.5, (D, D)),
                Wv=rng.normal(0, D**-0.5, (D, D)), Wo=rng.normal(0, D**-0.5, (D, D)),
                W1=rng.normal(0, D**-0.5, (D, 4*D)), W2=rng.normal(0, (4*D)**-0.5, (4*D, D)))

def attention(x, p):
    q, k, v = rope(x @ p["Wq"]), rope(x @ p["Wk"]), x @ p["Wv"]
    s = q @ k.T / np.sqrt(D)
    s = np.where(np.tril(np.ones((T, T))) == 1, s, -np.inf)
    return (softmax(s) @ v) @ p["Wo"]

def block(x, p):
    x = x + attention(rmsnorm(x), p)                     # sublayer 1: mix across positions
    x = x + (np.maximum(0, rmsnorm(x) @ p["W1"]) @ p["W2"])   # sublayer 2: per position
    return x

blocks = [make_block() for _ in range(8)]
x0 = rng.normal(size=(T, D))
x = x0.copy()
sizes = [np.abs(x).mean()]
for p in blocks:
    x = block(x, p)
    sizes.append(np.abs(x).mean())

print("average size of the residual stream, by depth:")
for L, v in enumerate(sizes):
    print(f"  after {L} blocks: {v:.3f}")

# ---------- 3. how much of the original survives? ----------
keep = float(np.abs(np.sum(x0*x)) / (np.linalg.norm(x0)*np.linalg.norm(x)))
print(f"\\ncosine between the input embedding and the final state: {keep:.3f}")
print("Nothing overwrote it. Every block only ever added.")

print(f"\\nparameters per block: {4*D*D + 8*D*D:,}    the 12*d^2 rule: {12*D*D:,}")`,
  solution: `import numpy as np

def rmsnorm(x, g=1.0, eps=1e-6):
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps) * g

def rope(x, base=10000):
    T, d = x.shape
    pos = np.arange(T)[:, None]
    i = np.arange(0, d, 2)[None, :]
    theta = pos / base ** (i / d)
    c, s = np.cos(theta), np.sin(theta)
    out = x.copy()
    even, odd = x[:, 0::2], x[:, 1::2]
    out[:, 0::2] = even*c - odd*s
    out[:, 1::2] = even*s + odd*c
    return out

d = 16
q0 = np.random.default_rng(0).normal(size=d)
k0 = np.random.default_rng(1).normal(size=d)

def score_at(m, n):
    a = np.zeros((max(m, n)+1, d)); a[m] = q0
    b = np.zeros((max(m, n)+1, d)); b[n] = k0
    return float(rope(a)[m] @ rope(b)[n])

print(f"query at 8,  key at 3   -> {score_at(8, 3):.6f}")
print(f"query at 30, key at 25  -> {score_at(30, 25):.6f}    same gap, same score")
print(f"query at 8,  key at 2   -> {score_at(8, 2):.6f}    different gap, different score")
assert abs(score_at(8, 3) - score_at(30, 25)) < 1e-6
assert abs(score_at(8, 3) - score_at(8, 2)) > 1e-3
print("PASS\\n")

rng = np.random.default_rng(0)
T, D, H = 12, 32, 4
def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

def make_block():
    return dict(Wq=rng.normal(0, D**-0.5, (D, D)), Wk=rng.normal(0, D**-0.5, (D, D)),
                Wv=rng.normal(0, D**-0.5, (D, D)), Wo=rng.normal(0, D**-0.5, (D, D)),
                W1=rng.normal(0, D**-0.5, (D, 4*D)), W2=rng.normal(0, (4*D)**-0.5, (4*D, D)))

def attention(x, p):
    q, k, v = rope(x @ p["Wq"]), rope(x @ p["Wk"]), x @ p["Wv"]
    s = q @ k.T / np.sqrt(D)
    s = np.where(np.tril(np.ones((T, T))) == 1, s, -np.inf)
    return (softmax(s) @ v) @ p["Wo"]

def block(x, p):
    x = x + attention(rmsnorm(x), p)
    x = x + (np.maximum(0, rmsnorm(x) @ p["W1"]) @ p["W2"])
    return x

blocks = [make_block() for _ in range(8)]
x0 = rng.normal(size=(T, D))
x = x0.copy()
sizes = [np.abs(x).mean()]
for p in blocks:
    x = block(x, p)
    sizes.append(np.abs(x).mean())

print("average size of the residual stream, by depth:")
for L, v in enumerate(sizes):
    print(f"  after {L} blocks: {v:.3f}")

keep = float(np.abs(np.sum(x0*x)) / (np.linalg.norm(x0)*np.linalg.norm(x)))
print(f"\\ncosine between the input embedding and the final state: {keep:.3f}")
print("Nothing overwrote it. Every block only ever added.")

print(f"\\nparameters per block: {4*D*D + 8*D*D:,}    the 12*d^2 rule: {12*D*D:,}")`,
  explain: `Part 1 confirms RoPE's defining property. The same *gap* between query and key gives the same score
wherever in the sequence the pair sits — 8-and-3 matches 30-and-25 to six decimals — while a different gap gives
a different score. The model therefore learns about relative distance rather than absolute slots, which is why
RoPE has no maximum position baked into it, unlike a lookup table of learned position vectors.

Part 3 is the residual-stream picture, measured. The stream's magnitude climbs monotonically with depth,
because each block *adds* and nothing subtracts — which is exactly why a final normalization before the output
head is standard practice rather than an afterthought. And the original embedding is still clearly present in
the final state: the cosine of about 0.28 should be read against the $1/\\sqrt{d}$ baseline from
[the first lesson](#/l/math-vectors), which for 384 numbers is about 0.05. Eight layers of additions later, the
input is still five times more aligned with the output than chance would explain. No layer ever replaced it.

That is the difference between the two mental models. In a pipeline picture, layer 8's output is a
transformation of layer 7's and the input is long gone. In the bus picture, the input is still sitting there
alongside eight layers' worth of additions, any of which a later layer can read from selectively. It is why
interpretability research talks about circuits that skip across layers, and why removing a single middle layer
from a trained transformer often barely changes its output.`,
},

'llm-pretraining': {
  title: 'Train a language model, then check whether it learned anything or just memorized',
  prompt: `1. Implement the backward pass and train a fixed-context character model — embed, concatenate,
   one hidden layer, softmax over the next character. Watch perplexity fall, then sample from it.
2. **Did it learn language, or the corpus?** The last block measures loss on the training text and on held-out
   sentences drawn from the same small vocabulary. **Predict both numbers before running.** Recall from
   [the information lesson](#/l/math-information) that a model which has learned nothing scores $\\log V$.`,
  hint: 'The softmax-plus-cross-entropy gradient is `p - onehot(y)`, divided by the batch size. From there it is the two-layer backward pass you have written before, plus `np.add.at(E, X, ...)` to scatter gradients back into the embedding rows that were used.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

pool = ["the quick brown fox jumps over the lazy dog. ",
        "the lazy dog sleeps by the warm fire. ",
        "a quick fox runs through the tall grass. ",
        "a brown dog jumps over a lazy fox. ",
        "the small cat sleeps on the soft rug. ",
        "a happy bird flies above the green hill. "]
train_txt = "".join(pool[:4]) * 30          # what the model sees
held_txt  = "".join(pool[4:])               # same words, sentences it never sees

chars = sorted(set("".join(pool))); stoi = {c: i for i, c in enumerate(chars)}
V, CTX, D = len(chars), 8, 24
train = np.array([stoi[c] for c in train_txt])
held  = np.array([stoi[c] for c in held_txt])

E  = rng.normal(0, 0.1, (V, D))             # token embeddings
Pe = rng.normal(0, 0.1, (CTX, D))           # positional embeddings
W1 = rng.normal(0, (CTX*D)**-0.5, (CTX*D, 64))
W2 = rng.normal(0, 64**-0.5, (64, V))

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

def batch(d, n=64):
    i = rng.integers(0, len(d)-CTX-1, n)
    return np.stack([d[j:j+CTX] for j in i]), np.array([d[j+CTX] for j in i])

for step in range(3001):
    X, y = batch(train)
    h0 = (E[X] + Pe[None]).reshape(len(X), -1)
    h1 = np.maximum(0, h0 @ W1)
    p  = softmax(h1 @ W2)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    # TODO: backprop into W2, W1, E and Pe, then take an SGD step at lr = 0.2.
    #       Use np.add.at(E, X, -lr * dh0) to accumulate into the used embedding rows.
    if step % 750 == 0:
        print(f"step {step:4d}  loss {loss:.4f}  perplexity {np.exp(loss):6.2f}")

assert loss < 0.5, "the model should fit this corpus easily -- check the backward pass"
print("PASS\\n")

# --- sample from it ---
ctx, out = [stoi[c] for c in "the quic"], "the quic"
for _ in range(70):
    x = np.array(ctx[-CTX:])
    p = softmax(np.maximum(0, (E[x] + Pe).reshape(1, -1) @ W1) @ W2)[0]
    nxt = rng.choice(V, p=p)
    out += chars[nxt]; ctx.append(nxt)
print("generated:", out)

# --- did it learn language, or this text? ---
def evaluate(d):
    X = np.stack([d[j:j+CTX] for j in range(len(d)-CTX-1)])
    y = np.array([d[j+CTX] for j in range(len(d)-CTX-1)])
    p = softmax(np.maximum(0, (E[X] + Pe[None]).reshape(len(X), -1) @ W1) @ W2)
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())

print(f"\\nloss on the training text : {evaluate(train):.3f}")
print(f"loss on held-out sentences: {evaluate(held):.3f}")
print(f"loss of a model that knows nothing (log V): {np.log(V):.3f}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

pool = ["the quick brown fox jumps over the lazy dog. ",
        "the lazy dog sleeps by the warm fire. ",
        "a quick fox runs through the tall grass. ",
        "a brown dog jumps over a lazy fox. ",
        "the small cat sleeps on the soft rug. ",
        "a happy bird flies above the green hill. "]
train_txt = "".join(pool[:4]) * 30
held_txt  = "".join(pool[4:])

chars = sorted(set("".join(pool))); stoi = {c: i for i, c in enumerate(chars)}
V, CTX, D = len(chars), 8, 24
train = np.array([stoi[c] for c in train_txt])
held  = np.array([stoi[c] for c in held_txt])

E  = rng.normal(0, 0.1, (V, D))
Pe = rng.normal(0, 0.1, (CTX, D))
W1 = rng.normal(0, (CTX*D)**-0.5, (CTX*D, 64))
W2 = rng.normal(0, 64**-0.5, (64, V))

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

def batch(d, n=64):
    i = rng.integers(0, len(d)-CTX-1, n)
    return np.stack([d[j:j+CTX] for j in i]), np.array([d[j+CTX] for j in i])

lr = 0.2
for step in range(3001):
    X, y = batch(train)
    h0 = (E[X] + Pe[None]).reshape(len(X), -1)
    h1 = np.maximum(0, h0 @ W1)
    p  = softmax(h1 @ W2)
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

assert loss < 0.5
print("PASS\\n")

ctx, out = [stoi[c] for c in "the quic"], "the quic"
for _ in range(70):
    x = np.array(ctx[-CTX:])
    p = softmax(np.maximum(0, (E[x] + Pe).reshape(1, -1) @ W1) @ W2)[0]
    nxt = rng.choice(V, p=p)
    out += chars[nxt]; ctx.append(nxt)
print("generated:", out)

def evaluate(d):
    X = np.stack([d[j:j+CTX] for j in range(len(d)-CTX-1)])
    y = np.array([d[j+CTX] for j in range(len(d)-CTX-1)])
    p = softmax(np.maximum(0, (E[X] + Pe[None]).reshape(len(X), -1) @ W1) @ W2)
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())

print(f"\\nloss on the training text : {evaluate(train):.3f}")
print(f"loss on held-out sentences: {evaluate(held):.3f}")
print(f"loss of a model that knows nothing (log V): {np.log(V):.3f}")`,
  explain: `Part 1: the loss falls to near zero and the samples read like the training text. Everything in the
pipeline works — this is a real language model, and it is essentially the architecture Bengio et al. published in
2003, before attention existed.

Part 2 is the uncomfortable part, and it is worth sitting with. Training loss is around 0.02. Held-out loss is
around **5.8**, which is *worse* than $\\log V \\approx 3.3$ — the loss of a model that has learned nothing at all
and guesses uniformly. This model did not learn English. It memorised four sentences, and on anything else it is
worse than useless, because it is confidently predicting the continuations it memorised.

The lesson claimed that a narrow objective produces broad capability *because the data is broad*. This is that
claim seen from the other side: the objective here is identical to GPT's, the architecture is a scaled-down
version of the same idea, and the result is a lookup table. Nothing was wrong with the objective. There were
four sentences.

That is also why the data pipeline in the lesson matters so much, and why deduplication is one of its stages.
Repeated text does not add information — it just gives the model more opportunity to do what this one did.`,
},

'llm-scaling': {
  title: 'Predict a big model from small ones, then find the compute-optimal split',
  prompt: `1. **The claim that justifies the spending.** Pretend you can only afford small runs. Fit a power law to
   four cheap measurements and use it to predict the loss of a run **1000× larger**, then check the prediction
   against the truth. This is exactly what a lab does before committing to a frontier run.
2. **How to spend a budget.** Write \`optimal(budget)\`, which finds the parameter/token split minimising loss
   subject to $C = 6ND$. Then look at where GPT-3, Chinchilla and Llama-3 actually sit.`,
  hint: 'A power law is a straight line in log-log space: $\\log(L - E) = \\log A - \\alpha \\log N$. So subtract the floor, take logs of both axes, and fit a straight line with `np.polyfit`. For part 2, sweep $N$ on a log grid, set $D = C/(6N)$, and keep the best.',
  starter: `import numpy as np

# the Chinchilla fitted form -- we will treat it as "the truth" and try to rediscover it
E, A, alpha, B, beta = 1.69, 406.4, 0.34, 410.7, 0.28
L = lambda N, D: E + A/N**alpha + B/D**beta

# ---------- 1. extrapolating from cheap runs ----------
small_N = np.array([3e7, 1e8, 3e8, 1e9])          # four models you can actually afford
D_fixed = 2e11
rng = np.random.default_rng(0)
measured = np.array([L(n, D_fixed) for n in small_N]) * (1 + rng.normal(0, 0.002, 4))

def fit_power_law(N, loss, floor):
    """Return (A_hat, alpha_hat) fitting loss = floor + A * N**(-alpha)."""
    # TODO: subtract the floor, take logs of both, fit a line, convert back
    return 0.0, 0.0

A_hat, a_hat = fit_power_law(small_N, measured, E + B/D_fixed**beta)
print(f"fitted from 4 small runs:  A = {A_hat:8.1f}   alpha = {a_hat:.4f}")
print(f"the true values:           A = {A:8.1f}   alpha = {alpha:.4f}\\n")

big_N = 1e12                                       # 1000x the largest one you fitted on
pred  = E + B/D_fixed**beta + A_hat * big_N**(-a_hat)
truth = L(big_N, D_fixed)
print(f"predicted loss at {big_N:.0e} params: {pred:.4f}")
print(f"actual loss                        : {truth:.4f}")
print(f"error: {abs(pred-truth):.5f} nats, from data spanning only 30x")
assert abs(pred - truth) < 0.02, "the extrapolation should be accurate to a couple of hundredths"
print("PASS\\n")

# ---------- 2. spending a budget ----------
def optimal(budget):
    """Return (loss, N, D) minimising L subject to 6*N*D = budget."""
    # TODO: sweep N over a log grid, set D from the budget, keep the lowest loss
    return np.inf, 0.0, 0.0

def fmt(x):
    for div, s in [(1e12, "T"), (1e9, "B"), (1e6, "M")]:
        if x >= div: return f"{x/div:.1f}{s}"
    return f"{x:.0f}"

print(f"{'budget':>10} {'params':>9} {'tokens':>9} {'tok/param':>10} {'loss':>8}")
for lg in [20, 21, 22, 23, 24]:
    loss, N, D = optimal(10**lg)
    print(f"  1e{lg:<7} {fmt(N):>9} {fmt(D):>9} {D/max(N,1):>10.1f} {loss:>8.4f}")

print("\\nreal models, at the compute they actually used:")
for name, N, D in [("GPT-3", 175e9, 300e9), ("Chinchilla", 70e9, 1.4e12),
                   ("Llama-3-8B", 8e9, 15e12)]:
    opt_loss, oN, oD = optimal(6*N*D)
    print(f"  {name:12s} {D/N:7.1f} tok/param   loss {L(N,D):.4f}   "
          f"best possible at that budget: {fmt(oN)} params, loss {opt_loss:.4f}")`,
  solution: `import numpy as np

E, A, alpha, B, beta = 1.69, 406.4, 0.34, 410.7, 0.28
L = lambda N, D: E + A/N**alpha + B/D**beta

small_N = np.array([3e7, 1e8, 3e8, 1e9])
D_fixed = 2e11
rng = np.random.default_rng(0)
measured = np.array([L(n, D_fixed) for n in small_N]) * (1 + rng.normal(0, 0.002, 4))

def fit_power_law(N, loss, floor):
    y = np.log(loss - floor)
    x = np.log(N)
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.exp(intercept)), float(-slope)

A_hat, a_hat = fit_power_law(small_N, measured, E + B/D_fixed**beta)
print(f"fitted from 4 small runs:  A = {A_hat:8.1f}   alpha = {a_hat:.4f}")
print(f"the true values:           A = {A:8.1f}   alpha = {alpha:.4f}\\n")

big_N = 1e12
pred  = E + B/D_fixed**beta + A_hat * big_N**(-a_hat)
truth = L(big_N, D_fixed)
print(f"predicted loss at {big_N:.0e} params: {pred:.4f}")
print(f"actual loss                        : {truth:.4f}")
print(f"error: {abs(pred-truth):.5f} nats, from data spanning only 30x")
assert abs(pred - truth) < 0.02
print("PASS\\n")

def optimal(budget):
    best = (np.inf, 0.0, 0.0)
    for lgN in np.arange(7, 13, 0.005):
        N = 10**lgN
        D = budget / (6*N)
        if D < 1e8: continue
        l = L(N, D)
        if l < best[0]: best = (l, N, D)
    return best

def fmt(x):
    for div, s in [(1e12, "T"), (1e9, "B"), (1e6, "M")]:
        if x >= div: return f"{x/div:.1f}{s}"
    return f"{x:.0f}"

print(f"{'budget':>10} {'params':>9} {'tokens':>9} {'tok/param':>10} {'loss':>8}")
for lg in [20, 21, 22, 23, 24]:
    loss, N, D = optimal(10**lg)
    print(f"  1e{lg:<7} {fmt(N):>9} {fmt(D):>9} {D/max(N,1):>10.1f} {loss:>8.4f}")

print("\\nreal models, at the compute they actually used:")
for name, N, D in [("GPT-3", 175e9, 300e9), ("Chinchilla", 70e9, 1.4e12),
                   ("Llama-3-8B", 8e9, 15e12)]:
    opt_loss, oN, oD = optimal(6*N*D)
    print(f"  {name:12s} {D/N:7.1f} tok/param   loss {L(N,D):.4f}   "
          f"best possible at that budget: {fmt(oN)} params, loss {opt_loss:.4f}")`,
  explain: `Part 1 is the reason frontier training is fundable. Four small runs, spanning a factor of 30 in size,
with measurement noise on top — and the fitted exponent lands on the true one, letting you predict the loss of a
model a *thousand times* larger to within a rounding error. That is not a property of models in general; it is a
property of power laws, which are straight lines in log-log space, and straight lines extrapolate.

Sit with what this buys. Before Kaplan et al., committing \\$100M to a training run meant hoping. After, you run
a ladder of cheap models over a weekend, fit a line, and know roughly what you are buying. The scaling law did
not make models better — it made the spending decision rational, and the spending is what made the models better.

Part 2: the optimal split keeps parameters and tokens growing together — every tenfold increase in budget buys
roughly three times the parameters and three times the data, rather than pouring it all into size. That balance
is Chinchilla's central finding. (The exact ratio this parametric form gives drifts from about 40 to about 100
tokens per parameter across these budgets; the widely quoted "20 tokens per parameter" comes from the paper's
other two estimation methods. The ratio is fit-dependent — the *balance* is the robust result.)

Then look at the three real models. GPT-3 sits at 1.7 tokens per parameter — trained under the earlier Kaplan
recipe, and a smaller model on the same compute would have scored better. Chinchilla sits right on the optimum,
which was the paper's demonstration. And Llama-3-8B sits at nearly 2000, a hundred times "optimal" — which is not
a mistake but a different objective. It gives up a little loss to be permanently cheaper to serve, and if you are
answering billions of requests, that trade is not close.`,
},

'llm-finetuning': {
  title: 'Implement LoRA, verify the merge, then test the bet it rests on',
  prompt: `1. Write the LoRA forward pass. Confirm it is *exactly* the identity at initialization, count the
   parameter saving, and check that merging $BA$ into $W$ gives identical outputs.
2. **The bet.** LoRA assumes the update a task needs is approximately low rank. Train a LoRA adapter at several
   ranks against two different targets: one whose required update genuinely is rank 2, and one whose required
   update is full rank. **Predict which curve flattens and which keeps improving.**`,
  hint: 'The forward pass is $xW + \\frac{\\alpha}{r}(xA)B$; merged, it is $x(W + \\frac{\\alpha}{r}AB)$. For part 2, with the loss being squared error and $W$ frozen, you can fit the adapter by gradient descent on $A$ and $B$ — the gradients are the ordinary chain rule through two matrix multiplies.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

d_in, d_out, r, alpha = 512, 512, 8, 16
W = rng.normal(0, d_in**-0.5, (d_in, d_out))
A = rng.normal(0, 0.01, (d_in, r))
B = np.zeros((r, d_out))              # zero, so the adapter starts as a no-op

def forward(x, A, B, merged=False):
    # TODO: x @ W + (alpha/r) * ((x @ A) @ B), or the merged equivalent
    return x @ W

x = rng.normal(size=(4, d_in))
assert np.allclose(forward(x, A, B), x @ W), "at init, LoRA must change nothing"
print("PASS at init\\n")

full, lora = d_in*d_out, d_in*r + r*d_out
print(f"full fine-tune : {full:,} parameters")
print(f"LoRA (r={r})    : {lora:,} parameters  ({lora/full:.2%})")
print(f"optimizer state saved: {(full-lora)*16/1e6:.0f} MB per layer")

Btrained = rng.normal(0, 0.05, B.shape)
print(f"\\nmerged == unmerged: "
      f"{np.allclose(forward(x, A, Btrained), forward(x, A, Btrained, merged=True), atol=1e-9)}")

# ---------- 2. is the update actually low rank? ----------
n, d = 256, 64
Wf = rng.normal(0, d**-0.5, (d, d))
X  = rng.normal(size=(n, d))

low  = rng.normal(size=(d, 2)) @ rng.normal(size=(2, d)) * 0.3     # a rank-2 change
fullr = rng.normal(0, 0.3/np.sqrt(d), (d, d))                       # a full-rank change

def fit_lora(target_delta, rank, steps=3000, lr=0.02):
    Y = X @ (Wf + target_delta)
    a = rng.normal(0, 0.02, (d, rank)); b = np.zeros((rank, d))
    for _ in range(steps):
        pred = X @ Wf + (X @ a) @ b
        g = 2*(pred - Y)/n
        ga = X.T @ (g @ b.T); gb = (X @ a).T @ g
        a -= lr*ga; b -= lr*gb
    return float(np.mean((X @ Wf + (X @ a) @ b - Y)**2) / np.mean((X @ target_delta)**2))

print("\\nrelative error of the fitted adapter (1.0 = learned nothing):")
print(f"{'rank':>6} {'target is rank 2':>18} {'target is full rank':>21}")
for rank in [1, 2, 4, 8, 16]:
    print(f"{rank:6d} {fit_lora(low, rank):18.4f} {fit_lora(fullr, rank):21.4f}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

d_in, d_out, r, alpha = 512, 512, 8, 16
W = rng.normal(0, d_in**-0.5, (d_in, d_out))
A = rng.normal(0, 0.01, (d_in, r))
B = np.zeros((r, d_out))

def forward(x, A, B, merged=False):
    if merged:
        return x @ (W + (alpha/r) * (A @ B))
    return x @ W + (alpha/r) * ((x @ A) @ B)

x = rng.normal(size=(4, d_in))
assert np.allclose(forward(x, A, B), x @ W)
print("PASS at init\\n")

full, lora = d_in*d_out, d_in*r + r*d_out
print(f"full fine-tune : {full:,} parameters")
print(f"LoRA (r={r})    : {lora:,} parameters  ({lora/full:.2%})")
print(f"optimizer state saved: {(full-lora)*16/1e6:.0f} MB per layer")

Btrained = rng.normal(0, 0.05, B.shape)
print(f"\\nmerged == unmerged: "
      f"{np.allclose(forward(x, A, Btrained), forward(x, A, Btrained, merged=True), atol=1e-9)}")

n, d = 256, 64
Wf = rng.normal(0, d**-0.5, (d, d))
X  = rng.normal(size=(n, d))

low  = rng.normal(size=(d, 2)) @ rng.normal(size=(2, d)) * 0.3
fullr = rng.normal(0, 0.3/np.sqrt(d), (d, d))

def fit_lora(target_delta, rank, steps=3000, lr=0.02):
    Y = X @ (Wf + target_delta)
    a = rng.normal(0, 0.02, (d, rank)); b = np.zeros((rank, d))
    for _ in range(steps):
        pred = X @ Wf + (X @ a) @ b
        g = 2*(pred - Y)/n
        ga = X.T @ (g @ b.T); gb = (X @ a).T @ g
        a -= lr*ga; b -= lr*gb
    return float(np.mean((X @ Wf + (X @ a) @ b - Y)**2) / np.mean((X @ target_delta)**2))

print("\\nrelative error of the fitted adapter (1.0 = learned nothing):")
print(f"{'rank':>6} {'target is rank 2':>18} {'target is full rank':>21}")
for rank in [1, 2, 4, 8, 16]:
    print(f"{rank:6d} {fit_lora(low, rank):18.4f} {fit_lora(fullr, rank):21.4f}")`,
  explain: `Part 1: because $B$ starts at zero, the adapter contributes exactly nothing at step 0 — the model you
begin fine-tuning from is bit-for-bit the model you started with, which is why LoRA never damages a checkpoint
just by being attached. And since $\\frac{\\alpha}{r}AB$ is only a matrix, you can add it into $W$ when you are
done: the merged model has the same shape, the same speed, and no adapter to carry around.

Part 2 is the empirical bet, isolated. Against the rank-2 target, error collapses as soon as the adapter has
rank 2 and adding more rank buys nothing — the adapter has *exactly* enough capacity, and any extra is wasted.
Against the full-rank target, every increase in rank keeps helping and even rank 16 leaves a lot on the table,
because a rank-16 matrix simply cannot express a rank-64 change.

That is the whole of LoRA's risk in two columns. It is not a compression trick that works generally; it works
when the change your task needs happens to be simple. Empirically, adapting a pretrained model to a domain or a
style usually is — which is why rank 8 or 16 is enough for most fine-tuning. Teaching a model a genuinely new
capability often is not, and that is when people find LoRA underperforming full fine-tuning and reach for a
higher rank.`,
},

'llm-decoding': {
  title: 'Implement every sampler, and watch the nucleus adapt',
  prompt: `Write temperature, top-k, top-p and min-p filtering. Then show that top-p keeps one token when the model is
confident and many when it is not — the adaptivity that made it the default.`,
  hint: 'For top-p: sort descending, cumulative-sum, keep up to the first index where the cumsum reaches p.',
  starter: `import numpy as np

vocab = ["the","a","this","my","our","cat","dog","quantum","zebra","xylophone"]
logits = np.array([3.9, 3.2, 2.4, 2.0, 1.6, 0.8, 0.5, -1.4, -2.2, -3.5])

def softmax(z, T=1.0):
    z = z/T; z = z - z.max(); e = np.exp(z); return e/e.sum()

def top_k(p, k):
    # TODO
    return p

def top_p(p, thresh):
    # TODO
    return p

def min_p(p, frac):
    # TODO: keep tokens with prob >= frac * max(prob)
    return p

def entropy(p):
    q = p[p > 0]; return -(q*np.log2(q)).sum()

print(f"{'strategy':20s} {'kept':>5} {'entropy':>8}  top-3")
for name, p in [("raw (T=1)", softmax(logits)), ("T=0.5", softmax(logits, 0.5)),
                ("T=1.5", softmax(logits, 1.5)), ("top-k=3", top_k(softmax(logits), 3)),
                ("top-p=0.9", top_p(softmax(logits), 0.9)),
                ("min-p=0.1", min_p(softmax(logits), 0.1))]:
    top3 = ", ".join(f"{vocab[i]}={p[i]:.2f}" for i in np.argsort(-p)[:3])
    print(f"{name:20s} {int((p>0).sum()):5d} {entropy(p):8.3f}  {top3}")

print("\\nnucleus size adapts to confidence:")
for desc, lg in [("confident", np.array([8.,1.,0.,0.,0.,0.,0.,0.,0.,0.])),
                 ("uncertain", np.array([1.,.9,.8,.8,.7,.7,.6,.6,.5,.5]))]:
    print(f"  {desc:10s} -> top-p 0.9 keeps {(top_p(softmax(lg), 0.9) > 0).sum()} tokens")`,
  solution: `import numpy as np

vocab = ["the","a","this","my","our","cat","dog","quantum","zebra","xylophone"]
logits = np.array([3.9, 3.2, 2.4, 2.0, 1.6, 0.8, 0.5, -1.4, -2.2, -3.5])

def softmax(z, T=1.0):
    z = z/T; z = z - z.max(); e = np.exp(z); return e/e.sum()

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
    q = p[p > 0]; return -(q*np.log2(q)).sum()

print(f"{'strategy':20s} {'kept':>5} {'entropy':>8}  top-3")
for name, p in [("raw (T=1)", softmax(logits)), ("T=0.5", softmax(logits, 0.5)),
                ("T=1.5", softmax(logits, 1.5)), ("top-k=3", top_k(softmax(logits), 3)),
                ("top-p=0.9", top_p(softmax(logits), 0.9)),
                ("min-p=0.1", min_p(softmax(logits), 0.1))]:
    top3 = ", ".join(f"{vocab[i]}={p[i]:.2f}" for i in np.argsort(-p)[:3])
    print(f"{name:20s} {int((p>0).sum()):5d} {entropy(p):8.3f}  {top3}")

print("\\nnucleus size adapts to confidence:")
for desc, lg in [("confident", np.array([8.,1.,0.,0.,0.,0.,0.,0.,0.,0.])),
                 ("uncertain", np.array([1.,.9,.8,.8,.7,.7,.6,.6,.5,.5]))]:
    print(f"  {desc:10s} -> top-p 0.9 keeps {(top_p(softmax(lg), 0.9) > 0).sum()} tokens")`,
},

'llm-prompting': {
  title: 'Simulate self-consistency and find where it stops helping',
  prompt: `Model majority voting over sampled reasoning chains. Show it helps when wrong answers scatter and does
nothing when the model is *consistently* wrong — the failure mode people forget.`,
  hint: 'Majority voting exploits the fact that there are many ways to be wrong and one way to be right.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def majority_correct(p_correct, n_samples, n_wrong_modes, trials=3000):
    """Fraction of trials where the majority answer is correct."""
    wins = 0
    for _ in range(trials):
        counts = {}
        for _ in range(n_samples):
            # TODO: with prob p_correct the answer is "correct",
            #       otherwise it is one of n_wrong_modes distinct wrong answers
            ans = "correct"
            counts[ans] = counts.get(ans, 0) + 1
        if max(counts, key=counts.get) == "correct": wins += 1
    return wins / trials

print("p=0.4, wrong answers scattered over 8 modes:")
for n in [1, 3, 5, 11, 21, 41]:
    print(f"  {n:3d} samples -> {majority_correct(0.4, n, 8):.3f}")

print("\\nsame p, but how concentrated the wrong answers are:")
for modes in [1, 2, 4, 8, 20]:
    print(f"  {modes:2d} wrong mode(s) -> {majority_correct(0.4, 21, modes):.3f}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def majority_correct(p_correct, n_samples, n_wrong_modes, trials=3000):
    wins = 0
    for _ in range(trials):
        counts = {}
        for _ in range(n_samples):
            if rng.random() < p_correct:
                ans = "correct"
            else:
                ans = f"wrong_{rng.integers(n_wrong_modes)}"
            counts[ans] = counts.get(ans, 0) + 1
        if max(counts, key=counts.get) == "correct": wins += 1
    return wins / trials

print("p=0.4, wrong answers scattered over 8 modes:")
for n in [1, 3, 5, 11, 21, 41]:
    print(f"  {n:3d} samples -> {majority_correct(0.4, n, 8):.3f}")

print("\\nsame p, but how concentrated the wrong answers are:")
for modes in [1, 2, 4, 8, 20]:
    print(f"  {modes:2d} wrong mode(s) -> {majority_correct(0.4, 21, modes):.3f}")
print("\\nWith 1 wrong mode, voting is useless: the model is systematically wrong.")`,
},

'llm-rag': {
  title: 'Build hybrid retrieval with reciprocal rank fusion',
  prompt: `Implement BM25 and combine it with dense similarity using RRF. Find a query where keyword search wins and
one where dense search wins, and confirm the hybrid gets both right.`,
  hint: 'RRF needs no score calibration: sum 1/(k + rank) across rankings, with k around 60.',
  starter: `import numpy as np, re
from collections import Counter

docs = [
 "The transformer architecture was introduced in Attention Is All You Need in 2017.",
 "LoRA freezes pretrained weights and injects trainable low-rank matrices.",
 "BPE tokenization merges the most frequent adjacent pair of symbols repeatedly.",
 "Chinchilla found compute-optimal training uses about 20 tokens per parameter.",
 "FlashAttention reduces attention memory from quadratic to linear by tiling.",
 "RMSNorm removes mean-subtraction from LayerNorm and is used in Llama.",
]
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
N = len(docs)
avgdl = np.mean([len(tok(d)) for d in docs])
df = Counter(w for d in docs for w in set(tok(d)))

def bm25(query, k1=1.5, b=0.75):
    scores = np.zeros(N)
    for i, d in enumerate(docs):
        tf, dl = Counter(tok(d)), len(tok(d))
        for w in tok(query):
            if w not in tf: continue
            # TODO: idf * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))
            pass
    return scores

def rrf(*rankings, k=60):
    score = np.zeros(N)
    # TODO: for each ranking, add 1/(k + rank + 1) to the doc at that rank
    return score

vocab = sorted({w for d in docs for w in tok(d)})
vi = {w:i for i,w in enumerate(vocab)}
W = np.random.default_rng(0).normal(size=(len(vocab), 32))
def embed(s):
    ws = [vi[w] for w in tok(s) if w in vi]
    if not ws: return np.zeros(32)
    v = W[ws].mean(0); return v/(np.linalg.norm(v)+1e-9)
D = np.stack([embed(d) for d in docs])

for q in ["how many tokens per parameter?", "what does LoRA freeze?"]:
    s_sparse, s_dense = bm25(q), D @ embed(q)
    print(f"\\nQ: {q}")
    for name, s in [("BM25", s_sparse), ("dense", s_dense), ("hybrid", rrf(s_sparse, s_dense))]:
        print(f"  {name:8s} -> {docs[int(np.argmax(s))][:60]}...")`,
  solution: `import numpy as np, re
from collections import Counter

docs = [
 "The transformer architecture was introduced in Attention Is All You Need in 2017.",
 "LoRA freezes pretrained weights and injects trainable low-rank matrices.",
 "BPE tokenization merges the most frequent adjacent pair of symbols repeatedly.",
 "Chinchilla found compute-optimal training uses about 20 tokens per parameter.",
 "FlashAttention reduces attention memory from quadratic to linear by tiling.",
 "RMSNorm removes mean-subtraction from LayerNorm and is used in Llama.",
]
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
N = len(docs)
avgdl = np.mean([len(tok(d)) for d in docs])
df = Counter(w for d in docs for w in set(tok(d)))

def bm25(query, k1=1.5, b=0.75):
    scores = np.zeros(N)
    for i, d in enumerate(docs):
        tf, dl = Counter(tok(d)), len(tok(d))
        for w in tok(query):
            if w not in tf: continue
            idf = np.log((N - df[w] + 0.5)/(df[w] + 0.5) + 1)
            scores[i] += idf * tf[w]*(k1+1) / (tf[w] + k1*(1-b+b*dl/avgdl))
    return scores

def rrf(*rankings, k=60):
    score = np.zeros(N)
    for r in rankings:
        for rank, idx in enumerate(np.argsort(-r)):
            score[idx] += 1/(k + rank + 1)
    return score

vocab = sorted({w for d in docs for w in tok(d)})
vi = {w:i for i,w in enumerate(vocab)}
W = np.random.default_rng(0).normal(size=(len(vocab), 32))
def embed(s):
    ws = [vi[w] for w in tok(s) if w in vi]
    if not ws: return np.zeros(32)
    v = W[ws].mean(0); return v/(np.linalg.norm(v)+1e-9)
D = np.stack([embed(d) for d in docs])

for q in ["how many tokens per parameter?", "what does LoRA freeze?"]:
    s_sparse, s_dense = bm25(q), D @ embed(q)
    print(f"\\nQ: {q}")
    for name, s in [("BM25", s_sparse), ("dense", s_dense), ("hybrid", rrf(s_sparse, s_dense))]:
        print(f"  {name:8s} -> {docs[int(np.argmax(s))][:60]}...")`,
  explain: 'Dense retrieval misses rare exact terms; BM25 misses paraphrases. RRF combines rankings without needing the two score scales to be comparable, which is why it is the standard hybrid-search recipe.',
},

'llm-moe': {
  title: 'Route tokens to experts, then fix the collapse',
  prompt: `Implement top-k routing and measure load imbalance. Then implement the loss-free balancing trick: a
per-expert bias nudged until the loads equalize.`,
  hint: 'Adjust each expert bias by -lr * sign(load - target) and re-route until balanced.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

E, k, d, T = 8, 2, 32, 2000
x = rng.normal(size=(T, d))
Wr = rng.normal(0, d**-0.5, (d, E))
Wr[:, :2] += 0.35                      # give experts 0,1 a head start

def route(x, Wr, bias=None):
    logits = x @ Wr
    if bias is not None: logits = logits + bias
    # TODO: take the top-k experts per token, return their indices
    return np.zeros((len(x), k), dtype=int)

def load_of(idx):
    return np.bincount(idx.ravel(), minlength=E)

load = load_of(route(x, Wr))
print(f"no balancing:  load {load}  CV={load.std()/load.mean():.3f}")

bias = np.zeros(E)
target = T * k / E
for step in range(60):
    load = load_of(route(x, Wr, bias))
    # TODO: nudge bias to equalize load
    pass
print(f"after balancing: load {load}  CV={load.std()/load.mean():.3f}")

dense = d * 4*d * 2
print(f"\\ndense FFN      : {dense:,} params")
print(f"MoE total      : {E*dense:,} ({E}x)")
print(f"MoE per token  : {k*dense:,} ({k}x)  <- compute grows by k, not E")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

E, k, d, T = 8, 2, 32, 2000
x = rng.normal(size=(T, d))
Wr = rng.normal(0, d**-0.5, (d, E))
Wr[:, :2] += 0.35

def route(x, Wr, bias=None):
    logits = x @ Wr
    if bias is not None: logits = logits + bias
    return np.argsort(-logits, axis=1)[:, :k]

def load_of(idx): return np.bincount(idx.ravel(), minlength=E)

load = load_of(route(x, Wr))
print(f"no balancing:    load {load}  CV={load.std()/load.mean():.3f}")

bias = np.zeros(E)
target = T * k / E
for step in range(60):
    load = load_of(route(x, Wr, bias))
    bias -= 0.02 * np.sign(load - target)
print(f"after balancing: load {load}  CV={load.std()/load.mean():.3f}")

dense = d * 4*d * 2
print(f"\\ndense FFN      : {dense:,} params")
print(f"MoE total      : {E*dense:,} ({E}x)")
print(f"MoE per token  : {k*dense:,} ({k}x)  <- compute grows by k, not E")`,
  explain: 'Without balancing the router funnels everything to its favourites and the other experts never train. The bias trick (DeepSeek-V3) equalizes load without adding an auxiliary loss that would fight the language modeling objective.',
},

'llm-evaluation': {
  title: 'Measure position and length bias in an LLM judge',
  prompt: `Simulate a pairwise judge with known biases. Show that comparing two *identical* models gives a win rate
well above 50% due to position bias, and that randomizing order fixes it.`,
  hint: 'Also check: how many comparisons do you need to detect a true 55% win rate?',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def judge(a_q, b_q, pos_bias=0.10, len_bias=0.0, a_len=1.0, b_len=1.0, noise=0.25):
    s = (a_q - b_q) + pos_bias + len_bias*(a_len - b_len) + rng.normal(0, noise)
    return s > 0

N = 4000
# TODO: fixed order -- two EQUAL models, how often does A win?
naive = 0.0
print(f"equal models, fixed order   : A wins {naive:.1%}")

# TODO: randomized presentation order
rand = 0.0
print(f"equal models, randomized    : A wins {rand:.1%}")

# TODO: a slightly worse (0.9 vs 1.0) but 2x longer answer, with len_bias=0.35
longer = 0.0
print(f"worse but 2x longer         : A wins {longer:.1%}")

print("\\ndetecting a true 55% win rate:")
for n in [30, 100, 300, 1000]:
    se = np.sqrt(0.55*0.45/n)
    sig = "significant" if 0.55 - 1.96*se > 0.5 else "NOT significant"
    print(f"  n={n:5d}: 95% CI [{0.55-1.96*se:.3f}, {0.55+1.96*se:.3f}]  {sig}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def judge(a_q, b_q, pos_bias=0.10, len_bias=0.0, a_len=1.0, b_len=1.0, noise=0.25):
    s = (a_q - b_q) + pos_bias + len_bias*(a_len - b_len) + rng.normal(0, noise)
    return s > 0

N = 4000
naive = sum(judge(1.0, 1.0) for _ in range(N)) / N
print(f"equal models, fixed order   : A wins {naive:.1%}  <- pure position bias")

wins = 0
for _ in range(N):
    if rng.random() < 0.5: wins += judge(1.0, 1.0)
    else:                  wins += not judge(1.0, 1.0)
print(f"equal models, randomized    : A wins {wins/N:.1%}  <- corrected")

wins = 0
for _ in range(N):
    if rng.random() < 0.5: wins += judge(0.9, 1.0, len_bias=0.35, a_len=2.0)
    else:                  wins += not judge(1.0, 0.9, len_bias=0.35, b_len=2.0)
print(f"worse but 2x longer         : A wins {wins/N:.1%}  <- length bias wins")

print("\\ndetecting a true 55% win rate:")
for n in [30, 100, 300, 1000]:
    se = np.sqrt(0.55*0.45/n)
    sig = "significant" if 0.55 - 1.96*se > 0.5 else "NOT significant"
    print(f"  n={n:5d}: 95% CI [{0.55-1.96*se:.3f}, {0.55+1.96*se:.3f}]  {sig}")`,
},

};

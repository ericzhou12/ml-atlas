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
  title: 'Train a BPE tokenizer',
  prompt: `Implement the merge loop: repeatedly find the most frequent adjacent symbol pair and merge it. Then encode a
word the tokenizer has never seen and confirm it still decomposes rather than failing.`,
  hint: 'Count pairs across all words weighted by word frequency, merge the argmax, and apply that merge everywhere.',
  starter: `from collections import Counter

corpus = ("the cat sat on the mat the cat ate the rat "
          "a rat sat on a hat the dog sat on the log").split()

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
    # TODO: pick the most frequent pair, stop if its count < 2,
    #       record it in merges, and apply it to every entry of splits
    break

print("merges learned:", len(merges))
for i, (a, b) in enumerate(merges[:8]):
    print(f"  {i+1}. {a!r} + {b!r} -> {a+b!r}")

def encode(word):
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

for w in ["the", "cat", "splat", "zebra"]:
    print(f"  {w:8s} -> {encode(w)}")`,
  solution: `from collections import Counter

corpus = ("the cat sat on the mat the cat ate the rat "
          "a rat sat on a hat the dog sat on the log").split()

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
            if i+1 < len(s) and s[i] == a and s[i+1] == b:
                out.append(a+b); i += 2
            else:
                out.append(s[i]); i += 1
        splits[w] = out
    print(f"merge {step+1:2d}: {a!r} + {b!r} -> {(a+b)!r}  (seen {count}x)")

def encode(word):
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

print()
for w in ["the", "cat", "splat", "zebra"]:
    print(f"  {w:8s} -> {encode(w)}")`,
  explain: 'Frequent words collapse to one token; unseen words fall back to pieces. Nothing is ever out-of-vocabulary, which is the entire point of subword tokenization.',
},

'llm-attention': {
  title: 'Causal self-attention, and why the 1/sqrt(d) matters',
  prompt: `Implement scaled dot-product attention with a causal mask. Then remove the scaling and show that as
d_k grows the softmax saturates into a hard argmax — which kills the gradient.`,
  hint: 'Dot products of random unit-variance vectors have variance d_k, so their standard deviation is sqrt(d_k).',
  starter: `import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def attention(X, Wq, Wk, Wv, causal=True, scale=True):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    # TODO: scores = QK^T (scaled), apply the causal mask, softmax, weight V
    return X, None

rng = np.random.default_rng(0)
T, d, dk = 6, 32, 16
X = rng.normal(size=(T, d))
Wq, Wk, Wv = (rng.normal(0, d**-0.5, (d, dk)) for _ in range(3))

out, A = attention(X, Wq, Wk, Wv)
assert A is not None and A.shape == (T, T), "return the attention matrix too"
assert np.allclose(A.sum(1), 1.0), "rows must sum to 1"
assert np.allclose(np.triu(A, 1), 0.0), "causal mask not applied"
print("PASS\\n")
print(np.round(A, 3))

print("\\nwithout scaling, as d_k grows:")
for dk_t in [16, 128, 1024]:
    q = rng.normal(size=dk_t); k = rng.normal(size=(8, dk_t))
    raw = k @ q
    print(f"  d_k={dk_t:5d}  logit sd {raw.std():7.2f}  "
          f"unscaled max weight {softmax(raw).max():.4f}  "
          f"scaled {softmax(raw/np.sqrt(dk_t)).max():.4f}")`,
  solution: `import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def attention(X, Wq, Wk, Wv, causal=True, scale=True):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T
    if scale: scores = scores / np.sqrt(Q.shape[-1])
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
assert np.allclose(A.sum(1), 1.0) and np.allclose(np.triu(A, 1), 0.0)
print("PASS\\n"); print(np.round(A, 3))

print("\\nwithout scaling, as d_k grows:")
for dk_t in [16, 128, 1024]:
    q = rng.normal(size=dk_t); k = rng.normal(size=(8, dk_t))
    raw = k @ q
    print(f"  d_k={dk_t:5d}  logit sd {raw.std():7.2f}  "
          f"unscaled max weight {softmax(raw).max():.4f}  "
          f"scaled {softmax(raw/np.sqrt(dk_t)).max():.4f}")`,
  explain: 'At d_k = 1024 the unscaled softmax puts essentially all mass on one key. A saturated softmax has near-zero gradient, so the model could never learn to attend differently. One division fixes it.',
},

'llm-transformer': {
  title: 'Assemble a transformer block with RoPE',
  prompt: `Build a pre-norm block: RMSNorm, multi-head causal attention with rotary embeddings, residual, RMSNorm,
SwiGLU MLP, residual. Then verify that RoPE makes attention scores depend only on relative position.`,
  hint: 'RoPE rotates pairs of dimensions by an angle proportional to position. Check that q at 8 vs k at 3 gives the same score as 30 vs 25.',
  starter: `import numpy as np

def rmsnorm(x, g, eps=1e-6):
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps) * g

def rope(x, base=10000):
    T, d = x.shape
    pos = np.arange(T)[:, None]
    i = np.arange(0, d, 2)[None, :]
    theta = pos / base ** (i / d)
    c, s = np.cos(theta), np.sin(theta)
    out = x.copy()
    # TODO: rotate (x[:,0::2], x[:,1::2]) by theta
    return out

# --- relative-position check ---
d = 16
q0 = np.random.default_rng(0).normal(size=d)
k0 = np.random.default_rng(1).normal(size=d)

def score_at(m, n):
    seq = np.zeros((max(m, n) + 1, d))
    seq[m] = q0; qm = rope(seq)[m]
    seq2 = np.zeros((max(m, n) + 1, d))
    seq2[n] = k0; kn = rope(seq2)[n]
    return qm @ kn

print(f"score(m=8,  n=3)  = {score_at(8, 3):.6f}")
print(f"score(m=30, n=25) = {score_at(30, 25):.6f}   <- same offset, same score")
assert abs(score_at(8,3) - score_at(30,25)) < 1e-6, "RoPE should depend only on m-n"
print("\\nPASS")`,
  solution: `import numpy as np

def rmsnorm(x, g, eps=1e-6):
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps) * g

def rope(x, base=10000):
    T, d = x.shape
    pos = np.arange(T)[:, None]
    i = np.arange(0, d, 2)[None, :]
    theta = pos / base ** (i / d)
    c, s = np.cos(theta), np.sin(theta)
    out = x.copy()
    out[:, 0::2] = x[:, 0::2]*c - x[:, 1::2]*s
    out[:, 1::2] = x[:, 0::2]*s + x[:, 1::2]*c
    return out

d = 16
q0 = np.random.default_rng(0).normal(size=d)
k0 = np.random.default_rng(1).normal(size=d)

def score_at(m, n):
    a = np.zeros((max(m,n)+1, d)); a[m] = q0
    b = np.zeros((max(m,n)+1, d)); b[n] = k0
    return rope(a)[m] @ rope(b)[n]

print(f"score(m=8,  n=3)  = {score_at(8, 3):.6f}")
print(f"score(m=30, n=25) = {score_at(30, 25):.6f}")
assert abs(score_at(8,3) - score_at(30,25)) < 1e-6
print("\\nPASS -- the score depends only on m-n, with no position vector added anywhere.")`,
},

'llm-pretraining': {
  title: 'Train a character-level language model',
  prompt: `Build a fixed-context neural language model (embed, concatenate, one hidden layer, softmax) and train it
until perplexity drops. Then sample from it and watch fluency improve with context length.`,
  hint: 'This is the Bengio et al. 2003 architecture — no attention, and it still works.',
  starter: `import numpy as np

text = ("the quick brown fox jumps over the lazy dog. "
        "the lazy dog sleeps. the quick fox runs. "
        "a brown dog jumps over a lazy fox. ") * 40

chars = sorted(set(text)); stoi = {c:i for i,c in enumerate(chars)}
V, CTX, D = len(chars), 8, 24
data = np.array([stoi[c] for c in text])

rng = np.random.default_rng(0)
E  = rng.normal(0, 0.1, (V, D))
Pe = rng.normal(0, 0.1, (CTX, D))
W1 = rng.normal(0, (CTX*D)**-0.5, (CTX*D, 64))
W2 = rng.normal(0, 64**-0.5, (64, V))

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

def batch(n=64):
    i = rng.integers(0, len(data)-CTX-1, n)
    return np.stack([data[j:j+CTX] for j in i]), np.array([data[j+CTX] for j in i])

for step in range(3001):
    X, y = batch()
    h0 = (E[X] + Pe[None]).reshape(len(X), -1)
    h1 = np.maximum(0, h0 @ W1)
    p = softmax(h1 @ W2)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    # TODO: backprop into W2, W1, E, Pe and take an SGD step at lr=0.2
    if step % 750 == 0:
        print(f"step {step:4d}  loss {loss:.4f}  perplexity {np.exp(loss):6.2f}")`,
  solution: `import numpy as np

text = ("the quick brown fox jumps over the lazy dog. "
        "the lazy dog sleeps. the quick fox runs. "
        "a brown dog jumps over a lazy fox. ") * 40

chars = sorted(set(text)); stoi = {c:i for i,c in enumerate(chars)}
V, CTX, D = len(chars), 8, 24
data = np.array([stoi[c] for c in text])

rng = np.random.default_rng(0)
E  = rng.normal(0, 0.1, (V, D))
Pe = rng.normal(0, 0.1, (CTX, D))
W1 = rng.normal(0, (CTX*D)**-0.5, (CTX*D, 64))
W2 = rng.normal(0, 64**-0.5, (64, V))

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

def batch(n=64):
    i = rng.integers(0, len(data)-CTX-1, n)
    return np.stack([data[j:j+CTX] for j in i]), np.array([data[j+CTX] for j in i])

lr = 0.2
for step in range(3001):
    X, y = batch()
    h0 = (E[X] + Pe[None]).reshape(len(X), -1)
    h1 = np.maximum(0, h0 @ W1)
    p = softmax(h1 @ W2)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()

    dlog = p.copy(); dlog[np.arange(len(y)), y] -= 1; dlog /= len(y)
    dW2 = h1.T @ dlog
    dh1 = (dlog @ W2.T) * (h1 > 0)
    dW1 = h0.T @ dh1
    dh0 = (dh1 @ W1.T).reshape(len(X), CTX, D)
    W2 -= lr*dW2; W1 -= lr*dW1
    np.add.at(E, X, -lr*dh0)
    Pe -= lr*dh0.sum(0)

    if step % 750 == 0:
        print(f"step {step:4d}  loss {loss:.4f}  perplexity {np.exp(loss):6.2f}")

ctx = [stoi[c] for c in "the quic"]; out = "the quic"
for _ in range(80):
    h = np.maximum(0, (E[np.array(ctx[-CTX:])] + Pe).reshape(1,-1) @ W1)
    nxt = rng.choice(V, p=softmax(h @ W2)[0])
    out += chars[nxt]; ctx.append(nxt)
print("\\ngenerated:", out)`,
},

'llm-scaling': {
  title: 'Find the compute-optimal model size',
  prompt: `Using the Chinchilla parametric loss, write a function that finds the optimal parameter/token split for a
given FLOP budget. Then check where GPT-3 and Llama-3 sit relative to it.`,
  hint: 'Compute is C = 6ND. Sweep N on a log grid, set D = C/(6N), and take the argmin of the loss.',
  starter: `import numpy as np

E, A, alpha, B, beta = 1.69, 406.4, 0.34, 410.7, 0.28
L = lambda N, D: E + A/N**alpha + B/D**beta

def optimal(budget):
    """Return (loss, N, D) minimizing L subject to 6ND = budget."""
    # TODO
    return np.inf, 0, 0

def fmt(x):
    for div, s in [(1e12,"T"), (1e9,"B"), (1e6,"M")]:
        if x >= div: return f"{x/div:.1f}{s}"
    return f"{x:.0f}"

print(f"{'budget':>10} {'params':>9} {'tokens':>9} {'tok/param':>10} {'loss':>8}")
for lg in [20, 21, 22, 23, 24]:
    loss, N, D = optimal(10**lg)
    print(f"  1e{lg:<7} {fmt(N):>9} {fmt(D):>9} {D/max(N,1):>10.1f} {loss:>8.4f}")

print("\\nreal models:")
for name, N, D in [("GPT-3", 175e9, 300e9), ("Chinchilla", 70e9, 1.4e12),
                   ("Llama-3-8B", 8e9, 15e12)]:
    opt_loss, oN, oD = optimal(6*N*D)
    print(f"  {name:12s} {D/N:7.1f} tok/param   loss {L(N,D):.4f}   "
          f"optimal would be {fmt(oN)} params at loss {opt_loss:.4f}")`,
  solution: `import numpy as np

E, A, alpha, B, beta = 1.69, 406.4, 0.34, 410.7, 0.28
L = lambda N, D: E + A/N**alpha + B/D**beta

def optimal(budget):
    best = (np.inf, 0, 0)
    for lgN in np.arange(7, 13, 0.005):
        N = 10**lgN
        D = budget / (6*N)
        if D < 1e8: continue
        l = L(N, D)
        if l < best[0]: best = (l, N, D)
    return best

def fmt(x):
    for div, s in [(1e12,"T"), (1e9,"B"), (1e6,"M")]:
        if x >= div: return f"{x/div:.1f}{s}"
    return f"{x:.0f}"

print(f"{'budget':>10} {'params':>9} {'tokens':>9} {'tok/param':>10} {'loss':>8}")
for lg in [20, 21, 22, 23, 24]:
    loss, N, D = optimal(10**lg)
    print(f"  1e{lg:<7} {fmt(N):>9} {fmt(D):>9} {D/N:>10.1f} {loss:>8.4f}")

print("\\nreal models:")
for name, N, D in [("GPT-3", 175e9, 300e9), ("Chinchilla", 70e9, 1.4e12),
                   ("Llama-3-8B", 8e9, 15e12)]:
    opt_loss, oN, oD = optimal(6*N*D)
    print(f"  {name:12s} {D/N:7.1f} tok/param   loss {L(N,D):.4f}   "
          f"optimal: {fmt(oN)} params, loss {opt_loss:.4f}")`,
  explain: 'The optimum sits near 20 tokens per parameter. GPT-3 at 1.7 is far undertrained; Llama-3-8B at 1875 is deliberately overtrained, trading a little loss for permanently cheaper inference.',
},

'llm-finetuning': {
  title: 'Implement LoRA and verify the merge',
  prompt: `Write the LoRA forward pass, confirm it is exactly the identity at initialization, count the parameter
saving, and check that merging BA into W gives bit-identical outputs.`,
  hint: 'B initializes to zero so BA = 0 at step 0. The scale factor is alpha/r.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

d_in, d_out, r, alpha = 512, 512, 8, 16
W = rng.normal(0, d_in**-0.5, (d_in, d_out))
A = rng.normal(0, 0.01, (d_in, r))
B = np.zeros((r, d_out))

def forward(x, merged=False):
    # TODO: return x @ W + (alpha/r) * ((x @ A) @ B), or the merged equivalent
    return x @ W

x = rng.normal(size=(4, d_in))
assert np.allclose(forward(x), x @ W), "at init, LoRA must be the identity"
print("PASS at init\\n")

full = d_in*d_out
lora = d_in*r + r*d_out
print(f"full fine-tune : {full:,} params")
print(f"LoRA (r={r})    : {lora:,} params  ({lora/full:.2%})")
print(f"optimizer state saved: {(full-lora)*16/1e6:.0f} MB per layer")

B[:] = rng.normal(0, 0.05, B.shape)     # pretend we trained
print(f"\\nmerged == unmerged: {np.allclose(forward(x), forward(x, merged=True), atol=1e-9)}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

d_in, d_out, r, alpha = 512, 512, 8, 16
W = rng.normal(0, d_in**-0.5, (d_in, d_out))
A = rng.normal(0, 0.01, (d_in, r))
B = np.zeros((r, d_out))

def forward(x, merged=False):
    if merged:
        return x @ (W + (alpha/r) * (A @ B))
    return x @ W + (alpha/r) * ((x @ A) @ B)

x = rng.normal(size=(4, d_in))
assert np.allclose(forward(x), x @ W)
print("PASS at init\\n")

full, lora = d_in*d_out, d_in*r + r*d_out
print(f"full fine-tune : {full:,} params")
print(f"LoRA (r={r})    : {lora:,} params  ({lora/full:.2%})")
print(f"optimizer state saved: {(full-lora)*16/1e6:.0f} MB per layer")

B[:] = rng.normal(0, 0.05, B.shape)
print(f"\\nmerged == unmerged: {np.allclose(forward(x), forward(x, merged=True), atol=1e-9)}")
print("-> adapters fold into W after training, so inference costs nothing extra.")`,
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

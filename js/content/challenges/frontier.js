/* ============================================================
   Challenges — track 9, frontier topics

   EDITOR NOTE: these are JS template literals, so an unescaped backtick
   silently terminates the string. Escape inline code in prose.
   ============================================================ */

export default {

'fr-interpretability': {
  title: 'Reproduce superposition in a toy model',
  prompt: `Train a model that must squeeze n features through m < n dimensions. Show that dense features get only m
orthogonal slots, while sparse features get packed in non-orthogonally with tolerable interference.`,
  hint: 'Sweep the sparsity. Measure how many features survive with large norm, and the mean absolute cosine between them.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

n_feat, m_dim = 12, 4

def train(sparsity, steps=3000, lr=0.02):
    W = rng.normal(0, 0.3, (n_feat, m_dim))
    b = np.zeros(n_feat)
    imp = 0.85 ** np.arange(n_feat)          # features differ in importance
    for _ in range(steps):
        B = 256
        # each feature is active with prob (1 - sparsity)
        x = rng.random((B, n_feat)) * (rng.random((B, n_feat)) > sparsity)
        h = x @ W
        out = np.maximum(0, h @ W.T + b)
        # TODO: squared error weighted by imp, then gradient step on W and b
    return W

print(f"{n_feat} features into {m_dim} dimensions\\n")
print(f"{'sparsity':>9} {'features stored':>16} {'mean |cos|':>12}")
for sp in [0.0, 0.5, 0.8, 0.95, 0.99]:
    W = train(sp)
    norms = np.linalg.norm(W, axis=1)
    stored = int((norms > 0.35*norms.max()).sum())
    Wn = W / (norms[:,None] + 1e-9)
    G = np.abs(Wn @ Wn.T)
    off = G[~np.eye(n_feat, dtype=bool)]
    print(f"{sp:9.2f} {stored:16d} {off.mean():12.3f}")

W_dense, W_sparse = train(0.0), train(0.99)
count = lambda W: int((np.linalg.norm(W, axis=1) > 0.1).sum())
assert count(W_sparse) > count(W_dense), \\
    "sparse features should let the model pack in more than the dimension allows"
print("\\nPASS")
`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

n_feat, m_dim = 12, 4

def train(sparsity, steps=3000, lr=0.02):
    W = rng.normal(0, 0.3, (n_feat, m_dim))
    b = np.zeros(n_feat)
    imp = 0.85 ** np.arange(n_feat)
    for _ in range(steps):
        B = 256
        x = rng.random((B, n_feat)) * (rng.random((B, n_feat)) > sparsity)
        h = x @ W
        out = np.maximum(0, h @ W.T + b)
        err = (out - x) * imp
        d = 2 * err * (out > 0) / B
        W -= lr * (d.T @ h + (x.T @ (d @ W)))
        b -= lr * d.sum(0)
    return W

print(f"{n_feat} features into {m_dim} dimensions\\n")
print(f"{'sparsity':>9} {'features stored':>16} {'mean |cos|':>12}")
for sp in [0.0, 0.5, 0.8, 0.95, 0.99]:
    W = train(sp)
    norms = np.linalg.norm(W, axis=1)
    stored = int((norms > 0.35*norms.max()).sum())
    Wn = W / (norms[:,None] + 1e-9)
    G = np.abs(Wn @ Wn.T)
    off = G[~np.eye(n_feat, dtype=bool)]
    print(f"{sp:9.2f} {stored:16d} {off.mean():12.3f}")
print("\\nDense: ~4 features stored orthogonally. Sparse: many more packed in,")
print("accepting interference because collisions are rare. That is superposition.")

W_dense, W_sparse = train(0.0), train(0.99)
count = lambda W: int((np.linalg.norm(W, axis=1) > 0.1).sum())
assert count(W_sparse) > count(W_dense), \\
    "sparse features should let the model pack in more than the dimension allows"
print("\\nPASS")
`,
  explain: 'The model discovers a compression scheme on its own. This is why individual neurons are polysemantic, and why recovering features needs a wider, sparse basis rather than reading neurons directly.',
},

'fr-reasoning': {
  title: 'Best-of-n, and finding where the verifier turns against you',
  prompt: `Implement single-sample, majority voting, verifier-selected, and pass@n. Then lower the verifier accuracy and
find the n at which best-of-n starts getting *worse* — Goodhart in one plot.`,
  hint: 'With a weak verifier, larger n means you are increasingly sampling its error tail rather than genuine correctness.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def trial(p_correct, n, strategy, verifier_acc=0.9, n_wrong=10):
    answers = ["right" if rng.random() < p_correct else f"wrong_{rng.integers(n_wrong)}"
               for _ in range(n)]
    if strategy == "single":   return answers[0] == "right"
    if strategy == "oracle":   return "right" in answers           # pass@n ceiling
    if strategy == "majority":
        # TODO: pick the modal answer
        return False
    if strategy == "verifier":
        # TODO: score each answer (higher if correct, noisily), take the argmax
        return False

print("per-attempt accuracy 0.35\\n")
print(f"{'n':>5} {'single':>8} {'majority':>10} {'verifier':>10} {'pass@n':>9}")
for n in [1, 4, 16, 64, 256]:
    row = [np.mean([trial(0.35, n, s) for _ in range(1500)])
           for s in ["single", "majority", "verifier", "oracle"]]
    print(f"{n:5d} {row[0]:8.3f} {row[1]:10.3f} {row[2]:10.3f} {row[3]:9.3f}")

print("\\na weak verifier stops helping as n grows:")
for va in [0.95, 0.85, 0.70, 0.60]:
    accs = [np.mean([trial(0.35, n, "verifier", va) for _ in range(1200)])
            for n in [4, 64, 256]]
    print(f"  verifier {va:.2f}: n=4 {accs[0]:.3f}  n=64 {accs[1]:.3f}  n=256 {accs[2]:.3f}")

single  = np.mean([trial(0.35, 1,  "single")             for _ in range(1500)])
strong  = np.mean([trial(0.35, 64, "verifier", 0.95)     for _ in range(1500)])
weak    = np.mean([trial(0.35, 64, "verifier", 0.60)     for _ in range(1500)])
assert strong > single, "a good verifier should turn extra samples into accuracy"
assert weak  < strong,  "a weak verifier converts far less of that potential"
print("\\nPASS")
`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def trial(p_correct, n, strategy, verifier_acc=0.9, n_wrong=10):
    answers = ["right" if rng.random() < p_correct else f"wrong_{rng.integers(n_wrong)}"
               for _ in range(n)]
    if strategy == "single":   return answers[0] == "right"
    if strategy == "oracle":   return "right" in answers
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

print("per-attempt accuracy 0.35\\n")
print(f"{'n':>5} {'single':>8} {'majority':>10} {'verifier':>10} {'pass@n':>9}")
for n in [1, 4, 16, 64, 256]:
    row = [np.mean([trial(0.35, n, s) for _ in range(1500)])
           for s in ["single", "majority", "verifier", "oracle"]]
    print(f"{n:5d} {row[0]:8.3f} {row[1]:10.3f} {row[2]:10.3f} {row[3]:9.3f}")

print("\\na weak verifier stops helping as n grows:")
for va in [0.95, 0.85, 0.70, 0.60]:
    accs = [np.mean([trial(0.35, n, "verifier", va) for _ in range(1200)])
            for n in [4, 64, 256]]
    print(f"  verifier {va:.2f}: n=4 {accs[0]:.3f}  n=64 {accs[1]:.3f}  n=256 {accs[2]:.3f}")

single  = np.mean([trial(0.35, 1,  "single")             for _ in range(1500)])
strong  = np.mean([trial(0.35, 64, "verifier", 0.95)     for _ in range(1500)])
weak    = np.mean([trial(0.35, 64, "verifier", 0.60)     for _ in range(1500)])
assert strong > single, "a good verifier should turn extra samples into accuracy"
assert weak  < strong,  "a weak verifier converts far less of that potential"
print("\\nPASS")
`,
  explain: 'Note the gap between pass@n and every extractable strategy. The model often *can* find the answer; the hard part is knowing which sample is right. A weak verifier makes that worse as n grows, because you are selecting on its noise.',
},

'fr-architectures': {
  title: 'Show why a fixed-size state cannot recall an arbitrary past',
  prompt: `Compare geometric decay across channels with attention's flat-then-nothing profile. Then do the information
counting: how long a sequence can a 256-dimensional state store losslessly?`,
  hint: 'A d-dimensional float32 state holds 32d bits. A length-L sequence over V symbols needs L*log2(V) bits.',
  starter: `import numpy as np

print("influence of a token d steps back, by decay rate a:")
print(f"{'a':>8}" + "".join(f"{'d='+str(d):>11}" for d in [10, 50, 200, 1000]))
for a in [0.9, 0.99, 0.999, 0.9999]:
    # TODO: print a**d for each distance
    pass

print("\\nHiPPO-style multi-scale channels:")
rates = 1 - np.logspace(-4, -0.5, 8)
for i, a in enumerate(rates):
    half_life = np.log(0.5)/np.log(a)
    print(f"  channel {i}: a={a:.5f}  half-life {half_life:8.1f} tokens")

print(f"\\n{'seq len':>9} {'bits needed':>13} {'bits in 256-dim state':>23} {'possible?':>11}")
for L in [10, 100, 1000, 10000]:
    # TODO: bits_needed = L * log2(64 symbols); bits_available = 256 * 32
    pass

assert 0.9**200 < 1e-6,    "a fast-decaying state has forgotten a token 200 steps back"
assert 0.9999**200 > 0.9,  "a slow-decaying channel still remembers it"
print("\\nPASS")
`,
  solution: `import numpy as np

print("influence of a token d steps back, by decay rate a:")
print(f"{'a':>8}" + "".join(f"{'d='+str(d):>11}" for d in [10, 50, 200, 1000]))
for a in [0.9, 0.99, 0.999, 0.9999]:
    print(f"{a:8.4f}" + "".join(f"{a**d:11.2e}" for d in [10, 50, 200, 1000]))

print("\\nHiPPO-style multi-scale channels:")
rates = 1 - np.logspace(-4, -0.5, 8)
for i, a in enumerate(rates):
    print(f"  channel {i}: a={a:.5f}  half-life {np.log(0.5)/np.log(a):8.1f} tokens")

print(f"\\n{'seq len':>9} {'bits needed':>13} {'bits in 256-dim state':>23} {'possible?':>11}")
for L in [10, 100, 1000, 10000]:
    need = L * np.log2(64)
    have = 256 * 32
    print(f"{L:9d} {need:13.0f} {have:23.0f} {'yes' if need <= have else 'NO':>11}")

print(f"\\n{'context':>9} {'attention FLOPs':>18} {'SSM FLOPs':>14} {'KV cache (GB)':>15}")
d, L = 4096, 32
for n in [2**k for k in (10, 13, 16, 19)]:
    print(f"{n:9d} {4*n*n*d*L:18.3e} {20*n*d*L:14.3e} {2*L*8*128*n*2/1e9:15.2f}")
print("\\nA transformer's cache grows with the sequence, so it never hits the wall.")
print("An SSM's state does not -- that is both its advantage and its ceiling.")

assert 0.9**200 < 1e-6,    "a fast-decaying state has forgotten a token 200 steps back"
assert 0.9999**200 > 0.9,  "a slow-decaying channel still remembers it"
print("\\nPASS")
`,
  explain: `The decay table is the whole tradeoff. A state-space model carries the past in a fixed-size state that is
multiplied by a decay factor at every step, so a token $d$ steps back has influence $a^d$ — and however you set
$a$, that is an exponential. Set it fast and the model forgets within tens of steps; set it slow enough to
remember a thousand and it barely distinguishes anything nearby. The multi-scale trick is to run many channels
at different rates at once, so *some* channel remembers at every timescale, which is what HiPPO-style
initializations are for.

The bits table makes the limit precise rather than suggestive. A fixed-size state holds a fixed number of bits,
and recalling an arbitrary token from an arbitrary position needs a number of bits that grows with the sequence
length. Past some length the state simply cannot hold the answer, whatever the architecture does with it. This
is not an engineering shortfall; it is counting.

Then the cost table shows the other side. Attention never hits that wall, because its "state" — the KV cache —
grows with the sequence. It pays for that in quadratic compute and linear memory that becomes ruinous at long
context. Neither design is strictly better, which is why the strongest recent architectures interleave the two:
a few attention layers for exact recall, many recurrent layers for cheap context.`,
},
'fr-limits': {
  title: 'Measure calibration error, then fix it with temperature scaling',
  prompt: `Implement expected calibration error and a reliability diagram. Simulate an overconfident RLHF-style model,
then find the temperature that recovers calibration.`,
  hint: 'ECE is the weighted average gap between stated confidence and observed accuracy across bins.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def make_model(overconfidence, n=4000):
    true_p = rng.beta(2, 2, n)
    stated = np.clip(true_p ** (1/overconfidence), 0, 1)
    correct = rng.random(n) < true_p
    return stated, correct

def ece(conf, correct, bins=10):
    """Expected calibration error."""
    # TODO: bin by confidence, accumulate weight * |accuracy - confidence|
    return 0.0

print(f"{'model':22s} {'ECE':>8} {'accuracy':>10} {'mean conf':>11}")
for name, oc in [("base (calibrated)", 1.0), ("mild overconfidence", 2.0),
                 ("RLHF-like", 4.0), ("extreme", 8.0)]:
    c, k = make_model(oc)
    print(f"{name:22s} {ece(c,k):8.4f} {k.mean():10.3f} {c.mean():11.3f}")

c, k = make_model(4.0)
print("\\nreliability diagram (RLHF-like):")
for lo in np.arange(0, 1, 0.2):
    m = (c >= lo) & (c < lo+0.2)
    if m.sum() > 20:
        print(f"  says {lo:.1f}-{lo+0.2:.1f}: actually right {k[m].mean():.3f}  "
              f"gap {k[m].mean()-c[m].mean():+.3f}")

# TODO: sweep T over logits = log(c/(1-c)) and find the T minimizing ECE

c_base, k_base = make_model(1.0)
c_over, k_over = make_model(3.0)
assert ece(c_over, k_over) > ece(c_base, k_base), \\
    "an overconfident model should have a larger calibration error"
assert best_e < ece(c, k), "temperature scaling should reduce the calibration error"
print("\\nPASS")
`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def make_model(overconfidence, n=4000):
    true_p = rng.beta(2, 2, n)
    stated = np.clip(true_p ** (1/overconfidence), 0, 1)
    correct = rng.random(n) < true_p
    return stated, correct

def ece(conf, correct, bins=10):
    total = 0.0
    edges = np.linspace(0, 1, bins+1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0: continue
        total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total

print(f"{'model':22s} {'ECE':>8} {'accuracy':>10} {'mean conf':>11}")
for name, oc in [("base (calibrated)", 1.0), ("mild overconfidence", 2.0),
                 ("RLHF-like", 4.0), ("extreme", 8.0)]:
    c, k = make_model(oc)
    print(f"{name:22s} {ece(c,k):8.4f} {k.mean():10.3f} {c.mean():11.3f}")

c, k = make_model(4.0)
print("\\nreliability diagram (RLHF-like):")
for lo in np.arange(0, 1, 0.2):
    m = (c >= lo) & (c < lo+0.2)
    if m.sum() > 20:
        gap = k[m].mean() - c[m].mean()
        print(f"  says {lo:.1f}-{lo+0.2:.1f}: actually right {k[m].mean():.3f}  "
              f"gap {gap:+.3f} " + "#"*int(abs(gap)*60))

logits = np.log(c/(1-c+1e-9) + 1e-9)
best_T, best_e = 1.0, 1e9
for T in np.arange(0.5, 5, 0.05):
    e = ece(1/(1+np.exp(-logits/T)), k)
    if e < best_e: best_T, best_e = T, e
print(f"\\ntemperature scaling: T={best_T:.2f} reduces ECE {ece(c,k):.4f} -> {best_e:.4f}")

c_base, k_base = make_model(1.0)
c_over, k_over = make_model(3.0)
assert ece(c_over, k_over) > ece(c_base, k_base), \\
    "an overconfident model should have a larger calibration error"
assert best_e < ece(c, k), "temperature scaling should reduce the calibration error"
print("\\nPASS")
`,
  explain: 'Temperature scaling is a single parameter fit on a validation set, and it recovers most of the calibration RLHF destroys — without touching the model or its accuracy.',
},

};

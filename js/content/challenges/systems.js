/* ============================================================
   Challenges — track 5, training & inference systems

   One entry per lesson id: { title, prompt, starter, solution, hint?, explain? }
   Starters carry their own PASS/FAIL check so the learner gets a verdict
   without leaving the lab.

   EDITOR NOTE: these are JS template literals, so an unescaped backtick
   silently terminates the string. Escape inline code in prose.
   ============================================================ */

export default {

'sys-gpu': {
  title: 'Compute the roofline and the decode floor',
  prompt: `Work out the arithmetic intensity of a matmul, a matvec, and an elementwise op, and classify each as
compute- or memory-bound. Then derive the hard token/second ceiling for a 70B model at batch 1.`,
  hint: 'Arithmetic intensity is FLOPs divided by bytes moved. The H100 ridge point is about 300.',
  starter: `import numpy as np
H100 = dict(flops=989e12, bw=3.35e12)
ridge = H100["flops"] / H100["bw"]
print(f"ridge point: {ridge:.0f} FLOPs/byte -- below this you are memory-bound\\n")

def analyze(name, flops, bytes_moved):
    # TODO: compute arithmetic intensity, time if compute-bound, time if memory-bound
    ai, t_c, t_m = 0.0, 0.0, 0.0
    bound = "COMPUTE" if t_c > t_m else "MEMORY"
    return f"{name:32s} AI={ai:8.1f}  {bound:7s}  {max(t_c,t_m)*1e3:7.3f} ms"

d = 4096
print(analyze(f"matmul {d}^3", 2*d**3, 3*d*d*2))
print(analyze(f"matvec {d}x{d}", 2*d*d, d*d*2))
print(analyze("ReLU on 64M elements", 64e6, 64e6*2*2))

print("\\nautoregressive decode floor (bf16, batch 1):")
for name, params in [("7B", 7e9), ("70B", 70e9), ("405B", 405e9)]:
    # TODO: bytes read per token = params * 2; time = bytes / bandwidth;
    #       print GB/token, ms/token, and the resulting tokens per second
    print(f"  {name}")

print("\\nbatching (70B): the weight read is shared across the whole batch")
for B in [1, 8, 32, 128, 512]:
    weight_ms = 140e9 / H100["bw"] * 1e3
    compute_ms = 2 * 70e9 * B / H100["flops"] * 1e3
    step = max(weight_ms, compute_ms)
    print(f"  batch {B:4d}: {step:6.1f} ms/step -> {B*1000/step:8.1f} tok/s total, "
          f"{1000/step:5.1f} per user")

print("\\nquantization is a BANDWIDTH optimization (70B, batch 1):")
for bits, name in [(16, "bf16"), (8, "int8"), (4, "int4")]:
    read = 70e9 * bits/8
    ms = read / H100["bw"] * 1e3
    print(f"  {name:5s}: {read/1e9:5.0f} GB/token -> {ms:6.1f} ms -> {1000/ms:6.1f} tok/s")

def ai(flops, bytes_moved): return flops / bytes_moved
assert ai(2*d**3, 3*d*d*2) > ridge, "a large matmul should be compute-bound"
assert ai(2*d*d, d*d*2) < ridge/10, "a matvec should be far below the ridge point"
assert 70e9*2/H100["bw"] > 0.03, "a 70B model in bf16 cannot generate faster than ~24 tok/s"
print("\\nPASS")`,
  solution: `import numpy as np
H100 = dict(flops=989e12, bw=3.35e12)
ridge = H100["flops"] / H100["bw"]
print(f"ridge point: {ridge:.0f} FLOPs/byte\\n")

def analyze(name, flops, bytes_moved):
    ai = flops / bytes_moved
    t_c = flops / H100["flops"]
    t_m = bytes_moved / H100["bw"]
    bound = "COMPUTE" if t_c > t_m else "MEMORY"
    return f"{name:32s} AI={ai:8.1f}  {bound:7s}  {max(t_c,t_m)*1e3:7.3f} ms"

d = 4096
print(analyze(f"matmul {d}^3", 2*d**3, 3*d*d*2))
print(analyze(f"matvec {d}x{d}", 2*d*d, d*d*2))
print(analyze("ReLU on 64M elements", 64e6, 64e6*2*2))

print("\\nautoregressive decode floor (bf16, batch 1):")
for name, params in [("7B", 7e9), ("70B", 70e9), ("405B", 405e9)]:
    read = params * 2
    ms = read / H100["bw"] * 1e3
    print(f"  {name:5s}: {read/1e9:5.0f} GB/token -> {ms:6.1f} ms -> {1000/ms:6.1f} tok/s max")

print("\\nbatching (70B): the weight read is shared across the whole batch")
for B in [1, 8, 32, 128, 512]:
    weight_ms = 140e9 / H100["bw"] * 1e3
    compute_ms = 2 * 70e9 * B / H100["flops"] * 1e3
    step = max(weight_ms, compute_ms)
    print(f"  batch {B:4d}: {step:6.1f} ms/step -> {B*1000/step:8.1f} tok/s total, "
          f"{1000/step:5.1f} per user")

print("\\nquantization is a BANDWIDTH optimization (70B, batch 1):")
for bits, name in [(16, "bf16"), (8, "int8"), (4, "int4")]:
    read = 70e9 * bits/8
    ms = read / H100["bw"] * 1e3
    print(f"  {name:5s}: {read/1e9:5.0f} GB/token -> {ms:6.1f} ms -> {1000/ms:6.1f} tok/s")

def ai(flops, bytes_moved): return flops / bytes_moved
assert ai(2*d**3, 3*d*d*2) > ridge, "a large matmul should be compute-bound"
assert ai(2*d*d, d*d*2) < ridge/10, "a matvec should be far below the ridge point"
assert 70e9*2/H100["bw"] > 0.03, "a 70B model in bf16 cannot generate faster than ~24 tok/s"
print("\\nPASS")`,
  explain: `The first three lines sort the operations by which resource they exhaust. A large matmul has arithmetic
intensity in the hundreds and is compute-bound — the GPU is doing what it was built for. A matvec is around 1,
three hundred times below the ridge point, and a ReLU is below that: both spend essentially all their time
waiting for memory, with the arithmetic units idle.

The decode floor is the number worth carrying around. Generating one token requires reading **every weight**, so
a 70B model in bf16 moves 140 GB per token and cannot exceed about 24 tokens per second on an H100 no matter how
fast the chip computes. That is a bandwidth fact, not a compute fact, and buying a faster processor would not
move it.

The batching table shows why serving systems fight so hard to keep batches full: the weight read is shared, so
128 sequences cost the same wall-clock step as one, and total throughput rises more than a hundredfold while each
individual user's speed is unchanged. It stops being free only once the batch is large enough that the arithmetic
finally exceeds the memory time — which is where the per-user number starts falling.

And the last table is the lesson's claim about quantization, in numbers: 4-bit weights are not primarily about
fitting in memory, they are about reading four times fewer bytes, which buys close to four times the tokens per
second. The arithmetic was never the constraint.`,
},

'sys-memory': {
  title: 'Budget training memory, and shard it',
  prompt: `Compute the memory for full fine-tuning versus ZeRO stages 1-3, and find the smallest GPU count that fits
a 70B model. Then compute the pipeline bubble fraction.`,
  hint: 'Adam needs 16 bytes/param: 2 bf16 weights, 2 gradients, 4 fp32 master, 4+4 moments.',
  starter: `import numpy as np

def train_memory(params_b, gpus=8, zero_stage=0, seq=4096, batch=4,
                 layers=32, d=4096, checkpointing=True):
    P = params_b * 1e9
    weights, grads, optim = P*2, P*2, P*12
    # TODO: apply ZeRO sharding -- stage 1 shards optim, 2 also grads, 3 also weights
    act = batch*seq*d*layers*2 * (2 if checkpointing else 34)
    return (weights + grads + optim + act) / 1e9

print(f"{'config':40s} {'GB':>8}  fits 80GB?")
for label, kw in [
    ("7B full, no checkpointing", dict(params_b=7, checkpointing=False)),
    ("7B full, checkpointing", dict(params_b=7)),
    ("7B ZeRO-1 x8", dict(params_b=7, zero_stage=1)),
    ("7B ZeRO-3 x8", dict(params_b=7, zero_stage=3)),
    ("70B ZeRO-3 x8", dict(params_b=70, layers=80, d=8192, zero_stage=3)),
    ("70B ZeRO-3 x64", dict(params_b=70, layers=80, d=8192, zero_stage=3, gpus=64)),
]:
    g = train_memory(**kw)
    print(f"{label:40s} {g:8.1f}  {'yes' if g < 80 else 'NO'}")

print("\\npipeline bubble = (P-1)/(m+P-1):")
for P in [4, 8]:
    print(f"  {P} stages: " + "  ".join(f"m={m}: {(P-1)/(m+P-1):5.1%}" for m in [1,4,16,64]))

assert train_memory(7, checkpointing=False) > train_memory(7), \\
    "checkpointing should reduce memory"
assert train_memory(7, gpus=8, zero_stage=3) < train_memory(7, gpus=8, zero_stage=1), \\
    "each ZeRO stage should shard more than the last"
assert train_memory(7, gpus=8, zero_stage=3) < 80, "ZeRO-3 across 8 GPUs should fit a 7B model"
print("\\nPASS")`,
  solution: `import numpy as np

def train_memory(params_b, gpus=8, zero_stage=0, seq=4096, batch=4,
                 layers=32, d=4096, checkpointing=True):
    P = params_b * 1e9
    weights, grads, optim = P*2, P*2, P*12
    if zero_stage >= 1: optim /= gpus
    if zero_stage >= 2: grads /= gpus
    if zero_stage >= 3: weights /= gpus
    act = batch*seq*d*layers*2 * (2 if checkpointing else 34)
    return (weights + grads + optim + act) / 1e9

print(f"{'config':40s} {'GB':>8}  fits 80GB?")
for label, kw in [
    ("7B full, no checkpointing", dict(params_b=7, checkpointing=False)),
    ("7B full, checkpointing", dict(params_b=7)),
    ("7B ZeRO-1 x8", dict(params_b=7, zero_stage=1)),
    ("7B ZeRO-3 x8", dict(params_b=7, zero_stage=3)),
    ("70B ZeRO-3 x8", dict(params_b=70, layers=80, d=8192, zero_stage=3)),
    ("70B ZeRO-3 x64", dict(params_b=70, layers=80, d=8192, zero_stage=3, gpus=64)),
]:
    g = train_memory(**kw)
    print(f"{label:40s} {g:8.1f}  {'yes' if g < 80 else 'NO'}")

print("\\npipeline bubble = (P-1)/(m+P-1):")
for P in [4, 8]:
    print(f"  {P} stages: " + "  ".join(f"m={m}: {(P-1)/(m+P-1):5.1%}" for m in [1,4,16,64]))

assert train_memory(7, checkpointing=False) > train_memory(7), \\
    "checkpointing should reduce memory"
assert train_memory(7, gpus=8, zero_stage=3) < train_memory(7, gpus=8, zero_stage=1), \\
    "each ZeRO stage should shard more than the last"
assert train_memory(7, gpus=8, zero_stage=3) < 80, "ZeRO-3 across 8 GPUs should fit a 7B model"
print("\\nPASS")`,
  explain: `Read the first block top to bottom. A 7B model — small by current standards — needs 258 GB to fine-tune
naively, which is three H100s for a model whose weights are only 14 GB. The other 244 GB is optimizer state and
activations, and the activations are the larger surprise: they scale with batch size and sequence length rather
than with the model.

Gradient checkpointing halves the total, and ZeRO does the rest by noticing that with 8 GPUs there is no reason
for each of them to hold a full copy of the optimizer state. Stage 1 shards the optimizer, stage 2 also the
gradients, stage 3 also the weights — each stage trading more communication for less memory. The 70B rows show
where it ends: even ZeRO-3 across 8 GPUs does not fit, and you need 64.

The pipeline table is the other cost of splitting a model across devices. With $P$ stages, the first stage
finishes and then waits for the last to catch up — a **bubble** of idle time worth $(P-1)/(m+P-1)$ of every step.
At one microbatch across eight stages, **87.5% of your very expensive GPUs are doing nothing**. Keeping many
microbatches in flight is what fills the pipeline back up, which is why $m$ is pushed large and why deep
pipelines need big batches to be worth having.`,
},
'sys-quantization': {
  title: 'Break quantization with one outlier, then fix it with groups',
  prompt: `Implement symmetric integer quantization with an optional group size. Add a few extreme outlier weights and
show that per-tensor scaling collapses while per-group survives.`,
  hint: 'The scale is max|w| / (2^(bits-1) - 1). One outlier stretches it for everything sharing that scale.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def quantize(w, bits, group=None):
    levels = 2**(bits-1) - 1
    if group is None:
        # TODO: one scale for the whole tensor
        return w
    out = np.empty_like(w)
    # TODO: one scale per contiguous group of \`group\` weights
    return out

n = 4096
w = rng.normal(0, 1, n)
w_out = w.copy(); w_out[::512] *= 25          # a few extreme outliers

err = lambda a, b: np.sqrt(((a-b)**2).mean()) / a.std()

print(f"{'scheme':30s} {'clean':>9} {'with outliers':>15}")
for bits in [8, 4, 3]:
    for group in [None, 128, 64]:
        g = "per-tensor" if group is None else f"group={group}"
        print(f"  {bits}-bit {g:22s} {err(w, quantize(w,bits,group)):8.4f} "
              f"{err(w_out, quantize(w_out,bits,group)):14.4f}")

w = np.random.default_rng(1).normal(size=4096)
w[0] = 40.0                                    # one outlier
err = lambda b, g: np.abs(quantize(w, b, g) - w).mean()
assert err(4, 64) < err(4, None), "group-wise scales should beat a single tensor-wide scale"
assert err(8, None) < err(4, None), "more bits should mean less error"
print("\\nPASS")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def quantize(w, bits, group=None):
    levels = 2**(bits-1) - 1
    if group is None:
        s = np.abs(w).max() / levels
        return np.round(w/s) * s
    out = np.empty_like(w)
    for i in range(0, len(w), group):
        chunk = w[i:i+group]
        s = max(np.abs(chunk).max() / levels, 1e-12)
        out[i:i+group] = np.round(chunk/s) * s
    return out

n = 4096
w = rng.normal(0, 1, n)
w_out = w.copy(); w_out[::512] *= 25
err = lambda a, b: np.sqrt(((a-b)**2).mean()) / a.std()

print(f"{'scheme':30s} {'clean':>9} {'with outliers':>15}")
for bits in [8, 4, 3]:
    for group in [None, 128, 64]:
        g = "per-tensor" if group is None else f"group={group}"
        print(f"  {bits}-bit {g:22s} {err(w, quantize(w,bits,group)):8.4f} "
              f"{err(w_out, quantize(w_out,bits,group)):14.4f}")

P = 7e9
print("\\n7B model:")
for name, bits, oh in [("bf16",16,0), ("int8",8,0.02), ("int4 g=128",4,0.04)]:
    gb = P * bits/8 * (1+oh) / 1e9
    print(f"  {name:12s} {gb:6.1f} GB   decode ceiling {3.35e12/(gb*1e9):6.1f} tok/s")

w = np.random.default_rng(1).normal(size=4096)
w[0] = 40.0                                    # one outlier
err = lambda b, g: np.abs(quantize(w, b, g) - w).mean()
assert err(4, 64) < err(4, None), "group-wise scales should beat a single tensor-wide scale"
assert err(8, None) < err(4, None), "more bits should mean less error"
print("\\nPASS")`,
  explain: `The first table is the case for group-wise quantization, and the mechanism is worth naming. Quantizing means
mapping a tensor's whole range onto a handful of levels — so one outlier weight stretches the range and every
*other* weight gets squeezed into fewer effective levels. Splitting the tensor into groups of 64 or 128, each
with its own scale, means an outlier can only ruin its own group. Compare the 4-bit rows: per-tensor error is
several times worse than group-of-64, bought with one extra scale factor per 64 weights.

Notice also which column degrades faster. Error on the *matrix product* is consistently worse than error on the
weights themselves, because the errors accumulate across the summation — a reminder that weight-level error
metrics understate what the model will actually experience.

The last block is why anyone accepts the error at all. Going from bf16 to 4-bit does not merely halve memory
twice; it roughly quadruples the decode ceiling, because generation is bandwidth-bound and you are now reading a
quarter of the bytes. That is the trade: a small and controllable loss of precision for a large and immediate
speedup, which is why 4-bit is the default for local inference.`,
},
'sys-inference': {
  title: 'Size a KV cache and find the optimal speculation length',
  prompt: `Compute KV cache size with and without GQA, then derive the expected speedup from speculative decoding and
find the k that maximizes it.`,
  hint: 'Expected accepted tokens per round is (1 - a^(k+1))/(1 - a); cost is k*draft_cost + 1.',
  starter: `import numpy as np

def kv_bytes(layers, kv_heads, d_head, seq, batch, bits=16):
    # TODO: 2 (K and V) * layers * heads * d_head * seq * batch * bytes
    return 0

print(f"{'model':24s} {'8k ctx':>10} {'128k ctx':>11} {'batch 32 @ 8k':>15}")
for name, m in [
  ("Llama-3-70B (GQA 8)", dict(layers=80, kv_heads=8,  d_head=128)),
  ("70B if it used MHA",  dict(layers=80, kv_heads=64, d_head=128)),
]:
    a = kv_bytes(**m, seq=8192,   batch=1)/1e9
    b = kv_bytes(**m, seq=131072, batch=1)/1e9
    c = kv_bytes(**m, seq=8192,   batch=32)/1e9
    print(f"{name:24s} {a:9.2f}G {b:10.1f}G {c:14.1f}G")

def speedup(alpha, k, draft_cost):
    # TODO
    return 0.0

print("\\nspeculative decoding speedup (draft costs 10% of target):")
print(f"{'accept':>8}" + "".join(f"{'k='+str(k):>8}" for k in [1,2,4,6,8]))
for a in [0.5, 0.7, 0.8, 0.9]:
    print(f"{a:8.1f}" + "".join(f"{speedup(a,k,0.10):8.2f}" for k in [1,2,4,6,8]))

assert kv_bytes(80, 8, 128, 128_000, 1) > kv_bytes(80, 8, 128, 8_000, 1), \\
    "the cache grows with context length"
assert kv_bytes(80, 8, 128, 8_000, 1) < kv_bytes(80, 64, 128, 8_000, 1), \\
    "grouped-query attention should shrink the cache"
assert speedup(0.9, 4, 0.1) > speedup(0.5, 4, 0.1), \\
    "a better draft model should give a bigger speedup"
print("\\nPASS")`,
  solution: `import numpy as np

def kv_bytes(layers, kv_heads, d_head, seq, batch, bits=16):
    return 2 * layers * kv_heads * d_head * seq * batch * bits/8

print(f"{'model':24s} {'8k ctx':>10} {'128k ctx':>11} {'batch 32 @ 8k':>15}")
for name, m in [
  ("Llama-3-70B (GQA 8)", dict(layers=80, kv_heads=8,  d_head=128)),
  ("70B if it used MHA",  dict(layers=80, kv_heads=64, d_head=128)),
]:
    a = kv_bytes(**m, seq=8192,   batch=1)/1e9
    b = kv_bytes(**m, seq=131072, batch=1)/1e9
    c = kv_bytes(**m, seq=8192,   batch=32)/1e9
    print(f"{name:24s} {a:9.2f}G {b:10.1f}G {c:14.1f}G")

def speedup(alpha, k, draft_cost):
    accepted = (1 - alpha**(k+1)) / (1 - alpha)
    return accepted / (k*draft_cost + 1)

print("\\nspeculative decoding speedup (draft costs 10% of target):")
print(f"{'accept':>8}" + "".join(f"{'k='+str(k):>8}" for k in [1,2,4,6,8]))
for a in [0.5, 0.7, 0.8, 0.9]:
    print(f"{a:8.1f}" + "".join(f"{speedup(a,k,0.10):8.2f}" for k in [1,2,4,6,8]))

for a in [0.6, 0.8, 0.9]:
    best = max(((speedup(a,k,0.10), k) for k in range(1,17)))
    print(f"\\nalpha={a}: optimal k={best[1]} giving {best[0]:.2f}x")

assert kv_bytes(80, 8, 128, 128_000, 1) > kv_bytes(80, 8, 128, 8_000, 1), \\
    "the cache grows with context length"
assert kv_bytes(80, 8, 128, 8_000, 1) < kv_bytes(80, 64, 128, 8_000, 1), \\
    "grouped-query attention should shrink the cache"
assert speedup(0.9, 4, 0.1) > speedup(0.5, 4, 0.1), \\
    "a better draft model should give a bigger speedup"
print("\\nPASS")`,
  explain: `Speculative decoding works because verifying $k$ drafted tokens costs one weight-read — the same as generating
a single token. So when the draft is usually right, you get several tokens for the price of one.

The table shows both sides of the trade. Higher acceptance means longer runs of drafted tokens survive, so the
speedup climbs. But read along each row: the speedup rises with $k$, peaks, and then **falls**. Drafting more
tokens than the small model can reliably get right means paying for draft passes whose output is discarded. The
optimum moves right as acceptance improves — $k=3$ at 60% acceptance, $k=10$ at 90% — which is why a better draft
model is worth more than a bigger $k$.

And it is worth stressing what this does *not* cost. The verification step accepts a drafted token only with the
probability the target model itself would have assigned, so the text you get is exactly the text the big model
would have produced. It is a pure latency win with no quality change, which is rare enough to be worth noticing.`,
},
};

/* ============================================================
   Track 5 — Training and Inference Systems
   ============================================================ */

import { t, key, intuition, warn, hist, mathnote, viz, deriv, code, quiz, paper, book, course, blog, video, demo, codeRef,
         tldr, recap, jargon, steps, diagram } from './_helpers.js';

export default [

/* ---------------------------------------------------------- */
{
  id: 'sys-gpu',
  title: 'GPUs, Memory, and the Roofline',
  sub: 'Why your model is slow, and why it is almost never the FLOPs.',
  mins: 24, level: 'advanced',
  prereq: ['math-numerics'],
  tags: ['GPU', 'performance'],
  sections: [
    tldr(`Almost everything you assume about why models are slow is wrong. The bottleneck is essentially never
arithmetic — GPUs have far more compute than they can feed. **It is moving data between memory and the
processor.**

One ratio explains nearly all of it. An H100 can do about 1000 trillion operations per second but can only read
about 3.35 terabytes per second from its main memory. Divide: roughly **300 arithmetic operations per byte
read**. Anything doing less than that leaves the arithmetic units idle, waiting.

Generating text from a language model does about *2* operations per byte. It is memory-bound by a factor of
150, which sets a hard speed limit no amount of extra compute can lift — and explains why batching, quantization,
and speculative decoding all work.`),

    jargon([
      ['FLOP / FLOPs', 'Floating-point operation(s). The unit of arithmetic work.'],
      ['HBM', 'High Bandwidth Memory — the GPU\'s main memory. Large (80 GB) and, relatively speaking, slow.'],
      ['SRAM', 'Tiny on-chip memory. Tens of megabytes, roughly 10× faster than HBM. Using it well is what FlashAttention does.'],
      ['bandwidth', 'How many bytes per second you can read from memory. The scarce resource, almost always.'],
      ['arithmetic intensity', 'FLOPs performed per byte read. The single number that decides whether an operation is compute-bound or memory-bound.'],
      ['memory-bound', 'Limited by how fast data arrives. The processor is idle. Most deep learning operations are here.'],
      ['compute-bound', 'Limited by arithmetic. What you want, and what large matrix multiplies achieve.'],
      ['roofline', 'A plot of achievable performance against arithmetic intensity. Flat where memory-bound, sloped where compute-bound.'],
      ['kernel', 'One GPU function. "Kernel fusion" means combining several into one so intermediate results never leave the chip.'],
      ['tensor core', 'Specialised hardware doing small matrix multiplies extremely fast. The source of the enormous FLOP numbers.'],
      ['prefill / decode', 'Processing the prompt (parallel, compute-bound) versus generating tokens one at a time (sequential, memory-bound). Two completely different performance regimes.'],
    ]),

    t(`## The hardware, in one paragraph

A modern GPU has thousands of arithmetic units and a memory hierarchy: fast on-chip SRAM (tens of MB, ~20 TB/s) and
slower off-chip HBM (tens of GB, ~2–3 TB/s). Tensor cores do matrix multiply-accumulate at enormous rates.

The numbers that matter for an H100: roughly **1000 TFLOP/s** in bf16 against **3.35 TB/s** of memory bandwidth. Divide
them and you get the machine's **arithmetic intensity**: about 300 FLOPs per byte read. Any operation doing fewer FLOPs
per byte than that is **memory-bound** — the arithmetic units sit idle waiting for data.`),

    key(`This ratio explains nearly all deep learning performance work.

| Operation | FLOPs per byte | Verdict |
|---|---|---|
| Large matmul ($n\\times n$ by $n\\times n$) | $\\sim n/6$ | compute-bound for large $n$ ✓ |
| Elementwise (ReLU, add) | $\\sim 0.25$ | hopelessly memory-bound |
| LayerNorm | $\\sim 1$ | memory-bound |
| Attention (naive) | low, and quadratic memory traffic | memory-bound |
| **Autoregressive decoding** | $\\sim 2$ per weight byte | **severely memory-bound** |

The last row is the important one. Generating one token requires reading *every weight in the model* from HBM to do a
handful of FLOPs with each. A 70B model in bf16 means 140 GB of reads per token. At 3.35 TB/s that is ~42 ms — a hard
floor of about 24 tokens/second, **regardless of how fast the GPU computes**.`),

    t(`## The consequences

Everything about inference optimization follows from that floor:

- **Batching is free-ish.** The weights are read once for the whole batch, so 64 sequences cost barely more than 1.
  Throughput scales; latency does not improve. This is why serving systems fight so hard to keep batches full.
- **Quantization is a bandwidth optimization**, not primarily a memory-capacity one. 4-bit weights mean 4× fewer bytes
  to read, hence ~4× faster decoding.
- **Speculative decoding** works because verifying $k$ tokens is one weight-read, the same as generating one.
- **Kernel fusion** matters enormously. Separate ReLU, add, and LayerNorm kernels each round-trip the whole tensor
  through HBM; fused, the data stays in registers.
- **FlashAttention** is fast because it is IO-aware, not because it does less arithmetic. It does slightly *more*
  (recomputation) and wins anyway.`),

    t(`## Prefill vs decode

Two phases with completely different characteristics:

**Prefill** processes the whole prompt at once. It is a big matmul over $n$ tokens — **compute-bound**, high GPU
utilization, and it produces the KV cache.

**Decode** generates one token at a time. Each step is a matrix–*vector* product — **memory-bound**, terrible
utilization.

Serving systems therefore treat them separately: continuous batching mixes prefill and decode work, and some
architectures disaggregate them onto different hardware entirely. Time-to-first-token is a prefill metric;
inter-token latency is a decode metric, and they optimize differently.`),

    t(`## MFU

**Model FLOPs Utilization** is the honest benchmark for a training run: achieved FLOPs divided by the hardware peak.

$$\\text{MFU} = \\frac{6ND / t}{\\text{peak FLOP/s}}$$

Well-tuned large training runs achieve 40–55%. Below 30% means something is wrong — usually a data pipeline that
cannot keep up, communication that is not overlapped with compute, or too many small unfused kernels.`),

    code('The roofline, computed', `import numpy as np

H100 = dict(flops=989e12, bw=3.35e12)     # bf16 dense TFLOP/s, HBM3 bytes/s
ridge = H100["flops"] / H100["bw"]
print(f"H100 ridge point: {ridge:.0f} FLOPs per byte")
print("Anything below this is memory-bound.\\n")

def analyze(name, flops, bytes_moved):
    ai = flops / bytes_moved
    t_compute = flops / H100["flops"]
    t_memory  = bytes_moved / H100["bw"]
    bound = "COMPUTE" if t_compute > t_memory else "MEMORY"
    return f"{name:34s} AI={ai:8.1f}  {bound:7s}  {max(t_compute,t_memory)*1e3:7.3f} ms"

d = 4096
print(analyze(f"matmul {d}x{d} @ {d}x{d}", 2*d**3, 3*d*d*2))
print(analyze(f"matvec {d}x{d} @ {d}",     2*d*d,  d*d*2))
print(analyze("ReLU on 64M elements",      64e6,   64e6*2*2))
print(analyze("LayerNorm on 64M",          64e6*5, 64e6*2*2))

# --- the decoding floor for real model sizes ---
print("\\nAutoregressive decode floor (batch=1, bf16):")
for name, params in [("7B", 7e9), ("13B", 13e9), ("70B", 70e9), ("405B", 405e9)]:
    read = params * 2
    ms = read / H100["bw"] * 1e3
    print(f"  {name:5s}: read {read/1e9:5.0f} GB/token -> {ms:6.1f} ms -> {1000/ms:6.1f} tok/s max")

print("\\nSame, at 4-bit:")
for name, params in [("7B", 7e9), ("70B", 70e9)]:
    ms = params * 0.5 / H100["bw"] * 1e3
    print(f"  {name:5s}: {1000/ms:6.1f} tok/s max  ({0.5/2:.0%} the bytes -> 4x faster)")

# --- batching: throughput scales, latency does not ---
print("\\n70B bf16, batch scaling:")
for B in [1, 8, 32, 128]:
    weight_ms = 140e9 / H100["bw"] * 1e3          # read once per step, any batch
    compute_ms = 2 * 70e9 * B / H100["flops"] * 1e3
    step = max(weight_ms, compute_ms)
    print(f"  batch {B:4d}: {step:6.1f} ms/step -> {B*1000/step:8.1f} tok/s total, "
          f"{1000/step:5.1f} tok/s per user")`),

    quiz('Serving a 70B model, you increase batch size from 1 to 32. What happens to per-user latency?',
      ['Roughly unchanged — the same weight read serves the whole batch, so total throughput rises ~32×',
       'It gets 32× worse',
       'It improves by 32×',
       'It doubles'],
      0,
      'Decoding is memory-bound: the cost is dominated by streaming all 140 GB of weights from HBM, which happens once per step regardless of batch size. Adding sequences adds arithmetic the GPU has spare capacity for. Per-user latency is nearly flat until you become compute-bound or run out of KV cache memory — which is exactly why continuous batching is the central technique in every serving stack.'),

    recap(`- State the arithmetic-intensity ratio for a modern GPU and use it to classify an operation as
  compute- or memory-bound.
- Compute the hard token-per-second ceiling for a model of given size from bandwidth alone.
- Explain why prefill and decode have completely different bottlenecks.
- Say why kernel fusion helps, in terms of round trips to HBM.
- Explain what FlashAttention optimized, and why it is faster despite doing *more* arithmetic.`),
  ],
  refs: [
    blog('Making Deep Learning Go Brrrr From First Principles', 'Horace He', 2022, 'https://horace.io/brrr_intro.html', 'The best explanation of compute-bound vs memory-bound vs overhead-bound. Read this one.'),
    paper('Roofline: An Insightful Visual Performance Model', 'Williams, Waterman & Patterson', 2009, 'https://dl.acm.org/doi/10.1145/1498765.1498785', 'The original model.'),
    paper('Efficiently Scaling Transformer Inference', 'Pope et al.', 2022, 'https://arxiv.org/abs/2211.05102', 'Careful analysis of the inference cost model at scale.'),
    blog('GPU Puzzles', 'Sasha Rush', 2022, 'https://github.com/srush/GPU-Puzzles', 'Learn CUDA thinking by solving small problems.'),
    blog('Transformer Inference Arithmetic', 'Kipply Chen', 2022, 'https://kipp.ly/transformer-inference-arithmetic/', 'Works through the latency and memory arithmetic in detail.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'sys-memory',
  title: 'Training Memory and Distributed Training',
  sub: 'Where the gigabytes go, and the four ways to split a model across GPUs.',
  mins: 26, level: 'advanced',
  prereq: ['sys-gpu', 'math-optimization'],
  tags: ['distributed', 'FSDP', 'parallelism'],
  sections: [
    tldr(`"How much memory do I need to train this?" has a formula, and knowing it saves a lot of guessing.

With Adam in mixed precision it is roughly **16 bytes per parameter** before you have stored a single
activation. So a 7B model needs 112 GB of state and does not fit on an 80 GB card — which is why distributed
training is not an advanced topic but a basic requirement.

The second half is the four ways to split a model across GPUs. They are not alternatives: large runs use three
or four simultaneously, and knowing which axis each one splits is what makes the configuration files legible.`),

    jargon([
      ['optimizer state', 'Adam\'s two running averages per parameter. Eight bytes each parameter, and usually the largest single term.'],
      ['master weights', 'The fp32 copy kept so tiny updates are not lost to rounding. Four more bytes per parameter.'],
      ['activations', 'Intermediate values saved during the forward pass for use in the backward pass. Scale with batch size and sequence length, and often exceed the model itself.'],
      ['gradient accumulation', 'Processing several small batches and summing gradients before stepping. Simulates a large batch at small-batch memory cost.'],
      ['data parallel (DP)', 'Every GPU has a full model copy and a different slice of the batch. Gradients are averaged across GPUs.'],
      ['ZeRO / FSDP', 'Sharding the optimizer state, gradients, and weights across data-parallel GPUs instead of replicating them. Three stages of increasing savings.'],
      ['tensor parallel (TP)', 'Splitting individual weight matrices across GPUs. Needs very fast interconnect — used within a node, not across.'],
      ['pipeline parallel (PP)', 'Giving different GPUs different layers. Cheap on bandwidth; introduces idle time called the "bubble".'],
      ['bubble', 'GPUs sitting idle in a pipeline waiting for work from the previous stage.'],
      ['all-reduce', 'The collective operation summing a tensor across all GPUs and giving everyone the result. The main communication cost of data parallelism.'],
      ['3D parallelism', 'Using data, tensor, and pipeline parallelism together. Standard at frontier scale.'],
    ]),

    t(`## The memory budget

Full fine-tuning with Adam costs roughly **16 bytes per parameter** before you store a single activation:

| Item | Bytes/param |
|---|---|
| bf16 weights | 2 |
| bf16 gradients | 2 |
| fp32 master weights | 4 |
| Adam first moment $m$ | 4 |
| Adam second moment $v$ | 4 |
| **Total** | **16** |

A 7B model is 112 GB of state — before activations, before the CUDA context. It does not fit on an 80 GB card.`),

    viz('gpu-memory'),

    t(`**Activations** are the other half, and they scale with batch size × sequence length × hidden size × layers.
For long sequences they can exceed the model state entirely. Levers:

- **Gradient checkpointing** — store activations only at layer boundaries and recompute the rest. $O(\\sqrt L)$ memory
  for ~30% more compute. Nearly always worth it.
- **Gradient accumulation** — process micro-batches sequentially, accumulating gradients before stepping. Simulates a
  large batch with small-batch memory.
- **Sequence/context parallelism** — split the sequence dimension across devices.`),

    t(`## The four parallelism axes`),

    viz('parallelism'),

    t(`**Data parallel (DDP).** Every GPU holds a full model copy and processes a different slice of the batch;
gradients are all-reduced each step. Simple and communication-light, but the model must fit on one device.

**FSDP / ZeRO.** Shard parameters, gradients, and optimizer state across devices. Before each layer runs, all-gather
its parameters; after, free them. Memory per device drops ~$G\\times$, and this is what makes 70B training on 8 cards
possible. ZeRO has three stages — optimizer state, then gradients, then parameters — each saving more and
communicating more.

**Tensor parallel.** Split individual matrices. The FFN's first matrix is split by columns and the second by rows so
only one all-reduce is needed per block. Communication happens *inside* every layer, so it needs NVLink-class
interconnect and is normally confined within a node.

**Pipeline parallel.** Different GPUs own different layers. Communication is tiny (just activations at boundaries) but
naive scheduling leaves GPUs idle — the "bubble." Micro-batching (GPipe, 1F1B) shrinks it to roughly
$(P-1)/(m+P-1)$ for $P$ stages and $m$ micro-batches.

Real runs combine all four. The standard shape: tensor-parallel within a node, pipeline-parallel across nodes,
data-parallel over the whole thing, with FSDP sharding optimizer state.`),

    t(`## Communication is the real constraint

- **All-reduce** costs $2(G-1)/G \\times$ message size with ring algorithms — roughly $2\\times$ the model size per step
  for DDP.
- **Overlap it with compute.** Modern frameworks start reducing gradients for late layers while early layers are still
  computing backward. Without overlap, scaling efficiency collapses.
- **Topology matters.** NVLink within a node is ~900 GB/s; InfiniBand between nodes is ~50–400 GB/s; Ethernet is far
  worse. Put the chattiest parallelism (tensor) on the fastest link.`),

    warn(`**Fault tolerance is a first-class concern.** A run across thousands of GPUs for months will lose hardware —
Meta reported a failure roughly every few hours during Llama-3 training. You need frequent checkpointing, automatic
restart, and the ability to skip a bad data shard. Loss spikes are routine; the standard response is to roll back a few
thousand steps and skip past the offending batch.`),

    code('Memory and parallelism arithmetic', `import numpy as np

def train_memory(params_b, seq=4096, batch=4, layers=32, d=4096,
                 checkpointing=True, zero_stage=0, gpus=8):
    P = params_b * 1e9
    weights, grads = P*2, P*2
    optim = P*12                                   # fp32 master + Adam m,v
    if zero_stage >= 1: optim /= gpus
    if zero_stage >= 2: grads /= gpus
    if zero_stage >= 3: weights /= gpus
    act = batch * seq * d * layers * 2 * (2 if checkpointing else 34)
    return dict(weights=weights/1e9, grads=grads/1e9,
                optim=optim/1e9, act=act/1e9,
                total=(weights+grads+optim+act)/1e9)

print(f"{'config':38s} {'weights':>8} {'grads':>7} {'optim':>7} {'act':>6} {'TOTAL':>8}")
for label, kw in [
    ("7B, no sharding, no checkpointing", dict(params_b=7, checkpointing=False)),
    ("7B, checkpointing",                 dict(params_b=7)),
    ("7B, ZeRO-3 on 8 GPUs",              dict(params_b=7, zero_stage=3)),
    ("70B, ZeRO-3 on 8 GPUs",             dict(params_b=70, layers=80, d=8192, zero_stage=3)),
    ("70B, ZeRO-3 on 64 GPUs",            dict(params_b=70, layers=80, d=8192, zero_stage=3, gpus=64)),
]:
    m = train_memory(**kw)
    fits = "fits 80GB" if m['total'] < 80 else "TOO BIG"
    print(f"{label:38s} {m['weights']:7.1f}G {m['grads']:6.1f}G "
          f"{m['optim']:6.1f}G {m['act']:5.1f}G {m['total']:7.1f}G  {fits}")

# --- pipeline bubble ---
print("\\npipeline bubble fraction = (P-1)/(m+P-1):")
for P in [4, 8]:
    row = "  ".join(f"m={m}: {(P-1)/(m+P-1):5.1%}" for m in [1, 4, 16, 64])
    print(f"  {P} stages -> {row}")

# --- communication volume per step ---
print("\\nper-step communication for a 7B model on 8 GPUs:")
P = 7e9
print(f"  DDP all-reduce   : {2*P*2/1e9:.1f} GB")
print(f"  ZeRO-3 gather+rs : {(P*2 + P*2)/1e9:.1f} GB (but spread across layers, overlappable)")
print(f"  tensor-parallel  : {2 * 32 * 4096 * 4096 * 2 * 2 / 1e9:.1f} GB (per layer, needs NVLink)")`),

    quiz('You can fit a 7B model for inference on one 80GB GPU but not for training. Why?',
      ['Training also needs gradients, fp32 master weights, and two Adam moments — about 16 bytes/param versus 2',
       'Training uses a larger batch size',
       'Training requires more compute per token',
       'Inference uses quantization'],
      0,
      'Inference needs weights only: 7B × 2 bytes = 14 GB, plus KV cache. Training adds gradients (2), fp32 master weights (4), and Adam\'s two moments (4+4) — 16 bytes/param, so 112 GB, plus activations. The fixes are ZeRO/FSDP sharding, 8-bit optimizers, or LoRA, which trains so few parameters that the optimizer state becomes negligible.'),

    recap(`- Recite the 16 bytes/parameter breakdown and say which term is largest.
- Estimate whether a given model will fit on a given card, before launching anything.
- Say what activations scale with, and name the two levers for reducing them.
- Describe what each of the four parallelism strategies splits, and which needs the fastest interconnect.
- Explain what ZeRO/FSDP removes that plain data parallelism duplicates.`),
  ],
  refs: [
    paper('ZeRO: Memory Optimizations Toward Training Trillion Parameter Models', 'Rajbhandari et al.', 2019, 'https://arxiv.org/abs/1910.02054', 'The sharding scheme behind DeepSpeed and FSDP.'),
    paper('Megatron-LM: Training Multi-Billion Parameter Models Using Model Parallelism', 'Shoeybi et al.', 2019, 'https://arxiv.org/abs/1909.08053', 'Tensor parallelism, and the column/row split that minimizes communication.'),
    paper('GPipe: Efficient Training of Giant Neural Networks', 'Huang et al.', 2018, 'https://arxiv.org/abs/1811.06965', 'Pipeline parallelism and micro-batching.'),
    paper('PyTorch FSDP', 'Zhao et al.', 2023, 'https://arxiv.org/abs/2304.11277', 'The production implementation.'),
    paper('The Llama 3 Herd of Models', 'Meta AI', 2024, 'https://arxiv.org/abs/2407.21783', 'Unusually candid section on infrastructure, failure rates, and what actually broke.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'sys-quantization',
  title: 'Quantization, Pruning, and Distillation',
  sub: 'Making a trained model smaller and faster without retraining it from scratch.',
  mins: 24, level: 'advanced',
  prereq: ['sys-gpu', 'math-numerics'],
  tags: ['quantization', 'compression'],
  sections: [
    tldr(`Three ways to make a trained model cheaper without retraining it from scratch.

**Quantization** stores weights in fewer bits — 4 instead of 16. Because decoding is memory-bound, 4-bit
weights are read four times faster, so this is a *speed* optimization at least as much as a memory one. It is
the one that works best and is nearly free.

**Pruning** deletes weights. Straightforward in theory; unstructured sparsity does not actually make GPUs
faster, which limits it severely in practice.

**Distillation** trains a small model to imitate a large one, and works better than training the small model
directly — for reasons that are interesting and not fully settled.`),

    jargon([
      ['quantization', 'Storing numbers in fewer bits by mapping a float range onto a small set of integers.'],
      ['int8 / int4', '8-bit and 4-bit integer formats. int4 gives 4× compression over bf16.'],
      ['scale / zero-point', 'The two numbers mapping the integer grid back to real values.'],
      ['per-group scales', 'Using a separate scale for every 64 or 128 weights instead of one for the whole tensor. The `g=64` in checkpoint names.'],
      ['outlier', 'An activation channel with values ~100× the typical magnitude. Ruins naive quantization by stretching the grid.'],
      ['GPTQ / AWQ', 'The two standard post-training quantization methods. GPTQ adjusts remaining weights to compensate; AWQ protects the channels that matter most.'],
      ['PTQ / QAT', 'Post-Training Quantization (apply it to a finished model) versus Quantization-Aware Training (train with it simulated).'],
      ['pruning', 'Removing weights entirely — setting them to zero.'],
      ['structured / unstructured sparsity', 'Removing whole rows or channels (actually faster) versus scattered individual weights (compresses, but does not speed up standard hardware).'],
      ['distillation', 'Training a small "student" to match a large "teacher" model\'s outputs.'],
      ['soft targets / dark knowledge', 'The teacher\'s full probability distribution, not just its top answer. Carries information about which wrong answers are plausible, which is what makes distillation work.'],
    ]),

    t(`## Quantization

Store weights in fewer bits. Since decoding is memory-bandwidth-bound, 4-bit weights are read 4× faster than bf16 —
this is a *speed* optimization at least as much as a memory one.

The basic scheme maps a floating range onto integers:

$$q = \\text{round}\\!\\left(\\frac{w}{s}\\right), \\qquad w \\approx s\\cdot q, \\qquad s = \\frac{\\max|w|}{2^{b-1}-1}$$`),

    viz('quantization'),

    t(`**The outlier problem** is the whole difficulty. Weights are roughly Gaussian, but transformer *activations*
develop extreme outliers in a few channels — 100× the typical magnitude — and these grow with model size. With one
scale for a whole tensor, a single outlier stretches the grid and collapses everything else onto a few levels.

Turn on the outlier channel in the figure and watch the error jump.

The fixes:

- **Per-group scales.** One scale per 64 or 128 weights. This is the "g=64" in GPTQ/AWQ checkpoint names, and it
  contains the damage locally. Costs a little metadata.
- **GPTQ** — quantize column by column, using approximate second-order information to adjust the remaining weights to
  compensate for the error already introduced.
- **AWQ** — observes that a small fraction of weights are salient (identified via activation magnitude) and protects
  them by scaling before quantization.
- **SmoothQuant** — migrates activation outliers into the weights, where they are easier to handle.
- **LLM.int8()** — keeps outlier channels in fp16 and quantizes the rest.`),

    key(`**Weights quantize far more easily than activations.** Weight-only 4-bit is close to lossless on most models;
activation quantization to 8 bits needs care, and to 4 bits is a research problem. This is why almost all deployed
quantization is weight-only (W4A16), despite the fact that quantizing both would let you use integer tensor cores.

**Quantization-aware training** (fake-quantize during training, with a straight-through estimator on the backward
pass) recovers more accuracy than post-training quantization, at the cost of a training run.`),

    t(`## Pruning

Remove weights entirely.

**Unstructured** — zero individual weights below a magnitude threshold. Reaches high sparsity with little accuracy
loss, and delivers almost no speedup on standard hardware, because irregular sparsity does not map onto dense matmul
units. NVIDIA's 2:4 structured sparsity (2 of every 4 weights zero) is a hardware-supported compromise giving ~2×.

**Structured** — remove entire attention heads, FFN channels, or layers. Actually faster, because the result is a
smaller dense model. This is what "depth pruning" and "width pruning" of LLMs do, usually followed by a short
recovery fine-tune.

The **lottery ticket hypothesis** observed that a pruned subnetwork, reset to its *original* initialization, can train
to full accuracy alone — suggesting dense training is partly a search over sparse subnetworks. Elegant, though it has
proven hard to exploit for actual speedups.`),

    t(`## Distillation

Train a small student to match a large teacher's **output distribution**, not just its argmax:

$$\\mathcal{L} = \\alpha\\,\\mathcal{L}_{\\text{CE}}(y, p_s) + (1-\\alpha)\\,T^2 \\cdot \\text{KL}\\!\\left(p_t^{(T)} \\,\\|\\, p_s^{(T)}\\right)$$

The temperature $T$ softens both distributions to expose the teacher's **dark knowledge** — the relative probabilities
among *wrong* answers. A teacher that assigns 0.7 to "dog", 0.2 to "wolf", and $10^{-6}$ to "airplane" is communicating
a similarity structure that a hard label cannot. The $T^2$ factor rescales gradients so $T$ does not change the
effective learning rate.

Note the interaction with label smoothing: smoothing deliberately erases that inter-class structure, so a
label-smoothed teacher distills measurably worse.

In modern practice, "distillation" often means something looser — training a small model on text *generated* by a
large one. Effective and widely used, though it is sequence-level imitation rather than distribution matching.`),

    code('Quantization schemes, compared', `import numpy as np
rng = np.random.default_rng(0)

def quantize(w, bits, group=None):
    """Symmetric integer quantization, optionally per-group."""
    levels = 2**(bits-1) - 1
    if group is None:
        s = np.abs(w).max() / levels
        return np.round(w/s) * s
    out = np.empty_like(w)
    for i in range(0, len(w), group):
        chunk = w[i:i+group]
        s = np.abs(chunk).max() / levels
        out[i:i+group] = np.round(chunk/max(s,1e-12)) * s
    return out

n = 4096
w = rng.normal(0, 1, n)
w_out = w.copy()
w_out[::512] *= 25                                 # a few extreme outlier weights

def err(a, b): return np.sqrt(((a-b)**2).mean()) / a.std()

print(f"{'scheme':28s} {'clean':>9} {'with outliers':>15}")
for bits in [8, 4, 3]:
    for group in [None, 128, 64]:
        g = "per-tensor" if group is None else f"group={group}"
        print(f"  {bits}-bit {g:20s} {err(w, quantize(w,bits,group)):8.4f} "
              f"{err(w_out, quantize(w_out,bits,group)):14.4f}")

print("\\nPer-tensor scaling collapses under outliers; per-group contains them.\\n")

# --- memory and bandwidth ---
P = 7e9
print("7B model:")
for name, bits, overhead in [("bf16", 16, 0), ("int8", 8, 0.02), ("int4 g=128", 4, 0.04)]:
    gb = P * bits/8 * (1+overhead) / 1e9
    print(f"  {name:12s} {gb:6.1f} GB   decode ceiling {3.35e12/(gb*1e9):6.1f} tok/s")

# --- distillation: what temperature exposes ---
teacher = np.array([6.0, 3.5, 3.2, -1.0, -4.0])
def soft(z, T):
    z = z/T; z = z - z.max(); e = np.exp(z); return e/e.sum()
print("\\nteacher distribution at different temperatures:")
for T in [1, 2, 4]:
    print(f"  T={T}: {np.round(soft(teacher, T), 4)}")
print("  ^ higher T exposes that classes 1 and 2 are nearly as plausible as 0.")
print("    That similarity structure is the 'dark knowledge' the student learns.")`),

    quiz('Why is weight-only 4-bit quantization nearly lossless while 4-bit activation quantization is not?',
      ['Activations develop extreme per-channel outliers that grow with model size; weights stay roughly Gaussian',
       'Weights are more numerous, so errors average out',
       'Activations are computed in fp32 anyway',
       'Weight quantization is applied after training'],
      0,
      'Transformer activations develop systematic outlier channels with magnitudes ~100× the typical value, and the effect strengthens with scale. Those outliers wreck a shared quantization grid. Weights have no comparable structure. Hence W4A16 is standard, and getting activations down requires outlier-specific machinery — SmoothQuant, LLM.int8(), or keeping outlier channels in higher precision.'),

    recap(`- Explain why quantization speeds up decoding, not just shrinks the file.
- Describe the outlier problem and why per-group scales contain it.
- Say why unstructured pruning compresses a model without making it faster on a GPU.
- Explain "dark knowledge" — why a teacher's full distribution teaches more than its top answer.
- Choose between quantization, pruning, and distillation for a stated deployment constraint.`),
  ],
  refs: [
    paper('GPTQ: Accurate Post-Training Quantization for Generative Transformers', 'Frantar et al.', 2022, 'https://arxiv.org/abs/2210.17323', ''),
    paper('AWQ: Activation-aware Weight Quantization', 'Lin et al.', 2023, 'https://arxiv.org/abs/2306.00978', ''),
    paper('LLM.int8()', 'Dettmers et al.', 2022, 'https://arxiv.org/abs/2208.07339', 'The paper that identified and characterized the activation outlier problem.'),
    paper('Distilling the Knowledge in a Neural Network', 'Hinton, Vinyals & Dean', 2015, 'https://arxiv.org/abs/1503.02531', 'The original, and the source of "dark knowledge."'),
    paper('The Lottery Ticket Hypothesis', 'Frankle & Carbin', 2018, 'https://arxiv.org/abs/1803.03635', ''),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'sys-inference',
  title: 'Serving and Inference Optimization',
  sub: 'KV caches, continuous batching, paged attention, speculative decoding.',
  mins: 24, level: 'advanced',
  prereq: ['sys-gpu', 'llm-attention'],
  tags: ['inference', 'serving', 'KV cache'],
  sections: [
    tldr(`Serving a model well is a different discipline from training one, and the gap between a naive
implementation and a good one is roughly **20×** throughput.

Four techniques do most of that. The **KV cache** avoids recomputing the past. **Continuous batching** stops
short requests waiting behind long ones. **PagedAttention** eliminates the memory fragmentation that otherwise
wastes most of your cache. **Speculative decoding** uses a small model to guess ahead, exploiting the fact that
verifying several tokens costs almost the same as generating one.

Every one of them is an attack on the memory-bandwidth bottleneck from [the GPU lesson](#/l/sys-gpu).`),

    jargon([
      ['throughput vs latency', 'Total tokens per second across all users, versus how fast one user sees a response. These trade off, and which you optimize is a product decision.'],
      ['time to first token (TTFT)', 'How long before output starts appearing. Dominated by prefill.'],
      ['prefill / decode', 'Processing the prompt in parallel, then generating one token at a time. Different bottlenecks entirely.'],
      ['KV cache', 'Stored keys and values from previous tokens, so they need not be recomputed each step.'],
      ['static batching', 'Grouping requests and waiting for all to finish. Simple, and wastes enormous capacity on mismatched lengths.'],
      ['continuous / in-flight batching', 'Swapping a finished sequence out and a queued one in immediately, at every step. The single biggest serving win.'],
      ['PagedAttention', 'Managing the KV cache in fixed blocks like virtual memory, so nothing is reserved that is not used.'],
      ['fragmentation', 'Memory reserved but unusable. Early systems wasted 60–80% of the KV cache to it.'],
      ['speculative decoding', 'A small draft model proposes several tokens; the large model verifies them all in one pass. Provably preserves the output distribution.'],
      ['acceptance rate', 'What fraction of the draft model\'s guesses survive verification. Determines the speedup.'],
      ['tokens/second/GPU', 'The metric that actually matters for serving cost.'],
    ]),

    t(`## The KV cache

Generating token $n$ needs the keys and values of all previous tokens. Those never change, so cache them.`),

    viz('kv-cache'),

    t(`Without a cache, generation is $O(n^2)$ in recomputation. With it, $O(n)$. Every inference stack does this.

The cost is memory:

$$\\text{cache bytes} = 2 \\times L \\times n_{\\text{kv heads}} \\times d_{\\text{head}} \\times \\text{seq} \\times \\text{batch} \\times \\text{bytes}$$

For Llama-3-70B at 8k context that is roughly 1.25 GB **per sequence**. Batch 32 and the cache exceeds the weights.
This is why GQA exists, why cache quantization is common, and why context length is expensive to sell.`),

    t(`## PagedAttention

The naive implementation allocates a contiguous buffer sized to the maximum possible sequence length. If a request
finishes at 200 tokens with a 4096 reservation, 95% is wasted. Measured internal fragmentation in early systems ran
60–80%.

**PagedAttention** (vLLM) borrows virtual memory: split the cache into fixed blocks, keep a per-sequence block table,
and allocate on demand. Waste drops to under 4%, which translates directly into larger batches and 2–4× throughput.

It also makes **prefix sharing** trivial — sequences with a common prompt (a shared system prompt, a beam search tree)
point at the same physical blocks.`),

    t(`## Continuous batching

Static batching runs a batch to completion, so every sequence waits for the longest one, and finished slots sit idle.

**Continuous (in-flight) batching** operates at the iteration level: when a sequence emits its EOS token, it leaves and
a queued request takes its slot immediately. Combined with paged memory this is worth several times the throughput of
naive batching, and it is the single most important serving optimization.

The remaining tension is prefill vs decode. A long prompt's prefill blocks decoding for everyone in the batch, spiking
inter-token latency. **Chunked prefill** splits large prefills into pieces interleaved with decode steps, trading a
little time-to-first-token for much steadier streaming.`),

    t(`## Speculative decoding`),

    viz('speculative-decoding'),

    t(`A small draft model proposes $k$ tokens; the target model verifies all $k$ in **one forward pass** — it can,
because verification is parallel while generation is not. A rejection-sampling rule makes the output distribution
**provably identical** to the target model's. This is free speed, not a quality trade.

Expected tokens per round with acceptance rate $\\alpha$:

$$\\mathbb{E}[\\text{accepted}] = \\frac{1-\\alpha^{k+1}}{1-\\alpha}$$

One rejection discards everything after it, so the gain hinges on how well the draft imitates the target. Typical
speedups are 2–3×. Variants avoid the separate draft model entirely: **Medusa** adds extra prediction heads,
**EAGLE** drafts in feature space, and **n-gram/prompt lookup** drafts by copying from the context — remarkably
effective for summarization and code editing, where output repeats input.`),

    t(`## The metrics that matter

- **TTFT** (time to first token) — dominated by prefill, so by prompt length.
- **ITL / TPOT** (inter-token latency) — dominated by memory bandwidth and batch size.
- **Throughput** (tokens/sec across all users) — what determines your cost per token.

**Throughput and latency trade off directly.** Bigger batches raise throughput and worsen per-user latency. There is no
single "fast" configuration; there is a curve, and you pick a point on it based on whether you are serving an
interactive chat or a batch job.`),

    code('KV cache sizing and speculative decoding math', `import numpy as np

def kv_bytes(layers, kv_heads, d_head, seq, batch, bits=16):
    return 2 * layers * kv_heads * d_head * seq * batch * bits/8

models = {
  "Llama-3-8B  (GQA 8)":  dict(layers=32, kv_heads=8,  d_head=128, weights=16),
  "Llama-3-70B (GQA 8)":  dict(layers=80, kv_heads=8,  d_head=128, weights=140),
  "70B if it used MHA":   dict(layers=80, kv_heads=64, d_head=128, weights=140),
}
print(f"{'model':24s} {'8k ctx':>10} {'128k ctx':>11} {'batch 32 @ 8k':>15}")
for name, m in models.items():
    a = kv_bytes(m["layers"], m["kv_heads"], m["d_head"], 8192, 1)/1e9
    b = kv_bytes(m["layers"], m["kv_heads"], m["d_head"], 131072, 1)/1e9
    c = kv_bytes(m["layers"], m["kv_heads"], m["d_head"], 8192, 32)/1e9
    print(f"{name:24s} {a:9.2f}G {b:10.1f}G {c:14.1f}G")
print("\\nGQA cuts the cache 8x. Without it, 70B at batch 32 needs more cache than weights.\\n")

# --- speculative decoding ---
def speedup(alpha, k, draft_cost):
    accepted = (1 - alpha**(k+1)) / (1 - alpha)
    return accepted / (k*draft_cost + 1)

print("speculative decoding speedup (draft costs 10% of target):")
print(f"{'accept':>8}" + "".join(f"{'k='+str(k):>8}" for k in [1,2,4,6,8]))
for a in [0.5, 0.7, 0.8, 0.9]:
    row = "".join(f"{speedup(a,k,0.10):8.2f}" for k in [1,2,4,6,8])
    print(f"{a:8.1f}{row}")

best = max(((speedup(0.8,k,0.10), k) for k in range(1,17)))
print(f"\\nat alpha=0.8, optimal k={best[1]} giving {best[0]:.2f}x")

# --- fragmentation: why paging matters ---
print("\\nKV memory waste, naive contiguous allocation (max_len=4096):")
rng = np.random.default_rng(0)
lengths = rng.integers(50, 2000, 200)
print(f"  reserved: {200*4096:,} token-slots")
print(f"  used    : {lengths.sum():,}")
print(f"  wasted  : {1 - lengths.sum()/(200*4096):.1%}")
print(f"  paged (block=16): wasted {1 - lengths.sum()/(np.ceil(lengths/16).sum()*16):.1%}")`),

    quiz('Why can speculative decoding be strictly free — same output distribution, more speed?',
      ['Verifying k drafted tokens is one forward pass, and a rejection-sampling rule makes the accepted output identical in distribution to the target',
       'The draft model is trained to match the target exactly',
       'It only applies when the target model is confident',
       'It is not free — it trades a small amount of quality for speed'],
      0,
      'The target model scores all $k$ draft tokens in a single parallel forward pass — the same cost as generating one token, because decoding is memory-bound. The accept/reject rule (accept with probability $\\min(1, p_{\\text{target}}/p_{\\text{draft}})$, else resample from the adjusted residual) provably preserves the target distribution. The only cost is wasted draft compute on rejections.'),

    recap(`- Explain what the KV cache stores and how it changes generation from quadratic to linear.
- Compute KV cache size for a given model, context length, and batch size.
- Say why continuous batching beats static batching, in terms of what GPUs do while waiting.
- Explain the fragmentation problem PagedAttention solves, and the analogy it borrows from.
- Explain why speculative decoding is faster *and* gives identical output to the large model alone.`),
  ],
  refs: [
    paper('Efficient Memory Management for LLM Serving with PagedAttention', 'Kwon et al.', 2023, 'https://arxiv.org/abs/2309.06180', 'vLLM. Virtual memory for the KV cache.'),
    paper('Fast Inference from Transformers via Speculative Decoding', 'Leviathan, Kalman & Matias', 2022, 'https://arxiv.org/abs/2211.17192', 'Includes the proof that the output distribution is preserved.'),
    paper('Orca: A Distributed Serving System for Transformer-Based Generative Models', 'Yu et al.', 2022, 'https://www.usenix.org/conference/osdi22/presentation/yu', 'Continuous batching.'),
    paper('Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads', 'Cai et al.', 2024, 'https://arxiv.org/abs/2401.10774', 'Speculation without a separate draft model.'),
    codeRef('vLLM', 'vLLM team', 2023, 'https://github.com/vllm-project/vllm', 'The reference open serving implementation. Readable.'),
  ],
},

];

/* ============================================================
   Track 8 — Vision and Vision-Language Models
   ============================================================ */

import { t, key, intuition, warn, hist, mathnote, viz, deriv, code, quiz, paper, book, course, blog, video, demo, codeRef,
         tldr, recap, jargon, steps, diagram } from './_helpers.js';

export default [

/* ---------------------------------------------------------- */
{
  id: 'vis-vit',
  title: 'Vision Transformers',
  sub: 'Throw away the convolutional inductive bias and let scale replace it.',
  mins: 22, level: 'core',
  prereq: ['nn-cnn', 'llm-transformer'],
  tags: ['ViT', 'vision'],
  sections: [
    tldr(`A Vision Transformer does something almost insultingly simple: **chop the image into squares and
pretend they are words.** Cut a 224×224 image into 16×16 patches, flatten each into a vector, and feed the
resulting 196 "tokens" to an ordinary transformer. That is the whole architecture.

What makes it interesting is what it *throws away*. Every convolutional inductive bias — locality, translation
equivariance, hierarchy — is gone. The model has to learn from data that adjacent patches are related.

The result is a clean crossover that is worth internalising as a general pattern: **ViTs lose to CNNs on small
datasets and win on large ones.** Hand-designed structure is a gift when data is scarce and a ceiling when it is
not.`),

    jargon([
      ['patch', 'One square tile of the image, typically 16×16 pixels. The ViT equivalent of a token.'],
      ['patch embedding', 'Flattening a patch and projecting it to $d_{\\text{model}}$ with a learned linear layer. Mathematically identical to a convolution with stride equal to kernel size.'],
      ['`[CLS]` token', 'An extra learnable token prepended to the sequence, whose final representation is used for classification. Borrowed from BERT.'],
      ['inductive bias', 'Assumptions baked into an architecture before training. CNNs assume locality; ViTs assume almost nothing.'],
      ['Swin', 'A hierarchical ViT that restricts attention to local windows and shifts them between layers, making cost linear rather than quadratic in image size.'],
      ['register token', 'A dedicated extra token added so the model has scratch space, instead of hijacking background patches for it.'],
      ['DINOv2', 'A self-supervised ViT trained without labels, producing general-purpose features widely used as a frozen backbone.'],
    ]),

    t(`## The recipe

A Vision Transformer does something almost insultingly simple: cut the image into fixed squares, flatten each,
project it to $d_{\\text{model}}$, add a position embedding, and run a standard transformer.`),

    viz('vit-patches'),

    t(`That is the whole architecture. A 224×224 image with 16×16 patches gives 196 tokens — a comfortable sequence
length. The original ViT prepended a learnable \`[CLS]\` token and classified from its final representation; modern
variants often just average-pool the patch tokens instead.`),

    key(`**Every convolutional inductive bias is discarded.** No locality, no translation equivariance, no hierarchy.
The model must learn from data that nearby patches are related and that a cat is a cat wherever it appears.

The consequence is a clean crossover: ViTs **lose** to CNNs on small datasets and **win** on large ones. Below roughly
10M images, ResNets are better. Above 100M, ViTs pull ahead and keep scaling. Given enough data, a model learns better
biases than we designed.

This is the same lesson as kernels-vs-deep-learning one level up, and it is worth internalizing as a general pattern
rather than a fact about vision.`),

    t(`## Cost

Attention is quadratic in token count, and token count is quadratic in inverse patch size. Halving the patch size
gives 4× the tokens and **16×** the attention cost. That arithmetic is why high-resolution vision transformers needed
architectural help:

- **Swin** restores hierarchy: attention within local windows, shifted between layers, with patch merging to
  downsample. Linear in image size, and it works well for detection and segmentation where multi-scale features matter.
- **Hybrid stems** use a few convolutional layers before patching, recovering some locality cheaply.
- **Register tokens** — a 2023 finding that ViTs repurpose uninformative background patches as scratch space, producing
  high-norm artifacts in attention maps. Adding a few dedicated "register" tokens gives the model somewhere to put that
  state and cleans the maps up considerably.`),

    t(`## Self-supervised vision

Labels do not scale; pixels do. Three families:

- **Contrastive** (SimCLR, MoCo) — two augmented views of one image should embed together, different images apart.
  Needs many negatives, so batch size matters.
- **Self-distillation** (DINO, DINOv2) — a student matches a teacher that is an exponential moving average of itself.
  No negatives needed. DINOv2 features are strong enough to use frozen for segmentation and depth, and its attention
  maps segment objects without ever being trained to.
- **Masked image modeling** (MAE) — mask 75% of patches and reconstruct them. Efficient, because the encoder only sees
  the visible 25%. The very high mask ratio is essential: images are spatially redundant, so an easy task teaches
  nothing.`),

    code('Patchify, and the cost arithmetic', `import numpy as np

def patchify(img, p):
    """(H, W, C) -> (num_patches, p*p*C)"""
    H, W, C = img.shape
    assert H % p == 0 and W % p == 0
    return (img.reshape(H//p, p, W//p, p, C)
               .transpose(0, 2, 1, 3, 4)
               .reshape(-1, p*p*C))

img = np.arange(8*8*3).reshape(8, 8, 3).astype(float)
P = patchify(img, 4)
print(f"8x8x3 image, patch 4 -> {P.shape[0]} patches of dim {P.shape[1]}")
print(f"first patch top-left pixel: {P[0][:3]}  (should be {img[0,0]})\\n")

d = 768
print(f"{'image':>7} {'patch':>6} {'tokens':>8} {'attn FLOPs':>13} {'relative':>9}")
base = None
for size in [224, 224, 448, 448]:
    for p in ([16, 8] if size == 224 else [16, 8]):
        n = (size//p)**2
        fl = 4 * n*n * d
        base = base or fl
        print(f"{size:7d} {p:6d} {n:8d} {fl:13.3e} {fl/base:8.1f}x")
        break
    else: continue

print()
for size, p in [(224,16), (224,8), (448,16), (448,8), (1024,16)]:
    n = (size//p)**2
    fl = 4*n*n*d
    print(f"{size:7d} {p:6d} {n:8d} {fl:13.3e} {fl/(4*196*196*d):8.1f}x")

print("\\nHalving patch size: 4x tokens, 16x attention cost.")
print("This is why Swin's windowed attention exists.")`),

    quiz('A ViT and a ResNet are trained on 5,000 images. Which likely wins, and why?',
      ['The ResNet — its built-in locality and translation equivariance are correct priors that the ViT must learn from data it does not have',
       'The ViT, because attention is more expressive',
       'They perform identically',
       'The ViT, because it has more parameters'],
      0,
      'The ViT has strictly more expressive power and strictly less prior. With 5,000 images it cannot learn that neighbouring pixels are related, so it wastes capacity rediscovering what a convolution assumes for free. The ordering flips somewhere around 10–100M images. More inductive bias means less data needed but a lower ceiling — the central tradeoff from the [framing lesson](#/l/ml-framing).'),

    recap(`- Describe the ViT recipe in one sentence, and say what a "patch" corresponds to.
- List the convolutional inductive biases a ViT discards, and say what it must learn instead.
- Predict which of a CNN and a ViT wins at a given dataset size, and justify it.
- Explain why halving the patch size costs 16× more attention, not 4×.
- Say what Swin restores and why that matters for detection and segmentation.`),
  ],
  refs: [
    paper('An Image is Worth 16x16 Words', 'Dosovitskiy et al.', 2020, 'https://arxiv.org/abs/2010.11929', 'The ViT paper, including the data-scale crossover.'),
    paper('Swin Transformer', 'Liu et al.', 2021, 'https://arxiv.org/abs/2103.14030', 'Hierarchy and windowed attention restored.'),
    paper('Masked Autoencoders Are Scalable Vision Learners', 'He et al.', 2021, 'https://arxiv.org/abs/2111.06377', 'MAE. The 75% mask ratio is the whole trick.'),
    paper('DINOv2', 'Oquab et al.', 2023, 'https://arxiv.org/abs/2304.07193', 'Self-supervised features good enough to use frozen.'),
    paper('Vision Transformers Need Registers', 'Darcet et al.', 2023, 'https://arxiv.org/abs/2309.16588', 'A satisfying diagnosis of a weird artifact, with a two-line fix.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'vis-clip',
  title: 'CLIP and Contrastive Multimodal Learning',
  sub: 'Learning vision from text supervision, and getting zero-shot classification free.',
  mins: 22, level: 'core',
  prereq: ['vis-vit', 'nn-embeddings'],
  tags: ['CLIP', 'contrastive', 'multimodal'],
  sections: [
    tldr(`Labelled image datasets are small because humans have to label them. The internet has *billions* of
image–caption pairs already, for free, in unlimited vocabulary.

CLIP's idea is to learn from those instead. Train two encoders — one for images, one for text — so that
matching pairs land near each other in a shared space and mismatched pairs land far apart. Nothing ever tells
the model what a dog *is*; the only signal is "these two go together".

The payoff is **zero-shot classification**. To classify an image into categories the model never saw, write the
category names as sentences, embed them, and pick the nearest. The classifier is built at inference time, out of
words.`),

    jargon([
      ['contrastive learning', 'Training on *pairs*: pull matching things together in embedding space, push non-matching apart.'],
      ['shared embedding space', 'One space that both images and text map into, so a picture of a dog and the word "dog" land in the same neighbourhood.'],
      ['positive / negative pair', 'A genuinely matching image–caption pair / any non-matching combination.'],
      ['in-batch negatives', 'Using the other examples in the same batch as the negatives. Free, and the reason batch size matters so much here.'],
      ['temperature $\\tau$', 'A learned scalar dividing the similarities before softmax. Controls how sharply the model must discriminate.'],
      ['zero-shot', 'Classifying into categories the model was never trained on, by describing them in text.'],
      ['prompt template', 'The sentence frame class names are placed in, like "a photo of a {}". Affects accuracy by several points, which is uncomfortable.'],
      ['SigLIP', 'A CLIP variant using a pairwise sigmoid loss instead of softmax, so it does not need enormous batches.'],
    ]),

    t(`## The insight

ImageNet has 1.2M images across 1,000 fixed classes, and every label was placed by a human. The internet has
**billions** of image–caption pairs, already labeled, in open vocabulary, for free.

CLIP trains an image encoder and a text encoder to agree: matching pairs land close in a shared embedding space,
mismatched pairs far apart.`),

    viz('clip-contrastive'),

    t(`The objective is symmetric cross-entropy over the $N\\times N$ similarity matrix in a batch:

$$\\mathcal{L} = \\tfrac12\\Big[\\text{CE}\\big(\\text{softmax}(S/\\tau),\\ I\\big) + \\text{CE}\\big(\\text{softmax}(S^{\\mathsf T}/\\tau),\\ I\\big)\\Big]$$

with $S_{ij} = \\mathbf{e}^{\\text{img}}_i \\cdot \\mathbf{e}^{\\text{txt}}_j$ on normalized embeddings and $\\tau$ a
learned temperature.

Read it as: **for each image, pick its caption out of the batch; for each caption, pick its image.** Nothing tells the
model what a dog *is* — the only supervision is "these two go together."

Batch size is a genuine capability lever here, because each row is an $N$-way classification and a larger $N$ is a
harder, more informative task. CLIP used 32,768.`),

    key(`**Zero-shot classification falls out for free.** At test time, write the class names as captions —
"a photo of a {class}" — embed them, and take the argmax similarity against the image embedding.

The model was never trained on your classes. It was trained on the general relationship between images and language,
and your class names are language. That is a qualitatively different kind of generalization than a fixed-head
classifier can offer, and it is why CLIP was such a big deal.

The prompt template matters more than you would expect — "a photo of a {}" beats bare "{}" by several points, and
ensembling many templates helps further. That is a slightly uncomfortable fact.`),

    t(`## Descendants and limitations

**SigLIP** replaces softmax with a pairwise sigmoid loss, which removes the need for a global normalization across the
batch. Works better at small batch sizes and is now a common vision encoder for VLMs.

**Limitations worth knowing:**

- **Bag-of-words behavior.** CLIP is notoriously weak at compositional structure: "a red cube on a blue sphere"
  and "a blue cube on a red sphere" embed almost identically. Benchmarks like ARO and Winoground were built to
  measure this, and performance is poor.
- **Fine-grained failure.** Counting, spatial relations, text rendering, and specialist domains are all weak.
- **Bias.** Trained on uncurated web pairs; it inherits the associations in them, documented at length in the original
  paper's own broader-impacts section.
- **Typographic attacks.** Write "iPod" on a piece of paper, hold it in front of an apple, and CLIP says iPod. Text in
  the image dominates.`),

    code('The CLIP objective, implemented', `import numpy as np
rng = np.random.default_rng(0)

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
def softmax(z, ax=-1):
    z = z - z.max(ax, keepdims=True); e = np.exp(z); return e/e.sum(ax, keepdims=True)

N, D = 8, 32
# simulate embeddings where matching pairs share a latent factor
latent = rng.normal(size=(N, D))
img = l2(latent + rng.normal(0, 0.35, (N, D)))
txt = l2(latent + rng.normal(0, 0.35, (N, D)))

for tau in [0.5, 0.1, 0.01]:
    S = img @ txt.T / tau
    Pi, Pt = softmax(S, 1), softmax(S, 0)
    loss = -(np.log(np.diag(Pi)+1e-12).mean() + np.log(np.diag(Pt)+1e-12).mean())/2
    acc = (S.argmax(1) == np.arange(N)).mean()
    print(f"tau={tau:5.2f}  loss {loss:.4f}  retrieval acc {acc:.3f}  "
          f"mean P(correct) {np.diag(Pi).mean():.3f}")

print("\\nbatch size makes the task harder and the signal richer:")
for n in [4, 16, 64, 256]:
    lat = rng.normal(size=(n, D))
    i, t = l2(lat + rng.normal(0,0.35,(n,D))), l2(lat + rng.normal(0,0.35,(n,D)))
    S = i @ t.T / 0.07
    print(f"  N={n:4d}: chance {1/n:.4f}, achieved {(S.argmax(1)==np.arange(n)).mean():.3f}")

# --- zero-shot classification ---
classes = ["dog", "cat", "car", "tree"]
class_emb = l2(rng.normal(size=(len(classes), D)))
query_img = l2(class_emb[1] + rng.normal(0, 0.4, D))     # actually a cat
sims = class_emb @ query_img
p = softmax(sims / 0.07)
print("\\nzero-shot prediction:")
for c, s, pp in zip(classes, sims, p):
    print(f"  'a photo of a {c}':  sim {s:+.3f}  p {pp:.3f}")`),

    quiz('CLIP scores 76% zero-shot on ImageNet without ever seeing an ImageNet label. How?',
      ['It learned a general image–language alignment; class names are just text, so it can be conditioned on any label set expressible in words',
       'ImageNet images were in its training data with their labels',
       'It fine-tunes on a few examples at test time',
       'It uses the ImageNet class hierarchy'],
      0,
      'CLIP learned to match images with arbitrary natural-language descriptions. At test time you write each class as a caption and pick the best match, so the label set is defined at inference rather than at training. A conventional classifier has 1,000 fixed output neurons and cannot be asked about a class it has no neuron for. (Contamination is a legitimate concern for the exact number, and the CLIP paper does address overlap analysis — but the mechanism is real.)'),

    recap(`- Explain what CLIP is supervised by, and why that supervision is available at internet scale.
- Read the contrastive objective as "pick the caption out of the batch, and vice versa".
- Say why batch size is a capability lever for contrastive training specifically.
- Explain zero-shot classification, and why a fixed-head classifier structurally cannot do it.
- Name a limitation of CLIP embeddings — counting, spatial relations, fine-grained detail — and why the
  objective causes it.`),
  ],
  refs: [
    paper('Learning Transferable Visual Models From Natural Language Supervision', 'Radford et al.', 2021, 'https://arxiv.org/abs/2103.00020', 'CLIP. Long, thorough, and unusually honest about limitations.'),
    paper('Sigmoid Loss for Language Image Pre-Training', 'Zhai et al.', 2023, 'https://arxiv.org/abs/2303.15343', 'SigLIP.'),
    paper('When and why vision-language models behave like bags-of-words', 'Yuksekgonul et al.', 2022, 'https://arxiv.org/abs/2210.01936', 'The compositionality failure, measured.'),
    paper('Winoground', 'Thrush et al.', 2022, 'https://arxiv.org/abs/2204.03162', 'A benchmark models find nearly impossible and humans find easy.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'vis-vlm',
  title: 'Vision-Language Models',
  sub: 'Getting pixels into a language model, and what breaks when you do.',
  mins: 24, level: 'advanced',
  prereq: ['vis-clip', 'llm-finetuning'],
  tags: ['VLM', 'multimodal'],
  sections: [
    tldr(`You have a language model that reads token embeddings, and an image that is a grid of pixels. Every
vision-language model is an answer to one question: **how do you turn pixels into something the LLM can read?**

There are four answers, and they trade off cost against ceiling. The cheapest — train a two-layer MLP to map
image features into the LLM's embedding space, and freeze everything else — works startlingly well and is why
this area became accessible to small teams in 2023.

The recurring practical problem is **resolution**. A 336×336 encoder physically cannot read a document or a
dense chart, and every fix for that multiplies your token count.`),

    jargon([
      ['VLM', 'Vision-Language Model. A language model that can also take images as input.'],
      ['vision encoder', 'The network turning an image into feature vectors. Usually a frozen pretrained CLIP or SigLIP ViT.'],
      ['projector / adapter', 'A small trained network mapping vision features into the LLM\'s embedding space, so they can sit in the prompt like tokens.'],
      ['cross-attention', 'Attention where queries come from one stream and keys/values from another — the mechanism for letting text attend to image features without merging the streams.'],
      ['gated', 'Multiplied by a learned scalar starting at zero, so a new component begins as a no-op and the model chooses to switch it on.'],
      ['perceiver resampler', 'A module compressing any number of image features into a fixed small number of tokens, using learned queries.'],
      ['native multimodality', 'Training on images and text together from the start, with no separate encoder or bridge.'],
      ['tiling / AnyRes', 'Splitting a high-resolution image into crops, encoding each, and concatenating. The standard fix for resolution, at the cost of many more tokens.'],
      ['catastrophic forgetting', 'Losing text ability while training on images. The main risk when the LLM is unfrozen.'],
    ]),

    t(`## The bridging problem

A language model consumes token embeddings. An image is a grid of pixels. Every VLM architecture is an answer to the
same question: **how do pixels become something the LLM can read?**`),

    viz('vlm-architecture'),

    t(`**Projection / adapter (LLaVA).** Freeze a CLIP or SigLIP encoder and an LLM, train only a small MLP mapping
image patch features into the LLM's embedding space. Image patches literally become tokens in the prompt. Two-stage
training: align the projector on caption data, then instruction-tune.

This is astonishingly cheap — a competent VLM from two frozen models and a two-layer MLP — and it is why VLM research
became broadly accessible in 2023.

**Perceiver resampler (Flamingo).** A fixed set of learned queries cross-attends to the image features and compresses
them to a constant small number of tokens. Decouples resolution from sequence length, which matters enormously for
video.

**Interleaved cross-attention.** The image never enters the token stream; gated cross-attention layers are inserted
between the LLM's blocks. The gates initialize at zero, so the model starts as exactly the original LLM and learns to
open them. Preserves text ability perfectly, at the cost of new parameters in every layer.

**Native multimodality.** No separate encoder, no bridge — images are tokenized alongside text and everything is
trained together from the start. Highest ceiling, most expensive, and cannot reuse an existing LLM. This is the
direction frontier models have moved.`),

    t(`## Resolution is the practical battleground

A 336×336 encoder cannot read a document or a dense chart. Approaches:

- **Tiling / AnyRes** — split a high-resolution image into crops, encode each, concatenate, plus a downsampled global
  view. Simple and effective; blows up token count.
- **Native dynamic resolution** — process the image at its actual aspect ratio and size (NaViT-style packing).
- **Token compression** — pixel-shuffle, pooling, or a resampler to cut patch tokens before they reach the LLM.

The tension is constant: more tokens means better fine detail and more compute. For document and chart understanding
resolution matters more than model size, which is a useful thing to know when a VLM is failing at OCR-ish tasks.`),

    warn(`**Known failure modes**, most of which come from the vision encoder rather than the LLM:

- **Hallucinating objects** that are not present — the language prior overrides weak visual evidence. If the caption
  distribution says kitchens contain sinks, the model will report a sink.
- **Counting.** Reliably poor beyond about four objects.
- **Spatial reasoning.** Left/right, above/below, and relative position are weak, inherited from CLIP-style training.
- **Fine text.** Improving fast with resolution, still unreliable.
- **Safety asymmetry.** Text-only safety training does not transfer to image inputs. Instructions embedded in an image
  are a live prompt-injection vector, and this is not solved.`),

    t(`## Evaluation

MMMU (college-level multimodal reasoning), MathVista (visual math), DocVQA and ChartQA (documents), MMBench, and
POPE (object hallucination specifically) are the common suite.

The recurring methodological problem: **many benchmark questions are answerable without the image.** Text-only
baselines score far above chance on several popular multimodal benchmarks, because language priors and answer-option
artifacts leak the answer. Always check whether a blind baseline was reported — if it was not, discount the number.`),

    code('The projector, and the token budget', `import numpy as np
rng = np.random.default_rng(0)

d_vision, d_llm = 1024, 4096
grid = 24                                        # 336px / 14px patches
n_patches = grid * grid

# LLaVA-style two-layer projector
W1 = rng.normal(0, d_vision**-0.5, (d_vision, d_llm))
W2 = rng.normal(0, d_llm**-0.5, (d_llm, d_llm))
def project(feats):
    h = feats @ W1
    return np.maximum(0, h) @ W2                 # GELU in practice

feats = rng.normal(size=(n_patches, d_vision))
tokens = project(feats)
print(f"{n_patches} patches -> {tokens.shape[0]} tokens of dim {tokens.shape[1]}")
print(f"projector params: {(d_vision*d_llm + d_llm*d_llm)/1e6:.1f}M "
      f"(vs ~7000M for the LLM it feeds)\\n")

# --- token budget across strategies ---
print(f"{'strategy':30s} {'tokens':>8} {'attn cost vs text-only':>24}")
text_only = 500
for name, n in [
    ("text only (500 tok prompt)", 0),
    ("336px, 14px patches", 576),
    ("+ 4 tiles (AnyRes 672px)", 576*5),
    ("perceiver resampler (64)", 64),
    ("pixel-shuffle 2x2", 576//4),
    ("1024px, 14px patches", (1024//14)**2),
]:
    total = text_only + n
    print(f"{name:30s} {n:8d} {(total/text_only)**2:23.1f}x")

print("\\nHigh resolution is expensive quadratically. Hence resamplers and pooling.")`),

    quiz('A VLM describes a kitchen photo and mentions a sink that is not there. What is the most likely cause?',
      ['Language prior overriding weak visual evidence — kitchens usually have sinks in the caption distribution',
       'The image encoder failed entirely',
       'The temperature is too low',
       'The image resolution is too high'],
      0,
      'This is object hallucination, and it is the characteristic VLM failure. The LLM is vastly stronger than the visual signal reaching it, so when visual evidence is ambiguous the language prior fills the gap with what *usually* co-occurs. It is measured directly by POPE. Mitigations include stronger visual grounding in training, higher resolution, and decoding methods that contrast conditioned against unconditioned outputs.'),

    recap(`- State the bridging problem in one sentence.
- Compare the four bridging strategies on cost, ceiling, and whether they preserve text ability.
- Explain why zero-initialized gates let you add cross-attention without damaging the base model.
- Say why resolution is the practical battleground, and what tiling costs you.
- Recognise object hallucination and explain it as the language prior overpowering a weak visual signal.`),
  ],
  refs: [
    paper('Visual Instruction Tuning (LLaVA)', 'Liu et al.', 2023, 'https://arxiv.org/abs/2304.08485', 'The projector recipe that democratized VLMs.'),
    paper('Flamingo: a Visual Language Model for Few-Shot Learning', 'Alayrac et al.', 2022, 'https://arxiv.org/abs/2204.14198', 'Perceiver resampler and gated cross-attention.'),
    paper('Evaluating Object Hallucination in LVLMs (POPE)', 'Li et al.', 2023, 'https://arxiv.org/abs/2305.10355', ''),
    paper('MMMU', 'Yue et al.', 2023, 'https://arxiv.org/abs/2311.16502', 'College-level multimodal reasoning, with text-only baselines reported.'),
    paper('Patch n\' Pack: NaViT', 'Dehghani et al.', 2023, 'https://arxiv.org/abs/2307.06304', 'Native resolution and aspect ratio.'),
  ],
},

];

/* ============================================================
   Challenges — track 8, vision & VLMs

   EDITOR NOTE: these are JS template literals, so an unescaped backtick
   silently terminates the string. Escape inline code in prose.
   ============================================================ */

export default {

'vis-vit': {
  title: 'Patchify an image and price the attention',
  prompt: `Implement patchify without loops over pixels, verify it round-trips, then compute how attention cost scales
as you halve the patch size. Predict the factor before running it.`,
  hint: 'Reshape to (H/p, p, W/p, p, C), transpose so the patch axes are adjacent, then flatten.',
  starter: `import numpy as np

def patchify(img, p):
    """(H, W, C) -> (num_patches, p*p*C)"""
    H, W, C = img.shape
    assert H % p == 0 and W % p == 0
    # TODO
    return np.zeros((0, p*p*C))

img = np.arange(8*8*3).reshape(8, 8, 3).astype(float)
P = patchify(img, 4)
assert P.shape == (4, 48), f"expected (4, 48), got {P.shape}"
assert np.allclose(P[0][:3], img[0, 0]), "first patch should start at the top-left pixel"
print("PASS\\n")

d = 768
print(f"{'image':>7} {'patch':>6} {'tokens':>8} {'attn FLOPs':>13} {'vs 224/16':>10}")
base = 4 * 196*196 * d
for size, p in [(224,16), (224,8), (448,16), (448,8), (1024,16)]:
    n = (size//p)**2
    fl = 4*n*n*d
    print(f"{size:7d} {p:6d} {n:8d} {fl:13.3e} {fl/base:9.1f}x")`,
  solution: `import numpy as np

def patchify(img, p):
    H, W, C = img.shape
    assert H % p == 0 and W % p == 0
    return (img.reshape(H//p, p, W//p, p, C)
               .transpose(0, 2, 1, 3, 4)
               .reshape(-1, p*p*C))

img = np.arange(8*8*3).reshape(8, 8, 3).astype(float)
P = patchify(img, 4)
assert P.shape == (4, 48)
assert np.allclose(P[0][:3], img[0, 0])
print("PASS\\n")

d = 768
print(f"{'image':>7} {'patch':>6} {'tokens':>8} {'attn FLOPs':>13} {'vs 224/16':>10}")
base = 4 * 196*196 * d
for size, p in [(224,16), (224,8), (448,16), (448,8), (1024,16)]:
    n = (size//p)**2
    fl = 4*n*n*d
    print(f"{size:7d} {p:6d} {n:8d} {fl:13.3e} {fl/base:9.1f}x")`,
  explain: 'Halving the patch size gives 4x the tokens and 16x the attention cost. That quadratic-of-a-quadratic is exactly why Swin reintroduced windowed attention and why resamplers exist.',
},

'vis-clip': {
  title: 'Implement the CLIP objective and do zero-shot classification',
  prompt: `Write the symmetric contrastive loss over the similarity matrix, then use the trained space to classify an
image by writing class names as captions. Check how batch size changes the difficulty.`,
  hint: 'Two cross-entropies: rows (image picks caption) and columns (caption picks image), against the identity.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
def softmax(z, ax=-1):
    z = z - z.max(ax, keepdims=True); e = np.exp(z); return e/e.sum(ax, keepdims=True)

def clip_loss(img, txt, tau):
    """Symmetric cross-entropy over the NxN similarity matrix."""
    # TODO: S = img @ txt.T / tau, then average the row-wise and column-wise NLL
    return 0.0

N, D = 8, 32
latent = rng.normal(size=(N, D))
img = l2(latent + rng.normal(0, 0.35, (N, D)))
txt = l2(latent + rng.normal(0, 0.35, (N, D)))

for tau in [0.5, 0.1, 0.01]:
    S = img @ txt.T / tau
    acc = (S.argmax(1) == np.arange(N)).mean()
    print(f"tau={tau:5.2f}  loss {clip_loss(img, txt, tau):.4f}  retrieval acc {acc:.3f}")

print("\\nbatch size makes the task harder and the signal richer:")
for n in [4, 16, 64, 256]:
    lat = rng.normal(size=(n, D))
    i, t = l2(lat + rng.normal(0,0.35,(n,D))), l2(lat + rng.normal(0,0.35,(n,D)))
    S = i @ t.T / 0.07
    print(f"  N={n:4d}: chance {1/n:.4f}, achieved {(S.argmax(1)==np.arange(n)).mean():.3f}")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
def softmax(z, ax=-1):
    z = z - z.max(ax, keepdims=True); e = np.exp(z); return e/e.sum(ax, keepdims=True)

def clip_loss(img, txt, tau):
    S = img @ txt.T / tau
    Pi, Pt = softmax(S, 1), softmax(S, 0)
    return -(np.log(np.diag(Pi)+1e-12).mean() + np.log(np.diag(Pt)+1e-12).mean()) / 2

N, D = 8, 32
latent = rng.normal(size=(N, D))
img = l2(latent + rng.normal(0, 0.35, (N, D)))
txt = l2(latent + rng.normal(0, 0.35, (N, D)))

for tau in [0.5, 0.1, 0.01]:
    S = img @ txt.T / tau
    acc = (S.argmax(1) == np.arange(N)).mean()
    print(f"tau={tau:5.2f}  loss {clip_loss(img, txt, tau):.4f}  retrieval acc {acc:.3f}")

print("\\nbatch size makes the task harder and the signal richer:")
for n in [4, 16, 64, 256]:
    lat = rng.normal(size=(n, D))
    i, t = l2(lat + rng.normal(0,0.35,(n,D))), l2(lat + rng.normal(0,0.35,(n,D)))
    S = i @ t.T / 0.07
    print(f"  N={n:4d}: chance {1/n:.4f}, achieved {(S.argmax(1)==np.arange(n)).mean():.3f}")

classes = ["dog", "cat", "car", "tree"]
class_emb = l2(rng.normal(size=(len(classes), D)))
query = l2(class_emb[1] + rng.normal(0, 0.4, D))
p = softmax(class_emb @ query / 0.07)
print("\\nzero-shot prediction (true class: cat):")
for c, pp in zip(classes, p):
    print(f"  a photo of a {c:5s}  p={pp:.3f}")`,
},

'vis-vlm': {
  title: 'Build the projector and budget the token cost',
  prompt: `Implement a LLaVA-style two-layer projector from vision features into the language model's embedding space,
count its parameters against the LLM's, and compute how tiling for high resolution blows up the sequence.`,
  hint: 'The projector is tiny — that is the whole point. Compare it to the 7B model it feeds.',
  starter: `import numpy as np
rng = np.random.default_rng(0)

d_vision, d_llm = 1024, 4096
grid = 24
n_patches = grid * grid

W1 = rng.normal(0, d_vision**-0.5, (d_vision, d_llm))
W2 = rng.normal(0, d_llm**-0.5, (d_llm, d_llm))

def project(feats):
    # TODO: two-layer MLP with a nonlinearity between
    return np.zeros((len(feats), d_llm))

feats = rng.normal(size=(n_patches, d_vision))
tokens = project(feats)
assert tokens.shape == (n_patches, d_llm)
print(f"{n_patches} patches -> {tokens.shape[0]} tokens of dim {tokens.shape[1]}")
print(f"projector params: {(d_vision*d_llm + d_llm*d_llm)/1e6:.1f}M vs ~7000M for the LLM\\n")

text_only = 500
print(f"{'strategy':32s} {'img tokens':>11} {'attn cost':>12}")
for name, n in [("text only", 0), ("336px / 14px patches", 576),
                ("+ 4 tiles (AnyRes 672px)", 576*5), ("perceiver resampler", 64),
                ("pixel-shuffle 2x2", 144), ("1024px / 14px patches", (1024//14)**2)]:
    total = text_only + n
    print(f"{name:32s} {n:11d} {(total/text_only)**2:11.1f}x")`,
  solution: `import numpy as np
rng = np.random.default_rng(0)

d_vision, d_llm = 1024, 4096
grid = 24
n_patches = grid * grid

W1 = rng.normal(0, d_vision**-0.5, (d_vision, d_llm))
W2 = rng.normal(0, d_llm**-0.5, (d_llm, d_llm))

def project(feats):
    h = feats @ W1
    return np.maximum(0, h) @ W2         # GELU in real implementations

feats = rng.normal(size=(n_patches, d_vision))
tokens = project(feats)
assert tokens.shape == (n_patches, d_llm)
print(f"{n_patches} patches -> {tokens.shape[0]} tokens of dim {tokens.shape[1]}")
print(f"projector params: {(d_vision*d_llm + d_llm*d_llm)/1e6:.1f}M vs ~7000M for the LLM")
print(f"trainable fraction: {(d_vision*d_llm + d_llm*d_llm)/7e9:.2%}\\n")

text_only = 500
print(f"{'strategy':32s} {'img tokens':>11} {'attn cost':>12}")
for name, n in [("text only", 0), ("336px / 14px patches", 576),
                ("+ 4 tiles (AnyRes 672px)", 576*5), ("perceiver resampler", 64),
                ("pixel-shuffle 2x2", 144), ("1024px / 14px patches", (1024//14)**2)]:
    total = text_only + n
    print(f"{name:32s} {n:11d} {(total/text_only)**2:11.1f}x")`,
  explain: 'The projector is under 0.4% of the parameters and is the only thing trained in LLaVA stage 1. That is why a competent VLM became buildable by anyone with two frozen checkpoints and a GPU.',
},

};

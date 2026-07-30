/* ============================================================
   content/index.js — the curriculum registry.

   To add a track: write js/content/<name>.js exporting an array of
   lessons, import it here, and add one entry to TRACKS.
   See AUTHORING.md for the full recipe.
   ============================================================ */

import math from './math.js';
import classical from './classical.js';
import nn from './nn.js';
import llm from './llm.js';
import systems from './systems.js';
import generative from './generative.js';
import rl from './rl.js';
import vision from './vision.js';
import frontier from './frontier.js';
import embodied from './embodied.js';
import practice from './practice.js';

export { CHALLENGES } from './challenges/index.js';

export const TRACKS = [
  {
    id: 'math',
    num: 1,
    name: 'Mathematical Foundations',
    color: '#5aa9ff',
    desc: 'Vectors, matrices, SVD, derivatives, probability, information theory, optimization, and floating point. Everything the rest of the atlas assumes.',
    intro: `Nothing here is mathematics for its own sake — every lesson exists because something later breaks without it.
If you already know this material, skim the interactive figures and move on; if you do not, this track is worth more
than the rest of the atlas combined.`,
    lessons: math,
  },
  {
    id: 'classical',
    num: 2,
    name: 'Classical Machine Learning',
    color: '#37d6a8',
    desc: 'Regression, regularization, trees, SVMs, clustering, and evaluation. The methods that still win on tabular data, and the concepts deep learning inherited.',
    intro: `Deep learning did not replace this material; it built on it. Overfitting, regularization, the bias-variance
decomposition, and honest evaluation are exactly as load-bearing for a 70B-parameter transformer as for a linear model —
they are just harder to see at that scale.`,
    lessons: classical,
  },
  {
    id: 'nn',
    num: 3,
    name: 'Neural Networks',
    color: '#b78bff',
    desc: 'Backpropagation, activations, initialization, normalization, CNNs, RNNs, and embeddings. The mechanics under every modern model.',
    intro: `This is the load-bearing track for everything that follows. Backpropagation in particular is worth
understanding to the point where you could re-derive it — nearly every training pathology you will ever debug is a
consequence of those four equations.`,
    lessons: nn,
  },
  {
    id: 'llm',
    num: 4,
    name: 'Transformers & LLMs',
    color: '#ff8f5a',
    desc: 'Tokenization, attention, the transformer block, pretraining, scaling laws, fine-tuning, decoding, prompting, RAG, MoE, and evaluation.',
    intro: `The architecture that ate the field. Attention is the only genuinely new idea here — everything else is
assembled from the previous three tracks. Work through the attention lesson slowly; the rest follows from it more
easily than you might expect.`,
    lessons: llm,
  },
  {
    id: 'systems',
    num: 5,
    name: 'Training & Inference Systems',
    color: '#4fd3e8',
    desc: 'GPUs and the roofline, distributed training, quantization, and serving. Why models are slow, and what to do about it.',
    intro: `Almost every performance question in deep learning reduces to one ratio: FLOPs per byte moved. This track
starts there and derives the rest — batching, quantization, speculative decoding, FlashAttention — as consequences of
memory bandwidth rather than as a bag of tricks.`,
    lessons: systems,
  },
  {
    id: 'generative',
    num: 6,
    name: 'Generative Models',
    color: '#ffd35a',
    desc: 'Autoencoders and VAEs, GANs, and diffusion — including flow matching and the score-based view.',
    intro: `Three attempts at the same problem: learn a distribution well enough to sample from it. The diffusion
lesson is the long one, and the figures there compute the exact analytic score of a Gaussian mixture, so what you see
is the real thing rather than an illustration.`,
    lessons: generative,
  },
  {
    id: 'rl',
    num: 7,
    name: 'Reinforcement Learning & Alignment',
    color: '#ff6b9d',
    desc: 'MDPs and Bellman, Q-learning and exploration, policy gradients and PPO, then RLHF, DPO, and reasoning models.',
    intro: `Learning from consequences instead of answers. The first three lessons are classical RL; the fourth is
where it meets language models, and it is the reason RL matters far more now than it did five years ago.`,
    lessons: rl,
  },
  {
    id: 'vision',
    num: 8,
    name: 'Vision & VLMs',
    color: '#a3b18a',
    desc: 'Vision transformers, CLIP and contrastive multimodal learning, and how pixels get into a language model.',
    intro: `Vision is where the "learned features beat designed features, given enough data" lesson was demonstrated
most clearly — twice, first by CNNs over hand-crafted descriptors and then by ViTs over CNNs.`,
    lessons: vision,
  },
  {
    id: 'frontier',
    num: 9,
    name: 'Frontier Topics',
    color: '#b78bff',
    desc: 'Mechanistic interpretability, reasoning and test-time compute, state-space models, and the problems nobody has solved.',
    intro: `This track is a map of terrain that is still moving. Treat it as a set of pointers to primary sources
rather than settled knowledge, and check the papers — several of these areas will look different in a year.`,
    lessons: frontier,
  },
  {
    id: 'embodied',
    num: 10,
    name: 'Physical & Embodied Intelligence',
    color: '#37d6a8',
    desc: 'Robot learning: why imitation drifts, visuomotor policies, vision-language-action models, world models, and how to evaluate any of it honestly.',
    intro: `Everything that made language models work is absent here — you cannot scrape robot data, actions have
consequences, and the data distribution depends on the policy. This track is organized around that single constraint,
because nearly every technique in modern robot learning is a strategy for needing less action-labeled data.`,
    lessons: embodied,
  },
  {
    id: 'practice',
    num: 11,
    name: 'Practice',
    color: '#8492a3',
    desc: 'Debugging models that do not work, running experiments you can trust, and reading papers efficiently.',
    intro: `The material that does not appear in courses and accounts for most of the difference between people who
ship working models and people who do not.`,
    lessons: practice,
  },
];

/* ---------------- derived indices ---------------- */

export const LESSONS = TRACKS.flatMap((t) => t.lessons);

const BY_ID = new Map(LESSONS.map((l) => [l.id, l]));
const TRACK_OF = new Map();
for (const tr of TRACKS) for (const l of tr.lessons) TRACK_OF.set(l.id, tr);

export const byId = (id) => BY_ID.get(id);
export const trackOf = (id) => TRACK_OF.get(id) || TRACKS[0];

/** All references across all lessons, deduplicated by title, with back-links. */
export function allRefs() {
  const seen = new Map();
  for (const l of LESSONS) {
    for (const r of l.refs || []) {
      const k = (r.title || '').toLowerCase().trim();
      if (!k) continue;
      if (!seen.has(k)) seen.set(k, { ...r, lessons: [] });
      seen.get(k).lessons.push(l.id);
    }
    for (const s of l.sections || []) {
      if (s.t === 'refs') {
        for (const r of s.items || []) {
          const k = (r.title || '').toLowerCase().trim();
          if (!seen.has(k)) seen.set(k, { ...r, lessons: [] });
          seen.get(k).lessons.push(l.id);
        }
      }
    }
  }
  return [...seen.values()];
}

/* ---------------- glossary ---------------- */

export const GLOSSARY = [
  { term: 'Activation function', sym: '$\\phi$', def: 'The elementwise nonlinearity between linear layers. Without one, stacking layers gains nothing.', see: 'nn-activations' },
  { term: 'AdamW', def: 'Adam with **decoupled** weight decay — the penalty is applied directly to the weights rather than added to the loss, so it is not distorted by the adaptive scaling. The default optimizer for transformers.', see: 'math-optimization' },
  { term: 'Attention', def: 'A soft dictionary lookup: score a query against every key, softmax the scores, return the weighted average of the values.' },
  { term: 'Autodiff (reverse mode)', def: 'Computing all derivatives of one scalar output with respect to every input in a single backward pass. Backpropagation is this applied to a neural network.', see: 'math-jacobian' },
  { term: 'Backpropagation', def: 'Reverse-mode autodiff on a network. Four equations: output delta, propagate backward, weight gradient, bias gradient.', see: 'nn-backprop' },
  { term: 'Bias (statistical)', def: 'How far the *average* model, over random training sets, is from the truth. High bias = underfitting.', see: 'ml-overfitting' },
  { term: 'BatchNorm', def: 'Normalize each feature across the batch. Transformative for CNNs; awkward for sequences, which is why transformers use LayerNorm.', see: 'nn-normalization' },
  { term: 'bf16', def: 'Brain float16: fp32 with the mantissa truncated. Same exponent range as fp32, so no loss scaling needed. The default training precision.', see: 'math-numerics' },
  { term: 'Condition number', sym: '$\\kappa$', def: 'Ratio of largest to smallest singular value. Measures numerical sensitivity and predicts how badly gradient descent will zig-zag.', see: 'math-eigen-svd' },
  { term: 'Convexity', def: 'Every chord lies above the curve. Implies any local minimum is global. Linear/logistic regression and SVMs are convex; neural networks are not.', see: 'math-optimization' },
  { term: 'Cross-entropy', sym: '$H(p,q)$', def: 'Average bits to encode samples from $p$ using a code built for $q$. Minimizing it equals maximizing likelihood equals minimizing $\\mathrm{KL}(p\\|q)$.', see: 'math-information' },
  { term: 'Curse of dimensionality', def: 'In high dimensions, volume concentrates near the surface, distances concentrate, and random vectors are nearly always orthogonal.', see: 'math-vectors' },
  { term: 'Determinant', sym: '$\\det A$', def: 'The signed factor by which a linear map scales volume. Zero means the map collapses space and is not invertible.', see: 'math-matrices' },
  { term: 'Double descent', def: 'Test error rises to a peak at the interpolation threshold ($d=n$) and then falls again as capacity grows further. The regime deep networks live in.', see: 'ml-overfitting' },
  { term: 'Dropout', sym: '$p$', def: 'Randomly zero units during training, scaling survivors by $1/(1-p)$. Trains an implicit ensemble of subnetworks.', see: 'nn-regularization' },
  { term: 'Eigenvector', sym: '$A\\mathbf{v}=\\lambda\\mathbf{v}$', def: 'A direction the matrix only stretches, never rotates. Hessian eigenvalues are curvatures; covariance eigenvectors are principal components.', see: 'math-eigen-svd' },
  { term: 'ELBO', def: 'Evidence lower bound. The VAE objective: reconstruction quality minus KL from the posterior to the prior.' },
  { term: 'Embedding', def: 'A learned dense vector for a discrete symbol. Implemented as a lookup table; semantically, geometry replaces one-hot orthogonality.', see: 'nn-embeddings' },
  { term: 'Empirical risk', sym: '$\\hat R$', def: 'Average loss on your sample — what you can compute. True risk is the expectation over the real distribution, which you cannot.', see: 'ml-framing' },
  { term: 'Entropy', sym: '$H(p)$', def: 'Average surprise, $-\\sum p\\log p$. The minimum bits needed to encode samples from $p$.', see: 'math-information' },
  { term: 'Gradient', sym: '$\\nabla f$', def: 'Vector of partial derivatives. Points in the direction of steepest ascent and is perpendicular to the level set.', see: 'math-derivatives' },
  { term: 'Gradient checkpointing', def: 'Store only some forward activations and recompute the rest during the backward pass. Trades ~30% compute for $O(\\sqrt L)$ memory.', see: 'nn-backprop' },
  { term: 'He initialization', def: 'Weights drawn with variance $2/n_{\\text{in}}$. The factor of 2 compensates for ReLU discarding half the signal.', see: 'nn-initialization' },
  { term: 'Hessian', sym: '$H$', def: 'Matrix of second derivatives. Its eigenvalues are the curvatures along the principal axes of the loss surface.', see: 'math-derivatives' },
  { term: 'Inductive bias', def: 'The assumptions a model makes before seeing data. More bias means less data needed but a lower ceiling if the bias is wrong.', see: 'ml-framing' },
  { term: 'Jacobian', sym: '$J$', def: 'Matrix of all partial derivatives of a vector-valued function — the best local linear map. Backprop applies these without ever building them.', see: 'math-jacobian' },
  { term: 'KL divergence', sym: '$D_{\\mathrm{KL}}(p\\|q)$', def: 'Excess bits from coding $p$ with a code built for $q$. Non-negative, asymmetric. Forward KL covers modes; reverse KL seeks them.', see: 'math-information' },
  { term: 'LayerNorm', def: 'Normalize across features within one example. Batch-size independent and identical at train and test — which is why transformers use it.', see: 'nn-normalization' },
  { term: 'Lasso (L1)', def: 'Penalty $\\lambda\\|\\mathbf{w}\\|_1$. Its constraint region has corners on the axes, so it produces coefficients that are exactly zero.', see: 'ml-regularization' },
  { term: 'Learning rate', sym: '$\\eta$', def: 'Step size. On a quadratic, stability requires $\\eta < 2/\\lambda_{\\max}$ — so the sharpest curvature anywhere caps it everywhere.', see: 'math-optimization' },
  { term: 'Likelihood', def: 'Probability of the observed data as a function of the parameters. Maximizing it is MLE; adding a prior makes it MAP.', see: 'math-probability' },
  { term: 'Logit', def: 'A pre-softmax (or pre-sigmoid) score. Equivalently, log-odds.', see: 'ml-logistic' },
  { term: 'LoRA', def: 'Fine-tune by adding a low-rank update $\\Delta W = BA$ instead of touching $W$. Trains ~0.1–1% as many parameters.', see: 'math-matrices' },
  { term: 'Momentum', sym: '$\\beta$', def: 'Accumulate a velocity across steps. Oscillating components cancel; consistent ones amplify by $1/(1-\\beta)$.', see: 'math-optimization' },
  { term: 'Overfitting', def: 'Fitting noise in the training data. Symptom: training error keeps falling while held-out error rises.', see: 'ml-overfitting' },
  { term: 'PCA', def: 'Project onto the top eigenvectors of the covariance. Equivalently, the subspace minimizing reconstruction error — the two objectives are identical.', see: 'ml-unsupervised' },
  { term: 'Perplexity', def: '$e^{\\text{cross-entropy}}$. The effective number of equally likely choices the model faces per token. Tokenizer-dependent.', see: 'math-information' },
  { term: 'Precision / Recall', def: 'Precision: of what you flagged, how much was right. Recall: of what was real, how much you caught. Rare classes destroy precision while leaving AUC intact.', see: 'ml-evaluation' },
  { term: 'Rank', def: 'Dimension of a matrix\'s output space. Low rank means compressible — the basis of PCA, LoRA, and recommender systems.', see: 'math-matrices' },
  { term: 'Residual connection', def: '$\\mathbf{y} = \\mathcal{F}(\\mathbf{x})+\\mathbf{x}$. Gives gradients an identity path home, which is why 100-layer networks train.', see: 'nn-cnn' },
  { term: 'Ridge (L2)', def: 'Penalty $\\lambda\\|\\mathbf{w}\\|^2$. Shrinks selectively along directions the data does not constrain; never produces exact zeros.', see: 'ml-regularization' },
  { term: 'RMSNorm', def: 'LayerNorm without mean subtraction. Cheaper, empirically equivalent, used by Llama and most recent models.', see: 'nn-normalization' },
  { term: 'SGD', def: 'Gradient descent on minibatches. The estimate is unbiased with variance $\\propto 1/B$, and that noise is itself a useful regularizer.', see: 'math-optimization' },
  { term: 'Softmax', def: '$e^{z_i}/\\sum_j e^{z_j}$ — turns scores into a distribution. Its gradient with cross-entropy collapses to $\\mathbf{p}-\\mathbf{y}$.', see: 'math-jacobian' },
  { term: 'Superposition', def: 'Packing more features than dimensions by using non-orthogonal directions, viable because features are sparse and rarely collide.', see: 'math-vectors' },
  { term: 'SVD', sym: '$A=U\\Sigma V^{\\mathsf T}$', def: 'Every matrix is a rotation, a stretch, and another rotation. Truncating it gives the provably optimal low-rank approximation.', see: 'math-eigen-svd' },
  { term: 'Universal approximation', def: 'One hidden layer can approximate any continuous function — given enough units. Says nothing about how many, or whether SGD will find them.', see: 'nn-perceptron-mlp' },
  { term: 'Vanishing gradient', def: 'The backward product of Jacobians decays exponentially with depth, so early layers stop learning. Fixed by ReLU, careful init, normalization, and residuals.', see: 'nn-backprop' },
  { term: 'Variance (statistical)', def: 'How much the fitted model changes across random training sets. High variance = overfitting.', see: 'ml-overfitting' },
  { term: 'Weight decay', sym: '$\\lambda$', def: 'L2 shrinkage applied to weights. In Bayesian terms, a zero-mean Gaussian prior. Exclude biases and norm parameters.', see: 'nn-regularization' },
  { term: 'Xavier / Glorot init', def: 'Weights with variance $1/n_{\\text{in}}$, keeping activation variance stable across depth for symmetric activations.', see: 'nn-initialization' },

  /* --- tracks 4–10 --- */
  { term: 'ALiBi', def: 'Positional information added as a distance-proportional penalty directly to attention scores. Extrapolates beyond training length.', see: 'llm-transformer' },
  { term: 'Arithmetic intensity', def: 'FLOPs performed per byte moved from memory. Below the hardware ratio (~300 on an H100) an operation is memory-bound and the compute units idle.', see: 'sys-gpu' },
  { term: 'BPE', def: 'Byte-pair encoding. Start from bytes, repeatedly merge the most frequent adjacent pair. The tokenizer behind most modern models.', see: 'llm-tokenization' },
  { term: 'Causal mask', def: 'Sets attention scores to $-\\infty$ for future positions so a token cannot see its own future. What makes a decoder a decoder.', see: 'llm-attention' },
  { term: 'Chinchilla-optimal', def: 'Roughly 20 training tokens per parameter — the allocation minimizing loss for a fixed training budget. Not the same as deployment-optimal.', see: 'llm-scaling' },
  { term: 'Classifier-free guidance', sym: '$w$', def: 'Extrapolate along the difference between conditional and unconditional predictions. Higher $w$ means better prompt adherence, less diversity.', see: 'gen-diffusion' },
  { term: 'Continuous batching', def: 'Add and remove sequences from an inference batch at each iteration rather than per batch. The single largest serving throughput win.', see: 'sys-inference' },
  { term: 'DDIM', def: 'Deterministic diffusion sampling via an ODE view, reaching good samples in 10–50 steps instead of 1000.', see: 'gen-diffusion' },
  { term: 'DPO', def: 'Direct preference optimization. Algebraically eliminates the reward model from RLHF, leaving supervised learning on preference pairs.', see: 'rl-rlhf' },
  { term: 'ELBO', def: 'Evidence lower bound — the VAE objective. Reconstruction quality minus the KL from the approximate posterior to the prior.', see: 'gen-autoencoders' },
  { term: 'FlashAttention', def: 'Exact attention computed with tiling and recomputation so the $n\\times n$ matrix is never materialized. $O(n)$ memory, faster in wall-clock.', see: 'llm-attention' },
  { term: 'FSDP / ZeRO', def: 'Shard parameters, gradients, and optimizer state across devices, gathering each layer only when needed.', see: 'sys-memory' },
  { term: 'GQA', def: 'Grouped-query attention. Several query heads share one key/value head, shrinking the KV cache proportionally.', see: 'llm-attention' },
  { term: 'GRPO', def: 'Group relative policy optimization. Uses the mean reward of several sampled responses as a baseline, removing the value network.', see: 'rl-policy-gradient' },
  { term: 'Induction head', def: 'A two-head circuit that finds a previous occurrence of the current token and copies what followed it. A major mechanism behind in-context learning.', see: 'llm-attention' },
  { term: 'In-context learning', def: 'Learning a task from examples in the prompt, with no weight updates. The "learning" happens entirely in the forward pass.', see: 'llm-prompting' },
  { term: 'KV cache', def: 'Stored keys and values for past tokens, making generation $O(n)$ instead of $O(n^2)$. Often the dominant memory cost at long context.', see: 'sys-inference' },
  { term: 'MFU', def: 'Model FLOPs utilization — achieved FLOPs over hardware peak. Well-tuned large training runs reach 40–55%.', see: 'sys-gpu' },
  { term: 'Mixture of experts', def: 'Replace the FFN with $E$ experts and route each token to $k$ of them. Parameters grow $E\\times$, compute per token only $k\\times$.', see: 'llm-moe' },
  { term: 'PagedAttention', def: 'Block-based KV cache allocation, borrowed from virtual memory. Cuts fragmentation from ~70% to under 4%.', see: 'sys-inference' },
  { term: 'PPO', def: 'Proximal policy optimization. Clips the importance ratio so the policy cannot drift far from the one that collected the data.', see: 'rl-policy-gradient' },
  { term: 'Prefill vs decode', def: 'Prefill processes the prompt in parallel and is compute-bound; decode emits one token at a time and is memory-bound. They optimize differently.', see: 'sys-gpu' },
  { term: 'Reparameterization trick', def: 'Write $z = \\mu + \\sigma\\epsilon$ so randomness sits in a parameter-free variable and gradients flow through $\\mu,\\sigma$.', see: 'gen-autoencoders' },
  { term: 'Reward hacking', def: 'Optimizing a learned proxy reward into regions where it no longer tracks real quality. Goodhart\'s law, and the reason for the KL penalty in RLHF.', see: 'rl-rlhf' },
  { term: 'RLHF', def: 'Train a reward model on human preference comparisons, then optimize the policy against it with a KL leash to the SFT model.', see: 'rl-rlhf' },
  { term: 'RLVR', def: 'RL with verifiable rewards — tests pass or the answer is right. Cannot be hacked like a learned reward model, and is how reasoning models are trained.', see: 'fr-reasoning' },
  { term: 'RoPE', def: 'Rotary position embedding. Rotates $q$ and $k$ by an angle proportional to position, so their dot product depends only on relative offset.', see: 'llm-transformer' },
  { term: 'Score', sym: '$\\nabla_x\\log p(x)$', def: 'The gradient of log-density. A diffusion model trained to predict noise is estimating this up to a known constant.', see: 'gen-diffusion' },
  { term: 'Speculative decoding', def: 'A draft model proposes $k$ tokens; the target verifies them in one pass. Provably preserves the output distribution.', see: 'sys-inference' },
  { term: 'Sparse autoencoder', def: 'A wide, L1-penalized autoencoder trained on activations to recover features stored in superposition.', see: 'fr-interpretability' },
  { term: 'SSM', def: 'State-space model. A linear recurrence with $O(1)$ state, parallelizable in training via a scan. Cheap at long context, weak at exact recall.', see: 'fr-architectures' },
  { term: 'SwiGLU', def: 'A gated FFN variant, $(\\text{Swish}(xW_1)\\odot xW_2)W_3$, standard in modern LLMs. Hidden dim shrunk to $\\tfrac23\\times$ to match parameter count.', see: 'nn-activations' },
  { term: 'Temperature', sym: '$T$', def: 'Divides logits before softmax. $T<1$ sharpens, $T>1$ flattens, $T\\to0$ is greedy.', see: 'llm-decoding' },
  { term: 'Top-p / nucleus sampling', def: 'Keep the smallest set of tokens whose cumulative probability reaches $p$. Adapts to how confident the model is.', see: 'llm-decoding' },
  { term: 'Superposition', def: 'Storing more features than dimensions as non-orthogonal directions, viable because sparse features rarely collide. Why neurons are polysemantic.', see: 'fr-interpretability' },
  { term: 'ViT', def: 'Vision transformer. Cut the image into patches, treat them as tokens, run a standard transformer. Beats CNNs given enough data.', see: 'vis-vit' },

  /* --- embodied --- */
  { term: 'Action chunking', sym: '$k$', def: 'Predict a sequence of $k$ future actions from one observation. Shortens the effective horizon, smooths motion, and amortizes inference latency.', see: 'emb-policies' },
  { term: 'Behavior cloning', def: 'Supervised imitation: regress from observations to expert actions. Simple and effective, but its error compounds as $O(\\epsilon T^2)$.', see: 'emb-why-robots' },
  { term: 'Covariate shift', def: 'The policy visits states the expert never demonstrated, where it is less accurate, so deviations compound. The central problem of imitation learning.', see: 'emb-why-robots' },
  { term: 'DAgger', def: 'Roll out the policy, have the expert label the states it actually reached, retrain. Restores the linear error bound.', see: 'emb-why-robots' },
  { term: 'Diffusion Policy', def: 'A visuomotor policy that denoises a chunk of continuous actions, representing multimodal action distributions that MSE regression cannot.', see: 'emb-policies' },
  { term: 'Domain randomization', def: 'Randomize simulator parameters widely so reality is one sample from the training distribution. Trades peak simulated performance for real-world robustness.', see: 'emb-benchmarks' },
  { term: 'Embodiment', def: 'The specific robot — kinematics, sensors, gripper. Policies are usually embodiment-specific, which is why cross-embodiment datasets matter.', see: 'emb-why-robots' },
  { term: 'Temporal ensembling', def: 'Average overlapping action predictions from successive chunks. Recovers smoothness without committing to a stale plan.', see: 'emb-policies' },
  { term: 'VLA', def: 'Vision-language-action model. A web-pretrained VLM with an action head, so robot data only has to teach control rather than semantics.', see: 'emb-vla' },
  { term: 'World model', def: 'A learned model of environment dynamics, used to train a policy on imagined rollouts. Horizons stay short because model error compounds.', see: 'emb-world-models' },
];

/* ---------------- learning paths ---------------- */

export const PATHS = [
  {
    name: 'Complete beginner',
    desc: 'No assumed background beyond high-school algebra. Go in order, do not skip the figures, and expect the math track to take the longest.',
    hours: '~15 hours',
    lessons: ['math-vectors', 'math-matrices', 'math-derivatives', 'math-probability',
      'ml-framing', 'ml-linear-regression', 'ml-overfitting', 'ml-logistic',
      'math-optimization', 'nn-perceptron-mlp', 'nn-backprop', 'nn-activations'],
  },
  {
    name: 'I can code, I want to understand deep learning',
    desc: 'Skips the classical-ML detour where possible and drives at the mechanics of training a network.',
    hours: '~9 hours',
    lessons: ['math-derivatives', 'math-jacobian', 'math-optimization', 'math-information',
      'nn-perceptron-mlp', 'nn-backprop', 'nn-activations', 'nn-initialization',
      'nn-normalization', 'nn-losses-training'],
  },
  {
    name: 'The mathematics, properly',
    desc: 'For people who want to read papers without skipping the equations. Do the derivation blocks on the first pass, not the second.',
    hours: '~7 hours',
    lessons: ['math-vectors', 'math-matrices', 'math-eigen-svd', 'math-derivatives',
      'math-jacobian', 'math-probability', 'math-information', 'math-optimization', 'math-numerics'],
  },
  {
    name: 'Tabular / applied ML',
    desc: 'What you actually need for problems with rows and columns, where gradient boosting still beats deep learning.',
    hours: '~6 hours',
    lessons: ['ml-framing', 'ml-linear-regression', 'ml-overfitting', 'ml-regularization',
      'ml-logistic', 'ml-trees-ensembles', 'ml-unsupervised', 'ml-evaluation'],
  },
  {
    name: 'Interview preparation',
    desc: 'The concepts that actually come up: the bias-variance decomposition, why cross-entropy, how backprop works, and how not to fool yourself when evaluating.',
    hours: '~5 hours',
    lessons: ['ml-overfitting', 'ml-regularization', 'ml-logistic', 'ml-evaluation',
      'ml-trees-ensembles', 'nn-backprop', 'math-optimization', 'nn-normalization'],
  },
  {
    name: 'Understand LLMs, end to end',
    desc: 'The shortest honest route from "what is a gradient" to "how does a reasoning model work." Skips vision and classical ML.',
    hours: '~12 hours',
    lessons: ['math-derivatives', 'math-information', 'nn-backprop', 'nn-embeddings',
      'nn-normalization', 'llm-tokenization', 'llm-attention', 'llm-transformer',
      'llm-pretraining', 'llm-scaling', 'llm-finetuning', 'llm-decoding',
      'llm-prompting', 'rl-rlhf', 'fr-reasoning'],
  },
  {
    name: 'Ship an LLM application',
    desc: 'What you need to build something real: prompting, retrieval, decoding, serving costs, and — most importantly — evaluation.',
    hours: '~7 hours',
    lessons: ['llm-tokenization', 'llm-prompting', 'llm-rag', 'llm-decoding',
      'llm-finetuning', 'llm-evaluation', 'sys-inference', 'sys-quantization',
      'fr-limits'],
  },
  {
    name: 'Generative models',
    desc: 'VAEs, GANs, and diffusion, with the score-matching and flow-matching views that unify them.',
    hours: '~6 hours',
    lessons: ['math-probability', 'math-information', 'gen-autoencoders', 'gen-gans',
      'gen-diffusion', 'vis-clip'],
  },
  {
    name: 'Systems and performance',
    desc: 'Why models are slow and expensive, and what actually fixes it. Assumes you can already train something.',
    hours: '~6 hours',
    lessons: ['math-numerics', 'sys-gpu', 'sys-memory', 'sys-quantization',
      'sys-inference', 'llm-moe'],
  },
  {
    name: 'Reinforcement learning',
    desc: 'From Bellman equations to the RL that post-trains frontier models.',
    hours: '~6 hours',
    lessons: ['math-probability', 'rl-mdp', 'rl-model-free', 'rl-policy-gradient',
      'rl-rlhf', 'fr-reasoning'],
  },
  {
    name: 'Robot learning / embodied AI',
    desc: 'From why imitation drifts to π₀ and world models. Assumes you know backprop and have seen diffusion.',
    hours: '~8 hours',
    lessons: ['gen-diffusion', 'vis-vlm', 'emb-why-robots', 'emb-policies',
      'emb-vla', 'emb-world-models', 'emb-benchmarks'],
  },
  {
    name: 'Research orientation',
    desc: 'For someone starting a PhD or moving into research: the open problems, how to run experiments that hold up, and how to read.',
    hours: '~8 hours',
    lessons: ['ml-framing', 'ml-overfitting', 'ml-evaluation', 'fr-interpretability',
      'fr-architectures', 'fr-limits', 'pr-experiments', 'pr-reading'],
  },
];

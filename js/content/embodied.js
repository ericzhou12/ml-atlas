/* ============================================================
   Track 11 — Physical & Embodied Intelligence
   ============================================================ */

import { t, key, intuition, warn, hist, mathnote, viz, deriv, code, quiz, paper, book, course, blog, video, demo, codeRef,
         tldr, recap, jargon, steps, diagram } from './_helpers.js';

export default [

/* ---------------------------------------------------------- */
{
  id: 'emb-why-robots',
  title: 'Why Robots Are Different',
  sub: 'Everything that made language models work is missing here.',
  mins: 24, level: 'advanced',
  prereq: ['ml-framing', 'nn-losses-training'],
  tags: ['robotics', 'embodied AI', 'imitation learning'],
  sections: [
    tldr(`Everything that made language models work is missing in robotics, and it is worth being precise about
why rather than assuming scale will eventually sort it out.

Text can be scraped; robot episodes must be physically performed in real time on hardware that breaks. A wrong
token is a typo; a wrong torque destroys a gripper. And the i.i.d. assumption fails *structurally* — what the
robot sees next is caused by what it just did.

That last point creates **compounding error**, the central problem of the field: a policy trained only on
expert demonstrations has never seen its own mistakes, so its first small error puts it somewhere it has no
idea how to handle.

The whole field is organised around **borrowing** representations from data that *can* be scraped — which is
why the following lessons are largely about attaching robots to vision-language models.`),

    jargon([
      ['embodied AI', 'AI that acts in the physical world through a body — a robot arm, a mobile base, a humanoid.'],
      ['policy', 'The function mapping what the robot senses to what it does. The thing being learned.'],
      ['episode / demonstration', 'One recorded attempt at a task. The unit of robot data, and expensive: minutes of human time each.'],
      ['teleoperation', 'A human driving the robot directly to produce demonstrations. How most robot data is made.'],
      ['imitation learning', 'Learning a policy by copying demonstrations. Supervised learning applied to control.'],
      ['behaviour cloning', 'The simplest form: supervised regression from observation to action.'],
      ['compounding error', 'Small mistakes moving the robot into states not in the training data, where it errs worse. The characteristic failure of imitation learning.'],
      ['covariate shift', 'The formal name for that: the distribution of states visited differs from the distribution trained on.'],
      ['DAgger', 'Dataset Aggregation. Run the policy, have an expert label the states it actually reaches, retrain. Directly attacks covariate shift.'],
      ['embodiment', 'The specific robot — its kinematics, sensors, and limits. Policies transfer poorly across embodiments.'],
      ['action space', 'What the policy outputs: joint positions, end-effector poses, velocities. The choice matters more than it looks.'],
    ]),

    t(`## The premise fails

Language models work because the internet is a free, enormous, pre-labeled dataset. Every assumption behind that
sentence breaks for robots:

- **Data cannot be scraped.** Every episode must be physically produced, in real time, on hardware that wears out.
- **Actions have consequences.** A wrong token is a typo. A wrong torque is a broken gripper.
- **The distribution depends on the policy.** What the robot sees next is a consequence of what it just did. This is
  the i.i.d. assumption failing structurally, not incidentally.
- **Evaluation requires the physical world.** You cannot benchmark on a held-out test set; you have to run the robot,
  and every trial takes a minute and needs a human to reset the scene.
- **Embodiments differ.** A policy trained on one arm does not transfer to another with different kinematics, and
  there is no common "vocabulary" the way there is for text.`),

    viz('robot-data-scale'),

    t(`Set that figure to ten thousand robots running around the clock for a decade and watch how little ground you
gain. This is not a problem you can spend your way out of at current data efficiency — which is why the entire field
is organized around **borrowing** representations from data that *can* be scraped.`),

    t(`## Imitation learning, and its central defect

The dominant approach is the simplest one: collect human teleoperation demonstrations, then train
$\\pi_\\theta(a \\mid o)$ by supervised learning. This is **behavior cloning**, and it is just regression from
observations to actions.

It works far better than it has any right to, and it has one structural flaw.`),

    viz('covariate-shift'),

    key(`A supervised model is accurate on its training distribution. A policy that makes one small error moves to a
state slightly *off* that distribution, where it is less accurate, which produces a larger error, which takes it
further off-distribution.

Ross & Bagnell showed the resulting error grows as $O(\\epsilon T^2)$ in the horizon rather than $O(\\epsilon T)$.
The quadratic term is the entire difficulty of imitation learning, and it is why a policy with 99% per-step accuracy
can still fail every episode.`),

    t(`Three responses, all in use:

- **DAgger** — roll out the policy, have the expert label the states it actually visited, retrain. Provably restores
  the linear bound. Expensive, because it needs an expert in the loop.
- **Deliberate recovery data.** Teleoperators are instructed to make mistakes and correct them, so the demonstrated
  distribution already covers off-nominal states. This is standard practice and is most of why modern datasets work.
- **Just collect much more data**, widening the covered band until drift rarely leaves it. Widen *demonstrated
  coverage* in the figure to see this working.`),

    t(`## The vocabulary

| Term | Meaning |
|---|---|
| **Embodiment** | The specific robot: kinematics, sensors, gripper. Policies are usually embodiment-specific. |
| **Teleoperation** | A human driving the robot to produce demonstrations. VR controllers, leader-follower arms (ALOHA), or handheld grippers (UMI). |
| **Proprioception** | The robot's sense of its own joint positions and velocities. |
| **End-effector** | The business end — gripper, hand, tool. |
| **Action space** | What the policy outputs: joint velocities, end-effector deltas, or absolute poses. This choice matters more than it sounds. |
| **Control frequency** | How often actions are issued. Manipulation wants 10–50 Hz; locomotion wants 100–1000 Hz. |
| **Horizon** | Steps per episode. Long horizons are where compounding error bites. |

**Action space choice is load-bearing.** Absolute end-effector poses are easier to learn and transfer across
embodiments; joint-space deltas are more precise but embodiment-specific. Delta actions accumulate drift; absolute
actions need accurate calibration.`),

    hist(`Robot learning has had several phases worth knowing about, because the field's vocabulary is layered.

**Classical robotics** (through ~2010) solved perception, planning, and control as separate engineered modules. It
works extremely well in structured environments — this is what runs in factories — and falls apart when the world is
unstructured.

**Deep RL on robots** (2015–2020) tried to learn control from reward. It worked in simulation and struggled on real
hardware: sample complexity in the millions of episodes, reward design difficulties, and safety problems. Real-world RL
remains hard.

**Imitation learning at scale** (2021–) turned out to work far better. Supervised learning is stable, demonstrations
are easier to provide than reward functions, and the approach scales with data. Essentially all current
manipulation systems are imitation-first, with RL used for post-training refinement.`),

    code('Behavior cloning, and watching it drift', `import numpy as np
rng = np.random.default_rng(0)

# Expert: a proportional controller tracking a reference path.
def reference(t): return np.sin(t * 1.1) * 0.9 + 0.15 * t
def expert_action(t, y): return np.clip((reference(t + 0.25) - y) * 2.2, -1.6, 1.6)

# --- collect demonstrations ---
def collect(n_demos, noise=0.0):
    X, A = [], []
    for _ in range(n_demos):
        y = reference(0) + rng.normal(0, noise)
        for t in np.arange(0, 9, 0.12):
            a = expert_action(t, y)
            X.append([t, y]); A.append(a)
            y += (a + rng.normal(0, noise)) * 0.12
    return np.array(X), np.array(A)

# --- a tiny policy: linear regression on [t, y, y^2, 1] ---
def feats(X): return np.c_[X[:, 0], X[:, 1], X[:, 1]**2, np.ones(len(X))]
def fit(X, A): return np.linalg.lstsq(feats(X), A, rcond=None)[0]

def rollout(w, steps=75):
    y, dev = reference(0), []
    for i, t in enumerate(np.arange(0, 9, 0.12)[:steps]):
        a = feats(np.array([[t, y]]))[0] @ w
        y += a * 0.12
        dev.append(abs(y - reference(t)))
    return np.array(dev)

print(f"{'demos':>7} {'injected noise':>16} {'final deviation':>17} {'mean':>8}")
for n, noise in [(2, 0.0), (2, 0.15), (10, 0.0), (10, 0.15), (50, 0.15)]:
    w = fit(*collect(n, noise))
    d = rollout(w)
    print(f"{n:7d} {noise:16.2f} {d[-1]:17.4f} {d.mean():8.4f}")

print("\\nNoise injection during collection widens the covered state distribution,")
print("which is a cheap stand-in for deliberate recovery demonstrations.\\n")

# --- the quadratic bound, empirically ---
w = fit(*collect(3, 0.0))
print("deviation vs horizon (no recovery data):")
d = rollout(w, steps=75)
for h in [10, 20, 40, 75]:
    print(f"  T={h:3d}: deviation {d[h-1]:.4f}   T^2 scaling would predict "
          f"{d[9]*(h/10)**2:.4f}")`),

    quiz('A behavior-cloned policy has 99% per-step accuracy but fails 60% of episodes over a 200-step horizon. Why is this not a contradiction?',
      ['Errors compound: one mistake moves the robot off the demonstrated distribution, where accuracy is lower, and the deviation grows',
       'The per-step accuracy metric is measured incorrectly',
       '99% is simply too low for robotics',
       'The policy is overfitting to the training set'],
      0,
      'Per-step accuracy is measured on the *expert\'s* state distribution. The moment the policy deviates, it is evaluating on states it was never trained on, and its accuracy there is unknown and worse. That is the covariate shift problem, and it is why the error bound is $O(\\epsilon T^2)$ rather than $O(\\epsilon T)$. Recovery demonstrations and DAgger both attack it by making the training distribution match the visited one.'),

    recap(`- List the five ways robotics breaks the assumptions that made language models work.
- Explain compounding error, and why it follows from the policy influencing its own data distribution.
- Say what DAgger does and why it directly targets covariate shift.
- Explain why the field borrows representations from web data rather than collecting more robot data.`),
  ],
  refs: [
    paper('A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning', 'Ross, Gordon & Bagnell', 2011, 'https://arxiv.org/abs/1011.0686', 'DAgger, and the $O(\\epsilon T^2)$ analysis. The foundational result for this track.'),
    paper('Open X-Embodiment: Robotic Learning Datasets and RT-X Models', 'Open X-Embodiment Collaboration', 2023, 'https://arxiv.org/abs/2310.08864', '34 labs pooling 22 embodiments into one dataset. The clearest statement of the data problem and one response to it.'),
    paper('DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset', 'Khazatsky et al.', 2024, 'https://arxiv.org/abs/2403.12945', '76k trajectories across 564 scenes, collected in real homes and offices rather than labs.'),
    course('Deep Reinforcement Learning (CS285)', 'Sergey Levine (Berkeley)', 2023, 'https://rail.eecs.berkeley.edu/deeprlcourse/', 'Lectures 2–3 cover imitation learning and covariate shift properly. Free video.'),
    book('Robotics, Vision and Control', 'Peter Corke', 2017, 'https://petercorke.com/rvc/home/', 'If you need the classical foundations — kinematics, dynamics, control — this is the accessible entry.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'emb-policies',
  title: 'Visuomotor Policies: ACT, Diffusion Policy, and Flow',
  sub: 'The architectural changes that made imitation learning actually work.',
  mins: 26, level: 'advanced',
  prereq: ['emb-why-robots', 'gen-diffusion'],
  tags: ['diffusion policy', 'ACT', 'flow matching'],
  sections: [
    tldr(`For years robot policies were mediocre for a reason that turns out to be sharp, fixable, and a
genuinely good lesson about loss functions.

Human demonstrations are **multimodal**: asked to avoid an obstacle, one demonstrator goes left, another goes
right, and both are correct. Train with mean squared error and you learn the conditional *mean* of the
demonstrations — which here is "straight into the obstacle".

This is not underfitting. A *perfect* MSE-optimal predictor does this, because the mean of a bimodal
distribution sits between the modes. The fix is to output a **distribution** over actions and sample from it,
and the three architectures that did so — ACT, Diffusion Policy, and flow matching — are what made imitation
learning work.`),

    jargon([
      ['multimodal (distribution)', 'Having several distinct peaks. Several genuinely different correct answers. Not the same as "multimodal" meaning images-plus-text.'],
      ['conditional mean', 'The average action given the observation. What MSE regression converges to, and often not a valid action at all.'],
      ['action chunking', 'Predicting a *sequence* of future actions from one observation instead of just the next one.'],
      ['ACT', 'Action Chunking Transformer. Predicts a chunk of actions with a transformer plus a small VAE for multimodality.'],
      ['Diffusion Policy', 'Generating actions with a diffusion model, which represents multimodal distributions naturally.'],
      ['flow matching', 'A diffusion-like generative method needing far fewer sampling steps. Important when you must act at 50 Hz.'],
      ['temporal ensembling', 'Averaging overlapping predicted action chunks for smoother motion.'],
      ['end-effector', 'The business end of the arm — the gripper or tool.'],
      ['horizon', 'How many decision steps until the task ends. Shorter effective horizons mean less compounding.'],
    ]),

    t(`## The regression trap

For years the standard policy was a CNN feeding an MLP that regressed to an action, trained with mean squared error.
It was mediocre, and the reason turns out to be sharp and fixable.

Human demonstrations are **multimodal**. Asked to go around an obstacle, one demonstrator goes left and another goes
right. Both are correct. MSE regression learns the conditional *mean* of the demonstrations.`),

    viz('multimodal-actions'),

    key(`The average of "go left" and "go right" is "go straight into the obstacle."

This is not an underfitting problem — a *perfect* MSE-optimal predictor does this, because the conditional mean of a
bimodal distribution sits between the modes. The only fix is a policy head that represents a **distribution** over
actions and samples from it rather than predicting a point.`),

    t(`## The three fixes that worked

**Action chunking (ACT, 2023).** Predict a *sequence* of $k$ future actions from one observation instead of a single
next action. This helps in three separate ways: it reduces the effective horizon (fewer decision points, so less
compounding), it produces temporally consistent motion, and it amortizes inference latency.`),

    viz('action-chunking'),

    t(`ACT pairs chunking with a CVAE, whose latent variable absorbs the demonstrator's stylistic variation so the
decoder does not have to average over it. Combined with **temporal ensembling** — averaging overlapping predictions
from successive chunks — you get smooth motion without committing to a stale plan.

ACT is what made the **ALOHA** system work: two low-cost arms, a leader-follower teleoperation rig, and roughly 50
demonstrations per task producing reliable fine manipulation. That data efficiency was the surprising part.

**Diffusion Policy (2023).** Represent $p(a_{t:t+k} \\mid o)$ with a conditional diffusion model, denoising a chunk of
continuous actions. Naturally multimodal, stable to train, and it handles high-dimensional action sequences well.
It became the default policy class almost immediately.

**Flow matching (2024–).** Same idea, straighter paths, far fewer integration steps. Since a robot needs actions at
30–50 Hz, the step count is not an aesthetic concern — it is the difference between usable and not. π₀ uses a
flow-matching action expert for exactly this reason.`),

    t(`## Getting the data in the first place

The policy architecture stopped being the bottleneck; data collection became it. The notable systems:

- **ALOHA / Mobile ALOHA** (Stanford) — leader-follower bimanual teleoperation at low cost. Mobile ALOHA added a
  wheeled base and showed household tasks like cooking shrimp and using an elevator.
- **UMI** (Universal Manipulation Interface, Stanford/Columbia) — a handheld gripper with a wrist camera that a human
  carries around. **No robot needed to collect data.** Demonstrations transfer to a real arm because the observation
  and action spaces are defined at the gripper.
- **DROID** — 76k trajectories across 564 real scenes, collected by 50 labs with a standardized rig.
- **RoboTurk / crowdsourced VR** — remote teleoperation at scale.

The pattern: **decouple data collection from the robot**. The most valuable recent contributions are as much about
data-collection hardware as about learning algorithms.`),

    code('Why MSE fails, and what a sampling head fixes', `import numpy as np
rng = np.random.default_rng(0)

# Two equally valid ways to complete the task: y = +1 or y = -1
def demo_actions(n, p_up=0.5):
    return np.where(rng.random(n) < p_up, 1.0, -1.0) + rng.normal(0, 0.12, n)

A = demo_actions(4000)
print("demonstration distribution: bimodal at +1 and -1")
print(f"  mean {A.mean():+.4f}   <- MSE regression predicts THIS")
print(f"  fraction near the mean (|a| < 0.4): {(np.abs(A) < 0.4).mean():.4f}")
print("  i.e. the MSE-optimal prediction is an action nobody ever demonstrated.\\n")

# --- a mixture density head instead ---
def fit_gmm(A, iters=60):
    mu = np.array([-0.5, 0.5]); sd = np.array([0.5, 0.5]); pi = np.array([0.5, 0.5])
    for _ in range(iters):
        r = pi * np.exp(-((A[:, None] - mu)**2) / (2*sd**2)) / sd
        r /= r.sum(1, keepdims=True)
        Nk = r.sum(0)
        pi = Nk / len(A)
        mu = (r * A[:, None]).sum(0) / Nk
        sd = np.sqrt((r * (A[:, None] - mu)**2).sum(0) / Nk) + 1e-6
    return pi, mu, sd

pi, mu, sd = fit_gmm(A)
print("mixture density network recovers both modes:")
for k in range(2):
    print(f"  component {k}: weight {pi[k]:.3f}  mean {mu[k]:+.3f}  sd {sd[k]:.3f}")

samples = np.where(rng.random(4000) < pi[0],
                   rng.normal(mu[0], sd[0], 4000), rng.normal(mu[1], sd[1], 4000))
print(f"  sampled actions near the mean: {(np.abs(samples) < 0.4).mean():.4f}  <- correctly rare\\n")

# --- action chunking: horizon reduction ---
print("effective decision points over a 400-step episode:")
for k in [1, 8, 20, 50]:
    print(f"  chunk k={k:3d}: {400//k:4d} decisions   "
          f"compounding error ~ (T/k)^2 = {(400/k)**2:9.0f}")
print("\\nChunking shortens the effective horizon, which attacks the quadratic term directly.")`),

    quiz('Why does Diffusion Policy outperform an MSE-regression policy on the same demonstrations?',
      ['It models a distribution over actions and can sample one mode, rather than predicting the mean of several valid behaviors',
       'It has more parameters',
       'Diffusion models are better at processing images',
       'It trains for longer'],
      0,
      'The demonstrations are multimodal — several distinct action sequences complete the task. MSE regression is provably driven to the conditional mean, which for a multimodal distribution is often an action that is not merely suboptimal but invalid (straight into the obstacle). A diffusion or flow head represents $p(a\\mid o)$ and samples from it, committing to one mode. Same data, same encoder — the loss function was the problem.'),

    recap(`- Explain why MSE regression fails on multimodal demonstrations, and why a *perfect* MSE model still
  fails.
- Say what action chunking is and name the three separate problems it helps with.
- Explain why a generative action head fixes what regression could not.
- Say why flow matching is preferred over diffusion when the robot must act at high frequency.`),
  ],
  refs: [
    paper('Diffusion Policy: Visuomotor Policy Learning via Action Diffusion', 'Chi et al.', 2023, 'https://arxiv.org/abs/2303.04137', 'Columbia/Stanford/TRI. The paper that changed the default policy class. Very readable.'),
    paper('Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT/ALOHA)', 'Zhao et al.', 2023, 'https://arxiv.org/abs/2304.13705', 'Stanford. Action chunking, temporal ensembling, and $20k of hardware doing precise bimanual work.'),
    paper('Mobile ALOHA', 'Fu, Zhao & Finn', 2024, 'https://arxiv.org/abs/2401.02117', 'Whole-body teleoperation and household tasks with ~50 demonstrations each.'),
    paper('Universal Manipulation Interface', 'Chi et al.', 2024, 'https://arxiv.org/abs/2402.10329', 'Collect demonstrations with a handheld gripper, no robot required. One of the more important practical ideas in the field.'),
    paper('Flow Matching for Generative Modeling', 'Lipman et al.', 2022, 'https://arxiv.org/abs/2210.02747', 'The generative formulation that π₀ adopts for its action expert.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'emb-vla',
  title: 'Vision-Language-Action Models',
  sub: 'Borrowing semantics from the web because robot data cannot supply it.',
  mins: 26, level: 'frontier',
  prereq: ['emb-policies', 'vis-vlm'],
  tags: ['VLA', 'pi-zero', 'RT-2', 'OpenVLA'],
  sections: [
    tldr(`A policy trained only on robot data can do the tasks it was shown, with the objects it was shown. It
cannot generalize to a new object or a new phrasing, because a few hundred thousand episodes contain almost no
semantic diversity.

A vision-language model trained on the web already has that semantics. So: **start from a pretrained VLM and
attach an action output.** The robot data then only has to teach control — not what a mug is, or what "the
extinct animal" means.

This is currently the dominant paradigm in robot learning, and the results include genuine emergent
generalization: instructions the model was never trained on, working because the underlying VLM understood
them.`),

    jargon([
      ['VLA', 'Vision-Language-Action model. A VLM with an action output, fine-tuned on robot data.'],
      ['co-training', 'Continuing to train on web data while fine-tuning on robot data, so semantic knowledge is not forgotten.'],
      ['action tokens', 'Expressing continuous actions as discrete tokens, so a language model can emit them directly.'],
      ['action expert / head', 'A separate module generating continuous actions, attached to the VLM\'s representation.'],
      ['cross-embodiment', 'Training one policy across data from many different robots. Open X-Embodiment is the standard dataset.'],
      ['emergent generalization', 'Handling instructions or objects absent from robot training data, because the VLM supplied the understanding.'],
      ['inference latency', 'How long a forward pass takes. A hard constraint: a 7B VLM cannot run at 50 Hz on a robot.'],
      ['open-loop / closed-loop', 'Executing a planned sequence blindly versus re-sensing and re-deciding continuously.'],
    ]),

    t(`## The bet

A policy trained only on robot data can execute the tasks it was shown. It cannot generalize to an object it has never
seen, or follow a phrasing it has never encountered, because a few hundred thousand episodes contain almost no
semantic diversity.

A vision-language model trained on the web has that semantics already. **A VLA starts from a pretrained VLM and
attaches an action output.** The robot data then only has to teach *control*, not what a mug is.`),

    viz('vla-architecture'),

    t(`## The lineage

**RT-1** (Google, 2022) — a transformer over image and language tokens producing discretized actions, trained on 130k
episodes over 17 months. Established that the recipe scales.

**RT-2** (2023) — the key move: take a *web-pretrained* VLM (PaLI-X/PaLM-E) and fine-tune it on robot data with
actions expressed as text tokens. Co-training on web data preserved semantic knowledge, producing genuine emergent
generalization — instructions like "pick up the extinct animal" worked, because the VLM knew what that meant and only
had to learn the picking up.

**RT-X / Open X-Embodiment** (2023) — train across 22 embodiments pooled from 34 institutions. Cross-embodiment
transfer improves performance on each individual robot, which was not obvious in advance.

**OpenVLA** (Stanford/Berkeley, 2024) — a 7B open-source VLA on Llama 2 + DINOv2/SigLIP, trained on 970k
Open X-Embodiment episodes. Beat the much larger closed RT-2-X on several evaluations, and made the whole area
reproducible.

**π₀ / π₀.₅** (Physical Intelligence, 2024–25) — a PaliGemma VLM with a separate **flow-matching action expert**
producing continuous 50 Hz action chunks. Trained on a large cross-embodiment mixture including the company's own
fleet. π₀.₅ pushed on open-world generalization — cleaning kitchens and bedrooms in homes the robot had never entered.

**Gemini Robotics** (DeepMind, 2025) and **GR00T N1** (NVIDIA, 2025) followed with the same overall shape: a large
pretrained multimodal backbone plus a fast action decoder, trained on heterogeneous embodiment data.`),

    key(`Two design choices separate these systems, and both are visible in the figure above.

**How actions leave the model.** Discretizing actions into bins and emitting them as text tokens (RT-2, OpenVLA)
requires no architectural change and reuses the language head — but autoregressive decoding is slow and binning caps
precision, giving ~3–10 Hz control. Attaching a continuous action expert trained by flow matching or diffusion (π₀)
gives smooth 50 Hz multimodal control at the cost of a bespoke architecture.

**Whether web data stays in the mixture.** Co-training on vision-language data throughout — not just pretraining then
fine-tuning — is what preserves generalization. Fine-tuning purely on robot data causes the model to forget the
semantics it was chosen for in the first place.`),

    t(`## What generalizes, and what does not

Honest current picture:

**Works.** Novel objects within familiar categories. Paraphrased instructions. New backgrounds and lighting. Some
compositional instruction following. Cross-embodiment transfer when action spaces are aligned.

**Does not, reliably.** Genuinely novel manipulation *skills* — the model recombines what it was shown rather than
inventing motions. Long-horizon tasks without intermediate structure. Precise force-controlled contact (insertion,
wiping, anything where feel matters more than sight). Recovery from unusual failures. Anything requiring real
multi-minute planning.

**The measurement problem.** Reported success rates come from small trial counts on specific hardware in specific
rooms. A "70% success rate" typically means 14 of 20 trials, and the binomial confidence interval on that is roughly
±20 points. Cross-paper comparison is largely not meaningful.`),

    warn(`**Be careful with demo videos.** Robot demonstrations are selected, often speeded up, and frequently show the
best of many attempts. This is not unique to robotics but the gap between demo and reliability is unusually large
here, because a 70% success rate produces a great video and an unusable product.

Ask: how many trials, in how many distinct scenes, with what reset protocol, and on hardware other than the authors'?`),

    code('Action tokenization, and why control frequency suffers', `import numpy as np

# --- RT-2 / OpenVLA style: discretize each action dimension into bins ---
def tokenize(action, n_bins=256, lo=-1.0, hi=1.0):
    a = np.clip(action, lo, hi)
    return np.round((a - lo) / (hi - lo) * (n_bins - 1)).astype(int)

def detokenize(tokens, n_bins=256, lo=-1.0, hi=1.0):
    return tokens / (n_bins - 1) * (hi - lo) + lo

action = np.array([0.137, -0.442, 0.891, 0.0, -0.213, 0.667, 1.0])   # 7-DoF
tok = tokenize(action)
rec = detokenize(tok)
print("action     ", np.round(action, 4))
print("tokens     ", tok)
print("recovered  ", np.round(rec, 4))
print(f"max quantization error: {np.abs(action-rec).max():.5f} "
      f"({np.abs(action-rec).max()/2*100:.3f}% of range)\\n")

for bins in [64, 256, 1024]:
    err = 1.0 / (bins - 1)
    print(f"  {bins:5d} bins -> resolution {err:.5f}  "
          f"({err*1000:.2f} mm if the range is 1 m)")

# --- the latency arithmetic ---
print("\\ncontrol frequency by action representation:")
rows = [
    ("token decoding, 7 dims, 1 step",   7,  0.020, 1),
    ("token decoding, 7 dims, chunk 10", 70, 0.020, 10),
    ("flow expert, 4 integration steps", 4,  0.012, 50),
]
for name, forwards, per_forward, chunk in rows:
    latency = forwards * per_forward
    hz = chunk / latency
    print(f"  {name:36s} {latency*1000:6.0f} ms -> {hz:6.1f} Hz")

print("\\nAutoregressive token decoding costs one forward pass PER DIMENSION.")
print("A flow-matching expert emits the whole chunk in a few passes, which is")
print("the difference between 5 Hz and 50 Hz control.")`),

    quiz('Why do VLAs co-train on web vision-language data rather than just fine-tuning a VLM on robot data?',
      ['Fine-tuning only on robot data causes the model to forget the semantic knowledge it was chosen for',
       'Web data provides more action labels',
       'It makes training faster',
       'Robot data is too noisy to train on alone'],
      0,
      'The reason to start from a VLM is its semantics — it knows what a mug is, what "extinct animal" means. Robot datasets contain almost no semantic diversity, so fine-tuning exclusively on them causes catastrophic forgetting of exactly the capability you wanted. Keeping web data in the mixture throughout preserves it. RT-2 demonstrated this directly: co-training produced emergent instruction generalization that robot-only fine-tuning did not.'),

    recap(`- State the bet a VLA makes: the web supplies semantics, robot data supplies control.
- Explain what co-training preserves and what happens without it.
- Give an example of emergent generalization and say where the understanding came from.
- Name the practical constraint that limits how large a VLA can be, and the usual architectural response.`),
  ],
  refs: [
    paper('RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control', 'Brohan et al.', 2023, 'https://arxiv.org/abs/2307.15818', 'Google DeepMind. The co-training insight, and the emergent generalization results.'),
    paper('OpenVLA: An Open-Source Vision-Language-Action Model', 'Kim et al.', 2024, 'https://arxiv.org/abs/2406.09246', 'Stanford/Berkeley/TRI. 7B, open weights, open data, reproducible. The best entry point for actually working on this.'),
    paper('π₀: A Vision-Language-Action Flow Model for General Robot Control', 'Black et al. (Physical Intelligence)', 2024, 'https://arxiv.org/abs/2410.24164', 'The flow-matching action expert, and 50 Hz continuous control.'),
    paper('π₀.₅: A VLA with Open-World Generalization', 'Physical Intelligence', 2025, 'https://arxiv.org/abs/2504.16054', 'Co-training across heterogeneous data sources, evaluated in homes the robot had never seen.'),
    paper('RT-1: Robotics Transformer for Real-World Control at Scale', 'Brohan et al.', 2022, 'https://arxiv.org/abs/2212.06817', 'The system that established the scaling recipe.'),
    paper('Gemini Robotics', 'Google DeepMind', 2025, 'https://arxiv.org/abs/2503.20020', 'A frontier multimodal model adapted to embodied control.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'emb-world-models',
  title: 'World Models and Learning in Imagination',
  sub: 'If real experience is expensive, learn a simulator and dream instead.',
  mins: 24, level: 'frontier',
  prereq: ['rl-model-free', 'gen-diffusion'],
  tags: ['world models', 'Dreamer', 'model-based RL'],
  sections: [
    tldr(`Model-free RL needs millions of environment steps. On a real robot at 10 Hz that is over a day of
flawless continuous operation — and it needs far more than a million.

A **world model** learns to predict the environment's dynamics. Then you train the policy *inside* the learned
model, where a step costs a forward pass instead of a second of real time. Real interaction is spent learning
the model; policy optimization becomes nearly free.

The key engineering insight is that the imagination happens **in latent space**, never in pixels. Predicting
the next 64×64 image is much harder than predicting the next compact latent, and the policy never needed pixels
anyway.`),

    jargon([
      ['world model', 'A learned model of environment dynamics: given a state and an action, predict what happens next.'],
      ['model-based RL', 'RL that uses such a model, as opposed to learning purely from real experience.'],
      ['imagination / rollout', 'Simulating a trajectory inside the learned model rather than in the real world.'],
      ['latent space', 'The compact learned representation the model operates in, instead of raw pixels.'],
      ['RSSM', 'Recurrent State-Space Model. The Dreamer line\'s architecture: a latent state carried forward and predicted.'],
      ['actor / critic', 'The policy and the value estimator, both trained on imagined trajectories.'],
      ['model exploitation', 'The policy finding and exploiting flaws in the learned model — high imagined reward, real-world failure. The characteristic failure mode.'],
      ['imagination horizon', 'How many steps forward you trust the model before compounding prediction error makes it useless.'],
      ['sample efficiency', 'How much real-world interaction is needed. The entire point of this approach.'],
    ]),

    t(`## The motivation

Model-free RL needs millions of environment steps. On a real robot at 10 Hz, a million steps is over a day of
continuous operation with no failures — and RL requires far more than a million.

A **world model** learns the environment's dynamics, then trains the policy inside the model. Real interaction is
spent learning the model; policy optimization becomes nearly free.`),

    t(`## Dreamer

The Dreamer line (Hafner et al.) is the clearest instance. Three components trained together:

1. A **recurrent state-space model** encoding observations into a compact latent $z_t$ and predicting
   $z_{t+1}$ from $z_t$ and $a_t$.
2. An **actor** trained entirely on trajectories imagined in latent space.
3. A **critic** bootstrapping value beyond the imagination horizon.

Crucially the imagination happens **in latent space**, never in pixels — predicting the next 64×64 image is far harder
than predicting the next 1024-dimensional latent, and the policy does not need pixels.

DreamerV3 was notable for using one hyperparameter configuration across more than 150 tasks — Atari, continuous
control, Minecraft — including collecting diamonds in Minecraft from scratch, a long-horizon sparse-reward task that
had resisted everything else. **DayDreamer** put the same algorithm on real robots and got a quadruped walking in
about an hour of real experience.`),

    viz('world-model-rollout'),

    key(`The catch is visible in that figure. A model with 2% one-step error diverges from reality as the horizon
grows, and a policy optimized against a diverged model learns to **exploit the model's errors** rather than solve the
task — the model-based analogue of reward hacking.

Two standard mitigations, both in the figure: keep the imagination horizon short (Dreamer uses ~15 steps and
bootstraps a value function past that), and **replan from a fresh real observation** every few steps, which is what
model-predictive control does.`),

    t(`## Video prediction as a policy

A different lineage treats video generation itself as the planner: predict *what should happen*, then work out the
actions that produce it.

- **UniPi** (Google, 2023) — generate a video of the task being completed, then run an inverse dynamics model to
  recover the actions between frames.
- **Dreamitate** (Columbia, 2024) — fine-tune a video model to imagine a human hand performing the task, then track
  the imagined motion and execute it.
- **Genie / Genie-2 / Genie-3** (DeepMind) — learn *action-controllable* world models from unlabeled internet video,
  producing interactive environments with no action labels at all.
- **GR00T N1** and related systems use video-pretrained representations as a backbone for control.

The appeal is enormous: **internet video is the closest thing to a scrapable dataset of physical dynamics**, and it
sidesteps the data problem from the first lesson. The difficulty is the same one: getting from "what should happen" to
"which torques produce it" needs an inverse dynamics model, and that needs action-labeled data again — just less of it.`),

    intuition(`The recurring theme across this whole track: **actions are the scarce modality.** Images and video are
abundant; language is abundant; state-action pairs on real hardware are not.

Every major approach is a strategy for reducing how much action-labeled data you need. VLAs borrow semantics from
web images and text. World models borrow dynamics from video. Sim-to-real borrows from physics engines. UMI borrows
from human hands. They are all answers to the same question.`),

    warn(`I should flag an uncertainty here. You mentioned **"dream2flow"** — I could not confirm a paper by that exact
name and I am not going to invent a citation for it. The name suggests the intersection covered above: a learned world
model ("dream") producing trajectories or flows used for control. If you meant something specific, send the link or
the authors and I will add it properly. Related work I *am* confident about is cited below.`),

    code('Model-based vs model-free sample efficiency', `import numpy as np
rng = np.random.default_rng(0)

# Simple 1-D system the agent must learn to control.
def true_step(x, v, a):
    nv = 0.98*v + 0.1*a - 0.02*x
    return x + nv, nv

# --- learn a linear dynamics model from N real transitions ---
def learn_model(n_real):
    X, Y = [], []
    x, v = rng.normal(), 0.0
    for _ in range(n_real):
        a = rng.normal()
        nx, nv = true_step(x, v, a)
        X.append([x, v, a]); Y.append([nx, nv])
        x, v = nx, nv
        if abs(x) > 6: x, v = rng.normal(), 0.0
    return np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)[0]

def rollout_error(W, horizon):
    """How far does the learned model drift from truth over a horizon?"""
    errs = []
    for _ in range(200):
        x = v = 0.0
        mx, mv = 0.0, 0.0
        for t in range(horizon):
            a = rng.normal()
            x, v = true_step(x, v, a)
            mx, mv = np.array([mx, mv, a]) @ W
            errs.append(abs(mx - x))
    return np.mean(errs)

print(f"{'real transitions':>18} {'H=1':>9} {'H=5':>9} {'H=15':>9} {'H=50':>9}")
for n in [50, 200, 1000, 5000]:
    W = learn_model(n)
    row = "".join(f"{rollout_error(W, h):9.4f}" for h in [1, 5, 15, 50])
    print(f"{n:18d}{row}")

print("\\nError compounds with horizon at every data budget. More data lowers the")
print("whole curve but does not change its shape -- which is why short imagination")
print("horizons plus a learned value function beat long rollouts.\\n")

# --- the sample-efficiency argument ---
print("to get 1M policy-training steps:")
print(f"  model-free, real robot at 10 Hz : {1e6/10/3600:8.1f} hours of robot time")
print(f"  model-based, 5k real + imagined : {5e3/10/3600:8.2f} hours of robot time")
print(f"  speedup: {1e6/5e3:.0f}x fewer real interactions")`),

    quiz('Why does Dreamer imagine only ~15 steps ahead rather than full episodes?',
      ['Model error compounds with horizon, so a policy optimized on long rollouts learns to exploit the model rather than solve the task',
       'Longer rollouts are too computationally expensive',
       'The latent state cannot represent long sequences',
       'Fifteen steps is enough to complete most tasks'],
      0,
      'Any learned model has one-step error, and it compounds. Past a modest horizon the imagined trajectory has little to do with reality, and the optimizer — which is very good at finding whatever maximizes its objective — will discover the model\'s failure modes instead of a good policy. Short horizons keep the imagination trustworthy; a learned value function supplies the return beyond them. It is the same structure as the KL leash in RLHF: constrain the optimizer to where the proxy is still valid.'),

    recap(`- Explain why a learned world model changes the economics of RL on real hardware.
- Say why imagination happens in latent space rather than in pixels.
- Describe model exploitation, and connect it to reward hacking in RLHF.
- Say what limits the imagination horizon.`),
  ],
  refs: [
    paper('Mastering Diverse Domains through World Models (DreamerV3)', 'Hafner et al.', 2023, 'https://arxiv.org/abs/2301.04104', 'One hyperparameter set across 150+ tasks, including Minecraft diamonds from scratch.'),
    paper('DayDreamer: World Models for Physical Robot Learning', 'Wu et al.', 2022, 'https://arxiv.org/abs/2206.14176', 'Dreamer on real hardware — a quadruped learning to walk in about an hour.'),
    paper('Learning Universal Policies via Text-Guided Video Generation (UniPi)', 'Du et al.', 2023, 'https://arxiv.org/abs/2302.00111', 'Generate the video, then invert it for actions.'),
    paper('Genie: Generative Interactive Environments', 'Bruce et al.', 2024, 'https://arxiv.org/abs/2402.15391', 'Action-controllable world models learned from unlabeled internet video. No action labels anywhere.'),
    paper('Dreamitate: Real-World Visuomotor Policy Learning via Video Generation', 'Liang et al.', 2024, 'https://arxiv.org/abs/2406.16862', 'Imagine a hand doing the task, track it, execute.'),
    paper('World Models', 'Ha & Schmidhuber', 2018, 'https://worldmodels.github.io/', 'The paper that popularized the framing, with an excellent interactive write-up.'),
  ],
},

/* ---------------------------------------------------------- */
{
  id: 'emb-benchmarks',
  title: 'Simulation, Benchmarks, and Honest Evaluation',
  sub: 'BEHAVIOR, sim-to-real, and why robotics results are hard to compare.',
  mins: 24, level: 'advanced',
  prereq: ['emb-why-robots', 'ml-evaluation'],
  tags: ['BEHAVIOR', 'sim2real', 'benchmarks'],
  sections: [
    tldr(`Simulation gives you unlimited data, instant resets, ground-truth state, and no broken hardware. It
also gives you a policy that works in a world that does not exist.

**Domain randomization** is the standard bridge, and its logic is worth appreciating: rather than trying to model
reality accurately, randomize the simulator's parameters so widely that *reality is one sample* from your
training distribution. The policy learns robustness instead of optimality.

The lesson closes on evaluation, which in robotics is genuinely harder than the modelling. Success rates are
reported on a handful of trials with unstated reset protocols, and results across papers are frequently not
comparable at all. Knowing how to read them sceptically is a real skill.`),

    jargon([
      ['sim-to-real gap', 'The performance drop when a policy trained in simulation meets the physical world.'],
      ['domain randomization', 'Randomizing simulator parameters — friction, mass, lighting, latency — so the policy must be robust rather than tuned to one setting.'],
      ['system identification', 'The opposite strategy: measuring reality carefully and matching the simulator to it.'],
      ['photorealism', 'Rendering fidelity. Matters less than physics fidelity for most manipulation.'],
      ['reset protocol', 'How the scene is restored between trials. Rarely stated, and it changes reported success rates substantially.'],
      ['success rate', 'The headline robotics metric. Nearly meaningless without a trial count and a confidence interval.'],
      ['BEHAVIOR', 'A large simulated benchmark of household activities, built for realistic long-horizon tasks.'],
      ['generalization axis', 'What is being varied at test time — new object, new position, new instruction, new scene. Papers often blur these.'],
    ]),

    t(`## Why simulate

Simulation gives you unlimited data, resettable environments, ground-truth state, parallelism, and no broken hardware.
It also gives you a policy that works in a world that does not exist.`),

    viz('sim2real'),

    t(`**Domain randomization** is the standard bridge: rather than modeling reality accurately, randomize the
simulator's parameters — friction, mass, lighting, textures, latency — widely enough that reality is *one sample* from
the training distribution. The policy learns to be robust rather than optimal.

Slide the randomization width in the figure. Narrow gives excellent simulated performance and total real-world
failure. Wide flattens the curve — worse in sim, but reality is inside the covered range. This tradeoff is the whole
of sim-to-real.

Then note what **50 real episodes of fine-tuning** does. A small amount of real data is worth an enormous amount of
randomization, which is the standing argument for real-robot data collection over ever-better simulators.`),

    t(`## BEHAVIOR and the Stanford ecosystem

**BEHAVIOR-1K** (Stanford Vision and Learning Lab) is the most ambitious embodied benchmark. Rather than picking tasks
researchers find convenient, it surveyed people about what they actually want robots to do, and built the resulting
**1,000 household activities** — cleaning, cooking, tidying, organizing — in 50 fully interactive scenes.

What makes it hard on purpose:

- **Realistic physics**, including fluids, deformables, heat, and cleanliness state.
- **Long horizons** — real household tasks take hundreds of steps and have many subgoals.
- **Semantic state**, not just geometric. "Is the counter clean" and "is the soup cooked" are predicates the simulator
  actually tracks.
- **Mobile manipulation** — navigation and manipulation together, which most benchmarks separate.

It runs on **OmniGibson**, built on NVIDIA Omniverse. Success rates on the full benchmark remain low, which is the
point: it was designed not to saturate.

The broader Stanford ecosystem is worth knowing as a whole: **iGibson** and **OmniGibson** (simulation), **BEHAVIOR**
(tasks), **ALOHA** and **Mobile ALOHA** (hardware and teleoperation), **UMI** (data collection), **OpenVLA**
(models), and **HAI**, whose annual **AI Index Report** is the most useful single source of measured, cited trend data
on the field — compute, cost, capability, investment, policy — and a good corrective to vibes-based claims about
progress.`),

    t(`## The other benchmarks

| Benchmark | What it measures |
|---|---|
| **BEHAVIOR-1K** | Long-horizon household activities, full physics, mobile manipulation |
| **RoboCasa** | Large-scale kitchen manipulation, heavily generative-AI-augmented scenes |
| **LIBERO** | Lifelong / continual robot learning across task suites |
| **CALVIN** | Long-horizon language-conditioned manipulation |
| **Meta-World** | 50 manipulation tasks for multi-task and meta-RL |
| **SimplerEnv** | Simulated evaluation *calibrated to correlate with real-robot results* |
| **Open X-Embodiment** | Not a benchmark — a pooled training dataset across 22 embodiments |

**SimplerEnv** deserves the highlight. The recurring complaint about simulated robot benchmarks is that they do not
predict real performance; SimplerEnv explicitly tries to make simulated rankings match real ones, which is a more
useful goal than realism for its own sake.`),

    warn(`**Robotics evaluation is genuinely unreliable, and you should read papers accordingly.**

- **Tiny trial counts.** "80% success" is usually 16 of 20 trials. The 95% confidence interval is roughly ±18 points.
  Two systems reporting 70% and 80% are not distinguishable.
- **Non-reproducible setups.** Different arm, gripper, camera placement, lighting, table height, and object instances.
  Nobody can rerun your experiment.
- **Reset protocol matters and is rarely specified.** Are objects placed identically each trial, or randomized within
  a region? This changes results dramatically.
- **Cherry-picked scenes.** The evaluated objects are often exactly the ones in the training set, described as "novel"
  because that instance was held out.
- **Simulation results that do not transfer.** A benchmark number is a claim about the simulator unless transfer is
  demonstrated.

Good practice, when you see it: ≥50 trials per condition, reported confidence intervals, a documented reset protocol,
evaluation on hardware the authors did not build, and released code plus checkpoints.`),

    t(`## Where this is heading

An honest read of the state of things:

- **Manipulation is where language was around 2019.** The architecture question is largely settled (pretrained
  multimodal backbone + expressive action head); the data question is not.
- **Data collection hardware is as important as modeling.** UMI-style approaches that decouple demonstrations from
  robots may matter more than the next architecture.
- **Video pretraining is the most promising route around the data wall**, because internet video is the only abundant
  source of physical dynamics.
- **Evaluation needs to improve before progress can be measured properly.** This is currently the field's weakest link.
- **Locomotion is much further along than manipulation** — RL in simulation with domain randomization genuinely works
  for legged robots, because the contact dynamics are simpler and simulators model them well.`),

    code('Confidence intervals on robot success rates', `import numpy as np
rng = np.random.default_rng(0)

def wilson(successes, n, z=1.96):
    """Wilson score interval - better than normal approximation for small n."""
    if n == 0: return (0, 1)
    p = successes / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, c-h), min(1, c+h)

print("what a reported success rate actually tells you:")
print(f"{'trials':>7} {'observed':>10} {'95% CI':>22} {'width':>8}")
for n in [10, 20, 50, 100, 500]:
    s = round(0.8 * n)
    lo, hi = wilson(s, n)
    print(f"{n:7d} {s/n:10.1%} {f'[{lo:.1%}, {hi:.1%}]':>22} {(hi-lo)*100:7.1f}pp")

print("\\ncan you distinguish two policies?")
for n in [20, 50, 200]:
    a_lo, a_hi = wilson(round(0.70*n), n)
    b_lo, b_hi = wilson(round(0.80*n), n)
    overlap = a_hi > b_lo
    print(f"  n={n:3d}: 70% [{a_lo:.2f},{a_hi:.2f}] vs 80% [{b_lo:.2f},{b_hi:.2f}]  "
          f"{'NOT distinguishable' if overlap else 'distinguishable'}")

# how many trials do you need?
print("\\ntrials needed to distinguish a true 70% from a true 80% (80% power):")
for n in range(10, 600, 10):
    wins = 0
    for _ in range(2000):
        a = rng.random(n) < 0.70
        b = rng.random(n) < 0.80
        if b.mean() > a.mean(): wins += 1
    if wins/2000 >= 0.80:
        print(f"  n = {n} trials per condition")
        break`),

    quiz('A paper reports 85% success on a new manipulation task, over 20 trials. What is the appropriate reading?',
      ['The 95% CI is roughly [64%, 95%] — consistent with anything from mediocre to excellent, and not comparable to another paper\'s number',
       'The policy succeeds 85% of the time',
       'The result is invalid',
       'It is better than a policy reporting 75% on the same task'],
      0,
      '17 of 20 successes gives a Wilson interval of about [64%, 95%]. That is compatible with a genuinely strong policy and with a fairly weak one, and it certainly does not separate 85% from another paper\'s 75%. Combined with different hardware, scenes, objects, and reset protocols, cross-paper comparison in robotics is mostly not meaningful. Look for trial counts, confidence intervals, and a stated reset protocol.'),

    recap(`- Explain domain randomization as "make reality one sample from the training distribution".
- Describe the tradeoff as randomization width increases, in both simulated and real performance.
- Say why 50 real episodes can outweigh a great deal of randomization.
- Read a robotics result sceptically: name three things you would ask for before believing a success rate.`),
  ],
  refs: [
    paper('BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark', 'Li et al. (Stanford Vision and Learning Lab)', 2022, 'https://arxiv.org/abs/2403.09227', '1,000 household activities chosen by surveying what people actually want, in full-physics interactive scenes.'),
    blog('AI Index Report', 'Stanford HAI', 2025, 'https://aiindex.stanford.edu/report/', 'The most useful single source of measured, cited trend data on the field. Read the robotics and compute chapters.'),
    paper('OmniGibson / iGibson', 'Stanford Vision and Learning Lab', 2023, 'https://behavior.stanford.edu/omnigibson/', 'The simulator BEHAVIOR runs on.'),
    paper('SimplerEnv: Evaluating Real-World Robot Manipulation Policies in Simulation', 'Li et al.', 2024, 'https://arxiv.org/abs/2405.05941', 'Simulated evaluation deliberately calibrated to correlate with real-robot rankings.'),
    paper('RoboCasa: Large-Scale Simulation of Everyday Tasks', 'Nasiriany et al.', 2024, 'https://arxiv.org/abs/2406.02523', ''),
    paper('Sim-to-Real Transfer of Robotic Control with Dynamics Randomization', 'Peng et al.', 2017, 'https://arxiv.org/abs/1710.06537', 'The domain randomization argument, made cleanly.'),
    paper('Learning Agile and Dynamic Motor Skills for Legged Robots', 'Hwangbo et al.', 2019, 'https://arxiv.org/abs/1901.08652', 'Why locomotion sim-to-real works better than manipulation.'),
  ],
},

];

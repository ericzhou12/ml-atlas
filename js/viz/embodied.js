/* ============================================================
   viz/embodied.js — robot learning and embodied AI figures.
   ============================================================ */

import { panel } from '../ui.js';
import { cssVar, alpha, mix, seqMap, divergeMap, LA, rng, fmt, SERIES } from '../plot.js';

const V = {};
export default V;

/* ============================================================
   Covariate shift — the central problem of imitation learning
   ============================================================ */

V['covariate-shift'] = (host) => {
  // Expert follows a smooth reference path. A behavior-cloned policy is
  // accurate ON the demonstrated states and degrades off them, so any
  // deviation compounds. DAgger re-collects expert labels on visited states.
  const expert = (x) => 0.9 * Math.sin(x * 1.1) + 0.15 * x;
  const expertAction = (x, y) => {
    const target = expert(x + 0.25);
    return Math.max(-1.6, Math.min(1.6, (target - y) * 2.2));
  };

  panel(host, {
    title: 'Why behavior cloning drifts',
    height: 330,
    plot: { xlim: [-0.3, 9], ylim: [-3.2, 3.2], xlabel: 'time / forward progress', ylabel: 'lateral state' },
    controls: [
      { type: 'slider', key: 'demos', label: 'demonstrations', min: 1, max: 60, step: 1, value: 6 },
      { type: 'slider', key: 'noise', label: 'policy error ε', min: 0, max: .5, step: .005, value: .12, fmt: (v) => v.toFixed(3) },
      { type: 'slider', key: 'band', label: 'demonstrated state coverage', min: .1, max: 2.5, step: .05, value: .45 },
      { type: 'check', key: 'dagger', label: 'DAgger (relabel on-policy states)', value: false },
      { type: 'check', key: 'shownoise', label: 'show demonstration data', value: true },
    ],
    draw(p, s, P) {
      p.clear().axes();
      p.clip();

      // --- demonstrated distribution ---
      const r = rng(7);
      const demoPts = [];
      for (let d = 0; d < Math.round(s.demos); d++) {
        const jitter = r.normal(0, s.band * 0.35);
        for (let x = 0; x <= 9; x += 0.25) {
          demoPts.push([x, expert(x) + jitter + r.normal(0, s.band * 0.12)]);
        }
      }
      if (s.shownoise) p.points(demoPts, { r: 1.6, color: alpha(cssVar('--s1'), .18) });

      // coverage band
      const up = [], dn = [];
      for (let x = 0; x <= 9; x += 0.1) { up.push([x, expert(x) + s.band]); dn.push([x, expert(x) - s.band]); }
      p.line(up, { color: alpha(cssVar('--s1'), .5), width: 1.2, dash: [4, 3] });
      p.line(dn, { color: alpha(cssVar('--s1'), .5), width: 1.2, dash: [4, 3] });

      // --- expert reference ---
      p.fn(expert, { color: cssVar('--s1'), width: 2.6, from: 0, to: 9 });

      // --- roll out the learned policy ---
      const rr = rng(1234);
      const roll = (dagger) => {
        let y = expert(0) + 0.05;
        const path = [[0, y]];
        const errs = [];
        for (let x = 0; x < 9; x += 0.12) {
          const dev = Math.abs(y - expert(x));
          // Accuracy degrades once the state leaves the demonstrated band.
          // DAgger collects expert labels wherever the policy actually goes,
          // so the error stays bounded instead of compounding.
          const off = dagger ? 0 : Math.max(0, dev - s.band) / Math.max(s.band, 1e-6);
          const err = s.noise * (1 + off * 3.2);
          const a = expertAction(x, y) + rr.normal(0, err) + off * s.noise * 1.5;
          y += a * 0.12;
          y = Math.max(-3.1, Math.min(3.1, y));
          path.push([x + 0.12, y]);
          errs.push(Math.abs(y - expert(x + 0.12)));
        }
        return { path, errs };
      };

      const bc = roll(false);
      p.line(bc.path, { color: cssVar('--s2'), width: 2.6 });
      if (s.dagger) {
        const dg = roll(true);
        p.line(dg.path, { color: cssVar('--s3'), width: 2.6 });
      }
      p.clip(false);

      p.legend([
        { label: 'expert trajectory', color: cssVar('--s1') },
        { label: 'demonstrated coverage', color: alpha(cssVar('--s1'), .5), dash: true },
        { label: 'behavior cloning', color: cssVar('--s2') },
        ...(s.dagger ? [{ label: 'DAgger', color: cssVar('--s3') }] : []),
      ], { pos: 'tl' });

      const final = bc.errs[bc.errs.length - 1] || 0;
      const mean = bc.errs.reduce((a, b) => a + b, 0) / Math.max(bc.errs.length, 1);
      P.readout({
        'ε (per-step error)': fmt(s.noise, 3),
        'final deviation': fmt(final, 3),
        'mean deviation': fmt(mean, 3),
        'BC error bound': 'O(ε T²) — quadratic in horizon',
        'DAgger bound': 'O(ε T) — linear',
      });
    },
    caption: 'The policy is only accurate on states the expert visited (the dashed band). One small error moves it slightly off that band, where it is *less* accurate, which produces a larger error — and the deviation compounds. Ross & Bagnell showed behavior cloning\'s error grows as $O(\\epsilon T^2)$ in the horizon, not $O(\\epsilon T)$. **Turn on DAgger**: by querying the expert on states the policy actually reaches, the training distribution matches the test distribution and the bound becomes linear. Widening *demonstrated coverage* achieves the same thing passively — which is why teleoperated data collection deliberately includes recoveries from mistakes.',
  });
};

/* ============================================================
   Multimodal action distributions — why MSE regression fails
   ============================================================ */

V['multimodal-actions'] = (host) => {
  panel(host, {
    title: 'Why robot policies stopped using mean-squared error',
    height: 340,
    plot: { xlim: [-3, 3], ylim: [-2.1, 2.4], equal: true, xlabel: 'x', ylabel: 'y' },
    controls: [
      {
        type: 'select', key: 'head', label: 'action head', value: 'mse',
        options: [
          { value: 'mse', label: 'MSE regression (predicts the mean)' },
          { value: 'gmm', label: 'mixture density network' },
          { value: 'diffusion', label: 'diffusion / flow policy (samples)' },
        ],
      },
      { type: 'slider', key: 'balance', label: 'demonstrations going left', min: 0, max: 1, step: .01, value: .5 },
      { type: 'slider', key: 'nsamples', label: 'policy samples drawn', min: 1, max: 40, step: 1, value: 12 },
      { type: 'check', key: 'demos', label: 'show demonstrations', value: true },
    ],
    draw(p, s, P) {
      p.clear().axes({ nx: 5, ny: 4 });
      p.clip();

      // obstacle
      const g = p.ctx;
      g.save();
      g.fillStyle = alpha(cssVar('--danger'), .22);
      g.strokeStyle = cssVar('--danger');
      g.lineWidth = 1.6;
      const ox = p.px(0), oy = p.py(0.15), rw = 0.55 * p.sx, rh = 0.75 * p.sy;
      g.fillRect(ox - rw, oy - rh, rw * 2, rh * 2);
      g.strokeRect(ox - rw, oy - rh, rw * 2, rh * 2);
      g.restore();
      p.text(0, 0.15, 'obstacle', { color: cssVar('--danger'), size: 10, align: 'center' });

      const start = [-2.5, -1.4], goal = [2.5, 1.6];

      // two valid demonstration modes: over the top, or under the bottom
      const modePath = (up) => {
        const pts = [];
        for (let t = 0; t <= 1.0001; t += 0.02) {
          const x = start[0] + (goal[0] - start[0]) * t;
          const bend = Math.sin(Math.PI * t) * (up ? 1.55 : -1.35);
          const y = start[1] + (goal[1] - start[1]) * t + bend;
          pts.push([x, y]);
        }
        return pts;
      };

      const r = rng(11);
      if (s.demos) {
        for (let k = 0; k < 14; k++) {
          const up = r() < s.balance;
          const path = modePath(up).map(([x, y]) => [x + r.normal(0, .05), y + r.normal(0, .07)]);
          p.line(path, { color: alpha(up ? cssVar('--s1') : cssVar('--s3'), .30), width: 1.3 });
        }
      }

      // --- what each action head produces ---
      const rr = rng(99);
      let hits = 0, total = 0;
      const inObstacle = (x, y) => Math.abs(x) < 0.55 && Math.abs(y - 0.15) < 0.75;

      if (s.head === 'mse') {
        // the conditional mean of a bimodal distribution
        const up = modePath(true), dn = modePath(false);
        const avg = up.map((q, i) => [q[0], s.balance * q[1] + (1 - s.balance) * dn[i][1]]);
        p.line(avg, { color: cssVar('--s2'), width: 3.2 });
        total = 1;
        if (avg.some(([x, y]) => inObstacle(x, y))) hits = 1;
      } else {
        const n = Math.round(s.nsamples);
        for (let k = 0; k < n; k++) {
          let up;
          if (s.head === 'gmm') up = rr() < s.balance;
          else up = rr() < s.balance;
          const jitter = s.head === 'diffusion' ? .10 : .05;
          const path = modePath(up).map(([x, y]) => [x, y + rr.normal(0, jitter)]);
          p.line(path, { color: alpha(up ? cssVar('--s2') : cssVar('--s5'), .75), width: 1.7 });
          total++;
          if (path.some(([x, y]) => inObstacle(x, y))) hits++;
        }
      }

      p.points([start], { r: 6, color: cssVar('--text') });
      p.points([goal], { r: 6, color: cssVar('--ok') });
      p.text(start[0], start[1] - .28, 'start', { color: cssVar('--text-dim'), size: 10, align: 'center' });
      p.text(goal[0], goal[1] + .28, 'goal', { color: cssVar('--ok'), size: 10, align: 'center' });
      p.clip(false);

      P.readout({
        'demonstrations': `${Math.round(s.balance * 100)}% over / ${Math.round((1 - s.balance) * 100)}% under`,
        'head': s.head === 'mse' ? 'predicts E[action | obs]' : 'samples from p(action | obs)',
        'trajectories hitting the obstacle': `${hits} / ${total}`,
        'verdict': s.head === 'mse' && Math.abs(s.balance - 0.5) < 0.35 ? 'COLLISION — the mean of two good paths is a bad path' : 'avoids the obstacle ✓',
      });
    },
    caption: 'Two demonstrations solve this task equally well: go over the obstacle, or go under. Both are correct. **MSE regression predicts the conditional mean of the demonstrations** — and the average of "over" and "under" goes straight through the obstacle. Slide the balance to 50/50 and watch it happen. This is not a capacity problem; a perfect MSE-optimal predictor does this. The fix is a policy head that can represent a *distribution* over actions and sample from it — which is exactly why Diffusion Policy, and later flow-matching policies like π₀, replaced regression heads in visuomotor learning.',
  });
};

/* ============================================================
   Action chunking
   ============================================================ */

V['action-chunking'] = (host) => {
  panel(host, {
    title: 'Action chunking: predicting a sequence, not a step',
    height: 300,
    plot: { xlim: [0, 60], ylim: [-1.6, 1.9], xlabel: 'control step', ylabel: 'commanded position' },
    controls: [
      { type: 'slider', key: 'chunk', label: 'chunk length k', min: 1, max: 32, step: 1, value: 1 },
      { type: 'slider', key: 'jitter', label: 'per-prediction noise', min: 0, max: .35, step: .005, value: .12, fmt: (v) => v.toFixed(3) },
      { type: 'slider', key: 'latency', label: 'inference latency (steps)', min: 0, max: 8, step: 1, value: 2 },
      { type: 'check', key: 'ensemble', label: 'temporal ensembling', value: false },
      { type: 'check', key: 'perturb', label: 'world changes at step 30', value: false },
    ],
    draw(p, s, P) {
      const target = (t) => Math.sin(t * 0.12) + 0.35 * Math.sin(t * 0.31);
      const shifted = (t) => target(t) + (s.perturb && t >= 30 ? 0.9 : 0);

      p.clear().axes();
      p.clip();
      const ref = [];
      for (let t = 0; t <= 60; t += 0.5) ref.push([t, shifted(t)]);
      p.line(ref, { color: alpha(cssVar('--s1'), .85), width: 2.4, dash: [5, 4] });

      const r = rng(5);
      const k = Math.round(s.chunk);
      const lat = Math.round(s.latency);
      const exec = [];
      const pending = [];   // for temporal ensembling: overlapping predictions

      for (let t = 0; t < 60; t += k) {
        // The policy observes at time t, but its output lands `lat` steps later.
        // With k=1 that latency is paid on every step; with a long chunk it is amortized.
        const obsTime = Math.max(0, t - lat);
        const bias = r.normal(0, s.jitter);
        for (let j = 0; j < k && t + j < 60; j++) {
          // predictions drift over the chunk because the model is extrapolating
          const drift = (j / Math.max(k, 1)) * s.jitter * 1.4;
          const v = shifted(obsTime + j) + bias + r.normal(0, drift);
          pending.push({ t: t + j, v, age: j });
        }
      }

      for (let t = 0; t < 60; t++) {
        const cands = pending.filter((q) => q.t === t);
        if (!cands.length) continue;
        let v;
        if (s.ensemble && cands.length > 1) {
          // exponentially weight newer predictions
          let wsum = 0, acc = 0;
          for (const c of cands) { const w = Math.exp(-0.3 * c.age); acc += w * c.v; wsum += w; }
          v = acc / wsum;
        } else v = cands[0].v;
        exec.push([t, v]);
      }

      p.line(exec, { color: cssVar('--s2'), width: 2.2 });
      p.points(exec, { r: 1.8, color: alpha(cssVar('--s2'), .7) });
      if (s.perturb) p.line([[30, -1.6], [30, 1.9]], { color: alpha(cssVar('--danger'), .8), width: 1.4, dash: [4, 3] });
      p.clip(false);

      p.legend([
        { label: 'ideal trajectory', color: cssVar('--s1'), dash: true },
        { label: 'executed', color: cssVar('--s2') },
      ], { pos: 'tl' });

      // metrics
      let jerk = 0, err = 0;
      for (let i = 1; i < exec.length - 1; i++) {
        jerk += Math.abs(exec[i + 1][1] - 2 * exec[i][1] + exec[i - 1][1]);
        err += Math.abs(exec[i][1] - shifted(exec[i][0]));
      }
      const react = s.perturb ? k + lat : 0;
      P.readout({
        'chunk length k': k,
        'policy calls per 60 steps': Math.ceil(60 / k),
        'smoothness (lower = smoother)': fmt(jerk / Math.max(exec.length, 1), 4),
        'tracking error': fmt(err / Math.max(exec.length, 1), 4),
        'steps to react to a change': s.perturb ? `${react}` : '—',
      });
    },
    caption: 'At **k = 1** the policy is queried every step, so independent prediction noise makes the command jitter, and inference latency is paid every step. Raise **k** and the trajectory smooths out and the compute drops — but the robot is now committed to a plan and cannot react for $k$ steps (turn on *world changes at step 30* to see the lag). **Temporal ensembling** — averaging overlapping predictions from successive chunks — recovers smoothness without the commitment, and is what ACT uses. Chunk lengths of 8–50 steps are typical; this single design choice was one of the larger practical wins in visuomotor policy learning.',
  });
};

/* ============================================================
   Sim-to-real and domain randomization
   ============================================================ */

V['sim2real'] = (host) => {
  panel(host, {
    title: 'Domain randomization: trading peak performance for coverage',
    height: 310,
    plot: { xlim: [0, 2], ylim: [0, 1.05], xlabel: 'physical parameter (e.g. friction coefficient)', ylabel: 'success rate' },
    controls: [
      { type: 'slider', key: 'width', label: 'randomization width', min: .01, max: .8, step: .01, value: .03 },
      { type: 'slider', key: 'simCenter', label: 'simulator calibration', min: .4, max: 1.6, step: .01, value: .8 },
      { type: 'slider', key: 'real', label: 'true value in the real world', min: .2, max: 1.9, step: .01, value: 1.15 },
      { type: 'check', key: 'adapt', label: 'add real-world fine-tuning (50 episodes)', value: false },
    ],
    draw(p, s, P) {
      p.clear().axes();
      p.clip();

      // A policy trained over a distribution of parameters generalizes across
      // roughly that range, with a peak that falls as the range widens.
      const w = s.width;
      const peak = 0.98 / (1 + 1.15 * w);
      const spread = 0.055 + w * 1.35;
      const perf = (x) => peak * Math.exp(-((x - s.simCenter) ** 2) / (2 * spread * spread));

      const adapted = (x) => {
        if (!s.adapt) return perf(x);
        // fine-tuning adds a narrow bump at the real value, on top of the prior
        const base = perf(x);
        const bump = 0.9 * Math.exp(-((x - s.real) ** 2) / (2 * 0.09 ** 2));
        return Math.min(0.99, base + bump * (0.35 + 0.55 * base));
      };

      // training distribution
      const dist = [];
      for (let x = 0; x <= 2; x += .005) {
        const d = Math.exp(-((x - s.simCenter) ** 2) / (2 * Math.max(w, .012) ** 2));
        dist.push([x, d * .22]);
      }
      p.area(dist, { color: alpha(cssVar('--s4'), .18) });
      p.line(dist, { color: alpha(cssVar('--s4'), .8), width: 1.4 });

      p.fn(adapted, { color: cssVar('--s2'), width: 2.8, n: 400 });
      p.line([[s.real, 0], [s.real, 1.05]], { color: cssVar('--s3'), width: 2, dash: [5, 3] });
      p.points([[s.real, adapted(s.real)]], { r: 6, color: cssVar('--s3'), stroke: cssVar('--bg-inset'), strokeWidth: 2 });
      p.clip(false);

      p.legend([
        { label: 'training parameter distribution', color: alpha(cssVar('--s4'), .8) },
        { label: 'policy success rate', color: cssVar('--s2') },
        { label: 'reality', color: cssVar('--s3'), dash: true },
      ], { pos: 'tr' });

      const realPerf = adapted(s.real);
      const simPerf = adapted(s.simCenter);
      P.readout({
        'success in simulation': fmt(simPerf * 100, 1) + '%',
        'success in the real world': fmt(realPerf * 100, 1) + '%',
        'sim-to-real gap': fmt((simPerf - realPerf) * 100, 1) + ' points',
        'verdict': realPerf < .25 ? 'policy does not transfer ✗' : realPerf < .6 ? 'partial transfer' : 'transfers ✓',
      });
    },
    caption: 'Start with a narrow randomization width: the policy is superb in simulation and **fails completely** in reality, because the real friction is not the simulated friction and nobody knew that in advance. Widen the randomization and the peak drops while the curve flattens — you give up simulator performance to buy robustness across a range that hopefully contains reality. This is the core sim-to-real tradeoff. Note what *fine-tuning on 50 real episodes* does: a small amount of real data is worth an enormous amount of randomization, which is the argument for real-robot data collection over better simulators.',
  });
};

/* ============================================================
   World models — imagination and compounding error
   ============================================================ */

V['world-model-rollout'] = (host) => {
  panel(host, {
    title: 'Learning in imagination, and where it breaks',
    height: 320,
    plot: { xlim: [0, 40], ylim: [-2.6, 2.6], xlabel: 'horizon (steps imagined)', ylabel: 'state' },
    controls: [
      { type: 'slider', key: 'err', label: 'one-step model error', min: 0, max: .12, step: .001, value: .02, fmt: (v) => v.toFixed(3) },
      { type: 'slider', key: 'horizon', label: 'imagination horizon H', min: 1, max: 40, step: 1, value: 15 },
      { type: 'slider', key: 'nroll', label: 'imagined rollouts', min: 1, max: 24, step: 1, value: 8 },
      { type: 'check', key: 'replan', label: 'replan from a real observation every 5 steps (MPC)', value: false },
    ],
    draw(p, s, P) {
      // true latent dynamics: a damped oscillator
      const step = (x, v, dt = 1) => {
        const nv = v + (-0.045 * x - 0.02 * v) * dt;
        return [x + nv * dt, nv];
      };

      p.clear().axes();
      p.clip();

      // ground truth
      let x = 1.8, v = 0.0;
      const truth = [[0, x]];
      for (let t = 1; t <= 40; t++) { [x, v] = step(x, v); truth.push([t, x]); }
      p.line(truth, { color: cssVar('--s1'), width: 2.8 });

      // imagined rollouts under a slightly wrong model
      const r = rng(21);
      const H = Math.round(s.horizon);
      let maxDiv = 0;
      for (let k = 0; k < Math.round(s.nroll); k++) {
        let ix = 1.8, iv = 0;
        const path = [[0, ix]];
        const bias = r.normal(0, s.err);          // systematic model bias for this rollout
        for (let t = 1; t <= H; t++) {
          if (s.replan && t % 5 === 0) {
            // re-ground on the true state — this is what MPC does
            ix = truth[t][1];
            iv = (truth[t][1] - truth[t - 1][1]);
          }
          const nv = iv + (-(0.045 + bias) * ix - 0.02 * iv);
          ix = ix + nv + r.normal(0, s.err * 0.6);
          iv = nv;
          path.push([t, ix]);
          maxDiv = Math.max(maxDiv, Math.abs(ix - truth[t][1]));
        }
        p.line(path, { color: alpha(SERIES()[k % 8], .55), width: 1.5 });
      }

      if (H < 40) p.line([[H, -2.6], [H, 2.6]], { color: alpha(cssVar('--text-faint'), .6), width: 1.2, dash: [4, 3] });
      p.clip(false);

      p.legend([
        { label: 'true dynamics', color: cssVar('--s1') },
        { label: 'imagined rollouts', color: alpha(cssVar('--s2'), .6) },
      ], { pos: 'tr' });

      P.readout({
        'one-step error': fmt(s.err, 3),
        'horizon H': H,
        'worst divergence at H': fmt(maxDiv, 3),
        'growth': s.replan ? 'bounded — re-grounded every 5 steps' : 'compounds with H',
        'why H is short': 'Dreamer imagines ~15 steps, not 1000',
      });
    },
    caption: 'A world model lets an agent train on **imagined** rollouts instead of real interaction — which is the whole point, since real robot time is the scarce resource. But a model with even 2% one-step error diverges from reality as the horizon grows, and a policy optimized against a diverged model exploits the model\'s errors rather than solving the task. This is why Dreamer imagines short horizons (~15 steps) and bootstraps a learned value function beyond them, and why model-predictive control **replans from a fresh observation** every few steps. Turn on replanning and watch the divergence stop compounding.',
  });
};

/* ============================================================
   The robot data problem
   ============================================================ */

V['robot-data-scale'] = (host) => {
  panel(host, {
    title: 'The data problem, to scale',
    height: 310,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'robots', label: 'robots collecting data', min: 1, max: 10000, step: 1, value: 100 },
      { type: 'slider', key: 'hours', label: 'hours per robot per day', min: 1, max: 24, step: 1, value: 8 },
      { type: 'slider', key: 'years', label: 'years of collection', min: .25, max: 10, step: .25, value: 1 },
    ],
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;

      // Rough public figures, in "tokens-equivalent" only as an order-of-magnitude device.
      const rows = [
        { name: 'Web text (frontier LLM pretraining)', v: 15e12, unit: 'tokens', c: cssVar('--s1') },
        { name: 'LAION-5B (image–text pairs)', v: 5e9, unit: 'pairs', c: cssVar('--s4') },
        { name: 'YouTube video uploaded per year', v: 3e10, unit: 'seconds', c: cssVar('--s7') },
        { name: 'Open X-Embodiment (2023)', v: 1e6, unit: 'episodes', c: cssVar('--s3') },
        { name: 'DROID (2024)', v: 76e3, unit: 'episodes', c: cssVar('--s5') },
        { name: 'A typical single-task paper', v: 100, unit: 'episodes', c: cssVar('--s6') },
      ];

      // user-configured collection effort, at ~120 episodes per robot-hour
      const episodes = Math.round(s.robots) * s.hours * 365 * s.years * 120;
      rows.push({ name: 'Your fleet (configured below)', v: episodes, unit: 'episodes', c: cssVar('--s2'), hi: true });

      const maxLog = Math.log10(15e12);
      const x0 = 8, labelW = Math.min(232, p.w * 0.46);
      const barX = x0 + labelW;
      const barW = p.w - barX - 92;
      const rowH = Math.min(30, (p.h - 44) / rows.length);

      g.font = '11px -apple-system, sans-serif';
      g.textBaseline = 'middle';
      rows.forEach((row, i) => {
        const y = 22 + i * rowH;
        g.fillStyle = row.hi ? cssVar('--text') : cssVar('--text-dim');
        g.textAlign = 'left';
        const label = row.name.length > 38 ? row.name.slice(0, 36) + '…' : row.name;
        g.fillText(label, x0, y);

        const frac = Math.max(0, Math.log10(Math.max(row.v, 1)) / maxLog);
        g.fillStyle = alpha(row.c, row.hi ? .95 : .6);
        g.fillRect(barX, y - rowH * .30, Math.max(barW * frac, 2), rowH * .52);
        if (row.hi) {
          g.strokeStyle = cssVar('--s2'); g.lineWidth = 1.5;
          g.strokeRect(barX, y - rowH * .30, Math.max(barW * frac, 2), rowH * .52);
        }

        g.fillStyle = row.hi ? cssVar('--s2') : cssVar('--text-faint');
        g.font = '10.5px ui-monospace, monospace';
        g.textAlign = 'left';
        g.fillText(fmtBig(row.v), barX + barW + 8, y);
        g.font = '11px -apple-system, sans-serif';
      });

      g.fillStyle = cssVar('--text-faint');
      g.font = '10px -apple-system, sans-serif';
      g.textAlign = 'left';
      g.fillText('log₁₀ scale — each bar length is orders of magnitude, not a ratio', x0, p.h - 8);

      const yearsToWeb = 15e12 / Math.max(episodes * 500, 1);
      P.readout({
        'episodes collected': fmtBig(episodes),
        'robot-hours': fmtBig(Math.round(s.robots) * s.hours * 365 * s.years),
        'vs Open X-Embodiment': fmt(episodes / 1e6, 2) + '×',
        'the gap': 'robot data is 6–8 orders of magnitude behind web text, and cannot be scraped',
      });
    },
    caption: 'Language models are trained on ~10¹³ tokens scraped for free. **Robot data has to be physically produced**, one episode at a time, on hardware that breaks. Open X-Embodiment pooled 22 embodiments from 34 labs and reached ~1M episodes — still six orders of magnitude below web text. Set the sliders to 10,000 robots running 24 hours a day for a decade and see how close you get. This single constraint explains most of the field: why VLAs bootstrap from web-pretrained vision-language models, why simulation and video pretraining matter, and why cross-embodiment transfer is treated as essential rather than optional.',
  });

  function fmtBig(x) {
    if (x >= 1e12) return (x / 1e12).toFixed(1) + 'T';
    if (x >= 1e9) return (x / 1e9).toFixed(1) + 'B';
    if (x >= 1e6) return (x / 1e6).toFixed(1) + 'M';
    if (x >= 1e3) return (x / 1e3).toFixed(1) + 'K';
    return String(Math.round(x));
  }
};

/* ============================================================
   VLA architecture
   ============================================================ */

V['vla-architecture'] = (host) => {
  panel(host, {
    title: 'How a vision-language-action model is put together',
    height: 300,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      {
        type: 'select', key: 'kind', label: 'action representation', value: 'flow',
        options: [
          { value: 'tokens', label: 'discretized action tokens (RT-2, OpenVLA)' },
          { value: 'flow', label: 'flow-matching action expert (π₀)' },
          { value: 'diffusion', label: 'diffusion head (Octo, RDT)' },
        ],
      },
      { type: 'check', key: 'cotrain', label: 'co-train on web vision-language data', value: true },
    ],
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const box = (x, y, w, h, label, col, sub) => {
        g.fillStyle = alpha(col, .18);
        g.strokeStyle = col; g.lineWidth = 1.6;
        g.fillRect(x, y, w, h); g.strokeRect(x, y, w, h);
        g.fillStyle = cssVar('--text');
        g.font = '600 11px -apple-system, sans-serif';
        g.textAlign = 'center'; g.textBaseline = 'middle';
        g.fillText(label, x + w / 2, y + h / 2 - (sub ? 7 : 0));
        if (sub) {
          g.fillStyle = cssVar('--text-faint');
          g.font = '9.5px -apple-system, sans-serif';
          g.fillText(sub, x + w / 2, y + h / 2 + 8);
        }
      };
      const arrow = (x1, y1, x2, y2, col) => {
        g.strokeStyle = col || cssVar('--text-faint'); g.lineWidth = 1.5;
        g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
        const a = Math.atan2(y2 - y1, x2 - x1);
        g.fillStyle = col || cssVar('--text-faint');
        g.beginPath();
        g.moveTo(x2, y2);
        g.lineTo(x2 - 6 * Math.cos(a - .4), y2 - 6 * Math.sin(a - .4));
        g.lineTo(x2 - 6 * Math.cos(a + .4), y2 - 6 * Math.sin(a + .4));
        g.fill();
      };

      const W = p.w, c1 = cssVar('--s1'), c2 = cssVar('--s2'), c3 = cssVar('--s3'), c4 = cssVar('--s4');
      const bw = Math.min(120, (W - 60) / 4);

      box(14, 28, bw, 34, 'camera(s)', c2, 'wrist + scene');
      box(14, 74, bw, 34, 'instruction', c1, '"put the mug away"');
      box(14, 120, bw, 34, 'proprioception', c4, 'joint angles');

      const mid = 20 + bw + 24;
      box(mid, 60, bw + 20, 62, 'pretrained VLM', c3,
          s.cotrain ? 'co-trained on web data' : 'robot data only');

      const rx = mid + bw + 56;
      const headLabel = {
        tokens: ['action tokens', 'binned, emitted like text'],
        flow: ['action expert', 'flow matching, 50 Hz'],
        diffusion: ['diffusion head', 'iterative denoising'],
      }[s.kind];
      box(rx, 60, bw + 20, 62, headLabel[0], c2, headLabel[1]);

      const ax = rx + bw + 56;
      if (ax + 70 < W) box(ax, 70, Math.min(80, W - ax - 10), 42, 'actions', cssVar('--ok'), 'chunk of k steps');

      for (const y of [45, 91, 137]) arrow(14 + bw + 4, y, mid - 4, 91);
      arrow(mid + bw + 20 + 4, 91, rx - 4, 91, c3);
      if (ax + 70 < W) arrow(rx + bw + 20 + 4, 91, ax - 4, 91, c2);

      const notes = {
        tokens: ['Discretize each action dimension into bins and emit them as ordinary tokens, reusing the language model head unchanged.',
                 'Simplest possible integration — no new architecture. But autoregressive token decoding is slow, and binning limits precision, so control frequency suffers.'],
        flow: ['A separate small "action expert" attached to the VLM generates continuous action chunks by flow matching, conditioned on the VLM\'s features.',
               'Continuous, high-frequency (50 Hz), and multimodal. One forward pass produces a whole chunk instead of decoding tokens one at a time. This is the π₀ design.'],
        diffusion: ['A diffusion head denoises a chunk of continuous actions conditioned on the VLM features.',
                    'Same motivation as flow matching — represent multimodal action distributions — but needs more denoising steps at inference, which costs control frequency.'],
      }[s.kind];

      g.fillStyle = cssVar('--text-dim');
      g.font = '11.5px -apple-system, sans-serif';
      g.textAlign = 'left';
      wrapText(g, notes[0], 14, p.h - 56, W - 28, 15);
      g.fillStyle = cssVar('--text-faint');
      wrapText(g, notes[1], 14, p.h - 26, W - 28, 15);

      P.readout({
        'why start from a VLM': s.cotrain
          ? 'web pretraining supplies semantics and generalization robot data cannot'
          : 'robot-only training does not generalize to unseen objects or phrasings',
        'control frequency': s.kind === 'tokens' ? '~3–10 Hz' : s.kind === 'flow' ? '~50 Hz' : '~10–20 Hz',
      });
    },
    caption: 'Every VLA makes the same bet: **a model that already understands images and language needs far less robot data to learn to act**. The open design question is the action head. Emitting discretized action tokens is the simplest integration and the slowest. Attaching a continuous action expert trained by flow matching — π₀\'s approach — gives smooth, multimodal, 50 Hz control at the cost of a bespoke architecture. Turn off web co-training and the model still learns the demonstrated tasks; it just stops generalizing to objects and instructions it has not seen.',
  });

  function wrapText(g, text, x, y, maxW, lh) {
    const words = text.split(' ');
    let line = '', yy = y;
    for (const w of words) {
      const t = line + w + ' ';
      if (g.measureText(t).width > maxW && line) { g.fillText(line, x, yy); line = w + ' '; yy += lh; }
      else line = t;
    }
    g.fillText(line, x, yy);
  }
};

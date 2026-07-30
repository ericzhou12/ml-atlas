/* ============================================================
   viz/rl.js — reinforcement learning figures.
   Real value iteration / Q-learning / bandits, run in-browser.
   ============================================================ */

import { panel } from '../ui.js';
import { cssVar, alpha, mix, seqMap, divergeMap, LA, rng, fmt, SERIES } from '../plot.js';

const V = {};
export default V;

/* ---------- shared gridworld ---------- */

const GW = {
  W: 7, H: 5,
  walls: new Set(['2,1', '2,2', '2,3', '4,1', '4,3']),
  goal: '6,2', trap: '5,0',
};
const key = (x, y) => `${x},${y}`;
const inBounds = (x, y) => x >= 0 && x < GW.W && y >= 0 && y < GW.H;
const isWall = (x, y) => GW.walls.has(key(x, y));
const isTerm = (x, y) => key(x, y) === GW.goal || key(x, y) === GW.trap;
const rewardOf = (x, y) => (key(x, y) === GW.goal ? 1 : key(x, y) === GW.trap ? -1 : 0);
const ACTIONS = [[0, 1], [1, 0], [0, -1], [-1, 0]];
const ARROW = ['↑', '→', '↓', '←'];

function stepEnv(x, y, a, slip, r) {
  let ai = a;
  if (r() < slip) ai = (a + (r() < .5 ? 1 : 3)) % 4;   // slip sideways
  const nx = x + ACTIONS[ai][0], ny = y + ACTIONS[ai][1];
  if (!inBounds(nx, ny) || isWall(nx, ny)) return [x, y];
  return [nx, ny];
}

function drawGrid(p, g, opts) {
  const { cell, x0, y0 } = opts;
  for (let y = 0; y < GW.H; y++) {
    for (let x = 0; x < GW.W; x++) {
      const X = x0 + x * cell, Y = y0 + (GW.H - 1 - y) * cell;
      let fill = cssVar('--bg-inset');
      if (isWall(x, y)) fill = cssVar('--border');
      else if (opts.color) fill = opts.color(x, y);
      g.fillStyle = fill;
      g.fillRect(X, Y, cell - 1.5, cell - 1.5);
      if (key(x, y) === GW.goal) {
        g.fillStyle = cssVar('--ok');
        g.font = '600 13px -apple-system, sans-serif';
        g.textAlign = 'center'; g.textBaseline = 'middle';
        g.fillText('+1', X + cell / 2, Y + cell / 2);
      } else if (key(x, y) === GW.trap) {
        g.fillStyle = cssVar('--danger');
        g.font = '600 13px -apple-system, sans-serif';
        g.textAlign = 'center'; g.textBaseline = 'middle';
        g.fillText('−1', X + cell / 2, Y + cell / 2);
      } else if (!isWall(x, y) && opts.label) {
        const t = opts.label(x, y);
        if (t) {
          g.fillStyle = opts.labelColor ? opts.labelColor(x, y) : cssVar('--text-dim');
          g.font = opts.labelFont || '10px ui-monospace, monospace';
          g.textAlign = 'center'; g.textBaseline = 'middle';
          g.fillText(t, X + cell / 2, Y + cell / 2);
        }
      }
    }
  }
}

/* ---------- value iteration ---------- */

V['value-iteration'] = (host) => {
  let Vs = {}, iter = 0, delta = 0;
  const reset = () => {
    Vs = {};
    for (let y = 0; y < GW.H; y++) for (let x = 0; x < GW.W; x++) Vs[key(x, y)] = 0;
    iter = 0; delta = 0;
  };
  reset();

  const bellmanQ = (x, y, a, gamma, slip, stepCost) => {
    // expected value of taking action a
    let q = 0;
    const outcomes = [[a, 1 - slip], [(a + 1) % 4, slip / 2], [(a + 3) % 4, slip / 2]];
    for (const [ai, pr] of outcomes) {
      const nx = x + ACTIONS[ai][0], ny = y + ACTIONS[ai][1];
      const ok = inBounds(nx, ny) && !isWall(nx, ny);
      const tx = ok ? nx : x, ty = ok ? ny : y;
      q += pr * (rewardOf(tx, ty) + stepCost + (isTerm(tx, ty) ? 0 : gamma * Vs[key(tx, ty)]));
    }
    return q;
  };

  panel(host, {
    title: 'Value iteration: the Bellman equation, applied until nothing changes',
    height: 300,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'gamma', label: 'discount γ', min: .5, max: .999, step: .001, value: .95, fmt: (v) => v.toFixed(3) },
      { type: 'slider', key: 'slip', label: 'slip probability', min: 0, max: .6, step: .02, value: .2 },
      { type: 'slider', key: 'stepCost', label: 'living reward', min: -.2, max: .05, step: .005, value: -.02, fmt: (v) => v.toFixed(3) },
      { type: 'check', key: 'policy', label: 'show greedy policy', value: true },
      { type: 'play' },
      { type: 'button', label: 'one sweep', onClick: (s) => sweep(s) },
      { type: 'button', label: '↺ reset', onClick: reset },
    ],
    animate(s, P) { sweep(s); if (delta < 1e-6) P.play(false); },
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const cell = Math.min((p.w - 40) / GW.W, (p.h - 40) / GW.H);
      const x0 = (p.w - cell * GW.W) / 2, y0 = 14;
      let lo = Infinity, hi = -Infinity;
      for (const k in Vs) { lo = Math.min(lo, Vs[k]); hi = Math.max(hi, Vs[k]); }
      const rng2 = Math.max(hi - lo, 1e-6);
      drawGrid(p, g, {
        cell, x0, y0,
        color: (x, y) => mix(cssVar('--bg-inset'), Vs[key(x, y)] >= 0 ? cssVar('--s3') : cssVar('--s2'), Math.min(1, Math.abs(Vs[key(x, y)]) / Math.max(Math.abs(hi), Math.abs(lo), .2) * .75)),
        label: (x, y) => {
          if (s.policy && !isTerm(x, y)) {
            let bi = 0, bv = -Infinity;
            for (let a = 0; a < 4; a++) { const q = bellmanQ(x, y, a, s.gamma, s.slip, s.stepCost); if (q > bv) { bv = q; bi = a; } }
            return ARROW[bi] + '\n';
          }
          return Vs[key(x, y)].toFixed(2);
        },
        labelFont: s.policy ? '15px -apple-system, sans-serif' : '10px ui-monospace, monospace',
      });
      // values under arrows
      if (s.policy) {
        g.font = '9px ui-monospace, monospace';
        g.fillStyle = cssVar('--text-faint');
        g.textAlign = 'center'; g.textBaseline = 'middle';
        for (let y = 0; y < GW.H; y++) for (let x = 0; x < GW.W; x++) {
          if (isWall(x, y) || isTerm(x, y)) continue;
          g.fillText(Vs[key(x, y)].toFixed(2), x0 + x * cell + cell / 2, y0 + (GW.H - 1 - y) * cell + cell * .76);
        }
      }
      P.readout({
        'sweeps': iter,
        'max change (Bellman residual)': delta.toExponential(2),
        'status': delta < 1e-6 ? 'converged — this is V* ✓' : 'iterating',
        'γ meaning': `a reward n steps away is worth γⁿ = ${fmt(s.gamma ** 10, 3)} at 10 steps`,
      });
    },
    caption: 'Each sweep replaces every state\'s value with $\\max_a \\mathbb E[r + \\gamma V(s\')]$ — "how good is this square, assuming I act well from here." Watch value **propagate outward from the goal**, one square per sweep: that is credit assignment happening. Lower γ and distant rewards stop mattering, so the policy near the far wall becomes indifferent. Raise the slip probability and the optimal policy starts avoiding tiles next to the −1 trap.',
  });

  function sweep(s) {
    let d = 0;
    const nV = {};
    for (let y = 0; y < GW.H; y++) for (let x = 0; x < GW.W; x++) {
      const k = key(x, y);
      if (isWall(x, y) || isTerm(x, y)) { nV[k] = rewardOf(x, y) && isTerm(x, y) ? 0 : 0; continue; }
      let best = -Infinity;
      for (let a = 0; a < 4; a++) best = Math.max(best, bellmanQ(x, y, a, s.gamma, s.slip, s.stepCost));
      nV[k] = best;
      d = Math.max(d, Math.abs(best - Vs[k]));
    }
    Vs = nV; iter++; delta = d;
  }
};

/* ---------- Q-learning ---------- */

V['q-learning'] = (host) => {
  let Q = {}, agent = [0, 0], eps = 0, episodes = 0, totalR = 0, rHist = [], stepsIn = 0;
  const reset = () => {
    Q = {};
    for (let y = 0; y < GW.H; y++) for (let x = 0; x < GW.W; x++) Q[key(x, y)] = [0, 0, 0, 0];
    agent = [0, 0]; episodes = 0; totalR = 0; rHist = []; stepsIn = 0;
  };
  reset();
  const r = rng(77);

  panel(host, {
    title: 'Q-learning: no model, just experience',
    height: 320,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'slider', key: 'alpha', label: 'learning rate α', min: .01, max: 1, step: .01, value: .3 },
      { type: 'slider', key: 'gamma', label: 'discount γ', min: .5, max: .999, step: .001, value: .95, fmt: (v) => v.toFixed(3) },
      { type: 'slider', key: 'eps', label: 'exploration ε', min: 0, max: 1, step: .01, value: .3 },
      { type: 'slider', key: 'slip', label: 'slip probability', min: 0, max: .5, step: .02, value: .1 },
      { type: 'play' },
      { type: 'button', label: 'run 300 steps', onClick: (s) => { for (let i = 0; i < 300; i++) tick(s); } },
      { type: 'button', label: '↺ forget everything', onClick: reset },
    ],
    animate(s) { for (let i = 0; i < 4; i++) tick(s); },
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const cell = Math.min((p.w - 40) / GW.W, (p.h - 60) / GW.H);
      const x0 = (p.w - cell * GW.W) / 2, y0 = 14;
      let hi = .01;
      for (const k in Q) hi = Math.max(hi, Math.max(...Q[k]));
      drawGrid(p, g, {
        cell, x0, y0,
        color: (x, y) => {
          const v = Math.max(...Q[key(x, y)]);
          return mix(cssVar('--bg-inset'), v >= 0 ? cssVar('--s3') : cssVar('--s2'), Math.min(1, Math.abs(v) / hi * .7));
        },
        label: (x, y) => (isTerm(x, y) ? '' : ARROW[argmax(Q[key(x, y)])]),
        labelFont: '15px -apple-system, sans-serif',
      });
      // agent
      g.fillStyle = cssVar('--s5');
      g.beginPath();
      g.arc(x0 + agent[0] * cell + cell / 2, y0 + (GW.H - 1 - agent[1]) * cell + cell / 2, cell * .22, 0, 6.2832);
      g.fill();
      g.strokeStyle = cssVar('--bg'); g.lineWidth = 2; g.stroke();
      // reward curve
      if (rHist.length > 2) {
        const W = p.w - 60, X = 30, Y = y0 + GW.H * cell + 8, H = p.h - Y - 6;
        g.strokeStyle = cssVar('--s1'); g.lineWidth = 1.6;
        g.beginPath();
        const mn = Math.min(...rHist), mxv = Math.max(...rHist);
        rHist.forEach((v, i) => {
          const px = X + i / (rHist.length - 1) * W;
          const py = Y + H - (v - mn) / Math.max(mxv - mn, 1e-6) * H;
          i ? g.lineTo(px, py) : g.moveTo(px, py);
        });
        g.stroke();
        g.fillStyle = cssVar('--text-faint');
        g.font = '9px -apple-system, sans-serif';
        g.textAlign = 'left';
        g.fillText('episode return', X, Y + 8);
      }
      P.readout({
        'episodes': episodes,
        'ε': fmt(s.eps, 2),
        'last 10 returns (avg)': rHist.length ? fmt(rHist.slice(-10).reduce((a, b) => a + b, 0) / Math.min(rHist.length, 10), 3) : '—',
        'update': 'Q(s,a) ← Q(s,a) + α[r + γ max Q(s′,·) − Q(s,a)]',
      });
    },
    caption: 'The agent has no map and no idea what the rewards are — it finds out by bumping into things. Set ε to 0 immediately and it usually gets stuck exploiting the first mediocre path it found; keep ε high and it never commits. **That is the exploration/exploitation dilemma in its simplest form.** Note Q-learning is *off-policy*: it explores randomly but learns the value of acting greedily, which is why the arrows converge to the optimal policy even while the agent is still wandering.',
  });

  function argmax(a) { let bi = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i; return bi; }
  function tick(s) {
    const [x, y] = agent;
    const a = r() < s.eps ? Math.floor(r() * 4) : argmax(Q[key(x, y)]);
    const [nx, ny] = stepEnv(x, y, a, s.slip, r);
    const rew = rewardOf(nx, ny) - .02;
    totalR += rew;
    stepsIn++;
    const target = rew + (isTerm(nx, ny) ? 0 : s.gamma * Math.max(...Q[key(nx, ny)]));
    Q[key(x, y)][a] += s.alpha * (target - Q[key(x, y)][a]);
    if (isTerm(nx, ny) || stepsIn > 200) {
      episodes++;
      rHist.push(totalR);
      if (rHist.length > 200) rHist.shift();
      totalR = 0; stepsIn = 0;
      agent = [0, 0];
    } else agent = [nx, ny];
  }
};

/* ---------- multi-armed bandit ---------- */

V['bandit'] = (host) => {
  const K = 6;
  const trueP = [.28, .45, .62, .35, .55, .20];
  let state = null;
  const reset = (algo) => {
    state = {
      n: new Array(K).fill(0), s: new Array(K).fill(0),
      t: 0, regret: 0, regretHist: [], algo,
    };
  };
  reset('eps');

  panel(host, {
    title: 'Exploration vs exploitation: the multi-armed bandit',
    height: 300,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'select', key: 'algo', label: 'strategy', value: 'eps', options: [{ value: 'greedy', label: 'pure greedy' }, { value: 'eps', label: 'ε-greedy' }, { value: 'ucb', label: 'UCB1' }, { value: 'thompson', label: 'Thompson sampling' }], onChange: (v) => reset(v) },
      { type: 'slider', key: 'eps', label: 'ε', min: 0, max: .5, step: .01, value: .1 },
      { type: 'play' },
      { type: 'button', label: 'pull ×200', onClick: (s) => { for (let i = 0; i < 200; i++) pull(s); } },
      { type: 'button', label: '↺ reset', onClick: (s) => reset(s.algo) },
    ],
    animate(s) { for (let i = 0; i < 3; i++) pull(s); },
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const bw = (p.w - 60) / K;
      const baseY = p.h - 70;
      for (let k = 0; k < K; k++) {
        const X = 30 + k * bw;
        const est = state.n[k] ? state.s[k] / state.n[k] : 0;
        const conf = state.n[k] ? Math.sqrt(2 * Math.log(Math.max(state.t, 2)) / state.n[k]) : 2;
        // true rate (outline) and estimate (filled)
        g.strokeStyle = alpha(cssVar('--text-faint'), .8); g.lineWidth = 1.5;
        g.strokeRect(X, baseY - trueP[k] * (baseY - 30), bw - 10, trueP[k] * (baseY - 30));
        g.fillStyle = alpha(cssVar('--s1'), .75);
        g.fillRect(X, baseY - est * (baseY - 30), bw - 10, est * (baseY - 30));
        // confidence whisker
        if (state.n[k] && s.algo === 'ucb') {
          const top = Math.min(1, est + conf);
          g.strokeStyle = cssVar('--s2'); g.lineWidth = 1.5;
          g.beginPath();
          g.moveTo(X + (bw - 10) / 2, baseY - est * (baseY - 30));
          g.lineTo(X + (bw - 10) / 2, baseY - top * (baseY - 30));
          g.stroke();
        }
        g.fillStyle = cssVar('--text-dim');
        g.font = '10px ui-monospace, monospace';
        g.textAlign = 'center';
        g.fillText(`${state.n[k]}`, X + (bw - 10) / 2, baseY + 14);
        g.fillStyle = cssVar('--text-faint');
        g.fillText(trueP[k].toFixed(2), X + (bw - 10) / 2, baseY + 28);
      }
      g.fillStyle = cssVar('--text-faint');
      g.font = '10px -apple-system, sans-serif';
      g.textAlign = 'left';
      g.fillText('outline = true rate · fill = estimate · numbers = pulls / true rate', 30, 18);
      // regret curve
      if (state.regretHist.length > 2) {
        const W = 150, X = p.w - W - 20, Y = 26, H = 50;
        g.strokeStyle = cssVar('--border'); g.strokeRect(X, Y, W, H);
        g.beginPath();
        const mx = Math.max(...state.regretHist, 1);
        state.regretHist.forEach((v, i) => {
          const px = X + i / (state.regretHist.length - 1) * W;
          const py = Y + H - v / mx * H;
          i ? g.lineTo(px, py) : g.moveTo(px, py);
        });
        g.strokeStyle = cssVar('--s2'); g.lineWidth = 1.6; g.stroke();
        g.fillStyle = cssVar('--text-faint'); g.font = '9px -apple-system, sans-serif';
        g.fillText('cumulative regret', X + 3, Y - 4);
      }
      const best = Math.max(...trueP);
      P.readout({
        'pulls': state.t,
        'cumulative regret': fmt(state.regret, 2),
        'regret per pull': state.t ? fmt(state.regret / state.t, 4) : '—',
        'best arm found': state.n.indexOf(Math.max(...state.n)) === trueP.indexOf(best) ? 'yes ✓' : 'not yet',
      });
    },
    caption: '**Regret** is what you lost by not always pulling the best arm. Pure greedy locks onto whichever arm got lucky first and can be wrong forever (regret grows linearly). ε-greedy fixes that but keeps wasting ε of every pull on known-bad arms. **UCB** is optimistic in proportion to uncertainty — arms it knows little about get a bonus, so exploration self-extinguishes. **Thompson sampling** just samples from the posterior and acts greedily on the sample. Both achieve logarithmic regret; watch the curve flatten.',
  });

  function pull(s) {
    const r = Math.random;
    let k;
    if (state.algo === 'greedy') {
      k = argmaxArr(state.n.map((n, i) => (n ? state.s[i] / n : 1)));
    } else if (state.algo === 'eps') {
      k = r() < s.eps ? Math.floor(r() * K) : argmaxArr(state.n.map((n, i) => (n ? state.s[i] / n : 1)));
    } else if (state.algo === 'ucb') {
      k = argmaxArr(state.n.map((n, i) => (n ? state.s[i] / n + Math.sqrt(2 * Math.log(Math.max(state.t, 2)) / n) : 1e6)));
    } else {
      k = argmaxArr(state.n.map((n, i) => betaSample(state.s[i] + 1, n - state.s[i] + 1)));
    }
    const win = r() < trueP[k] ? 1 : 0;
    state.n[k]++; state.s[k] += win;
    state.t++;
    state.regret += Math.max(...trueP) - trueP[k];
    if (state.t % 3 === 0) { state.regretHist.push(state.regret); if (state.regretHist.length > 300) state.regretHist.shift(); }
  }
  function argmaxArr(a) { let bi = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i; return bi; }
  function betaSample(a, b) {
    const g1 = gammaSample(a), g2 = gammaSample(b);
    return g1 / (g1 + g2);
  }
  function gammaSample(k) {
    // Marsaglia-Tsang
    if (k < 1) return gammaSample(k + 1) * Math.pow(Math.random(), 1 / k);
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (let i = 0; i < 200; i++) {
      let x, v;
      do { x = normSample(); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const u = Math.random();
      if (u < 1 - .0331 * x ** 4) return d * v;
      if (Math.log(u) < .5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
    return d;
  }
  function normSample() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
};

/* ---------- policy gradient ---------- */

V['policy-gradient'] = (host) => {
  let theta = 0, hist = [], step = 0, baseline = 0;
  const reset = () => { theta = -1.5; hist = []; step = 0; baseline = 0; };
  reset();

  panel(host, {
    title: 'REINFORCE: push up what worked, push down what did not',
    height: 300,
    plot: { xlim: [-4, 4], ylim: [0, 1.1], xlabel: 'action a', ylabel: 'π(a) and reward' },
    controls: [
      { type: 'slider', key: 'lr', label: 'learning rate', min: .001, max: .3, step: .001, value: .03, fmt: (v) => v.toFixed(3) },
      { type: 'slider', key: 'sigma', label: 'policy std σ (exploration)', min: .1, max: 2, step: .05, value: .8 },
      { type: 'check', key: 'base', label: 'subtract a baseline (variance reduction)', value: false },
      { type: 'slider', key: 'batch', label: 'episodes per update', min: 1, max: 64, step: 1, value: 8 },
      { type: 'play' },
      { type: 'button', label: '↺ reset', onClick: reset },
    ],
    animate(s) {
      const r = rng(2000 + step);
      let g = 0, mean = 0;
      const B = Math.round(s.batch);
      const samples = [];
      for (let i = 0; i < B; i++) {
        const a = theta + r.normal(0, s.sigma);
        const rew = reward(a);
        samples.push([a, rew]);
        mean += rew / B;
      }
      const b = s.base ? baseline : 0;
      for (const [a, rew] of samples) {
        // ∇ log π(a|θ) for Gaussian policy = (a-θ)/σ²
        g += (rew - b) * (a - theta) / (s.sigma ** 2) / B;
      }
      baseline = .9 * baseline + .1 * mean;
      theta += s.lr * g;
      theta = Math.max(-4, Math.min(4, theta));
      step++;
      hist.push(mean);
      if (hist.length > 300) hist.shift();
    },
    draw(p, s, P) {
      p.clear().axes();
      p.clip();
      p.fn((a) => reward(a), { color: cssVar('--s3'), width: 2.4 });
      const pol = [];
      for (let a = -4; a <= 4; a += .02) pol.push([a, Math.exp(-((a - theta) ** 2) / (2 * s.sigma ** 2))]);
      p.area(pol, { color: alpha(cssVar('--s1'), .2) });
      p.line(pol, { color: cssVar('--s1'), width: 2.4 });
      p.line([[theta, 0], [theta, 1.1]], { color: cssVar('--s2'), width: 1.6, dash: [4, 3] });
      p.clip(false);
      p.legend([{ label: 'reward R(a)', color: cssVar('--s3') }, { label: 'policy π(a|θ)', color: cssVar('--s1') }], { pos: 'tl' });
      const varEst = hist.length > 5 ? variance(hist.slice(-40)) : 0;
      P.readout({
        'θ': fmt(theta, 3),
        'updates': step,
        'mean reward': hist.length ? fmt(hist[hist.length - 1], 4) : '—',
        'reward variance (last 40)': fmt(varEst, 5),
        'baseline': s.base ? fmt(baseline, 3) : 'off',
      });
    },
    caption: 'The policy is a Gaussian over actions; the update nudges $\\theta$ toward actions that returned above-average reward. Two things to feel here. **(1)** With batch 1 and no baseline the updates are extremely noisy — that variance is the central practical problem of policy gradients. Turn on the baseline and watch the trajectory smooth out without changing what it converges to. **(2)** With small σ the policy stops exploring and can park on the local bump instead of finding the taller one.',
  });

  function reward(a) { return Math.exp(-((a - 1.7) ** 2) / .6) + .55 * Math.exp(-((a + 1.6) ** 2) / .5); }
  function variance(v) { const m = v.reduce((a, b) => a + b, 0) / v.length; return v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length; }
};

/* ---------- PPO clipping ---------- */

V['ppo-clip'] = (host) => {
  panel(host, {
    title: 'PPO: the clipped objective, and what it prevents',
    height: 290,
    plot: { xlim: [0, 2.5], ylim: [-2.5, 2.5], xlabel: 'probability ratio  r = π_new / π_old', ylabel: 'objective' },
    controls: [
      { type: 'slider', key: 'eps', label: 'clip range ε', min: .05, max: .6, step: .01, value: .2 },
      { type: 'slider', key: 'A', label: 'advantage Â', min: -2, max: 2, step: .05, value: 1 },
      { type: 'check', key: 'kl', label: 'also show KL penalty version', value: false },
    ],
    draw(p, s, P) {
      p.clear().axes();
      p.clip();
      const A = s.A, e = s.eps;
      const unclipped = (r) => r * A;
      const clipped = (r) => Math.min(r * A, Math.max(1 - e, Math.min(1 + e, r)) * A);
      p.fn(unclipped, { color: alpha(cssVar('--s2'), .7), width: 1.8, dash: [5, 3] });
      p.fn(clipped, { color: cssVar('--s1'), width: 3 });
      if (s.kl) p.fn((r) => r * A - 2 * (r * Math.log(Math.max(r, 1e-6)) - r + 1), { color: cssVar('--s4'), width: 2, dash: [3, 3] });
      p.line([[1 - e, -2.5], [1 - e, 2.5]], { color: alpha(cssVar('--text-faint'), .5), width: 1, dash: [3, 3] });
      p.line([[1 + e, -2.5], [1 + e, 2.5]], { color: alpha(cssVar('--text-faint'), .5), width: 1, dash: [3, 3] });
      p.line([[1, -2.5], [1, 2.5]], { color: alpha(cssVar('--text'), .35), width: 1 });
      p.clip(false);
      p.legend([
        { label: 'r·Â  (vanilla policy gradient)', color: cssVar('--s2'), dash: true },
        { label: 'PPO clipped objective', color: cssVar('--s1') },
        ...(s.kl ? [{ label: 'KL-penalized', color: cssVar('--s4'), dash: true }] : []),
      ], { pos: A > 0 ? 'tl' : 'bl' });
      P.readout({
        'clip range': `[${fmt(1 - e, 2)}, ${fmt(1 + e, 2)}]`,
        'advantage sign': A > 0 ? 'positive — action was better than expected' : 'negative — worse than expected',
        'effect': A > 0
          ? 'gradient vanishes once r > 1+ε → no reward for moving further'
          : 'gradient vanishes once r < 1−ε → cannot crush the action to zero in one update',
        'why it matters': 'keeps the new policy close to the data-collecting policy → the samples stay valid',
      });
    },
    caption: 'Policy gradient methods reuse a batch of experience for several update steps, but the data was collected under the *old* policy. If the new policy drifts too far, the importance ratio $r$ blows up and the estimate becomes garbage. PPO simply **flattens the objective outside $[1-\\epsilon, 1+\\epsilon]$** so there is nothing to gain from drifting. Flip the advantage negative and notice the clip acts on the other side — a beautifully asymmetric, one-line fix that replaced the much heavier trust-region machinery of TRPO.',
  });
};

/* ---------- RLHF pipeline ---------- */

V['rlhf-pipeline'] = (host) => {
  panel(host, {
    title: 'From pretrained model to assistant',
    height: 300,
    plot: { xlim: [0, 1], ylim: [0, 1] },
    controls: [
      { type: 'select', key: 'stage', label: 'stage', value: '0', options: [{ value: '0', label: '1 · pretraining' }, { value: '1', label: '2 · supervised fine-tuning' }, { value: '2', label: '3 · reward model' }, { value: '3', label: '4 · RL (PPO / GRPO)' }, { value: '4', label: 'alternative · DPO' }] },
    ],
    draw(p, s, P) {
      p.clear();
      const g = p.ctx;
      const stages = [
        { n: 'Pretraining', d: 'next-token prediction on trillions of tokens of web text', data: 'raw text', obj: '−Σ log p(xₜ | x₍₁..ₜ₋₁₎)', out: 'a base model: knows a great deal, follows no instructions' },
        { n: 'SFT', d: 'fine-tune on curated (prompt, ideal response) pairs', data: '10k–1M demonstrations', obj: 'same cross-entropy, but only on the response tokens', out: 'follows instructions; quality capped by the demonstrations' },
        { n: 'Reward model', d: 'train a model to score responses from human preference comparisons', data: 'pairs (chosen, rejected)', obj: '−log σ(r(chosen) − r(rejected))', out: 'a learned proxy for "which answer do people prefer"' },
        { n: 'RL fine-tuning', d: 'optimize the policy against the reward model, with a KL leash to SFT', data: 'prompts only — responses are sampled', obj: 'E[r(y)] − β·KL(π ‖ π_SFT)', out: 'the assistant. Can beat the demonstrations it learned from' },
        { n: 'DPO', d: 'skip the reward model: optimize preferences directly on the policy', data: 'the same preference pairs', obj: '−log σ(β log π(y⁺)/π_ref(y⁺) − β log π(y⁻)/π_ref(y⁻))', out: 'similar result, far simpler pipeline — now the common default' },
      ];
      const k = +s.stage;
      // pipeline strip
      const bw = (p.w - 40) / 4, y = 34;
      for (let i = 0; i < 4; i++) {
        const X = 20 + i * bw;
        const on = i === k || (k === 4 && i >= 2);
        g.fillStyle = on ? alpha(cssVar('--s1'), .6) : alpha(cssVar('--s1'), .13);
        g.fillRect(X, y, bw - 10, 34);
        g.fillStyle = on ? cssVar('--text') : cssVar('--text-faint');
        g.font = (on ? '600 ' : '') + '11.5px -apple-system, sans-serif';
        g.textAlign = 'center'; g.textBaseline = 'middle';
        g.fillText(stages[i].n, X + (bw - 10) / 2, y + 17);
        if (i < 3) {
          g.strokeStyle = cssVar('--text-faint'); g.lineWidth = 1.4;
          g.beginPath(); g.moveTo(X + bw - 9, y + 17); g.lineTo(X + bw - 1, y + 17); g.stroke();
        }
      }
      if (k === 4) {
        g.strokeStyle = cssVar('--s3'); g.lineWidth = 2; g.setLineDash([4, 3]);
        g.beginPath();
        g.moveTo(20 + bw * 1.5, y + 38); g.lineTo(20 + bw * 1.5, y + 54);
        g.lineTo(20 + bw * 3.5, y + 54); g.lineTo(20 + bw * 3.5, y + 38);
        g.stroke(); g.setLineDash([]);
        g.fillStyle = cssVar('--s3'); g.font = '600 11px -apple-system, sans-serif';
        g.fillText('DPO — one step instead of two', 20 + bw * 2.5, y + 66);
      }
      const S = stages[k];
      const lines = [['what happens', S.d], ['data', S.data], ['objective', S.obj], ['result', S.out]];
      let Y = y + (k === 4 ? 90 : 66);
      g.textAlign = 'left';
      for (const [a, b] of lines) {
        g.fillStyle = cssVar('--text-faint');
        g.font = '10px -apple-system, sans-serif';
        g.fillText(a.toUpperCase(), 22, Y);
        g.fillStyle = cssVar('--text');
        g.font = (a === 'objective' ? '11.5px ui-monospace, monospace' : '12px -apple-system, sans-serif');
        wrap(g, b, 22, Y + 16, p.w - 44, 15);
        Y += 16 + 15 * Math.ceil(g.measureText(b).width / (p.w - 44)) + 12;
      }
      P.readout({ 'stage': S.n, 'compute share': k === 0 ? '>99% of total' : '<1% of total' });
    },
    caption: 'Almost all of the capability comes from stage 1; almost all of the *behavior* comes from stages 2–4. The KL penalty in stage 4 is doing something specific and important: the reward model is only a proxy, and a policy left unleashed will find its adversarial holes — literal reward hacking. Constraining the policy to stay near the SFT model keeps it inside the region where the proxy is still trustworthy.',
  });

  function wrap(g, text, x, y, maxW, lh) {
    const words = text.split(' ');
    let line = '', yy = y;
    for (const w of words) {
      const test = line + w + ' ';
      if (g.measureText(test).width > maxW && line) { g.fillText(line, x, yy); line = w + ' '; yy += lh; }
      else line = test;
    }
    g.fillText(line, x, yy);
  }
};

/* ============================================================
   viz/index.js — the visualization registry.
   Every lesson section of type {t:'viz', id} resolves through here.
   ============================================================ */

import foundations from './foundations.js';
import math from './math.js';
import classical from './classical.js';
import nn from './nn.js';
import transformers from './transformers.js';
import generative from './generative.js';
import rl from './rl.js';
import frontier from './frontier.js';
import embodied from './embodied.js';

export const VIZ = Object.assign({}, foundations, math, classical, nn, transformers, generative, rl, frontier, embodied);

export const VIZ_IDS = Object.keys(VIZ).sort();

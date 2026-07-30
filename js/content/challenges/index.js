/* ============================================================
   challenges/index.js — one file per track, merged into a single
   { lessonId: challenge } map that main.js renders under each lesson.
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

export const CHALLENGES = Object.assign(
  {}, math, classical, nn, llm, systems, generative, rl, vision, frontier, embodied, practice,
);

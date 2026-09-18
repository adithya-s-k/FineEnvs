#!/usr/bin/env node

// Generate synthetic Trackio-like CSV data with realistic ML curves.
// - Steps are simple integers (e.g., 1..N)
// - Metrics: epoch, train_accuracy, val_accuracy, train_loss, val_loss
// - W&B-like run names (e.g., pleasant-flower-1)
// - Deterministic with --seed
//
// Usage:
//   node app/scripts/generate-trackio-data.mjs \
//     --runs 3 \
//     --steps 10 \
//     --out app/src/content/assets/data/trackio_wandb_synth.csv \
//     [--seed 42] [--epoch-max 3.0] [--amount 1.0] [--start 1]
//
// To overwrite the demo file used by the embed:
//   node app/scripts/generate-trackio-data.mjs --runs 3 --steps 10 --out app/src/content/assets/data/trackio_wandb_demo.csv --seed 1337

import fs from 'node:fs/promises';
import path from 'node:path';

function parseArgs(argv){
  const args = { runs: 3, steps: 10, out: '', seed: undefined, epochMax: 3.0, amount: 1, start: 1 };
  for (let i = 2; i < argv.length; i++){
    const a = argv[i];
    if (a === '--runs' && argv[i+1]) { args.runs = Math.max(1, parseInt(argv[++i], 10) || 3); continue; }
    if (a === '--steps' && argv[i+1]) { args.steps = Math.max(2, parseInt(argv[++i], 10) || 10); continue; }
    if (a === '--out' && argv[i+1]) { args.out = argv[++i]; continue; }
    if (a === '--seed' && argv[i+1]) { args.seed = Number(argv[++i]); continue; }
    if (a === '--epoch-max' && argv[i+1]) { args.epochMax = Number(argv[++i]) || 3.0; continue; }
    if (a === '--amount' && argv[i+1]) { args.amount = Number(argv[++i]) || 1.0; continue; }
    if (a === '--start' && argv[i+1]) { args.start = parseInt(argv[++i], 10) || 1; continue; }
  }
  if (!args.out) {
    args.out = path.join('app', 'src', 'content', 'assets', 'data', 'trackio_wandb_synth.csv');
  }
  return args;
}

function mulberry32(seed){
  let t = seed >>> 0;
  return function(){
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function makeRng(seed){
  if (Number.isFinite(seed)) return mulberry32(seed);
  return Math.random;
}

function randn(rng){
  // Box-Muller transform
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function clamp(x, lo, hi){
  return Math.max(lo, Math.min(hi, x));
}

function logistic(t, k=6, x0=0.5){
  // 1 / (1 + e^{-k (t - x0)}) in [0,1]
  return 1 / (1 + Math.exp(-k * (t - x0)));
}

function expDecay(t, k=3){
  // (1 - e^{-k t}) in [0,1]
  return 1 - Math.exp(-k * t);
}

function pick(array, rng){
  return array[Math.floor(rng() * array.length) % array.length];
}

function buildRunNames(count, rng){
  const adjectives = [
    'pleasant','brisk','silent','ancient','bold','gentle','rapid','shy','curious','lively',
    'fearless','soothing','glossy','hidden','misty','bright','calm','keen','noble','swift'
  ];
  const nouns = [
    'flower','glade','sky','river','forest','ember','comet','meadow','harbor','dawn',
    'mountain','prairie','breeze','valley','lagoon','desert','monsoon','reef','thunder','willow'
  ];
  const names = new Set();
  let attempts = 0;
  while (names.size < count && attempts < count * 20){
    attempts++;
    const left = pick(adjectives, rng);
    const right = pick(nouns, rng);
    const idx = 1 + Math.floor(rng() * 9);
    names.add(`${left}-${right}-${idx}`);
  }
  return Array.from(names);
}

function formatLike(value, decimals){
  return Number.isFinite(decimals) && decimals >= 0 ? value.toFixed(decimals) : String(value);
}

async function main(){
  const args = parseArgs(process.argv);
  const rng = makeRng(args.seed);

  // Steps: integers from start .. start+steps-1
  const steps = Array.from({ length: args.steps }, (_, i) => args.start + i);
  const stepNorm = (i) => (i - steps[0]) / (steps[steps.length-1] - steps[0]);

  const runs = buildRunNames(args.runs, rng);

  // Per-run slight variations
  const runParams = runs.map((_r, idx) => {
    const r = rng();
    // Final accuracies
    const trainAccFinal = clamp(0.86 + (r - 0.5) * 0.12 * args.amount, 0.78, 0.97);
    const valAccFinal = clamp(trainAccFinal - (0.02 + rng() * 0.05), 0.70, 0.95);
    // Loss plateau
    const lossStart = 7.0 + (rng() - 0.5) * 0.10 * args.amount; // ~7.0 ±0.05
    const lossPlateau = 6.78 + (rng() - 0.5) * 0.04 * args.amount; // ~6.78 ±0.02
    const lossK = 2.0 + rng() * 1.5; // decay speed
    // Acc growth steepness and midpoint
    const kAcc = 4.5 + rng() * 3.0;
    const x0Acc = 0.35 + rng() * 0.25;
    return { trainAccFinal, valAccFinal, lossStart, lossPlateau, lossK, kAcc, x0Acc };
  });

  const lines = [];
  lines.push('run,step,metric,value,stderr');

  // EPOCH: linear 0..epochMax across steps
  for (let r = 0; r < runs.length; r++){
    const run = runs[r];
    for (let i = 0; i < steps.length; i++){
      const t = stepNorm(steps[i]);
      const epoch = args.epochMax * t;
      lines.push(`${run},${steps[i]},epoch,${formatLike(epoch, 2)},`);
    }
  }

  // TRAIN LOSS & VAL LOSS
  for (let r = 0; r < runs.length; r++){
    const run = runs[r];
    const p = runParams[r];
    let prevTrain = null;
    let prevVal = null;
    for (let i = 0; i < steps.length; i++){
      const t = stepNorm(steps[i]);
      const d = expDecay(t, p.lossK); // 0..1
      let trainLoss = p.lossStart - (p.lossStart - p.lossPlateau) * d;
      let valLoss = trainLoss + 0.02 + (rng() * 0.03);
      // Add mild noise
      trainLoss += randn(rng) * 0.01 * args.amount;
      valLoss += randn(rng) * 0.012 * args.amount;
      // Keep reasonable and mostly monotonic (small upward blips allowed)
      if (prevTrain != null) trainLoss = Math.min(prevTrain + 0.01, trainLoss);
      if (prevVal != null) valLoss = Math.min(prevVal + 0.012, valLoss);
      prevTrain = trainLoss; prevVal = valLoss;
      const stderrTrain = clamp(0.03 - 0.02 * t + Math.abs(randn(rng)) * 0.003, 0.006, 0.04);
      const stderrVal = clamp(0.035 - 0.022 * t + Math.abs(randn(rng)) * 0.003, 0.008, 0.045);
      lines.push(`${run},${steps[i]},train_loss,${formatLike(trainLoss, 3)},${formatLike(stderrTrain, 3)}`);
      lines.push(`${run},${steps[i]},val_loss,${formatLike(valLoss, 3)},${formatLike(stderrVal, 3)}`);
    }
  }

  // TRAIN ACCURACY & VAL ACCURACY (logistic)
  for (let r = 0; r < runs.length; r++){
    const run = runs[r];
    const p = runParams[r];
    for (let i = 0; i < steps.length; i++){
      const t = stepNorm(steps[i]);
      const accBase = logistic(t, p.kAcc, p.x0Acc);
      let trainAcc = clamp(0.55 + accBase * (p.trainAccFinal - 0.55), 0, 1);
      let valAcc = clamp(0.52 + accBase * (p.valAccFinal - 0.52), 0, 1);
      // Gentle noise
      trainAcc = clamp(trainAcc + randn(rng) * 0.005 * args.amount, 0, 1);
      valAcc = clamp(valAcc + randn(rng) * 0.006 * args.amount, 0, 1);
      const stderrTrain = clamp(0.02 - 0.011 * t + Math.abs(randn(rng)) * 0.002, 0.006, 0.03);
      const stderrVal = clamp(0.022 - 0.012 * t + Math.abs(randn(rng)) * 0.002, 0.007, 0.032);
      lines.push(`${run},${steps[i]},train_accuracy,${formatLike(trainAcc, 4)},${formatLike(stderrTrain, 3)}`);
      lines.push(`${run},${steps[i]},val_accuracy,${formatLike(valAcc, 4)},${formatLike(stderrVal, 3)}`);
    }
  }

  // Ensure directory exists
  await fs.mkdir(path.dirname(args.out), { recursive: true });
  await fs.writeFile(args.out, lines.join('\n') + '\n', 'utf8');
  const relOut = path.relative(process.cwd(), args.out);
  console.log(`Synthetic CSV generated: ${relOut}`);
}

main().catch(err => { console.error(err?.stack || String(err)); process.exit(1); });

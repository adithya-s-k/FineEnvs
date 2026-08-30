#!/usr/bin/env node

// Jitter Trackio CSV data with small, controlled noise.
// - Preserves comments (# ...) and blank lines
// - Leaves 'epoch' values unchanged
// - Adds mild noise to train/val accuracy (clamped to [0,1])
// - Adds mild noise to train/val loss (kept >= 0)
// - Keeps steps untouched
// Usage:
//   node app/scripts/jitter-trackio-data.mjs \
//     --in app/src/content/assets/data/trackio_wandb_demo.csv \
//     --out app/src/content/assets/data/trackio_wandb_demo.jitter.csv \
//     [--seed 42] [--amount 1.0] [--in-place]

import fs from 'node:fs/promises';
import path from 'node:path';

function parseArgs(argv){
  const args = { in: '', out: '', seed: undefined, amount: 1, inPlace: false };
  for (let i = 2; i < argv.length; i++){
    const a = argv[i];
    if (a === '--in' && argv[i+1]) { args.in = argv[++i]; continue; }
    if (a === '--out' && argv[i+1]) { args.out = argv[++i]; continue; }
    if (a === '--seed' && argv[i+1]) { args.seed = Number(argv[++i]); continue; }
    if (a === '--amount' && argv[i+1]) { args.amount = Number(argv[++i]) || 3; continue; }
    if (a === '--in-place') { args.inPlace = true; continue; }
  }
  if (!args.in) throw new Error('--in is required');
  if (args.inPlace) args.out = args.in;
  if (!args.out) {
    const { dir, name, ext } = path.parse(args.in);
    args.out = path.join(dir, `${name}.jitter${ext || '.csv'}`);
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

function jitterValue(metric, value, amount, rng){
  const m = metric.toLowerCase();
  if (m === 'epoch') return value; // keep as-is
  if (m.includes('accuracy')){
    const n = Math.max(-0.02 * amount, Math.min(0.02 * amount, randn(rng) * 0.01 * amount));
    return Math.max(0, Math.min(1, value + n));
  }
  if (m.includes('loss')){
    const n = Math.max(-0.03 * amount, Math.min(0.03 * amount, randn(rng) * 0.01 * amount));
    return Math.max(0, value + n);
  }
  // default: tiny noise
  const n = Math.max(-0.01 * amount, Math.min(0.01 * amount, randn(rng) * 0.005 * amount));
  return value + n;
}

function formatNumberLike(original, value){
  const s = String(original);
  const dot = s.indexOf('.')
  const decimals = dot >= 0 ? (s.length - dot - 1) : 0;
  if (!Number.isFinite(value)) return s;
  if (decimals <= 0) return String(Math.round(value));
  return value.toFixed(decimals);
}

async function main(){
  const args = parseArgs(process.argv);
  const rng = makeRng(args.seed);
  const raw = await fs.readFile(args.in, 'utf8');
  const lines = raw.split(/\r?\n/);
  const out = new Array(lines.length);

  for (let i = 0; i < lines.length; i++){
    const line = lines[i];
    if (!line || line.trim().length === 0) { out[i] = line; continue; }
    if (/^\s*#/.test(line)) { out[i] = line; continue; }

    // Preserve header line unmodified
    if (i === 0 && /^\s*run\s*,\s*step\s*,\s*metric\s*,\s*value\s*,\s*stderr\s*$/i.test(line)) {
      out[i] = line; continue;
    }

    const cols = line.split(',');
    if (cols.length < 4) { out[i] = line; continue; }

    const [run, stepStr, metric, valueStr, stderrStr = ''] = cols;
    const trimmedMetric = (metric || '').trim();
    const valueNum = Number((valueStr || '').trim());

    if (!Number.isFinite(valueNum)) { out[i] = line; continue; }

    const jittered = jitterValue(trimmedMetric, valueNum, args.amount, rng);
    const valueOut = formatNumberLike(valueStr, jittered);

    // Reassemble with original column count and positions
    const result = [run, stepStr, metric, valueOut, stderrStr].join(',');
    out[i] = result;
  }

  const finalText = out.join('\n');
  await fs.writeFile(args.out, finalText, 'utf8');
  const relIn = path.relative(process.cwd(), args.in);
  const relOut = path.relative(process.cwd(), args.out);
  console.log(`Jittered data written: ${relOut} (from ${relIn})`);
}

main().catch(err => {
  console.error(err?.stack || String(err));
  process.exit(1);
});

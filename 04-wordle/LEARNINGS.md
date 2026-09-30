# What the CPU runs taught us

Every Wordle number is greedy solve rate on the same 200-word holdout, one seed. Every GeoGuesser number is a within-task group of 8, 20,000 tasks, from the medians in `03-geoguesser/LEARNINGS.md`.

## The headline

Paying for information gain beat paying for a solve. Skipping dead groups did not, on this environment, once the policy could already solve most of the holdout. The skip still did what it says on the tin: sparse GRPO ended with 19% dead groups, sparse alive with 5%.

| arm | best | at | end | dead at end |
|---|---:|---:|---:|---:|
| sparse-grpo | 0.770 | 75 | 0.745 | 0.20 |
| sparse-alive | 0.790 | 50 | 0.700 | 0.05 |
| process-grpo | 0.800 | 200 | 0.800 | 0.11 |
| process-alive | 0.820 | 175 | 0.760 | 0.03 |

## 1. Sparse GRPO peaked and then ate itself

Dead groups on sparse-grpo: 0.02 at the peak, 0.20 at step 200. Collapse — identical guess sequences — 0.00 to 0.08. Solve rate dropped off the peak. That is GeoGuesser run 1's 750 wasted steps, compressed into a few minutes of CPU.

Alive on the same reward kept collapse to 0.03 and dead groups to 0.05. It did not keep the solve rate. Filtering a zero-std group removes a wasted backward pass. It does not create a ranking the reward refused to give.

## 2. Process reward is the ranking the cliff destroyed

Two failed Wordle episodes, one that halved the remaining set and one that guessed `xylyl`, share a sparse reward of 0. They do not share a process reward. Once that ordering exists, ordinary GRPO has something to do, and on this run it used it: process-grpo finished at 0.800, above sparse-grpo's 0.745.

This is the same observation as GeoGuesser's mixture curve. `exp(-d/1492.7) − 0.13` floored 6.43% of within-task groups. The mixture they already train with floors 0.01% of them. Densify first, filter second.

## 3. Population zeros are not group zeros

29.5% of untrained GeoGuesser episodes *never submitted* (`03-geoguesser/LEARNINGS.md` table). If those were independent, P(eight missing) = 0.295⁸ ≈ 5.7×10⁻⁵. Subtract-and-floor zeros more than that — 77/200 in `scoring.py` — because far submitted guesses also floor. Measured within-task, under the game curve, 6.43% of groups are dead, and 6.42 of those 6.43 points are cliffs. One resample of the *same* task takes that to 4.56%: a hard location stays hard. Drawing a fresh location instead was a different experiment, and it is not what DAPO does.

`ACCUM=2` is the setting where this number is the within-task std. `ACCUM=4` mixes two tasks and the between-task variance never collapses, which is why run 2 looked stable and learned less.

## 4. Collapse and a cliff disagree about resampling

Eight identical city guesses: 100% collapse, GRPO scale 10,000×. Resampling the same task draws the same guess. Fifteen kilometres of jitter around 662 km: 0% dead, GRPO scale 277× median, 463× at p95. The group is live in the `std > 0` sense and the amplifier is still a 300× bet on noise.

Ranks on both of those groups are in [−0.5, 0.5]. On the identical-guess group they are zero, which is correct — there is no ordering — and the right move is to stop, not to divide by 10⁻⁴.

## 5. One seed, a pointer policy, no Qwen

The tiny policy is a 32-d embedding of the answer list, masked to the grey/green tracker a person keeps on paper. It is not an LLM. The 0.02 gaps between arms at a single checkpoint are smaller than eval noise at n = 200. The dead-group column (0.20 → 0.03) is the one that transfers to GeoGuesser without a further run.

A Qwen training run is `train/grpo_wordle.py`. It is not in this table.

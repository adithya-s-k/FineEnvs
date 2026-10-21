<div align="center">

<a href="https://github.com/attoloop/PermitForge"><img src="https://raw.githubusercontent.com/attoloop/PermitForge/55d4425/docs/banner.jpg" alt="PermitForge: a synthetic plant's stacks and buildings inside its fence, the receptors near the plant coloured by how much room the submitted design left under the hourly NO2 standard" width="100%"></a>

<h1>PermitForge</h1>

<h3>Make a plant meet every air-quality standard at the lowest cost</h3>

<p>Choose emissions controls and stack heights for a synthetic industrial plant under five years of real weather.
Thirty air-quality evaluations, then one submission, graded by EPA's own dispersion model.</p>

<a href="https://huggingface.co/spaces/Bhuminjay/PermitForge"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-Play-FFD21E?style=for-the-badge&labelColor=1a1a1a" alt="Play in the Space" height="32"></a>
<a href="https://huggingface.co/datasets/Attoloop/PermitForge"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Tasks%2C%20rollouts%20%26%20training-4F46E5?style=for-the-badge&labelColor=1a1a1a" alt="Dataset" height="32"></a>
<a href="https://huggingface.co/Attoloop/PermitForge-Qwen3.8-27B-GRPO"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-GRPO%20adapter-10B981?style=for-the-badge&labelColor=1a1a1a" alt="The trained adapter" height="32"></a>
<a href="https://github.com/attoloop/PermitForge"><img src="https://img.shields.io/badge/GitHub-attoloop%2FPermitForge-181717?style=for-the-badge&logo=github&labelColor=1a1a1a" alt="PermitForge on GitHub" height="32"></a>

<sub>Weather: NOAA ISD and IGRA2, 2020 to 2024, processed by EPA AERMET. Models: EPA AERMOD 26135 and BPIPPRM 04274. Facilities are synthetic.</sub>

</div>

---

## The job

A plant already meets its limits with every stack on maximum control at its tallest allowed height. The
agent makes it cheaper. For each source it picks one of three controls (none, moderate, maximum) and one
height from a menu that ends at the Good Engineering Practice limit. A design must meet every standard
(hourly and annual NO2, daily and annual PM2.5 with two annual limits, hourly, three-hour and annual SO2) at every one of 184
receptors, as five-year design values over real hourly weather, after adding the background already in the air.

| Tool | What it does |
|---|---|
| `get_scenario()` | Sources, controls, height menus, standards, the draft and the evaluations left |
| `check_design(design)` | Validity and exact annualized cost, free and unlimited |
| `evaluate_design(design)` | Design values, margins and binding receptors from precomputed unit fields, 30 per episode |
| `submit_design(design)` | Ends the episode, runs BPIP-PRIME and AERMOD and grades once |

The environment runs on [OpenEnv](https://github.com/huggingface/OpenEnv) over MCP. 360 tasks: 230 train
at 10 weather stations, 130 eval at 5 other stations.

## Data and licences

Surface weather comes from NOAA's Integrated Surface Database and upper air from its Integrated Global
Radiosonde Archive, for 15 US stations, processed by AERMET. The dispersion models are EPA's AERMOD and
BPIP-PRIME, built from EPA source with pinned hashes. Plants, emissions, controls, prices and backgrounds are
generated. NOAA and EPA inputs are U.S. Government works. Code and data are Apache-2.0. The repository's
`SOURCES.md` lists every input file with its hash.

## Grading

Reward is the fraction of the draft design's annualized cost that a feasible submission saves:
`(cost(draft) - cost(submission)) / cost(draft)`. Invalid, infeasible and missing submissions score zero, and
so does the unchanged draft. There is no proven optimum and no model ceiling: 1.0 would mean zero cost, which
no task allows. Every submission is verified by the native models, never by the fast evaluator the agent
explores with. Nothing is excused.

## Results

Four models, 10 held-out eval tasks, 3 repeats each, 40 turns, 30 evaluations, medium effort. The interval is
a station-clustered bootstrap. Every episode submitted a feasible design.

| Model | Mean saved fraction | 95% CI | USD per episode |
|---|---:|---|---:|
| GPT-6.1 Sol | 0.606 | 0.542 to 0.677 | subscription |
| GPT-5.6 Sol | 0.586 | 0.519 to 0.663 | subscription |
| Claude Opus 5.5 | 0.576 | 0.501 to 0.660 | 1.71 |
| Claude Sonnet 5.5 | 0.507 | 0.424 to 0.600 | 0.37 |

Model-free policies on the same tasks: the unchanged draft 0.00, cheapest-first greedy 0.33, random search
with 30 evaluations 0.31, best of eight random runs 0.46, a 60-second search 0.48. Card:
[results/eval-v2.json](results/eval-v2.json).

**Training.** Our first training run was small: Qwen3.8-27B with GRPO (LoRA, 30 steps on 30 train tasks, 6
rollouts per step, one H200), evaluated before and after on the same 10 held-out tasks with the same harness. It
did not change the score: base 0.439, trained 0.439. A larger run on all 230 train tasks is next. The per-task table, curves and every failure met on the way are in the repository's
`train/cloud/RESULTS.md`. Card: [results/eval-v2-qwen.json](results/eval-v2-qwen.json).
That harness differs from the frontier one, so the two cards are not compared.

## Links

- [Space](https://huggingface.co/spaces/Bhuminjay/PermitForge): the playground and the MCP endpoint.
- [Dataset](https://huggingface.co/datasets/Attoloop/PermitForge): tasks, weather, every rollout with its transcript, and the whole training run.
- [Model](https://huggingface.co/Attoloop/PermitForge-Qwen3.8-27B-GRPO): the GRPO adapter and its checkpoints.
- [Code](https://github.com/attoloop/PermitForge): environment, verifier, evaluation harness, protocol and training recipe.

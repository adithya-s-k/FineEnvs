"""Assemble a source-linked, image-free article handoff from accepted local results."""
import csv,hashlib,json,math,shutil
from datetime import datetime,timezone
from pathlib import Path
import mistune
OUT=Path(__file__).resolve().parent
WORK=OUT.parents[3];EXP=WORK/'experiments'
ROOT=EXP/'sft-rl-complete-comparison-20260928'
C=Path(json.loads((ROOT/'latest.json').read_text())['directory'])
E=Path(json.loads((ROOT/'latest_efficiency.json').read_text())['directory'])
comparison=json.loads((C/'comparison.json').read_text());eff=json.loads((E/'efficiency.json').read_text())
H=['opencode','claude-code','codex','mini-swe-agent'];N={'qwen':'Qwen3.5-2B','lfm':'LFM2.5-2.6B'}
for source,name in [(C/'comparison.json','results.json'),(C/'all_checkpoints.csv','all_checkpoints.csv'),(E/'efficiency.json','tool_efficiency.json'),(E/'efficiency.csv','tool_efficiency.csv')]:shutil.copy2(source,OUT/name)
# Verify referenced accepted scores have not changed since the plotting snapshot.
source_hashes=comparison['source_sha256'];verified=0
for path,expected in source_hashes.items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
 verified+=1
runs=[('LFM bash SFT, matched 907',EXP/'lfm25-sft-20260927',907,907,[114,228],'86340'),('LFM bash SFT, full 4,677',EXP/'lfm25-sft-full-20260927',4677,4677,[585,1170],'86367'),('LFM OpenCode SFT',EXP/'opencode-student-sft-20260927/lfm',801,4825,[604,1208],'86427'),('Qwen OpenCode SFT',EXP/'opencode-student-sft-20260927/qwen',801,4825,[604,1208],'86429'),('LFM multi-harness SFT',EXP/'multiharness-student-sft-20260928/lfm',888,17929,[2242,4484],'86472'),('Qwen multi-harness SFT',EXP/'multiharness-student-sft-20260928/qwen',888,17929,[2242,4484],'86473')]
training=[]
for label,root,tasks,examples,steps,job in runs:
 p=root/'production/metrics.jsonl';rows=[json.loads(l) for l in p.open() if l.strip()];ls=[r for r in rows if 'loss' in r]
 final=rows[-1];assert final['epoch']==2 and final['step']==steps[-1],label
 assert all(math.isfinite(r[k]) for r in rows for k in ['loss','grad_norm'] if k in r),label
 training.append(dict(label=label,root=str(root),tasks=tasks,examples=examples,epoch_steps=steps,job=job,metrics_source=str(p),metrics_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),runtime_seconds=final['train_runtime'],mean_train_loss=final['train_loss'],loss_first50=sum(r['loss'] for r in ls[:50])/len(ls[:50]),loss_last50=sum(r['loss'] for r in ls[-50:])/len(ls[-50:]),final=final))
(OUT/'training_summary.json').write_text(json.dumps(training,indent=2)+'\n')
lines=[]
def add(s):lines.extend(s.strip().splitlines());lines.append('')
def table(headers,rows):
 lines.append('| '+' | '.join(headers)+' |');lines.append('| '+' | '.join(['---']*len(headers))+' |')
 for row in rows:lines.append('| '+' | '.join(str(v).replace('|','/') for v in row)+' |')
 lines.append('')
def label(r):return r['label'].replace('|','/')+(' *' if not r['complete'] else '')+(' †' if r['overlap'] else '')
def pct(v):return 'unavailable' if v is None else f'{100*v:.1f}%'
def link(p,title=None):return f'[{title or Path(p).name}]({p})'
add('''# Article handoff: SFT and RL across agent harnesses

Prepared 29 September 2026 from accepted results through 28 September. This is an evidence packet for updating the existing article, not a claim that all methods were compared under identical training conditions. No images are embedded. The HTML version contains the same text and semantic tables.

## Start here

Extend the article from multi-harness RL to **learning from and through agent harnesses**: supervised imitation of successful teacher traces, then a comparison with the existing online RL experiments. Preserve the OpenEnv/Harbor/TiTO explanation and the earlier RL findings. Add the SFT evidence without replacing or rewriting the recorded RL history.

The central result is model-dependent. Qwen3.5-2B benefits strongly from multi-harness SFT: 14.6% baseline → 30.9% after one epoch → 32.7% after two. LFM2.5-2.6B improves with OpenCode-only SFT to 47.5%, but multi-harness SFT reaches only 43.1%, close to its approximately 42.2% baseline. LFM multi-harness RL retains the strongest final correctness at 54.2% and reduces native tool calls by 31.1% on baseline-matched successes. Multi-harness SFT also reduces tool use across all four harnesses in both models, despite having no explicit efficiency reward.

**Training objectives must be stated prominently:** Qwen RL used correctness reward only. LFM RL used correctness plus a tool-efficiency bonus. SFT used demonstration imitation with token-level cross-entropy. Tool efficiency is an evaluation measurement for every method, not a shared training objective.

The latest OpenCode-only and multi-harness SFT runs are complete: two epochs for both models, with all eight checkpoint evaluations graded at 1,000/1,000. The older 907-task bash SFT evaluations remain incomplete. The full 4,677-demonstration bash SFT evaluations are complete but have disclosed test overlap.

## 1. Evaluation protocol and definitions

- Fixed test cohort: 250 tasks, evaluated independently with OpenCode, Claude Code, Codex and Mini-SWE-Agent. One selected graded rollout per task/harness pair, pass@1; 1,000 cells per checkpoint.
- Difficulty per harness: 33 easy, 118 medium, 99 hard. Across four harnesses: 132, 472 and 396 cells respectively. Do not describe those as 1,000 unique tasks.
- Completed overall scores are averages over those 1,000 binary correctness outcomes. Infrastructure failures are retried, not counted as wrong answers. These operational retries are not pass@k sampling, but retry history remains a comparability limitation.
- Student SFT evaluations use temperature 0.8, at most 4,096 output tokens per model call, a 17-call/agent-step budget, and a 600-second agent timeout. Qwen evaluates with thinking disabled; LFM retains its qualified native serving configuration. Both reload the actual saved student checkpoint.
- SFT evaluation jobs initially use concurrency 100 on separate GPUs. LFM epoch-1 recovery used a fresh allocation at concurrency 10. Concurrency is a throughput setting, not the sample count.
- TiTO checks passed for all completed OpenCode and multi-harness SFT evaluations. This establishes the recorded capture checks, not statistical significance or a guarantee of task success.
- Qwen Harbor baseline: 146/1,000 = 14.6%. Native OpenCode has its own historical baseline, 15.9%, and a different serving path. LFM baseline: 421/998 = 42.184%, with two Mini-SWE-Agent grades missing. Preserve that qualification.

## 2. All SFT and selected RL outcomes

E1/E2 mean completed SFT epochs. RL best means the highest **complete** aggregate checkpoint observed on this evaluation set; final means step 1,000. Best is retrospective test-set selection, not an independently chosen validation checkpoint. Every per-harness value in an RL row comes from that same checkpoint.

`*` = incomplete evaluation, graded subset only. `†` = test overlap. Blank values would mean unavailable, never zero.
''')
for model,rr in comparison['panels'].items():
 add('### '+N[model])
 table(['Run','Graded','Overall','OpenCode','Claude Code','Codex','Mini-SWE'],[[label(r),f"{r['graded']}/1,000",pct(r['score']),*[pct(r['harnesses'][h]) for h in H]] for r in rr])
add('''### What can be said from these scores

- Qwen multi-harness SFT E2 is +18.1 percentage points above baseline and +6.2 points above OpenCode-only SFT E2. Compared with the best OpenCode SFT epoch, the gap is +4.3 points. Both SFT variants improve every evaluation harness over the Qwen base model.
- Qwen multi-harness SFT E2 (32.7%) exceeds final Harbor OpenCode RL (26.4%) and final Harbor multi-harness RL (26.3%). It remains below best Harbor OpenCode RL (39.5%) and best Harbor multi-harness RL (37.0%). Do not hide the peak-to-final RL decline or use these observations to claim SFT is categorically better.
- LFM OpenCode-only SFT E2 is 47.5%, about +5.3 points above its provisional baseline. LFM multi-harness SFT E2 is 43.1%, about +0.9 points above baseline and 4.4 points below OpenCode SFT E2. The approximately 0.9-point change is not established as statistically reliable.
- LFM multi-harness SFT recovers from 38.3% at E1 to 43.1% at E2. Improvements over baseline occur in OpenCode, Claude Code and Codex; Mini-SWE falls from approximately 62.1% to 45.2%. The overall number hides that imbalance.
- LFM multi-harness RL finishes at 54.2%, versus 52.3% for OpenCode RL, 47.5% for OpenCode SFT and 43.1% for multi-harness SFT. Relative to multi-harness SFT E2, the final RL gap is 11.1 points. These methods differ in objective, data exposure and budget.
- Full bash SFT reaches 41.3% at E1 and 40.8% at E2 despite using more demonstrations. Its overlap and different task distribution prevent a clean held-out or data-scaling claim. The older 907-task E2 result (46.6% on 850 cells) must not be ranked as a complete result.

## 3. Tool efficiency, measured alongside correctness

For each checkpoint, match task/harness pairs solved correctly by both the baseline and checkpoint. Count native ATIF agent tool invocations with valid unique tool-call IDs. A tool invocation is not a model turn or the number of shell commands inside a tool call. All graded rows used in this packet had native count coverage; missing/unverifiable trajectories would be excluded rather than assigned zero.

`tool-call savings (%) = 100 × (1 − sum(checkpoint calls) / sum(baseline calls))`

Positive means fewer calls; negative means more. This is a ratio of summed calls on matched successes, not an average of per-task percentage changes. Matched subsets vary by checkpoint, so this is conditional efficiency, not the total cost of solving the full test set. SFT did not optimize this metric directly.
''')
for model,rr in eff['models'].items():
 add('### '+N[model]+' tool use')
 table(['Run','Matched pairs','Base mean calls','Run mean calls','Savings','OC savings / n','Claude savings / n','Codex savings / n','Mini-SWE savings / n'],[[label(r),r['efficiency']['all']['matched_successes'],f"{r['efficiency']['all']['baseline_mean']:.2f}",f"{r['efficiency']['all']['checkpoint_mean']:.2f}",f"{r['efficiency']['all']['savings_pct']:+.1f}%",*[f"{r['efficiency'][h]['savings_pct']:+.1f}% / {r['efficiency'][h]['matched_successes']}" for h in H]] for r in rr])
add('''### Efficiency observations worth highlighting

- Qwen multi-harness SFT E2: mean native calls decrease 9.75 → 4.50 on 108 matched successful pairs, a 53.8% saving. Savings are positive for every harness. This accompanies a large correctness gain over baseline.
- LFM multi-harness SFT E2: 7.18 → 5.44 calls on 276 pairs, a 24.2% saving. All four harnesses show positive savings. It is more efficient conditionally, but correctness remains substantially below LFM multi-harness RL.
- LFM OpenCode SFT E2 saves 8.5% overall but increases calls under Claude Code (10.5% more) and Mini-SWE (4.8% more). Its higher correctness than mixed SFT therefore comes with weaker and less uniform tool savings.
- LFM final multi-harness RL: 7.43 → 5.12 calls on 356 pairs, a 31.1% saving, with positive savings in all four harnesses. OpenCode-only RL saves 11.4% overall but uses 9.6% more calls under Claude Code.
- Qwen Harbor RL often uses more calls on matched successes. It was trained on correctness only, so do not describe this as a failed tool-efficiency reward. Qwen native OpenCode RL has its own baseline and saves 16.3% overall.
- Some Qwen RL per-harness cells have very small samples. Final Harbor multi-harness RL under OpenCode has only one matched successful pair; its percentage is not a robust population claim. Show n, and avoid highlighting that cell as a general trend.
- Efficiency differences between Qwen and LFM do not isolate an architectural effect: baseline behavior, training objectives and cohorts differ. Native action counts are verified, but the semantic size of one tool call also differs by harness.

## 4. SFT data, preparation and recipe

Two independently initialized base models were used: `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc` and `LiquidAI/LFM2.5-2.6B` at `654f9463ce32b05d0429d76fe1f580b27d4c1ac0`. The full bash run started fresh, not from the 907-task checkpoint. Multi-harness SFT also started fresh, not from OpenCode SFT.

| Variant | Source and selection | Examples and weighting | Epoch checkpoints |
|---|---|---|---|
| LFM bash, matched | FineEnvs/SmolDataEnvs-sft; 907 available tasks matched to the original LFM RL pool, 376 medium / 531 hard | 907 conversations; assistant-only loss | 114, 228 |
| LFM bash, full | All 4,677 demonstrations: 1,402 easy / 2,640 medium / 635 hard | Full corpus, including disclosed test overlap | 585, 1,170 |
| OpenCode teacher SFT, both models | 801 successful OpenCode rollouts from 801 tasks: 356 medium / 445 hard | 4,825 assistant-turn examples | 604, 1,208 |
| Multi-harness teacher SFT, both models | 3,189 successful rollouts from 888 tasks: 377 medium / 511 hard | 17,929 assistant-turn examples | 2,242, 4,484 |

The teacher source is `AdithyaSK/qwen38-27b-harbor-rollouts`, pinned at `3d826b6854acdb4e4918e5047cdf9f70eb38602e`, collected with Qwen3.8-27B. Mixed data: OpenCode 801 rollouts / 4,825 turns; Claude Code 781 / 4,581; Codex 797 / 4,078; Mini-SWE-Agent 810 / 4,445. There are 692 tasks with successful published traces in all four harnesses, a possible matched subset for a future controlled study. Four successful credential-containing source rollouts were excluded from publication; do not reintroduce them.

Teacher token IDs and behavior logprobs are not student SFT targets. The actual runs used student-specific rendered messages, input IDs and completion masks. Exact model-specific labels were checked against native TRL preparation over the full mixed corpus. Tool responses and earlier history are context, with only the current assistant completion supervised. Qwen training preserves reasoning when present in the teacher response; evaluation still disables thinking as in its baseline. LFM's training-only prefix adjustment is removed when saving its original inference template.

Common recipe: TRL SFT, full parameter fine-tuning, two epochs, LR 3e-6 constant, per-device batch 1, gradient accumulation 8, BF16, paged AdamW 8-bit, gradient checkpointing, seed 42, no packing and no truncation. Save every 50 updates plus epoch boundaries; evaluate E1 and E2. The teacher-trace runs use completion-only cross-entropy; bash runs use assistant-only cross-entropy.

Multi-harness SFT shuffles turn records with standard completion-token loss. It does not balance tasks, rollouts or harnesses. Long answers and multi-turn rollouts contribute more supervised tokens. The 1,000-task RL pool is not the same as the 888 successful-demonstration tasks; these are not identical-data experiments.

### Measured SFT training, not queue-time estimates
''')
table(['Run','Tasks / examples','Epoch steps','GPU job','Trainer runtime','First 50 loss','Last 50 loss'],[[r['label'],f"{r['tasks']} / {r['examples']}",'/'.join(map(str,r['epoch_steps'])),r['job'],f"{r['runtime_seconds']/60:.1f} min",f"{r['loss_first50']:.3f}",f"{r['loss_last50']:.3f}"] for r in training])
add('''All six SFT trainers reached epoch 2. Logged training losses and gradient norms were finite. These losses are from different target distributions and must not be ranked across datasets. The runtimes are the trainer-reported train-loop times; they exclude queueing, most preparation and evaluation, and are not an end-to-end cost comparison.

Teacher-trace prepared token volumes per corpus pass:
''')
table(['Student','OpenCode input tokens','OpenCode supervised tokens','Mixed input tokens','Mixed supervised tokens'],[['LFM','39,730,621','588,577','168,836,025','4,098,011'],['Qwen','42,179,296','610,720','177,276,678','4,361,348']])
add('''These are serialized per-example input lengths, including repeated history, rather than distinct text tokens. Mixed SFT has roughly seven times the supervised target tokens of OpenCode SFT, not simply a different harness label. No claim of equal compute or exposure is justified.

The mixed-data pipeline initially failed before production training because a JSONL reader split embedded Unicode line separators. Reading physical lines fixed six affected records in each student file without removing or changing data. Qwen's reasoning-prefix mismatch was also caught before production and resolved with per-example template settings. Full-corpus label audits and save/resume/evaluation smokes passed before the successful runs. Final model checks confirmed changed weights. Evaluation retries preserved previous grades; transient failures should not be described as model errors.

## 5. RL context to preserve

Qwen's original three async runs used a 1,000-task pool (150 easy / 600 medium / 250 hard), but 1,000 optimizer steps did not guarantee full coverage. Observed distinct tasks were 482 for Harbor multi-harness, 523 for Harbor OpenCode-only and 566 for native OpenCode. They used correctness only. The source reports document different resume histories and rollout filtering.

LFM OpenCode-only and multi-harness RL used a fresh base model and a pool of 1,000 tasks (400 medium / 600 hard), with an explicit two-pass schedule and a 1,000-update cap. A multi-harness group uses one harness for that scheduled task, not all four harnesses per update. Configured recipe: AsyncGRPO, LR 3e-6, eight generations, max in-flight 32, staleness 4, batch 4 / accumulation 4, BF16, paged AdamW 8-bit, a 40,960-token packing budget and atomic rollout admission. A trainer GPU and separate inference GPU were allocated per run. The configured two-pass schedule is not itself evidence of complete realized task coverage.

LFM reward: `correctness × (1 + 0.1 × 15 / (15 + native_tool_calls))`, with no bonus for an unverified count. The tested implementation also withholds the efficiency bonus when the count is zero. Incorrect answers get no bonus. The primary evaluation score remains binary correctness; do not substitute the shaped training reward for pass@1.

Both RL families saved every 50 updates and evaluated scheduled 100-step checkpoints; extra resume/diagnostic checkpoints exist. Best and final comparisons use only completed eligible evaluations. Historical deployment, allocation and transport configurations changed, so do not infer exact executed settings solely from an old planned config's `status` field.

Earlier Qwen findings remain relevant: the multi-harness decline starts after step 500; the documented task-repetition resume bug occurs after step 684 and cannot explain the initial decline. The earlier report records increasing output truncation and 35–58% of updates with no fresh reward-contrast gradient. These are observed associations and accounting issues, not proven explanations of the SFT results or of LFM behavior.

A Qwen hard-task continuation from checkpoint 500 remains a separate, incomplete local evaluation cohort: its latest local step-1,000 score is 26.0% on 924/1,000 grades. Do not merge it into the original Qwen RL curve or present it as a successful hard-task curriculum. SETA native-bash-only evaluation uses a different protocol and is not included in the four-harness aggregates here.

## 6. Difficulty breakdown for completed SFT and selected RL

Values are averages over available graded cells within each difficulty. The full-dataset bash SFT overlap flag still applies. Per-checkpoint raw counts and definitions remain in the linked source JSONs.
''')
for model,rr in comparison['panels'].items():
 rows=[]
 for r in rr:
  if r.get('difficulty') and r['complete']:rows.append([label(r),*[pct(r['difficulty'].get(k)) for k in ['easy','medium','hard']]])
 add('### '+N[model]);table(['Run','Easy','Medium','Hard'],rows)
add('''## 7. Article update instructions

The local article checkout inspected for this handoff is `/tmp/data-agent-article-merge/content/articles/multi-harness-rl/`. Its checked chapters still describe the September Qwen RL experiments. This is a local reference, not a claim that it matches the latest article PR or deployed Space. Before editing, inspect the actual target branch and preserve newer content.

1. **Introduction / framing:** keep harnesses and OpenEnv/Harbor as the organizing idea. Add that the same trajectory infrastructure supports supervised imitation as well as online RL. A suitable subtitle is “Learning across agent harnesses with SFT and RL.”
2. **Data-agent / data preparation:** add the two teacher-trace SFT datasets, success filtering, task counts, current-response masks and student-specific tokenization. Distinguish raw rollouts from assistant-turn examples. Mention the ready-to-use public SFT export.
3. **Training-ablation chapter:** organize by model first, then SFT versus RL. Show both SFT epochs and best/final RL. Preserve the original Qwen RL resume and exposure findings. Add LFM RL as a separate later experiment, not a continuation of Qwen.
4. **Evaluation chapter:** keep pass@1 primary. Add native tool-call savings on matched successes, formula and n. State Qwen RL correctness-only, LFM RL correctness plus efficiency, and SFT imitation beside the figures. Do not call all plotted savings an optimized reward.
5. **Conclusions:** replace “multi-harness training always wins” language with the actual model-dependent picture. Qwen mixed SFT improves broadly. LFM mixed SFT reduces calls but does not match OpenCode SFT correctness. LFM mixed RL combines higher correctness with broad call savings.
6. **Reproduction:** link the public portable and student-tokenized dataset configurations, exact recipe and local evidence packet. Update historical HuggingEnvs display names/links to FineEnvs where the destination exists; verify remote links before publishing.
7. **Visuals:** retain the distinct baseline / SFT / RL sections. Show incomplete and overlap flags. Avoid ranking methods only by their best result, or removing the Qwen reward-objective annotation. Use the accompanying tables/JSON to build HTML visualizations; this handoff does not require image interpretation.

Useful next experiments can be proposed as future work: compare harness mixtures on the same 692 common tasks, equalize supervised target-token exposure, balance task/rollout contributions, and compare SFT→RL against RL from base. None of those controlled comparisons has been run in this packet.

### Claims to avoid

- “SFT fails on LFM”: OpenCode SFT improves to 47.5%; mixed SFT is different.
- “Mixed SFT proves harness diversity causes the gain”: data volume, targets and compute differ.
- “RL always beats SFT”: Qwen final RL scores are below mixed SFT, while best RL checkpoints are above it.
- “Qwen's efficiency reward failed”: Qwen did not have one.
- “More efficient means cheaper to solve the whole benchmark”: savings are conditional on pairs solved by both, and tokens/latency are different metrics.
- “TiTO passing proves the run is statistically or operationally perfect”: it validates specific capture checks.
- “All evaluations are complete”: the current teacher-trace SFT evaluations are complete, but historical 907-task SFT, LFM baseline and hard-task continuation have missing cells.
- “Full bash SFT is held out”: it includes test-notebook and question overlap by explicit user choice.
- SFT metrics originally logged locally were backfilled to the public Trackio project on 29 September 2026. See TRACKIO_PUBLICATION.md for the verified dashboard and commit-pinned citation.

## 8. Artifacts and exact provenance

### Public entry points

- Existing article reference: https://huggingface.co/spaces/FineEnvs/multi-harness-rl . Historical user-facing URL: https://huggingface.co/spaces/AdithyaSK/multi-harness-rl . Verify the current destination before publishing.
- Article review PR: https://github.com/adithya-s-k/FineEnvs/pull/13 . Inspect its current head before applying this handoff.
- Historical RL dashboard: https://huggingface.co/spaces/FineEnvs/data-agent-training-comparison-trackio/?project=data-agent-rl-comparison . The exact score artifacts in this packet take precedence over a stale dashboard view.
- Portable SFT dataset: https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-multiharness-sft . Configurations: `all`, four individual harnesses, `lfm25_2_6b`, `qwen35_2b`.
- Published training script: https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-multiharness-sft/blob/main/train_sft.py . Requirements and verification receipt live alongside it. Data/code revision tested during publication: `a57fbba49e892199d2b0c31f68034310bf2e281c`.
- Raw teacher trajectories: https://huggingface.co/datasets/AdithyaSK/qwen38-27b-harbor-rollouts . Bash demonstrations: https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-sft (revision `a9fa95ab5dff4f4522081f7eb003109e56b7a36d`).

The Hub SFT example provides a convenient simplified epoch-save recipe. The exact historical cluster runs additionally saved every 50 updates and used the local gated pipeline. Do not claim the public script automatically reproduces the cluster scheduler, full evaluation orchestration or all save policies.

### Local run roots and checkpoints
''')
for r in training:
 add(f"- **{r['label']}**: {link(r['root'],'run root')}. Checkpoints: `production/run/checkpoint-{r['epoch_steps'][0]}` and `checkpoint-{r['epoch_steps'][1]}`. Metrics: {link(r['metrics_source'])}. Local Trackio: `production/trackio/`. Evaluations: `production/evaluations/checkpoint-<step>/`.")
add(f'''The four teacher-trace student runs have `evaluation_verified.json` at both epoch checkpoints. For the two old incomplete bash-907 evaluations use `eval_progress.json`, not an invented complete receipt. Completion/optimizer evidence is in each production directory; mixed SFT has `training_verified_step_4484.json` and `eval_controller_summary.json`.

- Exact OpenCode SFT code: {link(EXP/'opencode-student-sft-20260927/code','code directory')} and {link(EXP/'opencode-student-sft-20260927/final_config.json')}.
- Exact mixed SFT code: {link(EXP/'multiharness-student-sft-20260928/code','code directory')} and per-model `production/config.json`; {link(EXP/'multiharness-student-sft-20260928/CONFIG.json')} is a historical plan whose status field is stale.
- LFM RL roots: {link(EXP/'lfm25-medium-hard1000-20260921/opencode','OpenCode RL')} and {link(EXP/'lfm25-medium-hard1000-20260921/multi-harness','multi-harness RL')}; configs, per-allocation train audits and canonical checkpoint scores are inside.
- Earlier Qwen explanation: {link(WORK/'HuggingEnvs/04-data-agent/results.md')} and {link(WORK/'HuggingEnvs/04-data-agent/reports/three-run-analysis-20260917/REPORT.md')} / {link(WORK/'HuggingEnvs/04-data-agent/reports/three-run-analysis-20260917/TRAINING.md')}.
- Dataset publication evidence: {link(EXP/'multiharness-sft-hub-20260928/RESULTS.md')}; full source export and Hub verification receipts live alongside it.
- All-run plotting code: {link(ROOT/'plot_all.py')}. Native-tool extraction and plots: {link(ROOT/'tool_efficiency.py')}.
- Latest correctness figures and HTML: {link(C,'artifact directory')}. Latest efficiency figures and method report: {link(E,'artifact directory')}. They are local artifacts, not published image URLs.

### Exact score source for each displayed row
''')
for model,rr in comparison['panels'].items():
 for r in rr:add(f"- **{N[model]}, {label(r)}:** {link(r['source'])}")
add('''### Machine-readable packet

- `results.json`: all displayed rows, all recorded comparison checkpoints, source hashes and caveats.
- `all_checkpoints.csv`: full available checkpoint history, with completeness and overlap flags.
- `tool_efficiency.json` / `tool_efficiency.csv`: matched-pair counts, before/after native call means, per-harness savings, and references to audited cohort caches with trajectory hashes.
- `training_summary.json`: run roots, optimizer steps, measured training runtime, training loss windows and metrics hashes.
- `manifest.json`: checksums for the packet and the exact input snapshots.

Local absolute paths identify the original evidence. If handing this to an agent without this filesystem, send the entire packet directory; its summary and tables are self-contained. Raw traces, full checkpoints and tokenized training datasets remain at the linked paths and are not copied into this packet.

## Suggested short message to the article-writing agent

Please update the existing multi-harness article to cover SFT as well as RL using this evidence packet. Keep the architecture and earlier RL history, then add a model-by-model comparison of baseline, both SFT epochs, and best/final RL. Qwen multi-harness SFT reaches 32.7% from 14.6%; LFM OpenCode SFT reaches 47.5%, mixed SFT 43.1%, and mixed RL finishes at 54.2%. Include native tool-call savings on matched successes, with sample sizes. Explicitly state that Qwen RL was correctness-only, LFM RL used an efficiency bonus, and SFT used imitation. Preserve overlap/incomplete flags and avoid causal claims because data and budgets differ. Use the HTML/JSON tables and exact source links, not numbers read from images. Check the current article branch before editing and do not overwrite newer material.
''')
if (OUT/'TRACKIO_PUBLICATION.md').exists():
 add('\n## SFT runtime metrics and public dashboard\n\n' + (OUT/'TRACKIO_PUBLICATION.md').read_text().replace('# Public SFT Trackio publication\n\n', ''))
text='\n'.join(lines).rstrip()+'\n';assert '—' not in text
(OUT/'HANDOFF.md').write_text(text)
renderer=mistune.create_markdown(plugins=['table']);body=renderer(text)
(OUT/'HANDOFF.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SFT and RL article handoff</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1450px;margin:36px auto;padding:0 28px;color:#172638}h1,h2,h3{line-height:1.25}h2{margin-top:42px;border-top:1px solid #dce4ee;padding-top:20px}table{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}th,td{padding:9px;border:1px solid #dce4ee;text-align:left}th{background:#edf4fc}tr:nth-child(even){background:#f8fafc}code{background:#eef1f5;padding:2px 4px}a{color:#185c9b;overflow-wrap:anywhere}li{margin:7px 0}</style></head><body>'+body+'</body></html>')
assert '<img' not in (OUT/'HANDOFF.html').read_text()
manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),comparison_snapshot=str(C),efficiency_snapshot=str(E),verified_score_sources=verified,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT);print('Source hashes verified:',verified,'Displayed rows:',sum(map(len,comparison['panels'].values())),'Historical rows:',len(comparison['all_checkpoints']),'Markdown lines:',len(lines))

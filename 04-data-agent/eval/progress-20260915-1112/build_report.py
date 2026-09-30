"""Read-only snapshot of fixed-set evaluations and the actual training resume chain."""
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

REPO = Path('/fsx/adithyaskolavi/projects/trl_prod')
LOGS = REPO / 'experiments/async_grpo_harbor_data_agent/logs'
OUT = Path(__file__).resolve().parent
MAIN = LOGS / 'multi4-long-prod-20260915'
BASE = LOGS / 'multi4-baseline-20260914'
CK100 = LOGS / 'multi4-long-bounded-20260915/checkpoint-evals/step-000100'
CK200 = MAIN / 'checkpoint-evals/step-000200'
HARNESSES = ['opencode', 'claude-code', 'codex', 'mini-swe-agent']
LEVELS = ['easy', 'medium', 'hard']
NAMES = dict(zip(HARNESSES, ['OpenCode', 'Claude Code', 'Codex', 'Mini-SWE-Agent']))


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines(keepends=True) if s.endswith('\n')]


def traces(root):
    selected = {}
    ungraded = 0
    for p in sorted(root.glob('*.jsonl')):
        for row in lines(p):
            if row.get('reward') in (0, 1) and row.get('n_turns', 0) > 0:
                assert row['rep'] == 0
                selected.setdefault((row['harness'], row['index']), row)
            else:
                ungraded += 1
    return selected, ungraded


def summary(rows):
    n = len(rows)
    correct = int(sum(x['reward'] for x in rows))
    return {'correct': correct, 'graded': n, 'pass_at_1': correct / n if n else None}


manifest = read(BASE / 'manifest.json')
identities = [(t['name'], t['question_hash'], t['difficulty']) for t in manifest['tasks']]
for path in (CK100 / 'manifest.json', CK200 / 'manifest.json'):
    assert [(t['name'], t['question_hash'], t['difficulty']) for t in read(path)['tasks']] == identities

base = read(BASE / 'job-78215/canonical_results.json')
selected = {'base': {(r['harness'], r['index']): r for r in base['selected']}}
selected['100'], excluded100 = traces(CK100 / 'job-79057/traces')
selected['200'], excluded200 = traces(CK200 / 'job-79092/traces')
expected = {(h, i) for h in HARNESSES for i in range(250)}
assert set(selected['base']) == set(selected['100']) == expected
assert set(selected['200']) <= expected
assert sum(r['reward'] for r in selected['base'].values()) == 146
assert sum(r['reward'] for r in selected['100'].values()) == 248
checkpoint200_final = read(CK200 / 'scores.json') if (CK200 / 'scores.json').exists() else {}
evaluations = {}
for name, rows in selected.items():
    evaluations[name] = {
        **summary(list(rows.values())),
        'coverage_complete': set(rows) == expected,
        'final_audited': name != '200' or checkpoint200_final.get('comparison_ready', False),
        'harnesses': {}, 'difficulty': {},
    }
    for h in HARNESSES:
        subset = [r for (harness, _), r in rows.items() if harness == h]
        evaluations[name]['harnesses'][h] = {**summary(subset), 'difficulty': {}}
        for level in LEVELS:
            values = [r for r in subset if manifest['tasks'][r['index']]['difficulty'] == level]
            evaluations[name]['harnesses'][h]['difficulty'][level] = summary(values)
    for level in LEVELS:
        evaluations[name]['difficulty'][level] = summary([
            r for r in rows.values() if manifest['tasks'][r['index']]['difficulty'] == level
        ])

common = set(selected['200'])
matched = {}
for name, rows in selected.items():
    matched[name] = {
        **summary([rows[key] for key in sorted(common)]),
        'harnesses': {h: summary([rows[k] for k in sorted(common) if k[0] == h]) for h in HARNESSES},
    }
paired = {}
for earlier in ('base', '100'):
    wins = sum(selected['200'][k]['reward'] > selected[earlier][k]['reward'] for k in common)
    losses = sum(selected['200'][k]['reward'] < selected[earlier][k]['reward'] for k in common)
    paired[earlier] = {'newly_solved': wins, 'newly_unsolved': losses, 'net': wins - losses}

# Follow checkpoint ancestry, retaining only steps actually inherited by each resume.
root = MAIN
upper = float('inf')
metrics = {}
provenance = []
while True:
    config = read(root / 'run_config.json')
    job = read(root / 'submission.json')['training']
    source = root / f'job-{job}/audit/metrics.jsonl'
    resume = config.get('resume_state') or {}
    lower = resume.get('step', 0)
    kept = [r for r in lines(source) if 'grad_norm' in r and lower < r['step'] <= upper]
    for row in kept:
        assert row['step'] not in metrics
        metrics[row['step']] = {**row, 'source_job': str(job)}
    provenance.append({'job': str(job), 'metrics_file': str(source), 'resume_step': lower,
                       'retained_steps': [r['step'] for r in kept],
                       'file_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    if not resume.get('checkpoint'):
        break
    upper = lower
    root = Path(resume['checkpoint']).parents[2]
rows = [metrics[k] for k in sorted(metrics)]
assert [r['step'] for r in rows] == list(range(1, rows[-1]['step'] + 1))
current = [r for r in rows if r['source_job'] == '79083']
bounded = [r for r in rows if r['step'] >= 54]
windows = []
for start, end in [(54, 100), (101, 150), (151, 200), (201, rows[-1]['step']),
                   (rows[-1]['step'] - 39, rows[-1]['step'] - 20),
                   (rows[-1]['step'] - 19, rows[-1]['step'])]:
    subset = [r for r in rows if start <= r['step'] <= end]
    windows.append({'start': start, 'end': end, 'updates': len(subset),
                    'mean_logged_reward': mean(r['reward'] for r in subset),
                    'nonzero_gradient_updates': sum(r['grad_norm'] > 0 for r in subset)})
monitor = read(MAIN / 'monitor/status.json')
training = {
    'latest_step': rows[-1]['step'], 'windows': windows,
    'current_job_updates': len(current),
    'current_job_nonzero_gradient_updates': sum(r['grad_norm'] > 0 for r in current),
    'current_job_nonfinite_updates': sum(any(not math.isfinite(r[k]) for k in
        ('loss', 'grad_norm', 'ratio', 'kl', 'entropy') if isinstance(r.get(k), (int, float))) for r in current),
    'current_job_max_sample_staleness': max(r.get('sample/staleness_max', 0) for r in current),
    'current_job_stale_rows_dropped': sum(r.get('sample/dropped_stale_total', 0) for r in current),
    'current_job_stale_rollouts_dropped': sum(r.get('admission/stale_rollouts_dropped_total', 0) for r in current),
    'current_job_oversize_rows_dropped': sum(r.get('batch/dropped_oversize_total', 0) for r in current),
    'current_job_stale_drop_steps': [r['step'] for r in current if r.get('sample/dropped_stale_total', 0)],
    'monitor': {k: monitor.get(k) for k in ('checked_at', 'optimizer_step', 'alerts', 'saved_checkpoints', 'tito')},
    'trackio': read(MAIN / 'trackio/status.json'),
    'semantics': 'Unweighted mean of logged per-update reward; not unique-rollout or fixed-task pass@1. Training task/harness mix varies. Plot starts at bounded-admission recipe step54.',
    'source_chain': provenance,
}
report = {'snapshot_utc': datetime.now(timezone.utc).isoformat(), 'evaluations': evaluations,
          'difficulty_task_counts': dict(Counter(t['difficulty'] for t in manifest['tasks'])),
          'matched_completed_cells': matched, 'paired_changes': paired, 'training': training,
          'excluded_ungraded_attempts': {'100': excluded100, '200': excluded200},
          'sources': {'base': str(BASE / 'job-78215/canonical_results.json'),
                      '100': str(CK100 / 'scores.json'), '200': str(CK200 / 'job-79092/traces')},
          'checkpoint200_missing': [{'harness': h, 'index': i, 'difficulty': manifest['tasks'][i]['difficulty']}
                                    for h, i in sorted(expected - set(selected['200']))]}
(OUT / 'snapshot.json').write_text(json.dumps(report, indent=2) + '\n')
with (OUT / 'training_metrics.csv').open('w') as stream:
    fields = ['step', 'source_job', 'reward', 'grad_norm', 'loss', 'ratio', 'kl',
              'sample/staleness_max', 'sample/dropped_stale_total', 'batch/samples_per_step']
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader(); writer.writerows(rows)
with (OUT / 'evaluation_breakdown.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=['checkpoint', 'harness', 'difficulty', 'correct', 'graded', 'pass_at_1'])
    writer.writeheader()
    for name, ev in evaluations.items():
        for h, hdata in ev['harnesses'].items():
            for level, counts in hdata['difficulty'].items():
                writer.writerow({'checkpoint': name, 'harness': h, 'difficulty': level, **counts})

def cell(x):
    return f"{100*x['pass_at_1']:.1f}% ({x['correct']}/{x['graded']})"

markdown = [f"# Evaluation and reward snapshot — {report['snapshot_utc']}", '',
    'Same 250 test tasks per harness: 33 easy, 118 medium, 99 hard. Metric: pass@1.', '',
    f"Checkpoint200 has {len(common)}/1000 graded cells. Its full coverage, harness-version and TiTO final audit remains pending.", '',
    '| Harness | Base | Checkpoint100 | Checkpoint200 — provisional |', '| --- | ---: | ---: | ---: |']
for h in HARNESSES:
    markdown.append('| ' + NAMES[h] + ' | ' + ' | '.join(cell(evaluations[n]['harnesses'][h]) for n in selected) + ' |')
markdown += ['| Overall | ' + ' | '.join(cell(evaluations[n]) for n in selected) + ' |', '',
             '| Harness | Difficulty | Base | Checkpoint100 | Checkpoint200 — provisional |',
             '| --- | --- | ---: | ---: | ---: |']
for h in HARNESSES:
    for level in LEVELS:
        markdown.append('| ' + NAMES[h] + ' | ' + level + ' | ' + ' | '.join(
            cell(evaluations[n]['harnesses'][h]['difficulty'][level]) for n in selected) + ' |')
markdown += ['', '| Difficulty, all harnesses | Base | Checkpoint100 | Checkpoint200 — provisional |',
             '| --- | ---: | ---: | ---: |']
for level in LEVELS:
    markdown.append('| ' + level + ' | ' + ' | '.join(cell(evaluations[n]['difficulty'][level]) for n in selected) + ' |')
markdown += ['', f"On the same {len(common)} completed task–harness pairs: " + '; '.join(
    f"{n}: {cell(matched[n])}" for n in selected) + '.', '',
    '## Training reward', '', '| Steps | Mean logged reward | Nonzero-gradient updates |', '| --- | ---: | ---: |']
for w in windows:
    markdown.append(f"| {w['start']}–{w['end']} | {w['mean_logged_reward']:.4f} | {w['nonzero_gradient_updates']}/{w['updates']} |")
markdown += ['', training['semantics'], '',
    f"Current trainer79083: step{training['latest_step']}, {training['current_job_nonzero_gradient_updates']}/{len(current)} nonzero-gradient updates; {training['current_job_nonfinite_updates']} nonfinite updates; observed maximum staleness{training['current_job_max_sample_staleness']}; {training['current_job_stale_rollouts_dropped']} whole-rollout stale rejections ({training['current_job_stale_rows_dropped']} rows), at steps{training['current_job_stale_drop_steps']}; {training['current_job_oversize_rows_dropped']} oversized-row rejections.", '',
    'Monitor/Trackio timestamps, TiTO evidence, exclusions, matched counts, source-chain cutoffs and all plotted values are in snapshot.json and the CSV files.']
(OUT / 'REPORT.md').write_text('\n'.join(markdown) + '\n')
print(json.dumps({'snapshot_utc': report['snapshot_utc'], 'evaluations': evaluations,
                  'matched': matched, 'paired': paired, 'windows': windows,
                  'training_stability': {k: v for k, v in training.items() if k not in ('source_chain', 'monitor', 'trackio', 'windows')}}, indent=2))

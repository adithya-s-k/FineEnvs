"""Plot only the frozen snapshot, with provisional evaluation points distinguished."""
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
report = json.loads((ROOT / 'snapshot.json').read_text())
with (ROOT / 'training_metrics.csv').open() as stream:
    rows = [{k: float(v) for k, v in row.items() if v and k != 'source_job'} for row in csv.DictReader(stream)]
rows = [r for r in rows if r['step'] >= 54]
steps = [r['step'] for r in rows]
reward = [r['reward'] for r in rows]
rolling = [mean(reward[max(0, i-19):i+1]) if i >= 19 else float('nan') for i in range(len(rows))]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})
fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.5), sharex=True, height_ratios=[2, 1])
axes[0].plot(steps, reward, color='#bcc9d5', linewidth=1, alpha=.75, label='Each optimizer update')
axes[0].plot(steps, rolling, color='#155e75', linewidth=2.6, label='20-update moving average')
axes[0].set(ylabel='Logged training reward', ylim=(-.03, 1.05), title='Training continues; reward is noisy and has declined from its earlier window')
axes[0].legend(loc='upper right', frameon=False, ncol=2)
axes[1].plot(steps, [r['grad_norm'] for r in rows], color='#8b5cf6', linewidth=1.3)
axes[1].set(xlabel='Optimizer step', ylabel='Gradient norm', title='Finite gradient updates continue (norm shown before clipping)')
for ax in axes:
    ax.grid(axis='y', alpha=.2)
    ax.axvline(196, color='#64748b', linestyle='--', linewidth=1)
    ax.set_xlim(54, steps[-1])
axes[0].text(197, .76, 'Resume on hopper-prod', fontsize=9, color='#475569', va='top')
fig.text(.08, .015, f"Snapshot {report['snapshot_utc'][:16].replace('T', ' ')} UTC. Bounded-admission recipe from step 54. "
         'Reward averages are not fixed-task pass@1; task and harness mix varies.', fontsize=9, color='#475569')
fig.tight_layout(rect=(0, .04, 1, 1))
for suffix in ('png', 'svg', 'pdf'):
    fig.savefig(ROOT / f'training_reward.{suffix}', dpi=170)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10.5, 5))
colors = ['#0284c7', '#d97706', '#7c3aed', '#059669']
labels = ['OpenCode', 'Claude Code', 'Codex', 'Mini-SWE-Agent']
harnesses = ['opencode', 'claude-code', 'codex', 'mini-swe-agent']
for h, label, color in zip(harnesses, labels, colors):
    ys = [report['evaluations'][str(k) if k else 'base']['harnesses'][h]['pass_at_1'] for k in (0, 100, 200)]
    ax.plot([0, 100], ys[:2], '-o', color=color, linewidth=2, label=label)
    ax.plot([100, 200], ys[1:], '--', color=color, linewidth=1.5)
    ax.scatter([200], [ys[2]], facecolor='white', edgecolor=color, linewidth=2, s=70, zorder=3)
    offset = -9 if h == 'opencode' else 9 if h == 'claude-code' else 0
    ax.annotate(f'{ys[2]:.1%}', (200, ys[2]), xytext=(10, offset), textcoords='offset points', va='center', color=color)
ax.set(xticks=[0, 100, 200], xticklabels=['Base', 'Checkpoint 100', 'Checkpoint 200*'], ylim=(0, .38), xlim=(-10, 225),
       ylabel='Pass@1 on graded cells', title='Fixed test evaluation by harness — checkpoint 200 is provisional')
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.grid(axis='y', alpha=.2)
ax.legend(loc='upper left', ncol=2, frameon=False)
n = report['evaluations']['200']['graded']
fig.text(.1, .015, f'* {n}/1,000 cells graded; incomplete denominators can bias the provisional average. '
         'See report for matched-cell comparison and counts.', fontsize=9, color='#475569')
fig.tight_layout(rect=(0, .05, 1, 1))
for suffix in ('png', 'svg', 'pdf'):
    fig.savefig(ROOT / f'evaluation_comparison.{suffix}', dpi=170)
print('Generated training reward/gradient and evaluation plots from the frozen snapshot.')

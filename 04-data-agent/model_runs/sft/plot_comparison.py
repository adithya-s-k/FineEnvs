"""Refresh local SFT/RL comparison when a verified mixed-harness score arrives."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'opencode-student-sft-20260927'
prior=json.loads((OLD/'latest_rl_comparison.json').read_text())
source=Path(prior['directory'])/'comparison.json'
base=json.loads(source.read_text())
rows={}
receipts=[]
for model,old in base['models'].items():
    rows[model]=[{**r,'label':r['label'].replace('SFT epoch','OpenCode SFT epoch')} for r in old[:3]]
    prep=ROOT/model/'data/preparation.json'
    steps=[2242,4484]
    if prep.exists():
        n=json.loads(prep.read_text())['training_examples'];steps=[(n+7)//8,2*((n+7)//8)]
    for epoch,step in enumerate(steps,1):
        path=ROOT/model/'production/evaluations'/f'checkpoint-{step}/evaluation_verified.json'
        row={'label':f'Multi-harness SFT epoch {epoch}','step':step,'score':None,'graded':0,'source':str(path)}
        if path.exists():
            record=json.loads(path.read_text());score=record['score']
            assert score['complete'] and score['graded_cells']==1000 and record['strictly_held_out']
            row.update(score=score['average_pass_at_1'],graded=score['graded_cells'],harnesses={h:v['pass_at_1'] for h,v in score['harnesses'].items()})
            receipts.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        rows[model].append(row)
    rows[model].extend(old[3:])
signature=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
latest=ROOT/'latest_comparison.json'
if latest.exists() and json.loads(latest.read_text()).get('signature')==signature:
    raise SystemExit(0)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
stamp=datetime.now(timezone.utc)
out=ROOT/'plots'/stamp.strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
(out/'comparison.json').write_text(json.dumps({'snapshot_utc':stamp.isoformat(),'models':rows,'source_comparison':str(source),'new_sources':receipts,'selection_note':'RL is the best complete observed checkpoint by overall score; training datasets and budgets differ.'},indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#D4DDE7','text.color':'#142338'})
colors=['#98A2B3','#6AA3E0','#2765AD','#58B69D','#087F69','#E9A33B','#C96518']
fig,axes=plt.subplots(1,2,figsize=(17,8.5));fig.subplots_adjust(left=.19,right=.96,top=.80,bottom=.19,wspace=.65)
fig.text(.07,.94,'Multi-harness SFT | Comparison with OpenCode SFT and RL',fontsize=23,weight='bold')
fig.text(.07,.89,'Same 250-task evaluation set × 4 harnesses · Pass@1 · Only complete checkpoint evaluations enter the scores',fontsize=12,color='#52647C')
for ax,(model,rr) in zip(axes,rows.items()):
    for i,r in enumerate(rr):
        if r['score'] is None:
            ax.text(1,i,'Pending evaluation',va='center',color='#94A3B8',fontsize=10)
            continue
        ax.barh(i,100*r['score'],height=.62,color=colors[i],hatch='///' if r['graded']<1000 else None,edgecolor='white')
        ax.text(100*r['score']+1,i,f"{100*r['score']:.1f}%"+('*' if r['graded']<1000 else ''),va='center',weight='bold')
    labels=[r['label']+(f" (step {r['step']})" if r['label'].startswith('RL ') else '') for r in rr]
    ax.set_yticks(range(7),labels);ax.invert_yaxis();ax.set_xlim(0,65);ax.set_xlabel('Pass@1 (%)');ax.set_axisbelow(True);ax.grid(axis='x',alpha=.15);ax.tick_params(length=0)
    ax.set_title('Qwen3.5-2B' if model=='qwen' else 'LFM2.5-2.6B',loc='left',fontsize=18,weight='bold',pad=18)
fig.text(.07,.12,'OpenCode SFT: 801 rollouts / 801 tasks. Multi-harness SFT: 3,189 rollouts / 888 tasks. Both train for two epochs.',fontsize=10,color='#52647C')
fig.text(.07,.083,'LFM RL uses 1,000 medium/hard tasks; Qwen RL uses different training data. Best RL is retrospective checkpoint selection.',fontsize=10,color='#52647C')
fig.text(.07,.046,'* LFM baseline: 998/1,000 graded. Other shown scores: 1,000/1,000. '+stamp.strftime('%d %b %Y %H:%M UTC'),fontsize=10,color='#52647C')
for ext in ['png','svg']:fig.savefig(out/f'comparison_overall.{ext}',dpi=180)
plt.close(fig)
H=['opencode','claude-code','codex','mini-swe-agent']
fig,axes=plt.subplots(1,2,figsize=(17,9));fig.subplots_adjust(left=.21,right=.97,top=.80,bottom=.19,wspace=.75)
fig.text(.07,.94,'Evaluation harness breakdown',fontsize=25,weight='bold')
fig.text(.07,.89,'Each RL row uses one checkpoint selected by overall score; blank rows are awaiting evaluation',fontsize=12,color='#52647C')
for ax,(model,rr) in zip(axes,rows.items()):
    mat=np.array([[100*r['harnesses'][h] for h in H] if r['score'] is not None else [np.nan]*4 for r in rr])
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#F1F5F9')
    ax.imshow(mat,cmap=cmap,vmin=0,vmax=75,aspect='auto')
    ax.set_yticks(range(7),[r['label']+(f"\nstep {r['step']}" if r['label'].startswith('RL ') else '') for r in rr])
    ax.set_xticks(range(4),['OpenCode','Claude\nCode','Codex','Mini-SWE\nAgent']);ax.tick_params(length=0,pad=7)
    ax.set_title('Qwen3.5-2B' if model=='qwen' else 'LFM2.5-2.6B',loc='left',fontsize=18,weight='bold',pad=18)
    for i in range(7):
        for j in range(4):
            v=mat[i,j]
            ax.text(j,i,'pending' if np.isnan(v) else f'{v:.1f}%',ha='center',va='center',fontsize=10,color=('#94A3B8' if np.isnan(v) else 'white' if v>43 else '#142338'))
fig.text(.07,.10,'Training datasets and budgets differ. LFM baseline Mini-SWE-Agent has 248/250 grades; all other populated cells have 250/250.',fontsize=10,color='#52647C')
for ext in ['png','svg']:fig.savefig(out/f'comparison_harnesses.{ext}',dpi=180)
plt.close(fig)
temp=ROOT/('latest_comparison.'+stamp.strftime('%H%M%S%f')+'.tmp')
temp.write_text(json.dumps({'directory':str(out),'signature':signature,'updated_utc':stamp.isoformat()},indent=2)+'\n');temp.replace(latest)
print(out)

"""Freeze shared tasks and paired harness schedules for the two LFM arms."""
import hashlib,json,random,shutil,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'experiments/lfm25-medium-hard1000-20260921'
SOURCE=ROOT/'experiments/async_grpo_harbor_data_agent/logs/multi4-long-rotation-20260914'
BASE=ROOT/'HuggingEnvs/04-data-agent/model_runs/lfm25/config.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 import sys
 sys.path.insert(0,str(ROOT/'experiments/async_grpo_harbor_data_agent/tools'))
 from prepare_multi4_run import prepare,question
 source=Path('/admin/home/adithyaskolavi/.cache/openenv/harbor-datasets/HuggingEnvs__data-agent-harbor-train')
 baseline=ROOT/'experiments/async_grpo_harbor_data_agent/logs/multi4-baseline-20260914'
 test=json.loads((baseline/'manifest.json').read_text());test_ids={t['source'] for t in test['tasks']};test_books={t['notebook'] for t in test['tasks']}
 test_questions=set();test_instructions=set()
 for t in test['tasks']:
  d=baseline/'dataset/tasks'/t['name'];test_questions.add(question(tomllib.loads((d/'task.toml').read_text())));test_instructions.add(' '.join((d/'instruction.md').read_text().split()))
 pool={'medium':[],'hard':[]};seen=set()
 for d in sorted((source/'tasks').iterdir()):
  conf=tomllib.loads((d/'task.toml').read_text());m=conf['metadata'];sid=m['source_row_id'];tier=m['difficulty_tier'];inst=' '.join((d/'instruction.md').read_text().split())
  if tier not in pool or sid in test_ids or sid.split('_qa_')[0] in test_books or question(conf) in test_questions or inst in test_instructions or inst in seen:continue
  seen.add(inst);pool[tier].append({'name':d.name,'source':sid,'notebook':sid.split('_qa_')[0],'difficulty':tier,'normalized_instruction_sha256':hashlib.sha256(inst.encode()).hexdigest()})
 rng=random.Random(42);selected=rng.sample(pool['medium'],400)+rng.sample(pool['hard'],600);rng.shuffle(selected)
 selection=OUT.parent/'lfm25-medium-hard1000-selection-20260921.json'
 write(selection,{'source_dataset':'FineEnvs/data-agent-harbor-train','cached_source_root':str(source),'selection_seed':42,'easy_start_task_count':0,'tasks':selected,'eligible_counts':{k:len(v) for k,v in pool.items()}})
 if not OUT.exists():prepare(selection,baseline,OUT,1000)
 manifest=json.loads((OUT/'manifest.json').read_text())
 if [t['name'] for t in manifest['tasks']] != [t['name'] for t in selected]:raise ValueError('Existing selection differs')
 selected=manifest['tasks']
 for t in selected:
  for rel,digest in t['file_hashes'].items():assert sha(OUT/'dataset'/rel)==digest
 names=sorted(t['name'] for t in selected);indices={n:i for i,n in enumerate(names)}
 tasks=[{'name':t['name'],'task_index':indices[t['name']],'difficulty':t['difficulty']} for t in selected]
 orders=[]
 for epoch in range(2):
  order=list(range(1000));random.Random(42+epoch).shuffle(order);orders.append(order)
 write(OUT/'manifest.json',{**manifest,'tasks':selected,'task_count':1000,'difficulty_counts':{'medium':400,'hard':600},'selection_seed':42,'source_manifest_sha256':sha(SOURCE/'manifest.json'),'test_overlap':{'source':0,'notebook':0,'instruction':0},'epochs':2,'status':'prepared; not launched'})
 for arm,harnesses in [('opencode',['opencode']),('multi-harness',['opencode','claude-code','codex','mini-swe-agent'])]:
  d=OUT/arm;d.mkdir(parents=True,exist_ok=True);groups=[]
  assignments={}
  for tier in ['medium','hard']:
   for i,row in enumerate(j for j,t in enumerate(tasks) if t['difficulty']==tier):assignments[row]=i%len(harnesses)
  for epoch,order in enumerate(orders):
   for row in order:
    t=tasks[row];groups.append({'group_in_cycle':len(groups),'pass_index':epoch,'task_row':row,'task_index':t['task_index'],'task_name':t['name'],'difficulty':t['difficulty'],'harness':harnesses[(assignments[row]+epoch)%len(harnesses)]})
  schedule={'schema_version':1,'mode':'one_harness_per_task_per_pass','seed':42,'harnesses':harnesses,'tasks':tasks,'task_count':1000,'groups_per_pass':1000,'passes_per_cycle':2,'groups_per_cycle':2000,'easy_start_task_count':0,'groups':groups}
  write(d/'harness_schedule.json',schedule);(d/'indices.txt').write_text(','.join(str(t['task_index']) for t in tasks)+'\n')
  c=json.loads(BASE.read_text());c['run_name']='LFM 2.6B · '+('OpenCode' if arm=='opencode' else 'Multi-harness')+' · Medium+hard 1000'
  c['training_harnesses']=harnesses;c['dataset']={'split':str(OUT/'dataset'),'task_count':1000,'difficulty_counts':{'medium':400,'hard':600},'passes':2,'group_limit':2000,'schedule_file':str(d/'harness_schedule.json'),'schedule_sha256':sha(d/'harness_schedule.json'),'manifest_sha256':sha(OUT/'manifest.json')}
  c['logging'].update(project='data-agent-rl-comparison',space_id='FineEnvs/data-agent-training-comparison-trackio')
  c['reward']={'implementation':'Scoped rollout callback; native ATIF action count, checked against captured actions','efficiency_weight':0.1,'tool_budget':15,'formula':'correctness * (1 + 0.1 * 15/(15+calls))','unknown_count':'no bonus; counted separately','eval_reward':'binary correctness only','qualification_required':True}
  c['readiness']={'dataset':'verified','optimizer_and_action_count_smokes':'required for this changed configuration','launch_authorized':False,'scheduler':'explicit two-pass adapter; unit tests pass','reward_callback':'wired; native count >= retained captured count, unknown count gets no bonus'}
  write(d/'config.json',c)
 print(OUT)
if __name__=='__main__':main()

"""Corroborate captured tool actions using Harbor's native ATIF artifact."""
import json,os
from pathlib import Path
from reward import shaped_reward

def native_count(path):
    d=json.loads(Path(path).read_text())
    if not str(d.get('schema_version','')).startswith('ATIF-'):raise ValueError('Unknown trajectory schema')
    calls=[c for s in d['steps'] if s.get('source')=='agent' for c in s.get('tool_calls',[]) or []]
    ids=[c.get('tool_call_id') for c in calls]
    if any(not x for x in ids) or len(set(ids))!=len(ids):raise ValueError('Ambiguous action identities')
    return len(calls)

def reward(outcome):
    metadata=outcome.trace[0].get('efficiency_evidence',{}) if outcome.trace else {}
    count=metadata.get('native_actions');verified=count is not None and count>=outcome.tool_call_count
    result=shaped_reward(outcome.env_reward,count,count_verified=verified)
    record={**metadata,'correctness':outcome.env_reward,'captured_actions':outcome.tool_call_count,'count_verified':verified,'reward':result,'bonus':None if result is None else result-outcome.env_reward}
    path=Path(os.environ['CURRICULUM_AUDIT'])/'efficiency'/f"{metadata.get('episode_id','unknown')}.json"
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record,indent=2)+'\n')
    return result

def install_audit():
    import training_audit
    original=training_audit.AuditedSession
    def fetch(self):
        trace=self.session.fetch_proxy_trace();result=self.session.result
        evidence={'episode_id':self.metadata['episode_id'],'harness':self.metadata['harness'],'group_id':self.metadata['group_id'],'native_actions':None}
        if result is not None:
            path=Path(os.environ['LFM_TRIALS_DIR'])/result.trial_name/'agent/trajectory.json'
            evidence['trajectory']=str(path)
            try:evidence['native_actions']=native_count(path)
            except (OSError,ValueError,KeyError,TypeError) as exc:evidence['count_error']=type(exc).__name__
        if trace:trace[0]={**trace[0],'efficiency_evidence':evidence}
        return trace
    original.fetch_proxy_trace=fetch

def worker_class():
    return EfficiencyWorker

from hard_curriculum_train import FiniteWorker
class EfficiencyWorker(FiniteWorker):
    def __init__(self,**kwargs):
        super().__init__(rollout_reward_fn=reward,**kwargs)

# Spawned rollout workers import this callback module independently of the trainer entry point.
install_audit()

from production import install_resume_policy
install_resume_policy()

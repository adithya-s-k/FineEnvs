import json
from collections import Counter
from pathlib import Path
import pytest
from reward import shaped_reward
ROOT=Path(__file__).resolve().parents[4]/'experiments/lfm25-medium-hard1000-20260921'
def test_shared_tasks_and_equal_exposure():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert Counter(t['difficulty'] for t in manifest['tasks']) == {'hard':600,'medium':400}
    assert len({t['name'] for t in manifest['tasks']})==1000
    a=json.loads((ROOT/'opencode/harness_schedule.json').read_text());b=json.loads((ROOT/'multi-harness/harness_schedule.json').read_text())
    assert [(g['task_name'],g['pass_index']) for g in a['groups']]==[(g['task_name'],g['pass_index']) for g in b['groups']]
    for schedule in [a,b]:
        assert len(schedule['groups'])==2000
        assert Counter(g['task_name'] for g in schedule['groups'])=={t['name']:2 for t in manifest['tasks']}
    assert Counter(g['harness'] for g in b['groups'])=={h:500 for h in b['harnesses']}
def test_reward_has_no_failure_or_missing_count_bonus():
    assert shaped_reward(None,1,count_verified=True) is None
    for calls in [0,1,15,100]:assert shaped_reward(0,calls,count_verified=True)==0
    assert shaped_reward(1,0,count_verified=True)==1
    assert shaped_reward(1,None)==1
    assert shaped_reward(1,1)==1
    assert 1 < shaped_reward(1,30,count_verified=True) < shaped_reward(1,15,count_verified=True) < shaped_reward(1,1,count_verified=True) <= 1.1
    with pytest.raises(ValueError):shaped_reward(1,-1,count_verified=True)
    with pytest.raises(ValueError):shaped_reward(.5,1,count_verified=True)

from schedule import validate_schedule
def test_schedule_runtime_validator():
    for arm in ['opencode','multi-harness']:
        s=json.loads((ROOT/arm/'harness_schedule.json').read_text())
        validate_schedule(s)
        s['groups'][0]['task_name']='wrong'
        with pytest.raises(ValueError):validate_schedule(s)

from copy import deepcopy
from types import SimpleNamespace

from harness_schedule import make_schedule, validate_schedule
from prepare_harbor_opencode_run import single_harness_schedule


def test_single_harness_preserves_reference_order_without_reshuffling():
    tasks=[{'name':f'task-{i}','task_index':999-i,'difficulty':tier}
           for i,tier in enumerate(['easy']*150+['medium']*600+['hard']*250)]
    reference=make_schedule(tasks,['opencode','claude-code','codex','mini-swe-agent'])
    before=deepcopy(reference)
    actual=single_harness_schedule(reference)
    assert reference==before
    assert actual['harnesses']==['opencode']
    assert actual['groups_per_cycle']==1000 and actual['passes_per_cycle']==1
    assert [g['task_index'] for g in actual['groups']]==[g['task_index'] for g in reference['groups'][:1000]]
    assert {g['harness'] for g in actual['groups']}=={'opencode'}
    validate_schedule(actual)


def test_all_generations_route_through_opencode_across_cycle_boundary():
    from multi_harness import pair_rows,MultiHarborSessionFactory
    tasks=[{'name':f'task-{i}','task_index':i,'difficulty':'easy'} for i in range(40)]
    schedule=single_harness_schedule(make_schedule(tasks,['opencode','claude-code','codex','mini-swe-agent']))
    factory=SimpleNamespace(schedule=schedule,harnesses=['opencode'],group_offset=0,
        prompt_rows=lambda:[{'task_name':t['name'],'task_index':t['task_index']} for t in tasks])
    rows=pair_rows(factory)
    for group in range(85):
        for generation in range(8):
            assert MultiHarborSessionFactory.harness_for(factory,group)=='opencode'
            assert rows[group%40]['task_index']==schedule['groups'][group%40]['task_index']

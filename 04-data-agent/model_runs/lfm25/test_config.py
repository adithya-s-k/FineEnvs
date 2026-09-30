import json
from pathlib import Path


def test_same_tasks_hyperparameters_and_cadence():
    here=Path(__file__).parent
    c=json.loads((here/'config.json').read_text())
    assert c['dataset']['task_count']==1000
    assert c['dataset']['difficulty_counts']=={'easy':150,'medium':600,'hard':250}
    assert c['training']['learning_rate']==3e-6
    assert c['training']['num_generations']==8
    assert c['training']['max_staleness']==4
    assert c['training']['max_inflight']==32
    assert c['training']['save_steps']==50
    assert c['evaluation']['interval_optimizer_steps']==100
    assert len(c['harnesses'])==4


def test_checkpoint_eval_cadence():
    from controller import due_steps
    assert due_steps([0,2,4,50,100,150,200,950,1000,1050]) == [100,200,1000]

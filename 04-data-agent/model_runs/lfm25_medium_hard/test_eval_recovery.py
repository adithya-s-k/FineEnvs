import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from eval_recovery import stage, install


def test_preserves_results_and_rejects_checkpoint_mix(tmp_path):
    source=tmp_path/'eval-1';output=tmp_path/'eval-2'
    for p in [source,output]:
        p.mkdir();(p/'services.json').write_text(json.dumps({'model':'checkpoint400','revision':'pinned'}))
    (source/'traces').mkdir();(source/'traces/result.jsonl').write_text('{"reward":1}\n')
    (source/'trials/trial1').mkdir(parents=True)
    stage(source,output)
    assert (output/'traces/result.jsonl').read_bytes()==(source/'traces/result.jsonl').read_bytes()
    assert (output/'trials/trial1').resolve()==(source/'trials/trial1').resolve()
    other=tmp_path/'eval-3';other.mkdir();(other/'services.json').write_text(json.dumps({'model':'different','revision':'pinned'}))
    with pytest.raises(ValueError,match='another checkpoint'):stage(source,other)


def test_preserves_loopback_endpoints_without_changing_job_identity(monkeypatch,tmp_path):
    import os
    monkeypatch.setenv('LFM_EVAL_RESUME_FROM',str(tmp_path/'eval-83980'))
    monkeypatch.setenv('SLURM_JOB_ID','84030')
    base=SimpleNamespace(services=lambda *args:os.environ['SLURM_JOB_ID'],evaluate=lambda *args:None)
    install(base)
    assert base.services(tmp_path,'checkpoint400',False)=='83980'
    assert os.environ['SLURM_JOB_ID']=='84030'

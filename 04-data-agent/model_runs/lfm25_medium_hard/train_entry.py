"""Scoped scheduler/reward overlay; immutable qualified sources remain unchanged."""
import os,sys,runpy
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import harness_schedule
from schedule import validate_schedule
harness_schedule.validate_schedule=validate_schedule
from efficiency import install_audit,reward
install_audit()
import hard_curriculum_train
from production import install_resume_policy
install_resume_policy()
from efficiency import EfficiencyWorker
hard_curriculum_train.FiniteWorker=EfficiencyWorker
if __name__=='__main__':
    sys.path.insert(0,str(HERE.parent/'lfm25'))
    runpy.run_path(str(HERE.parent/'lfm25/train_entry.py'),run_name='__main__')

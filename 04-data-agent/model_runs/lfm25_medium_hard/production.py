"""Production admission and checkpoint-resume checks."""
import json
from pathlib import Path
from hashlib import sha256


def check_qualification(root, code):
    for folder in sorted(Path(root).glob('train-smoke-*'), reverse=True):
        proof = folder / 'qualification_passed.json'
        if not proof.exists():
            continue
        record = json.loads(proof.read_text())
        expected = {'config_sha256': Path(root) / 'config.json',
                    'schedule_sha256': Path(root) / 'harness_schedule.json',
                    'reward_sha256': Path(code) / 'reward.py'}
        if all(record.get(k) == sha256(p.read_bytes()).hexdigest() for k, p in expected.items()):
            if record.get('optimizer_save_resume') and record.get('reloaded_checkpoint_eval'):
                return str(folder)
    raise RuntimeError('No matching passed GPU qualification')


def committed_state(state):
    """Retry uncommitted groups; never repeat an already admitted group."""
    result = dict(state)
    committed = set(state.get('admitted_rollouts', {}).values())
    result['settled_groups'] = sorted(committed)
    result['retry_uncommitted_groups'] = sorted(set(state.get('settled_groups', [])) - committed)
    result['resume_policy'] = ('Retry groups with no committed optimizer work. '
                               'Do not replay committed groups; incomplete committed tails are reported.')
    return result


def install_resume_policy():
    import hard_curriculum_train as finite
    if getattr(finite, "_lfm_resume_policy", False):
        return
    finite._lfm_resume_policy = True
    original_pending = finite.pending_groups
    original_save = finite.FiniteTrainer._save_checkpoint

    def pending(limit, state):
        return original_pending(limit, committed_state(state))

    def save(self, model, trial):
        original_save(self, model, trial)
        checkpoint = Path(self.args.output_dir) / f'checkpoint-{self.state.global_step}'
        path = checkpoint / 'curriculum_state.json'
        finite.write_json(path, committed_state(json.loads(path.read_text())))

    finite.pending_groups = pending
    finite.FiniteTrainer._save_checkpoint = save

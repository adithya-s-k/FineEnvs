"""Publish completed full-model checkpoints and stage read-only evaluation inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

READY = 'checkpoint.ready.json'
SAVED = 'checkpoint.saved.json'
REQUIRED = ('config.json', 'trainer_state.json', 'tokenizer.json', 'tokenizer_config.json',
            'training_args.bin', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth')
METADATA = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
            'preprocessor_config.json', 'video_preprocessor_config.json', 'processor_config.json', 'chat_template.json',
            'chat_template.jinja', 'special_tokens_map.json', 'vocab.json', 'merges.txt',
            'added_tokens.json', 'model.safetensors.index.json')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def model_files(checkpoint):
    from safetensors import safe_open
    index = checkpoint / 'model.safetensors.index.json'
    weight_map = json.loads(index.read_text())['weight_map'] if index.exists() else None
    names = set(weight_map.values()) if weight_map else {'model.safetensors'}
    if not names or any(Path(n).name != n or not n.endswith('.safetensors') for n in names):
        raise ValueError('Invalid checkpoint shard index')
    all_keys = set()
    for name in sorted(names):
        with safe_open(checkpoint / name, framework='numpy') as tensors:
            keys = set(tensors.keys())
            if not keys or all_keys.intersection(keys):
                raise ValueError('Empty shard or duplicated tensor keys')
            if weight_map and keys != {k for k, v in weight_map.items() if v == name}:
                raise ValueError(f'Shard contents disagree with index: {name}')
            all_keys.update(keys)
    return sorted(names)


def mark_saved(checkpoint, step, base_model, base_revision, *, final=False):
    """Publish a small handoff after save; leave weight hashing to the CPU watcher."""
    checkpoint = Path(checkpoint).resolve()
    for name in REQUIRED:
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise ValueError(f'Incomplete checkpoint: {name}')
    if json.loads((checkpoint / 'trainer_state.json').read_text())['global_step'] != step:
        raise ValueError('Checkpoint step disagrees with trainer state')
    names = set(REQUIRED) | set(model_files(checkpoint))
    names.update(n for n in METADATA if (checkpoint / n).exists())
    marker = {'schema_version': 1, 'checkpoint': str(checkpoint), 'step': step, 'final': final,
              'base_model': base_model, 'base_revision': base_revision,
              'file_stats': {n: [(checkpoint / n).stat().st_size, (checkpoint / n).stat().st_mtime_ns]
                             for n in sorted(names)}}
    write_json(checkpoint / SAVED, marker)
    return marker


def finalize_saved(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    marker = json.loads((checkpoint / SAVED).read_text())
    if marker['checkpoint'] != str(checkpoint):
        raise ValueError('Saved checkpoint path mismatch')
    for name, expected in marker['file_stats'].items():
        if Path(name).name != name:
            raise ValueError('Invalid saved checkpoint filename')
        stat = (checkpoint / name).stat()
        if [stat.st_size, stat.st_mtime_ns] != expected:
            raise ValueError(f'Checkpoint changed after save: {name}')
    return mark_ready(checkpoint, marker['step'], marker['base_model'], marker['base_revision'],
                      final=marker.get('final', False))


def mark_ready(checkpoint, step, base_model, base_revision, *, final=False):
    checkpoint = Path(checkpoint).resolve()
    for name in REQUIRED:
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise ValueError(f'Incomplete checkpoint: {name}')
    if json.loads((checkpoint / 'trainer_state.json').read_text())['global_step'] != step:
        raise ValueError('Checkpoint step disagrees with trainer state')
    names = model_files(checkpoint) + [n for n in METADATA if (checkpoint / n).exists()]
    names = sorted(set(names + ['trainer_state.json']))
    before = {n: ((checkpoint / n).stat().st_size, (checkpoint / n).stat().st_mtime_ns) for n in names}
    hashes = {n: digest(checkpoint / n) for n in names}
    after = {n: ((checkpoint / n).stat().st_size, (checkpoint / n).stat().st_mtime_ns) for n in names}
    if before != after:
        raise ValueError('Checkpoint changed during finalization')
    marker = {'schema_version': 1, 'step': step, 'checkpoint': str(checkpoint), 'final': final,
              'base_model': base_model, 'base_revision': base_revision,
              'files': hashes, 'file_stats': after, 'training_state_present': list(REQUIRED)}
    write_json(checkpoint / READY, marker)
    return marker


def verify_ready(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    marker = json.loads((checkpoint / READY).read_text())
    if marker['checkpoint'] != str(checkpoint):
        raise ValueError('Checkpoint path differs from its completion marker')
    for name, expected in marker['files'].items():
        if Path(name).name != name or digest(checkpoint / name) != expected:
            raise ValueError(f'Checkpoint changed after completion: {name}')
    model_files(checkpoint)
    return marker


def resume_info(checkpoint, base_model, base_revision):
    """Validate a completed local Trainer checkpoint, including its rollout cursor."""
    checkpoint = Path(checkpoint).resolve()
    path = checkpoint / SAVED if (checkpoint / SAVED).exists() else checkpoint / READY
    if not path.is_file():
        raise ValueError('Resume requires a completed checkpoint marker')
    marker = json.loads(path.read_text())
    if (marker['checkpoint'] != str(checkpoint) or marker['base_model'] != base_model
            or marker['base_revision'] != base_revision):
        raise ValueError('Resume checkpoint path or base model/revision differs')
    for name in REQUIRED + ('rollout_state.json',):
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise ValueError(f'Incomplete resume checkpoint: {name}')
    for name, expected in marker['file_stats'].items():
        if Path(name).name != name:
            raise ValueError('Invalid checkpoint filename')
        stat = (checkpoint / name).stat()
        if [stat.st_size, stat.st_mtime_ns] != expected:
            raise ValueError(f'Resume checkpoint changed after save: {name}')
    model_files(checkpoint)
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    rollout = json.loads((checkpoint / 'rollout_state.json').read_text())
    if state['global_step'] != marker['step'] or marker['step'] <= 0:
        raise ValueError('Resume step disagrees with completion marker')
    if any(type(rollout.get(k)) is not int or rollout[k] < 0 for k in ('prompt_index', 'model_version')):
        raise ValueError('Invalid rollout cursor/model version')
    return {'checkpoint': str(checkpoint), 'step': marker['step'],
            'group_offset': rollout['prompt_index'], 'model_version': rollout['model_version'],
            'rollout_state_sha256': digest(checkpoint / 'rollout_state.json')}


def stage_model(checkpoint, target, base_metadata):
    checkpoint, target, base_metadata = map(Path, (checkpoint, target, base_metadata))
    marker = verify_ready(checkpoint)
    target.mkdir(parents=True, exist_ok=False)
    for name in model_files(checkpoint):
        (target / name).symlink_to((checkpoint / name).resolve())
    origins = {}
    for name in METADATA:
        source = checkpoint / name
        if not source.exists():
            if name == 'config.json' or name == 'model.safetensors.index.json':
                continue
            source = base_metadata / name
        if source.exists():
            shutil.copy2(source, target / name)
            origins[name] = str(source.resolve())
    assert (target / 'config.json').read_bytes() == (checkpoint / 'config.json').read_bytes()
    record = {'checkpoint': marker, 'metadata_origins': origins,
              'files': {p.name: digest(p) for p in target.iterdir() if p.is_file()}}
    write_json(target / 'checkpoint_source.json', record)
    return record


def verify_stage(target):
    target = Path(target)
    record = json.loads((target / 'checkpoint_source.json').read_text())
    for name, expected in record['files'].items():
        if Path(name).name != name or digest(target / name) != expected:
            raise ValueError(f'Staged checkpoint changed: {name}')
    model_files(target)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify'])
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    record = verify_stage(args.directory)
    print(f"Verified checkpoint step {record['checkpoint']['step']}: {args.directory}")

"""Reload an SFT checkpoint into the already-qualified four-harness evaluator."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import shutil
from datetime import datetime, timezone

from eval_policy import environment_capacity
from local_logging import configure as configure_local_logging, log_evaluation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--concurrency', type=int, default=100)
    args = p.parse_args()
    root, output = args.workspace.resolve(), args.output.resolve()
    archive = None
    resume_manifest = False
    if args.resume:
        assert output == args.checkpoint.resolve().parent.parent / 'evaluations' / args.checkpoint.name
        assert not (output / 'evaluation_verified.json').exists()
        resume_manifest = (output / 'traces/eval_config.json').exists()
        if not resume_manifest:
            assert not any(p.stat().st_size for p in (output / 'traces').glob('*.jsonl')), 'Trace data exists without its configuration; manual recovery required'
        archive = output / ('recovery-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        archive.mkdir()
        for artifact in output.iterdir():
            if artifact.is_file():
                shutil.move(str(artifact), str(archive / artifact.name))
        if resume_manifest:
            shutil.copy2(output / 'traces/eval_config.json', archive / 'eval_config.json')
    else:
        output.mkdir(parents=True, exist_ok=False)
    run_root = args.checkpoint.resolve().parent.parent
    configure_local_logging(run_root)
    model_runs = root / 'HuggingEnvs/04-data-agent/model_runs'
    sys.path.insert(0, str(root / 'HuggingEnvs/04-data-agent/train'))
    sys.path.insert(0, str(model_runs / 'lfm25_medium_hard'))
    from dns_fallback import install
    install()
    spec = importlib.util.spec_from_file_location('qualified_lfm_eval', model_runs / 'lfm25/run.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    config = json.loads((root / 'experiments/lfm25-medium-hard1000-20260921/opencode/config.json').read_text())
    training_config = json.loads((run_root / 'config.json').read_text())
    config['model'] = training_config['model']
    config['model_revision'] = training_config['revision']
    qwen = config['model'] == 'Qwen/Qwen3.5-2B'
    from checkpoint_artifacts import finalize_saved
    if not (args.checkpoint / 'checkpoint.ready.json').exists():
        finalize_saved(args.checkpoint)
    base.C = config
    base.TRAIN_SPLIT = root / 'experiments/lfm25-medium-hard1000-20260921/dataset'
    original_start = base.rt.start

    def start(command, log, env=None):
        command = list(command)
        if qwen and '--tool-call-parser' in command:
            command[command.index('--tool-call-parser') + 1] = 'qwen3_xml'
            command[command.index('--default-chat-template-kwargs') + 1] = '{"enable_thinking":false}'
            command += ['--limit-mm-per-prompt', '{"image":0,"video":0}', '--gdn-prefill-backend', 'triton']
        if '--expose' in command:
            command[command.index('--expose') + 1] = 'cloudflare'
            env = {**(os.environ if env is None else env), 'TUNNEL_TRANSPORT_PROTOCOL': 'http2',
                   'TUNNEL_LOGFILE': str(Path(log).with_suffix('.tunnel.log')),
                   'MAX_CONCURRENT_ENVS': str(environment_capacity(4 if args.smoke else args.concurrency))}
        return original_start(command, log, env)

    base.rt.start = start
    original_command = base.eval_command

    def eval_command(folder, engine, server, smoke):
        command = original_command(folder, engine, server, smoke)
        concurrency = 4 if smoke else args.concurrency
        for key in ['--concurrency', '--server-concurrency', '--sandbox-concurrency']:
            command[command.index(key) + 1] = str(concurrency)
        return command

    base.eval_command = eval_command
    processes = []
    try:
        processes, engine, server = base.services(output, str(args.checkpoint.resolve()), False)
        if resume_manifest:
            manifest_path = output / 'traces/eval_config.json'
            previous = json.loads((archive / 'services.json').read_text())
            assert previous['model'] == config['model'] and previous['revision'] == config['model_revision']
            original = json.loads(manifest_path.read_text())
            assert {a['harness'] for a in original['arms']} == set(config['harnesses'])
            assert all(a['model'] == config['model'] for a in original['arms'])
            rebound = json.loads(json.dumps(original))
            rebound['server'] = server
            for arm in rebound['arms']:
                arm['base_url'] = engine + '/v1'
            manifest_path.write_text(json.dumps(rebound, indent=2) + '\n')
            (archive / 'endpoint_rebind.json').write_text(json.dumps({'checkpoint': str(args.checkpoint.resolve()), 'before': original, 'after': rebound}, indent=2) + '\n')
        score = base.evaluate(output, engine, server, args.smoke)
        preparation = training_config['data_preparation']
        overlap = {key: preparation.get(key, 0) for key in ['test_task_overlap', 'test_notebook_overlap', 'test_question_overlap']}
        record = {'checkpoint': str(args.checkpoint.resolve()), 'smoke': args.smoke,
                  'heldout_tasks': 2 if args.smoke else 250, 'harnesses': config['harnesses'],
                  'pass_k': 1, 'concurrency': 4 if args.smoke else args.concurrency,
                  'environment_capacity': environment_capacity(4 if args.smoke else args.concurrency),
                  'temperature': 0.8, 'max_output_tokens': 4096, 'agent_step_limit': 17,
                  'model': config['model'], 'revision': config['model_revision'],
                  'tool_parser': 'qwen3_xml' if qwen else 'lfm2',
                  'chat_template_kwargs': {'enable_thinking': False} if qwen else {'preserve_thinking': True},
                  'test_split': str(base.TEST_SPLIT), 'score': score,
                  'training_test_overlap': overlap,
                  'strictly_held_out': not any(overlap.values()),
                  'recovery_archive': str(archive) if archive else None,
                  'protocol': 'Fixed 250-task Harbor evaluator with model-specific serving settings. Training/test overlap is disclosed separately.'}
        (output / 'evaluation_verified.json').write_text(json.dumps(record, indent=2) + '\n')
        log_evaluation(output, run_root)
        print(json.dumps(record, indent=2), flush=True)
    finally:
        base.rt.stop(processes)


if __name__ == '__main__':
    main()

"""Portable Daytona adapter, derived from the frozen 20260915 comparison backend.

Catalog indices match Harbor's sorted task directories. The manifest retains
curriculum order, which is separate from the catalog's task_index namespace.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
import os
import tempfile
import threading
import uuid
from pathlib import Path
from service_policy import admission, workload

from harbor.environments.daytona.environment import DaytonaClientManager, DaytonaEnvironment
from harbor.models.task.task import Task as HarborTask
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.verifier import Verifier
from whitebox_bash.server.sandbox import ExecResult


class LoopRunner:
    """All Daytona sessions in this server share one long-lived event loop."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True, name='daytona-whitebox-io')
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, coroutine, timeout=600):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError('Daytona operation exceeded its deadline') from None


_RUNNER = None
_LOCK = threading.Lock()


def runner():
    global _RUNNER
    with _LOCK:
        if _RUNNER is None:
            _RUNNER = LoopRunner()
        return _RUNNER


def load_frozen_tasks(split):
    from whitebox_bash.tasks import Task

    if split not in {'train', 'test'}:
        raise KeyError('Frozen comparison supports only train and test; indices never shift by difficulty.')
    root = Path(os.environ['DAYTONA_COMPARISON_RUN']).resolve()
    manifest = json.loads((root / f'{split}_manifest.json').read_text())
    result = []
    for index, row in enumerate(sorted(manifest['tasks'], key=lambda row: row['name'])):
        relative = next(k for k in row['file_hashes'] if k.endswith('/task.toml'))
        task_dir = (root / 'datasets' / split / relative).parent
        instruction = (task_dir / 'instruction.md').read_text()
        result.append(Task(
            instruction=instruction, answer='', difficulty=row['difficulty'],
            metadata={'source': 'harbor-frozen', 'native_task_dir': str(task_dir),
                      'task_index': index, 'split': split, 'source_name': row['name']},
        ))
    return tuple(result)


class DaytonaSandbox:
    """Existing bash/SETA methods backed by one native Harbor Daytona environment."""

    def __init__(self, env, task, paths):
        self.handle = env
        self.task = task
        self.paths = paths
        self.cwd = '/workdir'
        self.calls = []
        self.submitted = None
        self._closed = False
        self._io_failed = False

    @classmethod
    def start(cls, *, task, timeout_s=900, envs=None):
        role = workload(task.metadata.get('split', 'test'))
        admission.acquire(role)
        try:
            sb = cls._start_reserved(task=task, timeout_s=timeout_s, envs=envs)
            sb._role = role
            return sb
        except BaseException:
            admission.release(role)
            raise

    @classmethod
    def _start_reserved(cls, *, task, timeout_s=900, envs=None):
        if task.metadata.get('source') != 'harbor-frozen':
            raise ValueError('Daytona comparison requires the frozen Harbor task source')
        native = HarborTask(task.metadata['native_task_dir'])
        name = 'wb-' + uuid.uuid4().hex
        root = Path(os.environ['DAYTONA_WHITEBOX_TRIALS']).resolve()
        paths = TrialPaths(root / name)
        paths.mkdir()
        env = DaytonaEnvironment(
            environment_dir=native.paths.environment_dir,
            environment_name='daytona-comparison-20260915', session_id=name,
            trial_paths=paths, task_env_config=native.config.environment,
            override_cpus=1, override_memory_mb=4096, auto_snapshot=True,
            labels={'experiment': 'daytona-harness-comparison', 'run': '20260915', 'arm': 'whitebox',
                    'owner': os.environ.get('RUN_OWNER', 'local')},
        )
        sb = cls(env, native, paths)

        async def create():
            try:
                await asyncio.wait_for(env.start(force_build=False), min(timeout_s, 900))
                await env.ensure_dirs(['/logs/agent', '/logs/verifier', '/artifacts'])
                await asyncio.wait_for(env.run_healthcheck(), 300)
            except BaseException:
                await asyncio.shield(env.stop(delete=True))
                raise

        runner().call(create(), timeout_s + 310)
        (paths.trial_dir / 'session.json').write_text(json.dumps({
            'sandbox_id': sb.sandbox_id, 'task_index': task.metadata['task_index'],
            'split': task.metadata['split'], 'source_name': task.metadata['source_name'],
            'provider': 'daytona', 'state': 'active',
        }, indent=2) + '\n')
        return sb

    @property
    def sandbox_id(self):
        return self.handle._sandbox.id if self.handle._sandbox is not None else ''

    def kill(self):
        if self._closed:
            return
        runner().call(self.handle.stop(delete=True), 120)
        self._closed = True
        admission.release(self._role)
        (self.paths.trial_dir / 'cleanup.json').write_text('{"deleted": true}\n')

    def bash(self, command, timeout_s=120):
        try:
            result = runner().call(self.handle.exec(command=command, cwd=self.cwd, timeout_sec=timeout_s), timeout_s + 15)
            return ExecResult(stdout=result.stdout or '', stderr=result.stderr or '', exit_code=result.return_code)
        except Exception as exc:
            self._io_failed = True
            return ExecResult(error=f'{type(exc).__name__}: sandbox command transport failed', exit_code=1)

    def _abs(self, path):
        return path if path.startswith('/') else self.cwd + '/' + path

    def read_file(self, path):
        # Missing files and permissions are agent-visible command outcomes, not transport failures.
        import shlex
        return self.bash('cat -- ' + shlex.quote(self._abs(path)))

    def write_file(self, path, content):
        import shlex
        remote = self._abs(path)
        parent = str(Path(remote).parent)
        result = self.bash('mkdir -p -- ' + shlex.quote(parent))
        if not result.ok:
            return result
        try:
            with tempfile.TemporaryDirectory(prefix='daytona-whitebox-') as temp:
                local = Path(temp) / 'upload'
                local.write_text(content)
                runner().call(self.handle.upload_file(local, remote), 120)
            return ExecResult(stdout=f'wrote {len(content.encode())} bytes to {path}')
        except Exception as exc:
            self._io_failed = True
            return ExecResult(error=f'{type(exc).__name__}: sandbox file transport failed', exit_code=1)

    def grade(self, submitted):
        if self._io_failed:
            verdict = {'reward': None, 'note': 'sandbox transport failure; ungraded'}
        else:
            # Preserve the original answer.txt instructions. The SETA terminator is an
            # equivalent submission method; writing answer.txt directly also remains valid.
            if submitted is not None:
                wrote = self.write_file('/workdir/answer.txt', submitted)
                if not wrote.ok:
                    verdict = {'reward': None, 'note': 'submission transport failure; ungraded'}
                    (self.paths.trial_dir / 'grade.json').write_text(json.dumps(verdict) + '\n')
                    return verdict
            try:
                verifier = Verifier(task=self.task, trial_paths=self.paths, environment=self.handle)
                result = runner().call(verifier.verify(), 180)
                reward = result.rewards.get('correctness', result.rewards.get('reward'))
                if not isinstance(reward, (int, float)) or not math.isfinite(reward):
                    raise ValueError('Verifier did not return a finite correctness reward')
                if reward not in (0, 1):
                    raise ValueError('Comparison requires binary correctness rewards')
                verdict = {'reward': float(reward), 'correct': bool(reward), 'note': 'frozen Harbor verifier', 'rewards': result.rewards}
            except Exception as exc:
                verdict = {'reward': None, 'note': f'{type(exc).__name__}: verifier failed; ungraded'}
        (self.paths.trial_dir / 'grade.json').write_text(json.dumps(verdict, indent=2) + '\n')
        return verdict


def shutdown():
    if _RUNNER is not None:
        async def close():
            manager = await DaytonaClientManager.get_instance()
            await manager._cleanup()
        _RUNNER.call(close(), 60)
        _RUNNER.loop.call_soon_threadsafe(_RUNNER.loop.stop)
        _RUNNER.thread.join(timeout=5)

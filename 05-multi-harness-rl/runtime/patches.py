"""Small, checked adapters to the immutable native environment dependencies."""


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Pinned runtime adapter no longer applies: {old[:80]}")
    return text.replace(old, new, 1)


def restore_trl(root):
    import subprocess
    relative = "trl/experimental/async_grpo/async_grpo_trainer.py"
    path = root / relative
    if not path.exists():
        return
    original = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{relative}"], text=True)
    adapted = original.replace('attn_implementation="kernels-community/flash-attn3",',
        'attn_implementation=model_init_kwargs.pop("attn_implementation", "kernels-community/flash-attn3"),', 1)
    if path.read_text() == adapted:
        path.write_text(original)


def apply_trl(root):
    path = root / "trl/experimental/async_grpo/async_grpo_trainer.py"
    old = 'attn_implementation="kernels-community/flash-attn3",'
    new = 'attn_implementation=model_init_kwargs.pop("attn_implementation", "kernels-community/flash-attn3"),'
    text = path.read_text()
    if text.count(new) == 1 and old not in text:
        return
    path.write_text(replace(text, old, new))


def apply(packages):
    path = packages / "data_agent_env/server/capture.py"
    path.write_text(replace(path.read_text(), "max_output_tokens=16384,",
                           'max_output_tokens=int(os.environ["OPENENV_MAX_OUTPUT_TOKENS"]),'))
    text = path.read_text()
    text = replace(text, "from openenv.core.harness.capture import CaptureServer, to_trace_entries",
                   "from openenv.core.harness.capture import CaptureServer\nfrom openenv.core.harness.capture.contract import to_training_trace")
    text = replace(text, "turns_from_capture(to_trace_entries(session.graph, document))",
                   "turns_from_capture(to_training_trace(session.graph, document).to_trace_entries())")
    path.write_text(text)
    path = packages / "data_agent_env/harness.py"
    text = replace(path.read_text(), "    def fetch_proxy_trace(self) -> list[TraceEntry]:",
                   "    def fetch_training_trace(self):\n"
                   "        from runtime.native_capture import training_trace\n"
                   "        return training_trace(self._result)\n\n"
                   "    def fetch_proxy_trace(self) -> list[TraceEntry]:")
    path.write_text(text)
    path = packages / "data_agent_env/server/rollout.py"
    text = replace(path.read_text(), 'opencode run --print-logs "$(cat {config.home}/workdir/task.md)"',
                   'opencode run --format json --print-logs "$(cat {config.home}/workdir/task.md)" > /tmp/native-events.jsonl')
    text = replace(text, "        timed_out = exit_code != 0", """        timed_out = exit_code != 0
        native_actions = None
        if not timed_out:
            from train.adapters import opencode_count
            try:
                native_actions = opencode_count(sandbox.read_text('/tmp/native-events.jsonl'))
            except (ValueError, KeyError, TypeError, OSError):
                pass""")
    text = replace(text, '                **metadata_for(task, grade, n_tool_calls),',
                   '                **metadata_for(task, grade, n_tool_calls),\n                "verified_native_actions": native_actions,')
    path.write_text(text)
    path = packages / "whitebox_bash/server/environment.py"
    text = replace(path.read_text(), "from .sandbox import WORKDIR, ExecResult, Sandbox",
                   "from .sandbox import ExecResult\nfrom daytona_whitebox_backend import DaytonaSandbox as Sandbox\nWORKDIR = '/workdir'")
    text = replace(text, "sb = Sandbox.start(timeout_s=SANDBOX_TIMEOUT_S,", "sb = Sandbox.start(task=task, timeout_s=SANDBOX_TIMEOUT_S,")
    start = text.index("            check_passed: bool | None = None")
    end = text.index("            return verdict.as_dict()", start) + len("            return verdict.as_dict()")
    text = replace(text, text[start:end], """            from runtime.whitebox_reward import grade
            try:
                return grade(s)
            finally:
                _release(session_id)""")
    path.write_text(text)
    path = packages / "whitebox_bash/tasks.py"
    text = replace(path.read_text(), '    if TASK_SOURCE == "data-agent" and split not in ("demo",):',
        "    if TASK_SOURCE == 'harbor-frozen':\n        from daytona_whitebox_backend import load_frozen_tasks\n"
        "        tasks = load_frozen_tasks(split)\n        _CACHE[split] = tasks\n        return tasks\n"
        '    if TASK_SOURCE == "data-agent" and split not in ("demo",):')
    text = replace(text, '        d.pop("check", None)', '        d.pop("check", None)\n        d.pop("metadata", None)\n        d.pop("setup", None)')
    path.write_text(text)
    path = packages / "whitebox_bash/client.py"
    text = path.read_text()
    start = text.index("        # TRL's reward column is a float")
    end = text.index("        if verdict.ungraded:", start)
    text = replace(text, text[start:end], "        # TRL ignores NaN rewards from ungraded episodes.\n")
    text = replace(text, "reporting 0.0 to the trainer", "excluding reward from the trainer")
    text = replace(text, "self._reward = 0.0 if verdict.reward is None else verdict.reward",
                   'self._reward = float("nan") if verdict.reward is None else verdict.reward')
    path.write_text(text)

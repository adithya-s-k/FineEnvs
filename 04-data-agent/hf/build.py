"""Package the frozen experiment and the checked-in HF launchers without secrets."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
PORTABLE_ROOT = "/workspace/repro"
REL_RUN = Path("experiments/daytona_harness_comparison/logs/20260915")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(path, before, after):
    text = path.read_text()
    assert text.count(before) == 1, f"Patch anchor changed: {path}: {before[:60]}"
    path.write_text(text.replace(before, after))


def verify_runtime_entrypoints(stage):
    required = ["hf/runtime/ui_smoke.py", "hf/runtime/setup_pipeline.py",
                "hf/runtime/coordinator.py", "hf/runtime/job.py",
                "hf/configs/deployment.json", "hf/locks/requirements-env.lock"]
    missing = [name for name in required if not (stage / name).is_file()]
    if missing:
        raise ValueError(f"Bundle is missing runtime entry points: {missing}")


def refresh_source_fixes(stage):
    """Idempotent compatibility fixes in the portable copy, never the frozen input."""
    # Checkpoint model cards query importlib.metadata.version("trl"). Ship the
    # upstream package metadata so bootstrap can install this exact source copy.
    for name in ("pyproject.toml", "VERSION", "README.md", "LICENSE"):
        shutil.copy2(WORKSPACE / "trl" / name, stage / REL_RUN / "source/trl" / name)
    whitebox_script = WORKSPACE / "experiments/daytona_harness_comparison/tools/train_whitebox_daytona.py"
    text = whitebox_script.read_text().replace(str(WORKSPACE), PORTABLE_ROOT)
    text = text.replace('os.environ.get("SLURM_JOB_ID","local")', 'os.environ.get("RUN_OWNER","local")')
    (stage / "experiments/daytona_harness_comparison/tools/train_whitebox_daytona.py").write_text(text)
    client = stage / REL_RUN / "source/trl/trl/generation/vllm_client.py"
    before = '        self._post(f"{self.base_url}/reset_prefix_cache")'
    after = '''        # vLLM acknowledges this control request with an empty HTTP 200 body.
        response = self.session.post(f"{self.base_url}/reset_prefix_cache")
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")'''
    text = client.read_text()
    if before in text:
        assert text.count(before) == 1
        client.write_text(text.replace(before, after))
    elif after not in text:
        raise ValueError("Frozen TRL cache-reset compatibility patch no longer matches")
    train = stage / REL_RUN / "source/HuggingEnvs/04-data-agent/train"
    for name in ("train_harbor_multi.py", "standalone_comparison.py", "train_standalone_comparison.py"):
        text = (HERE.parent / "train" / name).read_text()
        text = text.replace(str(WORKSPACE), PORTABLE_ROOT)
        text = text.replace("os.environ.get('SLURM_JOB_ID', 'local')", "os.environ.get('RUN_OWNER', 'local')")
        if name == "train_harbor_multi.py":
            text = text.replace("        llm_url=args.vllm_url,", "        llm_url=os.environ.get('ROLLOUT_LLM_URL', args.vllm_url),\n        api_key=os.environ.get('ROLLOUT_LLM_API_KEY', ''),")
        (train / name).write_text(text)
    # Refresh the actual standalone implementation explicitly; the historical
    # snapshot also has a package with this name and must not win by accident.
    native = stage / REL_RUN / "source/packages/data_agent_env"
    if native.exists():
        shutil.rmtree(native)
    native.mkdir(parents=True)
    source = HERE.parent / "envs/blackbox-opencode"
    for path in source.iterdir():
        if path.is_file() and (path.suffix in {".py", ".toml", ".md"} or path.name == "LICENSE"):
            shutil.copy2(path, native / path.name)
    for name in ("server", "sandbox"):
        shutil.copytree(source / name, native / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    evaluator = stage / "experiments/daytona_harness_comparison/tools/eval_whitebox_native.py"
    source = evaluator.read_text()
    evaluator.write_text(source.replace("stop if a ramp grades under 90%.",
                                        "stop if a ramp grades under 90 percent."))
    blackbox_eval = stage / REL_RUN / "eval-source/eval_concurrent.py"
    text = blackbox_eval.read_text()
    anchor = '        capture_file = str(path)\n'
    if 'from eval_evidence import persist_trial' not in text:
        assert text.count(anchor) == 1
        text = text.replace(anchor, anchor + '        from eval_evidence import persist_trial\n        persist_trial(args, result)\n', 1)
    blackbox_eval.write_text(text)
    launcher = stage / REL_RUN / "eval-source/serve_vllm_tunnel.sh"
    launcher.write_text(launcher.read_text().replace(
        'URL_FILE="/fsx/$USER/logs/vllm-tunnel-url-${SHORT_NAME}-${SLURM_JOB_ID:-$$}.txt"',
        'URL_FILE="${VLLM_URL_FILE:-/tmp/vllm-tunnel-url.txt}"'))
    src = stage / REL_RUN / "source/OpenEnv/src/openenv"
    # Native session metadata carries the split budget alongside its own inference endpoint.
    rollout = src / "harbor/rollout.py"
    text = rollout.read_text()
    anchor = "        task=task_dir.name,\n    )"
    if anchor in text:
        text = text.replace(anchor, "        task=task_dir.name,\n        dataset=dataset,\n        max_output_tokens=__import__('service_policy').output_limit(dataset),\n    )", 1)
    rollout.write_text(text)
    capture = src / "core/harness/capture/server.py"
    text = capture.read_text()
    anchor = "        clamped = clamp_output_tokens(chat_request, app.state.max_output_tokens)"
    if anchor in text:
        text = text.replace(anchor, "        output_cap = session.metadata.get('max_output_tokens') or app.state.max_output_tokens\n        if app.state.max_output_tokens and output_cap:\n            output_cap = min(output_cap, app.state.max_output_tokens)\n        clamped = clamp_output_tokens(chat_request, output_cap)", 1)
        text = text.replace('                app.state.max_output_tokens,\n            )', '                output_cap,\n            )', 1)
    capture.write_text(text)
    environment = src / "harbor/environment.py"
    text = environment.read_text()
    anchor = "        result = await _resolve_and_run()"
    if anchor in text:
        text = text.replace(anchor, "        from service_policy import admission\n        async with admission.slot(split):\n            result = await _resolve_and_run()", 1)
    environment.write_text(text)
    # Explicit close lets the interactive UI release an abandoned episode without grading it.
    whitebox = stage / REL_RUN / "source/packages/whitebox_bash/server/environment.py"
    text = whitebox.read_text()
    anchor = "        @mcp.tool\n        def submit_solution("
    if 'def close_episode(' not in text:
        text = text.replace(anchor, "        @mcp.tool\n        def close_episode(session_id: str) -> dict:\n            _release(session_id)\n            return {'closed': True}\n\n" + anchor, 1)
    whitebox.write_text(text)


def build(snapshot, out):
    out.mkdir(parents=True, exist_ok=True)
    stage = out / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    run = stage / REL_RUN
    run.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".env", "*.env", ".venv", "node_modules")
    for name in ("source", "eval-source", "datasets"):
        shutil.copytree(snapshot / name, run / name, ignore=ignore)
    for name in ("comparison.json", "train_manifest.json", "test_manifest.json", "train_indices.txt",
                 "test_indices.txt", "reference_schedule.json", "opencode_schedule.json", "provider_smoke.json",
                 "whitebox_tools_smoke.json", "source_input_hashes.json"):
        if (snapshot / name).exists():
            shutil.copy2(snapshot / name, run / name)
    for name in ("blackbox/canonical_scores.json", "blackbox/final_tito.json", "whitebox/baseline/canonical_scores.json",
                 "whitebox/baseline/scores.json"):
        target = run / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot / name, target)
    tools = stage / "experiments/daytona_harness_comparison/tools"
    tools.mkdir(parents=True)
    for name in ("train_whitebox_daytona.py", "eval_whitebox_native.py", "generation_routing.py",
                 "daytona_whitebox_backend.py", "validate_training_smoke.py", "cleanup_daytona.py",
                 "audit_blackbox.py", "score_comparison.py", "smoke_daytona.py", "smoke_whitebox.py"):
        source = snapshot.parents[1] / "tools" / name
        if source.exists():
            shutil.copy2(source, tools / name)
    shutil.copytree(HERE, stage / "hf", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"))

    # Runtime paths only. Original manifest contents are retained unchanged for provenance.
    for base in (run / "source", run / "eval-source", tools):
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".sh"):
                text = p.read_text()
                text = text.replace(str(WORKSPACE), PORTABLE_ROOT)
                text = text.replace("os.environ.get('SLURM_JOB_ID', 'local')", "os.environ.get('RUN_OWNER', 'local')")
                text = text.replace('os.environ.get("SLURM_JOB_ID", "local")', 'os.environ.get("RUN_OWNER", "local")')
                p.write_text(text)
    serve = run / "eval-source/serve_vllm_tunnel.sh"
    replace_once(serve, 'VLLM_LOG="/fsx/$USER/logs/vllm-server-${SLURM_JOB_ID:-$$}.log"',
                 'VLLM_LOG="${VLLM_LOG:-/tmp/vllm-server.log}"')
    # Retain the native serving implementation; only opt private Spaces into its existing tunnel path.
    serving = run / "source/OpenEnv/src/openenv/harbor/serving.py"
    replace_once(serving, "        public = space_public_url()", "        public = '' if os.environ.get('OPENENV_CAPTURE_TRANSPORT') == 'tunnel' else space_public_url()")
    trainer = run / "source/HuggingEnvs/04-data-agent/train/train_harbor_multi.py"
    replace_once(trainer, "        llm_url=args.vllm_url,", "        llm_url=os.environ.get('ROLLOUT_LLM_URL', args.vllm_url),\n        api_key=os.environ.get('ROLLOUT_LLM_API_KEY', ''),")
    # An isolated job has a single GPU pair and a stable explicit run owner.
    whitebox = tools / "train_whitebox_daytona.py"
    text = whitebox.read_text().replace('os.environ.get("SLURM_JOB_ID","local")', 'os.environ.get("RUN_OWNER","local")')
    whitebox.write_text(text)
    refresh_source_fixes(stage)
    verify_runtime_entrypoints(stage)
    # Source and packaging metadata are both bound to the archived runtime.
    paths = {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    manifest = {"schema": 1, "source_snapshot": str(snapshot), "files": paths,
                "patches": ["portable runtime paths", "HF sandbox owner labels", "private-Space capture tunnel", "remote rollout inference URL separate from local weight-sync URL"]}
    (stage / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = out / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for p in sorted(stage.iterdir()):
            tar.add(p, arcname=p.name)
    metadata = {"sha256": sha(archive), "bytes": archive.stat().st_size, "files": len(paths)}
    (out / "bundle.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copy2(HERE / "runtime/bootstrap.py", out / "bootstrap.py")
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, default=WORKSPACE / REL_RUN)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--refresh-runtime", action="store_true", help="Refresh only checked-in HF scripts in an existing bundle")
    a = p.parse_args()
    if a.refresh_runtime:
        out = a.out.resolve()
        with tempfile.TemporaryDirectory(prefix="daytona-bundle-") as temp:
            stage = Path(temp)
            with tarfile.open(out / "bundle.tar.gz") as tar:
                tar.extractall(stage, filter="data")
            shutil.rmtree(stage / "hf")
            shutil.copytree(HERE, stage / "hf", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"))
            refresh_source_fixes(stage)
            verify_runtime_entrypoints(stage)
            manifest = json.loads((stage / "bundle_manifest.json").read_text())
            manifest["files"] = {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob("*"))
                                 if p.is_file() and p.name != "bundle_manifest.json"}
            (stage / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            with tarfile.open(out / "bundle.tar.gz", "w:gz", compresslevel=6) as tar:
                for p in sorted(stage.iterdir()):
                    tar.add(p, arcname=p.name)
        metadata = {"sha256": sha(out / "bundle.tar.gz"), "bytes": (out / "bundle.tar.gz").stat().st_size,
                    "files": len(manifest["files"])}
        (out / "bundle.json").write_text(json.dumps(metadata, indent=2) + "\n")
        shutil.copy2(HERE / "runtime/bootstrap.py", out / "bootstrap.py")
        print(json.dumps(metadata))
    else:
        print(json.dumps(build(a.snapshot.resolve(), a.out.resolve())))

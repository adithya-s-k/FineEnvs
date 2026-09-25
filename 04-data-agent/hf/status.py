"""Read HF job state and small durable progress artifacts without streaming logs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from deploy import credentials


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jobs", nargs="+")
    p.add_argument("--env-file", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--logs", action="store_true")
    a = p.parse_args()
    from huggingface_hub import HfApi
    secret = credentials(a.env_file)
    api = HfApi(token=secret["HF_TOKEN"])
    files = ["status.json", "services.json", "eval_progress.json", "canonical_scores.json",
             "scalability.json", "upload_status.json", "upload_error.json", "training_recipe.json",
             "training_smoke_verified.json", "trackio_verified.json"]
    if a.logs:
        files += ["serving.log", "vllm.log", "bridge.log", "eval-stage0-c8.log",
                  "smoke-opencode.log", "eval-opencode.log", "train-first.log", "train-resumed.log", "trackio-sync.log",
                  "eval-stage1-c32.log", "eval-stage2-c48.log", "eval-stage2-c53.log"]

    def inspect(job_id):
        j = api.inspect_job(job_id=job_id, namespace="HuggingEnvs")
        owner = j.environment["RUN_OWNER"]
        dest = a.out / "downloads" / owner
        dest.mkdir(parents=True, exist_ok=True)
        prefix = j.environment["RUN_ID"] + "/jobs/" + owner + "/"
        requested = list(files)
        if j.labels.get("role") == "coordinator":
            if j.labels.get("phase") == "qualify":
                requested = ["qualification.json"]
            elif j.labels.get("phase") == "setup":
                prefix = j.environment["RUN_ID"] + "/pipelines/" + j.environment["BUNDLE_SHA256"] + "/"
                requested = ["pipeline.json"]
            else:
                parent = api.inspect_job(job_id=j.environment["TRAINING_JOB"], namespace="HuggingEnvs")
                prefix = j.environment["RUN_ID"] + "/coordination/" + parent.environment["RUN_OWNER"] + "/"
                requested = ["monitor.json", "state.json"]
        from huggingface_hub.errors import EntryNotFoundError
        try:
            available = {item.path for item in api.list_bucket_tree(j.environment["ARTIFACT_BUCKET"],
                         prefix=prefix, recursive=False)}
        except EntryNotFoundError:
            available = set()
        if j.labels.get("arm") == "opencode":
            for backend in ("daytona", "hf"):
                for sub in (backend, "smoke/" + backend):
                    try:
                        nested = {item.path for item in api.list_bucket_tree(j.environment["ARTIFACT_BUCKET"], prefix=prefix+sub+"/", recursive=False)}
                    except EntryNotFoundError:
                        nested = set()
                    available.update(nested)
                    requested += [sub + "/scores.json", sub + "/scalability.json", sub + "/configuration.json", sub + "/progress.json"]
                    (dest / sub).mkdir(parents=True, exist_ok=True)
        if a.logs and j.labels.get("role") != "coordinator":
            requested += sorted(Path(name).name for name in available
                                if Path(name).name.startswith("eval-stage")
                                and name.endswith(".log") and Path(name).name not in requested)
        downloads = [(prefix + name, str(dest / name)) for name in requested if prefix + name in available]
        if downloads:
            api.download_bucket_files(j.environment["ARTIFACT_BUCKET"], files=downloads, raise_on_missing_files=False)
        result = {"id": job_id, "owner": owner, "stage": j.status.stage}
        for name in requested:
            path = dest / name
            if not path.exists():
                continue
            if name.endswith(".json"):
                value = json.loads(path.read_text())
                if name == "canonical_scores.json" and j.labels.get("arm") != "opencode":
                    value = {k: value.get(k) for k in ["graded_cells", "expected_cells", "complete",
                        "average_pass_at_1", "comparison_ready", "ungraded_attempts", "tito_pass"]}
                result[name] = value
            else:
                content = "\n".join(path.read_text(errors="replace").splitlines()[-8:])
                for value in secret.values():
                    content = content.replace(value, "[REDACTED]")
                result[name] = content[-2000:]
        return result

    with ThreadPoolExecutor(max_workers=min(4, len(a.jobs))) as pool:
        for result in pool.map(inspect, a.jobs):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

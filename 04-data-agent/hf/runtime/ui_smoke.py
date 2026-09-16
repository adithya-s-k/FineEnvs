"""Exercise the deployed UIs and two isolated Daytona sessions using Gradio's client."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

from gradio_client import Client


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-file")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from dotenv import dotenv_values
    values = dotenv_values(a.env_file) if a.env_file else {}
    token = values.get("HF_API_KEY") or os.environ["HF_TOKEN"]
    repo = "HuggingEnvs/data-agent-seta-whitebox-env"
    clients = [Client(repo, token=token, verbose=False) for _ in range(2)]

    def one(i):
        c = clients[i]
        split, index = ("train", 895) if i == 0 else ("test", 166)
        instruction, _ = c.predict(split, index, api_name="/preview_task")
        try:
            prompt, _, state = c.predict(split, index, api_name="/start_task")
            assert prompt == instruction and state == "Active"
            content = f"ui-isolation-{i}"
            out = c.predict("write", "", "/workdir/ui-isolation.txt", content, "", api_name="/run_tool")
            assert "[error]" not in out
            out = c.predict("bash", "cat /workdir/ui-isolation.txt", "/workdir", "", "", api_name="/run_tool")
            assert content in out and f"ui-isolation-{1-i}" not in out
            grade, state = c.predict("__known_wrong_ui_smoke__", api_name="/grade_task")
            assert grade == "Reward: 0.0" and state == "Finished"
            return {"split": split, "index": index, "passed": True, "isolated_tools": True,
                    "graded_zero": True, "closed": True}
        finally:
            c.predict(api_name="/close_task")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(one, range(2)))
    for arm in ["opencode-blackbox", "seta-whitebox"]:
        c = Client("HuggingEnvs/data-agent-" + arm + "-env", token=token, verbose=False)
        for split, index in [("train", 895), ("test", 249)]:
            instruction, _ = c.predict(split, index, api_name="/preview_task")
            assert len(instruction) > 50
            results.append({"arm": arm, "split": split, "index": index, "preview_passed": True})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"passed": True, "checked_at": time.time(), "results": results}, indent=2) + "\n")
    print(a.out.read_text())


if __name__ == "__main__":
    main()

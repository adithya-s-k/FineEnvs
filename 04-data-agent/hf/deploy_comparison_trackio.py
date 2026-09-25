"""Deploy the public comparison dashboard using native Trackio and bucket storage."""
import os
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi, CommitOperationAdd

from consolidate_async_runs import REPO, SPACE, BUCKET, PROJECT, DEFAULT_OUT


def main():
    values = dotenv_values(REPO / 'experiments/.env')
    os.environ['HF_TOKEN'] = values.get('HF_API_KEY') or values['HF_TOKEN']
    os.environ['TRACKIO_PLOT_ORDER'] = 'eval/pass_at_1,eval/difficulty/*,eval/harness/*,eval/harness_difficulty/*,train/reward_rolling20,train/reward,train/loss,train/grad_norm,train/nonzero_gradient_rolling20,train/entropy,train/kl,train/learning_rate'
    from trackio.deploy import create_space_if_not_exists
    create_space_if_not_exists(SPACE, bucket_id=BUCKET, private=False)
    api = HfApi()
    api.update_repo_settings(SPACE, repo_type='space', private=False)
    app = f'''import os
os.environ.setdefault("TRACKIO_PLOT_ORDER", {os.environ['TRACKIO_PLOT_ORDER']!r})
import trackio
trackio.show(project={PROJECT!r})
'''
    readme = f'''---
title: Data Agent Training Comparison
emoji: 📈
colorFrom: indigo
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
---

# Qwen3.5-2B: Harbor and native OpenCode

Both complete training histories (steps1–1000), checkpoint pass@1, difficulty,
harness, harness × difficulty, and optimizer/rollout diagnostics in one Trackio project.

- [Complete checkpoint report](REPORT.md)
- [Comparison figure](comparison.png)
- [Vector figure](comparison.svg)

Every checkpoint score shown passed full coverage, TiTO and harness-version gates.
Remaining evaluations appear automatically after validation. Baselines are separate
measured E2B and Daytona cohorts, as documented in the report.

Trackio0.33.0, standard CPU Space, persistent HF bucket; independent CPU publisher.
Use zero smoothing to inspect exact checkpoint scores. Raw metrics and explicitly
named20-update rolling averages are both retained.
'''
    operations = [CommitOperationAdd(path_in_repo='app.py',path_or_fileobj=app.encode()),
        CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=readme.encode())]
    for name in ('REPORT.md','comparison.png','comparison.svg','comparison.pdf'):
        operations.append(CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(DEFAULT_OUT/name)))
    result = api.create_commit(SPACE, repo_type='space', operations=operations,
                              commit_message='Add unified Harbor and OpenCode comparison dashboard and report')
    print('space',SPACE,'commit',result.oid)


if __name__=='__main__':main()

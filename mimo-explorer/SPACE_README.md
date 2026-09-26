---
title: MiMo RL Environment Explorer
emoji: 🧭
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Explore the MiMo-V2.6 RL environments and run rollouts
hf_oauth: true
hf_oauth_expiration_minutes: 1440
hf_oauth_scopes:
  - inference-api
  - jobs
tags:
  - reinforcement-learning
  - rl-environments
  - agents
  - dataset-explorer
datasets:
  - XiaomiMiMo/MiMo-V2.6-RL-oss
---

# MiMo RL Environment Explorer

An unofficial explorer for [XiaomiMiMo/MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss):
7,780 RL environments across five domains. Browse them, open any task to see everything inside it, then
**run a rollout** with a model of your choice and watch it get graded.

| Domain | Environments | The task | Graded by |
|---|---:|---|---|
| Code | 2,698 | Fix a real issue in a real repo | Hidden tests |
| Webdev | 2,093 | Build a website from a brief | A vision model on a full-page render |
| Cyber | 1,000 | Reproduce a real crash (ARVO) | A root-owned server: must crash in the expected function |
| Music | 1,000 | Compose in ABC notation | 18 "human-likeness" features, no model |
| General | 989 | Knowledge work in a simulated workplace | Rule checks and model-judged rubric checks |

## Running a rollout

Sign in with Hugging Face, or paste an access token (a write token, or a fine-grained one with the Inference Providers
and Jobs permissions). A rollout runs on **your** account:

- an **HF Sandbox** started from the task's own Docker image (CPU, about $0.01 an hour),
- **OpenCode** as the agent harness, calling the model you pick through **HF Inference Providers**,
- then the task's **own grader**: Xiaomi's reference code from [verl](https://github.com/XiaomiMiMo/verl) and
  [mimoagent](https://github.com/XiaomiMiMo/mimoagent), vendored unchanged. Tasks graded by a model ask you to pick the judge.

Nothing that grades a task (hidden tests, rubric answers, the expected crash) is in the sandbox while the agent works,
and the agent has no web tools. Rollouts keep running if you close the page; find them under **Rollouts**.
Traces are kept in a private bucket. Your token is never stored with them: it lives only in an encrypted cookie in your browser, and the sandbox never sees it (model calls go through a per-rollout proxy). See [SECURITY.md](SECURITY.md).

## Rollouts and the community

Rollouts are **public by default**: they appear on their task and under **Community** for everyone, shown without
who ran them, with the model, provider (or "own endpoint"), judge, settings and the exact versions that produced
them. You can make any rollout private from its page, and private rollouts are visible only to you. **My rollouts**
always shows only your own.

Public rollouts help everyone compare models on the same tasks and learn from real traces. With that in mind, they
may later be released as an open dataset, for example as SFT or RL traces, with the same anonymisation. Private
rollouts are never included.

Something off? Please [open a discussion](https://huggingface.co/spaces/FineEnvs/MiMo-RL-Envs-Explorer/discussions).
To work on the explorer itself, see [CONTRIBUTING.md](CONTRIBUTING.md).

Built by Hugging Face and the FineEnvs team.

Rewards here will not match Xiaomi's published numbers exactly: the harness is OpenCode rather than theirs, the judge
is whichever model you pick, and Webdev uses their evaluation-mode grader (training used a group-relative ranking).

## Run it yourself

```bash
git clone https://github.com/adithya-s-k/FineEnvs && cd FineEnvs/mimo-explorer
uv run uvicorn app.main:app      # uses your HF token; traces go to ./.local-runs
```

Source: [adithya-s-k/FineEnvs/mimo-explorer](https://github.com/adithya-s-k/FineEnvs/tree/mimo-explorer-rollouts/mimo-explorer).

"""Xiaomi's MiMo-V2.6 RL release (XiaomiMiMo/MiMo-V2.6-RL-oss), as the MiMo RL Environment Explorer reads and runs it:
every one of its 7,780 tasks browsable in full (brief, agent prompt, systems and their data, workspace files, the
repository, the grader with its judge prompts), and rollouts on its own harness: OpenCode on an HF Sandbox, graded by
Xiaomi's graders (hidden tests, a fuzz crash, rubric checks with a judge model, a screenshot judge, the music scorer).

The code is the MiMo explorer's (tooling/mimo-rl-explorer/app), kept close to it so the two can be compared and synced:
    catalog.py   the task views and what the runner needs (data/: the browsing index, built by build_data.py)
    judges.py    the judge prompts, pinned
    previews.py  workspace file previews (spreadsheets, documents, slides, PDFs)
    runner/      one rollout start to finish, per domain (core.py, domains.py, opencode.py)
    vendor/      Xiaomi's graders: the music scorer, the webdev judge, the fuzz-crash server
    snapshot_repos.py, validate.py   tools: snapshot Code tasks' repositories; health checks with no model
The explorer reads it through app/envs/mimo.py (the environment contract) and runs it as the "mimo" runner.
config, store, models, endpoints and version here point at the explorer's own.
"""

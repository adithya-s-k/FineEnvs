# Local experiment snapshot, 30 September 2026

This snapshot preserves the pending environment, training, evaluation, HF runtime and reporting work. It also copies the SFT experiment source and the deployed SmolDataEnv RL/SFT dashboard source into the repository. Source manifests identify the original files and deployed revisions.

Validation against the current workspace (OpenEnv `781bfc9a`, TRL `3cb8efdd`):

- Training/checkpoint/evaluation regression tests: 54 passed.
- HF runtime tests: 92 passed, 21 subtests passed, one failed.
- LFM medium/hard configuration and reward tests: 12 passed.
- LFM configuration and packed-convolution tests: three passed.
- SFT epoch-checkpoint policy: one passed. The separate evaluation-watcher script passed without submitting jobs.
- Python syntax, JSON parsing and shell syntax checks passed. Credential scans found no matches in the files selected for Git.

The HF failure is `test_shared_service.py::SharedServiceTest::test_simultaneous_split_caps_reach_engine_unchanged`: the current OpenEnv capture server returned the global 16,384-token cap for the eval session instead of its requested 4,096-token cap. This is an unresolved compatibility check, not a passing production qualification. Historical experiment results retain their original runtime provenance.

The SFT `test_local_logging.py` and `test_opencode_messages.py` files are standalone audit commands requiring prepared run/data arguments, not pytest collection targets. No new GPU jobs, sandbox rollouts or cloud deployments were started for this snapshot.

Raw publisher logs, live Trackio databases and duplicate raw metric exports remain on disk and are excluded by `04-data-agent/.gitignore`. The small versioned SFT scalar archive is intentionally included because the SFT dashboard restores from it. Checkpoints, credentials and raw agent traces are not included.

# Scripts

Run from the worktree root with `../../launch image-text-gen-rl --exec python image-text-gen-rl/train/<script>.py`.

| Script | Purpose | Cost |
|---|---|---|
| `calibrate_verifier.py` | Sweep candidate VLMs on labelled synthetic renders; rank models and pairs | ~$2 for 280 images × 11 models |
| `screen_prompts.py` | Word list + Gemma 4 31B rubric screen of all source prompts → `exclusions.json` | ~$0.5 |
| `publish_dataset.py` | Publish screened splits to `AdithyaSK/image-text-gen-rl-prompts`, pin `PUBLISHED_REVISION` | free |
| `deploy_space.py` | Private Space `AdithyaSK/image-text-gen-rl-env` with dataset mount and audit bucket | Space hardware + verifier calls |

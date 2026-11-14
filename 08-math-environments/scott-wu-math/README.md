<div align="center">

<h1>Scott Wu Math</h1>

<h3>Make the number from every card once with + − × ÷</h3>

<p>The 24 game, then the current year from eight cards, then any four-digit number from nine, as an OpenEnv environment.</p>

<a href="https://huggingface.co/spaces/FineEnvs/ScottWuMath"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-Play%20it%20now-FFD21E?style=for-the-badge&labelColor=1a1a1a" alt="Play the environment" height="32"></a>
<a href="https://github.com/openenv/OpenEnv"><img src="https://img.shields.io/badge/framework-OpenEnv-3B82F6?style=for-the-badge&labelColor=1a1a1a" alt="OpenEnv" height="32"></a>
<a href="https://github.com/adithya-s-k/FineEnvs/tree/main/08-math-environments/scott-wu-math"><img src="https://img.shields.io/badge/GitHub-08--math--environments%2Fscott--wu--math-181717?style=for-the-badge&logo=github&labelColor=1a1a1a" alt="Code on GitHub" height="32"></a>

</div>

---

Scott Wu's card game as an OpenEnv environment: deal cards, name a number, make it with + − × ÷ using
every card once. Four cards make 24; eight make the current year; nine make any four-digit number someone
names. People play it in the browser and models play it through the API, against the same server.

- **Play:** https://huggingface.co/spaces/FineEnvs/ScottWuMath
- **Environment card** (rules, levels, reward, API, solver, results): [envs/scott_wu_math/README.md](envs/scott_wu_math/README.md)

A fan tribute, not affiliated with or endorsed by Scott Wu or Cognition.

<!-- BEGIN:matrix -->
| Env | Tools | Backend | `openenv` |
|---|---|---|---|
| **scott_wu_math** | 0 | `python-offline` | ✅ |
<!-- END:matrix -->

## Layout

```text
scott-wu-math/
├── envs/scott_wu_math/      the environment (OpenEnv server, play page, grader, solver, task splits)
│   ├── game/                cards, expression checker, solver, levels, reward, prompts (no server needed)
│   ├── server/              OpenEnv app + environment; web/ is the play page
│   ├── data/tasks/          train (3,639), test (200), video (2): built by scripts/build_tasks.py
│   └── tests/
└── eval/
    ├── models.json          12 models: OpenAI, Anthropic, open models on HF Inference Providers
    ├── rollout.py           parallel rollouts, one OpenEnv session per episode
    ├── eval_card.py         runs/test20-* -> results/eval-v1.json, the catalog's eval card
    ├── export_video.py      turns runs into data.ts for the hf-motion comparison video
    └── export_vs.py         the same for the side-by-side video against the clip
results/eval-v1.json         12 models x 3 modes on 20 test deals
```

## Run

```bash
cd envs/scott_wu_math
uv venv && uv pip install -e ".[dev,eval]"
uvicorn scott_wu_math.server.app:app --port 8012      # http://localhost:8012
pytest

cd ../..
python eval/rollout.py --split video --samples 3 --mode think  --out runs/video-think
python eval/rollout.py --split test --per-level 5 --mode direct --out runs/test20-direct
```

`--mode` is `think` (reasoning at high effort), `nothink` (reasoning off or at the API's floor) or
`direct` (reasoning off, answer tag only, 300 tokens). Runs resume: rerunning a command fills in only the
episodes that are missing. Keys come from the nearest `.env` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`HF_TOKEN`). `runs/` is gitignored.

## Results

See the environment card and [results/eval-v1.json](results/eval-v1.json). Short version: with reasoning, every model makes 6,843 from the video's nine
cards on the first try; answering at once in 300 tokens, only GPT-6.1 Sol and Claude Opus 5.5 do. On 20
test deals at once, GPT-6.1 Sol solves 14, Opus 5.5 8, the other ten models 1 to 5.

"""OpenEnv's Harbor server for QED-Nano tasks, with our qed-proof harness registered.

Same service as 05-multi-harness-rl/envs/harbor (task API, capture proxy, sandbox lifecycle), plus:
  - the `qed-proof` seam, so `HarborSession(harness="qed-proof")` runs our Harbor agent, pointed at the capture
    proxy with the session id as its key. OpenEnv documents `module:Class` harnesses but resolves only names in
    `seams.SEAMS`, and the trial runs in this process, so the seam is registered here, before any rollout;
  - a larger per-call output cap than 05's (proofs plus a critique run past 4,096 tokens);
  - `/qed/splits`, the prepared dataset folders.

The judge's key reaches each task's verifier through `[verifier] env = { JUDGE_API_KEY = "${JUDGE_API_KEY}" }`, which Harbor
resolves from this process's environment. It is never given to the agent.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

data = Path(os.environ.get("QED_DATA", "prepared")).resolve()
os.environ.setdefault("OPENENV_DATASETS", ",".join(str(data / "datasets" / s) for s in ("train", "test")))
os.environ.setdefault("OPENENV_MAX_OUTPUT_TOKENS", "8192")
os.environ.setdefault("MAX_CONCURRENT_ENVS", "40")
os.environ.setdefault("OPENENV_HARBOR_REWARD_KEY", "reward")

from openenv.harbor import seams  # noqa: E402

seams.SEAMS["qed-proof"] = seams.Seam(
    name="qed-proof",
    dialect="openai_chat",
    import_path="qed_proof_agent.agent:QedProofAgent",
    model_fmt="{model}",
    # The key goes in the agent's own kwargs, never in process env: process env stays set through the verifier.
    kwargs=lambda base_url, session, model: {
        "api_base": f"{base_url}/v1",
        "api_key": session,
        "rounds": os.environ.get("QED_ROUNDS", "1"),
        "max_tokens": os.environ.get("OPENENV_MAX_OUTPUT_TOKENS", "8192"),
    },
    status="untested",
)

from harbor_env.server.app import app as harbor_app  # noqa: E402


@asynccontextmanager
async def lifespan(app):
    async with harbor_app.router.lifespan_context(harbor_app):
        yield


# Register our routes before mounting Harbor's app, which owns the root.
app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/qed/splits")
def split_paths():
    return {s: str(data / "datasets" / s) for s in ("train", "test")}


app.mount("/", harbor_app)

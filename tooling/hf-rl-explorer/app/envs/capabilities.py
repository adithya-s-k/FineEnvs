"""Evidence-based capabilities shared by dataset adapters, live Spaces, UI and MCP.

A framework label describes a format, not permission or proof that it can run.
A dataset's native harness may be external; a live server may be asleep or may
speak HTTP instead of OpenEnv's WebSocket protocol. Keep those axes independent.
The contract is additive so older clients can keep using run.options/step_api.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Framework:
    id: str
    label: str
    task: str
    lifecycle: str
    reward: str
    docs: str


FRAMEWORKS = {
    f.id: f for f in (
        Framework("harbor", "Harbor", "Task folder or packed task files",
                  "Prepare sandbox → run agent → verify → collect artifacts",
                  "The task's verifier writes a scalar or named reward metrics.",
                  "https://www.harborframework.com/docs/tasks"),
        Framework("openenv", "OpenEnv", "Live environment with typed actions and observations",
                  "Open session → reset → step until done → close",
                  "The environment returns rewards and termination with observations.",
                  "https://huggingface.co/docs/openenv/main/reference/core"),
        Framework("verifiers", "Verifiers", "Rows loaded by an installed environment or taskset",
                  "Load environment/taskset → run its harness → score the rollout",
                  "The installed environment's rubric or judge defines the reward; rows alone do not include that runtime.",
                  "https://github.com/PrimeIntellect-ai/verifiers/tree/main/docs"),
        Framework("nemo-gym", "NeMo Gym", "Task data plus agent and resources-server configuration",
                  "Seed → interact → verify, or reset → step for Gymnasium servers",
                  "A resources server or embedded agent harness computes the reward.",
                  "https://docs.nvidia.com/nemo/gym/build-verifiers/"),
        Framework("verl", "verl", "Training rows with prompt and reward metadata",
                  "Generate responses → route to the configured reward function → train",
                  "data_source selects a scorer unless training config supplies a custom reward function.",
                  "https://verl.readthedocs.io/en/latest/advance/reward_loop.html"),
        Framework("skyrl", "SkyRL", "Training rows routed to a skyrl-gym environment",
                  "Load rows → run environment interactions → compute training rewards",
                  "The selected environment and its configuration define the reward.",
                  "https://skyrl.readthedocs.io/en/latest/"),
        Framework("mimo", "MiMo", "Tasks from the MiMo RL environment release",
                  "Prepare task → run the release's domain harness → grade",
                  "Each domain uses its own verifier; supported conversions also run through Harbor.",
                  "https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss"),
        Framework("gymnasium", "Gymnasium HTTP", "A server exposing reset and step over HTTP",
                  "Reset → step until terminated or truncated → close",
                  "The server returns step rewards; its HTTP schema defines the request and response.",
                  "https://gymnasium.farama.org/api/env/"),
        Framework("mcp", "MCP", "Tools exposed by a live server", "Connect → call advertised tools → close",
                  "A reward is available only when the server exposes a grader or returns one.",
                  "https://modelcontextprotocol.io/specification/2025-06-18/server/tools"),
        Framework("ors", "ORS", "A server with named environments and sessions",
                  "Create session → call environment → close session", "The server defines its grading contract.", ""),
        Framework("api", "HTTP API", "A live server's published operations", "Call the server's documented routes",
                  "Inspect the published schema and grading routes for this server.", ""),
        Framework("rows", "RL dataset", "Dataset rows", "Browse task data; use the source's environment to execute it",
                  "A dataset label or answer column alone does not establish a runnable grader.", ""),
    )
}
HOSTED_RUNNERS = frozenset({"harbor", "mimo", "nemo-gym"})


def framework(value: str | None) -> Framework:
    key = str(value or "").lower().replace("_", "-").replace(" ", "-")
    aliases = {"nemo": "nemo-gym", "nemogym": "nemo-gym", "generic": "rows", "rl": "rows", "harbor-(packed)": "harbor"}
    key = aliases.get(key, key)
    if key.startswith("mimo"):
        key = "mimo"
    return FRAMEWORKS.get(key, FRAMEWORKS["rows"])


def capability(id: str, label: str, state: str, detail: str) -> dict[str, str]:
    if state not in {"available", "external", "unavailable", "unknown"}:
        raise ValueError(f"invalid capability state: {state}")
    return {"id": id, "label": label, "state": state, "detail": detail}


def dataset_support(name: str, *, state: str = "ready", options: list[dict[str, Any]] | None = None,
                    note: str = "", revision: str | None = None, restricted: bool = False) -> dict[str, Any]:
    f = framework(name)
    ready = state == "ready"
    features = [capability("browse", "Browse tasks", "available" if ready else "unknown",
                           "Inspect the prompt, grading information and source files." if ready else "Task discovery is still in progress.")]
    if options is None:
        features.append(capability("run", "Run an agent", "unknown", "Open a task to check its harness and runtime requirements."))
    else:
        runnable = [o for o in options if o.get("ok") and o.get("runner") in HOSTED_RUNNERS and not o.get("href")]
        external = [o for o in options if o.get("href") or (not o.get("ok") and o.get("runner") in {"verifiers", "nemo-gym", "nemogym", "verl", "skyrl"})]
        why = "; ".join(dict.fromkeys(o.get("why") for o in options if o.get("why")))
        features.append(capability("run", "Run an agent", "available" if runnable else "external" if external else "unavailable",
                                   "Runs here with " + ", ".join(o["label"] for o in runnable) + "." if runnable else
                                   why or note or "This source has no executable harness configured here."))
    features.append(capability("mcp", "Explore with an agent", "unavailable" if restricted else "available" if ready else "unknown",
                               "The MCP bridge only serves public datasets; this source requires your sign-in." if restricted else
                               "MCP exposes public task browsing. Execution follows the runtime support above."))
    return {"version": 1, "framework": asdict(f), "source": "dataset", "evidence": "adapter", "revision": revision,
            "transport": None, "capabilities": features}


def space_support(info: dict[str, Any]) -> dict[str, Any]:
    up = bool(info.get("running"))
    f = framework(info.get("framework") or "api")
    transport = "openenv-ws" if info.get("openenv") else "http" if info.get("api") else "mcp-http" if info.get("mcp") else None
    can_play = bool(info.get("step_api") or info.get("mcp") or info.get("api"))
    status = "available" if up and can_play else "unavailable" if up else "unknown"
    detail = ("Play an episode over OpenEnv's WebSocket session." if transport == "openenv-ws" else
              "Call the published HTTP routes; each explorer session keeps its own cookies." if transport == "http" else
              "Call MCP tools. Episode state depends on the server's MCP implementation.") if can_play else "No supported interactive API was discovered."
    if not up:
        detail = "The Space is " + str(info.get("stage") or "unreachable").lower().replace("_", " ") + "; check its live interface when it is running."
    return {"version": 1, "framework": asdict(f), "source": "space", "evidence": "live" if up else "unverified",
            "transport": transport, "capabilities": [
                capability("play", "Interactive playground", status, detail),
                capability("browse", "Browse tasks", "available" if up and info.get("task_api") else "unavailable" if up else "unknown",
                           "Tasks are served by the environment's Task API." if info.get("task_api") else "No live Task API was discovered; source files may describe the tasks."),
                capability("mcp", "Connect an agent", status,
                           "The explorer bridges the advertised interface to MCP." if up and can_play else detail),
            ]}

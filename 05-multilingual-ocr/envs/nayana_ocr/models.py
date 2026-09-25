from openenv.core.env_server.types import Action, Observation
from pydantic import Field

# Part of the environment contract, not an implementation detail: a grading that fails
# this way did not consume the episode, so the caller may retry the step without
# resetting. The server builds its messages from these and the trainer matches on them;
# keeping one definition stops a reworded message from silently disabling every retry.
JUDGE_FAILED = "Judge request or verdict failed"
JUDGE_BUSY = "Judge busy"
RETRYABLE_GRADING_SIGNALS = (JUDGE_FAILED, JUDGE_BUSY)


class NayanaAction(Action):
    answer: str = Field(default="", max_length=65536)


class NayanaObservation(Observation):
    task_id: str = ""
    snapshot_id: str = ""
    split: str = ""
    language: str = ""
    family: str = ""
    prompt: str = ""
    asset_path: str = ""
    asset_sha256: str = ""
    mime: str = ""
    width: int = 0
    height: int = 0
    metrics: dict[str, float | bool] = Field(default_factory=dict)
    reading_order_policy: str = ""
    reading_order: list[int | str] = Field(default_factory=list)
    annotation_masked: bool = False
    grading_policy_id: str = ""

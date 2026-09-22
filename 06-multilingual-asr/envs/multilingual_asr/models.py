from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class AsrAction(Action):
    transcript: str = Field(default="", max_length=65536)


class AsrObservation(Observation):
    task_id: str = ""
    snapshot_id: str = ""
    split: str = ""
    language: str = ""
    language_name: str = ""
    family: str = ""
    prompt: str = ""
    asset_path: str = ""
    asset_sha256: str = ""
    mime: str = ""
    sampling_rate: int = 0
    num_samples: int = 0
    duration_seconds: float = 0.0
    error_unit: str = ""
    metrics: dict[str, float | bool] = Field(default_factory=dict)
    grading_policy_id: str = ""

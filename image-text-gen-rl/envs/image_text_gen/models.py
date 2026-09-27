from openenv.core.env_server.types import Action, Observation
from pydantic import Field

# Part of the environment contract, not an implementation detail: a grading that fails
# this way did not consume the episode, so the caller may retry the step without
# resetting. The server builds its messages from these and trainers match on them.
VERIFIER_FAILED = "Verifier request or transcription failed"
VERIFIER_BUSY = "Verifier busy"
RETRYABLE_GRADING_SIGNALS = (VERIFIER_FAILED, VERIFIER_BUSY)

# 8 MiB of image bytes is ~11 MiB of base64, inside uvicorn's 16 MiB WebSocket frame.
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_B64_CHARS = (MAX_IMAGE_BYTES * 4) // 3 + 4


class ImageTextGenAction(Action):
    # PNG, JPEG or WebP bytes, base64-encoded (a `data:image/...;base64,` prefix is accepted).
    image: str = Field(default="", max_length=MAX_IMAGE_B64_CHARS)


class ImageTextGenObservation(Observation):
    task_id: str = ""
    split: str = ""
    family: str = ""
    prompt: str = ""
    # The text the image must show. It is quoted verbatim in the prompt, so this is a
    # convenience for logging and curricula, not a leak of the grading reference.
    target_text: str = ""
    text_len: int = 0
    # Mean Qwen OCR score of five SD3 samples from the source dataset: a difficulty prior.
    baseline_ocr: float = 0.0
    metrics: dict[str, float | bool | int] = Field(default_factory=dict)
    # Per-verifier blind transcriptions, returned so reward behaviour can be audited.
    transcriptions: list[dict] = Field(default_factory=list)
    grading_policy_id: str = ""

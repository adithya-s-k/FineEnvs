import base64
import io

import openenv.core.env_client as _env_client
import requests
from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from .models import ImageTextGenAction, ImageTextGenObservation

ENV_NAME = "image_text_gen"

# The deployed Space is private (it spends its owner's inference credits), so every
# request needs a Hugging Face token. openenv 0.4.2's EnvClient cannot send WebSocket
# headers, so its module-level connect function is wrapped once to add them for the
# WebSocket URLs registered here.
_WS_HEADERS = {}
_ws_connect = _env_client.ws_connect


def _connect_with_headers(url, **kwargs):
    headers = _WS_HEADERS.get(url)
    if headers:
        kwargs.setdefault("additional_headers", headers)
    return _ws_connect(url, **kwargs)


_env_client.ws_connect = _connect_with_headers


def encode_image(image, format="PNG"):
    """PIL image (or raw bytes) to the base64 string an ImageTextGenAction carries."""
    if isinstance(image, bytes | bytearray):
        return base64.b64encode(image).decode()
    buffer = io.BytesIO()
    image.save(buffer, format)
    return base64.b64encode(buffer.getvalue()).decode()


class ImageTextGenClient(EnvClient[ImageTextGenAction, ImageTextGenObservation, State]):
    def __init__(self, *args, token=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        if token:
            _WS_HEADERS[self._ws_url] = self._headers

    def _step_payload(self, action):
        return action.model_dump()

    def _parse_result(self, data):
        observation = dict(data["observation"])
        observation.update(reward=data.get("reward"), done=data.get("done", False))
        return StepResult(
            observation=ImageTextGenObservation(**observation),
            reward=data.get("reward"),
            done=data.get("done", False),
            metadata=data.get("metadata") or None,
        )

    def _parse_state(self, data):
        return State(**data)

    def _http_base(self):
        return (
            self._ws_url.removesuffix("/ws")
            .replace("wss://", "https://")
            .replace("ws://", "http://")
            .rstrip("/")
        )

    def manifest(self):
        response = requests.get(
            f"{self._http_base()}/manifest", headers=self._headers, timeout=30
        )
        response.raise_for_status()
        return response.json()

    def num_tasks(self, split):
        response = requests.post(
            f"{self._http_base()}/{ENV_NAME}/num_tasks",
            json={"split": split},
            headers=self._headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["num_tasks"]

    def get_task_range(self, split, start=0, stop=None):
        response = requests.post(
            f"{self._http_base()}/{ENV_NAME}/task_range",
            json={"split": split, "start": start, "stop": stop},
            headers=self._headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["tasks"]


def connect(url, token=None):
    """Sync client. For a *.hf.space URL the local Hugging Face token is used by default."""
    if token is None and ".hf.space" in url:
        from huggingface_hub import get_token

        token = get_token()
    # Two verifier calls of up to 60 s each run in parallel, plus one retry.
    return ImageTextGenClient(
        base_url=url, connect_timeout_s=60, message_timeout_s=180, token=token
    ).sync()

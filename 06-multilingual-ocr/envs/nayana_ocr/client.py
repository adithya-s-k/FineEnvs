import requests
from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from .models import NayanaAction, NayanaObservation

ENV_NAME = "nayana_ocr"


class NayanaClient(EnvClient[NayanaAction, NayanaObservation, State]):
    def _step_payload(self, action):
        return action.model_dump()

    def _parse_result(self, data):
        observation = dict(data["observation"])
        observation.update(reward=data.get("reward"), done=data.get("done", False))
        return StepResult(
            observation=NayanaObservation(**observation),
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
        response = requests.get(f"{self._http_base()}/manifest", timeout=30)
        response.raise_for_status()
        return response.json()

    def num_tasks(self, split):
        response = requests.post(
            f"{self._http_base()}/{ENV_NAME}/num_tasks",
            json={"split": split},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["num_tasks"]

    def get_task_range(self, split, start=0, stop=None):
        response = requests.post(
            f"{self._http_base()}/{ENV_NAME}/task_range",
            json={"split": split, "start": start, "stop": stop},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["tasks"]


def connect(url):
    return NayanaClient(
        base_url=url, connect_timeout_s=60, message_timeout_s=180
    ).sync()

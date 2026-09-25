"""Apply the example's train/eval admission policy without rewriting OpenEnv source."""
from functools import wraps
import os

from service_policy import admission, output_limit


def install():
    from openenv.harbor import rollout, serving

    if getattr(rollout.run_rollout, "_data_agent_policy", False):
        return
    original = rollout.run_rollout

    @wraps(original)
    async def run(**kwargs):
        dataset = kwargs.get("dataset", "")
        observer = kwargs.get("on_session_created")

        def session_created(session_id):
            session = kwargs["registry"].get(session_id)
            if session is None:
                raise RuntimeError("Rollout session disappeared before policy setup")
            session.metadata["max_output_tokens"] = output_limit(dataset)
            if observer is not None:
                observer(session_id)

        async with admission.slot(dataset):
            return await original(**{**kwargs, "on_session_created": session_created})

    run._data_agent_policy = True
    rollout.run_rollout = run
    # A protected Space's app endpoint cannot authenticate a sandbox's model
    # credential as an HF token. Publish only capture through the existing tunnel.
    public_url = serving.space_public_url
    serving.space_public_url = lambda: "" if os.environ.get("OPENENV_CAPTURE_TRANSPORT") == "tunnel" else public_url()

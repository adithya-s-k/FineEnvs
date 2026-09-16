"""Optional HF-signed-in demos; these conversations are separate from eval ledgers."""
import inspect
import json
import os
import time
from typing import get_type_hints

import gradio as gr

from inference_providers import catalog, credentials, visitor_token


def enabled():
    # Never expose Gradio's local mocked-login flow in an unconfigured deployment.
    return os.environ.get("SYSTEM") == "spaces" and all(os.environ.get(k) for k in
        ("SPACE_ID", "OAUTH_CLIENT_ID", "OAUTH_CLIENT_SECRET", "OAUTH_SCOPES", "OPENID_PROVIDER_URL"))


def provider_choices():
    try:
        providers = sorted({p for p, _ in catalog.rows()})
        return gr.update(choices=providers, value=None), gr.update(choices=[], value=None)
    except Exception:
        raise gr.Error("The HF model catalog is unavailable. Try loading it again.") from None


def model_choices(provider):
    if not provider:
        return gr.update(choices=[], value=None)
    try:
        models = sorted(m for p, m in catalog.rows() if p == provider)
        return gr.update(choices=models, value=None)
    except Exception:
        raise gr.Error("The HF model catalog is unavailable. Try loading it again.") from None


def model_details(provider, model):
    if not provider or not model:
        return "Choose a provider and model."
    try:
        row = catalog.rows()[(provider, model)]
    except (KeyError, ValueError):
        return "This model is no longer available from that provider. Refresh the catalog."
    price = row.get("pricing", {})
    text = "Tool calling supported. "
    if "input" in price and "output" in price:
        text += f"Per million tokens: ${price['input']:g} input · ${price['output']:g} output. "
    if row.get("context_length"):
        text += f"Context: {row['context_length']:,} tokens. "
    return text + "Usage is charged to your HF account."


def selected(oauth, provider, model):
    try:
        visitor_token(oauth)
        return catalog.select(provider, model)
    except ValueError as exc:
        raise gr.Error(str(exc)) from None


def controls():
    gr.LoginButton()
    gr.Markdown("Sign in and choose a model that can use tools. Inference uses your HF account's credits; "
                "a demo runs for at most 10 minutes and 17 agent turns.")
    refresh = gr.Button("Load available models")
    with gr.Row():
        provider = gr.Dropdown([], label="Inference provider", interactive=True)
        model = gr.Dropdown([], label="Model", interactive=True)
    detail = gr.Markdown("Choose a provider and model.")
    refresh.click(provider_choices, outputs=[provider, model], api_visibility="private")
    provider.change(model_choices, provider, model, api_visibility="private")
    model.change(model_details, [provider, model], detail, api_visibility="private")
    return provider, model


def blackbox(split, index, harness, provider, model, oauth_token: gr.OAuthToken):
    from environment_ui import blackbox_run, LOCAL
    target = selected(oauth_token, provider, model)
    with credentials.issue(oauth_token, target) as key:
        yield from blackbox_run(split, index, harness, LOCAL + "/hf-inference/v1", target, key)


def tool_schemas(env):
    """Derive the demo schema from the environment's existing declared tool surface."""
    from whitebox_bash.tools import specs_for
    from pydantic import ConfigDict, create_model
    tools, methods = [], {}
    for spec in specs_for("bash,seta"):
        method = getattr(env, spec.name)
        params = inspect.signature(method).parameters
        hints = get_type_hints(method)
        arguments = create_model(spec.name, __config__=ConfigDict(extra="forbid"), **{
            name: (hints[name], ... if p.default is inspect.Parameter.empty else p.default)
            for name, p in params.items()})
        tools.append({"type": "function", "function": {"name": spec.name, "description": spec.summary,
            "parameters": arguments.model_json_schema()}})
        methods[spec.name] = method
    return tools, methods


def whitebox(ui, split, index, provider, model, oauth, request):
    from openai import OpenAI
    from environment_ui import LOCAL
    target = selected(oauth, provider, model)
    # Own the same browser workspace lock as manual actions for the whole episode.
    prompt, _, _ = ui.start(split, index, request)
    session = ui.session(request)
    with session.lock:
        try:
            env = session.env
            if env is None:
                raise gr.Error("The workspace was closed. Start the agent again.")
            tools, methods = tool_schemas(env)
            messages = [{"role": "system", "content": "You are a data-analysis agent in an isolated workspace. "
                         "Use the tools to inspect the data and compute the answer. Call submit_solution with the answer itself."},
                        {"role": "user", "content": prompt}]
            deadline = time.monotonic() + 600
            yield "Starting the agent…", {"state": "running", "model": target}
            with credentials.issue(oauth, target) as key, OpenAI(base_url=LOCAL + "/hf-inference/v1", api_key=key,
                                                                timeout=120, max_retries=0) as client:
                for turn in range(17):
                    if time.monotonic() >= deadline:
                        break
                    response = client.chat.completions.create(model=target, messages=messages, tools=tools,
                        tool_choice="auto", max_tokens=4096, temperature=0.8,
                        timeout=min(120, deadline-time.monotonic()))
                    msg = response.choices[0].message
                    messages.append(msg.model_dump(exclude_none=True))
                    for call in msg.tool_calls or []:
                        name = call.function.name
                        try:
                            if name not in methods:
                                raise ValueError("Unknown tool")
                            args = json.loads(call.function.arguments)
                            if not isinstance(args, dict):
                                raise ValueError("Tool arguments must be an object")
                            result = str(methods[name](**args))
                        except (ValueError, TypeError):
                            result = "[error] Invalid tool name or arguments; use the provided schema."
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                        if env._reward is not None:
                            break
                    session.used_at = time.monotonic()
                    yield json.dumps(messages, indent=2, ensure_ascii=False)[-120000:], {
                        "state": "running", "model": target, "turns": turn + 1}
                    if not msg.tool_calls or env._reward is not None:
                        break
            reward = env.get_reward()
            yield json.dumps(messages, indent=2, ensure_ascii=False)[-120000:], {
                "state": "finished", "reward": reward, "model": target,
                "capture_level": "text", "training_eligible": False}
        except gr.Error:
            raise
        except Exception as exc:
            raise gr.Error(f"The demo stopped ({type(exc).__name__}). Check your HF inference credits or try another model.") from None
        finally:
            ui.dispose(session)
            session.used_at = time.monotonic()

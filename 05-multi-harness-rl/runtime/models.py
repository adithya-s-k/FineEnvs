"""Render the same non-thinking prompt in the inference engine and trainer."""
from pathlib import Path


def training_attention_backend(capability):
    major, _ = capability
    if major == 9:
        return "kernels-community/flash-attn3"
    if major == 8:
        return "kernels-community/flash-attn2"
    raise ValueError(f"Unqualified training GPU capability: {capability}")


def tokenizer_for(cfg, template_path=None):
    from copy import deepcopy
    from transformers import AutoTokenizer

    profile = cfg["profile"]
    tokenizer = AutoTokenizer.from_pretrained(profile["id"], revision=profile["revision"])
    if cfg["model"] == "lfm":
        template = tokenizer.chat_template
        old = '{{- "<|im_start|>assistant\\n<think>" -}}'
        if template.count(old) != 1:
            raise ValueError("LFM template changed; requalify the non-thinking prefix")
        tokenizer.chat_template = template.replace(old, '{{- "<|im_start|>assistant\\n<think></think>" -}}')
        from trl.chat_template_utils import lfm2_2_5_template
        tokenizer.response_template = deepcopy(lfm2_2_5_template)
    else:
        from trl.chat_template_utils import qwen3_5_template
        tokenizer.response_template = deepcopy(qwen3_5_template)
        tokenizer.chat_template = "{%- set enable_thinking = false -%}\n" + tokenizer.chat_template
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "Say OK."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False, preserve_thinking=True)
    tail = rendered.rsplit("<|im_start|>assistant", 1)[-1]
    if "</think>" not in tail or tail.count("<think>") != tail.count("</think>"):
        raise ValueError("Generation prompt still opens a thinking block")
    if template_path:
        Path(template_path).write_text(tokenizer.chat_template)
    return tokenizer


def serving_command(cfg, model, *, training, template_path, port=8000):
    p = cfg["profile"]
    cmd = ["vllm", "serve", model, "--host", "127.0.0.1", "--port", str(port),
           "--served-model-name", p["id"], "--dtype", "bfloat16", "--tensor-parallel-size", "1",
           "--data-parallel-size", "1" if training else str(cfg["eval_dp"]),
           "--max-model-len", str(cfg["max_model_len"]), "--gpu-memory-utilization", "0.85",
           "--enable-auto-tool-choice", "--tool-call-parser", p["tool_parser"],
           "--chat-template", str(template_path),
           "--default-chat-template-kwargs", '{"enable_thinking":false,"preserve_thinking":true}',
           "--generation-config", "vllm", "--override-generation-config",
           '{"temperature":0.8,"top_p":1.0,"top_k":-1}',
           "--logprobs-mode", "processed_logprobs", "--return-tokens-as-token-ids",
           "--enforce-eager", "--no-enable-prefix-caching"]
    if model == p["id"]:
        cmd += ["--revision", p["revision"]]
    if training:
        cmd += ["--weight-transfer-config", '{"backend":"nccl"}']
    if cfg["model"] == "qwen":
        cmd += ["--limit-mm-per-prompt", '{"image":0,"video":0}', "--gdn-prefill-backend", "triton"]
    return cmd


def check_visible_response(cfg, url):
    """Catch non-thinking outputs being swallowed by a reasoning parser."""
    from openai import OpenAI

    checks = []
    with OpenAI(base_url=url.rstrip("/") + "/v1", api_key="unused", timeout=120) as client:
        for stream in (False, True):
            response = client.chat.completions.create(model=cfg["profile"]["id"],
                messages=[{"role": "user", "content": "Reply with exactly OK."}],
                temperature=0, max_tokens=32, stream=stream)
            messages = ([c.choices[0].delta for c in response if c.choices] if stream else
                        [response.choices[0].message])
            content = "".join(m.content or "" for m in messages)
            reasoning = "".join(str((m.model_extra or {}).get(key) or "")
                                for m in messages for key in ("reasoning", "reasoning_content"))
            if not content.strip() or reasoning:
                raise ValueError(f"Non-thinking response was empty or hidden in reasoning (stream={stream})")
            checks.append({"stream": stream, "visible_chars": len(content), "reasoning_chars": len(reasoning)})
    return checks

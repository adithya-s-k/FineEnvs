"""Avoid grouping diverged tool histories as repeated initial prompts."""


def group_size(prompts, requested):
    if requested < 1:
        raise ValueError("Group size must be positive")
    if len(prompts) % requested:
        return 1
    for start in range(0, len(prompts), requested):
        if any(p != prompts[start] for p in prompts[start:start + requested]):
            return 1
    return requested


def protect_generation(trainer):
    client = trainer.vllm_generation.vllm_client
    def reset_prefix_cache():
        # vLLM acknowledges cache resets with an empty HTTP 200 body.
        response = client.session.post(f"{client.base_url}/reset_prefix_cache")
        response.raise_for_status()
    client.reset_prefix_cache = reset_prefix_cache
    generate = trainer.vllm_generation.generate
    def checked(*args, **kwargs):
        prompts = kwargs.get("prompts", args[0] if args else None)
        kwargs = {**kwargs, "num_generations": group_size(prompts, kwargs.get("num_generations", 1))}
        output = generate(*args, **kwargs)
        returned, completions, logprobs, ids = output
        if returned != prompts:
            raise ValueError("vLLM changed the supplied token prompts")
        for completion, lps, token_ids in zip(completions, logprobs, ids, strict=True):
            if len(completion) != len(lps) or token_ids != [[token] for token in completion]:
                raise ValueError("Sampled token/logprob IDs do not agree")
        return output
    trainer.vllm_generation.generate = checked

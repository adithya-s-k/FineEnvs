"""Protect native server generation's duplicate-prompt optimization after tools."""


def generation_group_size(prompts, requested):
    if requested<1:
        raise ValueError('generation group size must be positive')
    if len(prompts)%requested:
        return 1
    for start in range(0,len(prompts),requested):
        if any(prompt!=prompts[start] for prompt in prompts[start:start+requested]):
            return 1
    return requested

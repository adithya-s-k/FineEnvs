"""Candidate reward; action-count qualification is required before training."""
def shaped_reward(correctness, calls, *, count_verified=False, weight=0.1, budget=15):
    if correctness is None:
        return None
    if correctness not in (0, 1):
        raise ValueError('Expected binary verifier correctness')
    if not 0 <= weight <= 0.1 or budget <= 0:
        raise ValueError('Invalid efficiency parameters')
    if not count_verified or calls is None:
        return float(correctness)
    if type(calls) is not int or calls < 0:
        raise ValueError('Action count must be a nonnegative integer')
    if calls == 0:
        return float(correctness)
    return correctness * (1 + weight * budget / (budget + calls))

"""Match retained supervised spans to distinct engine call occurrences exactly."""


def audit_rows(prompt_ids, completions, masks, logprobs, calls):
    candidates, checks = [], []
    for row_index, (root, completion, mask, row_logprobs) in enumerate(zip(prompt_ids, completions, masks, logprobs, strict=True)):
        assert len(completion) == len(mask) == len(row_logprobs)
        assert set(mask) <= {0, 1}
        full = root + completion
        assert len(full) <= 131072
        offset = 0
        while offset < len(mask):
            if not mask[offset]:
                offset += 1
                continue
            end = offset + 1
            while end < len(mask) and mask[end]:
                end += 1
            keep, start = end - offset, len(root) + offset
            matches = [index for index, call in enumerate(calls)
                       if call['prompt_ids'] == full[:start]
                       and len(call['completion_ids']) >= keep
                       and (len(call['completion_ids']) == keep or end == len(mask))
                       and call['completion_ids'][:keep] == completion[offset:end]
                       and call['logprobs'][:keep] == row_logprobs[offset:end]]
            assert matches, f'row {row_index} supervision lacks exact engine provenance at {offset}'
            candidates.append(matches)
            offset = end
        assert all(p == 0 for p, m in zip(row_logprobs, mask, strict=True) if not m)
        checks.append({'supervised': sum(mask), 'context': len(mask) - sum(mask), 'tito_pass': True})
    # A truncated span may match several otherwise distinct calls. Find a complete
    # one-to-one assignment rather than consuming the first match greedily.
    assignments = {}
    def assign(span, visited):
        for call in candidates[span]:
            if call in visited:
                continue
            visited.add(call)
            if call not in assignments or assign(assignments[call], visited):
                assignments[call] = span
                return True
        return False
    for span in range(len(candidates)):
        assert assign(span, set()), f'supervised span {span} has no distinct engine call occurrence'
    return checks

"""Typed producer boundary for the archived native OpenCode wire format."""
from openenv.core.harness import TrainingTrace, TrainingTurn


def training_trace(result):
    if result is None:
        raise RuntimeError("No completed native rollout")
    if result.rollout_type != "train":
        raise ValueError("Native rollout has no training capture")
    if any(f.startswith("[FATAL]") for f in result.metadata.get("capture_findings", [])):
        raise ValueError("Native capture has fatal validation findings")
    return TrainingTrace(turns=[TrainingTurn(
        node_id=turn.capture_metadata["node_id"],
        prompt_token_ids=turn.prompt_token_ids,
        completion_token_ids=turn.completion_token_ids,
        per_token_logps=turn.per_token_logps,
        loss_mask=turn.loss_mask,
        request={"messages": turn.request_messages, "tools": turn.request_tools},
        response={"choices": [{"message": {"role": "assistant", "content": turn.text,
            "tool_calls": turn.tool_calls or None}, "finish_reason": turn.finish_reason}]},
        metadata=turn.capture_metadata,
    ) for turn in result.turns])

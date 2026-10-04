import pytest

from app.dockerfile import NotReplayable, check, plan


@pytest.mark.parametrize("instruction", ["USER app", 'SHELL ["/bin/bash", "-c"]', "user 1000"])
def test_execution_semantics_cannot_be_silently_discarded(instruction):
    dockerfile = f"FROM python:3.12\n{instruction}\nRUN echo setup\n"
    with pytest.raises(NotReplayable, match=instruction.split()[0].upper()):
        plan(dockerfile)
    ok, reason = check(dockerfile)
    assert not ok and instruction.split()[0].upper() in reason


def test_standard_single_stage_setup_remains_replayable():
    result = plan("FROM python:3.12\nWORKDIR /app\nENV MODE=test\nRUN echo setup\n")
    assert result.base == "python:3.12"
    assert result.steps == [("workdir", "/app"), ("env", {"MODE": "test"}), ("run", "echo setup")]

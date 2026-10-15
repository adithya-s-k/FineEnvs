import pytest

from app import catalog, seo_tasks


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.setattr(catalog, "_cached", lambda key, ttl, fn: fn())


def test_space_coordinates_are_checked_before_fetch_and_fields_are_safe(monkeypatch):
    monkeypatch.setattr(seo_tasks.spaces_live, "last_seen", lambda spec: {"task_api": {"env": "game", "splits": [{"name": "train", "num_tasks": 2}]}})
    calls = []
    def task(*args):
        calls.append(args)
        return {"task": {"task_id": "a", "prompt": "Look at this image.", "n_frames": 24, "provider": "mapillary",
                         "answer": "secret", "email": "private@example.com", "token": "secret"}}
    monkeypatch.setattr(seo_tasks.spaces_live, "task", task)
    for env, split, index in (("unknown", "train", 0), ("game", "unknown", 0), ("game", "train", 2), ("game", "train", -1)):
        with pytest.raises(seo_tasks.Unavailable) as exc:
            seo_tasks.space("org/sp", env, split, index)
        assert exc.value.status == 404
    assert not calls
    result = seo_tasks.space("org/sp", "game", "train", 0)
    assert result["fields"] == {"n_frames": 24, "provider": "mapillary"}
    assert result["brief"] == "Look at this image." and "secret" not in str(result)


def test_row_reads_are_anonymous_and_only_include_the_public_prompt(monkeypatch):
    from app.envs import rows
    def task(spec, config, split, index, token):
        assert token is None
        return {"title": "A task", "total": 3, "sections": [
            {"id": "task", "kind": "blocks", "body": [{"type": "custom", "text": "User: write three words.", "data": {"ignored": "raw"}}]},
            {"id": "grading", "body": "not prompt content"}]}
    monkeypatch.setattr(rows, "task", task)
    result = seo_tasks.row("org/ds", "train/0")
    assert result["brief"] == "User: write three words." and result["total"] == 3
    monkeypatch.setattr(rows, "task", lambda *args, **kwargs: {"restricted": True, "title": "Private"})
    with pytest.raises(seo_tasks.Unavailable) as exc:
        seo_tasks.row("org/ds", "train/0")
    assert exc.value.status == 404


def test_task_reads_refuse_excess_concurrency():
    for _ in range(4):
        assert seo_tasks._reads.acquire(blocking=False)
    try:
        with pytest.raises(seo_tasks.Unavailable) as exc:
            seo_tasks.read(("limited",), lambda: pytest.fail("should not start a read"))
        assert exc.value.status == 503
    finally:
        for _ in range(4):
            seo_tasks._reads.release()


def test_ranges_require_real_counts_and_deduplicate():
    task_api = {"environments": [{"env": "game", "splits": [
        {"name": "train", "num_tasks": 2}, {"name": "train", "num_tasks": 2},
        {"name": "unknown", "num_tasks": None}, {"name": "bad", "num_tasks": True}]}]}
    assert seo_tasks.ranges(task_api) == [["game", "train", 2]]

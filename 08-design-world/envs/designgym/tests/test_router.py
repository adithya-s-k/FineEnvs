"""The multi-worker router (server/router.py): every request reaches the worker that holds its state."""

from __future__ import annotations

from designgym.server.router import Router

W = ["http://127.0.0.1:9001", "http://127.0.0.1:9002", "http://127.0.0.1:9003"]


def test_episode_requests_follow_the_worker_named_in_the_id():
    r = Router(W)
    assert r.for_path("/api/episodes/w2-3fa2b9c0d1e2/screen.png") == 2
    assert r.for_path("/api/episodes/w1-abc/ws") == 1
    assert r.for_path("/api/episodes/w9-abc") == 0  # no such worker: the default
    assert r.for_path("/design_canvas/splits") is None  # stateless: any worker


def test_client_chosen_episode_ids_are_learned_from_replies():
    r = Router(W)
    r.learn(1, b'{"type":"observation","data":{"metadata":{"episode_id":"my-run-7"}}}')
    assert r.for_path("/api/episodes/my-run-7/render.png") == 1


def test_the_playground_stays_on_one_worker():
    r = Router(W)
    assert {r.for_path(p) for p in ("/web/", "/gradio_api/queue/join", "/web/theme.css")} == {0}


def test_new_sessions_go_to_the_least_busy_worker():
    r = Router(W)
    r.load = [3, 0, 2]
    assert r.least_busy() == 1
    r.load = [1, 1, 1]
    assert {r.least_busy() for _ in range(6)} == {0, 1, 2}  # ties rotate


def test_the_browsers_host_reaches_the_worker():
    # the playground builds its URLs from Host: the worker's own 127.0.0.1 address breaks the page behind a proxy
    from designgym.server.router import HOP

    assert "host" not in HOP and {"connection", "upgrade", "content-length"} <= HOP

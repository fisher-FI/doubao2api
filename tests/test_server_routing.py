"""Integration tests for server backend routing (no real network/browser)."""
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import doubao2api.unified_server as us
from doubao2api.client import (
    CompletionChunk,
    CompletionResult,
    GeneratedVideo,
    VideoGenerationResult,
)
from doubao2api.volcano import VolcanoClient


def _poll_task(c, task_id, timeout=5.0):
    """Poll an async video task until it reaches a terminal state."""
    deadline = time.time() + timeout
    resp = None
    while time.time() < deadline:
        resp = c.get(f"/v1/video/generations/async/{task_id}").json()
        if resp["status"] in ("succeeded", "failed", "cancelled"):
            return resp
        time.sleep(0.05)
    return resp


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """Create the app with two fake free accounts + a fake volcano key."""
    acc_dir = tmp_path / "accounts"
    acc_dir.mkdir()
    for i in range(2):
        (acc_dir / f"acct{i}.json").write_text(
            json.dumps({"cookies": {"sessionid": f"s{i}"}, "params": {}}),
            encoding="utf-8",
        )
    monkeypatch.setenv("DOUBAO_ACCOUNTS_DIR", str(acc_dir))
    monkeypatch.delenv("VOLC_API_KEY", raising=False)
    monkeypatch.setenv("VOLC_API_KEY", "test-volc-key")

    from doubao2api.browser_client import BrowserClient
    from doubao2api import DoubaoChatClient

    app = None
    with patch.object(BrowserClient, "start", new=AsyncMock(return_value=None)), \
         patch.object(BrowserClient, "is_ready", new_callable=lambda: property(lambda self: False)):
        app = us.create_app(api_key=None)
        with TestClient(app) as c:
            yield c


def test_health_reports_pool_and_volcano(app_client):
    resp = app_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pool_ready"] is True
    assert data["pool_total"] == 2
    assert data["pool_healthy"] == 2
    assert data["volcano_configured"] is True


def test_models_includes_volc_when_configured(app_client):
    resp = app_client.get("/v1/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "doubao" in ids
    assert "doubao-video" in ids
    assert "volc-chat" in ids
    assert "volc-video" in ids


def test_chat_routes_to_volcano_backend(app_client):
    async def _fake_stream(*args, **kwargs):
        yield CompletionChunk(text="volcano-hello", conversation_id="c1")
        yield CompletionChunk(is_done=True)

    with patch.object(VolcanoClient, "chat_stream_completion", new=_fake_stream):
        resp = app_client.post(
            "/v1/chat/completions",
            json={"model": "volc-chat", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "volcano-hello"
    assert data["model"] == "volc-chat"


def test_chat_unknown_model_still_400(app_client):
    resp = app_client.post(
        "/v1/chat/completions",
        json={"model": "nope-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 400


def test_video_routes_to_free_pool(app_client):
    from doubao2api import DoubaoChatClient

    async def _fake_video(*args, **kwargs):
        return VideoGenerationResult(
            videos=[GeneratedVideo(video_url="https://cdn/v.mp4", duration=5.0)],
            prompt=kwargs.get("prompt", ""),
        )

    with patch.object(DoubaoChatClient, "generate_video", new=_fake_video):
        resp = app_client.post("/v1/video/generations", json={"prompt": "a cat"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["data"][0]["video_url"] == "https://cdn/v.mp4"


def test_admin_accounts_listing(app_client):
    resp = app_client.get("/admin/api/accounts")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    names = {a["name"] for a in data["accounts"]}
    assert names == {"acct0", "acct1"}


def test_admin_toggle_account(app_client):
    resp = app_client.post("/admin/api/accounts/acct0/toggle")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False

    listing = app_client.get("/admin/api/accounts").json()
    acct0 = next(a for a in listing["accounts"] if a["name"] == "acct0")
    assert acct0["enabled"] is False
    assert acct0["is_healthy"] is False
    assert listing["healthy"] == 1


def test_admin_channel_status(app_client):
    resp = app_client.get("/admin/api/channel")
    assert resp.status_code == 200
    data = resp.json()
    assert data["free_accounts"]["configured"] is True
    assert data["volcano"]["configured"] is True


@pytest.fixture
def xyq_app_client(tmp_path, monkeypatch):
    """App with a fake xiaoyquee cookies dir; engine.run patched to return a real file."""
    monkeypatch.delenv("VOLC_API_KEY", raising=False)
    cookies_dir = tmp_path / "xyq_cookies"
    cookies_dir.mkdir()
    (cookies_dir / "test_cookie.json").write_text(json.dumps({"sessionid": "xyq"}), encoding="utf-8")
    out_dir = tmp_path / "xyq_downloads"
    out_dir.mkdir()
    fake_mp4 = out_dir / "cat_10s_x.mp4"
    fake_mp4.write_bytes(b"fake mp4 bytes")
    monkeypatch.setenv("XYQ_COOKIES_DIR", str(cookies_dir))
    monkeypatch.setenv("XYQ_OUTPUT_DIR", str(out_dir))

    from doubao2api.browser_client import BrowserClient
    import doubao2api.xiaoyunque_engine as engine

    async def _fake_run(**kwargs):
        return str(fake_mp4)

    with patch.object(BrowserClient, "start", new=AsyncMock(return_value=None)), \
         patch.object(BrowserClient, "is_ready", new_callable=lambda: property(lambda self: False)), \
         patch.object(engine, "run", new=_fake_run):
        app = us.create_app(api_key=None)
        with TestClient(app) as c:
            yield c


def test_xyq_video_generation_end_to_end(xyq_app_client):
    c = xyq_app_client
    # models list includes xyq when cookies present
    ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
    assert "xyq-video" in ids and "xyq-video-pro" in ids

    resp = c.post("/v1/video/generations", json={
        "prompt": "a cat", "model": "xyq-video", "duration": 10,
    })
    assert resp.status_code == 200, resp.text
    url = resp.json()["data"][0]["video_url"]
    assert url.startswith("/xyq-files/")

    # generated file is served
    file_resp = c.get(url)
    assert file_resp.status_code == 200
    assert b"fake mp4" in file_resp.content


def test_xyq_admin_endpoints(xyq_app_client):
    c = xyq_app_client
    status = c.get("/admin/api/xyq").json()
    assert status["configured"] is True
    assert status["cookies"] == ["test_cookie.json"]
    assert "xyq-video-pro" in status["models"]

    channel = c.get("/admin/api/channel").json()
    assert channel["xiaoyunque"]["configured"] is True

    # delete cookie -> channel becomes unconfigured
    r = c.delete("/admin/api/xyq/cookies/test_cookie")
    assert r.status_code == 200
    assert r.json()["cookies"] == []
    assert c.get("/admin/api/channel").json()["xiaoyunque"]["configured"] is False


def test_xyq_video_without_cookies_503(tmp_path, monkeypatch):
    monkeypatch.delenv("VOLC_API_KEY", raising=False)
    cookies_dir = tmp_path / "empty_xyq"
    cookies_dir.mkdir()
    monkeypatch.setenv("XYQ_COOKIES_DIR", str(cookies_dir))
    monkeypatch.setenv("XYQ_OUTPUT_DIR", str(tmp_path / "out"))

    from doubao2api.browser_client import BrowserClient
    with patch.object(BrowserClient, "start", new=AsyncMock(return_value=None)), \
         patch.object(BrowserClient, "is_ready", new_callable=lambda: property(lambda self: False)):
        app = us.create_app(api_key=None)
        with TestClient(app) as c:
            resp = c.post("/v1/video/generations", json={"prompt": "x", "model": "xyq-video"})
            assert resp.status_code == 503


# ── Async video task API ──


def test_async_video_task_success(app_client):
    from doubao2api import DoubaoChatClient

    async def _fake_video(*args, **kwargs):
        return VideoGenerationResult(
            videos=[GeneratedVideo(video_url="https://cdn/async.mp4", duration=5.0)],
            prompt=kwargs.get("prompt", ""),
        )

    with patch.object(DoubaoChatClient, "generate_video", new=_fake_video):
        resp = app_client.post("/v1/video/generations/async", json={"prompt": "a cat"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "pending"
    assert body["task_id"].startswith("videotask-")

    final = _poll_task(app_client, body["task_id"])
    assert final["status"] == "succeeded"
    assert final["data"][0]["video_url"] == "https://cdn/async.mp4"

    # appears in the task list
    listing = app_client.get("/v1/video/tasks").json()["data"]
    assert any(t["task_id"] == body["task_id"] for t in listing)


def test_async_video_task_failure(app_client):
    from doubao2api import DoubaoChatClient

    async def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    with patch.object(DoubaoChatClient, "generate_video", new=_boom):
        resp = app_client.post("/v1/video/generations/async", json={"prompt": "cat"})
    task_id = resp.json()["task_id"]

    final = _poll_task(app_client, task_id)
    assert final["status"] == "failed"
    assert "boom" in final["error"]


def test_async_video_missing_prompt_400(app_client):
    resp = app_client.post("/v1/video/generations/async", json={})
    assert resp.status_code == 400


def test_async_video_unknown_task_404(app_client):
    resp = app_client.get("/v1/video/generations/async/videotask-nonexistent")
    assert resp.status_code == 404


def test_async_video_unconfigured_channel_fails_fast(tmp_path, monkeypatch):
    """volc-* without API key must fail at submit time, not consume a slot."""
    monkeypatch.delenv("VOLC_API_KEY", raising=False)
    monkeypatch.delenv("XYQ_COOKIES_DIR", raising=False)
    empty = tmp_path / "empty_xyq"
    empty.mkdir()
    monkeypatch.setenv("XYQ_COOKIES_DIR", str(empty))
    monkeypatch.delenv("DOUBAO_ACCOUNTS_DIR", raising=False)

    from doubao2api.browser_client import BrowserClient
    with patch.object(BrowserClient, "start", new=AsyncMock(return_value=None)), \
         patch.object(BrowserClient, "is_ready", new_callable=lambda: property(lambda self: False)):
        app = us.create_app(api_key=None)
        with TestClient(app) as c:
            resp = c.post("/v1/video/generations/async",
                          json={"prompt": "x", "model": "volc-video"})
            assert resp.status_code == 503


def test_xyq_async_video_flow(xyq_app_client):
    c = xyq_app_client
    resp = c.post("/v1/video/generations/async", json={
        "prompt": "sunset", "model": "xyq-video", "duration": 15,
    })
    assert resp.status_code == 200
    final = _poll_task(c, resp.json()["task_id"])
    assert final["status"] == "succeeded"
    assert final["data"][0]["video_url"].startswith("/xyq-files/")
    assert final["data"][0]["duration"] == 15.0


def test_xyq_image_to_video_end_to_end(xyq_app_client, monkeypatch):
    import doubao2api.backends as backends_mod
    import doubao2api.xiaoyunque_engine as engine

    async def _fake_download(url, max_mb=20.0):
        return b"imgbytes", "cat.png"

    monkeypatch.setattr(backends_mod, "download_image_bytes", _fake_download)

    captured = {}

    async def _fake_run(**kwargs):
        captured.update(kwargs)
        return None  # will produce failed task, enough to assert wiring

    with patch.object(engine, "run", new=_fake_run):
        resp = xyq_app_client.post("/v1/video/generations/async", json={
            "prompt": "make it move", "model": "xyq-video",
            "image_url": "https://example.com/cat.png",
        })
        assert resp.status_code == 200
        final = _poll_task(xyq_app_client, resp.json()["task_id"])

    assert final["status"] == "failed"  # engine mocked to fail after capture
    refs = captured.get("ref_images")
    assert refs and len(refs) == 1 and refs[0].endswith(".png")


def test_video_task_delete(app_client):
    resp = app_client.delete("/v1/video/tasks/videotask-nope")
    assert resp.status_code == 404


# ── Multipart upload / first-last-frame API ──


def test_multipart_image_to_video_doubao(app_client):
    from doubao2api import DoubaoChatClient

    async def _fake_upload(*args, **kwargs):
        return {"uri": "tos-multipart-1", "cdn_url": "https://cdn/x.png"}

    async def _fake_video(*args, **kwargs):
        assert kwargs.get("ref_image_key") == "tos-multipart-1"
        return VideoGenerationResult(
            videos=[GeneratedVideo(video_url="https://cdn/mp.mp4", duration=5.0)],
            prompt=kwargs.get("prompt", ""),
        )

    with patch.object(DoubaoChatClient, "upload_image", new=_fake_upload), \
         patch.object(DoubaoChatClient, "generate_video", new=_fake_video):
        resp = app_client.post(
            "/v1/video/generations",
            data={"prompt": "make it move"},
            files={"file": ("cat.png", b"fakepng", "image/png")},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"][0]["video_url"] == "https://cdn/mp.mp4"


def test_multipart_two_files_map_to_first_last(xyq_app_client):
    import doubao2api.xiaoyunque_engine as engine

    captured = {}

    async def _run(**kwargs):
        captured.update(kwargs)
        return None

    with patch.object(engine, "run", new=_run):
        resp = xyq_app_client.post(
            "/v1/video/generations/async",
            data={"prompt": "morph", "model": "xyq-video"},
            files=[
                ("images", ("first.png", b"aaa", "image/png")),
                ("images", ("last.png", b"bbb", "image/png")),
            ],
        )
    assert resp.status_code == 200, resp.text
    final = _poll_task(xyq_app_client, resp.json()["task_id"])
    assert final["status"] == "failed"  # engine mocked to fail after capture
    refs = captured.get("ref_images")
    assert refs and len(refs) == 2


def test_camera_movement_reaches_free_backend(app_client):
    from doubao2api import DoubaoChatClient

    seen = {}

    async def _fake_video(*args, **kwargs):
        seen.update(kwargs)
        return VideoGenerationResult(
            videos=[GeneratedVideo(video_url="https://cdn/cam.mp4", duration=5.0)],
            prompt="p",
        )

    with patch.object(DoubaoChatClient, "generate_video", new=_fake_video):
        resp = app_client.post("/v1/video/generations", json={
            "prompt": "city flythrough", "camera_movement": "推镜头",
        })
    assert resp.status_code == 200
    assert seen.get("camera_movement") == "推镜头"


def test_last_frame_on_doubao_returns_502(app_client):
    resp = app_client.post("/v1/video/generations", json={
        "prompt": "x",
        "first_frame_url": "https://x/a.png",
        "last_frame_url": "https://x/b.png",
    })
    assert resp.status_code == 502
    assert "last-frame" in resp.json()["error"]["message"] or \
           "last-frame" in resp.text

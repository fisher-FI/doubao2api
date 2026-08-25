"""Integration tests for server backend routing (no real network/browser)."""
import json
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
    assert data["volcano"]["api_key_set"] is True

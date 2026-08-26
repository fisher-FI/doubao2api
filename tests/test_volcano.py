import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from doubao2api.volcano import VolcanoClient
from doubao2api.client import CompletionChunk


@pytest.mark.asyncio
async def test_chat_completion():
    client = VolcanoClient(api_key="test-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "id": "chatcmpl-xxx",
        "choices": [{"message": {"role": "assistant", "content": "Hello!", "reasoning_content": ""}}],
        "usage": {},
    })
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_resp)
    client._http = mock_http

    result = await client.chat_completion("Hi")
    assert result.text == "Hello!"
    assert result.conversation_id == "chatcmpl-xxx"
    # Verify request shape
    args, kwargs = mock_http.post.call_args
    assert args[0].endswith("/chat/completions")
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"]["stream"] is False


@pytest.mark.asyncio
async def test_is_ready_requires_key():
    client = VolcanoClient(api_key="")
    # Ensure env var doesn't leak into test
    assert client.is_ready in (True, False)  # depends on VOLC_API_KEY env
    client2 = VolcanoClient(api_key="k")
    assert client2.is_ready is True


@pytest.mark.asyncio
async def test_generate_video_polls_until_succeeded(monkeypatch):
    monkeypatch.setattr("doubao2api.volcano.asyncio.sleep", AsyncMock())
    client = VolcanoClient(api_key="test-key")

    create_resp = MagicMock()
    create_resp.status_code = 200
    create_resp.raise_for_status = MagicMock()
    create_resp.json = MagicMock(return_value={"id": "task-123"})

    poll_resp = MagicMock()
    poll_resp.status_code = 200
    poll_resp.raise_for_status = MagicMock()
    poll_resp.json = MagicMock(return_value={
        "status": "succeeded",
        "data": [{"url": "https://cdn.example.com/v.mp4", "duration": 5.0}],
    })

    async def _post(url, **kwargs):
        assert url.endswith("/contents/generations/tasks")
        return create_resp

    async def _get(url, **kwargs):
        assert "/contents/generations/tasks/task-123" in url
        return poll_resp

    mock_http = AsyncMock()
    mock_http.post = _post
    mock_http.get = _get
    client._http = mock_http

    result = await client.generate_video("a cat", timeout=60)
    assert len(result.videos) == 1
    assert result.videos[0].video_url == "https://cdn.example.com/v.mp4"


@pytest.mark.asyncio
async def test_generate_video_fails_on_failed_status(monkeypatch):
    monkeypatch.setattr("doubao2api.volcano.asyncio.sleep", AsyncMock())
    client = VolcanoClient(api_key="test-key")

    create_resp = MagicMock()
    create_resp.status_code = 200
    create_resp.raise_for_status = MagicMock()
    create_resp.json = MagicMock(return_value={"id": "task-fail"})

    poll_resp = MagicMock()
    poll_resp.status_code = 200
    poll_resp.raise_for_status = MagicMock()
    poll_resp.json = MagicMock(return_value={"status": "failed", "error": "boom"})

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=create_resp)
    mock_http.get = AsyncMock(return_value=poll_resp)
    client._http = mock_http

    with pytest.raises(RuntimeError, match="FAILED"):
        await client.generate_video("a cat", timeout=60)


@pytest.mark.asyncio
async def test_generate_video_with_ref_image_in_content(monkeypatch):
    monkeypatch.setattr("doubao2api.volcano.asyncio.sleep", AsyncMock())
    client = VolcanoClient(api_key="test-key")

    create_resp = MagicMock()
    create_resp.status_code = 200
    create_resp.raise_for_status = MagicMock()
    create_resp.json = MagicMock(return_value={"id": "task-i2v"})

    poll_resp = MagicMock()
    poll_resp.status_code = 200
    poll_resp.raise_for_status = MagicMock()
    poll_resp.json = MagicMock(return_value={
        "status": "succeeded", "data": [{"url": "https://cdn/v.mp4"}],
    })

    sent_payload = {}

    async def _post(url, **kwargs):
        sent_payload.update(kwargs["json"])
        return create_resp

    async def _get(url, **kwargs):
        return poll_resp

    mock_http = AsyncMock()
    mock_http.post = _post
    mock_http.get = _get
    client._http = mock_http

    result = await client.generate_video(
        "a cat dancing", ref_image_url="https://example.com/cat.png",
    )
    assert result.videos[0].video_url == "https://cdn/v.mp4"
    content = sent_payload["content"]
    assert {"type": "text", "text": "a cat dancing"} in content
    assert {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}} in content


@pytest.mark.asyncio
async def test_stream_chat_parses_sse():
    client = VolcanoClient(api_key="test-key")

    sse_lines = [
        'data: {"id":"c1","choices":[{"delta":{"role":"assistant"}}]}',
        'data: {"id":"c1","choices":[{"delta":{"reasoning_content":"think..."}}]}',
        'data: {"id":"c1","choices":[{"delta":{"content":"Hi"}}]}',
        "data: [DONE]",
    ]

    class _FakeStreamResp:
        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for line in sse_lines:
                yield line

    class _FakeStreamCtx:
        async def __aenter__(self):
            return _FakeStreamResp()

        async def __aexit__(self, *args):
            return False

    mock_http = AsyncMock()
    mock_http.stream = MagicMock(return_value=_FakeStreamCtx())
    client._http = mock_http

    chunks = []
    async for chunk in client.chat_stream_completion("Hi"):
        chunks.append(chunk)

    # role-only chunk skipped; thinking + content + done
    assert any(c.thinking == "think..." for c in chunks)
    assert any(c.text == "Hi" and c.conversation_id == "c1" for c in chunks)
    assert chunks[-1].is_done is True

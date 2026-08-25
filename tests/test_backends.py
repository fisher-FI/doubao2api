import pytest

from doubao2api.backends import FreeAccountBackend, VolcanoBackend, BrowserBackend
from doubao2api.account_pool import AccountEntry, AccountPool
from doubao2api.client import (
    CompletionChunk,
    CompletionResult,
    ImageGenerationResult,
    GeneratedImage,
    VideoGenerationResult,
    GeneratedVideo,
)
from unittest.mock import AsyncMock, MagicMock


def _make_pool_with_entry(entry: AccountEntry) -> AccountPool:
    pool = MagicMock(spec=AccountPool)
    pool.select = MagicMock(return_value=entry)
    pool.healthy_count = 1
    return pool


@pytest.mark.asyncio
async def test_free_account_backend_chat_returns_dict():
    entry = AccountEntry(name="test", session_file="test.json", client=MagicMock())
    entry.client.chat_completion = AsyncMock(
        return_value=CompletionResult(text="Hi", conversation_id="conv-1")
    )
    pool = _make_pool_with_entry(entry)
    backend = FreeAccountBackend(pool)

    result = await backend.chat("Hello")
    assert result["text"] == "Hi"
    assert result["conversation_id"] == "conv-1"
    assert entry.success_count == 1


@pytest.mark.asyncio
async def test_free_account_backend_chat_completion_stream():
    entry = AccountEntry(name="test", session_file="test.json", client=MagicMock())

    async def _mock_stream(*args, **kwargs):
        yield CompletionChunk(text="Hello", conversation_id="c1")
        yield CompletionChunk(is_done=True)

    entry.client.chat_stream_completion = _mock_stream
    pool = _make_pool_with_entry(entry)
    backend = FreeAccountBackend(pool)

    events = []
    async for event in backend.chat_completion("Hi"):
        events.append(event)
    assert len(events) == 1
    assert events[0]["_event"] == "CHUNK_DELTA"
    assert events[0]["text"] == "Hello"
    assert events[0]["conversation_id"] == "c1"


@pytest.mark.asyncio
async def test_free_account_backend_thinking_toggle():
    """Thinking chunks must be wrapped in block_type=10040 toggles."""
    entry = AccountEntry(name="test", session_file="test.json", client=MagicMock())

    async def _mock_stream(*args, **kwargs):
        yield CompletionChunk(thinking="let me think", conversation_id="c1")
        yield CompletionChunk(text="answer", conversation_id="c1")
        yield CompletionChunk(is_done=True)

    entry.client.chat_stream_completion = _mock_stream
    pool = _make_pool_with_entry(entry)
    backend = FreeAccountBackend(pool)

    events = [ev async for ev in backend.chat_completion("q")]
    # toggle-in, thinking delta, toggle-out, content delta
    assert events[0]["content"]["content_block"][0]["block_type"] == 10040
    assert events[1]["text"] == "let me think"
    assert events[2]["content"]["content_block"][0]["block_type"] == 10040
    assert events[3]["text"] == "answer"


@pytest.mark.asyncio
async def test_free_account_backend_error_event_stops_stream():
    entry = AccountEntry(name="test", session_file="test.json", client=MagicMock())

    async def _mock_stream(*args, **kwargs):
        yield CompletionChunk(error_code=710022004, error_msg="risk")

    entry.client.chat_stream_completion = _mock_stream
    pool = _make_pool_with_entry(entry)
    backend = FreeAccountBackend(pool)

    events = []
    async for event in backend.chat_completion("q"):
        events.append(event)
    assert len(events) == 1
    assert events[0]["error_code"] == 710022004
    # record_failure should have marked the account
    backend.record_failure(710022004)
    assert entry.needs_captcha is True


@pytest.mark.asyncio
async def test_free_account_backend_generate_image():
    entry = AccountEntry(name="test", session_file="test.json", client=MagicMock())
    entry.client.generate_image = AsyncMock(return_value=ImageGenerationResult(
        images=[GeneratedImage(ori_url="https://example.com/img.png", width=512, height=512)],
        prompt="test",
    ))
    pool = _make_pool_with_entry(entry)
    backend = FreeAccountBackend(pool)

    result = await backend.generate_image("test", "1:1")
    assert result["images"][0]["url"] == "https://example.com/img.png"


@pytest.mark.asyncio
async def test_free_account_backend_generate_video_marks_quota():
    entry = AccountEntry(name="test", session_file="test.json", client=MagicMock())
    entry.client.generate_video = AsyncMock(return_value=VideoGenerationResult(
        videos=[GeneratedVideo(video_url="https://cdn/v.mp4", duration=5.0)],
        prompt="cat",
    ))
    pool = _make_pool_with_entry(entry)
    backend = FreeAccountBackend(pool)

    result = await backend.generate_video("cat", "16:9")
    assert result["videos"][0]["video_url"] == "https://cdn/v.mp4"
    assert entry.daily_quota_used == 1


@pytest.mark.asyncio
async def test_extract_conversation_id_from_event():
    backend = FreeAccountBackend(MagicMock(spec=AccountPool))
    assert backend.extract_conversation_id({"conversation_id": "c9"}) == "c9"
    assert backend.extract_conversation_id({}) is None


@pytest.mark.asyncio
async def test_volcano_backend_generate_video_dict():
    mock_client = MagicMock()
    mock_client.generate_video = AsyncMock(return_value=VideoGenerationResult(
        videos=[GeneratedVideo(video_url="https://v.mp4")],
        prompt="p",
    ))
    backend = VolcanoBackend(mock_client)
    result = await backend.generate_video("p")
    assert result["videos"][0]["video_url"] == "https://v.mp4"


@pytest.mark.asyncio
async def test_volcano_backend_music_raises():
    backend = VolcanoBackend(MagicMock())
    with pytest.raises(RuntimeError, match="Music"):
        await backend.generate_music("song")

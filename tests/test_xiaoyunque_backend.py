"""Tests for the Xiaoyunque (剪映 Seedance 2.0) backend."""
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from doubao2api.backends import XiaoyunqueBackend


def _make_backend(tmp_path, run_impl=None):
    engine = MagicMock()
    engine.config = MagicMock(cookies_dir=str(tmp_path / "cookies"), output_dir=str(tmp_path / "out"))
    engine.get_token_files = MagicMock(return_value=[])
    if run_impl is not None:
        engine.run = AsyncMock(side_effect=run_impl)
    else:
        engine.run = AsyncMock(return_value=None)
    backend = XiaoyunqueBackend(
        cookies_dir=str(tmp_path / "cookies"),
        output_dir=str(tmp_path / "out"),
        headless=True,
        engine=engine,
    )
    return backend, engine


@pytest.mark.asyncio
async def test_generate_video_returns_file_url(tmp_path):
    out_file = tmp_path / "out" / "cat_10s_1.mp4"
    out_file.parent.mkdir(parents=True)
    out_file.write_bytes(b"fake")

    async def _run(**kwargs):
        return str(out_file)

    backend, engine = _make_backend(tmp_path, run_impl=_run)
    result = await backend.generate_video("a cat", ratio="16:9", duration=10, quality="fast")

    assert result["videos"][0]["video_url"] == "/xyq-files/cat_10s_1.mp4"
    assert result["videos"][0]["duration"] == 10.0
    # engine called with coerced args
    kwargs = engine.run.call_args.kwargs
    assert kwargs["model"] == "fast"
    assert kwargs["ratio"] == "16:9"
    assert kwargs["ref_images"] is None  # text-to-video supported


@pytest.mark.asyncio
async def test_ratio_coerced_for_xyq(tmp_path):
    async def _run(**kwargs):
        return None

    backend, engine = _make_backend(tmp_path, run_impl=_run)
    with pytest.raises(RuntimeError):
        await backend.generate_video("cat", ratio="4:3")
    kwargs = engine.run.call_args.kwargs
    assert kwargs["ratio"] == "16:9"  # 4:3 not supported -> coerced


@pytest.mark.asyncio
async def test_quality_mapping(tmp_path):
    async def _run(**kwargs):
        return None

    backend, engine = _make_backend(tmp_path, run_impl=_run)
    with pytest.raises(RuntimeError):
        await backend.generate_video("cat", quality="2.0")
    assert engine.run.call_args.kwargs["model"] == "2.0"


@pytest.mark.asyncio
async def test_invalid_duration_falls_back_to_10(tmp_path):
    async def _run(**kwargs):
        return None

    backend, engine = _make_backend(tmp_path, run_impl=_run)
    with pytest.raises(RuntimeError):
        await backend.generate_video("cat", duration=7)
    assert engine.run.call_args.kwargs["duration"] == 10


@pytest.mark.asyncio
async def test_engine_failure_raises(tmp_path):
    async def _run(**kwargs):
        return None  # engine returns None on failure

    backend, _ = _make_backend(tmp_path, run_impl=_run)
    with pytest.raises(RuntimeError, match="no result"):
        await backend.generate_video("cat")


@pytest.mark.asyncio
async def test_is_ready_depends_on_cookies(tmp_path):
    backend, engine = _make_backend(tmp_path)
    (tmp_path / "cookies").mkdir(exist_ok=True)
    assert backend.is_ready is False
    engine.get_token_files = MagicMock(return_value=["a.json", "b.json"])
    assert backend.is_ready is True
    assert backend.cookie_names() == ["a.json", "b.json"]


@pytest.mark.asyncio
async def test_unsupported_capabilities_raise(tmp_path):
    backend, _ = _make_backend(tmp_path)
    for coro in (
        backend.chat("hi"),
        backend.generate_image("p"),
        backend.generate_music("m"),
        backend.upload_file(b"x", "f.txt"),
    ):
        with pytest.raises(RuntimeError, match="only supports video"):
            await coro


@pytest.mark.asyncio
async def test_image_to_video_passes_ref_images_and_cleans_up(tmp_path, monkeypatch):
    import doubao2api.backends as backends_mod

    async def _fake_download(url, max_mb=20.0):
        return b"fakeimg", "ref.png"

    monkeypatch.setattr(backends_mod, "download_image_bytes", _fake_download)

    captured = {}

    async def _run(**kwargs):
        captured.update(kwargs)
        # ref file must exist while engine runs
        assert kwargs["ref_images"] and kwargs["ref_images"][0].endswith(".png")
        with open(kwargs["ref_images"][0], "rb") as f:
            assert f.read() == b"fakeimg"
        return None

    backend, engine = _make_backend(tmp_path, run_impl=_run)
    with pytest.raises(RuntimeError, match="no result"):
        await backend.generate_video("cat", ref_image_url="https://x/a.png")

    assert engine.run.call_args.kwargs["ref_images"] is not None
    # temp file cleaned up after run
    import os
    assert not os.path.exists(captured["ref_images"][0])


@pytest.mark.asyncio
async def test_text_to_video_has_no_ref_images(tmp_path):
    async def _run(**kwargs):
        assert kwargs["ref_images"] is None
        return None

    backend, _ = _make_backend(tmp_path, run_impl=_run)
    with pytest.raises(RuntimeError, match="no result"):
        await backend.generate_video("cat")


@pytest.mark.asyncio
async def test_first_and_last_frame_pass_two_ordered_images(tmp_path, monkeypatch):
    import doubao2api.backends as backends_mod
    import os

    async def _fake_download(url, max_mb=20.0):
        return b"frame", "f.png"

    monkeypatch.setattr(backends_mod, "download_image_bytes", _fake_download)

    async def _run(**kwargs):
        refs = kwargs["ref_images"]
        assert refs and len(refs) == 2  # first + last in order
        return None

    backend, _ = _make_backend(tmp_path, run_impl=_run)
    with pytest.raises(RuntimeError, match="no result"):
        await backend.generate_video(
            "cat", first_frame_url="https://x/first.png",
            last_frame_url="https://x/last.png",
        )
    # both temp files cleaned
    assert not any(f.startswith("_ref_") for f in os.listdir(str(tmp_path / "out")))

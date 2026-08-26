"""Backend abstraction layer for Doubao API.

Provides three backends sharing one dict-based interface (compatible with
the existing ``BrowserClient`` method signatures so ``unified_server.py``
endpoints need minimal changes):

- ``BrowserBackend``: wraps the existing single-account ``BrowserClient``.
- ``FreeAccountBackend``: multi-account pool built on ``DoubaoChatClient``.
- ``VolcanoBackend``: official Volcano Engine API.

All backends expose:
  chat_completion(text, conversation_id, bot_id, use_deep_think) -> AsyncIterator[dict]
  chat(text, ...) -> {"text": ..., "conversation_id": ...}
  chat_with_file(text, file_uri, file_name, file_size, use_deep_think) -> dict
  generate_image(prompt, ratio, ref_image_key) -> {"images": [...]}
  generate_music(prompt, lyric, genre) -> {"tracks": [...]}
  generate_video(prompt, ratio) -> {"videos": [...]}
  upload_file(file_data, filename) -> dict
  get_file_download_url(uri, expire_seconds) -> str
  upload_image(image_bytes, filename) -> dict
  record_success() / record_failure(code) / extract_conversation_id(event)
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from .browser_client import BrowserClient
from .account_pool import AccountEntry, AccountPool
from .client import (
    CompletionChunk,
    CompletionResult,
    DoubaoChatClient,
    UploadedFile,
)
from .volcano import VolcanoClient

log = logging.getLogger(__name__)

_IMAGE_EXT_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}


async def download_image_bytes(url: str, max_mb: float = 20.0) -> "tuple[bytes, str]":
    """Fetch a reference image from an http(s)/data URL.

    Returns ``(bytes, filename)``. Raises RuntimeError on unsupported
    schemes or oversized payloads.
    """
    import base64 as _b64

    import httpx as _httpx

    url = (url or "").strip()
    if not url:
        raise RuntimeError("Empty image URL")
    if url.startswith("data:"):
        try:
            header, encoded = url.split(",", 1)
        except ValueError as exc:
            raise RuntimeError("Invalid data URI") from exc
        # Pre-decode size guard (~20MB decoded ≈ 27MB base64 chars)
        if len(encoded) > int(max_mb * 1024 * 1024 * 1.4):
            raise RuntimeError(f"Image too large (> {max_mb}MB)")
        mime = header[5:].split(";", 1)[0] if len(header) > 5 else "image/png"
        ext = _IMAGE_EXT_BY_MIME.get(mime, ".png")
        try:
            data = _b64.b64decode(encoded)
        except Exception as exc:
            raise RuntimeError("Invalid base64 in data URI") from exc
        return data, f"ref{ext}"
    if not (url.startswith("http://") or url.startswith("https://")):
        raise RuntimeError(f"Unsupported image URL scheme: {url[:60]}")

    async with _httpx.AsyncClient(timeout=60, follow_redirects=True) as cli:
        resp = await cli.get(url)
        resp.raise_for_status()
        data = resp.content
    if len(data) > int(max_mb * 1024 * 1024):
        raise RuntimeError(f"Image too large (> {max_mb}MB)")

    ctype = (resp.headers.get("content-type") or "image/png").split(";")[0].strip()
    name = url.rstrip("/#?").rsplit("/", 1)[-1].split("?")[0]
    if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")):
        name = f"ref{_IMAGE_EXT_BY_MIME.get(ctype, '.png')}"
    return data, name


class BrowserBackend:
    """Adapter wrapping the single-account BrowserClient (fallback mode)."""

    def __init__(self, client: BrowserClient):
        self._client = client

    async def is_alive(self) -> bool:
        return await self._client.is_alive()

    async def restart(self):
        await self._client.restart()

    @property
    def is_ready(self) -> bool:
        return self._client.is_ready

    @property
    def needs_captcha(self) -> bool:
        return self._client.needs_captcha

    @property
    def consecutive_failures(self) -> int:
        return self._client.consecutive_failures

    @property
    def last_error_code(self) -> int:
        return self._client.last_error_code

    def record_success(self):
        self._client.record_success()

    def record_failure(self, error_code: int = 0):
        self._client.record_failure(error_code)

    @staticmethod
    def extract_conversation_id(event: Dict[str, Any]) -> Optional[str]:
        return BrowserClient.extract_conversation_id(event)

    async def chat_completion(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        use_deep_think: int = 0,
    ) -> AsyncIterator[Dict[str, Any]]:
        async for event in self._client.chat_completion(
            text, conversation_id=conversation_id, bot_id=bot_id, use_deep_think=use_deep_think
        ):
            yield event

    async def chat(
        self, text: str, conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None, use_deep_think: int = 0,
    ) -> Dict[str, Any]:
        return await self._client.chat(
            text, conversation_id=conversation_id, bot_id=bot_id, use_deep_think=use_deep_think
        )

    async def chat_with_file(
        self, text: str, file_uri: Union[str, list], file_name: str,
        file_size: int, use_deep_think: int = 0,
    ) -> Dict[str, Any]:
        return await self._client.chat_with_file(
            text, file_uri, file_name, file_size, use_deep_think
        )

    async def generate_image(
        self, prompt: str, ratio: Optional[str] = None, ref_image_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._client.generate_image(prompt, ratio=ratio, ref_image_key=ref_image_key)

    async def generate_music(
        self, prompt: str, lyric: Optional[str] = None, genre: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._client.generate_music(prompt, lyric=lyric, genre=genre)

    async def generate_video(
        self, prompt: str, ratio: Optional[str] = None,
        ref_image_url: Optional[str] = None,
        first_frame_url: Optional[str] = None,
        last_frame_url: Optional[str] = None,
        camera_movement: Optional[str] = None,
    ) -> Dict[str, Any]:
        if any((ref_image_url, first_frame_url, last_frame_url, camera_movement)):
            raise RuntimeError(
                "Browser fallback channel does not support image-to-video / "
                "camera control; configure a free-account pool instead"
            )
        return await self._client.generate_video(prompt, ratio=ratio)

    async def upload_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        return await self._client.upload_file(file_data, filename)

    async def get_file_download_url(self, uri: str, expire_seconds: int = 3600) -> str:
        return await self._client.get_file_download_url(uri=uri, expire_seconds=expire_seconds)

    async def upload_image(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        return await self._client.upload_image(image_bytes=image_bytes, filename=filename)


def _chunk_to_events(chunk: CompletionChunk, thinking_toggle: bool):
    """Convert a CompletionChunk into simplified raw-dict events.

    Returns (events, new_thinking_toggle).
    """
    events: List[Dict[str, Any]] = []
    base = {"conversation_id": chunk.conversation_id}

    if chunk.error_code:
        events.append({"error_code": chunk.error_code, "error_msg": chunk.error_msg, **base})
        return events, thinking_toggle

    if chunk.thinking:
        if not thinking_toggle:
            events.append({
                "_event": "CONTENT_BLOCK",
                "content": {"content_block": [{"block_type": 10040}]},
                **base,
            })
            thinking_toggle = True
        events.append({"_event": "CHUNK_DELTA", "text": chunk.thinking, **base})
    elif chunk.text:
        if thinking_toggle:
            events.append({
                "_event": "CONTENT_BLOCK",
                "content": {"content_block": [{"block_type": 10040}]},
                **base,
            })
            thinking_toggle = False
        events.append({"_event": "CHUNK_DELTA", "text": chunk.text, **base})
    elif chunk.block_type == 10040:
        # Pure thinking toggle marker (no text attached)
        events.append({
            "_event": "CONTENT_BLOCK",
            "content": {"content_block": [{"block_type": 10040}]},
            **base,
        })
        thinking_toggle = not thinking_toggle

    return events, thinking_toggle


class FreeAccountBackend:
    """Multi-account backend using a DoubaoChatClient pool."""

    def __init__(self, pool: AccountPool):
        self._pool = pool
        self._next_account: Optional[AccountEntry] = None

    @property
    def is_ready(self) -> bool:
        return self._pool.healthy_count > 0

    @property
    def needs_captcha(self) -> bool:
        return any(a.needs_captcha for a in self._pool.accounts)

    @property
    def consecutive_failures(self) -> int:
        return 0

    @property
    def last_error_code(self) -> int:
        return 0

    def record_success(self):
        if self._next_account:
            self._next_account.mark_success()

    def record_failure(self, error_code: int = 0):
        if self._next_account:
            was_healthy = self._next_account.is_healthy()
            self._next_account.mark_failure(error_code)
            if was_healthy and not self._next_account.is_healthy():
                log.warning(
                    "Account %s unavailable after error %d, will select next",
                    self._next_account.name, error_code,
                )

    @staticmethod
    def extract_conversation_id(event: Dict[str, Any]) -> Optional[str]:
        cid = event.get("conversation_id")
        return cid or None

    def _select_account(self, *, for_video: bool = False) -> AccountEntry:
        entry = self._pool.select(for_video=for_video)
        self._next_account = entry
        return entry

    def _get_client(self) -> DoubaoChatClient:
        entry = self._select_account()
        if entry.client is None:
            raise RuntimeError(f"Account {entry.name} has no client initialized")
        return entry.client

    async def chat_completion(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        use_deep_think: int = 0,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield raw-dict events compatible with existing server helpers."""
        client = self._get_client()
        thinking_toggle = False
        async for chunk in client.chat_stream_completion(
            text=text, need_deep_think=use_deep_think, bot_id=bot_id,
        ):
            if chunk.is_done:
                break
            events, thinking_toggle = _chunk_to_events(chunk, thinking_toggle)
            for ev in events:
                yield ev
            if any(ev.get("error_code") for ev in events):
                return

    async def chat(
        self, text: str, conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None, use_deep_think: int = 0,
    ) -> Dict[str, Any]:
        entry = self._select_account()
        client = entry.client
        if client is None:
            raise RuntimeError(f"Account {entry.name} not initialized")
        result = await client.chat_completion(
            text=text, need_deep_think=use_deep_think, bot_id=bot_id,
        )
        entry.mark_success()
        return {"text": result.text, "conversation_id": result.conversation_id}

    async def chat_with_file(
        self, text: str, file_uri: Union[str, list], file_name: str,
        file_size: int, use_deep_think: int = 0,
    ) -> Dict[str, Any]:
        entry = self._select_account()
        client = entry.client
        if client is None:
            raise RuntimeError(f"Account {entry.name} not initialized")

        if isinstance(file_uri, list):
            file_refs = file_uri
        else:
            file_refs = [{"uri": file_uri, "name": file_name, "size": file_size}]

        attachments: List[UploadedFile] = []
        for ref in file_refs:
            uri = ref.get("uri", "")
            if not uri.startswith("tos-"):
                raise RuntimeError("Only TOS URIs supported for file chat")
            attachments.append(UploadedFile(
                uri=uri,
                name=ref.get("name", "file"),
                size=int(ref.get("size") or 0),
            ))

        result = await client.chat_completion(
            text=text, need_deep_think=use_deep_think, file_attachments=attachments,
        )
        entry.mark_success()
        return {"text": result.text, "conversation_id": result.conversation_id}

    async def generate_image(
        self, prompt: str, ratio: Optional[str] = None, ref_image_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry = self._select_account()
        client = entry.client
        if client is None:
            raise RuntimeError(f"Account {entry.name} not initialized")
        try:
            result = await client.generate_image(prompt, ratio=ratio, ref_image_key=ref_image_key)
        except Exception:
            entry.mark_failure()
            raise
        entry.mark_success()
        return {
            "images": [
                {
                    "key": img.key,
                    "url": img.ori_url or img.raw_url or img.thumb_url,
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                }
                for img in result.images
            ],
            "prompt": prompt,
        }

    async def generate_music(
        self, prompt: str, lyric: Optional[str] = None, genre: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry = self._select_account()
        client = entry.client
        if client is None:
            raise RuntimeError(f"Account {entry.name} not initialized")
        try:
            result = await client.generate_music(prompt, lyric=lyric, genre=genre)
        except Exception:
            entry.mark_failure()
            raise
        entry.mark_success()
        return {
            "tracks": [
                {
                    "audio_url": t.audio_url,
                    "title": t.title,
                    "duration": t.duration,
                    "lyrics": t.lyrics,
                    "cover_url": t.cover_url,
                    "vid": t.vid,
                }
                for t in result.tracks
            ],
            "prompt": prompt,
        }

    async def generate_video(
        self, prompt: str, ratio: Optional[str] = None,
        ref_image_url: Optional[str] = None,
        first_frame_url: Optional[str] = None,
        last_frame_url: Optional[str] = None,
        camera_movement: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry = self._select_account(for_video=True)
        client = entry.client
        if client is None:
            raise RuntimeError(f"Account {entry.name} not initialized")
        if last_frame_url:
            raise RuntimeError(
                "Doubao channel does not support last-frame control "
                "(use xyq-video / volc-video for first-&-last-frame)"
            )
        try:
            ref_key = None
            ref_source = ref_image_url or first_frame_url
            if ref_source:
                data, fname = await download_image_bytes(ref_source)
                att = await client.upload_image(image_bytes=data, filename=fname)
                ref_key = att.get("uri") or att.get("key") or ""
            result = await client.generate_video(
                prompt, ratio=ratio, camera_movement=camera_movement or None,
                ref_image_key=ref_key or None,
            )
        except Exception:
            entry.mark_failure()
            raise
        entry.mark_success()
        # Official quota accounting: 10s video = 2 points, 5s = 1 point
        total_points = sum(
            AccountEntry.quota_points_for_duration(v.duration) for v in result.videos
        )
        entry.increment_quota(total_points)
        return {
            "videos": [
                {
                    "video_url": v.video_url,
                    "cover_url": v.cover_url,
                    "width": v.width,
                    "height": v.height,
                    "duration": v.duration,
                }
                for v in result.videos
            ],
            "prompt": prompt,
        }

    async def upload_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        entry = self._select_account()
        client = entry.client
        if client is None:
            raise RuntimeError(f"Account {entry.name} not initialized")
        try:
            result = await client.upload_file(file_data, filename)
        except Exception:
            entry.mark_failure()
            raise
        entry.mark_success()
        return {
            "uri": result.uri,
            "name": result.name,
            "size": result.size,
            "file_type": result.file_type,
        }

    async def get_file_download_url(self, uri: str, expire_seconds: int = 3600) -> str:
        entry = self._select_account()
        client = entry.client
        if client is None:
            raise RuntimeError(f"Account {entry.name} not initialized")
        return await client.get_file_download_url(uri, expire_seconds)

    async def upload_image(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        entry = self._select_account()
        client = entry.client
        if client is None:
            raise RuntimeError(f"Account {entry.name} not initialized")
        try:
            result = await client.upload_image(image_bytes=image_bytes, filename=filename)
        except Exception:
            entry.mark_failure()
            raise
        entry.mark_success()
        return result


class VolcanoBackend:
    """Adapter wrapping VolcanoClient into the dict-based backend interface."""

    def __init__(self, client: VolcanoClient):
        self._client = client

    @property
    def is_ready(self) -> bool:
        return self._client.is_ready

    @property
    def needs_captcha(self) -> bool:
        return False

    @property
    def consecutive_failures(self) -> int:
        return 0

    @property
    def last_error_code(self) -> int:
        return 0

    def record_success(self):
        pass

    def record_failure(self, error_code: int = 0):
        pass

    @staticmethod
    def extract_conversation_id(event: Dict[str, Any]) -> Optional[str]:
        cid = event.get("conversation_id")
        return cid or None

    async def chat_completion(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        use_deep_think: int = 0,
    ) -> AsyncIterator[Dict[str, Any]]:
        thinking_toggle = False
        async for chunk in self._client.chat_stream_completion(
            text=text, need_deep_think=use_deep_think, bot_id=bot_id,
        ):
            if chunk.is_done:
                break
            events, thinking_toggle = _chunk_to_events(chunk, thinking_toggle)
            for ev in events:
                yield ev
            if any(ev.get("error_code") for ev in events):
                return

    async def chat(
        self, text: str, conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None, use_deep_think: int = 0,
    ) -> Dict[str, Any]:
        result = await self._client.chat_completion(
            text=text, need_deep_think=use_deep_think, bot_id=bot_id,
        )
        return {"text": result.text, "conversation_id": result.conversation_id}

    async def chat_with_file(
        self, text: str, file_uri: Union[str, list], file_name: str,
        file_size: int, use_deep_think: int = 0,
    ) -> Dict[str, Any]:
        # Official API doesn't support file chat through this path; fall back to plain chat
        return await self.chat(text, use_deep_think=use_deep_think)

    async def generate_image(
        self, prompt: str, ratio: Optional[str] = None, ref_image_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = await self._client.generate_image(prompt, ratio=ratio, ref_image_key=ref_image_key)
        return {
            "images": [
                {
                    "key": img.key,
                    "url": img.ori_url or img.raw_url or img.thumb_url,
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                }
                for img in result.images
            ],
            "prompt": prompt,
        }

    async def generate_music(
        self, prompt: str, lyric: Optional[str] = None, genre: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise RuntimeError("Music generation not supported via Volcano API")

    async def generate_video(
        self, prompt: str, ratio: Optional[str] = None,
        ref_image_url: Optional[str] = None,
        first_frame_url: Optional[str] = None,
        last_frame_url: Optional[str] = None,
        camera_movement: Optional[str] = None,  # not supported by Ark video API; ignored
    ) -> Dict[str, Any]:
        result = await self._client.generate_video(
            prompt, ratio=ratio, ref_image_url=ref_image_url,
            first_frame_url=first_frame_url, last_frame_url=last_frame_url,
        )
        return {
            "videos": [
                {
                    "video_url": v.video_url,
                    "cover_url": v.cover_url,
                    "width": v.width,
                    "height": v.height,
                    "duration": v.duration,
                }
                for v in result.videos
            ],
            "prompt": prompt,
        }

    async def upload_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        result = await self._client.upload_file(file_data, filename)
        return {
            "uri": result.uri,
            "name": result.name,
            "size": result.size,
            "file_type": result.file_type,
        }

    async def get_file_download_url(self, uri: str, expire_seconds: int = 3600) -> str:
        return await self._client.get_file_download_url(uri, expire_seconds)

    async def upload_image(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        return await self._client.upload_image(image_bytes=image_bytes, filename=filename)


class XiaoyunqueBackend:
    """小云雀 (剪映 Seedance 2.0) channel backed by the vendored Playwright engine.

    Only video generation is supported. The engine handles multi-cookie
    rotation, credit checks and downloads internally; generated files land in
    ``output_dir`` and are served by the server via ``/xyq-files/{name}``.
    """

    # quality label -> engine model key
    QUALITY_MAP = {"fast": "fast", "2.0": "2.0"}

    def __init__(self, cookies_dir: str, output_dir: str,
                 headless: bool = True, engine=None):
        if engine is None:
            from . import xiaoyunque_engine as engine
        self._engine = engine
        self._engine.config.cookies_dir = cookies_dir
        self._engine.config.output_dir = output_dir
        self._engine.config.headless = headless
        self._output_dir = output_dir

    @property
    def is_ready(self) -> bool:
        return len(self.cookie_names()) > 0

    @property
    def needs_captcha(self) -> bool:
        return False

    @property
    def consecutive_failures(self) -> int:
        return 0

    @property
    def last_error_code(self) -> int:
        return 0

    def record_success(self):
        pass

    def record_failure(self, error_code: int = 0):
        pass

    @staticmethod
    def extract_conversation_id(event: Dict[str, Any]) -> Optional[str]:
        return None

    def cookie_names(self) -> List[str]:
        try:
            return list(self._engine.get_token_files())
        except Exception:
            return []

    @staticmethod
    def _coerce_ratio(ratio: Optional[str]) -> str:
        # xiaoyquee web UI supports 16:9 / 9:16 only; engine maps 1:1 already
        if ratio in ("16:9", "9:16"):
            return ratio
        return "16:9"

    async def generate_video(
        self,
        prompt: str,
        ratio: Optional[str] = None,
        duration: int = 10,
        quality: str = "fast",
        ref_image_url: Optional[str] = None,
        first_frame_url: Optional[str] = None,
        last_frame_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a video via the xiaoyquee engine.

        Returns the standard dict shape: {"videos": [{"video_url": ...}], ...}
        where ``video_url`` points at the local file-serving endpoint.

        Image-to-video: reference images are downloaded to temp files and
        passed to the engine as ordered ``ref_images`` — 1 image = single
        reference/first frame, 2 images = first & last frame (the JianYing
        web API interprets them by order; there is no explicit role field).
        Temp files are removed afterwards.
        """
        import os as _os
        import uuid as _uuid

        duration = int(duration) if duration in (5, 10, 15, "5", "10", "15") else 10
        engine_model = self.QUALITY_MAP.get(quality, "fast")

        effective_first = first_frame_url or ref_image_url
        frame_urls = [u for u in (effective_first, last_frame_url) if u][:2]

        tmp_paths = []
        try:
            for u in frame_urls:
                data, fname = await download_image_bytes(u)
                _os.makedirs(self._output_dir, exist_ok=True)
                root, ext = _os.path.splitext(fname)
                tmp_path = _os.path.join(
                    self._output_dir, f"_ref_{_uuid.uuid4().hex[:8]}{ext or '.png'}"
                )
                with open(tmp_path, "wb") as f:
                    f.write(data)
                tmp_paths.append(tmp_path)

            out_path = await self._engine.run(
                prompt=prompt,
                duration=duration,
                ratio=self._coerce_ratio(ratio),
                model=engine_model,
                ref_images=tmp_paths or None,
                output_dir=self._output_dir,
                cookie_file=None,
            )
        finally:
            for p in tmp_paths:
                try:
                    _os.remove(p)
                except OSError:
                    pass

        if not out_path:
            raise RuntimeError("xiaoyunque generation returned no result")
        name = _os.path.basename(out_path)
        return {
            "videos": [
                {
                    "video_url": f"/xyq-files/{name}",
                    "cover_url": "",
                    "width": 0,
                    "height": 0,
                    "duration": float(duration),
                }
            ],
            "prompt": prompt,
        }

    # ── Unsupported capabilities (uniform dict interface) ──

    async def chat_completion(self, *args, **kwargs):
        raise RuntimeError("Xiaoyunque channel only supports video generation")

    async def chat(self, *args, **kwargs):
        raise RuntimeError("Xiaoyunque channel only supports video generation")

    async def chat_with_file(self, *args, **kwargs):
        raise RuntimeError("Xiaoyunque channel only supports video generation")

    async def generate_image(self, *args, **kwargs):
        raise RuntimeError("Xiaoyunque channel only supports video generation")

    async def generate_music(self, *args, **kwargs):
        raise RuntimeError("Xiaoyunque channel only supports video generation")

    async def upload_file(self, *args, **kwargs):
        raise RuntimeError("Xiaoyunque channel only supports video generation")

    async def get_file_download_url(self, uri: str, expire_seconds: int = 3600) -> str:
        raise RuntimeError("Xiaoyunque channel only supports video generation")

    async def upload_image(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        raise RuntimeError("Xiaoyunque channel only supports video generation")

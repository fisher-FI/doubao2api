"""Official Volcano Engine (Ark) API client.

Provides OpenAI-compatible chat, image, and video generation
using the official Doubao API via Volcano Engine Ark.

Video generation uses the async task API:
  POST /api/v3/contents/generations/tasks          (create)
  GET  /api/v3/contents/generations/tasks/{task_id} (poll)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .client import (
    CompletionChunk,
    CompletionResult,
    GeneratedImage,
    GeneratedVideo,
    ImageGenerationResult,
    MusicGenerationResult,
    UploadedFile,
    VideoGenerationResult,
)

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_CHAT_MODEL = "doubao-pro-32k-250715"
DEFAULT_IMAGE_MODEL = "doubao-seedream-5-0-pro-260128"
DEFAULT_VIDEO_MODEL = "doubao-seedance-1-5-pro-251215"


class VolcanoClient:
    """Client for Volcano Engine official Doubao API."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        chat_model: str = "",
        image_model: str = "",
        video_model: str = "",
    ):
        self._api_key = api_key or os.environ.get("VOLC_API_KEY", "")
        self._base_url = (base_url or os.environ.get("VOLC_BASE_URL", "") or DEFAULT_BASE_URL).rstrip("/")
        self._chat_model = chat_model or os.environ.get("VOLC_CHAT_MODEL", "") or DEFAULT_CHAT_MODEL
        self._image_model = image_model or os.environ.get("VOLC_IMAGE_MODEL", "") or DEFAULT_IMAGE_MODEL
        self._video_model = video_model or os.environ.get("VOLC_VIDEO_MODEL", "") or DEFAULT_VIDEO_MODEL
        self._http = httpx.AsyncClient(timeout=300)
        self._ready = bool(self._api_key)

    @property
    def is_ready(self) -> bool:
        return self._ready

    async def close(self):
        await self._http.aclose()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def chat_completion(
        self,
        text: str,
        need_deep_think: int = 0,
        bot_id: Optional[str] = None,
        image_attachments: Optional[List[Dict[str, str]]] = None,
        file_attachments: Optional[List[UploadedFile]] = None,
        conversation_id: Optional[str] = None,
    ) -> CompletionResult:
        """Non-streaming chat completion."""
        messages = [{"role": "user", "content": text}]
        if file_attachments:
            file_names = ", ".join(f.name for f in file_attachments)
            messages[0]["content"] = f"[Files: {file_names}]\n{text}"

        payload = {
            "model": self._chat_model,
            "messages": messages,
            "stream": False,
        }
        resp = await self._http.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        result = CompletionResult(
            text=msg.get("content", ""),
            thinking_text=msg.get("reasoning_content", ""),
        )
        if data.get("id"):
            result.conversation_id = data["id"]
        return result

    async def chat_stream_completion(
        self,
        text: str,
        need_deep_think: int = 0,
        bot_id: Optional[str] = None,
        image_attachments: Optional[List[Dict[str, str]]] = None,
        file_attachments: Optional[List[UploadedFile]] = None,
    ) -> AsyncIterator[CompletionChunk]:
        """Streaming chat completion via SSE."""
        messages = [{"role": "user", "content": text}]
        if file_attachments:
            file_names = ", ".join(f.name for f in file_attachments)
            messages[0]["content"] = f"[Files: {file_names}]\n{text}"

        payload = {
            "model": self._chat_model,
            "messages": messages,
            "stream": True,
        }
        async with self._http.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk_str = line[6:].strip()
                if chunk_str == "[DONE]":
                    yield CompletionChunk(is_done=True)
                    return
                try:
                    chunk = json.loads(chunk_str)
                except json.JSONDecodeError:
                    continue
                chunk_id = chunk.get("id", "")
                choices = chunk.get("choices") or [{}]
                delta = choices[0].get("delta", {})
                if delta.get("role") == "assistant":
                    continue
                content = delta.get("content", "")
                reasoning = delta.get("reasoning_content", "")
                if reasoning:
                    yield CompletionChunk(thinking=reasoning, conversation_id=chunk_id)
                if content:
                    yield CompletionChunk(text=content, conversation_id=chunk_id)

    async def generate_image(
        self,
        prompt: str,
        ratio: Optional[str] = None,
        ref_image_key: Optional[str] = None,
    ) -> ImageGenerationResult:
        """Generate image via official images/generations endpoint."""
        size = "1024x1024"
        if ratio:
            ratio_map = {"1:1": "1024x1024", "16:9": "1792x1024", "9:16": "1024x1792"}
            size = ratio_map.get(ratio, "1024x1024")
        payload = {
            "model": self._image_model,
            "prompt": prompt,
            "n": 1,
            "size": size,
        }
        resp = await self._http.post(
            f"{self._base_url}/images/generations",
            headers=self._headers(),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        images = []
        for item in data.get("data", []):
            images.append(GeneratedImage(
                ori_url=item.get("url", ""),
                raw_url=item.get("url", ""),
                width=item.get("width", 0),
                height=item.get("height", 0),
            ))
        return ImageGenerationResult(images=images, prompt=prompt)

    async def generate_video(
        self,
        prompt: str,
        ratio: Optional[str] = None,
        camera_movement: Optional[str] = None,
        ref_image_key: Optional[str] = None,
        timeout: float = 300,
    ) -> VideoGenerationResult:
        """Generate video via official Ark async task API.

        Reference: POST /api/v3/contents/generations/tasks (create),
        GET /api/v3/contents/generations/tasks/{task_id} (poll).
        """
        content = [{"type": "text", "text": prompt}]
        payload: Dict[str, Any] = {
            "model": self._video_model,
            "content": content,
        }
        if ratio:
            payload["ratio"] = ratio

        # Create task
        resp = await self._http.post(
            f"{self._base_url}/contents/generations/tasks",
            headers=self._headers(),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        task_id = data.get("id", "")
        if not task_id:
            raise RuntimeError(f"No video task ID returned from official API: {data}")

        # Poll for completion
        start = time.time()
        while time.time() - start < timeout:
            await asyncio.sleep(10)
            poll_resp = await self._http.get(
                f"{self._base_url}/contents/generations/tasks/{task_id}",
                headers=self._headers(),
            )
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            status = str(poll_data.get("status", "")).upper()
            if status in ("SUCCEEDED", "SUCCESS"):
                videos = []
                for item in poll_data.get("data", []):
                    videos.append(GeneratedVideo(
                        video_url=item.get("url", ""),
                        cover_url=item.get("cover_url", ""),
                        width=item.get("width", 0),
                        height=item.get("height", 0),
                        duration=item.get("duration", 0.0),
                    ))
                return VideoGenerationResult(videos=videos, prompt=prompt)
            elif status in ("FAILED", "EXPIRED", "CANCELLED", "CANCELED"):
                raise RuntimeError(
                    f"Video generation {status}: {poll_data.get('error', 'unknown')}"
                )
        raise TimeoutError("Video generation timeout")

    async def generate_music(
        self,
        prompt: str,
        lyric: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> MusicGenerationResult:
        raise NotImplementedError("Music generation not supported via Volcano API")

    async def upload_file(self, file_data: bytes, filename: str) -> UploadedFile:
        files = {"file": (filename, file_data, "application/octet-stream")}
        resp = await self._http.post(
            f"{self._base_url}/files",
            headers={"Authorization": f"Bearer {self._api_key}"},
            files=files,
        )
        resp.raise_for_status()
        data = resp.json()
        return UploadedFile(
            uri=data.get("id", ""),
            name=data.get("filename", filename),
            size=data.get("bytes", len(file_data)),
        )

    async def get_file_download_url(self, uri: str, expire_seconds: int = 3600) -> str:
        return f"{self._base_url}/files/{uri}/content"

    async def upload_image(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        result = await self.upload_file(image_bytes, filename)
        return {
            "uri": result.uri,
            "cdn_url": "",
            "url": "",
            "name": result.name,
            "format": filename.rsplit(".", 1)[-1] if "." in filename else "",
            "width": 0,
            "height": 0,
        }

# 豆包反代实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 基于 doubao2api 扩展多账号免费池 + 官方火山引擎通道 + Web 管理面板

**架构:** 新增 `Backend` 接口（方法签名与现有 `BrowserClient` 的 dict 返回兼容），提供三个实现：`BrowserBackend`（适配现有单账号）、`FreeAccountBackend`（多账号池）、`VolcanoBackend`（官方 API）。通道抽象层位于 `unified_server.py` 的 `_get_backend()` 中，按模型名选择后端，端点代码几乎不变。

**Tech Stack:** Python 3.10+, FastAPI, aiohttp, httpx, Playwright (保留), DoubaoChatClient (纯 HTTP)

---

## 文件结构

### 新建文件
| 路径 | 职责 |
|---|---|
| `doubao2api/account_pool.py` | `AccountEntry`、`AccountPool`（多 session 加载、轮询、故障转移、额度跟踪） |
| `doubao2api/backends.py` | `Backend` 协议基类 + `BrowserBackend`、`FreeAccountBackend`、`VolcanoBackend` |
| `doubao2api/volcano.py` | `VolcanoClient`（官方火山引擎 API 封装，返回 domain dataclasses） |
| `tests/test_account_pool.py` | AccountPool 单元测试 |
| `tests/test_backends.py` | Backend 单元测试 |

### 修改文件
| 文件 | 改动 |
|---|---|
| `doubao2api/unified_server.py` | 集成 `_get_backend()`、添加 Admin API（账号、通道）、更新 `/health`、动态 `/v1/models` |
| `doubao2api/static/admin.html` | 增加“账号管理”、“通道配置”两个 Tab |
| `doubao2api/__init__.py` | 导出新类型 |
| `.env.example` | 文档新增环境变量 |
| `README.md` | 记录新功能 |

---

## 任务分解

### Task 1: 账号池 (`account_pool.py`)

**Files:**
- Create: `doubao2api/account_pool.py`
- Test: `tests/test_account_pool.py`

- [ ] **Step 1: 定义 `AccountEntry` 数据类**

```python
# doubao2api/account_pool.py
from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import DoubaoChatClient

log = logging.getLogger(__name__)


def _parse_cookie_header(header: str) -> Dict[str, str]:
    """Parse a cookie header string into a name/value dict."""
    cookies: Dict[str, str] = {}
    for item in header.split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        name, value = item.split("=", 1)
        cookies[name.strip()] = value.strip()
    return cookies


@dataclass
class AccountEntry:
    name: str
    session_file: str
    enabled: bool = True
    weight: int = 1
    backend: str = "http"  # "http" | "browser"
    client: Optional[DoubaoChatClient] = None
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_used_at: float = 0.0
    last_error_code: int = 0
    needs_captcha: bool = False
    daily_quota_used: int = 0
    quota_date: str = ""

    def mark_success(self):
        self.success_count += 1
        self.consecutive_failures = 0
        self.last_error_code = 0
        self.needs_captcha = False
        self.last_used_at = time.time()

    def mark_failure(self, error_code: int = 0):
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_error_code = error_code
        if error_code == 710022004:
            self.needs_captcha = True
        if self.consecutive_failures >= 5:
            self.enabled = False
            log.warning("Account %s disabled after %d consecutive failures", self.name, self.consecutive_failures)
        self.last_used_at = time.time()

    def is_healthy(self) -> bool:
        return self.enabled and not self.needs_captcha and self.consecutive_failures < 5

    def increment_quota(self):
        today = date.today().isoformat()
        if self.quota_date != today:
            self.daily_quota_used = 0
            self.quota_date = today
        self.daily_quota_used += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "session_file": self.session_file,
            "enabled": self.enabled,
            "weight": self.weight,
            "backend": self.backend,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "last_used_at": self.last_used_at,
            "last_error_code": self.last_error_code,
            "needs_captcha": self.needs_captcha,
            "is_healthy": self.is_healthy(),
            "daily_quota_used": self.daily_quota_used,
            "quota_date": self.quota_date,
        }
```

- [ ] **Step 2: 定义 `AccountPool` 类**

```python
# doubao2api/account_pool.py (continued)

class AccountPool:
    """Manages multiple DoubaoChatClient accounts with round-robin selection."""

    def __init__(
        self,
        accounts_dir: str = "./accounts",
        captcha_handler: str = "auto",
        max_captcha_retries: int = 3,
    ):
        self._accounts_dir = Path(accounts_dir)
        self._accounts: List[AccountEntry] = []
        self._next_index: int = 0
        self._captcha_handler = captcha_handler
        self._max_captcha_retries = max_captcha_retries

    @property
    def accounts(self) -> List[AccountEntry]:
        return copy.deepcopy(self._accounts)

    @property
    def healthy_count(self) -> int:
        return sum(1 for a in self._accounts if a.is_healthy())

    @property
    def total_count(self) -> int:
        return len(self._accounts)

    def load_accounts(self):
        """Load account entries from config and directory (reload)."""
        self._accounts = []
        # Load accounts.json if exists (in root or accounts dir)
        config_paths = [Path("accounts.json"), self._accounts_dir / "accounts.json"]
        loaded_config = False
        for cfg_path in config_paths:
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                for acct in config.get("accounts", []):
                    session_file = acct.get("session_file", "")
                    if not os.path.isabs(session_file):
                        session_file = str(self._accounts_dir / session_file)
                    self._accounts.append(AccountEntry(
                        name=acct.get("name", os.path.basename(session_file)),
                        session_file=session_file,
                        enabled=acct.get("enabled", True),
                        weight=acct.get("weight", 1),
                        backend=acct.get("backend", "http"),
                    ))
                loaded_config = True
                break

        # Scan accounts_dir for *.json files not already referenced
        if self._accounts_dir.exists():
            existing_sessions = {a.session_file for a in self._accounts}
            for f in sorted(self._accounts_dir.iterdir()):
                if f.name == "accounts.json":
                    continue
                if f.suffix == ".json" and str(f) not in existing_sessions:
                    self._accounts.append(AccountEntry(
                        name=f.stem,
                        session_file=str(f),
                    ))

        if not self._accounts:
            # Fallback: check root .doubao_session.json
            legacy = Path(".doubao_session.json")
            if legacy.exists():
                self._accounts.append(AccountEntry(
                    name="default",
                    session_file=str(legacy),
                ))
            # Check DOUBAO_COOKIE env var
            cookie_header = os.environ.get("DOUBAO_COOKIE", "").strip()
            if cookie_header and not self._accounts:
                self._accounts_dir.mkdir(parents=True, exist_ok=True)
                session_file = str(self._accounts_dir / "env_cookie.json")
                Path(session_file).write_text(
                    json.dumps({"cookies": _parse_cookie_header(cookie_header), "params": {}}, ensure_ascii=False),
                    encoding="utf-8",
                )
                self._accounts.append(AccountEntry(
                    name="env",
                    session_file=session_file,
                ))

    async def start(self):
        """Initialize clients for all accounts (idempotent, supports reload)."""
        # Close any existing clients first (for reload)
        for entry in self._accounts:
            if entry.client:
                try:
                    await entry.client.__aexit__(None, None, None)
                except Exception:
                    pass
                entry.client = None
        for entry in self._accounts:
            try:
                if not os.path.exists(entry.session_file):
                    log.warning("Session file not found: %s", entry.session_file)
                    entry.enabled = False
                    continue
                client = DoubaoChatClient.from_session(
                    session_file=entry.session_file,
                    captcha_handler=self._captcha_handler,
                    max_captcha_retries=self._max_captcha_retries,
                )
                entry.client = client
                log.info("Account %s loaded from %s", entry.name, entry.session_file)
            except Exception as e:
                log.error("Failed to load account %s: %s", entry.name, e)
                entry.enabled = False

    async def stop(self):
        """Close all account clients."""
        for entry in self._accounts:
            if entry.client:
                try:
                    await entry.client.__aexit__(None, None, None)
                except Exception:
                    pass
        self._accounts.clear()

    def select(self, *, for_video: bool = False) -> AccountEntry:
        """Select next healthy account (round-robin)."""
        healthy = [a for a in self._accounts if a.is_healthy()]
        if not healthy:
            raise RuntimeError("No healthy accounts available")
        # Round-robin
        for _ in range(len(healthy)):
            entry = healthy[self._next_index % len(healthy)]
            self._next_index = (self._next_index + 1) % len(healthy)
            if entry.is_healthy():
                return entry
        raise RuntimeError("No healthy accounts available (all excluded)")

    def get_by_name(self, name: str) -> Optional[AccountEntry]:
        for a in self._accounts:
            if a.name == name:
                return a
        return None
```

- [ ] **Step 3: 写单元测试**

```python
# tests/test_account_pool.py
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from doubao2api.account_pool import AccountEntry, AccountPool

class TestAccountEntry:
    def test_mark_success_resets_counters(self):
        entry = AccountEntry(name="test", session_file="test.json")
        entry.failure_count = 5
        entry.consecutive_failures = 3
        entry.last_error_code = 710022004
        entry.needs_captcha = True
        entry.mark_success()
        assert entry.consecutive_failures == 0
        assert entry.last_error_code == 0
        assert entry.needs_captcha is False
        assert entry.success_count == 1

    def test_mark_failure_disables_after_5(self):
        entry = AccountEntry(name="test", session_file="test.json")
        for i in range(5):
            entry.mark_failure()
        assert entry.enabled is False
        assert entry.consecutive_failures == 5

    def test_mark_failure_710022004_sets_captcha(self):
        entry = AccountEntry(name="test", session_file="test.json")
        entry.mark_failure(710022004)
        assert entry.needs_captcha is True

    def test_increment_quota_resets_daily(self):
        entry = AccountEntry(name="test", session_file="test.json")
        entry.increment_quota()
        assert entry.daily_quota_used == 1
        entry.increment_quota()
        assert entry.daily_quota_used == 2

    def test_is_healthy(self):
        entry = AccountEntry(name="test", session_file="test.json", enabled=True)
        assert entry.is_healthy() is True
        entry.needs_captcha = True
        assert entry.is_healthy() is False

class TestAccountPool:
    def test_load_accounts_from_config(self, tmp_path):
        pool = AccountPool(accounts_dir=str(tmp_path))
        cfg = {
            "accounts": [
                {
                    "name": "主号",
                    "session_file": "main.json",
                    "enabled": True,
                    "weight": 2,
                    "backend": "http",
                }
            ]
        }
        cfg_path = tmp_path / "accounts.json"
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False))
        # Also create the session file
        session = {"cookies": {"sessionid": "test-session"}, "params": {}}
        (tmp_path / "main.json").write_text(json.dumps(session))
        pool.load_accounts()
        assert len(pool.accounts) == 1
        assert pool.accounts[0].name == "主号"
        assert pool.accounts[0].weight == 2

    def test_load_accounts_from_directory(self, tmp_path):
        pool = AccountPool(accounts_dir=str(tmp_path))
        # Create a session file
        session = {"cookies": {"sessionid": "test"}, "params": {}}
        (tmp_path / "my_account.json").write_text(json.dumps(session))
        pool.load_accounts()
        assert len(pool.accounts) == 1
        assert pool.accounts[0].name == "my_account"

    def test_select_round_robin(self, tmp_path):
        pool = AccountPool(accounts_dir=str(tmp_path))
        for i in range(3):
            pool._accounts.append(AccountEntry(
                name=f"acct{i}", session_file=f"acct{i}.json", enabled=True
            ))
        selected = [pool.select().name for _ in range(5)]
        assert selected == ["acct0", "acct1", "acct2", "acct0", "acct1"]

    def test_select_skips_unhealthy(self, tmp_path):
        pool = AccountPool(accounts_dir=str(tmp_path))
        pool._accounts = [
            AccountEntry(name="good", session_file="good.json", enabled=True),
            AccountEntry(name="bad", session_file="bad.json", enabled=True, needs_captcha=True),
            AccountEntry(name="also_good", session_file="also.json", enabled=True),
        ]
        selected = [pool.select().name for _ in range(4)]
        assert "bad" not in selected

    def test_select_raises_when_no_healthy(self, tmp_path):
        pool = AccountPool(accounts_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="No healthy accounts"):
            pool.select()
```

- [ ] **Step 4: 运行测试并确认通过**

Run: `python -m pytest tests/test_account_pool.py -v`
Expected: all tests PASS

- [ ] **Step 5: 提交**

```bash
git add doubao2api/account_pool.py tests/test_account_pool.py
git commit -m "feat: add AccountPool and AccountEntry"
```

---

### Task 2: 火山引擎通道 (`volcano.py`)

**Files:**
- Create: `doubao2api/volcano.py`
- Test: `tests/test_volcano.py`

- [ ] **Step 1: 实现 `VolcanoClient`**

```python
# doubao2api/volcano.py
"""Official Volcano Engine (Ark) API client.

Provides OpenAI-compatible chat, image, and video generation
using the official Doubao API via Volcano Engine.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, AsyncIterator

import httpx

from .client import (
    CompletionChunk,
    CompletionResult,
    GeneratedImage,
    ImageGenerationResult,
    GeneratedVideo,
    VideoGenerationResult,
    GeneratedMusic,
    MusicGenerationResult,
    UploadedFile,
)

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class VolcanoClient:
    """Client for Volcano Engine official Doubao API."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = DEFAULT_BASE_URL,
        chat_model: str = "doubao-pro-32k-250715",
        image_model: str = "doubao-seedream-5-0-pro-260128",
        video_model: str = "doubao-seedance-1-5-pro-251215",
    ):
        self._api_key = api_key or os.environ.get("VOLC_API_KEY", "")
        self._base_url = base_url or os.environ.get("VOLC_BASE_URL", DEFAULT_BASE_URL)
        self._chat_model = chat_model or os.environ.get("VOLC_CHAT_MODEL", "doubao-pro-32k-250715")
        self._image_model = image_model or os.environ.get("VOLC_IMAGE_MODEL", "doubao-seedream-5-0-pro-260128")
        self._video_model = video_model or os.environ.get("VOLC_VIDEO_MODEL", "doubao-seedance-1-5-pro-251215")
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
        file_attachments: Optional[List["UploadedFile"]] = None,
    ) -> CompletionResult:
        """Non-streaming chat completion."""
        # Build messages
        messages = [{"role": "user", "content": text}]
        # If file attachments, add them (simplified: just text prompt)
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
        file_attachments: Optional[List["UploadedFile"]] = None,
    ) -> AsyncIterator[CompletionChunk]:
        """Streaming chat completion."""
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
            reasoning_buffer = ""
            in_reasoning = False
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
                delta = chunk.get("choices", [{}])[0].get("delta", {})
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
        """Generate image via official API."""
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
        if ratio:
            content[0]["text"] += f" --ratio {ratio}"
        payload = {
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
                raise RuntimeError(f"Video generation {status}: {poll_data.get('error', 'unknown')}")
        raise TimeoutError("Video generation timeout")

    async def generate_music(
        self,
        prompt: str,
        lyric: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> MusicGenerationResult:
        # Official API for music may not exist; raise NotImplementedError
        raise NotImplementedError("Music generation not supported via Volcano API")

    async def upload_file(self, file_data: bytes, filename: str) -> UploadedFile:
        # Official API file upload
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
        # Use upload_file and return dict
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
```

- [ ] **Step 2: 写单元测试**

```python
# tests/test_volcano.py
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import httpx
from doubao2api.volcano import VolcanoClient
from doubao2api.client import CompletionChunk, CompletionResult

@pytest.mark.asyncio
async def test_chat_completion():
    client = VolcanoClient(api_key="test-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json = AsyncMock(return_value={
        "id": "chatcmpl-xxx",
        "choices": [{"message": {"role": "assistant", "content": "Hello!", "reasoning_content": ""}}],
        "usage": {},
    })
    mock_resp.raise_for_status = MagicMock()
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_resp)
    client._http = mock_http

    result = await client.chat_completion("Hi")
    assert result.text == "Hello!"
    assert result.conversation_id == "chatcmpl-xxx"
```

- [ ] **Step 3: 运行测试**

Run: `python -m pytest tests/test_volcano.py -v`
Expected: PASS

- [ ] **Step 4: 提交**

```bash
git add doubao2api/volcano.py tests/test_volcano.py
git commit -m "feat: add VolcanoClient for official API"
```

---

### Task 3: Backend 抽象层 (`backends.py`)

**Files:**
- Create: `doubao2api/backends.py`
- Test: `tests/test_backends.py`

- [ ] **Step 1: 实现 `BrowserBackend`（适配器）**

```python
# doubao2api/backends.py
"""Backend abstraction layer for Doubao API.

Provides three backends:
- BrowserBackend: wraps existing BrowserClient (dict-based)
- FreeAccountBackend: multi-account pool (DoubaoChatClient-based)
- VolcanoBackend: official API
"""
from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from . import BrowserClient, DoubaoChatClient
from .account_pool import AccountEntry, AccountPool
from .volcano import VolcanoClient
from .client import (
    CompletionChunk,
    CompletionResult,
    GeneratedImage,
    VideoGenerationResult,
    MusicGenerationResult,
    UploadedFile,
)

log = logging.getLogger(__name__)


class BrowserBackend:
    """Adapter that wraps BrowserClient into the dict-based backend interface.

    This is used as fallback when no accounts are configured.
    """

    def __init__(self, client: BrowserClient):
        self._client = client

    # noinspection PyPep8Naming,PyMethodMayBeStatic
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

    # noinspection PyMethodMayBeStatic
    def extract_conversation_id(self, event: Dict[str, Any]) -> Optional[str]:
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
        return await self._client.chat(text, conversation_id, bot_id, use_deep_think)

    async def chat_with_file(
        self, text: str, file_uri: Union[str, list], file_name: str,
        file_size: int, use_deep_think: int = 0,
    ) -> Dict[str, Any]:
        return await self._client.chat_with_file(text, file_uri, file_name, file_size, use_deep_think)

    async def generate_image(
        self, prompt: str, ratio: Optional[str] = None, ref_image_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._client.generate_image(prompt, ratio, ref_image_key)

    async def generate_music(
        self, prompt: str, lyric: Optional[str] = None, genre: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._client.generate_music(prompt, lyric, genre)

    async def generate_video(
        self, prompt: str, ratio: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._client.generate_video(prompt, ratio)

    async def upload_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        return await self._client.upload_file(file_data, filename)

    async def get_file_download_url(self, uri: str, expire_seconds: int = 3600) -> str:
        return await self._client.get_file_download_url(uri, expire_seconds)

    async def upload_image(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        return await self._client.upload_image(image_bytes, filename)
```

- [ ] **Step 2: 实现 `FreeAccountBackend`**

```python
# doubao2api/backends.py (continued)

class FreeAccountBackend:
    """Multi-account backend using DoubaoChatClient pool.

    Maps domain dataclasses to the dict-based interface expected by the server.
    """

    def __init__(self, pool: AccountPool):
        self._pool = pool
        self._next_account: Optional[AccountEntry] = None

    @property
    def is_ready(self) -> bool:
        return self._pool.healthy_count > 0

    @property
    def needs_captcha(self) -> bool:
        # Check if any account has captcha
        return any(a.needs_captcha for a in self._pool.accounts)

    @property
    def consecutive_failures(self) -> int:
        return 0  # Not applicable at pool level

    @property
    def last_error_code(self) -> int:
        return 0

    def record_success(self):
        if self._next_account:
            self._next_account.mark_success()

    def record_failure(self, error_code: int = 0):
        if self._next_account:
            self._next_account.mark_failure(error_code)
            if self._next_account.needs_captcha or not self._next_account.is_healthy():
                log.warning("Account %s unavailable after error %d, will select next",
                            self._next_account.name, error_code)

    def extract_conversation_id(self, event: Dict[str, Any]) -> Optional[str]:
        return event.get("conversation_id") or None

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
        # Convert DoubaoChatClient CompletionChunk stream to raw dict events
        thinking_toggle = False
        async for chunk in client.chat_stream_completion(
            text=text, need_deep_think=use_deep_think, bot_id=bot_id,
        ):
            if chunk.is_done:
                break
            # Include conversation_id in every event
            base = {"conversation_id": chunk.conversation_id}

            if chunk.error_code:
                yield {"error_code": chunk.error_code, "error_msg": chunk.error_msg, **base}
                return

            if chunk.thinking:
                # Emit thinking toggle if not already in thinking mode
                if not thinking_toggle:
                    yield {"_event": "CONTENT_BLOCK", "content": {"content_block": [{"block_type": 10040}]}, **base}
                    thinking_toggle = True
                yield {"_event": "CHUNK_DELTA", "text": chunk.thinking, **base}
            elif chunk.text:
                if thinking_toggle:
                    yield {"_event": "CONTENT_BLOCK", "content": {"content_block": [{"block_type": 10040}]}, **base}
                    thinking_toggle = False
                yield {"_event": "CHUNK_DELTA", "text": chunk.text, **base}
            elif chunk.block_type == 10040:
                # Thinking toggle marker (if no text but block_type)
                yield {"_event": "CONTENT_BLOCK", "content": {"content_block": [{"block_type": 10040}]}, **base}
                thinking_toggle = not thinking_toggle

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
        # Upload file and chat
        if isinstance(file_uri, list):
            file_refs = file_uri
        else:
            file_refs = [{"uri": file_uri, "name": file_name, "size": file_size}]
        # Upload each file reference
        attachments = []
        for ref in file_refs:
            # file_uri may be a TOS URI already; if not, upload
            if ref["uri"].startswith("tos-"):
                attachments.append(UploadedFile(uri=ref["uri"], name=ref.get("name", "file"), size=ref.get("size", 0)))
            else:
                # Assume it's a URL or data URI; unsupported for now
                raise RuntimeError("Only TOS URIs supported for file chat")
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
        result = await client.generate_image(prompt, ratio, ref_image_key)
        entry.mark_success()
        entry.increment_quota()
        return {
            "images": [
                {
                    "key": img.key,
                    "url": img.ori_url or img.raw_url or img.thumb_url,
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                } for img in result.images
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
        result = await client.generate_music(prompt, lyric, genre)
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
                } for t in result.tracks
            ],
            "prompt": prompt,
        }

    async def generate_video(
        self, prompt: str, ratio: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry = self._select_account(for_video=True)
        client = entry.client
        if client is None:
            raise RuntimeError(f"Account {entry.name} not initialized")
        try:
            result = await client.generate_video(prompt, ratio)
            entry.mark_success()
            entry.increment_quota()
        except Exception as e:
            entry.mark_failure()
            raise
        return {
            "videos": [
                {
                    "video_url": v.video_url,
                    "cover_url": v.cover_url,
                    "width": v.width,
                    "height": v.height,
                    "duration": v.duration,
                } for v in result.videos
            ],
            "prompt": prompt,
        }

    async def upload_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        entry = self._select_account()
        client = entry.client
        if client is None:
            raise RuntimeError(f"Account {entry.name} not initialized")
        result = await client.upload_file(file_data, filename)
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
        result = await client.upload_image(image_bytes, filename)
        entry.mark_success()
        return result
```

#### `VolcanoBackend`（官方通道适配器）

```python
# doubao2api/backends.py (continued)

class VolcanoBackend:
    """Adapter that wraps VolcanoClient into the dict-based backend interface."""

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

    def extract_conversation_id(self, event: Dict[str, Any]) -> Optional[str]:
        return event.get("conversation_id") or None

    async def chat_completion(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        use_deep_think: int = 0,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield simplified raw-dict events from official chat stream."""
        thinking_toggle = False
        async for chunk in self._client.chat_stream_completion(
            text=text, need_deep_think=use_deep_think, bot_id=bot_id,
        ):
            if chunk.is_done:
                break
            base = {"conversation_id": chunk.conversation_id}
            if chunk.error_code:
                yield {"error_code": chunk.error_code, "error_msg": chunk.error_msg, **base}
                return
            if chunk.thinking:
                if not thinking_toggle:
                    yield {"_event": "CONTENT_BLOCK", "content": {"content_block": [{"block_type": 10040}]}, **base}
                    thinking_toggle = True
                yield {"_event": "CHUNK_DELTA", "text": chunk.thinking, **base}
            elif chunk.text:
                if thinking_toggle:
                    yield {"_event": "CONTENT_BLOCK", "content": {"content_block": [{"block_type": 10040}]}, **base}
                    thinking_toggle = False
                yield {"_event": "CHUNK_DELTA", "text": chunk.text, **base}

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
        result = await self._client.generate_image(prompt, ratio, ref_image_key)
        return {
            "images": [
                {
                    "key": img.key,
                    "url": img.ori_url or img.raw_url or img.thumb_url,
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                } for img in result.images
            ],
            "prompt": prompt,
        }

    async def generate_music(
        self, prompt: str, lyric: Optional[str] = None, genre: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise RuntimeError("Music generation not supported via Volcano API")

    async def generate_video(
        self, prompt: str, ratio: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = await self._client.generate_video(prompt, ratio)
        return {
            "videos": [
                {
                    "video_url": v.video_url,
                    "cover_url": v.cover_url,
                    "width": v.width,
                    "height": v.height,
                    "duration": v.duration,
                } for v in result.videos
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
        return await self._client.upload_image(image_bytes, filename)
```

- [ ] **Step 3: 写单元测试**

```python
# tests/test_backends.py
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from doubao2api.backends import FreeAccountBackend
from doubao2api.account_pool import AccountEntry, AccountPool
from doubao2api.client import CompletionChunk, CompletionResult, ImageGenerationResult, GeneratedImage

@pytest.mark.asyncio
async def test_free_account_backend_chat_returns_dict():
    pool = MagicMock(spec=AccountPool)
    entry = AccountEntry(name="test", session_file="test.json", client=MagicMock())
    entry.client.chat_completion = AsyncMock(return_value=CompletionResult(text="Hi", conversation_id="conv-1"))
    pool.select = MagicMock(return_value=entry)
    backend = FreeAccountBackend(pool)
    result = await backend.chat("Hello")
    assert result["text"] == "Hi"
    assert result["conversation_id"] == "conv-1"

@pytest.mark.asyncio
async def test_free_account_backend_chat_completion_stream():
    pool = MagicMock(spec=AccountPool)
    entry = AccountEntry(name="test", session_file="test.json", client=MagicMock())

    async def _mock_stream(*args, **kwargs):
        yield CompletionChunk(text="Hello")
        yield CompletionChunk(is_done=True)

    entry.client.chat_stream_completion = _mock_stream
    pool.select = MagicMock(return_value=entry)
    backend = FreeAccountBackend(pool)
    events = []
    async for event in backend.chat_completion("Hi"):
        events.append(event)
    assert len(events) == 1
    assert events[0]["_event"] == "CHUNK_DELTA"
    assert events[0]["text"] == "Hello"

@pytest.mark.asyncio
async def test_free_account_backend_generate_image():
    pool = MagicMock(spec=AccountPool)
    entry = AccountEntry(name="test", session_file="test.json", client=MagicMock())
    entry.client.generate_image = AsyncMock(return_value=ImageGenerationResult(
        images=[GeneratedImage(ori_url="https://example.com/img.png", width=512, height=512)],
        prompt="test",
    ))
    pool.select = MagicMock(return_value=entry)
    backend = FreeAccountBackend(pool)
    result = await backend.generate_image("test", "1:1")
    assert result["images"][0]["url"] == "https://example.com/img.png"
```

- [ ] **Step 4: 运行测试**

Run: `python -m pytest tests/test_backends.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add doubao2api/backends.py tests/test_backends.py
git commit -m "feat: add Backend abstraction layer (BrowserBackend, FreeAccountBackend)"
```

---

### Task 4: 集成到 `unified_server.py`

**Files:**
- Modify: `doubao2api/unified_server.py`
- Modify: `doubao2api/__init__.py`

- [ ] **Step 1: 添加 `_get_backend()` 和通道选择逻辑**

在 `create_app` 中，在 `lifespan` 内初始化 `AccountPool` 和 `VolcanoClient`，供整个 app 生命周期使用。

首先在文件顶部添加导入：

```python
# doubao2api/unified_server.py (top)
from .account_pool import AccountPool
from .backends import FreeAccountBackend, VolcanoBackend
from .volcano import VolcanoClient
```

然后在 `lifespan` 内添加初始化逻辑：

```python
# Inside create_app, after existing browser client startup:

# ── Multi-account pool ──
_pool: Dict[str, Any] = {}
_volcano: Dict[str, Any] = {}

# After starting browser client, try to load account pool:
accounts_dir = os.environ.get("DOUBAO_ACCOUNTS_DIR", "./accounts")
pool = AccountPool(accounts_dir=accounts_dir)
pool.load_accounts()
if pool.total_count > 0:
    await pool.start()
    _pool["pool"] = pool
    log.info("Account pool loaded with %d accounts (%d healthy)", pool.total_count, pool.healthy_count)
else:
    log.info("No account pool configured, falling back to BrowserClient")

# ── Volcano Engine ──
volc_api_key = os.environ.get("VOLC_API_KEY", "")
if volc_api_key:
    from .backends import VolcanoBackend
    volc = VolcanoClient(api_key=volc_api_key)
    _volcano["client"] = volc
    _volcano["volcano_backend"] = VolcanoBackend(volc)
    log.info("Volcano Engine configured")
```

在 `_get_client()` 旁添加 `_get_backend()`：

```python
def _get_backend(model: str = "doubao") -> Any:
    """Return the appropriate backend for the given model name."""
    if model.startswith("volc-"):
        volc = _volcano.get("volcano_backend")
        if volc is None:
            raise HTTPException(status_code=503, detail="Volcano Engine not configured (set VOLC_API_KEY)")
        return volc
    # doubao-* models → free account pool or fallback BrowserClient/Volcano
    pool = _pool.get("pool")
    if pool and pool.healthy_count > 0:
        free_backend = _pool.get("free_backend")
        if free_backend is None:
            from .backends import FreeAccountBackend
            free_backend = FreeAccountBackend(pool)
            _pool["free_backend"] = free_backend
        return free_backend
    # Fallback to BrowserClient if logged in; otherwise official API
    browser = _browser.get("client")
    if browser and browser.is_ready:
        return _get_client()
    volc = _volcano.get("volcano_backend")
    if volc:
        return volc
    return _get_client()
```

- [ ] **Step 2: 重构端点，将 `_get_client()` 替换为 `_get_backend(model)`**

逐个修改端点（`/v1/chat/completions`、`/v1/images/generations`、`/v1/audio/generations`、`/v1/video/generations`、`/v1/files`、`/v1/files/download`、`/v1/images/upload`、`/v1/chat/completions/with-file`、`/admin/api/probe`）：

```python
# 在每个端点中将:
client = _get_client()
# 替换为:
backend = _get_backend(body.model)  # 或直接从 path 推断模型名
```

注意：
- `_stream_chat` 和 `_collect_chat_response` 的第一个参数名从 `client` 改为 `backend`，但它们调用 `backend.chat_completion`、`backend.extract_conversation_id`、`backend.record_success/failure`，这些方法在三个后端上都存在。
- 对于没有模型名的端点（如 `/v1/files`），使用 `_get_backend("doubao")` 作为默认。
- 对于 `/v1/video/generations` 和 `/v1/audio/generations`，读取请求体中的 `model` 字段（默认 `"doubao-video"` / `"doubao-music"`），再传给 `_get_backend(model)`。若请求体没有 model，可使用 `DOUBAO_VIDEO_CHANNEL` 环境变量决定默认通道。

- [ ] **Step 3: 更新 `/v1/models` 动态列表**

```python
@app.get("/v1/models")
async def list_models(request: Request):
    _check_auth(request)
    models = [
        {"id": m, "object": "model", "owned_by": "doubao", "created": 0}
        for m in CHAT_MODELS
    ] + [
        {"id": m, "object": "model", "owned_by": "qianwen", "created": 0}
        for m in QIANWEN_MODELS
    ] + [
        {"id": "doubao-image", "object": "model", "owned_by": "doubao", "created": 0},
        {"id": "doubao-music", "object": "model", "owned_by": "doubao", "created": 0},
        {"id": "doubao-video", "object": "model", "owned_by": "doubao", "created": 0},
    ]
    # Add Volcano models if configured
    volc = _volcano.get("client")
    if volc and volc.is_ready:
        models += [
            {"id": "volc-chat", "object": "model", "owned_by": "volcano", "created": 0},
            {"id": "volc-image", "object": "model", "owned_by": "volcano", "created": 0},
            {"id": "volc-video", "object": "model", "owned_by": "volcano", "created": 0},
        ]
    return {"object": "list", "data": models}
```

- [ ] **Step 4: 更新 `/health` 端点**

```python
@app.get("/health")
async def health():
    pool = _pool.get("pool")
    volc = _volcano.get("client")
    result = {
        "status": "ok",
        "pool_ready": pool is not None and pool.healthy_count > 0,
        "pool_total": pool.total_count if pool else 0,
        "pool_healthy": pool.healthy_count if pool else 0,
        "volcano_configured": volc is not None and volc.is_ready,
        "browser_ready": _browser.get("client") is not None and _browser["client"].is_ready,
    }
    # Add legacy fields for backward compat
    browser = _browser.get("client")
    if browser:
        result["logged_in"] = browser.is_ready
        result["consecutive_failures"] = browser.consecutive_failures
        result["needs_captcha"] = browser.needs_captcha
        result["last_error_code"] = browser.last_error_code
    result["expert_degraded"] = _expert_tracker.is_degraded
    qw = _qianwen.get("client")
    result["qianwen_ready"] = qw.is_ready if qw else False
    return result
```

- [ ] **Step 5: 添加 Admin API 端点**

在 `# ── Admin Dashboard & Auth ──` 区域添加：

```python
# ── Account Management API ──

@app.get("/admin/api/accounts")
async def admin_list_accounts(request: Request):
    _check_auth(request)
    pool = _pool.get("pool")
    if not pool:
        return JSONResponse({"accounts": [], "message": "No account pool configured"})
    return JSONResponse({
        "accounts": [a.to_dict() for a in pool.accounts],
        "total": pool.total_count,
        "healthy": pool.healthy_count,
    })

@app.post("/admin/api/accounts")
async def admin_add_account(request: Request):
    _check_auth(request)
    pool = _pool.get("pool")
    if not pool:
        raise HTTPException(503, "Account pool not initialized")
    form = await request.form()
    session_file = form.get("session_file", "").strip()
    name = form.get("name", "").strip() or os.path.splitext(os.path.basename(session_file))[0]
    if not session_file:
        raise HTTPException(400, "Missing session_file")
    # Copy file to accounts dir
    accounts_dir = os.environ.get("DOUBAO_ACCOUNTS_DIR", "./accounts")
    os.makedirs(accounts_dir, exist_ok=True)
    dest = os.path.join(accounts_dir, f"{name}.json")
    content = await form.get("file").read() if form.get("file") else open(session_file, "rb").read()
    with open(dest, "wb") as f:
        f.write(content)
    # Reload pool and rebuild clients
    pool.load_accounts()
    await pool.start()
    # Rebuild free backend
    _pool.pop("free_backend", None)
    return JSONResponse({"status": "ok", "name": name, "session_file": dest})

@app.delete("/admin/api/accounts/{name}")
async def admin_delete_account(name: str, request: Request):
    _check_auth(request)
    pool = _pool.get("pool")
    if not pool:
        raise HTTPException(503, "Account pool not initialized")
    entry = pool.get_by_name(name)
    if not entry:
        raise HTTPException(404, f"Account '{name}' not found")
    # Remove from pool
    pool._accounts = [a for a in pool._accounts if a.name != name]
    # Remove session file
    if os.path.exists(entry.session_file):
        os.remove(entry.session_file)
    await pool.start()
    _pool.pop("free_backend", None)
    return JSONResponse({"status": "ok"})

@app.post("/admin/api/accounts/{name}/toggle")
async def admin_toggle_account(name: str, request: Request):
    _check_auth(request)
    pool = _pool.get("pool")
    if not pool:
        raise HTTPException(503, "Account pool not initialized")
    entry = pool.get_by_name(name)
    if not entry:
        raise HTTPException(404, f"Account '{name}' not found")
    entry.enabled = not entry.enabled
    return JSONResponse({"status": "ok", "name": name, "enabled": entry.enabled})

@app.post("/admin/api/accounts/{name}/probe")
async def admin_probe_account(name: str, request: Request):
    _check_auth(request)
    pool = _pool.get("pool")
    if not pool:
        raise HTTPException(503, "Account pool not initialized")
    entry = pool.get_by_name(name)
    if not entry:
        raise HTTPException(404, f"Account '{name}' not found")
    if not entry.client:
        raise HTTPException(503, "Account client not initialized")
    try:
        result = await entry.client.chat("1+1=?只回答数字", need_deep_think=0)
        return JSONResponse({"status": "ok", "text": result.text, "ms": 0})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)})

@app.get("/admin/api/channel")
async def admin_channel_status(request: Request):
    _check_auth(request)
    pool = _pool.get("pool")
    volc = _volcano.get("client")
    return JSONResponse({
        "free_accounts": {
            "configured": pool is not None and pool.total_count > 0,
            "total": pool.total_count if pool else 0,
            "healthy": pool.healthy_count if pool else 0,
        },
        "volcano": {
            "configured": volc is not None and volc.is_ready,
            "api_key_set": bool(os.environ.get("VOLC_API_KEY", "")),
        },
        "default_channel": os.environ.get("DOUBAO_VIDEO_CHANNEL", "free"),
    })
```

- [ ] **Step 6: 提交**

```bash
git add doubao2api/unified_server.py doubao2api/__init__.py
git commit -m "feat: integrate AccountPool, VolcanoClient, and backend selection into server"
```

---

### Task 5: Admin 面板扩展

**Files:**
- Modify: `doubao2api/static/admin.html`

- [ ] **Step 1: 在现有 Vue 3 面板中添加两个新 Tab**

在 `tabs` 数组末尾添加：
```javascript
{name: '账号管理', id: 'accounts'},
{name: '通道配置', id: 'channel'},
```

在 `icons` 数组对应位置添加 SVG 图标。

- [ ] **Step 2: 实现「账号管理」Tab**

```html
<section v-show="activeTab===4">
  <div class="flex items-center gap-3 mb-4">
    <button @click="fetchAccounts" class="btn">刷新</button>
    <span class="text-xs text-slate-400">总计 {{accountData.total}} 账号，健康 {{accountData.healthy}}</span>
  </div>
  <div class="card p-0 overflow-x-auto">
    <table class="w-full text-sm">
      <thead><tr class="text-slate-400 text-xs border-b border-slate-700">
        <th class="text-left p-3">名称</th>
        <th class="text-left p-3">状态</th>
        <th class="text-left p-3">成功</th>
        <th class="text-left p-3">失败</th>
        <th class="text-left p-3">连续失败</th>
        <th class="text-left p-3">风控</th>
        <th class="text-left p-3">今日额度</th>
        <th class="text-left p-3">操作</th>
      </tr></thead>
      <tbody>
        <tr v-for="acct in accountData.accounts" :key="acct.name" class="border-b border-slate-800 hover:bg-slate-800/30">
          <td class="p-3">{{acct.name}}</td>
          <td class="p-3">
            <span :class="acct.is_healthy?'text-green-400':'text-red-400'">
              {{acct.enabled ? (acct.is_healthy ? '健康' : '异常') : '停用'}}
            </span>
          </td>
          <td class="p-3">{{acct.success_count}}</td>
          <td class="p-3">{{acct.failure_count}}</td>
          <td class="p-3">{{acct.consecutive_failures}}</td>
          <td class="p-3">{{acct.needs_captcha ? '⚠️' : '—'}}</td>
          <td class="p-3">{{acct.daily_quota_used}}</td>
          <td class="p-3">
            <button @click="toggleAccount(acct.name)" class="btn-sm">{{acct.enabled?'停用':'启用'}}</button>
            <button @click="probeAccount(acct.name)" class="btn-sm ml-1">探活</button>
            <button @click="deleteAccount(acct.name)" class="btn-sm ml-1 text-red-400">删除</button>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</section>
```

- [ ] **Step 3: 实现「通道配置」Tab**

```html
<section v-show="activeTab===5">
  <div class="card mb-4">
    <h3 class="text-sm font-medium mb-3">免费账号池</h3>
    <div class="grid grid-cols-2 gap-3 text-sm">
      <div><span class="text-slate-400">已配置</span>: {{channelData.free_accounts.configured ? '是' : '否'}}</div>
      <div><span class="text-slate-400">总数</span>: {{channelData.free_accounts.total}}</div>
      <div><span class="text-slate-400">健康</span>: {{channelData.free_accounts.healthy}}</div>
    </div>
  </div>
  <div class="card">
    <h3 class="text-sm font-medium mb-3">火山引擎官方通道</h3>
    <div class="grid grid-cols-2 gap-3 text-sm">
      <div><span class="text-slate-400">已配置</span>: {{channelData.volcano.configured ? '是' : '否'}}</div>
      <div><span class="text-slate-400">API Key</span>: {{channelData.volcano.api_key_set ? '已设置' : '未设置'}}</div>
    </div>
    <div class="mt-3">
      <span class="text-xs text-slate-500">默认视频通道: {{channelData.default_channel}}</span>
    </div>
  </div>
</section>
```

- [ ] **Step 4: 添加 Vue 方法**

```javascript
// 在 methods 中添加:
async fetchAccounts() {
  try {
    const r = await fetch('/admin/api/accounts' + (apiKey ? `?key=${apiKey}` : ''));
    this.accountData = await r.json();
  } catch(e) { this.accountData = {accounts: [], total: 0, healthy: 0}; }
},
async toggleAccount(name) {
  await fetch(`/admin/api/accounts/${name}/toggle`, {method:'POST'});
  this.fetchAccounts();
},
async probeAccount(name) {
  const r = await fetch(`/admin/api/accounts/${name}/probe`, {method:'POST'});
  alert(await r.text());
},
async deleteAccount(name) {
  if (!confirm(`删除账号 ${name}?`)) return;
  await fetch(`/admin/api/accounts/${name}`, {method:'DELETE'});
  this.fetchAccounts();
},
async fetchChannel() {
  const r = await fetch('/admin/api/channel' + (apiKey ? `?key=${apiKey}` : ''));
  this.channelData = await r.json();
}
```

- [ ] **Step 5: 在 `created` 钩子中调用 `fetchAccounts()` 和 `fetchChannel()`**

- [ ] **Step 6: 提交**

```bash
git add doubao2api/static/admin.html
git commit -m "feat: add account management and channel config tabs to admin UI"
```

---

### Task 6: 更新文档和环境变量

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: 更新 `.env.example`**

```bash
# 服务端口
DOUBAO_PORT=9090
# 监听地址
DOUBAO_HOST=127.0.0.1
# API 密钥 (用于保护 /v1/* 端点，留空则不鉴权)
DOUBAO_API_KEY=
# 每分钟请求限制
DOUBAO_RPM_LIMIT=50
# 多账号 session 目录 (默认 ./accounts)
DOUBAO_ACCOUNTS_DIR=./accounts
# 默认视频通道 (free 或 volc)
DOUBAO_VIDEO_CHANNEL=free

# 火山引擎官方 API
VOLC_API_KEY=
VOLC_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
VOLC_CHAT_MODEL=doubao-pro-32k-250715
VOLC_IMAGE_MODEL=doubao-seedream-5-0-pro-260128
VOLC_VIDEO_MODEL=doubao-seedance-1-5-pro-251215
```

- [ ] **Step 1b: 添加缺失依赖到 `requirements.txt`**

```bash
echo "httpx>=0.27" >> requirements.txt
```

- [ ] **Step 2: 更新 README**

在 README 中增加：
- 多账号配置说明（`accounts/` 目录 + `accounts.json`）
- 火山引擎通道配置说明
- Admin 面板新增功能说明

- [ ] **Step 3: 提交**

```bash
git add .env.example README.md
git commit -m "docs: update env example and README for multi-account and volcano features"
```

---

### Task 7: 集成测试与最终验证

- [ ] **Step 1: 运行所有测试**

```bash
python -m pytest tests/ -v --disable-warnings
```

Expected: 所有新测试 PASS。原有测试由于缺少 `base_url` fixture 仍报错，不影响。

- [ ] **Step 2: 手动验证后端选择**

启动服务：`python -m doubao2api`
检查 `/health` 返回 `pool_ready` 等字段。
访问 `/admin` 确认新 Tab 出现。

- [ ] **Step 3: 最终提交**

```bash
git add -A
git commit -m "feat: complete doubao reverse proxy with multi-account pool and volcano channel"
```
"""Multi-account pool for Doubao free accounts.

Loads multiple session files from a directory, creates one
``DoubaoChatClient`` per account, and provides round-robin selection
with health tracking and daily quota counters.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import DoubaoChatClient

log = logging.getLogger(__name__)

RISK_CAPTCHA_CODE = 710022004


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
    """One Doubao account backed by a session file."""

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
        if error_code == RISK_CAPTCHA_CODE:
            self.needs_captcha = True
            log.warning("Account %s needs captcha (710022004)", self.name)
        if self.consecutive_failures >= 5:
            self.enabled = False
            log.warning(
                "Account %s disabled after %d consecutive failures",
                self.name,
                self.consecutive_failures,
            )
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


class AccountPool:
    """Manages multiple DoubaoChatClient accounts with round-robin selection."""

    def __init__(
        self,
        accounts_dir: str = "./accounts",
        captcha_handler="auto",
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
        for cfg_path in config_paths:
            if cfg_path.exists():
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                except (json.JSONDecodeError, OSError) as exc:
                    log.error("Failed to read %s: %s", cfg_path, exc)
                    continue
                for acct in config.get("accounts", []):
                    session_file = acct.get("session_file", "")
                    if not session_file:
                        continue
                    if not os.path.isabs(session_file):
                        session_file = str(self._accounts_dir / session_file)
                    self._accounts.append(AccountEntry(
                        name=acct.get("name", os.path.basename(session_file)),
                        session_file=session_file,
                        enabled=acct.get("enabled", True),
                        weight=acct.get("weight", 1),
                        backend=acct.get("backend", "http"),
                    ))
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
                    json.dumps(
                        {"cookies": _parse_cookie_header(cookie_header), "params": {}},
                        ensure_ascii=False,
                    ),
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

import json
from pathlib import Path

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

    def test_increment_quota_increments(self):
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
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        # Also create the session file so the scan doesn't double-add
        session = {"cookies": {"sessionid": "test-session"}, "params": {}}
        (tmp_path / "main.json").write_text(json.dumps(session), encoding="utf-8")
        pool.load_accounts()
        assert len(pool.accounts) == 1
        assert pool.accounts[0].name == "主号"
        assert pool.accounts[0].weight == 2

    def test_load_accounts_from_directory(self, tmp_path):
        pool = AccountPool(accounts_dir=str(tmp_path))
        # Create a session file
        session = {"cookies": {"sessionid": "test"}, "params": {}}
        (tmp_path / "my_account.json").write_text(json.dumps(session), encoding="utf-8")
        pool.load_accounts()
        assert len(pool.accounts) == 1
        assert pool.accounts[0].name == "my_account"

    def test_load_accounts_skips_config_file_in_dir_scan(self, tmp_path):
        pool = AccountPool(accounts_dir=str(tmp_path))
        cfg = {"accounts": []}
        (tmp_path / "accounts.json").write_text(json.dumps(cfg), encoding="utf-8")
        pool.load_accounts()
        # accounts.json must not be treated as a session file
        assert all(a.session_file.endswith("accounts.json") is False for a in pool.accounts)
        assert pool.total_count == 0

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

    def test_get_by_name(self, tmp_path):
        pool = AccountPool(accounts_dir=str(tmp_path))
        pool._accounts.append(AccountEntry(name="alpha", session_file="a.json"))
        assert pool.get_by_name("alpha") is not None
        assert pool.get_by_name("missing") is None

import importlib
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.fixture
def app_module(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-that-is-long-enough-for-security-checks")
    monkeypatch.setenv("DATABASE_NAME", str(tmp_path / "production-foundation.db"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    sys.modules.pop("main", None)
    module = importlib.import_module("main")
    module.create_tables()
    return module


def test_quota_reservation_is_atomic_and_refundable(app_module):
    app_module.execute(
        "INSERT INTO users (name, email, password, request_balance) VALUES (?, ?, ?, ?)",
        ("Quota User", "quota@example.com", app_module.hash_password("safe-password"), 1),
    )
    user = app_module.fetch_one("SELECT * FROM users WHERE email = ?", ("quota@example.com",))

    app_module.reserve_request_quota(user["id"])
    reserved = app_module.fetch_one("SELECT request_count, request_balance FROM users WHERE id = ?", (user["id"],))
    assert (reserved["request_count"], reserved["request_balance"]) == (1, 0)

    with pytest.raises(app_module.HTTPException) as error:
        app_module.reserve_request_quota(user["id"])
    assert error.value.status_code == 402

    app_module.refund_request_quota(user["id"])
    refunded = app_module.fetch_one("SELECT request_count, request_balance FROM users WHERE id = ?", (user["id"],))
    assert (refunded["request_count"], refunded["request_balance"]) == (0, 1)

    app_module.record_audit_event(user["id"], "user.quota_checked", "user", user["id"], {"source": "test"})
    event = app_module.fetch_one("SELECT * FROM audit_events WHERE target_id = ?", (str(user["id"]),))
    assert event["action"] == "user.quota_checked"


def test_redis_rate_limit_is_shared_when_configured(app_module):
    class FakeRedis:
        def __init__(self):
            self.values = {}
            self.expiries = {}

        def incr(self, key):
            self.values[key] = self.values.get(key, 0) + 1
            return self.values[key]

        def expire(self, key, seconds):
            self.expiries[key] = seconds

        def ttl(self, key):
            return self.expiries.get(key, 0)

    fake_redis = FakeRedis()
    app_module.REDIS_CLIENT = fake_redis
    app_module.enforce_rate_limit("code", "same-user", limit=1, window_seconds=30)
    with pytest.raises(app_module.HTTPException) as error:
        app_module.enforce_rate_limit("code", "same-user", limit=1, window_seconds=30)

    assert error.value.status_code == 429
    assert fake_redis.expiries


def test_postgres_url_normalization_and_safe_release_archive(app_module, tmp_path):
    assert app_module.normalize_database_url("postgres://db.example/app") == (
        "postgresql+psycopg://db.example/app"
    )
    statement, bindings = app_module.qmark_to_named("SELECT '?' AS literal WHERE id = ?", (7,))
    assert statement == "SELECT '?' AS literal WHERE id = :p0"
    assert bindings == {"p0": 7}

    from tools.build_release import ROOT, build_release

    archive_path = tmp_path / "missingly-release.zip"
    assert build_release(archive_path) > 0
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert "main.py" in names
    assert ".env" not in names
    assert "users.db" not in names
    assert all(not Path(name).suffix == ".log" for name in names)
    assert ROOT / "main.py"

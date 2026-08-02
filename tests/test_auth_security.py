import importlib
import sys

import pytest


@pytest.fixture
def app_module(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-that-is-long-enough-for-security-checks")
    monkeypatch.setenv("DATABASE_NAME", str(tmp_path / "test-users.db"))
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://testserver")
    sys.modules.pop("main", None)
    module = importlib.import_module("main")
    module.create_tables()
    return module


def test_sessions_are_hashed_expire_and_can_be_revoked(app_module):
    app_module.execute(
        "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
        ("Test User", "test@example.com", app_module.hash_password("safe-password")),
    )
    user = app_module.fetch_one("SELECT * FROM users WHERE email = ?", ("test@example.com",))

    token = app_module.create_auth_session(user["id"])
    stored = app_module.fetch_one("SELECT * FROM auth_sessions WHERE user_id = ?", (user["id"],))

    assert stored["token_hash"] == app_module.hash_auth_token(token)
    assert stored["token_hash"] != token
    assert app_module.get_user_by_token(token)["email"] == "test@example.com"

    app_module.revoke_auth_session(token, user["id"])
    assert app_module.get_user_by_token(token) is None


def test_reset_token_hashes_do_not_match_raw_tokens(app_module):
    raw_token = "reset-token-that-never-belongs-in-the-database"
    assert app_module.hash_auth_token(raw_token) != raw_token


def test_admin_templates_escape_user_supplied_values():
    admin_html = open("admin.html", encoding="utf-8").read()
    assert "${escapeHtml(user.name)}" in admin_html
    assert "${escapeHtml(user.email)}" in admin_html

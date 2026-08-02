import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_and_module(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-that-is-long-enough-for-security-checks")
    monkeypatch.setenv("DATABASE_NAME", str(tmp_path / "projects.db"))
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://testserver")
    sys.modules.pop("main", None)
    module = importlib.import_module("main")
    module.create_tables()
    module.execute(
        "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
        ("Project Owner", "owner@example.com", module.hash_password("safe-password")),
    )
    module.execute(
        "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
        ("Other User", "other@example.com", module.hash_password("safe-password")),
    )
    owner = module.fetch_one("SELECT * FROM users WHERE email = ?", ("owner@example.com",))
    other = module.fetch_one("SELECT * FROM users WHERE email = ?", ("other@example.com",))
    return TestClient(module.app), module.create_auth_session(owner["id"]), module.create_auth_session(other["id"])


def test_project_revisions_are_private_and_versioned(client_and_module):
    client, owner_token, other_token = client_and_module
    owner_headers = {"Authorization": f"Bearer {owner_token}"}
    other_headers = {"Authorization": f"Bearer {other_token}"}

    created = client.post(
        "/projects",
        headers=owner_headers,
        json={"name": "  Landing Page  ", "current_code": "<h1>v1</h1>", "prompt": "make v1"},
    )
    assert created.status_code == 201
    project = created.json()["project"]
    assert project["name"] == "Landing Page"
    assert project["current_revision"] == 1

    updated = client.patch(
        f"/projects/{project['id']}",
        headers=owner_headers,
        json={"current_code": "<h1>v2</h1>", "prompt": "make v2"},
    )
    assert updated.status_code == 200
    assert updated.json()["project"]["current_revision"] == 2

    revisions = client.get(f"/projects/{project['id']}/revisions", headers=owner_headers)
    assert revisions.status_code == 200
    assert [item["revision_number"] for item in revisions.json()["revisions"]] == [2, 1]

    first_revision = client.get(f"/projects/{project['id']}/revisions/1", headers=owner_headers)
    assert first_revision.json()["revision"]["code"] == "<h1>v1</h1>"

    forbidden = client.get(f"/projects/{project['id']}", headers=other_headers)
    assert forbidden.status_code == 404

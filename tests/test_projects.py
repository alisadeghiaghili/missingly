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


def test_project_collaboration_export_and_sandboxed_publishing(client_and_module):
    client, owner_token, other_token = client_and_module
    owner_headers = {"Authorization": f"Bearer {owner_token}"}
    other_headers = {"Authorization": f"Bearer {other_token}"}
    created = client.post(
        "/projects",
        headers=owner_headers,
        json={
            "name": "Collaborative Site",
            "current_code": "<script>window.parent.postMessage('nope', '*')</script><main>safe snapshot</main>",
            "prompt": "make it collaborative",
        },
    )
    project_id = created.json()["project"]["id"]

    member = client.post(
        f"/projects/{project_id}/members",
        headers=owner_headers,
        json={"email": "other@example.com", "role": "editor"},
    )
    assert member.status_code == 201
    member_id = member.json()["member"]["id"]

    shared_list = client.get("/projects", headers=other_headers)
    assert shared_list.json()["projects"][0]["role"] == "editor"
    assert client.get(f"/projects/{project_id}", headers=other_headers).status_code == 200
    assert client.patch(
        f"/projects/{project_id}",
        headers=other_headers,
        json={"current_code": "<main>editor revision</main>", "prompt": "editor change"},
    ).status_code == 200

    changed = client.patch(
        f"/projects/{project_id}/members/{member_id}",
        headers=owner_headers,
        json={"role": "viewer"},
    )
    assert changed.status_code == 200
    assert client.patch(
        f"/projects/{project_id}", headers=other_headers, json={"name": "not allowed"}
    ).status_code == 403
    assert client.get(f"/projects/{project_id}/revisions", headers=other_headers).status_code == 200
    assert client.get(f"/projects/{project_id}/members", headers=other_headers).status_code == 404

    publication = client.post(
        f"/projects/{project_id}/publications",
        headers=owner_headers,
        json={"slug": "collaborative-demo", "revision_number": 1},
    )
    assert publication.status_code == 201
    assert publication.json()["publication"]["revision_number"] == 1
    public = client.get("/published/collaborative-demo")
    assert public.status_code == 200
    assert "sandbox='allow-scripts allow-forms'" in public.text
    assert "<script>window.parent.postMessage" not in public.text
    assert "&lt;script&gt;window.parent.postMessage" in public.text

    exported = client.get(f"/projects/{project_id}/export?revision_number=2", headers=other_headers)
    assert exported.status_code == 200
    assert exported.headers["content-disposition"].startswith("attachment;")
    assert "editor revision" in exported.text

    removed = client.delete(
        f"/projects/{project_id}/publications/{publication.json()['publication']['id']}",
        headers=owner_headers,
    )
    assert removed.status_code == 200
    assert client.get("/published/collaborative-demo").status_code == 404

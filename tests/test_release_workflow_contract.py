"""Contract tests for ``.github/workflows/release.yml`` (WP-01-R, F-050)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"
)
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
PUBLISH_ACTION = "pypa/gh-action-pypi-publish"
RELEASE_ACTION = "softprops/action-gh-release"


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(scope="module")
def jobs(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return workflow["jobs"]


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return job.get("steps", [])


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _ancestors(jobs: dict[str, dict[str, Any]], name: str) -> set[str]:
    seen: set[str] = set()
    pending = _needs(jobs[name])
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(_needs(jobs[current]))
    return seen


def _jobs_using(jobs: dict[str, dict[str, Any]], action: str) -> list[str]:
    return [
        name
        for name, job in jobs.items()
        if any(
            str(step.get("uses", "")).startswith(action + "@") for step in _steps(job)
        )
    ]


def _run_text(job: dict[str, Any]) -> str:
    return "\n".join(str(step.get("run", "")) for step in _steps(job))


def _job_permissions(job: dict[str, Any]) -> dict[str, str]:
    permissions = job.get("permissions")
    assert isinstance(permissions, dict), "job must declare explicit permissions"
    return permissions


def _test_and_build_jobs(jobs: dict[str, dict[str, Any]]) -> list[str]:
    return [
        name
        for name, job in jobs.items()
        if "pytest" in _run_text(job) and "build" in _run_text(job)
    ]


def test_workflow_default_permissions_are_read_only(workflow: dict[str, Any]) -> None:
    assert workflow.get("permissions") == {"contents": "read"}


def test_a_job_runs_tests_and_builds_distributions(
    jobs: dict[str, dict[str, Any]],
) -> None:
    assert _test_and_build_jobs(jobs), "no job runs pytest and builds the distributions"


def test_publish_job_needs_the_test_and_build_job(
    jobs: dict[str, dict[str, Any]],
) -> None:
    publishers = _jobs_using(jobs, PUBLISH_ACTION)
    assert len(publishers) == 1
    verified = set(_test_and_build_jobs(jobs))
    assert verified
    assert verified & _ancestors(jobs, publishers[0])


def test_build_job_uploads_the_distributions_and_publish_downloads_them(
    jobs: dict[str, dict[str, Any]],
) -> None:
    publisher = jobs[_jobs_using(jobs, PUBLISH_ACTION)[0]]
    assert _jobs_using(jobs, "actions/upload-artifact")
    assert any(
        str(step.get("uses", "")).startswith("actions/download-artifact@")
        for step in _steps(publisher)
    )


def test_tag_version_and_wheel_version_are_checked(
    jobs: dict[str, dict[str, Any]],
) -> None:
    (name,) = _test_and_build_jobs(jobs)
    check_steps = [
        step
        for step in _steps(jobs[name])
        if "GITHUB_REF_NAME" in str(step.get("run", ""))
        or "GITHUB_REF_NAME" in str(step.get("env", ""))
    ]
    assert check_steps, "no step reads the pushed tag"
    text = "\n".join(str(step.get("run", "")) for step in check_steps)
    assert "__version__" in text, "tag is not compared with missingly.__version__"
    assert "dist/" in text and ".whl" in text, (
        "tag is not compared with the built wheel"
    )
    assert "exit 1" in text or "sys.exit" in text or "raise SystemExit" in text


def test_publish_uses_trusted_publishing_only(jobs: dict[str, dict[str, Any]]) -> None:
    (name,) = _jobs_using(jobs, PUBLISH_ACTION)
    job = jobs[name]
    assert _job_permissions(job).get("id-token") == "write"
    assert job.get("environment"), "publish job needs a GitHub environment"
    for step in _steps(job):
        if str(step.get("uses", "")).startswith(PUBLISH_ACTION + "@"):
            inputs = step.get("with", {}) or {}
            assert "password" not in inputs
            assert "user" not in inputs
    assert "PYPI_API_TOKEN" not in WORKFLOW_PATH.read_text(encoding="utf-8")


def test_id_token_write_is_granted_to_the_publish_job_only(
    jobs: dict[str, dict[str, Any]],
) -> None:
    (publisher,) = _jobs_using(jobs, PUBLISH_ACTION)
    for name, job in jobs.items():
        if name != publisher:
            assert (job.get("permissions") or {}).get("id-token") != "write"


def test_github_release_is_a_separate_minimal_job(
    jobs: dict[str, dict[str, Any]],
) -> None:
    (releaser,) = _jobs_using(jobs, RELEASE_ACTION)
    (publisher,) = _jobs_using(jobs, PUBLISH_ACTION)
    assert releaser != publisher
    assert _job_permissions(jobs[releaser]) == {"contents": "write"}
    assert publisher in _ancestors(jobs, releaser)
    assert _job_permissions(jobs[publisher]).get("contents") != "write"


def test_third_party_actions_are_pinned_to_full_commit_shas(
    jobs: dict[str, dict[str, Any]],
) -> None:
    refs = [
        str(step["uses"])
        for job in jobs.values()
        for step in _steps(job)
        if "uses" in step
    ]
    assert refs
    for ref in refs:
        _, _, version = ref.partition("@")
        assert FULL_SHA.match(version), (
            f"{ref} is not pinned to a 40-character commit SHA"
        )

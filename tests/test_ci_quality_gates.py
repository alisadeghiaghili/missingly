"""Contracts for the CI workflows, coverage gate and fixture ignore rules.

These tests pin the quality gates that WP-02 repairs: the R-oracle workflow
must run when Missingly code changes, the coverage threshold must have a single
source of truth, fixture CSVs must never be git-ignored, actionlint must lint
the workflows, and third-party actions must be pinned to commit SHAs.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    _toml = pytest.importorskip("tomli")


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPOSITORY_ROOT / ".github" / "workflows"
CI_WORKFLOW = WORKFLOW_DIR / "ci.yml"
ORACLE_WORKFLOW = WORKFLOW_DIR / "r-oracle.yml"
DOCS_WORKFLOW = WORKFLOW_DIR / "docs-pages.yml"
PINNED_WORKFLOWS = (CI_WORKFLOW, ORACLE_WORKFLOW, DOCS_WORKFLOW)
FIXTURE_DIR = REPOSITORY_ROOT / "tests" / "fixtures"
HYPOTHETICAL_FIXTURE = "tests/fixtures/new_oracle/sample.csv"
MISSINGLY_PATH = "missingly/**"
ORACLE_FIXTURE_PATHS = (
    "tests/fixtures/little_mcar/**",
    "tests/fixtures/mice_numeric_fcs/**",
)
MINIMUM_COVERAGE = 90
COVERAGE_FLAG = "--cov-fail-under"
_SHA_PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
_SHA_PIN_WITH_TAG_COMMENT = re.compile(
    r"uses:\s*\S+@[0-9a-f]{40}\s+#\s*v\d+\.\d+\.\d+\s*$"
)
_USES_LINE = re.compile(r"^\s*(?:-\s*)?uses:\s*\S")


def _load_workflow(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _triggers(workflow: Dict[str, Any]) -> Dict[str, Any]:
    # PyYAML follows YAML 1.1, where a bare ``on`` key is parsed as ``True``.
    return workflow.get("on", workflow.get(True, {}))


def _steps(workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for job in workflow.get("jobs", {}).values():
        steps.extend(job.get("steps", []))
    return steps


def _uses_values(path: Path) -> List[str]:
    workflow = _load_workflow(path)
    return [step["uses"] for step in _steps(workflow) if "uses" in step]


def _pyproject() -> Dict[str, Any]:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        return _toml.load(handle)


def _git(*args: str) -> subprocess.CompletedProcess:
    git = shutil.which("git")
    if git is None or not (REPOSITORY_ROOT / ".git").exists():
        pytest.skip("requires a git checkout")
    return subprocess.run(
        [git, *args],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_oracle_workflow_runs_on_missingly_code_changes() -> None:
    triggers = _triggers(_load_workflow(ORACLE_WORKFLOW))
    for event in ("pull_request", "push"):
        paths = triggers[event]["paths"]
        assert MISSINGLY_PATH in paths, f"{event} trigger must watch {MISSINGLY_PATH}"


def test_oracle_workflow_keeps_fixture_triggers() -> None:
    triggers = _triggers(_load_workflow(ORACLE_WORKFLOW))
    for event in ("pull_request", "push"):
        paths = triggers[event]["paths"]
        for fixture_path in ORACLE_FIXTURE_PATHS:
            assert fixture_path in paths, f"{event} trigger lost {fixture_path}"


def test_coverage_is_configured_in_pyproject_with_branch_and_threshold() -> None:
    coverage = _pyproject()["tool"]["coverage"]
    assert coverage["run"]["branch"] is True
    assert coverage["run"]["relative_files"] is True
    assert coverage["report"]["fail_under"] >= MINIMUM_COVERAGE


def test_no_workflow_overrides_the_pyproject_coverage_threshold() -> None:
    for path in WORKFLOW_DIR.glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        assert COVERAGE_FLAG not in text, (
            f"{path.name} must rely on pyproject fail_under"
        )


def test_ci_pytest_measures_branch_coverage() -> None:
    commands = [step.get("run", "") for step in _steps(_load_workflow(CI_WORKFLOW))]
    pytest_runs = [
        command for command in commands if "pytest" in command and "--cov=" in command
    ]
    assert pytest_runs, "ci.yml must run pytest with coverage"
    assert all("--cov-branch" in command for command in pytest_runs)


def test_every_fixture_csv_on_disk_is_tracked_by_git() -> None:
    fixtures = sorted(FIXTURE_DIR.rglob("*.csv"))
    assert fixtures, "expected at least one oracle fixture CSV"
    for fixture in fixtures:
        relative = fixture.relative_to(REPOSITORY_ROOT).as_posix()
        result = _git("ls-files", "--error-unmatch", relative)
        assert result.returncode == 0, f"{relative} is not tracked by git"


def test_fixture_csv_path_is_not_ignored_for_new_fixtures() -> None:
    result = _git("check-ignore", "-q", "--no-index", HYPOTHETICAL_FIXTURE)
    assert result.returncode == 1, f"{HYPOTHETICAL_FIXTURE} is ignored by .gitignore"


def test_ci_lints_workflows_with_actionlint() -> None:
    steps = _steps(_load_workflow(CI_WORKFLOW))
    lint_runs = [
        step.get("run", "") for step in steps if "actionlint" in step.get("run", "")
    ]
    assert any("actionlint-py" in command for command in lint_runs)
    assert any(command.strip().endswith("actionlint") for command in lint_runs)


@pytest.mark.parametrize("path", PINNED_WORKFLOWS, ids=lambda p: p.name)
def test_third_party_actions_are_pinned_to_commit_shas(path: Path) -> None:
    for uses in _uses_values(path):
        if uses.startswith("./"):
            continue
        assert _SHA_PIN.match(uses), (
            f"{path.name}: {uses} is not pinned to a 40-char SHA"
        )


@pytest.mark.parametrize("path", PINNED_WORKFLOWS, ids=lambda p: p.name)
def test_pinned_actions_carry_a_version_comment(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if _USES_LINE.match(line) and not line.strip().split("uses:")[
            1
        ].strip().startswith("./"):
            assert _SHA_PIN_WITH_TAG_COMMENT.search(line), (
                f"{path.name}: missing '# vX.Y.Z': {line}"
            )

"""Regression contracts between ``pyproject.toml`` and the CI workflows.

A metadata-only edit once rewrote ``pyproject.toml`` and silently dropped the
optional extras, the dynamic version source and the tool configuration that the
workflows depend on.  pip only warns about an unknown extra, so the failure
surfaced later as missing test tools.  These tests make that drift fail fast.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Set

import pytest

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    _toml = pytest.importorskip("tomli")


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPOSITORY_ROOT / ".github" / "workflows"
# Extra names start with a letter, which keeps regex classes such as ``\.[0-9]``
# inside workflow shell scripts from being mistaken for pip extras.
_EXTRAS_PATTERN = re.compile(r"\.\[([A-Za-z][A-Za-z0-9_\-]*(?:\s*,\s*[A-Za-z][A-Za-z0-9_\-]*)*)\]")
_PY_CLASSIFIER = re.compile(r"^Programming Language :: Python :: (3\.\d+)$")


def _pyproject() -> Dict[str, Any]:
    """Return the parsed repository ``pyproject.toml``.

    Returns
    -------
    dict
        The TOML document as nested dictionaries.

    Examples
    --------
    >>> _pyproject()["project"]["name"]
    'missingly'
    """
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        return _toml.load(handle)


def _workflow_text() -> str:
    """Return every GitHub Actions workflow concatenated as one string.

    Returns
    -------
    str
        The UTF-8 contents of all ``*.yml`` files under ``.github/workflows``.

    Examples
    --------
    >>> "pytest" in _workflow_text()
    True
    """
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(WORKFLOW_DIR.glob("*.yml"))
    )


def _extras_installed_by_workflows() -> Set[str]:
    """Return the optional-dependency names the workflows install.

    Returns
    -------
    set of str
        Every name used in an editable or regular ``.[extra,...]`` install.

    Examples
    --------
    >>> "test" in _extras_installed_by_workflows()
    True
    """
    names: Set[str] = set()
    for group in _EXTRAS_PATTERN.findall(_workflow_text()):
        names.update(part.strip() for part in group.split(",") if part.strip())
    return names


def test_every_extra_installed_by_ci_is_declared() -> None:
    """pip ignores unknown extras with a warning, so CI would lose its tools."""
    declared = set(_pyproject()["project"].get("optional-dependencies", {}))
    missing = sorted(_extras_installed_by_workflows() - declared)
    assert not missing, f"Workflows install undeclared extras: {missing}"


def test_test_extra_provides_the_tools_ci_invokes() -> None:
    """The ``test`` extra must install pytest and pytest-cov used by CI."""
    test_extra = " ".join(_pyproject()["project"]["optional-dependencies"]["test"])
    for requirement in ("pytest", "pytest-cov"):
        assert re.search(rf"(^|\s){re.escape(requirement)}\b", test_extra), requirement


def test_dynamic_version_has_a_source() -> None:
    """A dynamic version without a source makes every build fail."""
    data = _pyproject()
    assert "version" in data["project"].get("dynamic", [])
    source = data.get("tool", {}).get("setuptools", {}).get("dynamic", {}).get("version", {})
    assert source.get("attr") == "missingly._version.__version__"


def test_python_classifiers_match_the_ci_test_matrix() -> None:
    """Advertised Python versions must be exactly the versions CI tests."""
    classifiers = _pyproject()["project"]["classifiers"]
    advertised = {m.group(1) for c in classifiers if (m := _PY_CLASSIFIER.match(c))}
    matrix = re.search(r"python-version:\s*\[([^\]]+)\]", _workflow_text())
    assert matrix is not None, "CI test matrix not found"
    tested = {v.strip().strip("'\"") for v in matrix.group(1).split(",")}
    assert advertised == tested


def test_lint_and_coverage_configuration_is_present() -> None:
    """Ruff rule selection and coverage scope are part of the quality gate."""
    tool = _pyproject().get("tool", {})
    assert tool.get("ruff", {}).get("lint", {}).get("select"), "ruff lint rules missing"
    assert tool.get("coverage", {}).get("run", {}).get("source") == ["missingly"]


def test_license_expression_is_not_duplicated_by_a_license_classifier() -> None:
    """PEP 639 forbids a License classifier alongside a license expression."""
    project = _pyproject()["project"]
    assert isinstance(project.get("license"), str)
    assert not [c for c in project["classifiers"] if c.startswith("License ::")]

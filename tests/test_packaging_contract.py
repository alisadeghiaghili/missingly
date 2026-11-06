"""Regression contracts between ``pyproject.toml`` and the CI workflows.

A metadata-only edit once rewrote ``pyproject.toml`` and silently dropped the
optional extras, the dynamic version source and the tool configuration that the
workflows depend on.  pip only warns about an unknown extra, so the failure
surfaced later as missing test tools.  These tests make that drift fail fast.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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
_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.\-]+)\s*>=\s*([0-9][0-9A-Za-z.]*)\s*$")
_LICENCE_FILES = ("LICENSE", "NOTICE", "XeroGraph_LICENSE")
# Floors the code needs: DataFrame.map (pandas 2.1), RandomForest with NaN
# features (scikit-learn 1.4; ``encoded_missing_value`` alone needs 1.1).
_REQUIRED_MINIMUMS = {"pandas": (2, 1), "scikit-learn": (1, 4)}
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


def _floors(lines: List[str]) -> Dict[str, str]:
    """Map each ``name>=floor`` requirement line to ``{name: floor}``.

    Parameters
    ----------
    lines : list of str
        Requirement lines; blank lines and comments are ignored.

    Returns
    -------
    dict
        Lower-cased distribution name to its minimum version string.

    Raises
    ------
    AssertionError
        If a line is not a plain ``name>=version`` requirement.

    Examples
    --------
    >>> _floors(["pandas>=2.1.0", "# note"])
    {'pandas': '2.1.0'}
    """
    floors: Dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _REQUIREMENT.match(line)
        assert match, f"unsupported requirement line: {line!r}"
        floors[match.group(1).lower()] = match.group(2)
    return floors


def _requirements_lines() -> List[str]:
    """Return the lines of the repository ``requirements.txt``.

    Returns
    -------
    list of str
        File lines without line terminators.

    Examples
    --------
    >>> any("pandas" in line for line in _requirements_lines())
    True
    """
    path = REPOSITORY_ROOT / "requirements.txt"
    return path.read_text(encoding="utf-8").splitlines()


def _version_tuple(version: str) -> Tuple[int, ...]:
    """Convert ``"2.1.0"`` to ``(2, 1, 0)`` for comparison.

    Parameters
    ----------
    version : str
        Dotted numeric version.

    Returns
    -------
    tuple of int
        The numeric components.

    Examples
    --------
    >>> _version_tuple("2.1.0")
    (2, 1, 0)
    """
    return tuple(int(part) for part in version.split("."))


def test_requirements_floors_equal_pyproject_floors() -> None:
    """requirements.txt must not drift from the declared dependency floors."""
    declared = _floors(_pyproject()["project"]["dependencies"])
    assert _floors(_requirements_lines()) == declared


def test_requirements_do_not_advertise_undefined_extras() -> None:
    """Install hints in requirements.txt may only name extras that exist."""
    declared = set(_pyproject()["project"].get("optional-dependencies", {}))
    text = " ".join(_requirements_lines())
    hinted = set(re.findall(r"missingly\[([A-Za-z0-9_\-,\s]+)\]", text))
    names = {part.strip() for group in hinted for part in group.split(",")}
    assert names <= declared, f"undefined extras advertised: {sorted(names - declared)}"


@pytest.mark.parametrize("name", sorted(_REQUIRED_MINIMUMS))
def test_dependency_floor_covers_the_apis_the_code_uses(name: str) -> None:
    """Floors must not sit below the release that introduced a used API."""
    floors = _floors(_pyproject()["project"]["dependencies"])
    needed = _REQUIRED_MINIMUMS[name]
    assert _version_tuple(floors[name])[:2] >= needed


def test_compare_module_comment_states_the_pandas_floor() -> None:
    """The documented pandas minimum must equal the declared floor."""
    floors = _floors(_pyproject()["project"]["dependencies"])
    pandas_floor = ".".join(floors["pandas"].split(".")[:2])
    source = (REPOSITORY_ROOT / "missingly" / "compare.py").read_text(encoding="utf-8")
    assert f"pandas {pandas_floor}+" in source


def test_licence_files_are_declared_for_distribution() -> None:
    """All three licence documents are listed in ``license-files``."""
    declared = _pyproject()["project"]["license-files"]
    assert sorted(declared) == sorted(_LICENCE_FILES)


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    """Build a wheel and an sdist once for the content checks below.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Provides the output directory.

    Returns
    -------
    dict
        ``{"wheel": path, "sdist": path}`` of the freshly built artefacts.
    """
    outdir = tmp_path_factory.mktemp("dist")
    subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(outdir)],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return {"wheel": next(outdir.glob("*.whl")), "sdist": next(outdir.glob("*.tar.gz"))}


def _wheel_names(wheel: Path) -> List[str]:
    """List the member names of a wheel."""
    with zipfile.ZipFile(wheel) as archive:
        return archive.namelist()


def _sdist_names(sdist: Path) -> List[str]:
    """List the member names of an sdist."""
    with tarfile.open(sdist) as archive:
        return archive.getnames()


@pytest.mark.parametrize("licence", _LICENCE_FILES)
def test_wheel_and_sdist_contain_licence_files(
    built_distributions: Dict[str, Path], licence: str
) -> None:
    """Licence and attribution texts must travel with every distribution."""
    wheel = _wheel_names(built_distributions["wheel"])
    sdist = _sdist_names(built_distributions["sdist"])
    assert any(n.endswith(f".dist-info/licenses/{licence}") for n in wheel)
    assert any(n.endswith(f"/{licence}") for n in sdist)


def test_sdist_does_not_ship_a_partial_test_tree(
    built_distributions: Dict[str, Path],
) -> None:
    """A tests/ tree without conftest and fixtures cannot run, so none ships."""
    names = _sdist_names(built_distributions["sdist"])
    assert not [n for n in names if "/tests/" in n or n.endswith("/tests")]

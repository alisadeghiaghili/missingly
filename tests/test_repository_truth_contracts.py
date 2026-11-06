"""Regression contracts for repository-maintained public guidance.

These tests deliberately validate documentation and packaging facts that are
easy to drift away from the executable public API.  They do not replace API
tests; they make those human-facing claims mechanically reviewable.
"""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    """Return a UTF-8 repository text file for a truth-contract test.

    Parameters
    ----------
    relative_path : str
        Path relative to the repository root.

    Returns
    -------
    str
        The complete UTF-8 file contents.

    Examples
    --------
    >>> "missingly" in _read("README.md")
    True
    """
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def _coverage_gate() -> int:
    """Return the enforced coverage threshold from ``pyproject.toml``.

    Returns
    -------
    int
        The ``[tool.coverage.report] fail_under`` value.

    Examples
    --------
    >>> _coverage_gate() > 0
    True
    """
    from tests.test_packaging_contract import _pyproject

    pyproject = _pyproject()
    assert pyproject["tool"]["coverage"]["run"]["branch"] is True
    return int(pyproject["tool"]["coverage"]["report"]["fail_under"])


def test_contributing_guide_states_the_runtime_and_documentation_contract() -> None:
    """Keep contributor policy aligned with packaged Python and docstring rules."""
    contributing = _read("CONTRIBUTING.md")
    gate = _coverage_gate()

    assert "**Target Python**: 3.9+" in contributing
    assert "Parameters, Returns, Examples" in contributing
    assert f"currently {gate}% line and branch coverage" in contributing
    assert "`fail_under`" in contributing
    assert "`pyproject.toml`" in contributing


def test_contributing_guide_only_names_supported_accessor_and_imputation_apis() -> None:
    """Reject stale API names in contributor-facing examples."""
    guidance = _read("CONTRIBUTING.md") + _read("README.md")

    for unsupported in (
        "df.miss.summary()",
        "df.miss.report()",
        "df.miss.compare(df2)",
        "impute_simple",
        "impute_iterative",
        "MultipleImputer",
        "MIWorkflow",
        "missing_rate=",
        "MissinglyImputer(method=",
        "test_mcar",
        "test_mar",
        "missing_pattern",
        "missing_mechanism",
        "compare_datasets",
        "compare_groups",
        "MissinglyReport",
        "TimeSeriesMissingnessAnalyzer",
    ):
        assert unsupported not in guidance

    contributing = _read("CONTRIBUTING.md")
    assert 'df.miss.impute(strategy="mean")' in contributing
    assert "frac=0.2" in contributing
    assert "from missingly.report import create_report" in contributing
    assert (
        "from missingly.timeseries import gap_table, miss_ts_summary, vis_ts_miss, impute_ts"
        in contributing
    )


def test_contributing_guide_only_documents_real_domain_exceptions_and_maturity() -> (
    None
):
    """Reject fictional exception names and unverified maturity claims."""
    guidance = _read("CONTRIBUTING.md") + _read("CONVENTIONS.md")

    for stale_claim in (
        "production-grade Python package",
        "InvalidMethodError",
        "ColumnNotFoundError",
        "MixedDtypeError",
        "requirements.txt          # pinned dependencies",
        "MCAR/MAR/MNAR tests, Little's test, etc.",
    ):
        assert stale_claim not in guidance

    contributing = _read("CONTRIBUTING.md")
    for public_exception in (
        "MissinglyError",
        "ImputationError",
        "InsufficientDataError",
        "InvalidStrategyError",
        "MissingColumnError",
        "ConfigurationError",
    ):
        assert f"`{public_exception}`" in contributing


def test_readme_time_series_example_matches_the_current_return_schema() -> None:
    """Keep the time-series quickstart executable against public functions."""
    readme = _read("README.md")

    assert "mi.miss_ts_summary(ts)" in readme
    for actual_column in ("n_miss", "pct_miss", "n_gaps", "mean_gap", "max_gap"):
        assert actual_column in readme
    for stale_column in ("mean_gap_len", "longest_gap_start", "longest_gap_end"):
        assert stale_column not in readme


def test_readme_does_not_advertise_mixed_knn_as_an_inductive_pipeline_mode() -> None:
    """Prevent an unsupported train/test Gower example from returning."""
    readme = _read("README.md")

    assert 'MissinglyImputer(strategy="knn", metric="mixed"' not in readme
    assert "rejected by `MissinglyImputer`" in readme


def test_legacy_requirements_share_the_packaging_dependency_floors() -> None:
    """Keep direct requirements installs consistent with project metadata."""
    from tests.test_packaging_contract import _floors, _pyproject, _requirements_lines

    requirements = _requirements_lines()
    declared = _floors(_pyproject()["project"]["dependencies"])

    assert declared
    assert _floors(requirements) == declared
    assert all("missingly[" not in line for line in requirements)


def test_translated_readmes_match_time_series_and_mixed_knn_contracts() -> None:
    """Keep localized quickstarts aligned with the executable public API."""
    german = _read("README_DE.md")
    persian = _read("README_FA.md")

    for readme in (german, persian):
        assert "miss_ts_summary(ts)" in readme
        assert "miss_ts_summary(ts, col=" not in readme
        assert 'metric="mixed"' in readme

    assert "Python-Version:** 3.9" in german
    assert 'lehnt `metric="mixed"` ab' in german
    assert "۳.۹ و بالاتر" in persian
    assert "رد می‌کند" in persian


def test_conventions_and_contributing_publish_current_verified_status() -> None:
    """Prevent stale status text from overriding the enforced source of truth."""
    contributing = _read("CONTRIBUTING.md")
    conventions = _read("CONVENTIONS.md")
    gate = _coverage_gate()
    stated_gate = f"currently {gate}% line and branch coverage"

    assert "Target Python 3.9+" in contributing
    assert "NumPy-style docstrings" in contributing
    assert stated_gate in contributing
    assert stated_gate in conventions
    for document in ("README.md", "README_DE.md", "README_FA.md", "CHANGELOG.md"):
        assert "92.50%" not in _read(document)
    for document in (contributing, conventions):
        assert "92.50%" not in document
        assert "3846/4158" not in document
        assert "enforcement remains 80%" not in document
        assert "threshold remains 80%" not in document
    assert "## [2026-08-15] Current verified status" in conventions
    assert "### Current active issue map" in conventions
    for issue in (
        "#69",
        "#60",
        "#61",
        "#71",
        "#40",
        "#78",
        "#79",
        "#81",
        "#83",
        "#84",
    ):
        assert issue in conventions


def test_removed_working_notes_are_gone_and_unreferenced() -> None:
    """Keep local working notes out of the repository and out of references."""
    root_markdown = {path.name for path in REPOSITORY_ROOT.glob("*.md")}
    assert not any(name.endswith("Visualiz.md") for name in root_markdown)
    for removed in (
        "SKILL_MISSINGLY.md",
        "instructions.md",
        "missingly-dev-skills.md",
        "BRANCH_INTEGRATION.md",
    ):
        assert removed not in root_markdown
        for document in (
            "README.md",
            "CONTRIBUTING.md",
            "CONVENTIONS.md",
            "TESTING.md",
        ):
            assert removed not in _read(document)


def test_repository_declares_a_cross_platform_text_normalization_policy() -> None:
    """Require Git attributes so generated artifacts cannot silently rewrite text."""
    attributes = REPOSITORY_ROOT / ".gitattributes"

    assert attributes.is_file()
    assert "* text=auto eol=lf" in attributes.read_text(encoding="utf-8")


def test_missingly_owns_no_dqt_integration_gate() -> None:
    """Keep the sister-package dependency and CI responsibility one-way."""
    workflow = _read(".github/workflows/ci.yml")

    assert "dqt-integration:" not in workflow
    assert "data_quality_toolkit" not in workflow
    assert "tests/test_dqt_integration.py" not in workflow


def test_readmes_document_plotly_as_an_explicit_optional_dependency() -> None:
    """Reject silent-fallback and dependency-free Plotly guidance."""
    english = _read("README.md")
    persian = _read("README_FA.md")

    assert 'pip install "missingly[interactive]"' in english
    assert "silently falls back" not in english
    assert "Delegates to data_quality_toolkit" not in english
    assert "with no extra dependencies" not in english
    assert "بدون هیچ وابستگی اضافه‌ای" not in persian

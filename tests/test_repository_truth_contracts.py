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


def test_skill_describes_the_supported_runtime_and_documentation_contract() -> None:
    """Keep the AI context aligned with packaged Python and docstring policy."""
    skill = _read("SKILL_MISSINGLY.md")

    assert "**Target Python**: 3.9+" in skill
    assert "Parameters, Returns, Examples" in skill
    assert "**89.64% (3589/4004)**" in skill
    assert "coverage-py312" in skill


def test_skill_only_names_supported_accessor_and_imputation_apis() -> None:
    """Reject stale API names that would cause generated user code to fail."""
    skill = _read("SKILL_MISSINGLY.md")

    for unsupported in (
        "df.miss.summary()",
        "df.miss.report()",
        "df.miss.compare(df2)",
        "impute_simple",
        "impute_iterative",
        "MultipleImputer",
        "MIWorkflow",
        "missing_rate=",
    ):
        assert unsupported not in skill

    assert "df.miss.impute(strategy=\"mean\")" in skill
    assert "frac=0.2" in skill


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
    assert "rejected by MissinglyImputer" in readme


def test_legacy_requirements_share_the_packaging_dependency_floors() -> None:
    """Keep direct requirements installs consistent with project metadata."""
    requirements = _read("requirements.txt")

    for requirement in (
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "statsmodels>=0.14.0",
        "jinja2>=3.1.0",
        "upsetplot>=0.9.0",
    ):
        assert requirement in requirements


def test_repository_declares_a_cross_platform_text_normalization_policy() -> None:
    """Require Git attributes so generated artifacts cannot silently rewrite text."""
    attributes = REPOSITORY_ROOT / ".gitattributes"

    assert attributes.is_file()
    assert "* text=auto eol=lf" in attributes.read_text(encoding="utf-8")

"""Independent numerical and failure-contract tests for Little's MCAR test."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from missingly.benchmark import BenchmarkManifest
from missingly.diagnostics import mcar_test


_FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "little_mcar"
_AIRQUALITY = _FIXTURE_DIR / "airquality.csv"
_MANIFEST = _FIXTURE_DIR / "benchmark_manifest.json"


@pytest.fixture(scope="module")
def airquality() -> pd.DataFrame:
    """Load the vendored R ``datasets::airquality`` oracle data set."""
    return pd.read_csv(_AIRQUALITY).drop(columns="rownames")


@pytest.fixture(scope="module")
def benchmark_manifest() -> BenchmarkManifest:
    """Load the machine-readable evidence contract for the frozen oracle."""
    return BenchmarkManifest.from_dict(json.loads(_MANIFEST.read_text(encoding="utf-8")))


def test_airquality_fixture_hash_matches_its_benchmark_manifest(benchmark_manifest):
    """Prevent a silent data change from invalidating the recorded oracle metrics."""
    assert sha256(_AIRQUALITY.read_bytes()).hexdigest() == benchmark_manifest.dataset_sha256


def test_imported_oracle_manifest_is_explicit_about_its_provenance_gap(benchmark_manifest):
    """Do not silently present an unversioned imported regression value as parity proof."""
    assert "not published" in benchmark_manifest.reference_tool_version
    assert "not published" in benchmark_manifest.reference_platform
    assert any("before claiming cross-tool parity" in item for item in benchmark_manifest.known_differences)


def test_little_mcar_matches_frozen_naniar_oracle(airquality, benchmark_manifest):
    """Match ``naniar::mcar_test(airquality)`` within numerical EM tolerance."""
    result = mcar_test(airquality)

    expected = benchmark_manifest.expected_metrics
    tolerances = benchmark_manifest.tolerances
    assert result["missing_patterns"] == pytest.approx(
        expected["missing_patterns"], abs=tolerances["missing_patterns"]
    )
    assert result["df"] == pytest.approx(
        expected["degrees_of_freedom"], abs=tolerances["degrees_of_freedom"]
    )
    assert result["chi_square"] == pytest.approx(
        expected["chi_square"], abs=tolerances["chi_square"]
    )
    assert result["p_value"] == pytest.approx(
        expected["p_value"], abs=tolerances["p_value"]
    )
    assert result["em_iterations"] > 0


def test_all_missing_pattern_does_not_change_little_statistic(airquality):
    """An all-missing row contributes no observed mean contrast to the statistic."""
    with_all_missing = pd.concat(
        [airquality, pd.DataFrame([{column: np.nan for column in airquality.columns}])],
        ignore_index=True,
    )

    result = mcar_test(with_all_missing)

    assert result["missing_patterns"] == 5
    assert result["chi_square"] == pytest.approx(35.1061229683641, abs=1e-4)


def test_rejects_all_missing_column():
    """Fail loudly when a Gaussian mean parameter is unidentified."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [np.nan] * 4})

    with pytest.raises(ValueError, match="entirely missing"):
        mcar_test(frame)


def test_rejects_high_dimensional_input_without_enough_information():
    """Do not manufacture a chi-square result when n is not larger than p."""
    frame = pd.DataFrame(
        [[1.0, np.nan, 2.0], [2.0, 3.0, np.nan], [np.nan, 1.0, 2.0]]
    )

    with pytest.raises(ValueError, match="more informative rows"):
        mcar_test(frame)


def test_nonconvergent_em_raises_actionable_error(airquality):
    """Never return a statistic from an unconverged EM fit."""
    with pytest.raises(RuntimeError, match="did not converge"):
        mcar_test(airquality, max_iter=1)


def test_rejects_non_numeric_and_infinite_observations():
    """Require a finite numeric matrix rather than coercing arbitrary values."""
    non_numeric = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": ["a", "b", "c"]})
    infinite = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0], "y": [1.0, np.inf, 3.0, 4.0]})

    with pytest.raises(ValueError, match="numeric columns"):
        mcar_test(non_numeric)
    with pytest.raises(ValueError, match="infinite"):
        mcar_test(infinite)

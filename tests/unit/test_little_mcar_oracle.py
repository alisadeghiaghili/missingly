"""Independent numerical and failure-contract tests for Little's MCAR test."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from missingly.diagnostics import mcar_test


_FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "little_mcar"
_AIRQUALITY = _FIXTURE_DIR / "airquality.csv"
_AIRQUALITY_SHA256 = "65d2c4afd976c169af9bb0bd97e9e78e1e8a185f1b52e2e3153e30f90c7fb5f8"


@pytest.fixture(scope="module")
def airquality() -> pd.DataFrame:
    """Load the vendored R ``datasets::airquality`` oracle data set."""
    return pd.read_csv(_AIRQUALITY).drop(columns="rownames")


def test_airquality_fixture_hash_is_frozen():
    """Prevent a silent fixture change from invalidating the R oracle values."""
    assert sha256(_AIRQUALITY.read_bytes()).hexdigest() == _AIRQUALITY_SHA256


def test_little_mcar_matches_frozen_naniar_oracle(airquality):
    """Match ``naniar::mcar_test(airquality)`` within numerical EM tolerance."""
    result = mcar_test(airquality)

    assert result["missing_patterns"] == 4
    assert result["df"] == 14
    assert result["chi_square"] == pytest.approx(35.1061288689702, abs=1e-4)
    assert result["p_value"] == pytest.approx(0.00141778113856683, abs=1e-7)
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

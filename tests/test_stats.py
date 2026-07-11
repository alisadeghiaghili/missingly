"""Tests for statistical tests (migrated from missingly.stats -> missingly.diagnostics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.diagnostics import mcar_test, mar_mnar_test, diagnose_missing


@pytest.fixture()
def numeric_df():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "a": rng.standard_normal(100),
            "b": rng.standard_normal(100),
            "c": rng.standard_normal(100),
        }
    )
    mask = rng.random((100, 3)) < 0.2
    df[mask] = np.nan
    return df


@pytest.fixture()
def no_missing_df():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


# ---------------------------------------------------------------------------
# mcar_test
# ---------------------------------------------------------------------------

class TestMcarTest:
    def test_returns_dict(self, numeric_df):
        result = mcar_test(numeric_df)
        assert isinstance(result, dict)

    def test_has_required_keys(self, numeric_df):
        result = mcar_test(numeric_df)
        for key in ("statistic", "df", "p_value"):
            assert key in result

    def test_p_value_range(self, numeric_df):
        result = mcar_test(numeric_df)
        assert 0.0 <= result["p_value"] <= 1.0 or np.isnan(result["p_value"])

    def test_no_missing_raises_or_nan(self, no_missing_df):
        """With no missing data, mcar_test should raise or return NaN p-value."""
        try:
            result = mcar_test(no_missing_df)
            assert np.isnan(result["p_value"])
        except (ValueError, TypeError):
            pass


# ---------------------------------------------------------------------------
# mar_mnar_test
# ---------------------------------------------------------------------------

class TestMarMnarTest:
    def test_returns_dataframe(self, numeric_df):
        result = mar_mnar_test(numeric_df)
        assert isinstance(result, pd.DataFrame)

    def test_no_missing_empty_or_raises(self, no_missing_df):
        try:
            result = mar_mnar_test(no_missing_df)
            assert len(result) == 0
        except (ValueError, TypeError):
            pass


# ---------------------------------------------------------------------------
# diagnose_missing
# ---------------------------------------------------------------------------

class TestDiagnoseMissing:
    def test_returns_dict(self, numeric_df):
        result = diagnose_missing(numeric_df)
        assert isinstance(result, dict)

    def test_has_mcar_key(self, numeric_df):
        result = diagnose_missing(numeric_df)
        assert "mcar" in result

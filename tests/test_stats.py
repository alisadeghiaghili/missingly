"""Tests for statistical tests (migrated from missingly.stats -> missingly.diagnostics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.diagnostics import mcar_test, mar_mnar_test, diagnose_missing
import missingly



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

def test_mcar_test_amount_missing_shape(numeric_df_with_missing):
    """amount_missing must be a (2, n_cols) DataFrame."""
    result = mcar_test(numeric_df_with_missing)
    am = result['amount_missing']
    assert isinstance(am, pd.DataFrame)
    assert am.shape == (2, numeric_df_with_missing.shape[1])


def test_mcar_test_raises_on_all_complete(no_missing_df):
    """mcar_test raises ValueError when DataFrame has no missing values."""
    with pytest.raises(ValueError, match="at least one missing"):
        mcar_test(no_missing_df)


def test_mcar_test_sentinel(numeric_df_with_missing):
    """mcar_test handles sentinel missing_values without raising."""
    df = numeric_df_with_missing.fillna(-99)
    result = mcar_test(df, missing_values=[-99])
    assert isinstance(result, dict)


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

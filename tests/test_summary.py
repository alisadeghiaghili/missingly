"""Tests for missingly summary functions.

Covers correctness of all summary statistics and the critical
bind_shadow NaN + sentinel bug fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.summary import (
    bind_shadow,
    n_miss,
    n_complete,
    pct_miss,
    pct_complete,
    miss_var_summary,
    miss_case_summary,
)


@pytest.fixture
def df():
    """Small DataFrame with NaN and a sentinel value."""
    return pd.DataFrame({
        "A": [1.0, np.nan, -99.0],
        "B": [np.nan, 2.0, 3.0],
    })


# ---------------------------------------------------------------------------
# bind_shadow
# ---------------------------------------------------------------------------

def test_bind_shadow_doubles_columns(df):
    """bind_shadow returns a DataFrame with 2x the columns."""
    result = bind_shadow(df)
    assert result.shape[1] == df.shape[1] * 2


def test_bind_shadow_suffix(df):
    """Shadow columns are named with _NA suffix."""
    result = bind_shadow(df)
    assert "A_NA" in result.columns
    assert "B_NA" in result.columns


def test_bind_shadow_nan_always_detected(df):
    """NaN values are detected even when missing_values is provided."""
    # Without missing_values: NaN in A[1] and B[0] should be True
    result = bind_shadow(df)
    assert result.loc[1, "A_NA"] == True   # A[1] is NaN
    assert result.loc[0, "B_NA"] == True   # B[0] is NaN


def test_bind_shadow_sentinel_also_detected(df):
    """Sentinel values in missing_values are flagged as missing."""
    result = bind_shadow(df, missing_values=[-99])
    # A[2] = -99.0 should be flagged
    assert result.loc[2, "A_NA"] == True


def test_bind_shadow_nan_plus_sentinel(df):
    """Both NaN and sentinels are flagged when missing_values is given."""
    result = bind_shadow(df, missing_values=[-99])
    # A[1] = NaN → still True
    assert result.loc[1, "A_NA"] == True
    # A[2] = -99 → True
    assert result.loc[2, "A_NA"] == True
    # A[0] = 1.0 → False
    assert result.loc[0, "A_NA"] == False


def test_bind_shadow_does_not_mutate(df):
    """bind_shadow does not mutate the original DataFrame."""
    original = df.copy()
    bind_shadow(df, missing_values=[-99])
    pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# n_miss / n_complete
# ---------------------------------------------------------------------------

def test_n_miss_basic(df):
    """n_miss counts NaN values correctly."""
    assert n_miss(df) == 2  # A[1] and B[0]


def test_n_miss_with_sentinel(df):
    """n_miss counts sentinels when missing_values is provided."""
    assert n_miss(df, missing_values=[-99]) == 3  # A[1], B[0], A[2]


def test_n_complete_basic(df):
    """n_complete + n_miss == total cells."""
    assert n_miss(df) + n_complete(df) == df.size


# ---------------------------------------------------------------------------
# pct_miss / pct_complete
# ---------------------------------------------------------------------------

def test_pct_miss_range(df):
    """pct_miss is between 0 and 100."""
    assert 0.0 <= pct_miss(df) <= 100.0


def test_pct_complete_complement(df):
    """pct_miss and pct_complete sum to 100."""
    assert abs(pct_miss(df) + pct_complete(df) - 100.0) < 1e-9


def test_pct_miss_empty_df():
    """pct_miss returns 0 for an empty DataFrame."""
    assert pct_miss(pd.DataFrame()) == 0.0


# ---------------------------------------------------------------------------
# miss_var_summary
# ---------------------------------------------------------------------------

def test_miss_var_summary_shape(df):
    """miss_var_summary has one row per column."""
    result = miss_var_summary(df)
    assert len(result) == len(df.columns)


def test_miss_var_summary_values(df):
    """miss_var_summary n_miss column matches expected counts."""
    result = miss_var_summary(df)
    a_row = result[result["variable"] == "A"].iloc[0]
    assert a_row["n_miss"] == 1  # only A[1] is NaN


def test_miss_var_summary_sentinel(df):
    """miss_var_summary counts sentinels when missing_values is provided."""
    result = miss_var_summary(df, missing_values=[-99])
    a_row = result[result["variable"] == "A"].iloc[0]
    assert a_row["n_miss"] == 2  # A[1]=NaN and A[2]=-99


# ---------------------------------------------------------------------------
# miss_case_summary
# ---------------------------------------------------------------------------

def test_miss_case_summary_shape(df):
    """miss_case_summary has one row per DataFrame row."""
    result = miss_case_summary(df)
    assert len(result) == len(df)


def test_miss_case_summary_pct_range(df):
    """miss_case_summary pct_miss is between 0 and 100."""
    result = miss_case_summary(df)
    assert (result["pct_miss"] >= 0).all()
    assert (result["pct_miss"] <= 100).all()

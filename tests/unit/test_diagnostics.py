"""Unit tests for missingly.diagnostics."""

import numpy as np
import pandas as pd
import pytest

from missingly.diagnostics import (
    bind_shadow,
    diagnose_missing,
    mar_mnar_test,
    mcar_test,
    miss_case_summary,
    miss_scan_count,
    miss_summary,
    miss_var_summary,
    n_complete,
    n_miss,
    pct_complete,
    pct_miss,
    summarise,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_simple():
    return pd.DataFrame({
        "a": [1.0, np.nan, 3.0, 4.0],
        "b": [np.nan, 2.0, np.nan, 4.0],
    })


@pytest.fixture
def df_no_missing():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


@pytest.fixture
def df_all_missing():
    return pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})


@pytest.fixture
def df_mixed():
    return pd.DataFrame({
        "age": [25.0, np.nan, 35.0, 40.0, 50.0],
        "city": ["Berlin", "Munich", np.nan, "Berlin", "Hamburg"],
    })


# ---------------------------------------------------------------------------
# n_miss / n_complete
# ---------------------------------------------------------------------------

class TestNMissNComplete:
    def test_n_miss_basic(self, df_simple):
        assert n_miss(df_simple) == 3

    def test_n_miss_no_missing(self, df_no_missing):
        assert n_miss(df_no_missing) == 0

    def test_n_miss_all_missing(self, df_all_missing):
        assert n_miss(df_all_missing) == 4

    def test_n_complete_basic(self, df_simple):
        assert n_complete(df_simple) == 5  # 8 cells - 3 missing

    def test_n_miss_sentinel(self):
        df = pd.DataFrame({"a": [-99, 1, -99]})
        assert n_miss(df, missing_values=[-99]) == 2

    def test_n_miss_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            n_miss([1, None, 3])


# ---------------------------------------------------------------------------
# pct_miss / pct_complete
# ---------------------------------------------------------------------------

class TestPctMissPctComplete:
    def test_pct_miss_range(self, df_simple):
        p = pct_miss(df_simple)
        assert 0.0 <= p <= 100.0

    def test_pct_miss_all_missing(self, df_all_missing):
        assert pct_miss(df_all_missing) == pytest.approx(100.0)

    def test_pct_complete_complements_miss(self, df_simple):
        assert pct_miss(df_simple) + pct_complete(df_simple) == pytest.approx(100.0)

    def test_empty_df_returns_zero(self):
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        assert pct_miss(df) == 0.0


# ---------------------------------------------------------------------------
# bind_shadow
# ---------------------------------------------------------------------------

class TestBindShadow:
    def test_doubles_columns(self, df_simple):
        result = bind_shadow(df_simple)
        assert result.shape[1] == df_simple.shape[1] * 2

    def test_shadow_column_names(self, df_simple):
        result = bind_shadow(df_simple)
        assert "a_NA" in result.columns
        assert "b_NA" in result.columns

    def test_shadow_values_correct(self, df_simple):
        result = bind_shadow(df_simple)
        assert result.loc[1, "a_NA"] is True or result.loc[1, "a_NA"] == True
        assert result.loc[0, "b_NA"] is True or result.loc[0, "b_NA"] == True

    def test_sentinel_values_marked(self):
        df = pd.DataFrame({"a": [-99, 1, 2]})
        result = bind_shadow(df, missing_values=[-99])
        assert result.loc[0, "a_NA"] == True

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            bind_shadow([1, 2, 3])


# ---------------------------------------------------------------------------
# miss_var_summary
# ---------------------------------------------------------------------------

class TestMissVarSummary:
    def test_schema(self, df_simple):
        result = miss_var_summary(df_simple)
        assert list(result.columns) == ["variable", "n_miss", "pct_miss"]

    def test_counts_correct(self, df_simple):
        result = miss_var_summary(df_simple)
        row_a = result[result["variable"] == "a"].iloc[0]
        assert row_a["n_miss"] == 1
        row_b = result[result["variable"] == "b"].iloc[0]
        assert row_b["n_miss"] == 2

    def test_no_missing_returns_zeros(self, df_no_missing):
        result = miss_var_summary(df_no_missing)
        assert result["n_miss"].sum() == 0


# ---------------------------------------------------------------------------
# miss_case_summary
# ---------------------------------------------------------------------------

class TestMissCaseSummary:
    def test_schema(self, df_simple):
        result = miss_case_summary(df_simple)
        assert list(result.columns) == ["case", "n_miss", "pct_miss"]

    def test_row_count_matches_input(self, df_simple):
        result = miss_case_summary(df_simple)
        assert len(result) == len(df_simple)


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------

class TestSummarise:
    def test_single_row(self, df_simple):
        result = summarise(df_simple)
        assert len(result) == 1

    def test_columns(self, df_simple):
        result = summarise(df_simple)
        for col in ("n_rows", "n_cols", "n_cells", "n_miss", "n_complete",
                    "pct_miss", "pct_complete"):
            assert col in result.columns

    def test_values_consistent(self, df_simple):
        result = summarise(df_simple)
        assert result.loc[0, "n_miss"] + result.loc[0, "n_complete"] == result.loc[0, "n_cells"]


# ---------------------------------------------------------------------------
# miss_scan_count
# ---------------------------------------------------------------------------

class TestMissScanCount:
    def test_finds_sentinel(self):
        df = pd.DataFrame({"a": [-99, 1, -99], "b": [2, -99, 3]})
        result = miss_scan_count(df, search=[-99])
        assert len(result) == 2
        assert result[result["variable"] == "a"]["n"].values[0] == 2

    def test_empty_when_not_found(self, df_no_missing):
        result = miss_scan_count(df_no_missing, search=[-99])
        assert len(result) == 0

    def test_detects_nan_as_sentinel(self):
        df = pd.DataFrame({"a": [np.nan, 1.0, np.nan]})
        result = miss_scan_count(df, search=[float("nan")])
        assert result.loc[0, "n"] == 2


# ---------------------------------------------------------------------------
# miss_summary
# ---------------------------------------------------------------------------

class TestMissSummary:
    def test_keys(self, df_simple):
        result = miss_summary(df_simple)
        assert set(result.keys()) == {"summary", "by_variable", "by_case"}

    def test_rounded(self, df_simple):
        result = miss_summary(df_simple, digits=1)
        pct = result["by_variable"]["pct_miss"].values
        for v in pct:
            assert round(v, 1) == v


# ---------------------------------------------------------------------------
# mcar_test
# ---------------------------------------------------------------------------

class TestMcarTest:
    def test_returns_expected_keys(self, df_simple):
        result = mcar_test(df_simple)
        for key in ("chi_square", "df", "p_value", "missing_patterns", "amount_missing"):
            assert key in result

    def test_p_value_in_range(self, df_simple):
        result = mcar_test(df_simple)
        p = result["p_value"]
        if not np.isnan(p):
            assert 0.0 <= p <= 1.0

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            mcar_test([[1, None], [None, 2]])


# ---------------------------------------------------------------------------
# diagnose_missing
# ---------------------------------------------------------------------------

class TestDiagnose:
    def test_mechanism_valid_value(self, df_simple):
        result = diagnose_missing(df_simple)
        assert result["mechanism"] in (
            "MCAR", "MAR", "possible_MNAR", "insufficient_data"
        )

    def test_recommendation_is_string(self, df_simple):
        result = diagnose_missing(df_simple)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0

    def test_strategy_hint_is_string(self, df_simple):
        result = diagnose_missing(df_simple)
        assert isinstance(result["strategy_hint"], str)

    def test_high_missingness_cols_flagged(self):
        df = pd.DataFrame({
            "a": [1.0] + [np.nan] * 9,
            "b": [np.nan] + list(range(9)),
        })
        result = diagnose_missing(df)
        assert "a" in result["high_missingness_cols"]

    def test_insufficient_data_mechanism_for_single_col(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        result = diagnose_missing(df)
        assert result["mechanism"] == "insufficient_data"

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError):
            diagnose_missing("not a df")


# ---------------------------------------------------------------------------
# Backward-compat: stats.py and summary.py shims still work
# ---------------------------------------------------------------------------

def test_stats_shim_still_importable():
    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        from missingly.stats import mcar_test as _mcar  # noqa: F401
    assert callable(_mcar)


def test_summary_shim_still_importable():
    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        from missingly.summary import n_miss as _n_miss  # noqa: F401
    assert callable(_n_miss)

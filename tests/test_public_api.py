"""Tests for the v1 public API surface of missingly.

Checks:
- All v1 surface symbols exist in the missingly namespace and are callable.
- Legacy / experimental functions emit FutureWarning when called.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import missingly


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    """A small DataFrame with some missing values."""
    return pd.DataFrame({
        "a": [1.0, np.nan, 3.0, np.nan, 5.0],
        "b": [np.nan, 2.0, 3.0, 4.0, np.nan],
        "c": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


# ---------------------------------------------------------------------------
# v1 surface \u2014 existence + callable
# ---------------------------------------------------------------------------

V1_SYMBOLS = [
    # summary
    "bind_shadow", "n_miss", "n_complete", "pct_miss", "pct_complete",
    "miss_var_summary", "miss_case_summary",
    # diagnosis
    "mcar_test", "mar_mnar_test", "diagnose_missing",
    # visualisation core
    "matrix", "bar", "miss_case", "miss_var_pct", "vis_miss", "miss_which",
    "upset", "miss_patterns", "miss_cooccurrence", "heatmap", "miss_cluster",
    # report
    "create_report",
    # timeseries (v1 subset)
    "miss_ts_summary", "vis_ts_miss", "vis_miss_over_time", "impute_ts",
    # evaluation
    "compare_imputations", "cv_compare_imputations",
    # sklearn-style
    "MissinglyImputer", "FittedImputer", "make_imputer",
    # impute helpers
    "impute_mean", "impute_median", "impute_mode",
    "impute_knn", "impute_mice", "impute_rf", "impute_gb",
    # manipulation helpers
    "replace_with_na", "replace_with_na_all",
]


@pytest.mark.parametrize("symbol", V1_SYMBOLS)
def test_v1_symbol_exists_and_callable(symbol):
    """Every v1 surface symbol must exist in missingly and be callable."""
    obj = getattr(missingly, symbol, None)
    assert obj is not None, (
        f"missingly.{symbol} is missing from the package namespace"
    )
    assert callable(obj), (
        f"missingly.{symbol} exists but is not callable"
    )


def test_v1_symbols_in_all():
    """Core v1 symbols must appear in __all__."""
    for sym in V1_SYMBOLS:
        assert sym in missingly.__all__, (
            f"{sym} is a v1 symbol but is absent from missingly.__all__"
        )


# ---------------------------------------------------------------------------
# df.miss accessor
# ---------------------------------------------------------------------------

def test_miss_accessor_registered(simple_df):
    """df.miss accessor must be registered after importing missingly."""
    assert hasattr(simple_df, "miss"), "df.miss accessor not registered"


def test_miss_accessor_n_miss(simple_df):
    assert simple_df.miss.n_miss() == 4


def test_miss_accessor_summary(simple_df):
    result = simple_df.miss.miss_var_summary()
    assert isinstance(result, pd.DataFrame)
    assert "n_miss" in result.columns


# ---------------------------------------------------------------------------
# Legacy / experimental API \u2014 FutureWarning
# ---------------------------------------------------------------------------

LEGACY_SYMBOLS = [
    "simulate_mcar", "simulate_mar", "simulate_mnar", "simulate_mixed",
    "gap_table", "vis_gap_lengths",
    "clean_names", "remove_empty", "coalesce_columns", "miss_as_feature",
    "chunk_apply", "memory_usage_mb", "optimize_dtypes",
    "hotelling_test", "pattern_monotone_test", "missing_correlation_matrix",
]


@pytest.mark.parametrize("sym", LEGACY_SYMBOLS)
def test_legacy_symbol_exists(sym):
    """Legacy symbols must still exist in the namespace."""
    assert hasattr(missingly, sym), f"missingly.{sym} is missing"
    assert callable(getattr(missingly, sym))


def test_simulate_mcar_works():
    """simulate_mcar should work without FutureWarning (no longer deprecated)."""
    df_complete = pd.DataFrame({
        "x": list(range(1, 11, 1)),
        "y": list(range(10, 0, -1)),
    }, dtype=float)
    result = missingly.simulate_mcar(df_complete, frac=0.2, random_state=0)
    assert result is not None
    assert result.shape == df_complete.shape


def test_missing_correlation_matrix_emits_future_warning():
    """missing_correlation_matrix must emit FutureWarning (legacy API)."""
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]})
    with pytest.warns(FutureWarning, match="experimental / legacy"):
        missingly.missing_correlation_matrix(frame=df)


def test_pattern_monotone_test_emits_future_warning():
    """pattern_monotone_test must emit FutureWarning (legacy API)."""
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0, np.nan],
        "b": [np.nan, 2.0, np.nan, 4.0],
    })
    with pytest.warns(FutureWarning, match="experimental / legacy"):
        missingly.pattern_monotone_test(frame=df)


def test_miss_as_feature_emits_future_warning():
    """miss_as_feature must emit FutureWarning (legacy API)."""
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]})
    with pytest.warns(FutureWarning, match="experimental / legacy"):
        missingly.miss_as_feature(df=df)


# ---------------------------------------------------------------------------
# Backward-compat aliases (private — not in __all__)
# ---------------------------------------------------------------------------

def test_test_hotelling_alias_exists():
    """test_hotelling is a private alias for hotelling_test."""
    from missingly import _test_hotelling
    assert _test_hotelling is missingly.hotelling_test


def test_test_pattern_monotone_alias_exists():
    """test_pattern_monotone is a private alias for pattern_monotone_test."""
    from missingly import _test_pattern_monotone
    assert _test_pattern_monotone is missingly.pattern_monotone_test

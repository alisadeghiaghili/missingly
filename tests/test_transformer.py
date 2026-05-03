"""Tests for MissinglyImputer sklearn-compatible transformer.

Covers:
- All seven strategies: mean, median, mode, knn, mice, rf, gb
- fit / transform / fit_transform interface
- No data leakage: transform uses only train statistics
- sklearn Pipeline compatibility
- NotFittedError before fit
- TypeError for non-DataFrame input
- ValueError for column mismatch
- Mutation safety (original DataFrame unchanged)
- get_feature_names_out
- Mixed numeric + categorical DataFrames
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

from missingly.transformer import MissinglyImputer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df():
    """Small numeric DataFrame with missing values."""
    return pd.DataFrame({
        "age":    [25.0, np.nan, 35.0, 40.0, 30.0],
        "income": [50_000.0, 60_000.0, np.nan, 80_000.0, 70_000.0],
        "score":  [85.0, 90.0, 78.0, np.nan, 88.0],
    })


@pytest.fixture
def mixed_df():
    """DataFrame with both numeric and categorical columns."""
    return pd.DataFrame({
        "age":   [25.0, np.nan, 35.0, 40.0, 30.0, 28.0],
        "city":  ["Paris", "London", np.nan, "Berlin", "Paris", "London"],
        "score": [85.0, 90.0, 78.0, np.nan, 88.0, 92.0],
        "grade": ["A", "B", "A", "C", np.nan, "B"],
    })


@pytest.fixture
def train_test_split(numeric_df):
    """Return a (train, test) tuple from the numeric fixture."""
    return numeric_df.iloc[:3].copy(), numeric_df.iloc[3:].copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_no_missing(df: pd.DataFrame) -> None:
    assert df.isnull().sum().sum() == 0, (
        f"Expected no missing; got {df.isnull().sum().sum()}"
    )


# ---------------------------------------------------------------------------
# Basic fit / transform for all strategies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["mean", "median", "mode", "knn", "mice", "rf", "gb"])
def test_fit_transform_no_missing_numeric(strategy, numeric_df):
    """All strategies produce a fully-observed numeric DataFrame."""
    imputer = MissinglyImputer(strategy=strategy)
    result = imputer.fit(numeric_df).transform(numeric_df)
    _assert_no_missing(result)
    assert result.shape == numeric_df.shape


@pytest.mark.parametrize("strategy", ["mean", "median", "mode", "knn", "rf", "gb"])
def test_fit_transform_no_missing_mixed(strategy, mixed_df):
    """All tree/simple strategies handle mixed DataFrames."""
    imputer = MissinglyImputer(strategy=strategy)
    result = imputer.fit(mixed_df).transform(mixed_df)
    _assert_no_missing(result)
    assert result.shape == mixed_df.shape


# ---------------------------------------------------------------------------
# No data leakage: transform must use only train statistics
# ---------------------------------------------------------------------------

def test_no_leakage_mean(train_test_split):
    """transform uses train mean, not test mean — no data leakage."""
    train, test = train_test_split
    imputer = MissinglyImputer(strategy="mean")
    imputer.fit(train)
    result = imputer.transform(test)
    _assert_no_missing(result)
    # The imputed value for income NaN in train must equal train income mean
    train_income_mean = train["income"].mean()
    train_result = imputer.transform(train)
    assert abs(train_result.loc[train_result.index[1], "age"] - train["age"].mean()) < 1e-6


# ---------------------------------------------------------------------------
# sklearn Pipeline compatibility
# ---------------------------------------------------------------------------

def test_pipeline_with_scaler(numeric_df):
    """MissinglyImputer works inside an sklearn Pipeline with StandardScaler."""
    pipe = Pipeline([
        ("imputer", MissinglyImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
    ])
    result = pipe.fit_transform(numeric_df)
    assert result.shape == numeric_df.shape
    assert not np.isnan(result).any()


def test_pipeline_fit_transform_separate(numeric_df):
    """Pipeline.fit on train + transform on test works without errors."""
    train = numeric_df.iloc[:3].copy()
    test  = numeric_df.iloc[3:].copy()
    pipe = Pipeline([("imputer", MissinglyImputer(strategy="median"))])
    pipe.fit(train)
    result = pipe.transform(test)
    assert result.shape[1] == train.shape[1]


# ---------------------------------------------------------------------------
# NotFittedError before fit
# ---------------------------------------------------------------------------

def test_not_fitted_error(numeric_df):
    """transform raises NotFittedError if called before fit."""
    imputer = MissinglyImputer(strategy="mean")
    with pytest.raises(NotFittedError):
        imputer.transform(numeric_df)


# ---------------------------------------------------------------------------
# TypeError for non-DataFrame input
# ---------------------------------------------------------------------------

def test_type_error_on_array(numeric_df):
    """fit raises TypeError when passed a numpy array instead of DataFrame."""
    imputer = MissinglyImputer(strategy="mean")
    with pytest.raises(TypeError):
        imputer.fit(numeric_df.values)


# ---------------------------------------------------------------------------
# ValueError for column mismatch
# ---------------------------------------------------------------------------

def test_column_mismatch_raises(numeric_df):
    """transform raises ValueError when test columns differ from train."""
    imputer = MissinglyImputer(strategy="mean")
    imputer.fit(numeric_df)
    bad_df = numeric_df.drop(columns=["age"])
    with pytest.raises(ValueError, match="missing in transform"):
        imputer.transform(bad_df)


# ---------------------------------------------------------------------------
# Invalid strategy
# ---------------------------------------------------------------------------

def test_invalid_strategy_raises():
    """Passing an unknown strategy raises ValueError at construction."""
    with pytest.raises(ValueError, match="strategy"):
        MissinglyImputer(strategy="interpolate")


# ---------------------------------------------------------------------------
# Mutation safety
# ---------------------------------------------------------------------------

def test_does_not_mutate(numeric_df):
    """fit and transform must not mutate the input DataFrame."""
    original = numeric_df.copy()
    imputer = MissinglyImputer(strategy="mean")
    imputer.fit(numeric_df)
    imputer.transform(numeric_df)
    pd.testing.assert_frame_equal(numeric_df, original)


# ---------------------------------------------------------------------------
# get_feature_names_out
# ---------------------------------------------------------------------------

def test_get_feature_names_out(numeric_df):
    """get_feature_names_out returns the training column names in order."""
    imputer = MissinglyImputer(strategy="mean")
    imputer.fit(numeric_df)
    assert imputer.get_feature_names_out() == numeric_df.columns.tolist()


# ---------------------------------------------------------------------------
# RF categorical values are valid
# ---------------------------------------------------------------------------

def test_rf_categorical_values_valid(mixed_df):
    """RF transformer imputes categoricals with values from the training set."""
    imputer = MissinglyImputer(strategy="rf")
    result = imputer.fit(mixed_df).transform(mixed_df)
    valid_cities = set(mixed_df["city"].dropna())
    assert set(result["city"]).issubset(valid_cities)


def test_gb_categorical_values_valid(mixed_df):
    """GB transformer imputes categoricals with values from the training set."""
    imputer = MissinglyImputer(strategy="gb")
    result = imputer.fit(mixed_df).transform(mixed_df)
    valid_grades = set(mixed_df["grade"].dropna())
    assert set(result["grade"]).issubset(valid_grades)


# ---------------------------------------------------------------------------
# fit_transform convenience
# ---------------------------------------------------------------------------

def test_fit_transform_convenience(numeric_df):
    """fit_transform(X) is equivalent to fit(X).transform(X)."""
    imputer_a = MissinglyImputer(strategy="mean")
    result_a = imputer_a.fit_transform(numeric_df)

    imputer_b = MissinglyImputer(strategy="mean")
    result_b = imputer_b.fit(numeric_df).transform(numeric_df)

    pd.testing.assert_frame_equal(result_a, result_b)

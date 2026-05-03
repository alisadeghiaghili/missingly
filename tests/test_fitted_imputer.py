"""Tests for FittedImputer and make_imputer.

Conventions
-----------
* Each test creates its own train/test split to verify that transform()
  uses only the statistics learned from the training set.
* We do NOT test imputation quality — that is covered by test_impute.py.
  Here we only test the fit/transform contract and leakage-prevention.
"""

import numpy as np
import pandas as pd
import pytest

from missingly.impute import FittedImputer, make_imputer
import missingly


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def train_df():
    """Numeric training DataFrame with missing values."""
    return pd.DataFrame({
        'a': [1.0, np.nan, 3.0, 4.0, 5.0],
        'b': [10.0, 20.0, np.nan, 40.0, 50.0],
    })


@pytest.fixture
def test_df():
    """Numeric test DataFrame with missing values — different from train."""
    return pd.DataFrame({
        'a': [np.nan, 2.0],
        'b': [15.0, np.nan],
    })


@pytest.fixture
def mixed_train_df():
    """Mixed numeric + categorical training DataFrame."""
    return pd.DataFrame({
        'age':  [25.0, np.nan, 35.0, 40.0],
        'city': ['Paris', 'Lyon', None, 'Paris'],
    })


# ---------------------------------------------------------------------------
# make_imputer factory
# ---------------------------------------------------------------------------

def test_make_imputer_returns_fitted_imputer():
    """make_imputer() must return a FittedImputer instance."""
    imp = make_imputer('mean')
    assert isinstance(imp, FittedImputer)


def test_make_imputer_not_fitted_initially():
    """A freshly created FittedImputer must report is_fitted=False."""
    imp = make_imputer('mean')
    assert imp.is_fitted is False


def test_make_imputer_all_strategies(train_df):
    """make_imputer() must work for every supported strategy."""
    for strategy in ('mean', 'median', 'mode', 'knn', 'mice'):
        imp = make_imputer(strategy)
        result = imp.fit_transform(train_df)
        assert isinstance(result, pd.DataFrame)
        assert result.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# FittedImputer.fit / transform contract
# ---------------------------------------------------------------------------

def test_fit_returns_self(train_df):
    """fit() must return the FittedImputer instance (for chaining)."""
    imp = make_imputer('mean')
    result = imp.fit(train_df)
    assert result is imp


def test_is_fitted_after_fit(train_df):
    """is_fitted must be True after fit() is called."""
    imp = make_imputer('mean')
    imp.fit(train_df)
    assert imp.is_fitted is True


def test_transform_raises_before_fit(test_df):
    """transform() must raise RuntimeError if called before fit()."""
    imp = make_imputer('mean')
    with pytest.raises(RuntimeError, match=r"fit\(\)"):
        imp.transform(test_df)


def test_transform_returns_dataframe(train_df, test_df):
    """transform() must return a pd.DataFrame."""
    imp = make_imputer('mean')
    imp.fit(train_df)
    result = imp.transform(test_df)
    assert isinstance(result, pd.DataFrame)


def test_transform_no_missing(train_df, test_df):
    """Transformed output must have zero missing values."""
    imp = make_imputer('mean')
    imp.fit(train_df)
    result = imp.transform(test_df)
    assert result.isnull().sum().sum() == 0


def test_transform_uses_train_statistics_not_test(train_df, test_df):
    """transform() must use train means, not test means.

    train mean of 'a' = (1 + 3 + 4 + 5) / 4 = 3.25
    test mean  of 'a' = 2.0 (only non-missing value)

    The imputed value for test row 0 ('a' is NaN) must be 3.25, not 2.0.
    """
    imp = make_imputer('mean')
    imp.fit(train_df)
    result = imp.transform(test_df)
    train_mean_a = train_df['a'].mean()  # 3.25
    imputed_a = result.loc[0, 'a']
    assert abs(imputed_a - train_mean_a) < 1e-9, (
        f"Expected train mean {train_mean_a:.4f}, got {imputed_a:.4f}. "
        f"This suggests transform() is re-fitting on test data (data leakage)."
    )


def test_fit_transform_equivalent_to_fit_then_transform(train_df):
    """fit_transform(X) must produce the same result as fit(X).transform(X)."""
    imp1 = make_imputer('mean')
    result1 = imp1.fit_transform(train_df)

    imp2 = make_imputer('mean')
    imp2.fit(train_df)
    result2 = imp2.transform(train_df)

    pd.testing.assert_frame_equal(result1, result2)


def test_transform_preserves_columns(train_df, test_df):
    """Transformed DataFrame must have the same columns as the input."""
    imp = make_imputer('mean')
    imp.fit(train_df)
    result = imp.transform(test_df)
    assert list(result.columns) == list(test_df.columns)


def test_transform_preserves_index(train_df, test_df):
    """Transformed DataFrame must preserve the original row index."""
    imp = make_imputer('mean')
    imp.fit(train_df)
    result = imp.transform(test_df)
    pd.testing.assert_index_equal(result.index, test_df.index)


def test_mixed_dtype_fit_transform(mixed_train_df):
    """FittedImputer handles mixed numeric + categorical DataFrames."""
    imp = make_imputer('mode')
    result = imp.fit_transform(mixed_train_df)
    assert isinstance(result, pd.DataFrame)
    assert result.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr_unfitted():
    imp = make_imputer('knn')
    assert 'unfitted' in repr(imp)
    assert 'knn' in repr(imp)


def test_repr_fitted(train_df):
    imp = make_imputer('knn')
    imp.fit(train_df)
    assert 'fitted' in repr(imp)


# ---------------------------------------------------------------------------
# top-level import
# ---------------------------------------------------------------------------

def test_make_imputer_importable_from_top_level():
    assert callable(missingly.make_imputer)


def test_fitted_imputer_importable_from_top_level():
    assert missingly.FittedImputer is FittedImputer

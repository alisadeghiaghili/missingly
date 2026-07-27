"""Coverage for MissinglyImputer with model-based strategies (knn, mice, rf, gb).

This file specifically targets the gap left by stripping knn/mice from
test_make_imputer_all_strategies: make_imputer() intentionally rejects
these strategies (FittedImputer only supports simple scalars), so the
coverage must live in MissinglyImputer tests instead.

Focus areas
-----------
* CategoricalDtype is preserved after fit_transform for every strategy.
* No missing values remain after imputation.
* make_imputer raises InvalidStrategyError for model-based strategies.
"""

import numpy as np
import pandas as pd
import pytest

from missingly.transformer import MissinglyImputer
from missingly.impute import make_imputer
from missingly.exceptions import InvalidStrategyError


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mixed_df():
    """Small mixed numeric + CategoricalDtype DataFrame with missing values."""
    return pd.DataFrame({
        "num": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0],
        "cat": pd.Categorical(["a", np.nan, "b", "a", "b", "a", np.nan]),
    })


# ---------------------------------------------------------------------------
# CategoricalDtype preservation — MissinglyImputer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["knn", "mice", "rf", "gb"])
def test_categorical_dtype_preserved_model_strategies(mixed_df, strategy):
    """CategoricalDtype must survive fit_transform for model-based strategies."""
    imputer = MissinglyImputer(strategy=strategy)
    result = imputer.fit_transform(mixed_df)

    assert isinstance(result["cat"].dtype, pd.CategoricalDtype), (
        f"Strategy '{strategy}' lost CategoricalDtype: got {result['cat'].dtype}"
    )
    assert result["cat"].isna().sum() == 0, (
        f"Strategy '{strategy}' left NaN in categorical column"
    )
    assert result["num"].isna().sum() == 0, (
        f"Strategy '{strategy}' left NaN in numeric column"
    )


@pytest.mark.parametrize("strategy", ["knn", "mice", "rf", "gb"])
def test_no_missing_after_model_strategy(mixed_df, strategy):
    """fit_transform must produce a fully-imputed DataFrame for every model strategy."""
    result = MissinglyImputer(strategy=strategy).fit_transform(mixed_df)
    assert result.isnull().sum().sum() == 0


@pytest.mark.parametrize("strategy", ["knn", "mice", "rf", "gb"])
def test_fit_transform_returns_dataframe(mixed_df, strategy):
    """fit_transform must return a pd.DataFrame."""
    result = MissinglyImputer(strategy=strategy).fit_transform(mixed_df)
    assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize("strategy", ["knn", "mice", "rf", "gb"])
def test_column_names_preserved(mixed_df, strategy):
    """fit_transform must not change column names."""
    result = MissinglyImputer(strategy=strategy).fit_transform(mixed_df)
    assert list(result.columns) == list(mixed_df.columns)


# ---------------------------------------------------------------------------
# make_imputer correctly rejects model-based strategies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["knn", "mice", "rf", "gb"])
def test_make_imputer_raises_for_model_strategies(strategy):
    """make_imputer must raise InvalidStrategyError for model-based strategies.

    FittedImputer supports only {mean, median, mode} — strategies whose
    fit-state is a scalar per column.  Model-based strategies must be
    routed through MissinglyImputer instead.
    """
    with pytest.raises(InvalidStrategyError):
        make_imputer(strategy)

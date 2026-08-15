"""Public fallback contracts for random-forest and gradient-boosting imputers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from missingly.exceptions import ImputationError, InsufficientDataError
from missingly.impute import impute_gb, impute_rf


_CONDITIONAL_IMPUTERS: tuple[tuple[str, Callable[..., pd.DataFrame]], ...] = (
    ("rf", impute_rf),
    ("gb", impute_gb),
)
_CONDITIONAL_WRAPPERS: tuple[Callable[..., pd.DataFrame], ...] = (
    impute_rf,
    impute_gb,
)


@pytest.mark.parametrize("imputer", _CONDITIONAL_WRAPPERS, ids=["rf", "gb"])
def test_conditional_imputers_use_numeric_fallback_without_predictors(
    imputer: Callable[..., pd.DataFrame],
) -> None:
    """A valid one-column numeric frame uses the documented mean fallback."""
    frame = pd.DataFrame({"score": [1.0, np.nan, 3.0]})
    original = frame.copy(deep=True)

    result = imputer(frame, random_state=11, strict_mode=False)

    assert result["score"].tolist() == [1.0, 2.0, 3.0]
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("imputer", _CONDITIONAL_WRAPPERS, ids=["rf", "gb"])
def test_conditional_imputers_use_categorical_fallback_without_predictors(
    imputer: Callable[..., pd.DataFrame],
) -> None:
    """A categorical fallback keeps dtype, observed values, and valid support."""
    dtype = pd.CategoricalDtype(["low", "high"], ordered=True)
    frame = pd.DataFrame({"grade": pd.Series(["low", "high", pd.NA], dtype=dtype)})
    original = frame.copy(deep=True)

    result = imputer(frame, random_state=11, strict_mode=False)

    assert result["grade"].dtype == dtype
    assert result.loc[2, "grade"] in set(frame["grade"].dropna())
    pd.testing.assert_series_equal(result.loc[:1, "grade"], frame.loc[:1, "grade"])
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(("strategy", "imputer"), _CONDITIONAL_IMPUTERS)
@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"score": [1.0, np.nan, 3.0]}),
        pd.DataFrame({"grade": pd.Categorical(["low", "high", pd.NA])}),
    ],
    ids=["numeric", "categorical"],
)
def test_conditional_imputers_raise_typed_error_in_strict_mode(
    strategy: str,
    imputer: Callable[..., pd.DataFrame],
    frame: pd.DataFrame,
) -> None:
    """Strict mode exposes an estimator failure without changing caller data."""
    original = frame.copy(deep=True)

    with pytest.raises(ImputationError) as exc_info:
        imputer(frame, random_state=11, strict_mode=True)

    assert exc_info.value.strategy == strategy
    assert isinstance(exc_info.value.original, ValueError)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("imputer", _CONDITIONAL_WRAPPERS, ids=["rf", "gb"])
def test_conditional_imputers_reject_all_missing_target_before_estimation(
    imputer: Callable[..., pd.DataFrame],
) -> None:
    """Public conditional APIs report an all-missing target through typed data error."""
    frame = pd.DataFrame({"score": [np.nan, np.nan], "age": [20.0, 30.0]})
    original = frame.copy(deep=True)

    with pytest.raises(InsufficientDataError) as exc_info:
        imputer(frame, random_state=11, strict_mode=False)

    assert exc_info.value.column == "score"
    assert exc_info.value.n_observed == 0
    pd.testing.assert_frame_equal(frame, original)

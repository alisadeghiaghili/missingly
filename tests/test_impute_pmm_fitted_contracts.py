"""Public behavioral contracts for PMM and fitted simple imputers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missingly.exceptions import InsufficientDataError
from missingly.impute import impute_pmm, make_imputer


def test_impute_pmm_uses_mode_for_observed_non_numeric_data_without_mutation() -> None:
    """PMM delegates an observed categorical-only frame to mode imputation safely."""
    frame = pd.DataFrame(
        {"city": pd.Series(["Tehran", None, "Tehran"], dtype="string")}
    )
    original = frame.copy(deep=True)

    result = impute_pmm(frame, random_state=7)

    assert result["city"].tolist() == ["Tehran", "Tehran", "Tehran"]
    assert str(result["city"].dtype) == "string"
    pd.testing.assert_frame_equal(frame, original)


def test_impute_pmm_single_numeric_feature_uses_observed_mean(
) -> None:
    """PMM has a deterministic donor-free fallback for a single numeric feature."""
    frame = pd.DataFrame({"score": [1.0, np.nan, 3.0]})
    original = frame.copy(deep=True)

    result = impute_pmm(frame, random_state=7)

    assert result["score"].tolist() == [1.0, 2.0, 3.0]
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    ("frame", "column"),
    [
        (pd.DataFrame({"score": [np.nan, np.nan], "age": [20.0, 30.0]}), "score"),
        (pd.DataFrame({"city": pd.Series([None, None], dtype="string")}), "city"),
    ],
    ids=["numeric-target", "categorical-only-target"],
)
def test_impute_pmm_rejects_targets_without_observed_donors(
    frame: pd.DataFrame,
    column: str,
) -> None:
    """PMM reports no-donor targets with the package's typed data error."""
    original = frame.copy(deep=True)

    with pytest.raises(InsufficientDataError) as exc_info:
        impute_pmm(frame, random_state=7)

    assert exc_info.value.column == column
    assert exc_info.value.n_observed == 0
    pd.testing.assert_frame_equal(frame, original)


def test_impute_pmm_samples_only_observed_donors_deterministically() -> None:
    """PMM uses reproducible observed donor values and preserves observed cells."""
    frame = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0, 60.0],
            "income": [100.0, 150.0, np.nan, 250.0, 300.0],
        }
    )
    original = frame.copy(deep=True)

    first = impute_pmm(frame, random_state=23, n_nearest_donors=2)
    second = impute_pmm(frame, random_state=23, n_nearest_donors=2)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[2, "income"] in set(frame["income"].dropna())
    observed = ~frame["income"].isna()
    pd.testing.assert_series_equal(
        first.loc[observed, "income"],
        frame.loc[observed, "income"],
    )
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "test_frame",
    [
        pd.DataFrame({"age": [np.nan], "city": ["Tehran"], "extra": [np.nan]}),
        pd.DataFrame({"age": [np.nan]}),
        pd.DataFrame({"extra": [np.nan]}),
    ],
    ids=["extra-column", "missing-column", "unknown-only-column"],
)
def test_fitted_imputer_rejects_transform_schema_mismatches(
    test_frame: pd.DataFrame,
) -> None:
    """Transform rejects extra, missing, and unknown columns before any fill."""
    train = pd.DataFrame({"age": [20.0, 40.0], "city": ["Tehran", "Shiraz"]})
    original = test_frame.copy(deep=True)
    imputer = make_imputer("mean").fit(train)

    with pytest.raises(ValueError, match="columns"):
        imputer.transform(test_frame)

    pd.testing.assert_frame_equal(test_frame, original)


def test_fitted_imputer_rejects_all_missing_training_columns() -> None:
    """Fit fails before state changes when it cannot learn a column fill value."""
    train = pd.DataFrame(
        {
            "age": [20.0, 40.0],
            "city": pd.Series([None, None], dtype="string"),
        }
    )
    original = train.copy(deep=True)
    imputer = make_imputer("mode")

    with pytest.raises(InsufficientDataError) as exc_info:
        imputer.fit(train)

    assert exc_info.value.column == "city"
    assert imputer._is_fitted is False
    pd.testing.assert_frame_equal(train, original)


def test_fitted_imputer_preserves_nullable_dtype_and_training_statistics() -> None:
    """Transform retains extension dtypes and applies only learned train statistics."""
    train = pd.DataFrame(
        {
            "age": pd.Series([20, 40, None], dtype="Int64"),
            "city": pd.Series(["Tehran", "Shiraz", "Tehran"], dtype="string"),
        }
    )
    test = pd.DataFrame(
        {
            "age": pd.Series([None, 99], dtype="Int64"),
            "city": pd.Series([None, "Shiraz"], dtype="string"),
        }
    )
    original = test.copy(deep=True)

    result = make_imputer("mean").fit(train).transform(test)

    assert result["age"].tolist() == [30, 99]
    assert result["city"].tolist() == ["Tehran", "Shiraz"]
    assert str(result["age"].dtype) == "Int64"
    assert str(result["city"].dtype) == "string"
    pd.testing.assert_frame_equal(test, original)

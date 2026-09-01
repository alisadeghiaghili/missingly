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
from sklearn.base import clone
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
    train_result = imputer.transform(train)
    assert abs(train_result.loc[train_result.index[1], "age"] - train["age"].mean()) < 1e-6


@pytest.mark.parametrize("strategy", ["logreg", "polyreg", "polr"])
def test_non_inductive_regression_strategies_fail_loudly(strategy, numeric_df):
    """A sklearn transformer must never refit a regression model on test rows."""
    with pytest.raises(NotImplementedError, match="inductive"):
        MissinglyImputer(strategy=strategy).fit(numeric_df)


def test_mixed_knn_fails_loudly_in_inductive_mode(mixed_df):
    """Mixed Gower KNN must not combine evaluation rows with training donors."""
    with pytest.raises(NotImplementedError, match="transform rows as donors"):
        MissinglyImputer(strategy="knn", metric="mixed").fit(mixed_df)


@pytest.mark.parametrize("method", ["sequential", "weighted"])
def test_non_inductive_hotdeck_variants_fail_loudly(method, numeric_df):
    """Transformer hot-deck variants need an explicit test-donor policy."""
    with pytest.raises(NotImplementedError, match="Only random hot-deck"):
        MissinglyImputer(strategy="hotdeck", hotdeck_method=method).fit(numeric_df)


def test_random_hotdeck_uses_training_donors_only():
    """Extreme observed values in test data must never become hot-deck donors."""
    train = pd.DataFrame({"value": [10.0, 20.0], "feature": [1.0, 2.0]})
    test = pd.DataFrame({"value": [np.nan, 9999.0], "feature": [3.0, 4.0]})

    result = MissinglyImputer(strategy="hotdeck", random_state=0).fit(train).transform(test)

    assert result.loc[0, "value"] in {10.0, 20.0}


def test_hotdeck_rejects_a_column_without_training_donors():
    """Inductive hot-deck must fail rather than invent donors for a column."""
    train = pd.DataFrame({"value": [np.nan], "feature": [1.0]})
    serving = pd.DataFrame({"value": [np.nan], "feature": [2.0]})

    with pytest.raises(ValueError, match="no observed training donors"):
        MissinglyImputer(strategy="hotdeck", random_state=0).fit(train).transform(serving)


def test_pmm_transform_draws_only_from_training_target_donors():
    """PMM serving output must be a training donor, never a test-row value."""
    train = pd.DataFrame({"target": [10.0, 20.0, 30.0], "feature": [1.0, 2.0, 3.0]})
    serving = pd.DataFrame({"target": [np.nan], "feature": [2.0]})

    result = MissinglyImputer(strategy="pmm", random_state=0).fit(train).transform(serving)

    assert result.loc[0, "target"] in set(train["target"])


def test_pmm_transform_uses_the_training_mean_for_a_single_numeric_column():
    """PMM has a deterministic valid fallback when no predictors exist."""
    train = pd.DataFrame({"target": [10.0, 20.0, 30.0]})
    serving = pd.DataFrame({"target": [np.nan]})

    result = MissinglyImputer(strategy="pmm", random_state=0).fit(train).transform(serving)

    assert result.loc[0, "target"] == pytest.approx(20.0)


def test_gb_transform_is_invariant_to_other_test_rows():
    """GB feature filling must use training means, not aggregate test statistics."""
    train = pd.DataFrame({
        "target": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
        "feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    recipient = pd.DataFrame({"target": [np.nan], "feature": [np.nan]})
    batch = pd.concat(
        [recipient, pd.DataFrame({"target": [1000.0], "feature": [1_000_000.0]})],
        ignore_index=True,
    )
    imputer = MissinglyImputer(strategy="gb", random_state=0)
    imputer.fit(train)

    alone = imputer.transform(recipient)
    together = imputer.transform(batch)

    assert alone.loc[0, "target"] == pytest.approx(together.loc[0, "target"])


@pytest.mark.parametrize("strategy", ["knn", "mice", "rf", "gb", "hotdeck"])
def test_single_row_all_missing_categorical_column_is_served(strategy):
    """A single all-missing categorical serving row must not crash."""
    train = pd.DataFrame(
        {
            "number": [1.0, 2.0, 3.0, 4.0],
            "category": ["a", "b", "a", "b"],
        }
    )
    serving = pd.DataFrame({"number": [1.5], "category": [np.nan]})

    result = MissinglyImputer(strategy=strategy, random_state=0).fit(train).transform(serving)

    assert result.loc[0, "number"] == 1.5
    assert result.loc[0, "category"] in {"a", "b"}


@pytest.mark.parametrize("strategy", ["rf", "gb"])
def test_tree_transform_is_invariant_to_input_column_order(strategy):
    """Tree models must use the feature order learned during fit."""
    rng = np.random.default_rng(12)
    feature = np.linspace(-2.0, 2.0, 80)
    nuisance = rng.normal(size=80)
    target = 10.0 * feature + rng.normal(scale=0.1, size=80)
    train = pd.DataFrame({"target": target, "feature": feature, "nuisance": nuisance})
    train.loc[::5, "target"] = np.nan
    imputer = MissinglyImputer(strategy=strategy, random_state=0).fit(train)

    canonical = pd.DataFrame({"target": [np.nan], "feature": [1.0], "nuisance": [0.0]})
    reordered = canonical[["nuisance", "feature", "target"]]

    canonical_result = imputer.transform(canonical)
    reordered_result = imputer.transform(reordered)

    assert reordered_result.loc[0, "target"] == pytest.approx(
        canonical_result.loc[0, "target"]
    )
    assert reordered_result.columns.tolist() == train.columns.tolist()


@pytest.mark.parametrize("strategy", ["knn", "mice"])
def test_observed_unseen_categories_are_not_imputed_as_missing(strategy):
    """An observed category outside the training vocabulary must be preserved."""
    train = pd.DataFrame(
        {
            "number": [1.0, 2.0, 3.0, 4.0],
            "category": ["a", "b", "a", "b"],
        }
    )
    serving = pd.DataFrame({"number": [10.0, 11.0], "category": ["unseen", "a"]})

    result = MissinglyImputer(strategy=strategy, random_state=0).fit(train).transform(serving)

    assert result.loc[0, "category"] == "unseen"
    assert result.loc[1, "category"] == "a"


@pytest.mark.parametrize("strategy", ["knn", "mice"])
def test_observed_unseen_categorical_dtype_is_preserved(strategy):
    """Restoring an unseen category must retain a valid categorical dtype."""
    train = pd.DataFrame(
        {
            "number": [1.0, 2.0, 3.0, 4.0],
            "category": pd.Categorical(["a", "b", "a", "b"]),
        }
    )
    serving = pd.DataFrame(
        {
            "number": [10.0, 11.0],
            "category": pd.Categorical(["unseen", "a"], categories=["a", "b", "unseen"]),
        }
    )

    result = MissinglyImputer(strategy=strategy, random_state=0).fit(train).transform(serving)

    assert isinstance(result["category"].dtype, pd.CategoricalDtype)
    assert "unseen" in result["category"].cat.categories
    assert result.loc[0, "category"] == "unseen"


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


def test_extra_columns_raises(numeric_df):
    """transform raises ValueError when test has columns not seen during fit."""
    imputer = MissinglyImputer(strategy="mean")
    imputer.fit(numeric_df)
    extra_df = numeric_df.copy()
    extra_df["extra"] = 1.0
    with pytest.raises(ValueError, match="not seen during fit"):
        imputer.transform(extra_df)


# ---------------------------------------------------------------------------
# Invalid strategy
# ---------------------------------------------------------------------------

def test_invalid_strategy_raises():
    """Passing an unknown strategy raises ValueError at construction."""
    with pytest.raises(ValueError, match="strategy"):
        MissinglyImputer(strategy="interpolate")


def test_set_params_rejects_invalid_strategy():
    """set_params validates strategy before a later fit can silently no-op."""
    imputer = MissinglyImputer()

    with pytest.raises(ValueError, match="strategy"):
        imputer.set_params(strategy="interpolate")


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
    """get_feature_names_out follows sklearn's ndarray output contract."""
    imputer = MissinglyImputer(strategy="mean")
    imputer.fit(numeric_df)
    feature_names = imputer.get_feature_names_out()

    assert isinstance(feature_names, np.ndarray)
    assert feature_names.dtype == object
    assert feature_names.tolist() == numeric_df.columns.tolist()


def test_clone_preserves_public_parameters_without_fitted_state(numeric_df):
    """sklearn.clone recreates an unfitted estimator from constructor parameters."""
    fitted = MissinglyImputer(strategy="median", n_neighbors=3).fit(numeric_df)

    cloned = clone(fitted)

    assert cloned.get_params(deep=False) == fitted.get_params(deep=False)
    assert cloned._is_fitted is False


def test_pmm_single_feature_uses_only_training_donors_without_mutation():
    """PMM's no-predictor fallback samples observed donors rather than a mean."""
    train = pd.DataFrame({"score": [1.0, 3.0]})
    serving = pd.DataFrame({"score": [np.nan, 99.0, np.nan]})
    original = serving.copy(deep=True)

    result = MissinglyImputer(strategy="pmm", random_state=7).fit(train).transform(
        serving
    )

    assert set(result.loc[[0, 2], "score"]).issubset({1.0, 3.0})
    assert result.loc[1, "score"] == 99.0
    pd.testing.assert_frame_equal(serving, original)


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


# ---------------------------------------------------------------------------
# Categorical dtype preservation (Bug #1)
# ---------------------------------------------------------------------------

def test_categorical_dtype_preserved_by_strategy():
    """CategoricalDtype must be preserved after imputation for all strategies."""
    df = pd.DataFrame({
        "num": [1.0, np.nan, 3.0, 4.0, 5.0],
        "cat": pd.Categorical(["a", np.nan, "b", "a", "b"]),
    })
    for strategy in ["mean", "median", "mode", "knn", "rf", "gb"]:
        imputer = MissinglyImputer(strategy=strategy)
        result = imputer.fit_transform(df)
        assert isinstance(result["cat"].dtype, pd.CategoricalDtype), (
            f"Strategy '{strategy}' lost CategoricalDtype: "
            f"got {result['cat'].dtype}"
        )
        assert not result["cat"].isna().any(), (
            f"Strategy '{strategy}' left NaN in categorical column"
        )

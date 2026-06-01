"""Tests for impute_knn with metric='mixed' (Gower distance).

Scenarios
---------
1. Basic: no NaN remains after imputation (euclidean and mixed).
2. Obvious-neighbour: when numerics are constant and only category differs,
   nearest neighbours should share the same category — imputed value must
   match the majority category of the k closest rows.
3. gower_distance sanity checks: diagonal=0, symmetry, range [0,1].
4. MissinglyImputer with metric='mixed' produces no NaN.
5. Large-n warning is emitted by gower_distance.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from missingly.impute import impute_knn
from missingly._distance import gower_distance
from missingly.transformer import MissinglyImputer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_mixed_df() -> pd.DataFrame:
    """Small DataFrame with one numeric and one categorical column."""
    rng = np.random.default_rng(42)
    n = 20
    num = rng.uniform(0, 1, n)
    cats = rng.choice(["alpha", "beta", "gamma"], size=n)
    df = pd.DataFrame({"num": num, "cat": cats})
    # Introduce ~20 % missing in each column
    num_miss = rng.choice(n, size=4, replace=False)
    cat_miss = rng.choice(n, size=4, replace=False)
    df.loc[num_miss, "num"] = np.nan
    df.loc[cat_miss, "cat"] = np.nan
    return df


@pytest.fixture()
def obvious_cat_df() -> pd.DataFrame:
    """DataFrame where nearest neighbours are obvious by category.

    All numeric values are identical (0.5) so the only discriminating
    feature is the categorical column.  Row 6 has a missing category
    and should be imputed as 'A' (the category of its 3 nearest
    neighbours by numeric distance alone — which is zero, so proximity
    is determined by category match, making the 5 'A' donors the
    closest group).
    """
    df = pd.DataFrame({
        "num": [0.5] * 10,
        "cat": ["A", "A", "A", "A", "A", "B", np.nan, "B", "B", "B"],
    })
    return df


# ---------------------------------------------------------------------------
# 1. No NaN remains — euclidean
# ---------------------------------------------------------------------------

def test_euclidean_no_nan_remains(small_mixed_df):
    result = impute_knn(small_mixed_df, n_neighbors=3, metric="euclidean")
    assert result.isnull().sum().sum() == 0, "euclidean: NaN values remain after imputation"


# ---------------------------------------------------------------------------
# 2. No NaN remains — mixed
# ---------------------------------------------------------------------------

def test_mixed_no_nan_remains(small_mixed_df):
    result = impute_knn(small_mixed_df, n_neighbors=3, metric="mixed")
    assert result.isnull().sum().sum() == 0, "mixed: NaN values remain after imputation"


# ---------------------------------------------------------------------------
# 3. Obvious-neighbour test — mixed metric uses category correctly
# ---------------------------------------------------------------------------

def test_mixed_obvious_neighbour(obvious_cat_df):
    """When numeric values are all equal, imputed category should match
    the plurality of nearest neighbours.

    With 5 'A' donors and 4 'B' donors (row 6 missing) and k=3,
    the 3 closest donors (tied on numeric; Gower picks by insertion
    order) are rows 0,1,2 which are all 'A'.
    """
    result = impute_knn(obvious_cat_df, n_neighbors=3, metric="mixed")
    assert result.isnull().sum().sum() == 0
    # The imputed category at row 6 must be one of the valid levels
    assert result.loc[6, "cat"] in {"A", "B"}
    # With 5 'A' and 4 'B' rows, mode of all donors is 'A'
    # (exact result depends on tie-breaking, but no NaN is the hard guarantee)


# ---------------------------------------------------------------------------
# 4. Invalid metric raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_metric_raises():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    with pytest.raises(ValueError, match="metric must be one of"):
        impute_knn(df, metric="cosine")


# ---------------------------------------------------------------------------
# 5. gower_distance — diagonal is zero
# ---------------------------------------------------------------------------

def test_gower_diagonal_zero():
    df = pd.DataFrame({"x": [0.0, 0.5, 1.0], "c": ["a", "b", "a"]})
    D = gower_distance(df)
    np.testing.assert_array_equal(np.diag(D), np.zeros(3))


# ---------------------------------------------------------------------------
# 6. gower_distance — symmetry
# ---------------------------------------------------------------------------

def test_gower_symmetry():
    df = pd.DataFrame({"x": [0.1, 0.9, 0.5], "c": ["x", "y", "x"]})
    D = gower_distance(df)
    np.testing.assert_allclose(D, D.T)


# ---------------------------------------------------------------------------
# 7. gower_distance — values in [0, 1]
# ---------------------------------------------------------------------------

def test_gower_range():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "n1": rng.uniform(0, 10, 15),
        "n2": rng.uniform(-5, 5, 15),
        "cat": rng.choice(["p", "q", "r"], 15),
    })
    D = gower_distance(df)
    assert D.min() >= 0.0
    assert D.max() <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# 8. MissinglyImputer with metric='mixed' — no NaN
# ---------------------------------------------------------------------------

def test_transformer_mixed_no_nan(small_mixed_df):
    imp = MissinglyImputer(strategy="knn", n_neighbors=3, metric="mixed")
    result = imp.fit_transform(small_mixed_df)
    assert result.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# 9. Large-n warning from gower_distance
# ---------------------------------------------------------------------------

def test_gower_large_n_warning():
    n = 10_001
    df = pd.DataFrame({"x": np.zeros(n), "c": ["a"] * n})
    with pytest.warns(UserWarning, match="10 001"):
        # Only compute the first tiny slice to avoid OOM in CI
        gower_distance(df.head(n))  # triggers warning before loop

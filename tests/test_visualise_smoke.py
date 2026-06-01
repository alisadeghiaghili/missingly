"""Smoke tests: call every public visualisation function once on a 100x8 DataFrame.

Goal
----
No exceptions raised in either static (matplotlib) or ``interactive=False``
mode.  Interactive=True (Plotly) paths are covered separately in
``test_visualise_interactive.py`` and ``test_visualise_interactive_phase2.py``.

Coverage
--------
All functions listed in ``missingly.visualise.__all__`` are exercised, plus
some extra flag combinations (``cluster=True``, ``normalize=False``, etc.).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from missingly import visualise
from missingly.visualisation import (
    matrix, bar, miss_case, miss_var_pct, vis_miss, miss_which,
    upset, miss_patterns, miss_cooccurrence,
    heatmap, dendrogram, miss_cluster,
    miss_row_profile, scatter_miss, shadow_scatter,
    vis_miss_cumsum_var, vis_miss_cumsum_case, vis_miss_span,
    vis_parallel_coords, vis_miss_fct, vis_miss_by_group,
    vis_impute_dist, miss_impute_compare,
)


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


@pytest.fixture
def df_medium() -> pd.DataFrame:
    """100x8 DataFrame with numeric + categorical columns and ~15% missing."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "age":    rng.integers(18, 80, n).astype(float),
        "income": rng.uniform(20_000, 120_000, n),
        "score":  rng.standard_normal(n),
        "weight": rng.uniform(50, 120, n),
        "height": rng.uniform(150, 200, n),
        "visits": rng.integers(0, 50, n).astype(float),
        "city":   rng.choice(["A", "B", "C"], n),
        "status": rng.choice(["active", "inactive", None], n),
    })
    for col in ["age", "income", "score", "weight", "height", "visits"]:
        idx = rng.choice(n, size=int(n * 0.15), replace=False)
        df.loc[idx, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Basic
# ---------------------------------------------------------------------------

def test_matrix_smoke(df_medium):
    ax = matrix(df_medium)
    assert ax is not None


def test_bar_smoke(df_medium):
    ax = bar(df_medium)
    assert ax is not None


def test_miss_case_smoke(df_medium):
    ax = miss_case(df_medium)
    assert ax is not None


def test_miss_var_pct_smoke(df_medium):
    ax = miss_var_pct(df_medium)
    assert ax is not None


def test_vis_miss_smoke(df_medium):
    ax = vis_miss(df_medium)
    assert ax is not None


def test_vis_miss_clustered_smoke(df_medium):
    ax = vis_miss(df_medium, cluster=True)
    assert ax is not None


def test_miss_which_smoke(df_medium):
    ax = miss_which(df_medium)
    assert ax is not None


# ---------------------------------------------------------------------------
# Pattern analysis
# ---------------------------------------------------------------------------

def test_upset_smoke(df_medium):
    result = upset(df_medium)
    assert isinstance(result, dict)
    assert "intersections" in result


def test_miss_patterns_smoke(df_medium):
    ax = miss_patterns(df_medium)
    assert ax is not None


def test_miss_cooccurrence_smoke(df_medium):
    ax = miss_cooccurrence(df_medium)
    assert ax is not None


def test_miss_cooccurrence_raw_counts_smoke(df_medium):
    ax = miss_cooccurrence(df_medium, normalize=False)
    assert ax is not None


# ---------------------------------------------------------------------------
# Correlation / clustering
# ---------------------------------------------------------------------------

def test_heatmap_smoke(df_medium):
    ax = heatmap(df_medium)
    assert ax is not None


def test_heatmap_mask_insignificant_smoke(df_medium):
    ax = heatmap(df_medium, mask_insignificant=True)
    assert ax is not None


def test_dendrogram_smoke(df_medium):
    ax = dendrogram(df_medium)
    assert ax is not None


def test_miss_cluster_smoke(df_medium):
    ax = miss_cluster(df_medium)
    assert ax is not None


# ---------------------------------------------------------------------------
# Row / variable profiles
# ---------------------------------------------------------------------------

def test_miss_row_profile_smoke(df_medium):
    ax = miss_row_profile(df_medium)
    assert ax is not None


def test_vis_miss_cumsum_var_smoke(df_medium):
    ax = vis_miss_cumsum_var(df_medium)
    assert ax is not None


def test_vis_miss_cumsum_case_smoke(df_medium):
    ax = vis_miss_cumsum_case(df_medium)
    assert ax is not None


def test_vis_miss_span_smoke(df_medium):
    ax = vis_miss_span(df_medium, column="age", span_every=10)
    assert ax is not None


def test_vis_parallel_coords_smoke(df_medium):
    ax = vis_parallel_coords(df_medium)
    assert ax is not None


# ---------------------------------------------------------------------------
# Scatter / MAR / shadow
# ---------------------------------------------------------------------------

def test_scatter_miss_smoke(df_medium):
    ax = scatter_miss(df_medium, x="age", y="income")
    assert ax is not None


def test_shadow_scatter_smoke(df_medium):
    ax = shadow_scatter(df_medium, x="age", y="income", shadow_col="score")
    assert ax is not None


# ---------------------------------------------------------------------------
# Factor / group breakdown
# ---------------------------------------------------------------------------

def test_vis_miss_fct_smoke(df_medium):
    ax = vis_miss_fct(df_medium, fct="city")
    assert ax is not None


def test_vis_miss_by_group_smoke(df_medium):
    ax = vis_miss_by_group(df_medium, group_col="city")
    assert ax is not None


# ---------------------------------------------------------------------------
# Imputation diagnostics
# ---------------------------------------------------------------------------

def test_vis_impute_dist_smoke(df_medium):
    imputed = df_medium.fillna(df_medium.mean(numeric_only=True))
    ax = vis_impute_dist(df_medium, imputed, column="age")
    assert ax is not None


def test_miss_impute_compare_smoke(df_medium):
    imputed = df_medium.fillna(df_medium.mean(numeric_only=True))
    fig = miss_impute_compare(df_medium, imputed)
    assert fig is not None


# ---------------------------------------------------------------------------
# Sentinel / extra missing_values
# ---------------------------------------------------------------------------

def test_sentinel_missing_values_smoke(df_medium):
    """Passing custom sentinel values must not raise."""
    df2 = df_medium.copy()
    df2.loc[0, "age"] = -99
    ax = bar(df2, missing_values=[-99])
    assert ax is not None


# ---------------------------------------------------------------------------
# Facade + subpackage import checks
# ---------------------------------------------------------------------------

def test_facade_imports():
    """All public names must be accessible via missingly.visualise directly."""
    for name in visualise.__all__:
        assert hasattr(visualise, name), f"missingly.visualise.{name} not found"


def test_new_subpackage_imports():
    """All public names must be accessible via missingly.visualisation."""
    from missingly import visualisation
    for name in visualisation.__all__:
        assert hasattr(visualisation, name), f"missingly.visualisation.{name} not found"
